from typing import List, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, VisionDataset

from egg.zoo.coco_game.utils.utils import console


class DummyData(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        base_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        perc_ids (int, optional): Max number of ids to load from the dataset, -1 for all
    """

    def __init__(
            self,
            data_len,
            image_size: int = 224,
            distractors: int = 1,
            data_seed: int = 42,
    ):
        """
        Custom Dataset
        Args:
            root: path to the coco dataset
            ann_file: path to coco annotations file
            base_transform: transformation used for the sender input
        """

        self.data_len = data_len
        self.distractors = distractors
        self.random_state = np.random.RandomState(data_seed)
        self.image_size = image_size

    def get_images(
            self, img_id: List[int], image_anns: List[int], size: Tuple[int, int]
    ) -> List[np.array]:
        """
        Get images, draw bbox with class name, resize and return
        """

        imgs = [
            self.random_state.random((self.image_size, self.image_size))
            for _ in range(img_id)
        ]
        return imgs

    def __getitem__(self, index: int):
        """
        Function called by the epoch iterator
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """

        # preprocess segments
        chosen_sgm = self.random_state.random((3, self.image_size, self.image_size))
        resized_image = self.random_state.random((3, self.image_size, self.image_size))
        distractors_sgm = [
            self.random_state.random((3, self.image_size, self.image_size))
            for _ in range(self.distractors)
        ]

        chosen_sgm = torch.Tensor(chosen_sgm)
        resized_image = torch.Tensor(resized_image)
        distractors_sgm = [torch.Tensor(x) for x in distractors_sgm]
        segments = [chosen_sgm] + distractors_sgm

        # define sender input
        sender_inp = torch.stack((resized_image, segments[0]))

        # shuffle segments
        segments = [x.unsqueeze(dim=0) for x in segments]
        indices = list(range(len(segments)))
        self.random_state.shuffle(indices)
        shuffled_segs = [segments[idx] for idx in indices]
        segments = torch.cat(shuffled_segs, dim=0)

        # labels are : position of true seg, category of segment, image id, annotation id
        labels = [
            torch.LongTensor([indices[0], 1, 2, 3]) for _ in range(self.distractors + 1)
        ]
        labels = [x.unsqueeze(dim=0) for x in labels]
        labels = torch.cat(labels, dim=0)

        return sender_inp, segments, labels

    def __len__(self):
        return self.data_len


def get_dummy_data(data_len, opts):
    d = DummyData(
        data_len=data_len,
        image_size=opts.image_resize,
        distractors=opts.distractors,
        data_seed=opts.data_seed,
    )

    d = DataLoader(
        d,
        shuffle=False,
        drop_last=False,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
    )

    return d


def collate(
        batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Manage input (samples, segmented) and labels to feed at sender and reciever

    :return : sender input, labels, reciever input, aux_info
            (( image, segment), bboxs, image), target dict
    """

    # extract infos
    sender_inp = [elem[0] for elem in batch]
    seg = [elem[1] for elem in batch]
    labels = [elem[2] for elem in batch]

    # save_data(labels, f"data_diff2.csv")

    # stack on torch tensor
    sender_inp = torch.stack(sender_inp).contiguous()
    seg = torch.stack(seg).contiguous()
    labels = torch.stack(labels).contiguous()

    # transpose to have [batch,discriminants ...]-> [discriminants, batch, ...]
    # seg= np.transpose(seg, axes=(1, 0, 2, 3, 4))
    # labels= np.transpose(labels, axes=(1, 0, 2))

    return sender_inp, labels, seg


def filter_distractors(
        train_data: CocoDetection, val_data: CocoDetection, min_annotations: int
):
    """
    Filter both train and val given a minimum number of distractors per image.
    An image is valid if the number of annotations in it is >= distractors+1
    """
    min_perc_valid = 0.7
    train_cats = train_data.coco.cats
    train_cats = [x["name"] for x in train_cats.values()]

    stop = False
    while not stop:
        stop = inner_filtering(train_data, val_data, min_annotations, min_perc_valid)

    new_train_cats = train_data.coco.cats
    new_train_cats = [x["name"] for x in new_train_cats.values()]

    removed_cats = set(train_cats) - set(new_train_cats)

    console.log(
        f"A total of {len(new_train_cats)}/{len(train_cats)} classes are left from the distractor filtering\n"
        f"Filtered classes are {len(removed_cats)} : {removed_cats}"
    )


def get_annotation_stats(
        coco: COCO, min_annotations: int, min_perc_valid: float
) -> List[str]:
    """
    Return the discarded categories and removes annotations/images and cats from coco
    """
    cat2remove = {id_: dict(total=0, valid=0) for id_ in coco.cats.keys()}

    imgs_to_rm = []
    anns2remove = []

    for img_id, anns in coco.imgToAnns.items():

        valid = 1 if len(anns) > min_annotations else 0

        if not valid:
            imgs_to_rm.append(img_id)
            anns2remove += [x["id"] for x in anns]

        for anns in anns:
            ann_id = anns["category_id"]
            cat2remove[ann_id]["total"] += 1
            cat2remove[ann_id]["valid"] += valid

    cat2remove = {k: v["valid"] / v["total"] for k, v in cat2remove.items()}
    cat2remove = [k for k, v in cat2remove.items() if v < min_perc_valid]

    anns2remove += [k for k, v in coco.anns.items() if v["category_id"] in cat2remove]

    anns2remove = set(anns2remove)

    for img_id in imgs_to_rm:
        coco.imgs.pop(img_id)

    for ann_id in anns2remove:
        coco.anns.pop(ann_id)

    for cat in cat2remove:
        coco.cats.pop(cat)

    return cat2remove


def remove_cats(coco: COCO, cat_list: set):
    """
    Removes a category (annotations) from coco
    """
    ans2remove = []
    for cat in cat_list:
        images = coco.catToImgs[cat]
        for img in images:
            annotations = coco.imgToAnns[img]
            for key in range(len(annotations)):
                anns = annotations[key]
                if anns["category_id"] == cat:
                    ans2remove.append(anns["id"])

        coco.cats.pop(cat)
    ans2remove = set(ans2remove)
    for ans in ans2remove:
        coco.anns.pop(ans)


def inner_filtering(
        train_coco: CocoDetection,
        val_coco: CocoDetection,
        min_annotations: int,
        min_perc_valid: float,
) -> bool:
    """
    After independently removing images/anns from train and val based on min distractors, cross check to see if val and
     train have the same categories. If not rremoves the surplus cats and returns false
    """
    train_stats = get_annotation_stats(train_coco.coco, min_annotations, min_perc_valid)
    val_stats = get_annotation_stats(val_coco.coco, min_annotations, min_perc_valid)

    train_coco.init_dicts()
    val_coco.init_dicts()

    cats2rm_train = set(val_stats) - set(train_stats)
    cats2rm_val = set(train_stats) - set(val_stats)

    if len(cats2rm_val) != 0 or len(cats2rm_train) != 0:
        stop = False
        remove_cats(train_coco.coco, cats2rm_train)
        remove_cats(val_coco.coco, cats2rm_val)

        train_coco.init_dicts()
        val_coco.init_dicts()
    else:
        stop = True

    return stop


def save_data(data_list, data_file="data.csv"):
    data_list = [x.tolist() for x in data_list]
    data_list = [x for sub in data_list for x in sub]
    data_list = [",".join([str(y) for y in x]) for x in data_list]
    data_list = [f"{x}\n" for x in data_list]
    with open(data_file, "a") as f:
        f.writelines(data_list)


def check_data(train_data: CocoDetection, val_data: CocoDetection):
    coco_train = train_data.coco
    coco_val = val_data.coco

    train_imgs = coco_train.imgs.keys()
    val_imgs = coco_val.imgs.keys()

    train_imgs = set(train_imgs)
    val_imgs = set(val_imgs)

    same_imgs = train_imgs.intersection(val_imgs)

    if len(same_imgs) > 0:
        raise AttributeError("Test data and train data share same images")

    train_anns = coco_train.anns.keys()
    val_anns = coco_val.anns.keys()

    train_anns = set(train_anns)
    val_anns = set(val_anns)

    same_anns = train_anns.intersection(val_anns)

    if len(same_anns) > 0:
        raise AttributeError("Test data and train data share same annotations")


def split_dataset(dataset: DataLoader):
    data_len = len(dataset)
    train_len = int(data_len * 0.8)
    val_len = data_len - train_len
    train, val = torch.utils.data.random_split(dataset, [train_len, val_len])

    return train.dataset, val.dataset
