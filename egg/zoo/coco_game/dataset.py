import os
import random
from argparse import Namespace
from functools import reduce
from typing import Dict, List, Tuple

import PIL
import albumentations as album
import cv2
import lycon
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from egg.zoo.coco_game.utils.dataset_utils import check_same_classes
from egg.zoo.coco_game.utils.utils import console
from egg.zoo.coco_game.utils.vis_utils import visualize_bbox


class CocoDetection(VisionDataset):
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
            root: str,
            ann_file: str,
            base_transform: album.Compose,
            perc_ids: float = 1,
            distractors:int=1,
    ):
        """
        Custom Dataset
        Args:
            root: path to the coco dataset
            ann_file: path to coco annotations file
            base_transform: transformation used for the sender input
            perc_ids: percentage of ids to keep
        """
        super(CocoDetection, self).__init__(root, None, None, None)

        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.base_transform = base_transform
        self.distractors=distractors

        ############
        # Filtering
        ############
        # randomly drop perc_ids
        self.ids = self.delete_rand_items(self.ids, int((1 - perc_ids) * len(self.ids)))

        self.ids = sorted(self.ids)

    @staticmethod
    def delete_rand_items(items: list, n: int):
        original_len = len(items)
        to_delete = set(random.sample(range(len(items)), n))
        new_len = len(items) - n

        console.log(
            f"Perc filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )
        return [x for i, x in enumerate(items) if not i in to_delete]

    def get_class_weights(self) -> Dict[int, int]:
        dataset = self.coco.dataset

        ans = dataset["annotations"]
        class_weights = {}

        for idx in range(len(ans)):
            cat = ans[idx]["category_id"]
            if cat not in class_weights.keys():
                class_weights[cat] = 0
            class_weights[cat] += 1

        class_weights = {k: v / idx for k, v in class_weights.items()}
        class_weights = {k: 1 - v for k, v in class_weights.items()}
        return class_weights

    def filter_anns_classes(self, num_classes: int, over_presented: int):
        """
        Filter annotations based on minimum (mean) area
        """

        original_len = len(self.ids)

        dataset = self.coco.dataset
        cats = dataset["categories"]
        cats = cats[over_presented: num_classes + over_presented]
        ans = dataset["annotations"]
        ids = [elem["id"] for elem in cats]
        cat_map = {}
        for idx in range(len(cats)):
            cat_map[cats[idx]["id"]] = idx
            cats[idx]["id"] = idx

        dataset["categories"] = cats

        for idx in range(len(ans)):
            cat = ans[idx]["category_id"]
            if cat not in ids:
                ans[idx] = None
            else:
                ans[idx]["category_id"] = cat_map[ans[idx]["category_id"]]

        dataset["annotations"] = [elem for elem in ans if elem is not None]

        self.coco.dataset = dataset
        self.coco.createIndex()
        self.ids = list(self.coco.anns.keys())
        new_len = len(self.ids)

        console.log(
            f"Classes filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )

    def init_dicts(self):
        imgToAnns, catToImgs = {}, {}

        for ann in self.coco.anns.values():
            if ann["image_id"] not in imgToAnns.keys():
                imgToAnns[ann["image_id"]] = []
            if ann["category_id"] not in catToImgs.keys():
                catToImgs[ann["category_id"]] = []

            imgToAnns[ann["image_id"]].append(ann)
            catToImgs[ann["category_id"]].append(ann["image_id"])

        imgs2rm=set(self.coco.imgs.keys())-set(imgToAnns.keys())
        self.coco.imgs= {k: v for k, v in self.coco.imgs.items() if k not in imgs2rm}

        self.coco.imgToAnns = imgToAnns
        self.coco.catToImgs = catToImgs
        self.ids = list(self.coco.imgs.keys())


    def filter_anns_area(self, min_area=0.1):
        """
        Filter annotations based on minimum (mean) area
        """
        eps = 0.0001
        original_len = len(self.ids)

        for idx in range(len(self.ids)):
            an_id = self.ids[idx]
            an = self.coco.anns[an_id]
            img = self.coco.loadImgs(an["image_id"])[0]

            # get normalized area
            area = (
                    (an["bbox"][2] + eps)
                    / img["width"]
                    * (an["bbox"][3] + eps)
                    / img["height"]
            )
            if area < min_area:
                self.ids[idx] = None

        self.ids = [x for x in self.ids if x is not None]

        new_len = len(self.ids)

        console.log(
            f"Area filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )

    def get_images(
            self, img_id: List[int], image_anns: List[int], size: Tuple[int, int]
    ) -> List[np.array]:
        """
        Get images, draw bbox with class name, resize and return
        """

        infos = self.coco.loadImgs(img_id)
        paths = [os.path.join(self.root, pt["file_name"]) for pt in infos]
        anns = self.coco.loadAnns(image_anns)
        cats_name = [self.coco.cats[x["category_id"]]["name"] for x in anns]
        bboxs = [x["bbox"] for x in anns]

        imgs = [lycon.load(pt) for pt in paths]
        imgs = [
            visualize_bbox(img, bbox, class_name, (0, 255, 0))
            for img, bbox, class_name in zip(imgs, bboxs, cats_name)
        ]

        imgs = [
            cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC) for img in imgs
        ]

        return imgs

    def __getitem__(
            self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Function called by the epoch iterator
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # get annotation id-> target -> image id
        img_id = self.ids[index]
        targets = self.coco.imgToAnns[img_id].copy()
        chosen_target = random.choice(targets)
        targets.remove(chosen_target)
        distractors = random.choices(targets, k=self.distractors)

        path = self.coco.loadImgs(img_id)[0]["file_name"]

        # get image
        path = os.path.join(self.root, path)
        img_original = PIL.Image.open(path)

        # convert to rgb if grayscale
        if img_original.mode != "RGB":
            img_original = img_original.convert("RGB")

        # extract segment and choose target
        chosen_sgm = self.extract_segmented(img_original, chosen_target)
        distractors_sgm = [self.extract_segmented(img_original, x) for x in distractors]

        # if segmented area is empty get next item
        if reduce(lambda x, y: x * y, chosen_sgm.size) == 0:
            return self.__getitem__(index + 1)

        try:

            # Resize and normalize images
            resized_image = self.base_transform(img_original)

            chosen_sgm = self.base_transform(chosen_sgm)

            distractors_sgm = [self.base_transform(x) for x in distractors_sgm]

        except cv2.error:
            print(f"Faulty image at index {index}")
            return self.__getitem__(index + 1)

        # preprocess segments
        segments = [chosen_sgm] + distractors_sgm

        # define sender input
        sender_inp = torch.cat((resized_image, segments[0]), dim=-1)

        # shuffle segments
        segments = [x.unsqueeze(dim=0) for x in segments]
        indices = list(range(len(segments)))
        random.shuffle(indices)
        shuffled_segs = [segments[idx] for idx in indices]
        segments = torch.cat(shuffled_segs, dim=0)

        labels = [chosen_target] + distractors
        # labels are : position of true seg, category of segment, image id, annotation id
        labels = [
            torch.LongTensor([indices[0], x["category_id"], img_id, x["id"]])
            for x in labels
        ]
        labels = [x.unsqueeze(dim=0) for x in labels]
        labels = torch.cat(labels, dim=0)

        return sender_inp, segments, labels

    @staticmethod
    def extract_segmented(img: PIL.Image, target: dict) -> PIL.Image:
        """
        Choose random target and extract segment
        """
        bbox = target["bbox"]
        # from (x,y,w,h) to (x,y,x2,y2)
        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        sgm = img.crop(bbox)

        return sgm

    def __len__(self):
        return len(self.ids)


def torch_transformations(input_size: int) -> album.Compose:
    """
    Return transformation for sender and receiver
    Check albumentations site for other transformations
    https://albumentations-demo.herokuapp.com/
    """

    base_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ],
    )

    return base_transform


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

    # stack on torch tensor
    sender_inp = torch.stack(sender_inp).contiguous()
    seg = torch.stack(seg).contiguous()
    labels = torch.stack(labels).contiguous()

    # transpose to have [batch,discriminants ...]-> [discriminants, batch, ...]
    # seg= np.transpose(seg, axes=(1, 0, 2, 3, 4))
    # labels= np.transpose(labels, axes=(1, 0, 2))

    return sender_inp, labels, seg




def get_data(
        opts: Namespace,
):
    """
    Get train and validation data loader
    the path should be of the form:
    - path2data

    -- images
    --- train2017

    -- annotations
    --- instances_train2017.json
    """

    path2imgs = opts.data_root + "/"
    path2json = opts.data_root + "/annotations/"

    base_trans = torch_transformations(opts.image_resize)

    # generate datasets
    coco_train = CocoDetection(
        root=path2imgs + "train2017",
        ann_file=path2json + "instances_train2017.json",
        perc_ids=opts.train_data_perc,
        base_transform=base_trans,
        distractors=opts.distractors,
    )

    coco_val = CocoDetection(
        root=path2imgs + "val2017",
        ann_file=path2json + "instances_val2017.json",
        perc_ids=opts.val_data_perc,
        base_transform=base_trans,
        distractors=opts.distractors,
    )

    check_same_classes(train_data=coco_train,val_data=coco_val, min_annotations=opts.distractors+1)

    if opts.num_workers > 0:
        timeout = 10
    else:
        timeout = 0

    # generate dataloaders
    coco_train = DataLoader(
        coco_train,
        shuffle=True,
        drop_last=True,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
        timeout=timeout,
    )
    coco_val = DataLoader(
        coco_val,
        shuffle=True,
        drop_last=True,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
        timeout=timeout,
    )

    return coco_train, coco_val
