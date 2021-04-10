import os
from argparse import Namespace
from functools import reduce
from pathlib import Path
from typing import Tuple

import PIL
import cv2
import numpy as np
import torch
import torchvision
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import Compose

from egg.zoo.coco_game.utils.dataset_utils import collate, filter_distractors
from egg.zoo.coco_game.utils.utils import console
from egg.zoo.coco_game.utils.vis_utils import visualize_bbox, numpy2pil


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
            base_transform: Compose,
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
        super(CocoDetection, self).__init__(root, None, None, None)

        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.base_transform = base_transform
        self.distractors = distractors
        self.random_state = np.random.RandomState(data_seed)


    def delete_rand_items(self, perc_ids: float):
        """
        Delete percentage of dataset randomly for testing
        """
        n = int((1 - perc_ids) * len(self.ids))
        original_len = len(self.ids)
        to_delete = self.random_state.choice(
            range(len(self.ids)), size=n, replace=False
        )
        to_delete = set(to_delete)

        ids = [x for i, x in enumerate(self.ids) if i not in to_delete]
        ids = sorted(ids)
        new_len = len(ids)

        console.log(
            f"Perc filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )
        self.ids = ids

    def init_dicts(self):
        imgToAnns, catToImgs = {}, {}

        for ann in self.coco.anns.values():
            if ann["image_id"] not in imgToAnns.keys():
                imgToAnns[ann["image_id"]] = []
            if ann["category_id"] not in catToImgs.keys():
                catToImgs[ann["category_id"]] = []

            imgToAnns[ann["image_id"]].append(ann)
            catToImgs[ann["category_id"]].append(ann["image_id"])

        imgs2rm = set(self.coco.imgs.keys()) - set(imgToAnns.keys())
        self.coco.imgs = {k: v for k, v in self.coco.imgs.items() if k not in imgs2rm}

        self.coco.imgToAnns = imgToAnns
        self.coco.catToImgs = catToImgs
        self.ids = list(self.coco.imgs.keys())

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
        chosen_target = self.random_state.choice(targets)
        targets.remove(chosen_target)
        distractors = self.random_state.choice(targets, size=self.distractors)
        distractors = list(distractors)

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
            print(f"Segment area is zero at index {index}")
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
        sender_inp = torch.stack((resized_image, segments[0]))

        # shuffle segments
        segments = [imgs.unsqueeze(dim=0) for imgs in segments]
        indices = list(range(len(segments)))
        self.random_state.shuffle(indices)
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


def torch_transformations(input_size: int) -> Compose:
    """
    Return transformation for sender and receiver
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
        base_transform=base_trans,
        distractors=opts.distractors,
        data_seed=opts.data_seed,
    )



    coco_val = CocoDetection(
        root=path2imgs + "val2017",
        ann_file=path2json + "instances_val2017.json",
        base_transform=base_trans,
        distractors=opts.distractors,
        data_seed=opts.data_seed + 1,

    )

    filter_distractors(
        train_data=coco_train, val_data=coco_val, min_annotations=opts.distractors + 1
    )

    coco_train.delete_rand_items(opts.train_data_perc)
    coco_val.delete_rand_items(opts.val_data_perc)

    if opts.num_workers > 0:
        timeout = 10
    else:
        timeout = 0

    # generate dataloaders
    coco_train = DataLoader(
        coco_train,
        shuffle=True,
        drop_last=False,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
        timeout=timeout,
    )
    coco_val = DataLoader(
        coco_val,
        shuffle=True,
        drop_last=False,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
        timeout=timeout,
    )

    return coco_train, coco_val
