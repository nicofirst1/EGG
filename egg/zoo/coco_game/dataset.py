import os
import random
from argparse import Namespace
from typing import List, Tuple

import albumentations as album
import cv2
import lycon
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

from egg.zoo.coco_game.utils.utils import console


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
            num_classes: int = 90,
            min_area: float = 0,
            skip_first: int = 5,
    ):
        """
        Custom Dataset
        Args:
            root: path to the coco dataset
            ann_file: path to coco annotations file
            base_transform: transformation used for the sender input
            perc_ids: percentage of ids to keep
            num_classes: number of classes to keep
            skip_first: number of first classes to skip, since the first 5 in coco are over represented
        """
        super(CocoDetection, self).__init__(root, None, None, None)

        self.coco = COCO(ann_file)
        self.anns_ids = list(self.coco.anns.keys())

        ############
        # Filtering
        ############
        # filter per number of classes
        # The first 5 classes are over represented
        self.over_presented = skip_first
        self.filter_anns_classes(num_classes, over_presented=self.over_presented)

        # randomly drop perc_ids
        self.anns_ids = self.delete_rand_items(
            self.anns_ids, int((1 - perc_ids) * len(self.anns_ids))
        )

        # filter per area
        self.filter_anns_area(min_area=min_area)
        # sort anns
        self.anns_ids = sorted(self.anns_ids)

        self.base_transform = base_transform

    @staticmethod
    def delete_rand_items(items: list, n: int):
        original_len = len(items)
        to_delete = set(random.sample(range(len(items)), n))
        new_len = len(items) - n

        console.log(
            f"Perc filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )
        return [x for i, x in enumerate(items) if not i in to_delete]

    def filter_anns_classes(self, num_classes: int, over_presented: int):
        """
        Filter annotations based on minimum (mean) area
        """

        original_len = len(self.anns_ids)

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
        self.anns_ids = list(self.coco.anns.keys())
        new_len = len(self.anns_ids)

        console.log(
            f"Classes filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )

    def filter_anns_area(self, min_area=0.1):
        """
        Filter annotations based on minimum (mean) area
        """
        eps = 0.0001
        original_len = len(self.anns_ids)

        for idx in range(len(self.anns_ids)):
            an_id = self.anns_ids[idx]
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
                self.anns_ids[idx] = None

        self.anns_ids = [x for x in self.anns_ids if x is not None]

        new_len = len(self.anns_ids)

        console.log(
            f"Area filtered len : {new_len}/{original_len} ({new_len / original_len * 100:.3f}%)"
        )

    def get_images(self, img_id: int, size: Tuple[int, int]) -> List[np.array]:

        paths = self.coco.loadImgs(img_id)
        paths = [os.path.join(self.root, pt['file_name']) for pt in paths]

        imgs = [lycon.load(pt) for pt in paths]
        imgs = [cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC) for img in imgs]

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
        ann_ids = self.anns_ids[index]
        target = self.coco.loadAnns(ann_ids)[0]
        img_id = target["image_id"]

        path = self.coco.loadImgs(img_id)[0]["file_name"]

        # get image
        path = os.path.join(self.root, path)
        img_original = lycon.load(path)
        # extract segment and choose target
        sgm = self.extract_segmented(img_original, target)

        # if segmented is empty get next item
        if len(sgm) == 0:
            return self.__getitem__(index + 1)

        try:

            # Resize and normalize images
            resized_image = self.base_transform(
                image=img_original,
            )['image']

            sgm = self.base_transform(
                image=sgm,
            )['image']



        except cv2.error:
            print(f"Faulty image at index {index}")
            return self.__getitem__(index + 1)

        # we save the receiver distorted image and bboxes
        labels = target["category_id"]


        # the images are of size [h,w, channels] but the model requires [channels,w,h]
        sgm = np.transpose(sgm, axes=(2, 0, 1))
        resized_image = np.transpose(resized_image, axes=(2, 0, 1))

        # transform  in torch tensor
        resized_image = torch.FloatTensor(resized_image)
        sgm = torch.FloatTensor(sgm.copy())
        labels = torch.LongTensor([labels, img_id])

        return resized_image, sgm, labels

    @staticmethod
    def extract_segmented(img: np.array, target: dict):
        """
        Choose random target and extract segment
        """
        bbox = target["bbox"]
        bbox = [int(elem) for elem in bbox]
        # order bbox for numpy crop
        bbox = [bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]]
        sgm = img[bbox[0]: bbox[2], bbox[1]: bbox[3]]

        return sgm

    def __len__(self):
        return len(self.anns_ids)


def transformations(input_size: int) -> album.Compose:
    """
    Return transformation for sender and receiver
    Check albumentations site for other transformations
    https://albumentations-demo.herokuapp.com/
    """

    base_transform = album.Compose(
        [
            album.Resize(input_size, input_size),
            album.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], max_pixel_value=255),
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
    resized_image = [elem[0] for elem in batch]
    seg = [elem[1] for elem in batch]
    labels = [elem[2] for elem in batch]

    # stack on torch tensor
    resized_image = torch.stack(resized_image).contiguous()
    seg = torch.stack(seg).contiguous()
    labels = torch.stack(labels).contiguous()

    # concat image and seg (only way to pass it to sender)
    sender_inp = torch.cat((resized_image, seg), dim=-1)

    return sender_inp, labels, resized_image


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

    path2imgs = opts.data_root + "/images/"
    path2json = opts.data_root + "/annotations/"

    base_trans = transformations(opts.image_resize)

    # generate datasets
    coco_train = CocoDetection(
        root=path2imgs + "train2017",
        ann_file=path2json + "instances_train2017.json",
        perc_ids=opts.train_data_perc,
        base_transform=base_trans,
        num_classes=opts.num_classes,
        skip_first=opts.skip_first,
        min_area=opts.min_area,
    )

    coco_val = CocoDetection(
        root=path2imgs + "val2017",
        ann_file=path2json + "instances_val2017.json",
        perc_ids=opts.test_data_perc,
        base_transform=base_trans,
        num_classes=opts.num_classes,
        skip_first=opts.skip_first,
        min_area=opts.min_area,
    )

    if opts.num_workers>0:
        timeout=10
    else:
        timeout=0

    # generate dataloaders
    coco_train = DataLoader(
        coco_train,
        shuffle=True,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
        timeout=timeout
    )
    coco_val = DataLoader(
        coco_val,
        shuffle=True,
        num_workers=opts.num_workers,
        batch_size=opts.batch_size,
        collate_fn=collate,
        timeout=timeout

    )

    return coco_train, coco_val
