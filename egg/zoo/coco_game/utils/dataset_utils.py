from typing import List

from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

from egg.zoo.coco_game.utils.utils import console


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
        f"A total of {len(new_train_cats)} classes are left from the distractor filtering\n"
        f"Filtered classes are : {removed_cats}"
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
