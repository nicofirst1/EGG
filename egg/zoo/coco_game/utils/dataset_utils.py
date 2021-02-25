from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

from egg.zoo.coco_game.utils.utils import console


def check_same_classes(train_data: CocoDetection, val_data: CocoDetection, min_annotations):
    min_perc_valid = 0.7
    train_cats = train_data.coco.cats
    train_cats = [x['name'] for x in train_cats.values()]

    stop = False
    while not stop:
        stop = filter(train_data, val_data, min_annotations, min_perc_valid)

    new_train_cats = train_data.coco.cats
    new_train_cats = [x['name'] for x in new_train_cats.values()]

    removed_cats=set(train_cats)-set(new_train_cats)

    console.log(f"A total of {len(new_train_cats)} classes are left from the distractor filtering\nFiltered classes are : {removed_cats}")


def get_annotation_stats(coco: COCO, min_annotations, min_perc_valid):
    counter = {id_: dict(total=0, valid=0) for id_ in coco.cats.keys()}

    imgs_to_rm = []
    anns2remove = []

    for img_id, anns in coco.imgToAnns.items():

        valid = 1 if len(anns) > min_annotations else 0

        if not valid:
            imgs_to_rm.append(img_id)
            anns2remove += [x["id"] for x in anns]

        for anns in anns:
            ann_id = anns["category_id"]
            counter[ann_id]["total"] += 1
            counter[ann_id]["valid"] += valid

    counter = {k: v["valid"] / v["total"] for k, v in counter.items()}
    counter = [k for k, v in counter.items() if v < min_perc_valid]

    anns2remove += [
        k for k, v in coco.anns.items() if v["category_id"] in counter
    ]

    anns2remove = set(anns2remove)

    for img_id in imgs_to_rm:
        coco.imgs.pop(img_id)

    for ann_id in anns2remove:
        coco.anns.pop(ann_id)

    for cat in counter:
        coco.cats.pop(cat)

    return counter


def remove_cats(coco: COCO, cat_list):
    ans2remove = []
    for cat in cat_list:
        images = coco.catToImgs[cat]
        for img in images:
            annotations = coco.imgToAnns[img]
            for key in range(len(annotations)):
                anns = annotations[key]
                if anns['category_id'] == cat:
                    ans2remove.append(anns['id'])

        coco.cats.pop(cat)
    ans2remove=set(ans2remove)
    for ans in ans2remove:
        coco.anns.pop(ans)


def filter(train_coco: CocoDetection, val_coco: CocoDetection, min_annotations, min_perc_valid):
    train_stats = get_annotation_stats(train_coco.coco, min_annotations, min_perc_valid)
    val_stats = get_annotation_stats(val_coco.coco, min_annotations, min_perc_valid)

    train_coco.init_dicts()
    val_coco.init_dicts()

    cats2rm_train = set(val_stats) - set(train_stats)
    cats2rm_val = set(train_stats) - set(val_stats)

    if len(cats2rm_val) != 0 or len(cats2rm_train) != 0:
        stop = False
    else:
        stop = True

    remove_cats(train_coco.coco, cats2rm_train)
    remove_cats(val_coco.coco, cats2rm_val)

    train_coco.init_dicts()
    val_coco.init_dicts()

    return stop


def filter_anns_coocurence(self, train_coco, val_colo, min_annotations: int, min_perc_valid: float = 0.77):
    """
    Filter annotations based on minimum number of co-occurences
    """

    first_len = len(self.ids)
    original_len = -1
    new_len = 0

    # need a while since every time you delete some annotations from images, some images may have not enough annotations anymore
    while original_len != new_len:
        original_len = len(self.ids)
        filter(self)
        new_len = len(self.ids)

    imgs_to_rm = set(self.coco.imgs.keys()) - set(self.coco.imgToAnns.keys())
    for img in imgs_to_rm:
        self.coco.imgs.pop(img)

    self.init_dicts()
    new_len = len(self.ids)

    console.log(
        f"Co presence filtered len : {new_len}/{first_len} ({new_len / first_len * 100:.3f}%)"
    )
