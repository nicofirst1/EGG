from egg.zoo.coco_game.archs.flats import *
from egg.zoo.coco_game.archs.heads import *

HEAD_CHOICES = dict(
    only_signal=OnlySignal,
    only_image=OnlyImage,
    signal_expansion=SignalExpansion,
    feature_reduction=FeatureReduction,
    random_signal=RandomSignal,
    random_signal_img=RandomSignalImg,
    Conv=Conv,
    Conv2=Conv2,
)

FLAT_CHOICES = dict(
    AvgPool=AvgPool,
    MaxPool=MaxPool,
)


def get_head(head_choice: str) -> HeadModule:
    if head_choice in HEAD_CHOICES.keys():
        return HEAD_CHOICES[head_choice]
    else:
        raise KeyError(f"No head module '{head_choice}' found")


def get_flat(flat_choice: str) -> FlatModule:
    if flat_choice in FLAT_CHOICES.keys():
        return FLAT_CHOICES[flat_choice]
    else:
        raise KeyError(f"No flat module '{flat_choice}' found")


def get_out_features(flat_choice):
    flat = get_flat(flat_choice)
    return flat.out_dim
