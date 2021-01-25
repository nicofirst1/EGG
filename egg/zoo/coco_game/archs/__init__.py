from egg.zoo.coco_game.archs.heads import *

HEAD_CHOICES = dict(
    simple=SimpleHead,
    only_signal=OnlySignal,
    only_image=OnlyImage,
    signal_expansion_mul=SignalExpansion,
    feature_reduction__mul=FeatureReduction,
    RandomSignal=RandomSignal,
    RandomSignalImg=RandomSignalImg,
)


def get_head(head_choice: str) -> HeadModule:
    if head_choice in HEAD_CHOICES.keys():
        return HEAD_CHOICES[head_choice]
    else:
        raise KeyError(f"No head module '{head_choice}' found")


