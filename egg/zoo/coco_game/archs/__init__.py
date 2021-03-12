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



def get_head(head_choice: str) -> HeadModule:
    if head_choice in HEAD_CHOICES.keys():
        return HEAD_CHOICES[head_choice]
    else:
        raise KeyError(f"No head module '{head_choice}' found")


