from egg.zoo.coco_game.archs.heads import *

HEAD_CHOICES = dict(
    simple=SimpleHead,
    sequential=SequentialHead,
    only_signal=OnlySignal,
    signal_expansion_mul=SignalExpansionMul,
    feature_reduction_mul=FeatureReductionMul,
    signal_expansion_cat=SignalExpansionCat,
    feature_reduction_cat=FeatureReductionCat
)


def get_head(head_choice: str) -> HeadModule:
    if head_choice in HEAD_CHOICES.keys():
        return HEAD_CHOICES[head_choice]
    else:
        raise KeyError(f"No head module '{head_choice}' found")


