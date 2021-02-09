import numpy as np
import torch
from torch import nn

from egg.zoo.coco_game.archs import FlatModule, HeadModule, get_flat, get_head


def build_receiver(feature_extractor: nn.Module, opts) -> nn.Module:
    head_module: HeadModule = get_head(opts.head_choice)
    flat_module: FlatModule = get_flat(opts.flat_choice_receiver)()

    head_module = head_module(
        signal_dim=opts.sender_hidden,
        vision_dim=flat_module.out_dim,
        num_classes=opts.num_classes,
        hidden_dim=opts.box_head_hidden,
        distractors=opts.distractors,
        batch_size=opts.batch_size,
    )

    rec = Receiver(
        feature_extractor=feature_extractor,
        head_module=head_module,
        flat_module=flat_module,
    )
    return rec


class Receiver(nn.Module):
    """
    Recevier  module, directly concatenates the output of the vision module with the signal and pass it to a fc
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        head_module: HeadModule,
        flat_module,
    ):
        super(Receiver, self).__init__()
        self.feature_extractor = feature_extractor
        self.flat_module = flat_module

        self.box_module = head_module

    def forward(self, signal, image):
        # image is of dimension [discriminants, batch, channels=3, img_h, img_w]
        vision_out = []
        for idx in range(image.shape[1]):
            seg = image[:, idx]
            vso = self.feature_extractor(seg)
            vso = self.flat_module(vso)
            vision_out.append(vso)

        vision_out = torch.stack(vision_out)
        # bring to [batch, distractors, features]
        vision_out = vision_out.permute((1, 0, 2))

        signal = signal.float()
        # copy the signal N times where N is the number of distractors +1
        signal = signal.unsqueeze(dim=0).repeat(self.box_module.output_size, 1, 1)
        signal = signal.permute((1, 0, 2))

        class_logits = self.box_module(signal, vision_out)
        # class_logits [batch, num_classes]

        return class_logits
