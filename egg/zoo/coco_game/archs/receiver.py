import torch
from torch import nn

from egg.zoo.coco_game.archs.heads import Flat, HeadModule, get_head, get_out_features


def build_receiver(feature_extractor: nn.Module, opts) -> nn.Module:
    head_module = get_head(opts.head_choice)
    num_features = get_out_features() + opts.sender_hidden

    head_module = head_module(
        num_features=num_features,
        num_classes=opts.num_classes,
        hidden_dim=opts.box_head_hidden,
    )

    rec = Receiver(
        feature_extractor=feature_extractor,
        head_module=head_module,
        embedding_size=opts.embedding_size,
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
        embedding_size: int,
    ):
        super(Receiver, self).__init__()
        self.embedding_size = embedding_size
        self.feature_extractor = feature_extractor
        self.cat_flat = Flat()

        self.box_module = head_module

    def forward(self, signal, image):
        # image is of dimension [batch, channels=3, img_h, img_w] img_h=img_w=299
        vision_out = self.feature_extractor(image)
        vision_out = self.cat_flat(vision_out)

        signal = signal.float()
        # signal [batch, vocab]

        # concat vision_out\signal to aggregate info
        out = torch.cat((vision_out, signal), dim=1)
        # out [batch, out_features + sender_hidden]

        class_logits = self.box_module(out)
        # class_logits [batch, num_classes]

        return class_logits
