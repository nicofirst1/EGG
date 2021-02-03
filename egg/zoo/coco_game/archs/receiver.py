from torch import nn

from egg.zoo.coco_game.archs import HeadModule, get_head, get_flat, FlatModule


def build_receiver(feature_extractor: nn.Module, opts) -> nn.Module:
    head_module: HeadModule = get_head(opts.head_choice)
    flat_module: FlatModule = get_flat(opts.flat_choice_receiver)()

    head_module = head_module(
        signal_dim=opts.sender_hidden,
        vision_dim=flat_module.out_dim,
        num_classes=opts.num_classes,
        hidden_dim=opts.box_head_hidden,
    )

    rec = Receiver(
        feature_extractor=feature_extractor,
        head_module=head_module,
        flat_module=flat_module
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
        # image is of dimension [batch, channels=3, img_h, img_w]
        vision_out = self.feature_extractor(image)
        vision_out = self.flat_module(vision_out)

        signal = signal.float()
        # signal [batch, vocab]
        # signal= torch.rand(signal.shape).to(signal)

        class_logits = self.box_module(signal, vision_out)
        # class_logits [batch, num_classes]

        return class_logits
