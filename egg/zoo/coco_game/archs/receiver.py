import torch
from torch import nn

from egg.zoo.coco_game.archs import HeadModule, get_head, get_vision_dim


def build_receiver(feature_extractor: nn.Module, opts) -> nn.Module:
    head_module: HeadModule = get_head(opts.head_choice)

    head_module = head_module(
        signal_dim=opts.sender_hidden,
        vision_dim=get_vision_dim(),
        hidden_dim=opts.box_head_hidden,
        distractors=opts.distractors,
        batch_size=opts.batch_size,
    )

    rec = Receiver(
        feature_extractor=feature_extractor,
        head_module=head_module,
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
    ):
        super(Receiver, self).__init__()
        self.feature_extractor = feature_extractor

        self.head_module = head_module

    def forward(self, signal, image):
        # image is of dimension [ batch, discriminants, channels=3, img_h, img_w]

        with torch.no_grad():
            self.feature_extractor.eval()
            vision_out=self.extract_features(image)


        class_logits = self.head_module(signal, vision_out)
        # class_logits [batch, distractors]

        return class_logits

 

    def extract_features(self, image):
        vision_out = []
        for idx in range(image.shape[1]):
            seg = image[:, idx]
            vso = self.feature_extractor(seg)
            vso = vso.squeeze()
            vision_out.append(vso)

        vision_out = torch.stack(vision_out)
        # from [distractors,batch, features] to [batch, distractors, features]
        vision_out = vision_out.permute((1, 0, 2))
        return vision_out