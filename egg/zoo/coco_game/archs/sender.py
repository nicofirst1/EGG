from pathlib import Path

import torch
from torch import nn

from egg.zoo.coco_game.archs import get_flat, FlatModule
from egg.zoo.coco_game.utils.utils import load_pretrained_sender


def build_sender(feature_extractor, opts, pretrain=False):
    flat_module = get_flat(opts.flat_choice_sender)()

    sender = VisionSender(
        feature_extractor,
        flat_module=flat_module,
        image_size=opts.image_resize,
        image_type=opts.image_type,
        image_union=opts.image_union,
        n_hidden=opts.sender_hidden,
        num_classes=opts.num_classes,
        pretraining=pretrain,
    )

    if opts.sender_pretrain != "":
        load_pretrained_sender(Path(opts.sender_pretrain), sender)
    return sender


class VisionSender(nn.Module):
    """
    Vision sender module. Uses pretrained model for latent space of image and pass it to fc to build message.

    Depending on image_type and image_union the process changes as follows

    - image_type in ["seg", "img"] : the sender input is either the entire image (does not know what to point at)
     or the segment. The input is passed to the vision pretrained model and its output to a fc

                                                    vision                              fc
     Dimensions schema : [batch size, 3, img_h, img_w] -> [batch size, vision model out] -> [batch size, n_hidden]

    - image_type=="both" & image_union=="mul" : The input is a tuple where both the original image and the segment
     are present. Both are passed to the vision module separately and the result is multiplied together before being
      passed to the final fc

                                                            vision                               mul
      Dimensions schema : 2x  [batch size, 3, img_h, img_w] -> 2x [batch size, vision model out] ->
                                        fc
      1 x [batch size, vision model out] -> [batch size, n_hidden]


    - image_type=="both" & image_union=="cat" : The input is a tuple where both the original image and the segment
     are present. Both are passed to the vision module separately and the result is concatenated together.
     The resulting tensor is passed to another fc (cat_fc) and then to the final one


                                                            vision                               cat
      Dimensions schema : 2x  [batch size, 3, img_h, img_w] -> 2x [batch size, vision model out] ->
                                            cat_fc                            fc
      1 x [batch size, 2* vision model out] -> [batch size, vision model out] -> [batch size, n_hidden]


    """

    def __init__(
            self, model, flat_module: FlatModule, image_size: int, image_type: str, image_union: str, num_classes: int,
            n_hidden: int = 10, pretraining: bool = False,
    ):
        super(VisionSender, self).__init__()

        self.vision = model
        self.image_size = image_size
        self.image_type = image_type
        self.image_union = image_union
        self.out_features = flat_module.out_dim

        self.fc = nn.Linear(self.out_features, n_hidden)
        self.flat_module = flat_module

        if image_type == "both" and image_union == "cat":
            self.cat_fc = nn.Linear(2 * self.out_features, self.out_features)

        self.class_fc = nn.Linear(n_hidden, num_classes)
        self.pretraining = pretraining
        self.fc_out = None

    def predict_class(self, inp):
        class_logits = self.class_fc(self.fc_out)
        return class_logits

    def forward(self, inp):
        """
        inp : tuple (image, segmented): containing original image and segmented part
        """

        # inp is of dimension [batch, channels=3, img_h, img_w * 2]
        # get the output of the vision module
        vision_out = self.process_image(inp)

        # vision out [batch , vision out]
        # then fc on vision out
        fc_out = self.fc(vision_out)
        self.fc_out = fc_out

        # if pretrain then train the sender to recognize the class based on the fc_out
        if self.pretraining:
            class_logit = self.class_fc(fc_out)
            return class_logit

        # fc_out [batch, hidden size]
        return fc_out

    def combine_images(self, segment_out, image_out):
        """
        Get the vision out of both segment and image and decide how to combine them, either by mul or cat

        Returns:

        """
        # if the image union is concatenate then vision_in [batch size, 2* vision_out],
        # so use linear to get to  [batch size, vision_out]
        if self.image_union == "cat":
            vision_in = torch.cat((segment_out, image_out), dim=1)
            vision_out = self.cat_fc(vision_in)
        # else just multiply
        else:
            vision_out = torch.mul(segment_out, image_out)

        return vision_out

    def process_image(self, inp):
        """
        process the correct images based on user preferences
        Args:
            inp: input passed to forward, it is made of the original image and the segment concatenated on the last dim

        Returns: vision module output

        """

        # split image and segment
        img = inp[:, :, :, : self.image_size]
        seg = inp[:, :, :, self.image_size:]
        # img and seg = [batch, channels, image_size, image_size]

        if self.image_type == "seg":
            out = self.vision(seg)
            out = self.flat_module(out)

        elif self.image_type == "img":
            out = self.vision(img)
            out = self.flat_module(out)

        else:
            # both, apply vision to both
            segment_out = self.vision(seg)
            image_out = self.vision(img)
            segment_out = self.flat_module(segment_out)
            image_out = self.flat_module(image_out)
            out = self.combine_images(segment_out, image_out)

        return out
