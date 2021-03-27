import torch
from torch import nn

from egg.zoo.coco_game.archs import get_vision_dim
from egg.zoo.coco_game.utils.utils import console


def build_sender(feature_extractor, opts):
    sender = VisionSender(
        feature_extractor,
        image_size=opts.image_resize,
        image_type=opts.image_type,
        image_union=opts.image_union,
        n_hidden=opts.sender_hidden,
        out_features=get_vision_dim(),

    )

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
            self,
            model,
            image_size: int,
            image_type: str,
            image_union: str,
            out_features: int,
            n_hidden: int = 10,
    ):
        super(VisionSender, self).__init__()

        self.feature_extractor = model
        self.image_size = image_size
        self.image_type = image_type
        self.image_union = image_union

        vision_dim = get_vision_dim()

        if vision_dim == n_hidden:
            console.log("Using Identity for sender out")
            self.fc = nn.Identity()
        else:
            console.log(f"Using Linear vision2hidden= [{vision_dim}]->[{n_hidden}] for sender out")
            self.fc = nn.Linear(vision_dim, n_hidden)

        if image_type == "both" and image_union == "cat":
            self.cat_fc = nn.Linear(2 * out_features, out_features)

    def forward(self, inp):
        """
        inp : tuple (image, segmented): containing original image and segmented part
        """

        # inp is of dimension [batch,2,  channels=3, img_h, img_w]
        # get the output of the vision module
        with torch.no_grad():
            self.feature_extractor.eval()
            vision_out = self.process_image(inp)


        # vision out [batch , vision out]
        # then fc on vision out
        fc_out = self.fc(vision_out)

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
            vision_in = vision_in.squeeze()
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
        img = inp[:, 0, :, :]
        seg = inp[:, 1, :, :]
        # img and seg = [batch, channels, image_size, image_size]

        if self.image_type == "seg":
            out = self.feature_extractor(seg)

        elif self.image_type == "img":
            out = self.feature_extractor(img)

        else:
            # both, apply vision to both
            segment_out = self.feature_extractor(seg)
            image_out = self.feature_extractor(img)
            out = self.combine_images(segment_out, image_out)

        out = out.squeeze()
        return out
