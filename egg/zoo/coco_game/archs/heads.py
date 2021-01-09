# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision import models


def initialize_model():
    """
    Get pretrained model. Since we are going to use the frozen pretrained model for both receiver and sender it would
     be a waste of space to instantiate two different models and change the last layer as described in the tutorial.
     So we take one model, remove the last fc layer and directly map the output in our sender receiver.
    """
    input_size = 224
    model = models.resnet18(pretrained=True)
    # remove last fully connected
    model = nn.Sequential(*list(model.children())[:-2])

    # Parameters of newly constructed modules have requires_grad=True by default
    for param in model.parameters():
        param.requires_grad = False

    model = model.eval()
    return model, input_size


def get_out_features():
    return 512 * 2


class HeadModule(nn.Module):
    """
    Common implementation for the box modules, must take as input the given args
    """

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int):
        """
        Args:
            num_features: number of features coming from the feature extraction model
            num_classes: number of classes for the classification task
            hidden_dim: hidden dimension for the Linear modules
        """
        super(HeadModule, self).__init__()
        self.output_size = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: features coming from the concatenation of the feature extractor output and the sender signal

        Returns: Logits on classes. Note that the softmax MUST NOT be computed on the classes output since it is computed later on

        """
        raise NotImplemented()


def get_head(head_choice: str) -> HeadModule:
    if head_choice == "single":
        return SingleModule
    else:
        raise KeyError(f"No head module '{head_choice}' found")


class Flat(nn.Module):
    """
    Module to flatten the output of the feature extraction model.
    """

    def __init__(self, sz=(1, 1)):
        super(Flat, self).__init__()
        self.amp = nn.AdaptiveMaxPool2d(sz)
        self.aap = nn.AdaptiveAvgPool2d(sz)

    def forward(self, x):
        cat = torch.cat((self.amp(x), self.aap(x)), dim=1)
        flat = cat.view([cat.shape[0], -1])
        return flat


class SingleModule(HeadModule):
    """
    Usa single sequential model for both classification and regression.
    Use sigmoid for both classification and regression
    """

    def __init__(self, num_features: int, num_classes: int, hidden_dim: int):
        super(SingleModule, self).__init__(num_features, num_classes, hidden_dim)

        self.head_class = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        class_logits = self.head_class(features).clone()

        return class_logits
