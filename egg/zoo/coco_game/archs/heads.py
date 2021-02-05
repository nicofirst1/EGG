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
    model = models.resnet18(pretrained=True)
    # remove last fully connected
    model = nn.Sequential(*list(model.children())[:-2])

    # Parameters of newly constructed modules have requires_grad=True by default
    for param in model.parameters():
        param.requires_grad = False

    model = model.eval()
    return model


class HeadModule(nn.Module):
    """
    Common implementation for the box modules, must take as input the given args
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        """
        Args:
            signal_dim: length of the signal coming from the sender
            vision_dim: dimension of the pretrained vision out
            num_classes: number of classes for the classification task
            hidden_dim: hidden dimension for the Linear modules
        """
        super().__init__()
        self.output_size = num_classes

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal: The sender signal
            vision_features: Feature extractor output

        Returns: Logits on classes. Note that the softmax MUST NOT be computed on the classes output since it is computed later on

        """
        raise NotImplemented()


class SimpleHead(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        feature_num = signal_dim + vision_dim

        self.head_class = nn.Linear(feature_num, num_classes)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        out = torch.cat((vision_features, signal), dim=1)
        class_logits = self.head_class(out).clone()
        class_logits = torch.relu(class_logits)

        return class_logits


class OnlySignal(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        self.head_class = nn.Linear(signal_dim, self.output_size)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        class_logits = self.head_class(signal).clone()

        return class_logits


class RandomSignal(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        self.head_class = nn.Linear(signal_dim, self.output_size)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        signal = torch.rand(signal.shape).to(signal.device)
        class_logits = self.head_class(signal).clone()

        return class_logits


class RandomSignalImg(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        feature_num = signal_dim + vision_dim

        self.head_class = nn.Linear(feature_num, num_classes)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        signal = torch.rand(signal.shape).to(signal.device)

        out = torch.cat((vision_features, signal), dim=1)
        class_logits = self.head_class(out).clone()

        return class_logits


class OnlyImage(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        self.head_class = nn.Linear(vision_dim, self.output_size)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        class_logits = self.head_class(vision_features).clone()

        return class_logits


class SignalExpansion(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        self.signal_expansion = nn.Linear(signal_dim, vision_dim)
        self.head_class = nn.Linear(vision_dim, num_classes)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        signal = self.signal_expansion(signal)

        out = torch.mul(signal, vision_features)

        class_logits = self.head_class(out).clone()

        return class_logits


class FeatureReduction(HeadModule):
    """
    Usa single sequential model for  classification
    """

    def __init__(self, signal_dim: int, vision_dim: int, num_classes: int, hidden_dim: int):
        super().__init__(signal_dim, vision_dim, num_classes, hidden_dim)

        self.feature_reduction = nn.Linear(vision_dim, signal_dim)
        self.head_class = nn.Linear(signal_dim, num_classes)

    def forward(self, signal: torch.Tensor, vision_features: torch.Tensor) -> torch.Tensor:
        vision_features = self.feature_reduction(vision_features)

        out = torch.mul(signal, vision_features)

        class_logits = self.head_class(out).clone()

        return class_logits
