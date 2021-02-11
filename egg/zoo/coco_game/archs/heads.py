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


########################################################
#       ABSTRACT CLASS
#########################################################


class HeadModule(nn.Module):
    """
    Common implementation for the head modules which is used in the receiver taking as input the vision feature and the signal
    """

    def __init__(
        self,
        signal_dim: int,
        vision_dim: int,
        num_classes: int,
        hidden_dim: int,
        distractors: int,
        batch_size: int,
    ):
        """
        Args:
            signal_dim: length of the signal coming from the sender
            vision_dim: dimension of the pretrained vision out
            num_classes: number of classes for the classification task
            hidden_dim: hidden dimension for the Linear modules
            distractors: number of distractors present in the input
        """
        super().__init__()
        self.output_size = distractors + 1
        self.signal_dim = signal_dim
        self.vision_dim = vision_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self) -> torch.nn.Sequential:
        """
        Build the model used for prediction
        """
        raise NotImplemented()

    def signal_expansion(self, signal):
        """
        Repeat the signal vectors for every input image
        """
        # copy the signal N times where N is the number of distractors +1
        signal = signal.unsqueeze(dim=0).repeat(self.output_size, 1, 1)
        signal = signal.permute((1, 0, 2))
        return signal

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            signal: The sender signal
            vision_features: Feature extractor output

        Returns: Logits on classes. Note that the softmax MUST NOT be computed on the classes output since it is computed later on

        """
        raise NotImplemented()


########################################################
#       LINEAR HEADS
#########################################################


class FeatureReduction(HeadModule):
    """

    Consider the number N=distractors +1


    Feature reduction works as the default discrimination pipeline.
    It reduces the number of vision features to match the signal length
    [batch size, N, vision features]  -> [batch size, N, signal length]

    It then adds a dimension to the signal
    [batch size, signal length]  -> [batch size, signal length, 1]

    In order to allow for a matrix multiplication between vision feature and signal
                vision                          signal
     [batch size, N, signal length] x  [batch size, signal length, 1]

    which yields a class logit of size
    [batch size, N 1]

    Where the last dimension is removed with a squeeze

    """

    def build_model(self) -> torch.nn.Sequential:
        return nn.Sequential(nn.Linear(self.vision_dim, self.signal_dim))

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        vision_features = self.model(vision_features).tanh()
        signal = torch.unsqueeze(signal, dim=-1)
        out = torch.matmul(vision_features, signal)
        out = out.squeeze()

        return out


class SignalExpansion(HeadModule):
    """
    Consider the number N=distractors +1


    It works in the opposite way as FeatureReduction
    It increases the number of signal length to match vision features
    [batch size, N, signal length] -> [batch size, N, vision features]

    It then adds a dimension to the signal
    [batch size, vision features]  -> [batch size, vision features, 1]

    In order to allow for a matrix multiplication between vision feature and signal which yields a class logit of size
    [batch size, N 1]

    Where the last dimension is removed with a squeeze
    """

    def build_model(self) -> torch.nn.Sequential:
        return nn.Sequential(nn.Linear(self.signal_dim, self.vision_dim))

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        signal = self.model(signal).tanh()
        signal = torch.unsqueeze(signal, dim=-1)
        out = torch.matmul(vision_features, signal)
        out = out.squeeze()

        return out


########################################################
#       CONV HEADS
#########################################################


class Conv(HeadModule):
    """

    Consider the number N=distractors +1

    In a similar fashion to FeatureReduction, reduces the dimension of the vision to the signal one.
    [batch size, N, vision features] ->  [batch size, N, signal length]

    Then it repeats the signal on the second dimension to match the vision one
    [batch size, signal length] ->  [batch size, N, signal length]

    Then it multiplies the signal with the vision vector

    The output's dimension is increased to reach 4
    [batch size, N, signal length, 1]

    Then a convolution can be applied. The convolution is applied in order to reduce the second dimension from 'N' to
     '1' with the features and the kernel is a non squared size of (signal length - N,1)
     to reduce the last dimension to the output.
     [batch size, N, signal length, 1] ->[batch size, 1, N, 1]

     The result is squeezed to get a class logits of size
    [batch size, N]

    """

    def build_model(self) -> torch.nn.Sequential:
        self.embed = nn.Linear(self.vision_dim, self.signal_dim)
        return nn.Sequential(
            nn.Conv2d(
                self.output_size, 1, kernel_size=(self.signal_dim - self.output_size, 1)
            ),
        )

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        vision_features = self.embed(vision_features)
        signal = self.signal_expansion(signal)
        out = torch.mul(vision_features, signal)
        out = out.unsqueeze(dim=-1)
        class_logits = self.model(out).clone()
        class_logits = class_logits.squeeze()

        return class_logits


class Conv2(HeadModule):
    """

    Consider the number N=distractors +1

    In a similar fashion to FeatureReduction, reduces the dimension of the vision to the signal one.
    [batch size, N, vision features] ->  [batch size, N, signal length]

    Then it repeats the signal on the second dimension to match the vision one
    [batch size, signal length] ->  [batch size, N, signal length]

    Then it multiplies the signal with the vision vector

    The output's dimension is increased to reach 4
    [batch size, N, signal length, 1]

    Then a convolution can be applied. The convolution is applied with a   non squared  kernel of size size (signal length ,1)
     to reduce the last dimension to the output.
     [batch size, N, signal length, 1] ->[batch size, N, 1, 1]

     The result is squeezed to get a class logits of size
    [batch size, N]

    """

    def build_model(self) -> torch.nn.Sequential:
        self.embed = nn.Linear(self.vision_dim, self.signal_dim)
        return nn.Sequential(
            nn.Conv2d(
                self.output_size, self.output_size, kernel_size=(self.signal_dim, 1)
            ),
        )

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        vision_features = self.embed(vision_features)
        signal = self.signal_expansion(signal)
        out = torch.mul(vision_features, signal)
        out = out.unsqueeze(dim=-1)
        class_logits = self.model(out).clone()
        class_logits = class_logits.squeeze()

        return class_logits


########################################################
#       CONTROL HEADS
#########################################################


class OnlySignal(HeadModule):
    """
    Uses only the given signal with a linear for classification
    """

    def build_model(self) -> torch.nn.Sequential:
        return nn.Sequential(nn.Linear(self.signal_dim, self.output_size))

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        class_logits = self.model(signal).clone()

        return class_logits


class RandomSignal(HeadModule):
    """
    Generates a random signal and uses it with a linear for classification
    """

    def build_model(self) -> torch.nn.Sequential:
        return nn.Sequential(nn.Linear(self.signal_dim, self.output_size))

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        signal = torch.rand(signal.shape).to(signal.device)
        class_logits = self.model(signal).clone()

        return class_logits


class RandomSignalImg(HeadModule):
    """
    Generates a randomm signal and uses it with the image in a similar fashion to FeatureReduction
    """

    def build_model(self) -> torch.nn.Sequential:
        return nn.Sequential(nn.Linear(self.vision_dim, self.signal_dim))

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        signal = torch.rand(signal.shape).to(signal.device)

        vision_features = self.model(vision_features).tanh()
        signal = torch.unsqueeze(signal, dim=-1)
        class_logits = torch.matmul(vision_features, signal)
        class_logits = class_logits.squeeze()
        return class_logits


class OnlyImage(HeadModule):
    """
    Uses only the image for classification with a linear model
    """

    def build_model(self) -> torch.nn.Sequential:
        return nn.Sequential(nn.Linear(self.vision_dim, 1))

    def forward(
        self, signal: torch.Tensor, vision_features: torch.Tensor
    ) -> torch.Tensor:
        class_logits = self.model(vision_features).clone()
        class_logits = torch.squeeze(class_logits)

        return class_logits
