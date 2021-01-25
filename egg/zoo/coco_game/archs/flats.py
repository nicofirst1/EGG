import torch
from torch import nn


class FlatModule(nn.Module):
    """
    Module to flatten the output of the feature extraction model.
    """

    out_dim: int = 512

    def __init__(self, size: int):
        super(FlatModule, self).__init__()
        size = tuple([size for _ in range(2)])
        self.out_dim *= size[0] * size[1]
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class AvgPool(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    def __init__(self, size):
        super().__init__(size)
        self.aap = nn.AdaptiveAvgPool2d(self.size)

    def forward(self, x):
        x = self.aap(x)
        flat = x.view([x.shape[0], -1])
        return flat


class MaxPool(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    def __init__(self, size):
        super().__init__(size)
        self.amp = nn.AdaptiveMaxPool2d(self.size)

    def forward(self, x):
        x = self.amp(x)
        flat = x.view([x.shape[0], -1])
        return flat


class AvgMaxPoolCat(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    def __init__(self, size):
        super().__init__(size)
        self.amp = nn.AdaptiveMaxPool2d(self.size)
        self.aap = nn.AdaptiveAvgPool2d(self.size)
        self.out_dim *= 2

    def forward(self, x):
        x1 = self.aap(x)
        x2 = self.amp(x)
        x3 = torch.cat((x1, x2), dim=-1)
        flat = x3.view([x3.shape[0], -1])
        return flat


class AvgMaxPoolMul(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    def __init__(self, size):
        super().__init__(size)
        self.amp = nn.AdaptiveMaxPool2d(self.size)
        self.aap = nn.AdaptiveAvgPool2d(self.size)

    def forward(self, x):
        x1 = self.aap(x)
        x2 = self.amp(x)
        x3 = torch.mul(x1, x2)
        flat = x3.view([x3.shape[0], -1])
        return flat
