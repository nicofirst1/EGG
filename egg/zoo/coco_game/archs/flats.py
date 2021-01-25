import torch
from torch import nn


class FlatModule(nn.Module):
    """
    Module to flatten the output of the feature extraction model.
    """

    out_dim: int

    def __init__(self):
        super(FlatModule,self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class AvgPool(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    out_dim: int = 512

    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(sz)

    def forward(self, x):
        x = self.aap(x)
        flat = x.view([x.shape[0], -1])
        return flat


class MaxPool(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    out_dim: int = 512

    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.amp = nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        x = self.amp(x)
        flat = x.view([x.shape[0], -1])
        return flat


class AvgMaxPoolCat(FlatModule):
    """
    Module to flatten the output of the feature extraction model.
    """

    out_dim: int = 512 * 2

    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.amp = nn.AdaptiveMaxPool2d(sz)
        self.aap = nn.AdaptiveAvgPool2d(sz)

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

    out_dim: int = 512

    def __init__(self, sz=(1, 1)):
        super().__init__()
        self.amp = nn.AdaptiveMaxPool2d(sz)
        self.aap = nn.AdaptiveAvgPool2d(sz)

    def forward(self, x):
        x1 = self.aap(x)
        x2 = self.amp(x)
        x3 = torch.mul(x1, x2)
        flat = x3.view([x3.shape[0], -1])
        return flat
