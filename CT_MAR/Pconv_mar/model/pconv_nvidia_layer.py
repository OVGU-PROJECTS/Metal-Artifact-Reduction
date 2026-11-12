"""Adapter around NVIDIA's original "PartialConv2d".

The operator is vendored unchanged in "model/nvidia_partialconv2d.py", from
NVIDIA/partialconv commit a99cd7c under its BSD license. This only adds
PConv-MAR's same padding and the mask arguments. The layer stays named
"img_conv" so the old "*.img_conv.weight" / ".bias" checkpoint keys still load.
"""
import torch.nn as nn

from model.nvidia_partialconv2d import PartialConv2d


class PConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.img_conv = PartialConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,   # same padding, odd kernels
            bias=True,
            multi_channel=True,
            return_mask=True,
        )
        # match the legacy operator's init; nn.Conv2d defaults to Kaiming-uniform
        # with non-zero bias, which would change the starting point of the study.
        nn.init.xavier_uniform_(self.img_conv.weight, gain=1.0)
        nn.init.zeros_(self.img_conv.bias)

    def forward(self, img, mask):
        return self.img_conv(img, mask)
