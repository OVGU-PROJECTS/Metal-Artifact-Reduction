"""Embedding networks for DML-based invariance evaluation.

The embedding follows Eulig et al.'s evaluation idea: a ResNet maps CT patches
to a compact, L2-normalized feature vector. Triplet training then makes
same-case invariance samples close to the original input and different-case
samples farther away.
"""

from __future__ import annotations

from argparse import Namespace

import torch
import torch.nn as nn
from torchvision import models


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, models.ResNet101_Weights.IMAGENET1K_V1),
}


class MetricResNet(nn.Module):
    """ResNet feature encoder with an L2-normalized embedding head."""

    def __init__(
        self,
        *,
        in_ch: int = 1,
        embedding_dim: int = 128,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. Use one of {sorted(_BACKBONES)}.")

        builder, weights_enum = _BACKBONES[backbone]
        weights = weights_enum if pretrained else None
        self.backbone = builder(weights=weights)

        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_ch,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        if pretrained:
            with torch.no_grad():
                if in_ch == 1:
                    self.backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif in_ch == 3:
                    self.backbone.conv1.weight.copy_(old_conv.weight)
                else:
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    self.backbone.conv1.weight.copy_(mean_weight.repeat(1, in_ch, 1, 1) / float(in_ch))

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(in_features, int(embedding_dim)),
        )

    @classmethod
    def from_args(cls, args: Namespace) -> "MetricResNet":
        return cls(
            in_ch=int(getattr(args, "in_ch", 1)),
            embedding_dim=int(getattr(args, "embedding_dim", 128)),
            backbone=str(getattr(args, "backbone", "resnet50")),
            pretrained=bool(getattr(args, "pretrained", True)),
            dropout=float(getattr(args, "dropout", 0.0)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.head(features)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)

