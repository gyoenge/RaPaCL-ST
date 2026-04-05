from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

import densenet


class STNet(nn.Module):
    def __init__(self, num_genes: int = 250, pretrained: bool = True):
        super().__init__()

        if pretrained:
            self.backbone = models.densenet121(weights="DEFAULT").features
        else:
            self.backbone = densenet._densenet121().feature

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def build_model(
    num_genes: int = 250,
    pretrained: bool = True,
    backbone_name: str = "densenet121",
) -> STNet:
    return STNet(
        num_genes=num_genes,
        backbone_name=backbone_name,
        pretrained=pretrained,
    )