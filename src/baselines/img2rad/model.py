from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class PatchImgEncoder(nn.Module):
    def __init__(self, backbone_weight_path: Optional[str] = None, device: str = "cpu"):
        super().__init__()
        self.backbone = models.densenet121(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 1024

        if backbone_weight_path is not None:
            if not os.path.exists(backbone_weight_path):
                raise FileNotFoundError(
                    f"Backbone checkpoint not found: {backbone_weight_path}"
                )
            state_dict = torch.load(backbone_weight_path, map_location=device)
            self.backbone.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x


class ImgToRadiomicsModel(nn.Module):
    def __init__(
        self,
        radiomics_dim: int,
        backbone_weight_path: Optional[str] = None,
        device: str = "cpu",
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_encoder = PatchImgEncoder(backbone_weight_path, device)

        h1, h2 = hidden_dims
        self.rad_head = nn.Sequential(
            nn.Linear(self.img_encoder.output_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, radiomics_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb = self.img_encoder(x)
        rad_pred = self.rad_head(img_emb)
        return rad_pred

    def forward_with_feature(self, x: torch.Tensor):
        img_emb = self.img_encoder(x)
        rad_pred = self.rad_head(img_emb)
        return img_emb, rad_pred


class ImgFeaturePlusRadPredToGeneModel(nn.Module):
    def __init__(
        self,
        pretrained_img2rad_model: ImgToRadiomicsModel,
        num_genes: int,
        radiomics_dim: int,
        freeze_img2rad: bool = False,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img2rad = pretrained_img2rad_model

        if freeze_img2rad:
            for p in self.img2rad.parameters():
                p.requires_grad = False

        h1, h2 = hidden_dims
        self.gene_head = nn.Sequential(
            nn.Linear(self.img2rad.img_encoder.output_dim + radiomics_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, num_genes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img_emb, rad_pred = self.img2rad.forward_with_feature(x)
        fused = torch.cat([img_emb, rad_pred], dim=1)
        gene_pred = self.gene_head(fused)
        return gene_pred