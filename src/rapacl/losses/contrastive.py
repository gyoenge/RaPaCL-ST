import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalNTXentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, patch_proj, radiomics_multiview_proj, sample_ids):
        # patch_proj: [B, P]
        # radiomics_multiview_proj: [B, V, P]
        ...
        return loss