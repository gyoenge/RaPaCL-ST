import torch.nn as nn


class PatchClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)


class RadiomicsClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.loss_fn(logits, targets)