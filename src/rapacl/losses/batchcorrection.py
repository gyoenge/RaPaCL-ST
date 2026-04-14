import torch.nn as nn


class BatchCorrectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch_logits, batch_labels):
        return self.loss_fn(batch_logits, batch_labels)
