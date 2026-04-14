import torch
import torch.nn as nn

from .contrastive import MultiModalNTXentLoss
from .classification import PatchClassificationLoss, RadiomicsClassificationLoss
from .batchcorrection import BatchCorrectionLoss


class RaPaCLCriterion(nn.Module):
    def __init__(
        self,
        lambda_contrastive=1.0,
        lambda_patch_cls=1.0,
        lambda_rad_cls=1.0,
        lambda_batch=1.0,
        temperature=0.07,
        use_batch_correction=True,
    ):
        super().__init__()
        self.lambda_contrastive = lambda_contrastive
        self.lambda_patch_cls = lambda_patch_cls
        self.lambda_rad_cls = lambda_rad_cls
        self.lambda_batch = lambda_batch
        self.use_batch_correction = use_batch_correction

        self.contrastive_loss = MultiModalNTXentLoss(temperature=temperature)
        self.patch_cls_loss = PatchClassificationLoss()
        self.rad_cls_loss = RadiomicsClassificationLoss()
        self.batch_loss = BatchCorrectionLoss()

    def forward(self, outputs, batch):
        loss_dict = {}

        loss_dict["contrastive"] = self.contrastive_loss(
            outputs["patch_proj"],
            outputs["radiomics_multiview_proj"],
            batch["sample_ids"],
        )

        loss_dict["patch_cls"] = self.patch_cls_loss(
            outputs["patch_logits"],
            batch["labels"],
        )

        loss_dict["radiomics_cls"] = self.rad_cls_loss(
            outputs["radiomics_logits"],
            batch["labels"],
        )

        if self.use_batch_correction:
            loss_dict["patch_batch"] = self.batch_loss(
                outputs["patch_batch_logits"],
                batch["batch_labels"],
            )
            loss_dict["radiomics_batch"] = self.batch_loss(
                outputs["radiomics_batch_logits"],
                batch["batch_labels"],
            )
            loss_dict["batch"] = loss_dict["patch_batch"] + loss_dict["radiomics_batch"]
        else:
            loss_dict["batch"] = 0.0 * loss_dict["contrastive"]

        total = (
            self.lambda_contrastive * loss_dict["contrastive"]
            + self.lambda_patch_cls * loss_dict["patch_cls"]
            + self.lambda_rad_cls * loss_dict["radiomics_cls"]
            + self.lambda_batch * loss_dict["batch"]
        )
        loss_dict["total"] = total
        return loss_dict