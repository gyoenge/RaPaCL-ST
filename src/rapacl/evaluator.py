import os
from typing import Dict, List, Optional, Any

import torch


class RaPaCLEvaluator:
    def __init__(
        self,
        model,
        criterion,
        device: torch.device,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def load_checkpoint(self, checkpoint_path: str, strict: bool = True) -> Dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        else:
            self.model.load_state_dict(checkpoint, strict=strict)

        return checkpoint

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        self.model.eval()

        running = {
            "total": 0.0,
            "contrastive": 0.0,
            "patch_cls": 0.0,
            "radiomics_cls": 0.0,
            "batch": 0.0,
        }

        total_samples = 0
        patch_correct = 0
        radiomics_correct = 0
        patch_count = 0
        radiomics_count = 0

        for batch in data_loader:
            batch = self._move_batch_to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    patch_x=batch["patch_x"],
                    radiomics_x=batch["radiomics_x"],
                )
                loss_dict = self.criterion(outputs, batch)

            bs = batch["labels"].shape[0]
            total_samples += bs

            for key in running.keys():
                if key in loss_dict:
                    running[key] += float(loss_dict[key].detach().item()) * bs

            labels = batch["labels"]

            if "patch_logits" in outputs:
                patch_logits = outputs["patch_logits"]
                if patch_logits.shape[-1] == 1:
                    patch_pred = (torch.sigmoid(patch_logits.squeeze(-1)) > 0.5).long()
                else:
                    patch_pred = torch.argmax(patch_logits, dim=1)
                patch_correct += (patch_pred == labels).sum().item()
                patch_count += bs

            if "radiomics_logits" in outputs:
                rad_logits = outputs["radiomics_logits"]
                if rad_logits.shape[-1] == 1:
                    rad_pred = (torch.sigmoid(rad_logits.squeeze(-1)) > 0.5).long()
                else:
                    rad_pred = torch.argmax(rad_logits, dim=1)
                radiomics_correct += (rad_pred == labels).sum().item()
                radiomics_count += bs

        if total_samples == 0:
            raise RuntimeError("Empty dataloader passed to evaluator.")

        metrics = {}
        for key, value in running.items():
            metrics[key] = value / total_samples

        if patch_count > 0:
            metrics["patch_acc"] = patch_correct / patch_count
        if radiomics_count > 0:
            metrics["radiomics_acc"] = radiomics_correct / radiomics_count

        return metrics

    @torch.no_grad()
    def collect_outputs(self, data_loader) -> Dict[str, Any]:
        self.model.eval()

        collected = {
            "labels": [],
            "sample_ids": [],
            "patch_logits": [],
            "radiomics_logits": [],
            "patch_feat": [],
            "patch_proj": [],
            "radiomics_cls_feat": [],
            "radiomics_ctr_feat": [],
            "radiomics_ctr_proj": [],
            "radiomics_multiview_feat": [],
            "radiomics_multiview_proj": [],
        }

        if hasattr(self.model, "use_batch_correction") and self.model.use_batch_correction:
            collected["patch_batch_logits"] = []
            collected["radiomics_batch_logits"] = []

        for batch in data_loader:
            batch = self._move_batch_to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    patch_x=batch["patch_x"],
                    radiomics_x=batch["radiomics_x"],
                )

            if "labels" in batch:
                collected["labels"].append(batch["labels"].detach().cpu())
            if "sample_ids" in batch:
                collected["sample_ids"].append(batch["sample_ids"].detach().cpu())

            for key in list(collected.keys()):
                if key in outputs:
                    collected[key].append(outputs[key].detach().cpu())

        merged = {}
        for key, value_list in collected.items():
            if len(value_list) == 0:
                continue

            first = value_list[0]
            if torch.is_tensor(first):
                merged[key] = torch.cat(value_list, dim=0)
            else:
                merged[key] = value_list

        return merged
