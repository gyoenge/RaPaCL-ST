import os
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader


class RaPaCLTrainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device: torch.device,
        output_dir: str,
        scheduler=None,
        use_amp: bool = False,
        log_interval: int = 10,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.use_amp = use_amp
        self.log_interval = log_interval

        os.makedirs(self.output_dir, exist_ok=True)

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _move_batch_to_device(self, batch: Dict):
        moved = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                moved[k] = v.to(self.device)
            else:
                moved[k] = v
        return moved

    def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()

        running = {
            "total": 0.0,
            "contrastive": 0.0,
            "patch_cls": 0.0,
            "radiomics_cls": 0.0,
            "batch": 0.0,
        }

        num_steps = 0

        for step, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    patch_x=batch["patch_x"],
                    radiomics_x=batch["radiomics_x"],
                )
                loss_dict = self.criterion(outputs, batch)
                loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            for key in running.keys():
                if key in loss_dict:
                    running[key] += float(loss_dict[key].detach().item())

            num_steps += 1

            if step % self.log_interval == 0:
                msg = (
                    f"[Train] Epoch {epoch} Step {step}/{len(train_loader)} "
                    f"total={loss_dict['total'].item():.4f}"
                )
                if "contrastive" in loss_dict:
                    msg += f" contrastive={loss_dict['contrastive'].item():.4f}"
                if "patch_cls" in loss_dict:
                    msg += f" patch_cls={loss_dict['patch_cls'].item():.4f}"
                if "radiomics_cls" in loss_dict:
                    msg += f" rad_cls={loss_dict['radiomics_cls'].item():.4f}"
                if "batch" in loss_dict:
                    msg += f" batch={loss_dict['batch'].item():.4f}"
                print(msg)

        if num_steps == 0:
            raise RuntimeError("Empty train_loader.")

        for key in running.keys():
            running[key] /= num_steps

        return running

    @torch.no_grad()
    def validate_one_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()

        running = {
            "total": 0.0,
            "contrastive": 0.0,
            "patch_cls": 0.0,
            "radiomics_cls": 0.0,
            "batch": 0.0,
        }

        num_steps = 0

        for step, batch in enumerate(val_loader):
            batch = self._move_batch_to_device(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(
                    patch_x=batch["patch_x"],
                    radiomics_x=batch["radiomics_x"],
                )
                loss_dict = self.criterion(outputs, batch)

            for key in running.keys():
                if key in loss_dict:
                    running[key] += float(loss_dict[key].detach().item())

            num_steps += 1

        if num_steps == 0:
            raise RuntimeError("Empty val_loader.")

        for key in running.keys():
            running[key] /= num_steps

        print(
            f"[Val] Epoch {epoch} "
            f"total={running['total']:.4f} "
            f"contrastive={running['contrastive']:.4f} "
            f"patch_cls={running['patch_cls']:.4f} "
            f"rad_cls={running['radiomics_cls']:.4f} "
            f"batch={running['batch']:.4f}"
        )

        return running

    def save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
    ) -> str:
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        ckpt_path = os.path.join(self.output_dir, filename)

        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "criterion_state_dict": self.criterion.state_dict() if hasattr(self.criterion, "state_dict") else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        torch.save(state, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
        return ckpt_path

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
        save_every: int = 1,
        monitor: str = "total",
    ):
        best_val = float("inf")

        history = {
            "train": [],
            "val": [],
        }

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_one_epoch(train_loader, epoch)
            history["train"].append(train_metrics)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.validate_one_epoch(val_loader, epoch)
                history["val"].append(val_metrics)

                current = val_metrics.get(monitor, val_metrics["total"])
                if current < best_val:
                    best_val = current
                    self.save_checkpoint(
                        epoch=epoch,
                        train_metrics=train_metrics,
                        val_metrics=val_metrics,
                        filename="best.pt",
                    )

            if epoch % save_every == 0:
                self.save_checkpoint(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    filename=f"epoch_{epoch}.pt",
                )

        return history
