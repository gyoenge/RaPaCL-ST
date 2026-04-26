import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model_radiomics_from_full_checkpoint(
    model_radiomics: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = False,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)

    prefixes = [
        "model_radiomics.",
        "radiomics_model.",
        "module.model_radiomics.",
        "module.radiomics_model.",
    ]

    model_keys = set(model_radiomics.state_dict().keys())
    radiomics_state = {}

    for key, value in state_dict.items():
        loaded = False

        for prefix in prefixes:
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                radiomics_state[new_key] = value
                loaded = True
                break

        if not loaded and key in model_keys:
            radiomics_state[key] = value

    if len(radiomics_state) == 0:
        raise ValueError(
            f"No model_radiomics weights found in checkpoint: {checkpoint_path}"
        )

    result = model_radiomics.load_state_dict(radiomics_state, strict=strict)

    print(f"[Checkpoint] Loaded from: {checkpoint_path}")
    print(f"[Checkpoint] Loaded keys: {len(radiomics_state)}")
    print(f"[Checkpoint] Missing keys: {len(result.missing_keys)}")
    print(f"[Checkpoint] Unexpected keys: {len(result.unexpected_keys)}")

    if result.missing_keys:
        print("[Checkpoint] Missing:")
        for key in result.missing_keys:
            print(f"  - {key}")

    if result.unexpected_keys:
        print("[Checkpoint] Unexpected:")
        for key in result.unexpected_keys:
            print(f"  - {key}")


def save_checkpoint(
    output_dir: str,
    model_radiomics: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    args: Optional[Dict[str, Any]] = None,
    name: str = "checkpoint",
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"{name}_epoch{epoch}.pth")

    torch.save(
        {
            "epoch": epoch,
            "model_radiomics": model_radiomics.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "args": args or {},
        },
        save_path,
    )

    return save_path


def train_one_epoch(
    model_radiomics,
    loader,
    optimizer,
    criterion,
    device,
    epoch: int,
    scaler=None,
):
    model_radiomics.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Train Epoch {epoch}")

    for batch in pbar:
        radiomics_features = batch["radiomics_features"]
        labels = torch.tensor(batch["labels"], device=device).long()

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                embeddings, logits = model_radiomics(radiomics_features)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            embeddings, logits = model_radiomics(radiomics_features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(
            {
                "loss": loss_meter.avg,
                "acc": acc_meter.avg,
            }
        )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
    }


@torch.no_grad()
def evaluate(
    model_radiomics,
    loader,
    criterion,
    device,
    epoch: int,
):
    model_radiomics.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc=f"Val Epoch {epoch}")

    for batch in pbar:
        radiomics_features = batch["radiomics_features"]
        labels = torch.tensor(batch["labels"], device=device).long()

        embeddings, logits = model_radiomics(radiomics_features)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(acc, batch_size)

        pbar.set_postfix(
            {
                "loss": loss_meter.avg,
                "acc": acc_meter.avg,
            }
        )

    return {
        "loss": loss_meter.avg,
        "acc": acc_meter.avg,
    }

