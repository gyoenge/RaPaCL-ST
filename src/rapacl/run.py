import os
import random
import argparse
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from src.rapacl.models.rapacl import RaPaCL
from src.rapacl.losses.rapacl_criterion import RaPaCLCriterion
from src.rapacl.trainers import RaPaCLTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RaPaCL training entrypoint.")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./outputs/rapacl")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--patch_pretrained", action="store_true")
    parser.add_argument("--patch_feat_dim", type=int, default=1024)

    parser.add_argument("--radiomics_hidden_dim", type=int, default=128)
    parser.add_argument("--num_sub_cols", type=int, nargs="+", default=[93, 62, 31, 15, 7, 3, 1])
    parser.add_argument("--separate_contrast_token", action="store_true")

    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--projection_use_mlp", action="store_true")
    parser.add_argument("--projection_hidden_dim", type=int, default=None)
    parser.add_argument("--projection_dropout", type=float, default=0.0)

    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--classifier_dropout", type=float, default=0.0)

    parser.add_argument("--use_batch_correction", action="store_true")
    parser.add_argument("--num_batch_labels", type=int, default=None)
    parser.add_argument("--batch_disc_hidden_dim", type=int, default=None)
    parser.add_argument("--batch_disc_nlayers", type=int, default=2)
    parser.add_argument("--batch_disc_dropout", type=float, default=0.1)
    parser.add_argument("--batch_disc_grl_lambda", type=float, default=1.0)

    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_patch_cls", type=float, default=1.0)
    parser.add_argument("--lambda_rad_cls", type=float, default=1.0)
    parser.add_argument("--lambda_batch", type=float, default=1.0)

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_radiomics_features", type=int, default=93)

    parser.add_argument("--debug_forward", action="store_true")
    parser.add_argument("--use_dummy_data", action="store_true")

    return parser.parse_args()


class DummyRaPaCLDataset(Dataset):
    def __init__(
        self,
        length: int,
        image_size: int,
        num_radiomics_features: int,
        num_classes: int,
        use_batch_correction: bool,
        num_batch_labels: Optional[int],
    ) -> None:
        self.length = length
        self.image_size = image_size
        self.num_radiomics_features = num_radiomics_features
        self.num_classes = num_classes
        self.use_batch_correction = use_batch_correction
        self.num_batch_labels = num_batch_labels

        self.columns = [f"rad_feat_{i}" for i in range(num_radiomics_features)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        patch_x = torch.randn(3, self.image_size, self.image_size)
        radiomics_x = np.random.randn(self.num_radiomics_features).astype(np.float32)
        label = np.random.randint(0, self.num_classes)
        sample_id = idx

        item = {
            "patch_x": patch_x,
            "radiomics_x": radiomics_x,
            "labels": label,
            "sample_ids": sample_id,
        }

        if self.use_batch_correction:
            item["batch_labels"] = np.random.randint(0, self.num_batch_labels)

        return item


def dummy_collate_fn(batch):
    patch_x = torch.stack([item["patch_x"] for item in batch], dim=0)

    rad_array = np.stack([item["radiomics_x"] for item in batch], axis=0)
    rad_columns = [f"rad_feat_{i}" for i in range(rad_array.shape[1])]
    radiomics_x = pd.DataFrame(rad_array, columns=rad_columns)

    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    sample_ids = torch.tensor([item["sample_ids"] for item in batch], dtype=torch.long)

    output = {
        "patch_x": patch_x,
        "radiomics_x": radiomics_x,
        "labels": labels,
        "sample_ids": sample_ids,
    }

    if "batch_labels" in batch[0]:
        output["batch_labels"] = torch.tensor(
            [item["batch_labels"] for item in batch],
            dtype=torch.long,
        )

    return output


def build_model(args) -> RaPaCL:
    numerical_columns = [f"rad_feat_{i}" for i in range(args.num_radiomics_features)]

    model = RaPaCL(
        patch_pretrained=args.patch_pretrained,
        patch_feat_dim=args.patch_feat_dim,
        radiomics_hidden_dim=args.radiomics_hidden_dim,
        proj_dim=args.proj_dim,
        projection_use_mlp=args.projection_use_mlp,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        num_classes=args.num_classes,
        classifier_dropout=args.classifier_dropout,
        use_batch_correction=args.use_batch_correction,
        num_batch_labels=args.num_batch_labels,
        batch_disc_hidden_dim=args.batch_disc_hidden_dim,
        batch_disc_nlayers=args.batch_disc_nlayers,
        batch_disc_dropout=args.batch_disc_dropout,
        batch_disc_grl_lambda=args.batch_disc_grl_lambda,
        separate_contrast_token=args.separate_contrast_token,
        numerical_columns=numerical_columns,
        num_sub_cols=args.num_sub_cols,
        hidden_dim=args.radiomics_hidden_dim,
        device=args.device,
    )
    return model


def build_criterion(args) -> RaPaCLCriterion:
    return RaPaCLCriterion(
        lambda_contrastive=args.lambda_contrastive,
        lambda_patch_cls=args.lambda_patch_cls,
        lambda_rad_cls=args.lambda_rad_cls,
        lambda_batch=args.lambda_batch,
        temperature=args.temperature,
        use_batch_correction=args.use_batch_correction,
    )


def build_dummy_loaders(args):
    train_dataset = DummyRaPaCLDataset(
        length=32,
        image_size=args.image_size,
        num_radiomics_features=args.num_radiomics_features,
        num_classes=args.num_classes,
        use_batch_correction=args.use_batch_correction,
        num_batch_labels=args.num_batch_labels,
    )
    val_dataset = DummyRaPaCLDataset(
        length=16,
        image_size=args.image_size,
        num_radiomics_features=args.num_radiomics_features,
        num_classes=args.num_classes,
        use_batch_correction=args.use_batch_correction,
        num_batch_labels=args.num_batch_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dummy_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dummy_collate_fn,
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model(args).to(device)
    criterion = build_criterion(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer = RaPaCLTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        output_dir=args.output_dir,
        use_amp=args.use_amp,
        log_interval=10,
    )

    print("[INFO] Model / criterion / optimizer / trainer initialized.")

    if args.use_dummy_data:
        train_loader, val_loader = build_dummy_loaders(args)
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.epochs,
            save_every=1,
            monitor="total",
        )
        print("[INFO] Training finished.")
        return

    print("[INFO] No real dataloader is connected yet.")
    print("[INFO] Add your actual Dataset/DataLoader in run.py and pass to trainer.fit(...).")


if __name__ == "__main__":
    main()
