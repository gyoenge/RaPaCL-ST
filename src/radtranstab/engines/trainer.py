import os
import json
import argparse

import torch
from torch.utils.data import DataLoader

import transtab

from radtranstab.data.dataset import MyDataset, radiomics_collate_fn
from radtranstab.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from radtranstab.engines.trainer_utils import (
    set_seed,
    load_model_radiomics_from_full_checkpoint,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train radiomics-only TransTab model.")

    parser.add_argument("--train_jsonl_file", type=str, required=True)
    parser.add_argument("--val_jsonl_file", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--hdf5_file", type=str, required=True)

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--num_class", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layer", type=int, default=2)
    parser.add_argument("--projection_dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--use_amp", action="store_true")

    return parser.parse_args()


def build_model_radiomics(args, device):
    model_radiomics = transtab.build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=args.num_class,
        hidden_dim=args.hidden_dim,
        num_layer=args.num_layer,
        hidden_dropout_prob=args.dropout,
        projection_dim=args.projection_dim,
        activation="leakyrelu",
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
        ape_drop_rate=0.0,
        device=device,
    )

    # APE 없이 학습하므로 아래 로직은 사용하지 않는다.
    # model_radiomics.input_encoder.feature_extractor.update(num=ape_cols)

    return model_radiomics.to(device)


def build_dataloaders(args):
    train_dataset = MyDataset(
        jsonl_file=args.train_jsonl_file,
        hdf5_file=args.hdf5_file,
        root_dir=args.root_dir,
        is_train=True,
    )

    val_dataset = MyDataset(
        jsonl_file=args.val_jsonl_file,
        hdf5_file=args.hdf5_file,
        root_dir=args.root_dir,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=radiomics_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=radiomics_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model_radiomics = build_model_radiomics(args, device)

    if args.checkpoint_path is not None:
        load_model_radiomics_from_full_checkpoint(
            model_radiomics=model_radiomics,
            checkpoint_path=args.checkpoint_path,
            device=device,
            strict=False,
        )

    train_loader, val_loader = build_dataloaders(args)

    optimizer = torch.optim.AdamW(
        model_radiomics.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.amp.GradScaler() if args.use_amp else None

    best_val_loss = float("inf")
    history = {
        "train": {},
        "val": {},
    }

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model_radiomics=model_radiomics,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model_radiomics=model_radiomics,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        history["train"][epoch] = train_metrics
        history["val"][epoch] = val_metrics

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_path = save_checkpoint(
                output_dir=args.output_dir,
                model_radiomics=model_radiomics,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                args=vars(args),
                name="best_model_radiomics",
            )

            print(f"Saved best checkpoint: {save_path}")

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()

