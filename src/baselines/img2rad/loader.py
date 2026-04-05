from __future__ import annotations

import logging
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from .cache import load_samplewise_radiomics_targets
from .dataset import RadiomicsTargetDataset, STNetDataset


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ]
    )
    return train_transform, eval_transform


def build_fold_csv_paths(bench_data_root: str, outer_fold: int):
    train_csv = os.path.join(bench_data_root, f"splits/train_{outer_fold}.csv")
    test_csv = os.path.join(bench_data_root, f"splits/test_{outer_fold}.csv")

    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    return train_csv, test_csv


def _dataloader_kwargs(cfg: dict) -> dict:
    train_cfg = cfg.get("train", {})
    return {
        "num_workers": int(train_cfg.get("num_workers", 0)),
        "pin_memory": bool(train_cfg.get("pin_memory", True)),
    }


def build_radiomics_dataloaders(
    cfg: dict,
    gene_list_path: str,
    outer_fold: int,
    logger: logging.Logger,
):
    bench_data_root = cfg["paths"]["bench_data_root"]
    batch_size = int(cfg["train"]["batch_size"])
    normalize_expression = bool(cfg.get("data", {}).get("normalize_gene_expression", True))

    radiomics_parquet_dir = cfg["data"]["radiomics_parquet_dir"]
    radiomics_key_column = cfg.get("data", {}).get("radiomics_key_column", "barcode")
    radiomics_ignore_columns = cfg.get("data", {}).get(
        "radiomics_ignore_columns",
        ["sample_id", "patch_idx", "patch_id", "barcode", "status"],
    )

    train_transform, eval_transform = build_transforms()
    train_csv, test_csv = build_fold_csv_paths(bench_data_root, outer_fold)

    train_base_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_csv_path=train_csv,
        transforms_=train_transform,
        normalize_expression=normalize_expression,
    )
    test_base_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_csv_path=test_csv,
        transforms_=eval_transform,
        normalize_expression=normalize_expression,
    )

    train_rad_targets, feature_names = load_samplewise_radiomics_targets(
        base_dataset=train_base_dataset,
        radiomics_parquet_dir=radiomics_parquet_dir,
        logger=logger,
        key_column=radiomics_key_column,
        ignore_columns=radiomics_ignore_columns,
    )
    test_rad_targets, _ = load_samplewise_radiomics_targets(
        base_dataset=test_base_dataset,
        radiomics_parquet_dir=radiomics_parquet_dir,
        logger=logger,
        key_column=radiomics_key_column,
        ignore_columns=radiomics_ignore_columns,
    )

    apply_train_split_scaling = bool(
        cfg.get("data", {}).get("radiomics_apply_train_split_scaling", False)
    )

    if apply_train_split_scaling:
        logger.info("[Radiomics] Applying train-split normalization")

        train_mean = train_rad_targets.mean(dim=0, keepdim=True)
        train_std = train_rad_targets.std(dim=0, keepdim=True).clamp_min(1e-6)

        train_rad_targets = (train_rad_targets - train_mean) / train_std
        test_rad_targets = (test_rad_targets - train_mean) / train_std

    radiomics_dim = int(train_rad_targets.shape[1])

    train_dataset = RadiomicsTargetDataset(train_base_dataset, train_rad_targets)
    test_dataset = RadiomicsTargetDataset(test_base_dataset, test_rad_targets)

    dl_kwargs = _dataloader_kwargs(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    return train_loader, test_loader, radiomics_dim, feature_names


def build_gene_dataloaders(
    cfg: dict,
    gene_list_path: str,
    outer_fold: int,
):
    bench_data_root = cfg["paths"]["bench_data_root"]
    batch_size = int(cfg["train"]["batch_size"])
    normalize_expression = bool(cfg.get("data", {}).get("normalize_gene_expression", True))

    train_transform, eval_transform = build_transforms()
    train_csv, test_csv = build_fold_csv_paths(bench_data_root, outer_fold)

    train_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_csv_path=train_csv,
        transforms_=train_transform,
        normalize_expression=normalize_expression,
    )
    test_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_csv_path=test_csv,
        transforms_=eval_transform,
        normalize_expression=normalize_expression,
    )

    _, sample_target = train_dataset[0]
    num_genes = int(sample_target.shape[0])

    dl_kwargs = _dataloader_kwargs(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dl_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )

    return train_loader, test_loader, num_genes


def build_test_loader(
    cfg: dict,
    gene_list_path: str,
    outer_fold: int,
):
    bench_data_root = cfg["paths"]["bench_data_root"]
    batch_size = int(cfg["train"]["batch_size"])
    normalize_expression = bool(cfg.get("data", {}).get("normalize_gene_expression", True))

    _, eval_transform = build_transforms()
    _, test_csv = build_fold_csv_paths(bench_data_root, outer_fold)

    test_dataset = STNetDataset(
        bench_data_root=bench_data_root,
        gene_list_path=gene_list_path,
        split_csv_path=test_csv,
        transforms_=eval_transform,
        normalize_expression=normalize_expression,
    )

    _, sample_target = test_dataset[0]
    num_genes = int(sample_target.shape[0])

    dl_kwargs = _dataloader_kwargs(cfg)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dl_kwargs,
    )
    return test_loader, num_genes