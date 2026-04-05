from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from .dataset import STNetDataset


def safe_load_gene_names(gene_list_path: str) -> List[str]:
    with open(gene_list_path, "r", encoding="utf-8") as f:
        gene_info = json.load(f)

    if isinstance(gene_info, dict):
        if "genes" in gene_info:
            return gene_info["genes"]
        if "gene_names" in gene_info:
            return gene_info["gene_names"]

    if isinstance(gene_info, list):
        return gene_info

    raise ValueError(f"Unsupported gene list format: {gene_list_path}")


def load_sample_radiomics_parquet(
    parquet_path: str,
    key_column: str,
    ignore_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Radiomics parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    if key_column not in df.columns:
        raise ValueError(
            f"Key column '{key_column}' not found in parquet: {parquet_path}. "
            f"Available columns: {list(df.columns)}"
        )

    ignore_columns = set(ignore_columns or [])
    ignore_columns.add(key_column)

    feature_columns = [col for col in df.columns if col not in ignore_columns]

    if not feature_columns:
        raise ValueError(
            f"No feature columns found in parquet: {parquet_path}. "
            f"key_column={key_column}, ignore_columns={sorted(ignore_columns)}"
        )

    df = df.copy()
    df[key_column] = df[key_column].astype(str)

    if df[key_column].duplicated().any():
        dup_keys = df.loc[df[key_column].duplicated(), key_column].unique().tolist()[:10]
        raise ValueError(
            f"Duplicated keys found in {parquet_path} for column '{key_column}': {dup_keys}"
        )

    return df, feature_columns


def load_samplewise_radiomics_targets(
    base_dataset: STNetDataset,
    radiomics_parquet_dir: str,
    logger: logging.Logger,
    key_column: str = "barcode",
    ignore_columns: list[str] | None = None,
) -> Tuple[torch.Tensor, List[str]]:
    unique_sample_ids = sorted({meta["sample_id"] for meta in base_dataset.patch_meta})
    logger.info("[RadiomicsParquet] unique samples in dataset = %d", len(unique_sample_ids))

    sample_feature_df_map: Dict[str, pd.DataFrame] = {}
    global_feature_names: list[str] | None = None

    for sample_id in unique_sample_ids:
        parquet_path = os.path.join(radiomics_parquet_dir, f"{sample_id}.parquet")
        df, feature_columns = load_sample_radiomics_parquet(
            parquet_path=parquet_path,
            key_column=key_column,
            ignore_columns=ignore_columns,
        )

        if global_feature_names is None:
            global_feature_names = feature_columns
        else:
            if feature_columns != global_feature_names:
                raise ValueError(
                    f"Feature columns mismatch for sample_id={sample_id}\n"
                    f"expected={global_feature_names[:10]} ... ({len(global_feature_names)} cols)\n"
                    f"got={feature_columns[:10]} ... ({len(feature_columns)} cols)"
                )

        sample_feature_df_map[sample_id] = df.set_index(key_column)

        logger.info(
            "[RadiomicsParquet] loaded sample=%s rows=%d features=%d",
            sample_id,
            len(df),
            len(feature_columns),
        )

    assert global_feature_names is not None

    all_features = []
    missing_rows = []

    for idx, meta in enumerate(base_dataset.patch_meta):
        sample_id = meta["sample_id"]
        barcode = str(meta["barcode"])

        if sample_id not in sample_feature_df_map:
            raise KeyError(f"sample_id missing in radiomics parquet map: {sample_id}")

        sample_df = sample_feature_df_map[sample_id]

        if barcode not in sample_df.index:
            missing_rows.append((idx, sample_id, barcode))
            continue

        feat = sample_df.loc[barcode, global_feature_names].to_numpy(dtype=np.float32)

        if feat.ndim != 1:
            raise ValueError(
                f"Expected 1D feature vector, got shape={feat.shape} "
                f"for sample_id={sample_id}, barcode={barcode}"
            )

        all_features.append(feat)

    if missing_rows:
        preview = missing_rows[:10]
        raise KeyError(
            f"{len(missing_rows)} barcodes from STNetDataset were not found in radiomics parquet. "
            f"Examples: {preview}"
        )

    rad_targets = torch.tensor(np.stack(all_features, axis=0), dtype=torch.float32)
    logger.info(
        "[RadiomicsParquet] assembled targets shape = %s",
        tuple(rad_targets.shape),
    )

    return rad_targets, global_feature_names