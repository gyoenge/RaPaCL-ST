from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import load_yaml, parse_common_args, apply_cli_overrides
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything

from transtab import (
    load_data, 
    build_contrastive_learner,
    train
)

def save_column_info(
    run_dir: Path,
    categorical_columns: list[str],
    numerical_columns: list[str],
    binary_columns: list[str],
) -> None:
    info = {
        "categorical_columns": categorical_columns,
        "numerical_columns": numerical_columns,
        "binary_columns": binary_columns,
        "num_categorical": len(categorical_columns),
        "num_numerical": len(numerical_columns),
        "num_binary": len(binary_columns),
    }

    with open(run_dir / "column_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_common_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg["mode"] = args.mode

    seed = cfg.get("seed", 42)
    seed_everything(seed)

    log_dir = cfg["paths"]["log_dir"]
    timestamp, logger = setup_logger(log_dir, name="pretrain_transtab")

    logger.info("Loaded config from: %s", args.config)
    logger.info("Execution mode: %s", args.mode)
    logger.info("Device: %s", cfg["runtime"].get("device"))
    logger.info("Batch size: %s", cfg["train"].get("batch_size"))
    logger.info("Learning rate: %s", cfg["train"].get("lr"))

    output_root = ensure_dir(cfg["paths"]["output_root"])
    run_dir = ensure_dir(output_root / f"run_{timestamp}")
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])

    if cfg["experiment"].get("save_config", True):
        save_yaml(cfg, run_dir / "config_final.yaml")
        logger.info("Final config saved to: %s", run_dir / "config_final.yaml")

    logger.info("Preparing custom TransTab dataset from: %s", cfg["paths"]["data_root"])

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = load_data([f'{cfg["paths"]["data_root"]}'])
    
    logger.info("Detected column types:")
    logger.info("  categorical: %d", len(cat_cols))
    logger.info("  numerical  : %d", len(num_cols))
    logger.info("  binary     : %d", len(bin_cols))
    save_column_info(
        run_dir=run_dir,
        categorical_columns=cat_cols,
        numerical_columns=num_cols,
        binary_columns=bin_cols,
    )

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    model, collate_fn = build_contrastive_learner(
        cat_cols=cat_cols,
        num_cols=num_cols,
        bin_cols=bin_cols,
        supervised=model_cfg["supervised"],
        num_partition=model_cfg["num_partition"],
        overlap_ratio=model_cfg["overlap_ratio"],
    )

    if args.mode in {"all", "train"}:
        logger.info("Start training...")
        train(
            model=model,
            trainset=trainset,
            valset=valset,
            num_epoch=train_cfg["max_epochs"],
            batch_size=train_cfg["batch_size"],
            eval_batch_size=train_cfg["eval_batch_size"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            patience=train_cfg["patience"],
            warmup_ratio=train_cfg["warmup_ratio"],
            warmup_steps=train_cfg["warmup_steps"],
            eval_metric=train_cfg["eval_metric"],
            output_dir=str(checkpoint_dir),
            collate_fn=collate_fn,
            num_workers=train_cfg["num_workers"],
            ignore_duplicate_cols=model_cfg["ignore_duplicate_cols"],
            eval_less_is_better=train_cfg["eval_less_is_better"],
        )
        logger.info("Training finished.")

if __name__ == "__main__":
    main()
