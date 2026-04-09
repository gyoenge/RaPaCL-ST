from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import load_yaml, parse_common_args, apply_cli_overrides
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything

import transtab
from transtab import constants
from transtab.modeling_transtab import TransTabForCL
from transtab.trainer import Trainer
from transtab.trainer_utils import TransTabCollatorForCL

try:
    import wandb
except ImportError:
    wandb = None


def read_feature_list(txt_path: str | Path) -> list[str]:
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Feature txt file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip().lower() for line in f if line.strip()]


def load_custom_transtab_dataset(
    data_root: str | Path,
    data_csv: str,
    numerical_feature_file: str,
    target_col: str,
    binary_feature_file: str | None = None,
):
    data_root = Path(data_root)

    csv_path = data_root / data_csv
    num_path = data_root / numerical_feature_file

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not num_path.exists():
        raise FileNotFoundError(f"Numerical feature file not found: {num_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(c).lower() for c in df.columns]
    target_col = target_col.lower()

    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    numerical_columns = read_feature_list(num_path)

    binary_columns = []
    if binary_feature_file is not None:
        bin_path = data_root / binary_feature_file
        if bin_path.exists():
            binary_columns = read_feature_list(bin_path)

    numerical_columns = [c for c in numerical_columns if c in df.columns and c != target_col]
    binary_columns = [c for c in binary_columns if c in df.columns and c != target_col]

    used = set(numerical_columns) | set(binary_columns) | {target_col}
    categorical_columns = [c for c in df.columns if c not in used]

    x = df.drop(columns=[target_col])
    y = df[target_col]

    return (x, y), categorical_columns, numerical_columns, binary_columns


def build_contrastive_learner(
    categorical_columns=None,
    numerical_columns=None,
    binary_columns=None,
    projection_dim=128,
    num_partition=3,
    overlap_ratio=0.5,
    supervised=True,
    hidden_dim=128,
    num_layer=2,
    num_attention_head=8,
    hidden_dropout_prob=0,
    ffn_dim=256,
    activation="relu",
    device="cuda:0",
    checkpoint=None,
    ignore_duplicate_cols=True,
    **kwargs,
):
    model = TransTabForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        num_partition=num_partition,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        supervised=supervised,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        overlap_ratio=overlap_ratio,
        activation=activation,
        device=device,
    )

    if checkpoint is not None:
        model.load(checkpoint)

    collate_fn = TransTabCollatorForCL(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        overlap_ratio=overlap_ratio,
        num_partition=num_partition,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )

    if checkpoint is not None:
        extractor_state_dir = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_state_dir):
            collate_fn.feature_extractor.load(extractor_state_dir)

    return model, collate_fn


def train(
    model,
    trainset,
    valset=None,
    num_epoch=10,
    batch_size=64,
    eval_batch_size=256,
    lr=1e-4,
    weight_decay=0,
    patience=5,
    warmup_ratio=None,
    warmup_steps=None,
    eval_metric="auc",
    output_dir="./ckpt",
    collate_fn=None,
    num_workers=0,
    balance_sample=False,
    load_best_at_last=True,
    ignore_duplicate_cols=False,
    eval_less_is_better=False,
    **kwargs,
):
    if isinstance(trainset, tuple):
        trainset = [trainset]

    train_args = {
        "num_epoch": num_epoch,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "patience": patience,
        "warmup_ratio": warmup_ratio,
        "warmup_steps": warmup_steps,
        "eval_metric": eval_metric,
        "output_dir": output_dir,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "balance_sample": balance_sample,
        "load_best_at_last": load_best_at_last,
        "ignore_duplicate_cols": ignore_duplicate_cols,
        "eval_less_is_better": eval_less_is_better,
    }

    trainer = Trainer(
        model,
        trainset,
        valset,
        **train_args,
    )
    trainer.train()


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def init_wandb(cfg: dict[str, Any], run_dir: Path):
    wandb_cfg = cfg.get("wandb", {})
    enabled = wandb_cfg.get("enabled", False)

    if not enabled:
        return None

    if wandb is None:
        raise ImportError("wandb is not installed. Please `pip install wandb`.")

    run = wandb.init(
        project=wandb_cfg.get("project", cfg["experiment"].get("project_name", "default-project")),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("run_name") or cfg["experiment"]["name"],
        tags=wandb_cfg.get("tags"),
        notes=wandb_cfg.get("notes"),
        config=_flatten_dict(cfg),
        dir=str(run_dir),
    )

    wandb.save(str(run_dir / "config_final.yaml"))
    return run


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
    trainset, categorical_columns, numerical_columns, binary_columns = load_custom_transtab_dataset(
        data_root=cfg["paths"]["data_root"],
        data_csv=cfg["data"]["data_csv"],
        numerical_feature_file=cfg["data"]["numerical_feature_file"],
        binary_feature_file=cfg["data"].get("binary_feature_file"),
        target_col=cfg["data"]["target_col"],
    )

    logger.info("Detected column types:")
    logger.info("  categorical: %d", len(categorical_columns))
    logger.info("  numerical  : %d", len(numerical_columns))
    logger.info("  binary     : %d", len(binary_columns))

    save_column_info(
        run_dir=run_dir,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
    )

    wb_run = init_wandb(cfg, run_dir)
    if wb_run is not None:
        wandb.config.update(
            {
                "num_categorical_columns": len(categorical_columns),
                "num_numerical_columns": len(numerical_columns),
                "num_binary_columns": len(binary_columns),
            }
        )

    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    # model, collate_fn = build_contrastive_learner(
    #     categorical_columns=categorical_columns,
    #     numerical_columns=numerical_columns,
    #     binary_columns=binary_columns,
    #     projection_dim=model_cfg["projection_dim"],
    #     num_partition=model_cfg["num_partition"],
    #     overlap_ratio=model_cfg["overlap_ratio"],
    #     supervised=model_cfg["supervised"],
    #     hidden_dim=model_cfg["hidden_dim"],
    #     num_layer=model_cfg["num_layer"],
    #     num_attention_head=model_cfg["num_attention_head"],
    #     hidden_dropout_prob=model_cfg["hidden_dropout_prob"],
    #     ffn_dim=model_cfg["ffn_dim"],
    #     activation=model_cfg["activation"],
    #     device=cfg["runtime"]["device"],
    #     ignore_duplicate_cols=model_cfg["ignore_duplicate_cols"],
    # )
    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols=categorical_columns,
        num_cols=numerical_columns,
        bin_cols=binary_columns,
        supervised=model_cfg["supervised"],
        num_partition=model_cfg["num_partition"],
        overlap_ratio=model_cfg["overlap_ratio"],
    )

    if args.mode in {"all", "train"}:
        logger.info("Start training...")
        train(
            model=model,
            trainset=trainset,
            valset=None,
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

    if wb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
