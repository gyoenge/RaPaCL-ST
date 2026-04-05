from __future__ import annotations

import os
from pathlib import Path

import torch

from src.baselines.img2rad.evaluator import run_all_folds_pcc_eval
from src.baselines.img2rad.trainer import run_all_folds_training
from src.common.config import apply_cli_overrides, load_yaml, parse_common_args
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything


def build_gene_list_path(cfg: dict) -> str:
    bench_data_root = cfg["paths"]["bench_data_root"]
    genes_criteria = cfg["model"]["genes_criteria"]
    num_genes = cfg["model"]["num_genes"]
    return os.path.join(
        bench_data_root,
        f"{genes_criteria}_{num_genes}genes.json",
    )


def resolve_device(cfg: dict) -> torch.device:
    requested = cfg["runtime"].get("device", "cpu")
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_common_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed = int(cfg.get("seed", 42))
    seed_everything(seed)

    log_dir = cfg["paths"].get("log_dir", "logs")
    timestamp, logger = setup_logger(log_dir=log_dir, name="img2rad")

    device = resolve_device(cfg)
    logger.info("device: %s", device)

    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])
    gene_list_path = build_gene_list_path(cfg)

    save_yaml(cfg, Path(checkpoint_dir) / f"config_{timestamp}.yaml")

    logger.info("Configuration:")
    for top_key, top_value in cfg.items():
        logger.info("  %s: %s", top_key, top_value)

    train_reports = None
    if args.mode == "train" or cfg["runtime"].get("run_train", True):
        train_reports = run_all_folds_training(
            cfg=cfg,
            gene_list_path=gene_list_path,
            device=device,
            logger=logger,
        )

    if args.mode == "eval" or cfg["runtime"].get("run_eval", True):
        radiomics_dim = cfg["model"].get("radiomics_dim")
        if radiomics_dim is None:
            if train_reports is None or len(train_reports) == 0:
                raise ValueError(
                    "radiomics_dim is not available. "
                    "Either run training first or set model.radiomics_dim in config."
                )
            radiomics_dim = int(train_reports[0]["radiomics_dim"])

        aggregate_summary = run_all_folds_pcc_eval(
            cfg=cfg,
            gene_list_path=gene_list_path,
            radiomics_dim=int(radiomics_dim),
            device=device,
            timestamp=timestamp,
            logger=logger,
        )

        logger.info("=" * 100)
        logger.info("Evaluation done")
        if aggregate_summary is not None:
            logger.info(
                "Final macro mean PCC across folds = %.6f",
                aggregate_summary["macro_mean_pcc_across_folds"],
            )

    logger.info("=" * 100)
    logger.info("Pipeline finished")


if __name__ == "__main__":
    main()