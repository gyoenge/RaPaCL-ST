from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42) -> None:
    """seed everything"""
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """ensure directory"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_yaml(data: dict[str, Any], save_path: str | Path) -> None:
    """save yaml"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


### 

def unwrap_dataset(ds):
    """unwrap dataset: TransTab load_data 결과가 (X, y) 일 수도 있고 [(X, y)] 일 수도 있어서 정리"""
    if isinstance(ds, (list, tuple)) and len(ds) == 1:
        inner = ds[0]
        if isinstance(inner, (list, tuple)) and len(inner) == 2:
            return inner[0], inner[1]
    if isinstance(ds, (list, tuple)) and len(ds) == 2:
        return ds[0], ds[1]
    raise ValueError(f"Unexpected dataset format: type={type(ds)}, repr={repr(ds)[:300]}")

def save_column_info(
    run_dir: Path,
    categorical_columns: list[str],
    numerical_columns: list[str],
    binary_columns: list[str],
) -> None:
    """save column information"""
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