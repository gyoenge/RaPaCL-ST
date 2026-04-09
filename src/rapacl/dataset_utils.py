from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def normalize_column_names(df: pd.DataFrame, lowercase: bool = True) -> pd.DataFrame:
    df = df.copy()
    if lowercase:
        df.columns = [str(col).lower() for col in df.columns]
    return df


def normalize_config_columns(dataset_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dataset_cfg.copy()

    def lower_list(values: list[str] | None) -> list[str]:
        if not values:
            return []
        return [str(v).lower() for v in values]

    cfg["meta_columns"] = lower_list(cfg.get("meta_columns", []))
    cfg["exclude_columns"] = lower_list(cfg.get("exclude_columns", []))
    cfg["exclude_prefixes"] = lower_list(cfg.get("exclude_prefixes", []))

    target_column = cfg.get("target_column")
    if target_column is not None:
        cfg["target_column"] = str(target_column).lower()

    column_types = cfg.setdefault("column_types", {})
    for key in ["numerical", "binary", "categorical"]:
        block = column_types.setdefault(key, {})
        block["include_columns"] = lower_list(block.get("include_columns", []))
        block["exclude_columns"] = lower_list(block.get("exclude_columns", []))
        block["include_prefixes"] = lower_list(block.get("include_prefixes", []))

    preprocessing = cfg.setdefault("preprocessing", {})
    preprocessing.setdefault("lowercase_columns", True)
    preprocessing.setdefault("fill_missing_numerical", "mode")
    preprocessing.setdefault("fill_missing_categorical", "mode")
    preprocessing.setdefault("fill_missing_binary", "mode")
    preprocessing.setdefault("scale_numerical", True)
    preprocessing.setdefault("scaler", "minmax")
    preprocessing.setdefault("encode_categorical", False)

    return cfg


def _matches_prefix(col: str, prefixes: list[str]) -> bool:
    return any(col.startswith(prefix) for prefix in prefixes)


def infer_column_types(df: pd.DataFrame, dataset_cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    all_columns = list(df.columns)

    meta_columns = set(dataset_cfg.get("meta_columns", []))
    exclude_columns = set(dataset_cfg.get("exclude_columns", []))
    exclude_prefixes = dataset_cfg.get("exclude_prefixes", [])
    target_column = dataset_cfg.get("target_column")

    valid_columns = []
    for col in all_columns:
        if col == target_column:
            continue
        if col in meta_columns:
            continue
        if col in exclude_columns:
            continue
        if _matches_prefix(col, exclude_prefixes):
            continue
        valid_columns.append(col)

    col_cfg = dataset_cfg.get("column_types", {})

    num_cfg = col_cfg.get("numerical", {})
    bin_cfg = col_cfg.get("binary", {})
    cat_cfg = col_cfg.get("categorical", {})

    numerical_columns = []
    binary_columns = []
    categorical_columns = []

    for col in valid_columns:
        if col in num_cfg.get("exclude_columns", []):
            continue
        if col in bin_cfg.get("exclude_columns", []):
            continue
        if col in cat_cfg.get("exclude_columns", []):
            continue

        if col in num_cfg.get("include_columns", []) or _matches_prefix(col, num_cfg.get("include_prefixes", [])):
            numerical_columns.append(col)
            continue

        if col in bin_cfg.get("include_columns", []):
            binary_columns.append(col)
            continue

        if col in cat_cfg.get("include_columns", []):
            categorical_columns.append(col)
            continue

        categorical_columns.append(col)

    return sorted(categorical_columns), sorted(numerical_columns), sorted(binary_columns)


def _fill_mode(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return series.fillna(0)
    return series.fillna(series.mode(dropna=True).iloc[0])


def _convert_binary_value(value: Any, pos_values: set[str], neg_values: set[str]) -> int:
    if pd.isna(value):
        return value

    v = str(value).strip().lower()
    if v in pos_values:
        return 1
    if v in neg_values:
        return 0
    return value


def preprocess_dataframe(
    df: pd.DataFrame,
    dataset_cfg: dict[str, Any],
    categorical_columns: list[str],
    numerical_columns: list[str],
    binary_columns: list[str],
    fit: bool = True,
    scaler: MinMaxScaler | None = None,
    cat_encoder: OrdinalEncoder | None = None,
) -> tuple[pd.DataFrame, MinMaxScaler | None, OrdinalEncoder | None]:
    df = df.copy()

    prep_cfg = dataset_cfg["preprocessing"]
    binary_indicator = dataset_cfg.get("binary_indicator", {})
    pos_values = set(str(v).lower() for v in binary_indicator.get("positive", ["1", "true", "yes", "y", "t"]))
    neg_values = set(str(v).lower() for v in binary_indicator.get("negative", ["0", "false", "no", "n", "f"]))

    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = _fill_mode(df[col])

    for col in categorical_columns:
        df[col] = df[col].astype("object")
        df[col] = _fill_mode(df[col])

    for col in binary_columns:
        df[col] = df[col].apply(lambda x: _convert_binary_value(x, pos_values, neg_values))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = _fill_mode(df[col]).astype(int)

    if prep_cfg.get("scale_numerical", True) and numerical_columns:
        if prep_cfg.get("scaler", "minmax") != "minmax":
            raise ValueError("Currently only MinMaxScaler is supported.")

        if fit:
            scaler = MinMaxScaler()
            df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        else:
            if scaler is None:
                raise ValueError("Scaler must be provided when fit=False.")
            df[numerical_columns] = scaler.transform(df[numerical_columns])

    encode_cat = prep_cfg.get("encode_categorical", False)
    if encode_cat and categorical_columns:
        if fit:
            cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            df[categorical_columns] = cat_encoder.fit_transform(df[categorical_columns])
        else:
            if cat_encoder is None:
                raise ValueError("cat_encoder must be provided when fit=False.")
            df[categorical_columns] = cat_encoder.transform(df[categorical_columns])

    return df, scaler, cat_encoder


def split_xy(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")

    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def prepare_tabular_dataset(
    csv_path: str | Path,
    dataset_cfg: dict[str, Any],
    categorical_columns: list[str] | None = None,
    numerical_columns: list[str] | None = None,
    binary_columns: list[str] | None = None,
    fit: bool = True,
    scaler: MinMaxScaler | None = None,
    cat_encoder: OrdinalEncoder | None = None,
) -> tuple[
    tuple[pd.DataFrame, pd.Series],
    list[str],
    list[str],
    list[str],
    MinMaxScaler | None,
    OrdinalEncoder | None,
]:
    df = load_csv(csv_path)
    df = normalize_column_names(df, lowercase=dataset_cfg["preprocessing"].get("lowercase_columns", True))

    if categorical_columns is None or numerical_columns is None or binary_columns is None:
        categorical_columns, numerical_columns, binary_columns = infer_column_types(df, dataset_cfg)

    processed_df, scaler, cat_encoder = preprocess_dataframe(
        df=df,
        dataset_cfg=dataset_cfg,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        fit=fit,
        scaler=scaler,
        cat_encoder=cat_encoder,
    )

    x, y = split_xy(processed_df, dataset_cfg["target_column"])

    return (x, y), categorical_columns, numerical_columns, binary_columns, scaler, cat_encoder