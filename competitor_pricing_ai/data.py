"""Data loading and contract validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from competitor_pricing_ai.config import PipelineConfig, resolve_project_path


class DataContractError(ValueError):
    """Raised when input data does not satisfy the configured data contract."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if not input_path.exists():
        raise DataContractError(f"Input data file does not exist: {input_path}")

    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(input_path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise DataContractError(f"Unsupported input format: {suffix}")


def load_configured_dataset(config: PipelineConfig) -> pd.DataFrame:
    return load_dataset(resolve_project_path(config.data.input_path, config.root_dir))


def detect_competitor_columns(df: pd.DataFrame, config: PipelineConfig) -> list[str]:
    configured = [column for column in config.data.competitor_columns if column in df.columns]
    if configured:
        return configured

    if config.data.competitor_column_regex:
        pattern = re.compile(config.data.competitor_column_regex)
        detected = [column for column in df.columns if pattern.search(column)]
        if detected:
            return detected

    raise DataContractError("No competitor premium columns were found in the input data")


def validate_input_data(df: pd.DataFrame, config: PipelineConfig) -> dict[str, Any]:
    required = [config.data.date_column]
    required.extend(config.data.id_columns)
    required.extend(config.data.categorical_columns)
    required.extend(config.data.numeric_columns)
    required.extend(config.data.comparability_columns)
    if config.data.own_premium_column:
        required.append(config.data.own_premium_column)
    if config.data.conversion_column:
        required.append(config.data.conversion_column)
    if config.data.weight_column:
        required.append(config.data.weight_column)
    required.extend(config.data.competitor_columns)

    missing_required = sorted(set(required) - set(df.columns))
    if missing_required:
        raise DataContractError(f"Missing required columns: {missing_required}")

    competitor_columns = detect_competitor_columns(df, config)
    if len(competitor_columns) < config.data.target.top_n:
        raise DataContractError(
            "The configured target needs at least "
            f"{config.data.target.top_n} competitor columns, found {len(competitor_columns)}"
        )

    date_series = pd.to_datetime(df[config.data.date_column], errors="coerce")
    invalid_dates = int(date_series.isna().sum())
    if invalid_dates:
        raise DataContractError(
            f"{invalid_dates} rows have invalid values in {config.data.date_column}"
        )

    competitor_numeric = df[competitor_columns].apply(pd.to_numeric, errors="coerce")
    non_positive = int((competitor_numeric <= 0).sum().sum())
    missing_comp_prices = int(competitor_numeric.isna().sum().sum())
    complete_rows = int(competitor_numeric.notna().sum(axis=1).ge(config.data.target.top_n).sum())
    fixed_panel_rows = int(competitor_numeric.notna().all(axis=1).sum())
    target_eligible_rows = (
        fixed_panel_rows
        if config.data.target.missing_panel_policy == "complete"
        else complete_rows
    )

    if target_eligible_rows == 0:
        raise DataContractError(
            "No rows have enough competitor prices to calculate the configured target"
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "date_min": str(date_series.min().date()),
        "date_max": str(date_series.max().date()),
        "competitor_columns": competitor_columns,
        "competitor_quote_rates": {
            column: float(competitor_numeric[column].gt(0).mean())
            for column in competitor_columns
        },
        "missing_competitor_price_cells": missing_comp_prices,
        "non_positive_competitor_price_cells": non_positive,
        "rows_with_enough_competitors": complete_rows,
        "rows_with_complete_competitor_panel": fixed_panel_rows,
        "rows_eligible_for_target": target_eligible_rows,
        "target_missing_panel_policy": config.data.target.missing_panel_policy,
        "comparability_missing": {
            column: int(df[column].isna().sum())
            for column in config.data.comparability_columns
            if column in df.columns
        },
        "premium_basis": config.data.premium_basis,
        "premium_currency": config.data.premium_currency,
    }


def coerce_basic_types(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    typed = df.copy()
    typed[config.data.date_column] = pd.to_datetime(typed[config.data.date_column], errors="coerce")

    numeric_candidates = set(config.data.numeric_columns)
    numeric_candidates.update(detect_competitor_columns(typed, config))
    if config.data.own_premium_column:
        numeric_candidates.add(config.data.own_premium_column)
    if config.data.conversion_column and config.data.conversion_column in typed.columns:
        numeric_candidates.add(config.data.conversion_column)
    if config.data.weight_column and config.data.weight_column in typed.columns:
        numeric_candidates.add(config.data.weight_column)

    for column in numeric_candidates:
        if column in typed.columns:
            typed[column] = pd.to_numeric(typed[column], errors="coerce")

    for column in config.data.categorical_columns:
        if column in typed.columns:
            typed[column] = typed[column].astype("string")

    return typed
