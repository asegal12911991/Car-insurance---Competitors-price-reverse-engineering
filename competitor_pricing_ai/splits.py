"""Time-aware train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from competitor_pricing_ai.config import PipelineConfig


@dataclass
class SplitResult:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    metadata: dict[str, Any]


def time_based_split(df: pd.DataFrame, config: PipelineConfig) -> SplitResult:
    date_column = config.data.date_column
    target_column = config.data.target.name
    working = df.dropna(subset=[target_column]).copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column]).sort_values(date_column).reset_index(drop=True)

    if len(working) < 10:
        raise ValueError("At least 10 rows with valid dates and target values are required")

    if config.split.train_end_date or config.split.validation_end_date:
        train_end = pd.to_datetime(config.split.train_end_date)
        validation_end = pd.to_datetime(config.split.validation_end_date)
        if pd.isna(train_end) or pd.isna(validation_end):
            raise ValueError("Both train_end_date and validation_end_date are required together")
        train = working[working[date_column] <= train_end]
        validation = working[(working[date_column] > train_end) & (working[date_column] <= validation_end)]
        test = working[working[date_column] > validation_end]
    else:
        n_rows = len(working)
        test_size = max(1, int(round(n_rows * config.split.test_fraction)))
        validation_size = max(1, int(round(n_rows * config.split.validation_fraction)))
        train_end_idx = n_rows - validation_size - test_size
        validation_end_idx = n_rows - test_size
        train_end_date = working.iloc[train_end_idx - 1][date_column]
        validation_end_date = working.iloc[validation_end_idx - 1][date_column]
        train = working[working[date_column] <= train_end_date]
        validation = working[
            (working[date_column] > train_end_date)
            & (working[date_column] <= validation_end_date)
        ]
        test = working[working[date_column] > validation_end_date]

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError(
            "Time split produced an empty train, validation, or test partition. "
            "Adjust split fractions or explicit cut-off dates."
        )

    metadata = {
        "strategy": "time",
        "date_column": date_column,
        "row_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "date_ranges": {
            "train": date_range(train, date_column),
            "validation": date_range(validation, date_column),
            "test": date_range(test, date_column),
        },
    }
    return SplitResult(train=train, validation=validation, test=test, metadata=metadata)


def date_range(df: pd.DataFrame, date_column: str) -> dict[str, str | None]:
    if df.empty:
        return {"min": None, "max": None}
    dates = pd.to_datetime(df[date_column], errors="coerce")
    return {"min": str(dates.min().date()), "max": str(dates.max().date())}
