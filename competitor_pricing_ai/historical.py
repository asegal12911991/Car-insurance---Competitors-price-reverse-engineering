"""Leakage-safe historical market anchors for downstream demand modelling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.models import build_sklearn_pipeline


def build_historical_market_features(
    data: pd.DataFrame,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create expanding-window predictions using only observations from earlier months.

    The warm-up period is intentionally left without predictions. Backfilling it with a model
    trained on future observations would make the downstream demand model look better than it
    can perform in production.
    """
    date_column = config.data.date_column
    target_column = config.data.target.name
    working = data.dropna(subset=[date_column]).copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column]).sort_values(date_column)
    working["__score_period"] = working[date_column].dt.to_period("M")
    working["market_anchor"] = np.nan
    working["anchor_training_cutoff"] = pd.NaT

    scored_periods = 0
    for period in sorted(working["__score_period"].unique()):
        score_mask = working["__score_period"].eq(period)
        period_start = period.to_timestamp()
        train_mask = working[date_column].lt(period_start)
        train = working.loc[train_mask & working[target_column].notna()]
        if len(train) < config.historical_predictions.min_train_rows:
            continue

        model = build_sklearn_pipeline(config, categorical_columns, numeric_columns)
        fit_kwargs: dict[str, Any] = {}
        if config.data.weight_column and config.data.weight_column in train:
            fit_kwargs["model__sample_weight"] = train[config.data.weight_column].to_numpy()
        model.fit(train[feature_columns], train[target_column], **fit_kwargs)
        working.loc[score_mask, "market_anchor"] = model.predict(
            working.loc[score_mask, feature_columns]
        )
        working.loc[score_mask, "anchor_training_cutoff"] = train[date_column].max()
        scored_periods += 1

    own_column = config.data.own_premium_column
    if own_column and own_column in working:
        anchor = working["market_anchor"].clip(lower=1e-6)
        own = pd.to_numeric(working[own_column], errors="coerce").clip(lower=1e-6)
        working["relative_price_ratio"] = own / anchor
        working["log_relative_price"] = np.log(own / anchor)
        working["price_gap_to_market_anchor"] = own - anchor

    working["anchor_method"] = "rolling_origin_expanding_window"
    working["anchor_is_frozen_for_optimization"] = True
    working["anchor_model_backend"] = "sklearn"

    keep = list(dict.fromkeys(
        config.data.id_columns
        + [date_column]
        + config.data.categorical_columns
        + config.data.numeric_columns
        + [column for column in [own_column, config.data.conversion_column,
                                  config.data.weight_column] if column]
        + [target_column, "market_anchor", "relative_price_ratio", "log_relative_price",
           "price_gap_to_market_anchor", "anchor_training_cutoff", "anchor_method",
           "anchor_is_frozen_for_optimization", "anchor_model_backend"]
    ))
    output = working[[column for column in keep if column in working]].copy()
    metadata = {
        "method": "rolling_origin_expanding_window",
        "backend": "sklearn",
        "minimum_training_rows": config.historical_predictions.min_train_rows,
        "rows_total": int(len(output)),
        "rows_scored": int(output["market_anchor"].notna().sum()),
        "rows_warmup_unscored": int(output["market_anchor"].isna().sum()),
        "periods_scored": scored_periods,
        "leakage_rule": "Each anchor uses competitor observations strictly before its month.",
    }
    return output, metadata


def save_historical_market_features(
    data: pd.DataFrame,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> tuple[str, dict[str, Any]]:
    frame, metadata = build_historical_market_features(
        data, config, feature_columns, categorical_columns, numeric_columns
    )
    path = output_dir / "historical_market_features.csv"
    frame.to_csv(path, index=False)
    return str(path), metadata
