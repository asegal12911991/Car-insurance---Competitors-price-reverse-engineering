from __future__ import annotations

import pandas as pd

from competitor_pricing_ai.config import (
    DataConfig,
    HistoricalPredictionsConfig,
    PipelineConfig,
    ProjectConfig,
)
from competitor_pricing_ai.features import add_temporal_features
from competitor_pricing_ai.historical import build_historical_market_features


def test_historical_anchors_only_use_prior_months() -> None:
    dates = pd.date_range("2025-01-01", periods=90, freq="D")
    frame = pd.DataFrame({
        "quote_id": range(90),
        "quote_date": dates,
        "region": ["a", "b"] * 45,
        "driver_age": [30 + i % 20 for i in range(90)],
        "own_premium": [600 + i for i in range(90)],
        "converted": [i % 2 for i in range(90)],
        "avg_top_3_competitor_premium": [550 + i for i in range(90)],
    })
    temporal = add_temporal_features(frame, "quote_date")
    frame.loc[70, "avg_top_3_competitor_premium"] = None
    config = PipelineConfig(
        project=ProjectConfig(),
        data=DataConfig(
            input_path="unused.csv",
            date_column="quote_date",
            own_premium_column="own_premium",
            conversion_column="converted",
            competitor_columns=["a", "b", "c"],
            categorical_columns=["region"],
            numeric_columns=["driver_age"],
            id_columns=["quote_id"],
        ),
        historical_predictions=HistoricalPredictionsConfig(min_train_rows=20),
    )
    result, metadata = build_historical_market_features(
        frame,
        config,
        ["region", "driver_age", *temporal],
        ["region"],
        ["driver_age", *temporal],
    )
    scored = result.dropna(subset=["market_anchor"])
    assert metadata["rows_scored"] > 0
    assert metadata["lookback_months"] == 4
    assert (
        pd.to_datetime(scored["anchor_training_cutoff"])
        < pd.to_datetime(scored["quote_date"]).dt.to_period("M").dt.to_timestamp()
    ).all()
    assert scored["anchor_is_frozen_for_optimization"].all()
    assert pd.notna(result.loc[result["quote_id"].eq(70), "market_anchor"]).all()
