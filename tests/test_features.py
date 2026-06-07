from __future__ import annotations

import pandas as pd

from competitor_pricing_ai.config import (
    DataConfig,
    FeaturesConfig,
    PipelineConfig,
    ProjectConfig,
    SplitConfig,
    TargetConfig,
)
from competitor_pricing_ai.features import engineer_market_features, select_model_features


def make_config() -> PipelineConfig:
    return PipelineConfig(
        project=ProjectConfig(),
        data=DataConfig(
            input_path="unused.csv",
            date_column="quote_date",
            own_premium_column="own_premium",
            competitor_columns=["comp_a", "comp_b", "comp_c"],
            target=TargetConfig(name="avg_top_2_competitor_premium", top_n=2),
            categorical_columns=["region"],
            numeric_columns=["driver_age", "own_premium", "comp_a"],
            id_columns=["quote_id"],
        ),
        features=FeaturesConfig(top_ns=[2]),
        split=SplitConfig(),
    )


def test_engineer_market_features_average_top_n_and_rank() -> None:
    df = pd.DataFrame(
        {
            "quote_id": [1, 2],
            "quote_date": ["2025-01-01", "2025-01-02"],
            "region": ["center", "north"],
            "driver_age": [40, 21],
            "own_premium": [105, 150],
            "comp_a": [100, 200],
            "comp_b": [120, 130],
            "comp_c": [90, 140],
        }
    )
    engineered, metadata = engineer_market_features(df, make_config())

    assert engineered["avg_top_2_competitor_premium"].tolist() == [95.0, 135.0]
    assert engineered["rank_own_premium"].tolist() == [3.0, 3.0]
    assert metadata.target_column == "avg_top_2_competitor_premium"


def test_select_model_features_excludes_competitor_and_target_derived_columns() -> None:
    config = make_config()
    df = pd.DataFrame(
        {
            "quote_id": [1],
            "quote_date": ["2025-01-01"],
            "region": ["center"],
            "driver_age": [40],
            "own_premium": [105],
            "comp_a": [100],
            "comp_b": [120],
            "comp_c": [90],
        }
    )
    engineered, metadata = engineer_market_features(df, config)
    feature_columns, categorical, numeric = select_model_features(engineered, config, metadata)

    assert "comp_a" not in feature_columns
    assert "avg_top_2_competitor_premium" not in feature_columns
    assert "rank_own_premium" not in feature_columns
    assert categorical == ["region"]
    assert "driver_age" in numeric
    assert "own_premium" in numeric
