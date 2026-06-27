from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from competitor_pricing_ai.config import DataConfig, ModelConfig, PipelineConfig, ProjectConfig
from competitor_pricing_ai.models import (
    load_sklearn_bundle,
    predict_with_bundle,
    train_sklearn_model,
)
from competitor_pricing_ai.splits import SplitResult


def _synthetic_split(n_per_split: int = 60) -> tuple[SplitResult, PipelineConfig]:
    rng = np.random.default_rng(0)

    def make_block(start_date: str, n: int) -> pd.DataFrame:
        driver_age = rng.uniform(20, 70, size=n)
        region = rng.choice(["north", "south"], size=n)
        noise = rng.normal(0, 5, size=n)
        target = 100 + driver_age * 2 + np.where(region == "north", 10, 0) + noise
        return pd.DataFrame(
            {
                "quote_date": pd.date_range(start_date, periods=n, freq="D"),
                "driver_age": driver_age,
                "region": region,
                "avg_top_3_competitor_premium": np.maximum(target, 1.0),
            }
        )

    train = make_block("2024-01-01", n_per_split)
    validation = make_block("2024-06-01", n_per_split)
    test = make_block("2024-09-01", n_per_split)

    config = PipelineConfig(
        project=ProjectConfig(),
        data=DataConfig(
            input_path="unused.csv",
            date_column="quote_date",
            competitor_columns=["comp_a", "comp_b", "comp_c"],
            categorical_columns=["region"],
            numeric_columns=["driver_age"],
        ),
        model=ModelConfig(backend="sklearn"),
    )
    split = SplitResult(train=train, validation=validation, test=test, metadata={})
    return split, config


def test_train_sklearn_model_produces_a_loadable_bundle_with_expected_metrics(
    tmp_path: Path,
) -> None:
    split, config = _synthetic_split()
    feature_columns = ["driver_age", "region"]

    result = train_sklearn_model(
        split, config, feature_columns, ["region"], ["driver_age"], tmp_path
    )

    assert result.backend == "sklearn"
    assert set(result.metrics) == {"train", "validation", "test"}
    # The synthetic target is a clean function of the features, so the model should fit well.
    assert result.metrics["test"]["d2"] > 0.5

    bundle = load_sklearn_bundle(result.model_path)
    assert bundle["feature_columns"] == feature_columns
    assert bundle["target_column"] == "avg_top_3_competitor_premium"

    preds = predict_with_bundle(bundle, split.test[feature_columns])
    assert len(preds) == len(split.test)
    assert np.all(np.isfinite(preds))


def test_predict_with_bundle_raises_a_clear_error_for_missing_feature_columns() -> None:
    bundle = {
        "model": object(),
        "backend": "sklearn",
        "feature_columns": ["driver_age", "region"],
    }
    frame = pd.DataFrame({"driver_age": [25, 30]})

    with pytest.raises(ValueError, match="missing model feature columns"):
        predict_with_bundle(bundle, frame)
