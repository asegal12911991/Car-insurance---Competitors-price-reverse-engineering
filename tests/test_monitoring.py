from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from competitor_pricing_ai.config import (
    DataConfig,
    MonitoringConfig,
    PipelineConfig,
    ProjectConfig,
)
from competitor_pricing_ai.monitoring import (
    build_refresh_recommendation,
    calculate_drift,
    categorical_psi,
    distribution_psi,
    numeric_psi,
)


def _config(**monitoring_overrides) -> PipelineConfig:
    return PipelineConfig(
        project=ProjectConfig(),
        data=DataConfig(
            input_path="unused.csv",
            date_column="quote_date",
            competitor_columns=["comp_a", "comp_b", "comp_c"],
        ),
        monitoring=MonitoringConfig(**monitoring_overrides),
    )


def test_distribution_psi_is_zero_for_identical_distributions() -> None:
    dist = pd.Series([0.5, 0.5], index=["a", "b"])
    assert distribution_psi(dist, dist) == pytest.approx(0.0)


def test_numeric_psi_detects_a_shifted_distribution() -> None:
    expected = pd.Series(range(1, 101), dtype=float)
    actual = expected + 50  # whole distribution shifted up
    assert numeric_psi(expected, actual) > 0.5


def test_numeric_psi_is_near_zero_for_resampled_same_distribution() -> None:
    expected = pd.Series(range(1, 101), dtype=float)
    assert numeric_psi(expected, expected) == pytest.approx(0.0, abs=1e-6)


def test_categorical_psi_treats_missing_as_its_own_bucket() -> None:
    expected = pd.Series(["a", "a", "b", "b"])
    actual = pd.Series(["a", None, None, "b"])
    psi = categorical_psi(expected, actual)
    assert psi > 0.0


def test_calculate_drift_flags_a_feature_missing_from_current_data() -> None:
    reference = pd.DataFrame({"premium_zone": ["north", "south"]})
    current = pd.DataFrame({"other_column": ["north"]})
    drift = calculate_drift(
        reference, current, feature_columns=["premium_zone"], categorical_columns={"premium_zone"},
        psi_threshold=0.2,
    )
    assert drift[0]["status"] == "missing"
    assert drift[0]["psi"] is None


def test_refresh_recommendation_flags_unknown_d2_instead_of_masking_it_as_no_drop(
    tmp_path: Path,
) -> None:
    # Baseline metrics.json predates the d2 metric (only has r2). Falling back to
    # comparing current d2 against itself would silently report a zero drop.
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"test": {"r2": 0.8, "rmse": 50.0}}', encoding="utf-8")
    current_performance = {"d2": 0.7, "rmse": 50.0, "gini": 0.4, "mape": 5.0}

    recommendation = build_refresh_recommendation(
        drift=[],
        current_performance=current_performance,
        training_metrics_path=metrics_path,
        config=_config(),
    )

    assert recommendation["retrain_recommended"] is True
    assert any("D² unavailable" in reason for reason in recommendation["reasons"])


def test_refresh_recommendation_triggers_on_real_d2_drop(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"test": {"d2": 0.80, "rmse": 50.0}}', encoding="utf-8")
    current_performance = {"d2": 0.60, "rmse": 50.0}

    recommendation = build_refresh_recommendation(
        drift=[],
        current_performance=current_performance,
        training_metrics_path=metrics_path,
        config=_config(performance_d2_drop_threshold=0.05),
    )

    assert recommendation["retrain_recommended"] is True
    assert any("D² dropped" in reason for reason in recommendation["reasons"])


def test_refresh_recommendation_is_quiet_when_metrics_are_stable(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text('{"test": {"d2": 0.80, "rmse": 50.0}}', encoding="utf-8")
    current_performance = {"d2": 0.79, "rmse": 51.0}

    recommendation = build_refresh_recommendation(
        drift=[],
        current_performance=current_performance,
        training_metrics_path=metrics_path,
        config=_config(),
    )

    assert recommendation["retrain_recommended"] is False
