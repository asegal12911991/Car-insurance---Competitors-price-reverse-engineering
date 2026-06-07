"""Model and market drift monitoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from competitor_pricing_ai.config import PipelineConfig, load_config, resolve_project_path
from competitor_pricing_ai.data import coerce_basic_types, load_dataset, validate_input_data
from competitor_pricing_ai.features import engineer_market_features
from competitor_pricing_ai.metrics import regression_metrics
from competitor_pricing_ai.models import load_sklearn_bundle, predict_with_sklearn_bundle
from competitor_pricing_ai.reporting import write_json


def run_monitoring(config_or_path: PipelineConfig | str | Path) -> dict[str, Any]:
    config = load_config(config_or_path) if not isinstance(config_or_path, PipelineConfig) else config_or_path
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    current_path = config.monitoring.current_data_path
    if not current_path:
        raise ValueError("monitoring.current_data_path is required for monitoring")

    model_path = output_dir / "model.joblib"
    if not model_path.exists():
        raise ValueError(f"Monitoring currently expects a sklearn model bundle at {model_path}")

    reference_path = (
        resolve_project_path(config.monitoring.drift_reference_path, config.root_dir)
        if config.monitoring.drift_reference_path
        else output_dir / "reference_features.csv"
    )
    if not reference_path.exists():
        raise ValueError(f"Reference feature file does not exist: {reference_path}")

    bundle = load_sklearn_bundle(model_path)
    reference = pd.read_csv(reference_path)
    current_raw = load_dataset(resolve_project_path(current_path, config.root_dir))
    validate_input_data(current_raw, config)
    current_typed = coerce_basic_types(current_raw, config)
    current_engineered, _ = engineer_market_features(current_typed, config)
    current_engineered["prediction"] = predict_with_sklearn_bundle(bundle, current_engineered)

    feature_columns = bundle["feature_columns"]
    categorical_columns = set(bundle["categorical_columns"])
    drift = calculate_drift(
        reference,
        current_engineered,
        feature_columns,
        categorical_columns,
        config.monitoring.psi_threshold,
    )

    target_column = bundle["target_column"]
    performance = None
    if target_column in current_engineered.columns and current_engineered[target_column].notna().any():
        performance = regression_metrics(
            current_engineered[target_column], current_engineered["prediction"]
        )

    refresh_recommendation = build_refresh_recommendation(
        drift,
        performance,
        output_dir / "metrics.json",
        config,
    )
    metrics = {
        "drift": drift,
        "current_performance": performance,
        "refresh_recommendation": refresh_recommendation,
    }
    write_json(metrics, output_dir / "monitoring_metrics.json")
    write_monitoring_report(metrics, output_dir / "monitoring_report.md", config)
    return metrics


def calculate_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: set[str],
    psi_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in feature_columns:
        if column not in reference.columns or column not in current.columns:
            rows.append(
                {
                    "feature": column,
                    "psi": None,
                    "status": "missing",
                    "detail": "Feature missing from reference or current data",
                }
            )
            continue
        if column in categorical_columns:
            psi = categorical_psi(reference[column], current[column])
            kind = "categorical"
        else:
            psi = numeric_psi(reference[column], current[column])
            kind = "numeric"
        rows.append(
            {
                "feature": column,
                "type": kind,
                "psi": psi,
                "status": "review" if psi >= psi_threshold else "ok",
                "detail": f"PSI {psi:.4f}",
            }
        )
    return sorted(rows, key=lambda item: -1 if item["psi"] is None else item["psi"], reverse=True)


def numeric_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_values = pd.to_numeric(expected, errors="coerce").dropna()
    actual_values = pd.to_numeric(actual, errors="coerce").dropna()
    if expected_values.empty or actual_values.empty:
        return 0.0
    if expected_values.nunique() <= 2:
        return categorical_psi(expected_values.astype(str), actual_values.astype(str))

    quantiles = np.linspace(0, 1, bins + 1)
    cut_points = np.unique(np.quantile(expected_values, quantiles))
    if len(cut_points) < 3:
        return categorical_psi(expected_values.astype(str), actual_values.astype(str))

    expected_bins = pd.cut(expected_values, bins=cut_points, include_lowest=True, duplicates="drop")
    actual_bins = pd.cut(actual_values, bins=cut_points, include_lowest=True, duplicates="drop")
    return distribution_psi(
        expected_bins.value_counts(normalize=True, sort=False),
        actual_bins.value_counts(normalize=True, sort=False),
    )


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected_values = expected.astype("string").fillna("__missing__")
    actual_values = actual.astype("string").fillna("__missing__")
    expected_dist = expected_values.value_counts(normalize=True)
    actual_dist = actual_values.value_counts(normalize=True)
    return distribution_psi(expected_dist, actual_dist)


def distribution_psi(expected_dist: pd.Series, actual_dist: pd.Series) -> float:
    eps = 1e-6
    labels = expected_dist.index.union(actual_dist.index)
    expected = expected_dist.reindex(labels, fill_value=0).astype(float).clip(lower=eps)
    actual = actual_dist.reindex(labels, fill_value=0).astype(float).clip(lower=eps)
    return float(((actual - expected) * np.log(actual / expected)).sum())


def build_refresh_recommendation(
    drift: list[dict[str, Any]],
    current_performance: dict[str, float] | None,
    training_metrics_path: Path,
    config: PipelineConfig,
) -> dict[str, Any]:
    drift_reviews = [
        item for item in drift if item.get("status") in {"review", "missing"}
    ]
    reasons = []
    if drift_reviews:
        reasons.append(f"{len(drift_reviews)} features exceed drift thresholds or are missing")

    if current_performance and training_metrics_path.exists():
        import json

        training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
        test_metrics = training_metrics.get("test", {})
        # D² drop (falls back to R² for old metrics.json without d2 key)
        baseline_d2 = test_metrics.get("d2", test_metrics.get("r2", current_performance.get("d2", current_performance.get("r2", 0.0))))
        current_d2 = current_performance.get("d2", current_performance.get("r2", 0.0))
        d2_drop = baseline_d2 - current_d2
        rmse_increase = current_performance["rmse"] - test_metrics.get(
            "rmse", current_performance["rmse"]
        )
        gini_drop = test_metrics.get("gini", current_performance.get("gini", 0.0)) - current_performance.get("gini", 0.0)
        mape_increase = current_performance.get("mape", 0.0) - test_metrics.get("mape", current_performance.get("mape", 0.0))
        if d2_drop >= config.monitoring.performance_d2_drop_threshold:
            reasons.append(f"D² dropped by {d2_drop:.4f}")
        if rmse_increase >= config.monitoring.performance_rmse_increase_threshold:
            reasons.append(f"RMSE increased by {rmse_increase:.2f}")
        if gini_drop >= config.monitoring.performance_gini_drop_threshold:
            reasons.append(f"Gini dropped by {gini_drop:.4f}")
        if mape_increase >= config.monitoring.performance_mape_increase_threshold:
            reasons.append(f"MAPE increased by {mape_increase:.2f}%")

    return {
        "retrain_recommended": bool(reasons),
        "reasons": reasons or ["No retraining trigger exceeded configured thresholds"],
    }


def write_monitoring_report(
    metrics: dict[str, Any], output_path: str | Path, config: PipelineConfig
) -> None:
    lines = [
        f"# {config.project.name} Monitoring Report",
        "",
        "## Refresh Recommendation",
        "",
        f"- Retrain recommended: `{metrics['refresh_recommendation']['retrain_recommended']}`",
    ]
    for reason in metrics["refresh_recommendation"]["reasons"]:
        lines.append(f"- {reason}")

    lines.extend(["", "## Current Performance", ""])
    if metrics["current_performance"]:
        row = metrics["current_performance"]
        d2      = row.get("d2", float("nan"))
        gini    = row.get("gini", float("nan"))
        bias    = row.get("mean_bias_pct", float("nan"))
        d2_str   = f", D²={d2:.4f}"       if not np.isnan(d2)   else ""
        gini_str = f", Gini={gini:.4f}"   if not np.isnan(gini) else ""
        bias_str = f", Bias%={bias:+.2f}%" if not np.isnan(bias) else ""
        lines.append(
            f"Gini={gini:.4f}, MAPE={row['mape']:.2f}%{bias_str}"
            f"{d2_str}, RMSE={row['rmse']:.2f}, N={row['n']:,}"
        )
    else:
        lines.append("No actual target values were available for current performance monitoring.")

    lines.extend(["", "## Feature Drift", "", "| Feature | Type | PSI | Status |", "|---|---|---:|---|"])
    for item in metrics["drift"]:
        psi = "" if item.get("psi") is None else f"{item['psi']:.4f}"
        lines.append(f"| `{item['feature']}` | {item.get('type', '')} | {psi} | {item['status']} |")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
