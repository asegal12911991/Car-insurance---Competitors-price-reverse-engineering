"""Reporting and QA artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.features import FeatureMetadata


def write_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def build_qa_checklist(
    config: PipelineConfig,
    data_quality: dict[str, Any],
    split_metadata: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    feature_metadata: FeatureMetadata,
    model_path: str | None,
    mojo_path: str | None,
    runtime_seconds: float,
    onnx_path: str | None = None,
) -> dict[str, Any]:
    validation_metrics = metrics["validation"]
    test_metrics = metrics["test"]
    checks = [
        {
            "name": "data_contract_valid",
            "passed": data_quality["rows_eligible_for_target"] > 0,
            "blocking": True,
            "detail": f"{data_quality['rows_eligible_for_target']} rows can calculate target",
        },
        {
            "name": "time_based_split",
            "passed": split_metadata["strategy"] == "time",
            "blocking": True,
            "detail": split_metadata["date_ranges"],
        },
        {
            "name": "validation_d2_threshold",
            "passed": validation_metrics.get("d2", validation_metrics.get("r2", 0)) >= config.evaluation.d2_min,
            "blocking": True,
            "detail": f"validation D²={validation_metrics.get('d2', float('nan')):.4f}; target >= {config.evaluation.d2_min}",
        },
        {
            "name": "validation_rmse_threshold",
            "passed": validation_metrics["rmse"] <= config.evaluation.rmse_max,
            "blocking": True,
            "detail": f"validation RMSE={validation_metrics['rmse']:.4f}; target <= {config.evaluation.rmse_max}",
        },
        {
            "name": "test_d2_threshold",
            "passed": test_metrics.get("d2", test_metrics.get("r2", 0)) >= config.evaluation.d2_min,
            "blocking": True,
            "detail": f"test D²={test_metrics.get('d2', float('nan')):.4f}; target >= {config.evaluation.d2_min}",
        },
        {
            "name": "test_rmse_threshold",
            "passed": test_metrics["rmse"] <= config.evaluation.rmse_max,
            "blocking": True,
            "detail": f"test RMSE={test_metrics['rmse']:.4f}; target <= {config.evaluation.rmse_max}",
        },
        {
            "name": "validation_mape_threshold",
            "passed": validation_metrics.get("mape", float("inf")) <= config.evaluation.mape_max,
            "blocking": False,
            "detail": f"validation MAPE={validation_metrics.get('mape', float('nan')):.2f}%; target <= {config.evaluation.mape_max}%",
        },
        {
            "name": "validation_gini_threshold",
            "passed": validation_metrics.get("gini", 0.0) >= config.evaluation.gini_min,
            "blocking": False,
            "detail": f"validation Gini={validation_metrics.get('gini', float('nan')):.4f}; target >= {config.evaluation.gini_min}",
        },
        {
            "name": "validation_bias_threshold",
            "passed": abs(validation_metrics.get("mean_bias_pct", float("inf"))) <= config.evaluation.mean_bias_pct_max,
            "blocking": False,
            "detail": f"validation Bias%={validation_metrics.get('mean_bias_pct', float('nan')):+.2f}%; threshold ±{config.evaluation.mean_bias_pct_max}%",
        },
        {
            "name": "model_export_available",
            "passed": bool(model_path),
            "blocking": True,
            "detail": model_path or "No model path returned",
        },
        {
            "name": "mojo_export_available_when_h2o",
            "passed": config.model.backend != "h2o" or bool(mojo_path),
            "blocking": True,
            "detail": mojo_path or "Not applicable for sklearn backend",
        },
        {
            "name": "onnx_export_available_when_requested",
            "passed": not config.model.export_onnx or bool(onnx_path),
            "blocking": True,
            "detail": onnx_path if onnx_path else ("Not requested" if not config.model.export_onnx else "ONNX export was not produced"),
        },
        {
            "name": "onnx_prediction_parity_when_requested",
            "passed": (
                not config.model.export_onnx
                or bool(onnx_path and (Path(onnx_path).parent / "onnx_parity.json").exists())
            ),
            "blocking": True,
            "detail": "Python/ONNX parity artifact required for ONNX handoff",
        },
        {
            "name": "runtime_budget",
            "passed": runtime_seconds <= config.model.max_runtime_seconds,
            "blocking": True,
            "detail": f"runtime={runtime_seconds:.1f}s; budget={config.model.max_runtime_seconds}s",
        },
        {
            "name": "leakage_review",
            "passed": len(feature_metadata.leakage_warnings) == 0,
            "blocking": False,
            "detail": feature_metadata.leakage_warnings or "No leakage warnings detected",
        },
    ]
    return {
        "overall_passed": all(check["passed"] for check in checks if check.get("blocking", True)),
        "human_review_required": any(
            not check["passed"] for check in checks if not check.get("blocking", True)
        ),
        "checks": checks,
    }


def individual_competitors_table(results: dict[str, Any]) -> str:
    lines = [
        "| Competitor | R2 | RMSE | MAPE | Gini | Bias% | Coverage | N (test) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comp_col, result in sorted(results.items()):
        test = result.metrics["test"]
        gini = test.get("gini", float("nan"))
        bias_pct = test.get("mean_bias_pct", float("nan"))
        coverage = 1.0 - result.missing_rate_train
        bias_str = f"{bias_pct:+.2f}%" if not np.isnan(bias_pct) else "—"
        gini_str = f"{gini:.4f}" if not np.isnan(gini) else "—"
        lines.append(
            f"| `{comp_col}` | {test['r2']:.4f} | {test['rmse']:.2f} | "
            f"{test['mape']:.2f}% | {gini_str} | {bias_str} | "
            f"{coverage:.0%} | {result.n_test:,} |"
        )
    return "\n".join(lines)


def write_business_report(
    output_path: str | Path,
    config: PipelineConfig,
    data_quality: dict[str, Any],
    split_metadata: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    feature_importance: pd.DataFrame,
    feature_metadata: FeatureMetadata,
    qa_checklist: dict[str, Any],
    artifacts: dict[str, str | None],
    runtime_seconds: float,
    individual_results: dict[str, Any] | None = None,
    historical_metadata: dict[str, Any] | None = None,
    demand_readiness: dict[str, Any] | None = None,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    top_features = feature_importance.head(15)
    lines = [
        f"# {config.project.name} Business Report",
        "",
        "## Objective",
        "",
        "Reverse engineer competitor pricing strategy for car insurance pricing optimisation "
        f"using `{config.data.target.name}` as the primary market component target.",
        "",
        "## Run Summary",
        "",
        f"- Backend: `{config.model.backend}`",
        f"- Target: `{config.data.target.name}`",
        f"- Runtime: {runtime_seconds:.1f} seconds",
        f"- Input rows: {data_quality['rows']:,}",
        f"- Competitor columns: {', '.join(data_quality['competitor_columns'])}",
        f"- Date range: {data_quality['date_min']} to {data_quality['date_max']}",
        "",
        "## Time Split",
        "",
        metrics_table(split_metadata["row_counts"], split_metadata["date_ranges"]),
        "",
        "## Model Performance",
        "",
        performance_table(metrics),
        "",
        "## QA Checklist",
        "",
        qa_table(qa_checklist),
        "",
        "## Strongest Pricing Drivers",
        "",
        feature_table(top_features),
        "",
    ]
    if individual_results:
        lines += [
            "## Individual Competitor Models",
            "",
            individual_competitors_table(individual_results),
            "",
        ]
    if historical_metadata:
        lines += [
            "## Demand-Model Handoff",
            "",
            f"- Historical rows scored: {historical_metadata['rows_scored']:,}",
            f"- Warm-up rows intentionally unscored: "
            f"{historical_metadata['rows_warmup_unscored']:,}",
            "- Anchor rule: prior-month observations only; anchor stays frozen during optimisation.",
            f"- Standalone demand diagnostic: `{(demand_readiness or {}).get('status', 'not run')}`",
            "- Final elasticity and optimisation acceptance remains downstream.",
            "",
        ]
    lines += [
        "## Pricing Interpretation",
        "",
        interpretation_text(metrics, feature_importance, config),
        "",
        "## Leakage And Governance Notes",
        "",
        leakage_text(feature_metadata),
        "",
        "## Artifacts",
        "",
    ]
    for name, path in artifacts.items():
        lines.append(f"- `{name}`: `{path}`")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def metrics_table(row_counts: dict[str, int], date_ranges: dict[str, dict[str, str]]) -> str:
    lines = ["| Sample | Rows | Date Min | Date Max |", "|---|---:|---|---|"]
    for sample in ["train", "validation", "test"]:
        date_range = date_ranges[sample]
        lines.append(
            f"| {sample} | {row_counts[sample]:,} | {date_range['min']} | {date_range['max']} |"
        )
    return "\n".join(lines)


def performance_table(metrics: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Sample | Gini | MAPE | Bias% | RMSLE | D² | R² | RMSE | N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sample in ["train", "validation", "test"]:
        row = metrics[sample]
        d2      = row.get("d2", float("nan"))
        r2      = row.get("r2", float("nan"))
        bias    = row.get("mean_bias_pct", float("nan"))
        gini    = row.get("gini", float("nan"))
        rmsle   = row.get("rmsle", float("nan"))
        d2_str    = f"{d2:.4f}"    if not np.isnan(d2)   else "—"
        r2_str    = f"{r2:.4f}"    if not np.isnan(r2)   else "—"
        bias_str  = f"{bias:+.2f}%" if not np.isnan(bias) else "—"
        gini_str  = f"{gini:.4f}"  if not np.isnan(gini) else "—"
        rmsle_str = f"{rmsle:.4f}" if not np.isnan(rmsle) else "—"
        lines.append(
            f"| {sample} | {gini_str} | {row['mape']:.2f}% | {bias_str} | "
            f"{rmsle_str} | {d2_str} | {r2_str} | {row['rmse']:.2f} | {row['n']:,} |"
        )
    return "\n".join(lines)


def qa_table(qa_checklist: dict[str, Any]) -> str:
    lines = ["| Check | Status | Detail |", "|---|---|---|"]
    for check in qa_checklist["checks"]:
        status = "PASS" if check["passed"] else ("REVIEW" if not check.get("blocking", True) else "FAIL")
        detail = str(check["detail"]).replace("\n", " ")
        lines.append(f"| {check['name']} | {status} | {detail} |")
    return "\n".join(lines)


def feature_table(features: pd.DataFrame) -> str:
    if features.empty:
        return "No feature-importance values were produced."
    lines = ["| Feature | Importance | Std |", "|---|---:|---:|"]
    for _, row in features.iterrows():
        importance = row.get("importance_mean", np.nan)
        std = row.get("importance_std", np.nan)
        std_text = "" if pd.isna(std) else f"{std:.5f}"
        lines.append(f"| `{row['feature']}` | {importance:.5f} | {std_text} |")
    return "\n".join(lines)


def interpretation_text(
    metrics: dict[str, dict[str, float]],
    feature_importance: pd.DataFrame,
    config: PipelineConfig,
) -> str:
    test = metrics["test"]
    d2 = test.get("d2", test.get("r2", 0.0))
    threshold_text = (
        "meets"
        if d2 >= config.evaluation.d2_min and test["rmse"] <= config.evaluation.rmse_max
        else "does not yet meet"
    )
    top_feature = (
        f"`{feature_importance.iloc[0]['feature']}`"
        if not feature_importance.empty
        else "the available feature set"
    )
    gini = test.get("gini", float("nan"))
    bias_pct = test.get("mean_bias_pct", float("nan"))
    gini_text = f" Gini={gini:.3f} (discrimination)." if not np.isnan(gini) else ""
    if not np.isnan(bias_pct):
        bias_direction = "within" if abs(bias_pct) <= config.evaluation.mean_bias_pct_max else "outside"
        bias_text = f" Systematic bias: {bias_pct:+.2f}% ({bias_direction} ±{config.evaluation.mean_bias_pct_max}% threshold)."
    else:
        bias_text = ""
    return (
        f"The out-of-time test model {threshold_text} the configured deployment thresholds "
        f"(D² >= {config.evaluation.d2_min}, RMSE <= {config.evaluation.rmse_max}).{gini_text}{bias_text} "
        f"The strongest observed pricing driver is {top_feature}. Use the predictions as a "
        "frozen market-price anchor for downstream relative-price construction and scenario "
        "testing. Human review is still required before production pricing action."
    )


def leakage_text(feature_metadata: FeatureMetadata) -> str:
    if not feature_metadata.leakage_warnings:
        return (
            "No automatic leakage warnings were detected. Own premium and post-offer fields "
            "are hard-excluded; continue to review competitor-panel construction."
        )
    return "\n".join(f"- {warning}" for warning in feature_metadata.leakage_warnings)
