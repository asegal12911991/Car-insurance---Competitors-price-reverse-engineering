"""End-to-end training pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from competitor_pricing_ai.basket import generate_basket_artefacts
from competitor_pricing_ai.config import PipelineConfig, dump_resolved_config, load_config
from competitor_pricing_ai.data import coerce_basic_types, load_configured_dataset, validate_input_data
from competitor_pricing_ai.features import FeatureMetadata, engineer_market_features, select_model_features
from competitor_pricing_ai.metrics import lift_table
from competitor_pricing_ai.models import train_individual_competitor_models, train_model
from competitor_pricing_ai.reporting import build_qa_checklist, write_business_report, write_json
from competitor_pricing_ai.splits import time_based_split
from competitor_pricing_ai.tuning import apply_tuned_params, tune_hyperparameters


@dataclass
class PipelineRunResult:
    output_dir: Path
    metrics: dict[str, dict[str, float]]
    qa_checklist: dict[str, Any]
    artifacts: dict[str, str | None]
    individual_competitor_metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def run_training_pipeline(config_or_path: PipelineConfig | str | Path) -> PipelineRunResult:
    config = load_config(config_or_path) if not isinstance(config_or_path, PipelineConfig) else config_or_path
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    raw = load_configured_dataset(config)
    data_quality = validate_input_data(raw, config)
    typed = coerce_basic_types(raw, config)
    engineered, feature_metadata = engineer_market_features(typed, config)
    split = time_based_split(engineered, config)
    feature_columns, categorical_columns, numeric_columns = select_model_features(
        engineered, config, feature_metadata
    )

    if config.tuning.enabled:
        best_params = tune_hyperparameters(
            split=split,
            config=config,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            output_dir=output_dir,
        )
        config = apply_tuned_params(config, best_params)

    training_result = train_model(
        split=split,
        config=config,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        output_dir=output_dir,
    )

    individual_results: dict = {}
    if config.individual_competitor_models.enabled:
        individual_results = train_individual_competitor_models(
            split=split,
            config=config,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            competitor_columns=feature_metadata.competitor_columns,
            output_dir=output_dir,
        )

    runtime_seconds = time.perf_counter() - start

    artifacts = save_training_artifacts(
        config=config,
        output_dir=output_dir,
        data_quality=data_quality,
        split_metadata=split.metadata,
        feature_metadata=feature_metadata.to_dict(),
        training_result=training_result,
        split=split,
    )

    artifacts["market_data"] = save_market_data(split, config, feature_metadata, output_dir)

    try:
        artifacts.update(generate_basket_artefacts(training_result, split, config, output_dir))
    except Exception as exc:  # noqa: BLE001
        import warnings
        warnings.warn(f"Reference basket generation failed (non-fatal): {exc}")

    if individual_results:
        artifacts.update(save_individual_competitor_artifacts(output_dir, individual_results))

    qa_checklist = build_qa_checklist(
        config=config,
        data_quality=data_quality,
        split_metadata=split.metadata,
        metrics=training_result.metrics,
        feature_metadata=feature_metadata,
        model_path=training_result.model_path,
        mojo_path=training_result.mojo_path,
        onnx_path=training_result.onnx_path,
        runtime_seconds=runtime_seconds,
    )
    write_json(qa_checklist, output_dir / "qa_checklist.json")

    artifacts["qa_checklist"] = str(output_dir / "qa_checklist.json")
    artifacts["business_report"] = str(output_dir / "business_report.md")

    write_business_report(
        output_path=output_dir / "business_report.md",
        config=config,
        data_quality=data_quality,
        split_metadata=split.metadata,
        metrics=training_result.metrics,
        feature_importance=training_result.feature_importance,
        feature_metadata=feature_metadata,
        qa_checklist=qa_checklist,
        artifacts=artifacts,
        runtime_seconds=runtime_seconds,
        individual_results=individual_results or None,
    )

    return PipelineRunResult(
        output_dir=output_dir,
        metrics=training_result.metrics,
        qa_checklist=qa_checklist,
        artifacts=artifacts,
        individual_competitor_metrics={
            comp_col: result.metrics for comp_col, result in individual_results.items()
        },
    )


def save_training_artifacts(
    config: PipelineConfig,
    output_dir: Path,
    data_quality: dict[str, Any],
    split_metadata: dict[str, Any],
    feature_metadata: dict[str, Any],
    training_result: Any,
    split: Any,
) -> dict[str, str | None]:
    metrics_path = output_dir / "metrics.json"
    data_quality_path = output_dir / "data_quality.json"
    split_path = output_dir / "split_metadata.json"
    feature_metadata_path = output_dir / "feature_metadata.json"
    feature_importance_path = output_dir / "feature_importance.csv"
    model_features_path = output_dir / "model_features.json"
    config_path = output_dir / "run_config_resolved.yml"

    write_json(training_result.metrics, metrics_path)

    lift_table_path = output_dir / "lift_table_test.json"
    test_preds = training_result.predictions["test"]
    target_column = training_result.target_column
    if target_column in test_preds.columns and "prediction" in test_preds.columns:
        write_json(
            lift_table(test_preds[target_column], test_preds["prediction"]),
            lift_table_path,
        )

    write_json(data_quality, data_quality_path)
    write_json(split_metadata, split_path)
    write_json(feature_metadata, feature_metadata_path)
    write_json(
        {
            "feature_columns": training_result.feature_columns,
            "categorical_columns": training_result.categorical_columns,
            "numeric_columns": training_result.numeric_columns,
            "target_column": training_result.target_column,
        },
        model_features_path,
    )
    dump_resolved_config(config, config_path)
    training_result.feature_importance.to_csv(feature_importance_path, index=False)

    prediction_paths = {}
    for sample_name, prediction_frame in training_result.predictions.items():
        path = output_dir / f"predictions_{sample_name}.csv"
        prediction_frame.to_csv(path, index=False)
        prediction_paths[f"predictions_{sample_name}"] = str(path)

    reference_path = output_dir / "reference_features.csv"
    reference = build_reference_feature_frame(
        split.test,
        training_result.predictions["test"],
        training_result.feature_columns,
        config,
    )
    reference.to_csv(reference_path, index=False)

    return {
        "model": training_result.model_path,
        "mojo": training_result.mojo_path,
        "onnx": training_result.onnx_path,
        "metrics": str(metrics_path),
        "data_quality": str(data_quality_path),
        "split_metadata": str(split_path),
        "feature_metadata": str(feature_metadata_path),
        "feature_importance": str(feature_importance_path),
        "model_features": str(model_features_path),
        "reference_features": str(reference_path),
        "lift_table_test": str(lift_table_path),
        "run_config": str(config_path),
        **prediction_paths,
    }


def save_market_data(
    split: Any,
    config: PipelineConfig,
    feature_metadata: FeatureMetadata,
    output_dir: Path,
) -> str:
    """Save a combined market data CSV (all splits) for use by the dashboard."""
    keep: set[str] = set(config.data.id_columns)
    keep.add(config.data.date_column)
    if config.data.own_premium_column:
        keep.add(config.data.own_premium_column)
    if config.data.conversion_column:
        keep.add(config.data.conversion_column)
    keep.update(feature_metadata.competitor_columns)
    keep.update(feature_metadata.market_component_columns)
    keep.update(config.data.categorical_columns)
    keep.update(config.data.numeric_columns)
    keep.update(feature_metadata.temporal_columns)
    keep.add(feature_metadata.target_column)

    frames = []
    for label, df in [("train", split.train), ("validation", split.validation), ("test", split.test)]:
        cols = [c for c in df.columns if c in keep]
        part = df[cols].copy()
        part["_split"] = label
        frames.append(part)

    path = output_dir / "market_data.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return str(path)


def save_individual_competitor_artifacts(
    output_dir: Path,
    individual_results: dict,
) -> dict[str, str]:
    artifact_paths: dict[str, str] = {}
    for comp_col, result in individual_results.items():
        safe_name = comp_col.replace(" ", "_")

        metrics_path = output_dir / f"metrics_{safe_name}.json"
        write_json(result.metrics, metrics_path)
        artifact_paths[f"metrics_{safe_name}"] = str(metrics_path)

        lt_path = output_dir / f"lift_table_test_{safe_name}.json"
        test_preds = result.predictions["test"]
        if comp_col in test_preds.columns and "prediction" in test_preds.columns:
            write_json(lift_table(test_preds[comp_col], test_preds["prediction"]), lt_path)
            artifact_paths[f"lift_table_test_{safe_name}"] = str(lt_path)

        for sample_name, pred_frame in result.predictions.items():
            pred_path = output_dir / f"predictions_{sample_name}_{safe_name}.csv"
            pred_frame.to_csv(pred_path, index=False)
            artifact_paths[f"predictions_{sample_name}_{safe_name}"] = str(pred_path)

        fi_path = output_dir / f"feature_importance_{safe_name}.csv"
        result.feature_importance.to_csv(fi_path, index=False)
        artifact_paths[f"feature_importance_{safe_name}"] = str(fi_path)

        artifact_paths[f"model_{safe_name}"] = result.model_path

    return artifact_paths


def build_reference_feature_frame(
    test_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    config: PipelineConfig,
) -> pd.DataFrame:
    keep_columns = [
        column
        for column in config.data.id_columns + [config.data.date_column, config.data.target.name]
        if column in test_frame.columns
    ]
    reference = test_frame[keep_columns + feature_columns].copy()
    prediction_columns = [column for column in ["prediction", "residual", "absolute_error"] if column in prediction_frame]
    reference = reference.join(prediction_frame[prediction_columns])
    return reference
