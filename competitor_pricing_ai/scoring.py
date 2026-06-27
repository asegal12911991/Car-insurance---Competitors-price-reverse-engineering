"""Standalone batch scoring contract for a frozen competitor market anchor."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from competitor_pricing_ai.config import PipelineConfig, load_config, resolve_project_path
from competitor_pricing_ai.data import load_dataset
from competitor_pricing_ai.features import add_temporal_features
from competitor_pricing_ai.governance import verify_manifest_artifact
from competitor_pricing_ai.models import load_sklearn_bundle, predict_with_bundle


def score_market_anchor(
    config_or_path: PipelineConfig | str | Path,
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path | None = None,
) -> Path:
    config = load_config(config_or_path) if not isinstance(config_or_path, PipelineConfig) else config_or_path
    model_file = Path(model_path) if model_path else config.output_dir / "model.joblib"
    if not model_file.exists():
        raise ValueError(f"Model bundle does not exist: {model_file}")
    verify_manifest_artifact(model_file.parent, "model", model_file)
    source = resolve_project_path(input_path, config.root_dir)
    frame = load_dataset(source).copy()
    bundle = load_sklearn_bundle(model_file)

    date_column = config.data.date_column
    if date_column not in frame:
        raise ValueError(f"Scoring input is missing date column: {date_column}")
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    if frame[date_column].isna().any():
        raise ValueError("Scoring input contains invalid quote dates")
    for column in config.data.numeric_columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in config.data.categorical_columns:
        if column in frame:
            frame[column] = frame[column].astype("string")
    add_temporal_features(frame, date_column)

    missing = sorted(set(bundle["feature_columns"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Scoring input cannot create required model features: {missing}")
    anchor = np.maximum(predict_with_bundle(bundle, frame), 1e-6)

    keep = [column for column in config.data.id_columns + [date_column] if column in frame]
    output = frame[keep].copy()
    output["market_anchor"] = anchor
    own_column = config.data.own_premium_column
    if own_column and own_column in frame:
        own = pd.to_numeric(frame[own_column], errors="coerce").clip(lower=1e-6)
        output[own_column] = own
        output["relative_price_ratio"] = own / anchor
        output["log_relative_price"] = np.log(own / anchor)
        output["price_gap_to_market_anchor"] = own - anchor
    output["anchor_is_frozen_for_optimization"] = True
    output["anchor_model_sha256"] = hashlib.sha256(model_file.read_bytes()).hexdigest()
    output["anchor_training_cutoff"] = bundle.get("training_cutoff")
    cutoff = pd.to_datetime(bundle.get("training_cutoff"), errors="coerce")
    if pd.notna(cutoff):
        output["prediction_horizon_days"] = (frame[date_column] - cutoff).dt.days

    destination = resolve_project_path(output_path, config.root_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination
