"""Standalone diagnostic of the market anchor's incremental demand signal."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from competitor_pricing_ai.config import PipelineConfig


def evaluate_demand_readiness(
    historical: pd.DataFrame, config: PipelineConfig
) -> dict[str, Any]:
    conversion = config.data.conversion_column
    own_column = config.data.own_premium_column
    if not conversion or not own_column or conversion not in historical or own_column not in historical:
        return {"status": "not_available", "reason": "Conversion and own premium are required"}

    working = historical.dropna(subset=[conversion, own_column, "market_anchor"]).copy()
    working[conversion] = pd.to_numeric(working[conversion], errors="coerce")
    working = working[working[conversion].isin([0, 1])]
    if len(working) < config.demand_readiness.minimum_rows or working[conversion].nunique() < 2:
        return {"status": "insufficient_data", "rows": int(len(working))}

    working["log_own_premium"] = np.log(pd.to_numeric(working[own_column]).clip(lower=1e-6))
    working["log_market_anchor"] = np.log(working["market_anchor"].clip(lower=1e-6))
    working = working.sort_values(config.data.date_column)
    cut = int(len(working) * (1 - config.demand_readiness.test_fraction))
    train, test = working.iloc[:cut], working.iloc[cut:]

    categoricals = [c for c in config.data.categorical_columns if c in working]
    risk_numeric = [c for c in config.data.numeric_columns if c in working]
    baseline = categoricals + risk_numeric + ["log_own_premium"]
    enhanced = baseline + ["log_market_anchor"]
    baseline_metrics = _fit_demand_model(train, test, conversion, baseline, categoricals, config)
    enhanced_metrics = _fit_demand_model(train, test, conversion, enhanced, categoricals, config)
    return {
        "status": "diagnostic_only",
        "purpose": "Standalone proxy; final acceptance belongs in the governed demand workflow.",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline": baseline_metrics,
        "with_market_anchor": enhanced_metrics,
        "incremental_log_loss_improvement": (
            baseline_metrics["log_loss"] - enhanced_metrics["log_loss"]
        ),
        "incremental_brier_improvement": (
            baseline_metrics["brier"] - enhanced_metrics["brier"]
        ),
        "passes_incremental_signal_check": (
            enhanced_metrics["log_loss"] < baseline_metrics["log_loss"]
        ),
    }


def _fit_demand_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    features: list[str],
    categoricals: list[str],
    config: PipelineConfig,
) -> dict[str, float]:
    numeric = [column for column in features if column not in categoricals]
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), categoricals),
    ])
    model = Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=config.project.random_seed)),
    ])
    fit_kwargs = {}
    if config.data.weight_column and config.data.weight_column in train:
        fit_kwargs["model__sample_weight"] = train[config.data.weight_column].to_numpy()
    model.fit(train[features], train[target], **fit_kwargs)
    prediction = model.predict_proba(test[features])[:, 1]
    actual = test[target].to_numpy()
    return {
        "log_loss": float(log_loss(actual, prediction)),
        "brier": float(brier_score_loss(actual, prediction)),
        "auc": float(roc_auc_score(actual, prediction)),
        "mean_prediction": float(np.mean(prediction)),
        "actual_rate": float(np.mean(actual)),
        "calibration_bias": float(np.mean(prediction) - np.mean(actual)),
    }
