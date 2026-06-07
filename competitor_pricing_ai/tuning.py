"""Optuna hyperparameter search for all model backends.

Each backend exposes the parameters that matter most:
  catboost  — depth, learning_rate, l2_leaf_reg
  lightgbm  — num_leaves, learning_rate, reg_lambda, min_child_samples
  sklearn   — max_leaf_nodes, learning_rate, l2_regularization

The validation set (already out-of-time by construction) is used as the
objective, so no k-fold is needed — and standard k-fold would be wrong
here because it would leak future prices into past predictions.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.metrics import regression_metrics
from competitor_pricing_ai.reporting import write_json
from competitor_pricing_ai.splits import SplitResult

logger = logging.getLogger(__name__)

_DIRECTION: dict[str, str] = {
    "mape": "minimize", "rmsle": "minimize", "rmse": "minimize",
    "mae": "minimize",  "mean_bias": "minimize",
    "d2": "maximize",   "gini": "maximize",
}


def tune_hyperparameters(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Run Optuna TPE search and return the best hyperparameter dict."""
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Tuning requires optuna. Install with: python -m pip install -e \"[tuning]\""
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    metric    = config.tuning.metric
    direction = _DIRECTION.get(metric, "minimize")
    backend   = config.model.backend

    objective = _build_objective(
        backend=backend,
        split=split,
        config=config,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        metric=metric,
    )

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=config.project.random_seed),
    )
    study.optimize(
        objective,
        n_trials=config.tuning.n_trials,
        timeout=config.tuning.timeout_seconds,
        show_progress_bar=config.tuning.show_progress_bar,
        catch=(Exception,),
    )

    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        raise RuntimeError(
            "All Optuna trials failed. Check that the backend is installed and "
            "the metric name is valid."
        )

    best_params = study.best_params
    best_value  = study.best_value
    logger.info(
        "Tuning complete — %d/%d trials succeeded, best %s=%.4f, params=%s",
        len(completed), len(study.trials), metric, best_value, best_params,
    )

    write_json(
        {
            "backend": backend,
            "metric": metric,
            "direction": direction,
            "n_trials": len(study.trials),
            "n_completed": len(completed),
            "best_value": round(float(best_value), 6),
            "best_params": best_params,
            "all_trials": [
                {"number": t.number, "value": t.value,
                 "params": t.params, "state": str(t.state)}
                for t in study.trials
            ],
        },
        output_dir / "tuning_results.json",
    )
    return best_params


def apply_tuned_params(config: PipelineConfig, best_params: dict[str, Any]) -> PipelineConfig:
    """Return a deep copy of config with Optuna best params written into the active backend."""
    config = deepcopy(config)
    backend_cfg = getattr(config.model, config.model.backend, None)
    if backend_cfg is not None:
        for key, value in best_params.items():
            if hasattr(backend_cfg, key):
                setattr(backend_cfg, key, value)
    return config


# ──────────────────────────────────────────────
# Objective dispatcher
# ──────────────────────────────────────────────

def _build_objective(
    backend: str,
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    metric: str,
) -> Callable:
    if backend == "catboost":
        return _catboost_objective(split, config, feature_columns, categorical_columns, metric)
    if backend == "lightgbm":
        return _lightgbm_objective(
            split, config, feature_columns, categorical_columns, numeric_columns, metric
        )
    if backend == "h2o":
        raise RuntimeError("Hyperparameter tuning is not supported for the h2o backend.")
    return _sklearn_objective(
        split, config, feature_columns, categorical_columns, numeric_columns, metric
    )


# ──────────────────────────────────────────────
# CatBoost
# ──────────────────────────────────────────────

def _catboost_objective(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    metric: str,
) -> Callable:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError("catboost is not installed") from exc

    cb     = config.model.catboost
    target = config.data.target.name
    # Precompute data slices once — shared across all trials
    train_x = split.train[feature_columns]
    train_y = split.train[target].to_numpy()
    val_x   = split.validation[feature_columns]
    val_y   = split.validation[target].to_numpy()

    def objective(trial):
        model = CatBoostRegressor(
            depth=trial.suggest_int("depth", 4, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 15.0, log=True),
            loss_function=cb.loss_function,
            iterations=cb.iterations,
            random_seed=config.project.random_seed,
            cat_features=categorical_columns,
            early_stopping_rounds=cb.early_stopping_rounds,
            verbose=False,
        )
        model.fit(train_x, train_y, eval_set=(val_x, val_y), use_best_model=True)
        return float(regression_metrics(val_y, model.predict(val_x))[metric])

    return objective


# ──────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────

def _lightgbm_objective(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    metric: str,
) -> Callable:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("lightgbm is not installed") from exc

    from competitor_pricing_ai.models import build_sklearn_pipeline

    lgb_cfg = config.model.lightgbm
    target  = config.data.target.name

    # Precompute preprocessing once — not part of the search space
    preprocessor = (
        build_sklearn_pipeline(config, categorical_columns, numeric_columns)
        .named_steps["preprocess"]
    )
    preprocessor.fit(split.train[feature_columns])
    X_train = preprocessor.transform(split.train[feature_columns])
    X_val   = preprocessor.transform(split.validation[feature_columns])
    train_y = split.train[target].to_numpy()
    val_y   = split.validation[target].to_numpy()

    def objective(trial):
        model = lgb.LGBMRegressor(
            objective=lgb_cfg.objective,
            num_leaves=trial.suggest_int("num_leaves", 15, 127),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
            n_estimators=lgb_cfg.n_estimators,
            random_state=config.project.random_seed,
            verbose=-1,
        )
        model.fit(
            X_train, train_y,
            eval_set=[(X_val, val_y)],
            callbacks=[
                lgb.early_stopping(lgb_cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        return float(regression_metrics(val_y, model.predict(X_val))[metric])

    return objective


# ──────────────────────────────────────────────
# sklearn HistGradientBoosting
# ──────────────────────────────────────────────

def _sklearn_objective(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    metric: str,
) -> Callable:
    from competitor_pricing_ai.models import build_sklearn_pipeline

    target  = config.data.target.name
    train_x = split.train[feature_columns]
    train_y = split.train[target].to_numpy()
    val_x   = split.validation[feature_columns]
    val_y   = split.validation[target].to_numpy()

    def objective(trial):
        trial_config = deepcopy(config)
        trial_config.model.sklearn.max_leaf_nodes  = trial.suggest_int("max_leaf_nodes", 15, 255)
        trial_config.model.sklearn.learning_rate   = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        trial_config.model.sklearn.l2_regularization = trial.suggest_float("l2_regularization", 0.0, 1.0)

        pipeline = build_sklearn_pipeline(trial_config, categorical_columns, numeric_columns)
        pipeline.fit(train_x, train_y)
        return float(regression_metrics(val_y, pipeline.predict(val_x))[metric])

    return objective
