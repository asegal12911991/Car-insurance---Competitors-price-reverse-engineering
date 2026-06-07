"""Model training backends."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.metrics import regression_metrics
from competitor_pricing_ai.splits import SplitResult


@dataclass
class IndividualCompetitorModelResult:
    competitor_column: str
    model_path: str
    metrics: dict[str, dict[str, float]]
    feature_importance: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    n_train: int
    n_validation: int
    n_test: int
    missing_rate_train: float


@dataclass
class ModelTrainingResult:
    backend: str
    model_path: str | None
    mojo_path: str | None
    onnx_path: str | None
    metrics: dict[str, dict[str, float]]
    feature_importance: pd.DataFrame
    predictions: dict[str, pd.DataFrame]
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    target_column: str


def train_model(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> ModelTrainingResult:
    dispatch = {
        "sklearn":   train_sklearn_model,
        "catboost":  train_catboost_model,
        "lightgbm":  train_lightgbm_model,
        "h2o":       train_h2o_model,
    }
    return dispatch[config.model.backend](
        split, config, feature_columns, categorical_columns, numeric_columns, output_dir
    )


def train_sklearn_model(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> ModelTrainingResult:
    target_column = config.data.target.name
    pipeline = build_sklearn_pipeline(config, categorical_columns, numeric_columns)

    # log1p transform is redundant when using gamma loss (which already handles
    # multiplicative structure via its log link). Only apply for squared_error loss.
    if config.model.target_transform == "log1p" and config.model.sklearn.loss == "squared_error":
        pipeline = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    train_x, train_y = split.train[feature_columns], split.train[target_column]
    validation_x, validation_y = split.validation[feature_columns], split.validation[target_column]
    test_x, test_y = split.test[feature_columns], split.test[target_column]

    pipeline.fit(train_x, train_y)

    validation_pred = pipeline.predict(validation_x)
    test_pred = pipeline.predict(test_x)
    train_pred = pipeline.predict(train_x)

    metrics = {
        "train": regression_metrics(train_y, train_pred),
        "validation": regression_metrics(validation_y, validation_pred),
        "test": regression_metrics(test_y, test_pred),
    }

    feature_importance = calculate_permutation_importance(
        pipeline,
        split.validation,
        feature_columns,
        target_column,
        config,
    )

    bundle = {
        "model": pipeline,
        "backend": "sklearn",
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "target_column": target_column,
        "target_transform": config.model.target_transform,
    }
    model_path = output_dir / "model.joblib"
    joblib.dump(bundle, model_path)
# Commit test
    onnx_path = None
    if config.model.export_onnx:
        onnx_path = export_sklearn_onnx(pipeline, feature_columns, categorical_columns, output_dir)

    predictions = {
        "train": make_prediction_frame(split.train, target_column, train_pred, config),
        "validation": make_prediction_frame(split.validation, target_column, validation_pred, config),
        "test": make_prediction_frame(split.test, target_column, test_pred, config),
    }

    return ModelTrainingResult(
        backend="sklearn",
        model_path=str(model_path),
        mojo_path=None,
        onnx_path=onnx_path,
        metrics=metrics,
        feature_importance=feature_importance,
        predictions=predictions,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target_column=target_column,
    )


def build_sklearn_pipeline(
    config: PipelineConfig,
    categorical_columns: list[str],
    numeric_columns: list[str],
) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    estimator = HistGradientBoostingRegressor(
        loss=config.model.sklearn.loss,
        max_iter=config.model.sklearn.max_iter,
        learning_rate=config.model.sklearn.learning_rate,
        max_leaf_nodes=config.model.sklearn.max_leaf_nodes,
        l2_regularization=config.model.sklearn.l2_regularization,
        random_state=config.project.random_seed,
        early_stopping=True,
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def export_sklearn_onnx(
    pipeline: Any,
    feature_columns: list[str],
    categorical_columns: list[str],
    output_dir: Path,
) -> str:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export was requested but skl2onnx is not installed. "
            'Install with: python -m pip install -e "[onnx]"'
        ) from exc

    # TransformedTargetRegressor (log1p) is not supported by skl2onnx — use the inner pipeline
    inner = getattr(pipeline, "regressor_", pipeline)
    cat_set = set(categorical_columns)
    initial_types = [
        (col, StringTensorType([None, 1])) if col in cat_set else (col, FloatTensorType([None, 1]))
        for col in feature_columns
    ]
    onnx_model = convert_sklearn(inner, initial_types=initial_types)
    onnx_path = output_dir / "model.onnx"
    with onnx_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())
    return str(onnx_path)


def calculate_permutation_importance(
    fitted_model: Any,
    validation: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    config: PipelineConfig,
) -> pd.DataFrame:
    sample_size = min(len(validation), config.evaluation.importance_sample_size)
    sample = validation.sample(
        n=sample_size,
        random_state=config.project.random_seed,
    )
    importance = permutation_importance(
        fitted_model,
        sample[feature_columns],
        sample[target_column],
        n_repeats=config.evaluation.permutation_importance_repeats,
        random_state=config.project.random_seed,
        scoring="r2",
    )
    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    )
    return frame.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def make_prediction_frame(
    frame: pd.DataFrame,
    target_column: str,
    predictions: np.ndarray,
    config: PipelineConfig,
) -> pd.DataFrame:
    keep_columns = [
        column
        for column in config.data.id_columns + [config.data.date_column, target_column]
        if column in frame.columns
    ]
    result = frame[keep_columns].copy()
    result["prediction"] = predictions
    result["residual"] = result[target_column] - result["prediction"]
    result["absolute_error"] = result["residual"].abs()
    result["absolute_percentage_error"] = (
        result["absolute_error"] / result[target_column].replace(0, np.nan)
    ) * 100
    return result


def train_individual_competitor_models(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    competitor_columns: list[str],
    output_dir: Path,
) -> dict[str, IndividualCompetitorModelResult]:
    """Train one sklearn model per competitor column, filtering rows where that competitor has no quote."""
    threshold = config.individual_competitor_models.skip_missing_threshold
    results: dict[str, IndividualCompetitorModelResult] = {}

    for comp_col in competitor_columns:
        if comp_col not in split.train.columns:
            continue
        missing_rate = float(split.train[comp_col].isna().mean())
        if missing_rate > threshold:
            continue

        train_df = split.train[split.train[comp_col].notna()].copy()
        val_df = split.validation[split.validation[comp_col].notna()].copy()
        test_df = split.test[split.test[comp_col].notna()].copy()

        if len(train_df) < 20 or len(val_df) < 10 or len(test_df) < 10:
            continue

        pipeline = build_sklearn_pipeline(config, categorical_columns, numeric_columns)
        if config.model.target_transform == "log1p" and config.model.sklearn.loss == "squared_error":
            pipeline = TransformedTargetRegressor(
                regressor=pipeline,
                func=np.log1p,
                inverse_func=np.expm1,
                check_inverse=False,
            )

        pipeline.fit(train_df[feature_columns], train_df[comp_col])
        train_pred = pipeline.predict(train_df[feature_columns])
        val_pred = pipeline.predict(val_df[feature_columns])
        test_pred = pipeline.predict(test_df[feature_columns])

        metrics = {
            "train": regression_metrics(train_df[comp_col], train_pred),
            "validation": regression_metrics(val_df[comp_col], val_pred),
            "test": regression_metrics(test_df[comp_col], test_pred),
        }
        importance = calculate_permutation_importance(
            pipeline, val_df, feature_columns, comp_col, config
        )

        safe_name = comp_col.replace(" ", "_")
        model_path = output_dir / f"model_{safe_name}.joblib"
        joblib.dump(
            {
                "model": pipeline,
                "backend": "sklearn",
                "feature_columns": feature_columns,
                "categorical_columns": categorical_columns,
                "numeric_columns": numeric_columns,
                "target_column": comp_col,
                "target_transform": config.model.target_transform,
            },
            model_path,
        )

        predictions = {
            "train": make_prediction_frame(train_df, comp_col, train_pred, config),
            "validation": make_prediction_frame(val_df, comp_col, val_pred, config),
            "test": make_prediction_frame(test_df, comp_col, test_pred, config),
        }

        results[comp_col] = IndividualCompetitorModelResult(
            competitor_column=comp_col,
            model_path=str(model_path),
            metrics=metrics,
            feature_importance=importance,
            predictions=predictions,
            n_train=len(train_df),
            n_validation=len(val_df),
            n_test=len(test_df),
            missing_rate_train=missing_rate,
        )

    return results


def train_catboost_model(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> ModelTrainingResult:
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost backend selected but catboost is not installed. "
            'Install with: python -m pip install -e "[catboost]"'
        ) from exc

    target_column = config.data.target.name

    # CatBoost handles string/object categoricals natively — no OrdinalEncoder needed
    train_x = split.train[feature_columns]
    val_x   = split.validation[feature_columns]
    test_x  = split.test[feature_columns]
    train_y, val_y, test_y = (
        split.train[target_column],
        split.validation[target_column],
        split.test[target_column],
    )

    cb = CatBoostConfig = config.model.catboost
    model = CatBoostRegressor(
        loss_function=cb.loss_function,
        iterations=cb.iterations,
        learning_rate=cb.learning_rate,
        depth=cb.depth,
        l2_leaf_reg=cb.l2_leaf_reg,
        random_seed=config.project.random_seed,
        cat_features=categorical_columns,
        early_stopping_rounds=cb.early_stopping_rounds,
        verbose=False,
    )
    model.fit(train_x, train_y, eval_set=(val_x, val_y), use_best_model=True)

    train_pred = model.predict(train_x)
    val_pred   = model.predict(val_x)
    test_pred  = model.predict(test_x)

    metrics = {
        "train":      regression_metrics(train_y, train_pred),
        "validation": regression_metrics(val_y, val_pred),
        "test":       regression_metrics(test_y, test_pred),
    }

    importance = calculate_permutation_importance(model, split.validation, feature_columns, target_column, config)

    # Native .cbm save + joblib bundle for monitoring compatibility
    cbm_path = output_dir / "model.cbm"
    model.save_model(str(cbm_path))
    bundle_path = output_dir / "model.joblib"
    joblib.dump(
        {"model": model, "backend": "catboost", "feature_columns": feature_columns,
         "categorical_columns": categorical_columns, "numeric_columns": numeric_columns,
         "target_column": target_column, "target_transform": "none"},
        bundle_path,
    )

    onnx_path = None
    if config.model.export_onnx:
        onnx_p = output_dir / "model.onnx"
        model.save_model(str(onnx_p), format="onnx")  # CatBoost native ONNX export
        onnx_path = str(onnx_p)

    predictions = {
        "train":      make_prediction_frame(split.train,      target_column, train_pred, config),
        "validation": make_prediction_frame(split.validation, target_column, val_pred,   config),
        "test":       make_prediction_frame(split.test,       target_column, test_pred,  config),
    }

    return ModelTrainingResult(
        backend="catboost",
        model_path=str(bundle_path),
        mojo_path=None,
        onnx_path=onnx_path,
        metrics=metrics,
        feature_importance=importance,
        predictions=predictions,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target_column=target_column,
    )


class _WrappedLGBM:
    """Thin wrapper so permutation_importance can score a preprocessor+LightGBM pair."""
    def __init__(self, preprocessor: Any, model: Any) -> None:
        self.preprocessor = preprocessor
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self.preprocessor.transform(X))

    def score(self, X: pd.DataFrame, y: Any) -> float:
        from sklearn.metrics import r2_score
        return float(r2_score(y, self.predict(X)))


def train_lightgbm_model(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> ModelTrainingResult:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM backend selected but lightgbm is not installed. "
            'Install with: python -m pip install -e "[lightgbm]"'
        ) from exc

    target_column = config.data.target.name

    # Reuse sklearn ColumnTransformer for preprocessing; pass encoded arrays to LightGBM
    sklearn_pipeline = build_sklearn_pipeline(config, categorical_columns, numeric_columns)
    preprocessor = sklearn_pipeline.named_steps["preprocess"]
    preprocessor.fit(split.train[feature_columns])

    X_train = preprocessor.transform(split.train[feature_columns])
    X_val   = preprocessor.transform(split.validation[feature_columns])
    X_test  = preprocessor.transform(split.test[feature_columns])
    train_y, val_y, test_y = (
        split.train[target_column].to_numpy(),
        split.validation[target_column].to_numpy(),
        split.test[target_column].to_numpy(),
    )

    lgb_cfg = config.model.lightgbm
    model = lgb.LGBMRegressor(
        objective=lgb_cfg.objective,
        n_estimators=lgb_cfg.n_estimators,
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        reg_lambda=lgb_cfg.reg_lambda,
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

    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)
    test_pred  = model.predict(X_test)

    metrics = {
        "train":      regression_metrics(train_y, train_pred),
        "validation": regression_metrics(val_y, val_pred),
        "test":       regression_metrics(test_y, test_pred),
    }

    wrapped = _WrappedLGBM(preprocessor, model)
    importance = calculate_permutation_importance(
        wrapped, split.validation, feature_columns, target_column, config
    )

    model_path = output_dir / "model.joblib"
    joblib.dump(
        {"model": model, "preprocessor": preprocessor, "backend": "lightgbm",
         "feature_columns": feature_columns, "categorical_columns": categorical_columns,
         "numeric_columns": numeric_columns, "target_column": target_column,
         "target_transform": "none"},
        model_path,
    )

    predictions = {
        "train":      make_prediction_frame(split.train,      target_column, train_pred, config),
        "validation": make_prediction_frame(split.validation, target_column, val_pred,   config),
        "test":       make_prediction_frame(split.test,       target_column, test_pred,  config),
    }

    return ModelTrainingResult(
        backend="lightgbm",
        model_path=str(model_path),
        mojo_path=None,
        onnx_path=None,
        metrics=metrics,
        feature_importance=importance,
        predictions=predictions,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target_column=target_column,
    )


def load_sklearn_bundle(model_path: str | Path) -> dict[str, Any]:
    """Load any joblib model bundle (sklearn, catboost, or lightgbm backend)."""
    return joblib.load(model_path)


def predict_with_sklearn_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    return predict_with_bundle(bundle, frame)


def predict_with_bundle(bundle: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    X = frame[bundle["feature_columns"]]
    if bundle.get("backend") == "lightgbm" and "preprocessor" in bundle:
        return bundle["model"].predict(bundle["preprocessor"].transform(X))
    return bundle["model"].predict(X)


def train_h2o_model(
    split: SplitResult,
    config: PipelineConfig,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    output_dir: Path,
) -> ModelTrainingResult:
    try:
        import h2o
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
    except ImportError as exc:
        raise RuntimeError(
            "The H2O backend was selected, but h2o is not installed. "
            "Install with: python -m pip install -e \".[h2o]\""
        ) from exc

    h2o.init(max_mem_size=f"{config.model.memory_gb}G")

    target_column = config.data.target.name
    train_hf = to_h2o_frame(split.train, feature_columns, target_column, categorical_columns, h2o)
    validation_hf = to_h2o_frame(
        split.validation, feature_columns, target_column, categorical_columns, h2o
    )
    test_hf = to_h2o_frame(split.test, feature_columns, target_column, categorical_columns, h2o)

    model = H2OGradientBoostingEstimator(
        distribution=config.model.h2o.distribution,
        ntrees=config.model.h2o.ntrees,
        max_depth=config.model.h2o.max_depth,
        learn_rate=config.model.h2o.learn_rate,
        seed=config.project.random_seed,
        stopping_rounds=10,
        stopping_metric="deviance",
        score_tree_interval=10,
    )
    model.train(
        x=feature_columns,
        y=target_column,
        training_frame=train_hf,
        validation_frame=validation_hf,
    )

    train_pred = h2o_predict(model, train_hf)
    validation_pred = h2o_predict(model, validation_hf)
    test_pred = h2o_predict(model, test_hf)

    metrics = {
        "train": regression_metrics(split.train[target_column], train_pred),
        "validation": regression_metrics(split.validation[target_column], validation_pred),
        "test": regression_metrics(split.test[target_column], test_pred),
    }

    try:
        feature_importance = model.varimp(use_pandas=True).rename(
            columns={
                "variable": "feature",
                "relative_importance": "importance_mean",
                "percentage": "importance_percentage",
            }
        )
        if "importance_mean" not in feature_importance.columns:
            feature_importance["importance_mean"] = np.nan
    except Exception:
        feature_importance = pd.DataFrame(
            {"feature": feature_columns, "importance_mean": np.nan, "importance_std": np.nan}
        )

    model_path = h2o.save_model(model=model, path=str(output_dir), force=True)
    mojo_path = None
    if config.model.h2o.export_mojo:
        mojo_path = model.download_mojo(path=str(output_dir), get_genmodel_jar=False)

    predictions = {
        "train": make_prediction_frame(split.train, target_column, train_pred, config),
        "validation": make_prediction_frame(split.validation, target_column, validation_pred, config),
        "test": make_prediction_frame(split.test, target_column, test_pred, config),
    }

    return ModelTrainingResult(
        backend="h2o",
        model_path=model_path,
        mojo_path=mojo_path,
        onnx_path=None,
        metrics=metrics,
        feature_importance=feature_importance,
        predictions=predictions,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target_column=target_column,
    )


def to_h2o_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    categorical_columns: list[str],
    h2o_module: Any,
):
    h2o_frame = h2o_module.H2OFrame(frame[feature_columns + [target_column]])
    for column in categorical_columns:
        if column in h2o_frame.columns:
            h2o_frame[column] = h2o_frame[column].asfactor()
    return h2o_frame


def h2o_predict(model: Any, frame: Any) -> np.ndarray:
    return model.predict(frame).as_data_frame(use_multi_thread=True).iloc[:, 0].to_numpy()
