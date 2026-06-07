"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a pipeline configuration is invalid."""


@dataclass
class ProjectConfig:
    name: str = "competitor_pricing_ai"
    random_seed: int = 42
    output_dir: str = "output/run"


@dataclass
class TargetConfig:
    name: str = "avg_top_3_competitor_premium"
    top_n: int = 3


@dataclass
class DataConfig:
    input_path: str
    date_column: str
    own_premium_column: str | None = None
    conversion_column: str | None = None
    id_columns: list[str] = field(default_factory=list)
    competitor_columns: list[str] = field(default_factory=list)
    competitor_column_regex: str | None = None
    target: TargetConfig = field(default_factory=TargetConfig)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    leakage_columns: list[str] = field(default_factory=list)


@dataclass
class SegmentAggressivenessConfig:
    enabled: bool = False
    segment_columns: list[str] = field(default_factory=list)


@dataclass
class FeaturesConfig:
    top_ns: list[int] = field(default_factory=lambda: [3, 5])
    add_competitor_distribution: bool = True
    add_relative_position: bool = True
    add_temporal_features: bool = True
    add_segment_aggressiveness: SegmentAggressivenessConfig = field(
        default_factory=SegmentAggressivenessConfig
    )


@dataclass
class SplitConfig:
    strategy: str = "time"
    validation_fraction: float = 0.15
    test_fraction: float = 0.20
    train_end_date: str | None = None
    validation_end_date: str | None = None


@dataclass
class SklearnConfig:
    max_iter: int = 400
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.01
    loss: str = "gamma"


@dataclass
class H2OConfig:
    algorithm: str = "gbm"
    ntrees: int = 3000
    max_depth: int = 6
    learn_rate: float = 0.03
    export_mojo: bool = True
    distribution: str = "gamma"


@dataclass
class CatBoostConfig:
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    loss_function: str = "Tweedie:variance_power=2"  # Gamma deviance
    early_stopping_rounds: int = 50


@dataclass
class LightGBMConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.05
    num_leaves: int = 31
    reg_lambda: float = 0.1
    objective: str = "gamma"  # Gamma deviance
    early_stopping_rounds: int = 50


@dataclass
class ModelConfig:
    backend: str = "catboost"
    algorithm: str = "hist_gradient_boosting"
    objective: str = "regression"
    target_transform: str = "none"
    max_runtime_seconds: int = 3600
    memory_gb: int = 8
    export_onnx: bool = False
    sklearn: SklearnConfig = field(default_factory=SklearnConfig)
    h2o: H2OConfig = field(default_factory=H2OConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)


@dataclass
class EvaluationConfig:
    d2_min: float = 0.75
    rmse_max: float = 60.0
    mape_max: float = 15.0
    gini_min: float = 0.30
    mean_bias_pct_max: float = 5.0
    permutation_importance_repeats: int = 5
    importance_sample_size: int = 5000


@dataclass
class MonitoringConfig:
    drift_reference_path: str | None = None
    current_data_path: str | None = None
    psi_threshold: float = 0.20
    performance_d2_drop_threshold: float = 0.05
    performance_rmse_increase_threshold: float = 15.0
    performance_gini_drop_threshold: float = 0.05
    performance_mape_increase_threshold: float = 3.0


@dataclass
class TuningConfig:
    enabled: bool = True
    n_trials: int = 50
    metric: str = "mape"          # metric to optimise: mape, d2, gini, rmsle, rmse
    timeout_seconds: int | None = None   # optional wall-clock cap
    show_progress_bar: bool = True


@dataclass
class IndividualCompetitorModelsConfig:
    enabled: bool = False
    skip_missing_threshold: float = 0.40


@dataclass
class PipelineConfig:
    project: ProjectConfig
    data: DataConfig
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    individual_competitor_models: IndividualCompetitorModelsConfig = field(
        default_factory=IndividualCompetitorModelsConfig
    )
    tuning: TuningConfig = field(default_factory=TuningConfig)
    config_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("config_path", None)
        return result

    @property
    def root_dir(self) -> Path:
        if self.config_path:
            config_parent = Path(self.config_path).resolve().parent
            if config_parent.name.lower() in {"config", "configs"}:
                return config_parent.parent
            return config_parent
        return Path.cwd()

    @property
    def output_dir(self) -> Path:
        return resolve_project_path(self.project.output_dir, self.root_dir)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    try:
        project = ProjectConfig(**raw.get("project", {}))
        target = TargetConfig(**raw.get("data", {}).get("target", {}))
        data_raw = dict(raw.get("data", {}))
        data_raw["target"] = target
        data = DataConfig(**data_raw)

        features_raw = dict(raw.get("features", {}))
        segment_raw = features_raw.get("add_segment_aggressiveness", {})
        if isinstance(segment_raw, bool):
            segment_raw = {"enabled": segment_raw}
        features_raw["add_segment_aggressiveness"] = SegmentAggressivenessConfig(**segment_raw)
        features = FeaturesConfig(**features_raw)

        split = SplitConfig(**raw.get("split", {}))

        model_raw = dict(raw.get("model", {}))
        model_raw["sklearn"] = SklearnConfig(**model_raw.get("sklearn", {}))
        model_raw["h2o"] = H2OConfig(**model_raw.get("h2o", {}))
        model_raw["catboost"] = CatBoostConfig(**model_raw.get("catboost", {}))
        model_raw["lightgbm"] = LightGBMConfig(**model_raw.get("lightgbm", {}))
        model = ModelConfig(**model_raw)

        evaluation = EvaluationConfig(**raw.get("evaluation", {}))
        monitoring = MonitoringConfig(**raw.get("monitoring", {}))
        individual_raw = raw.get("individual_competitor_models", {})
        if isinstance(individual_raw, bool):
            individual_raw = {"enabled": individual_raw}
        individual_competitor_models = IndividualCompetitorModelsConfig(**individual_raw)
    except TypeError as exc:
        raise ConfigError(f"Invalid configuration structure: {exc}") from exc

    config = PipelineConfig(
        project=project,
        data=data,
        features=features,
        split=split,
        model=model,
        evaluation=evaluation,
        monitoring=monitoring,
        individual_competitor_models=individual_competitor_models,
        tuning=TuningConfig(**raw.get("tuning", {})),
        config_path=str(config_path),
    )
    validate_config(config)
    return config


def validate_config(config: PipelineConfig) -> None:
    if not config.data.input_path:
        raise ConfigError("data.input_path is required")
    if not config.data.date_column:
        raise ConfigError("data.date_column is required")
    if not config.data.competitor_columns and not config.data.competitor_column_regex:
        raise ConfigError(
            "Provide either data.competitor_columns or data.competitor_column_regex"
        )
    if config.data.target.top_n <= 0:
        raise ConfigError("data.target.top_n must be positive")
    if any(top_n <= 0 for top_n in config.features.top_ns):
        raise ConfigError("features.top_ns must contain only positive integers")
    if config.split.strategy != "time":
        raise ConfigError("Only split.strategy: time is currently supported")
    if not (0 < config.split.validation_fraction < 1):
        raise ConfigError("split.validation_fraction must be between 0 and 1")
    if not (0 < config.split.test_fraction < 1):
        raise ConfigError("split.test_fraction must be between 0 and 1")
    if config.split.validation_fraction + config.split.test_fraction >= 0.8:
        raise ConfigError("Validation plus test fractions leave too little training data")
    if config.model.backend not in {"sklearn", "h2o", "catboost", "lightgbm"}:
        raise ConfigError("model.backend must be one of: sklearn, h2o, catboost, lightgbm")
    if config.model.objective != "regression":
        raise ConfigError("Only regression objective is currently supported")
    if config.model.target_transform not in {"none", "log1p"}:
        raise ConfigError("model.target_transform must be 'none' or 'log1p'")


def resolve_project_path(path: str | Path, root_dir: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root_dir or Path.cwd()).resolve() / candidate


def dump_resolved_config(config: PipelineConfig, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
