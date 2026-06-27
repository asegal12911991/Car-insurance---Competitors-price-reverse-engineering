# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Install (core dependencies only)
python -m pip install -e .

# Install all optional extras for local development
python -m pip install -e ".[catboost,lightgbm,tuning,onnx,plots,dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_features.py

# Run a single test by name
pytest tests/test_features.py::test_engineer_market_features_average_top_n_and_rank

# Lint
ruff check competitor_pricing_ai/ tests/

# Train
competitor-pricing-ai train --config configs/config.example.yml

# Validate config without training
competitor-pricing-ai validate-config --config configs/config.example.yml

# Run monitoring
competitor-pricing-ai monitor --config configs/config.example.yml

# Score a standalone market-anchor handoff file
competitor-pricing-ai score --config configs/config.example.yml --input quotes.csv --output anchors.csv

# Generate synthetic data for testing
python scripts/generate_sample_data.py --output data/raw/competitor_quotes.csv

# Launch interactive dashboard
streamlit run scripts/dashboard.py -- --output-dir output/run_example
```

Line length limit is 100 characters (enforced by ruff). Target is Python 3.10+.

## Architecture

The package is `competitor_pricing_ai/`. The pipeline is orchestrated by `pipeline.py:run_training_pipeline()`, which calls each module in sequence. The CLI in `cli.py` is the only entry point — it calls `run_training_pipeline()` for `train`, `run_monitoring()` for `monitor`, and `validate_config()` for `validate-config`.

### Data flow through the pipeline

```
config.py        → PipelineConfig dataclass (load_config / validate_config)
data.py          → load + validate raw DataFrame, write data_quality.json
features.py      → engineer market features, return FeatureMetadata
splits.py        → time_based_split → SplitResult (train/validation/test DataFrames)
tuning.py        → [optional] Optuna TPE search, return best_params dict
models.py        → train_model (dispatches to backend) → ModelTrainingResult
                   train_individual_competitor_models → list[IndividualCompetitorModelResult]
basket.py        → generate_basket_artefacts → reference_basket.csv + reference_basket_index.csv
metrics.py       → regression_metrics, lift_table
reporting.py     → build_qa_checklist, write_business_report
monitoring.py    → run_monitoring (PSI drift + performance degradation)
```

All outputs land in `output/<run_name>/` as configured by `project.output_dir`.

### Key design constraints

**Time-based splitting only.** `splits.py` refuses k-fold by design — k-fold leaks future competitor prices into past training rows. Validation and test sets are always strictly out-of-time.

**Target leakage prevention.** `features.py` engineers market component columns (`avg_top_N_competitor_premium`, ratios, rank, dispersion) but `select_model_features()` excludes all of them from the model feature set by default. Only user-configured `categorical_columns` and `numeric_columns` (plus temporal columns) reach the model.

**Gamma/Tweedie loss across all backends.** Insurance premiums are positive and right-skewed. All four backends use Gamma or Tweedie loss. D² (Gamma deviance explained) is the primary deployment gate, not R².

**Model serialisation.** `model.joblib` is a dict `{model, backend, feature_columns, categorical_columns, numeric_columns, target_column, target_transform}`, not a bare model object. `monitoring.py` and `basket.py` use `load_sklearn_bundle()` / `predict_with_bundle()` from `models.py` to score saved models — never unpickle and call `.predict()` directly.

**Frozen market-anchor contract.** Own premium, conversion, competitor observations, IDs,
weights, and target-derived fields are hard-excluded from competitor-model predictors.
`historical.py` creates finite-lookback prior-month predictions for demand development;
`scoring.py` creates batch-scored frozen anchors for downstream handoff.

**Dynamic-market recency.** Evaluation training, rolling-origin historical anchors, and the
final deployable refit use the configured recent lookback. Exponential half-life weights can
further emphasize recent observations. The evaluated model is archived separately from the
latest-window production model.

**Incomplete competitor panels.** The default target requires at least `top_n` observed
premiums rather than a fully complete panel. `panel.py` produces monthly coverage, eligibility,
top-N composition, and incomplete-panel bias diagnostics used by QA and the dashboard.

**Standalone boundary.** This repository does not call an external demand or optimisation
platform. `demand_readiness.json` is a local signal diagnostic; final acceptance is downstream.

### Four model backends

| Backend | Module path | Serialisation |
|---|---|---|
| `sklearn` (default) | `train_sklearn_model()` | `model.joblib` |
| `catboost` | `train_catboost_model()` | `model.joblib` + `model.cbm` |
| `lightgbm` | `train_lightgbm_model()` | `model.joblib` |
| `h2o` | `train_h2o_model()` | `model.joblib` + `model_mojo.zip` |

H2O does not support Optuna tuning (raises an explicit error if both are enabled).
ONNX export (`model.onnx`) is sklearn-only, via `skl2onnx`.

### Config loading

`config.py:load_config(path)` accepts a file path or an existing `PipelineConfig` instance (used by tests). Relative paths in the YAML are resolved relative to the YAML file's directory, not the working directory. `validate_config()` is called automatically inside `load_config()`.

### Dashboard

`scripts/dashboard.py` is a self-contained Streamlit app (8 tabs). It reads only from the `output/` directory — no pipeline code is imported at dashboard runtime. It discovers run directories automatically and loads whichever JSON/CSV artifacts are present, rendering charts conditionally when optional artifacts (e.g. `tuning_results.json`, per-competitor model files) exist. The eighth tab covers downstream handoff readiness and run governance.
