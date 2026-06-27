# Competitor Pricing AI — System Blueprint

## Purpose

A configurable machine-learning pipeline that reverse engineers car insurance competitor
pricing behaviour and converts the output into actionable pricing intelligence.

The modelled target is an aggregated market component — by default
`avg_top_3_competitor_premium` — because top-N competitor prices are more stable than
individual competitor prices and more directly useful for optimisation.

For production handoff the prediction is called `market_anchor`. It is independent of own
premium and remains fixed while a separate optimiser varies candidate own price.

---

## Pipeline Architecture

```
Raw quotes CSV
      │
      ▼
1. Data ingestion & validation
      │
      ▼
2. Market feature engineering
      │
      ▼
3. Time-based train / validation / test split
      │
      ▼
4a. [Optional] Optuna hyperparameter tuning  ──► tuning_results.json
      │
      ▼
4b. Model training  ──────────────────────────► model.joblib / model.cbm
      │
      ▼
4c. Reference basket index  ──────────────────► reference_basket_index.csv
      │
      ▼
4d. [Optional] Individual competitor models  ──► model_{competitor}.joblib
      │
      ▼
5. Evaluation & QA checklist  ────────────────► metrics.json, qa_checklist.json
      │
      ▼
6. Artefact export  ──────────────────────────► predictions_*.csv, feature_importance.csv
      │
      ▼
7. Business report  ──────────────────────────► business_report.md
      │
      ▼
8. [Optional] Monitoring  ────────────────────► monitoring_metrics.json, monitoring_report.md
```

---

### Standalone demand-model contract

- `historical_market_features.csv` uses only competitor observations before each scoring month.
- Warm-up observations are unscored; they are never backfilled using future information.
- `score` accepts risk/date fields without requiring competitor premiums.
- `market_anchor` is frozen; only own candidate premium moves in downstream optimisation.
- `log_relative_price = log(candidate own premium / market_anchor)` is the recommended input.
- `demand_readiness.json` is diagnostic only and does not replace downstream governance.

## Step Details

### 1 · Data Ingestion & Validation (`data.py`)

- Loads any CSV quoted with a configurable path.
- Detects competitor columns by explicit list (`data.competitor_columns`) or regex
  (`data.competitor_column_regex`).
- Validates quote dates, non-positive prices, and sufficient competitor panel completeness.
- Coerces dates, numeric fields, and categorical fields consistently.
- Writes `data_quality.json` with row counts, date ranges, and coverage statistics.

### 2 · Market Feature Engineering (`features.py`)

Built fields (all excluded from model features by default to prevent target leakage):

| Field | Description |
|---|---|
| `avg_top_N_competitor_premium` | Mean of the N cheapest competitor quotes per row |
| `min_competitor_premium` | Cheapest quote on the row |
| `max_competitor_premium` | Most expensive quote on the row |
| `competitor_premium_std` | Intra-row dispersion |
| `competitor_count` | Number of competitors quoting on the row |
| `market_price_index` | Row target / overall target median |
| `own_to_{target}_ratio` | Own premium / avg_top_N |
| `price_gap_to_{target}` | Own premium − avg_top_N |
| `own_to_min_competitor_ratio` | Own premium / cheapest competitor |
| `rank_own_premium` | Rank of own premium among all competitors |
| `top_{N}_indicator` | 1 if own premium is in cheapest N |
| `segment_market_aggressiveness` | Segment mean target / overall mean target |
| `quote_year/month/quarter/weekofyear/dayofweek` | Temporal features derived from date column |

### 3 · Time-Based Splitting (`splits.py`)

- Strategy: `time` only — rows sorted chronologically, then split by fraction or explicit date cutoffs.
- Default fractions: 65% train / 15% validation / 20% test.
- Validation set is out-of-time relative to training; test set is out-of-time relative to both.
- Standard k-fold is intentionally not supported — it would leak future prices into past predictions.

### 4a · Hyperparameter Tuning (`tuning.py`)

Enabled via `tuning.enabled: true` in config.

- Sampler: Optuna TPE (Tree-structured Parzen Estimator).
- Objective: validation-set metric (no k-fold).
- Supported metrics: `mape`, `d2`, `gini`, `rmsle`, `rmse`.
- Per-backend search spaces:

| Backend | Tuned parameters |
|---|---|
| `sklearn` | `max_leaf_nodes`, `learning_rate`, `l2_regularization` |
| `catboost` | `depth`, `learning_rate`, `l2_leaf_reg` |
| `lightgbm` | `num_leaves`, `learning_rate`, `reg_lambda`, `min_child_samples` |
| `h2o` | Not supported (raises an explicit error) |

- Best params are applied to the config before final model training.
- Writes `tuning_results.json` with per-trial values and best hyperparameters.

### 4b · Model Training (`models.py`)

Four backends, all using Gamma / Tweedie loss appropriate for positive right-skewed prices:

| Backend | Key notes |
|---|---|
| `sklearn` (default) | `HistGradientBoostingRegressor`, Gamma loss, OrdinalEncoder for categoricals |
| `catboost` | Native string categorical handling, Tweedie:variance_power=2 loss, early stopping on validation set |
| `lightgbm` | sklearn ColumnTransformer preprocessing + LGBMRegressor Gamma loss, early stopping |
| `h2o` | H2O GBM Gamma distribution, MOJO export, requires JVM; no Optuna tuning, no ONNX export |

All backends produce:
- `model.joblib` — serialised model bundle (model + preprocessor + metadata)
- `predictions_{train,validation,test}.csv` — actual, predicted, residual, absolute error, APE
- `feature_importance.csv` — permutation importance on validation set
- `lift_table_test.json` — decile lift (mean actual vs mean predicted, pred/actual ratio)
- `metrics.json` — full metric set for all three splits

### 4c · Reference Basket (`basket.py`)

Automatically generated after every training run (all backends except H2O).

**Purpose:** isolate genuine competitor rate changes from risk-mix shift in monthly data.
Because different risk profiles are observed each month (~1,500 policies), raw monthly
averages confound rate changes with mix changes. The reference basket fixes this.

**Construction:**
- Categorical columns: every unique level observed in training data.
- Numeric columns: [p25, p50, p75] of training distribution.
- Full cross-product, sampled to 2,000 profiles maximum.

**Index computation:**
- The trained model is applied to the fixed basket at each calendar month.
- Outputs: monthly mean, p25, p75 of predicted `avg_top_N` across the basket.
- Saves `reference_basket.csv` (the fixed profiles) and `reference_basket_index.csv`.

### 4d · Individual Competitor Models (`models.py`)

Enabled via `individual_competitor_models.enabled: true`.

- Trains one sklearn model per competitor column.
- Filters to rows where that competitor actually quoted (non-null, non-zero price).
- Skips competitors with missing rate above `skip_missing_threshold` (default 40%).
- Saves per-competitor `model_{competitor}.joblib`, metrics, lift table, predictions, and feature importance.
- Enables exact own-price rank computation in the Profile Explorer dashboard tab.

### 5 · Evaluation & QA (`metrics.py`, `reporting.py`)

Metrics computed on all three splits:

| Metric | Role |
|---|---|
| D² (Gamma deviance explained) | Primary goodness-of-fit; analogous to R² but correct for multiplicative models |
| Gini | Actuarial rank-ordering discrimination |
| MAPE | Scale-invariant % pricing error |
| Bias % | Systematic over/under-pricing |
| RMSLE | Log-scale error; natural for multiplicative pricing |
| RMSE | Absolute error scale |
| R² | Retained as legacy reference only |
| MAE | Mean absolute error |

**Deployment gates (blocking):**

| Check | Default threshold |
|---|---|
| D² ≥ d2_min | 0.75 |
| RMSE ≤ rmse_max | 60 |

**Advisory checks (non-blocking):**

| Check | Default threshold |
|---|---|
| MAPE ≤ mape_max | 15% |
| Gini ≥ gini_min | 0.30 |
| Bias% ≤ mean_bias_pct_max | ±5% |

### 6 · Artefact Export

All outputs written to `output/<run_name>/`:

```
model.joblib                        # model bundle
model.cbm / model.onnx              # CatBoost / ONNX exports (backend-dependent)
metrics.json
data_quality.json
split_metadata.json
feature_metadata.json
model_features.json
run_config_resolved.yml
predictions_{train,validation,test}.csv
feature_importance.csv
lift_table_test.json
reference_features.csv              # test set features + predictions (monitoring reference)
reference_basket.csv
reference_basket_index.csv
tuning_results.json                 # only when tuning.enabled: true
market_data.csv                     # all splits combined, for dashboard
qa_checklist.json
business_report.md
model_{competitor}.joblib           # one per competitor, when individual models enabled
metrics_{competitor}.json
predictions_{split}_{competitor}.csv
feature_importance_{competitor}.csv
lift_table_test_{competitor}.json
```

### 7 · Business Report (`reporting.py`)

`business_report.md` contains:
- Model performance summary (all metrics, all splits)
- QA checklist status (pass / fail / advisory)
- Top feature drivers with actuarial interpretation
- Pricing insights (bias direction, market positioning)
- Governance and monitoring notes

### 8 · Monitoring (`monitoring.py`)

Compares current data distribution against `reference_features.csv` from the last training run.
Writes `monitoring_metrics.json` (structured drift and performance data) and `monitoring_report.md`
(human-readable summary with retraining recommendation).

- PSI per feature, flagged above `monitoring.psi_threshold` (default 0.20).
- Performance degradation alerts:

| Alert | Default threshold |
|---|---|
| D² drop | > 0.05 |
| RMSE increase | > 15 |
| Gini drop | > 0.05 |
| MAPE increase | > 3% |

---

## Dashboard (`scripts/dashboard.py`)

7-tab Streamlit application. Run with:

```bash
streamlit run scripts/dashboard.py -- --output-dir output/run_example
```

| Tab | Contents |
|---|---|
| 1 · Market Overview | Competitor price trends, own vs market, ratio, conversion, heatmap, **mix-adjusted index** |
| 2 · Competitor-Level | Price curves, missing rates, aggressiveness heatmap, individual model results |
| 3 · Market Components | Actual vs predicted, rank distribution, price gap, dispersion |
| 4 · Model Performance | D²/Gini/MAPE/Bias%/RMSLE KPIs, scatter, residuals, lift chart, feature importance, **Optuna trial history** |
| 5 · Monitoring | Data quality, performance by sample, prediction drift, segment MAPE |
| 6 · Pricing Action | Segment positioning, opportunity map, MAPE by segment |
| 7 · Profile Explorer | Raw-data drill-down — categorical + numeric filters, observed own vs competitor prices, rank distribution, conversion rate, missing quote rate |

---

## Configuration (`configs/config.yml`)

The active run config. Key sections:

```yaml
project:
  name:        # run identifier, used in output directory naming
  random_seed: # reproducibility

data:
  input_path:              # path to quotes CSV
  date_column:             # chronological split column
  own_premium_column:      # for ratio / rank / conversion features
  conversion_column:       # optional; enables conversion rate charts
  competitor_columns:      # explicit list, or use competitor_column_regex
  target:
    name:   avg_top_3_competitor_premium
    top_n:  3

features:
  top_ns: [3, 5]
  add_competitor_distribution: true
  add_relative_position: true
  add_temporal_features: true
  add_segment_aggressiveness:
    enabled: true
    segment_columns: [region, channel]

split:
  strategy: time
  validation_fraction: 0.15
  test_fraction: 0.20

model:
  backend: sklearn   # sklearn | catboost | lightgbm | h2o

tuning:
  enabled: false
  n_trials: 50
  metric: mape       # mape | d2 | gini | rmsle | rmse
  timeout_seconds:   # optional wall-clock cap

evaluation:
  d2_min: 0.75
  rmse_max: 60
  mape_max: 15.0
  gini_min: 0.30
  mean_bias_pct_max: 5.0

monitoring:
  drift_reference_path: output/run_example/reference_features.csv
  current_data_path:    data/raw/current_competitor_quotes.csv
  psi_threshold: 0.20
  performance_d2_drop_threshold: 0.05
  performance_rmse_increase_threshold: 15
  performance_gini_drop_threshold: 0.05
  performance_mape_increase_threshold: 3.0

individual_competitor_models:
  enabled: false
  skip_missing_threshold: 0.40
```

---

## CLI

```bash
# Train
competitor-pricing-ai train --config configs/config.yml

# Validate config only
competitor-pricing-ai validate-config --config configs/config.yml

# Monitor drift
competitor-pricing-ai monitor --config configs/config.yml

# All commands also available without installing the console script:
# python -m competitor_pricing_ai.cli <subcommand> --config configs/config.yml
```

---

## Governance Guardrails

- Use only legally and ethically obtained competitor observations.
- Do not use competitor intelligence for coordination or anti-competitive conduct.
- Review whether `own_premium` creates target circularity before production deployment
  (the pipeline raises a leakage warning when correlation with the target exceeds 0.85).
- Preserve human review gates before pricing actions are deployed.
- Monitor model drift frequently in dynamic markets — typically every 2–4 weeks.
- The reference basket index is the correct tool for trend monitoring; raw monthly averages
  are misleading when the risk-profile mix of observed quotes shifts month to month.
