# Competitor Pricing Analysis System

Configurable AI/ML system for reverse engineering car insurance competitor pricing
strategies and turning market prices into pricing-actionable intelligence.

The production purpose is to create a **frozen competitor market anchor** for a separate
demand model and price-optimisation workflow. This standalone repository does not connect to
an external optimisation platform or deploy pricing decisions. It creates governed CSV/ONNX
handoff artifacts that can be imported and independently approved downstream.

1. Validate and align competitor quote data.
2. Engineer aggregated market features such as average top-N competitor premium.
3. Use strict time-based train/validation/test splits.
4. Train explainable ML models for competitor market components.
5. Export scoring artifacts, including optional H2O MOJO and ONNX exports.
6. Generate QA, feature-importance, and business insight reports.
7. Monitor drift and refresh need.
8. Create rolling-origin historical demand features and batch-scored production anchors.

## Quick Start

Install the package in editable mode:

```powershell
python -m pip install -e .
```

Generate synthetic car-insurance competitor data:

```powershell
python scripts/generate_sample_data.py --output data/raw/competitor_quotes.csv
```

Run the full training workflow:

```powershell
competitor-pricing-ai train --config configs/config.example.yml
```

Or run without installing the console script:

```powershell
python -m competitor_pricing_ai.cli train --config configs/config.example.yml
```

Outputs are written to the configured `project.output_dir`, including:

- `model.joblib` — serialised model bundle (sklearn/CatBoost/LightGBM).
- `metrics.json` — D², R², RMSE, RMSLE, MAE, MAPE, Bias%, Gini for train/validation/test.
- `lift_table_test.json` — decile lift table for the out-of-time test set.
- `feature_importance.csv` — permutation feature importance on the validation set.
- `predictions_validation.csv` and `predictions_test.csv`.
- `reference_basket.csv` and `reference_basket_index.csv` — mix-adjusted market price index.
- `market_data.csv` — all splits combined; consumed by the dashboard.
- `qa_checklist.json` — deployment gate and advisory QA check results.
- `business_report.md` — human-readable run summary with performance and governance notes.
- `run_config_resolved.yml` — fully resolved configuration snapshot.
- `historical_market_features.csv` — prior-period-only anchors for demand development.
- `demand_readiness.json` — local incremental-signal diagnostic when conversion exists.
- `run_manifest.json` — hashes and provenance for every artifact in the run.

## Demand-Model Handoff

The model never uses own premium, conversion, competitor observations, identifiers, or
target-derived fields as predictors. Historical anchors use expanding-window monthly fits;
the initial warm-up period is left unscored rather than backfilled with future information.

```text
market_anchor = predicted comparable competitor premium
relative_price_ratio = candidate_own_premium / market_anchor
log_relative_price = log(candidate_own_premium / market_anchor)
```

`market_anchor` must stay frozen while the optimiser varies candidate own premium.

```powershell
competitor-pricing-ai score `
  --config configs/config.example.yml `
  --input data/raw/current_quotes.csv `
  --output output/current_market_anchor.csv
```

`demand_readiness.json` is only a standalone proxy. Final elasticity, calibration, profit,
and optimisation-stability acceptance belongs in the governed demand/optimisation process.

## Backends

Four model backends are available. Set `model.backend` in the config:

| Backend | Install | Best for | Loss function | Production export |
|---|---|---|---|---|
| `sklearn` | built-in | Default, zero extra deps | Gamma | ONNX via skl2onnx |
| `catboost` | `pip install -e "[catboost]"` | Many categoricals | Tweedie (variance_power=1.9) | Native `.cbm` |
| `lightgbm` | `pip install -e "[lightgbm]"` | Speed, large datasets | Gamma | — |
| `h2o` | `pip install -e "[h2o]"` | MOJO export, enterprise deployment | Gamma | H2O MOJO |

H2O does **not** support CatBoost — they are entirely separate frameworks.

## Hyperparameter Tuning

Optuna TPE search is available for the sklearn, CatBoost, and LightGBM backends.
Enable it in the config:

```yaml
tuning:
  enabled: true
  n_trials: 50
  metric: mape        # mape | d2 | gini | rmsle | rmse
  timeout_seconds: 600
```

The validation set is the tuning objective — no k-fold, which would leak future prices
into past predictions. Best parameters are applied before the final training run. Results
are saved to `tuning_results.json` and visualised in the **Model Performance** dashboard tab.

## Optional ONNX Export

To export a platform-portable `model.onnx` file (sklearn backend only):

```powershell
python -m pip install -e ".[onnx]"
```

Then set:

```yaml
model:
  export_onnx: true
```

The ONNX file can be scored in any ONNX-compatible runtime (Python `onnxruntime`, Java,
C++, or another compatible scoring runtime). If `target_transform: log1p` is configured, the ONNX model
outputs log-scale values — apply `expm1` after scoring to recover the original premium scale.

## Optional H2O MOJO Backend

To train with H2O and export a production MOJO:

```powershell
python -m pip install -e ".[h2o]"
```

Then set:

```yaml
model:
  backend: h2o
  h2o:
    export_mojo: true
```

The H2O backend trains an H2O GBM and saves the MOJO under the run output folder when
the installed H2O version supports MOJO export for that model.

## Individual Competitor Models

Enable per-competitor sklearn models to predict each competitor's price separately.
This allows exact own-price rank computation and is required for the per-competitor
charts in the Competitor-Level dashboard tab.

```yaml
individual_competitor_models:
  enabled: true
  skip_missing_threshold: 0.40   # skip competitors with > 40% missing quotes
```

Per-competitor versions of all standard outputs (model, metrics, predictions, feature importance, lift table) are written alongside the aggregate model outputs.

## Reference Basket Index

After every training run (all backends except H2O), the pipeline generates a
mix-adjusted competitor price index:

1. A fixed reference basket of synthetic risk profiles is constructed from the training
   data (categorical levels × numeric percentiles, up to 2,000 profiles).
2. The trained model is applied to that basket at each calendar month.
3. The resulting monthly mean / p25 / p75 predictions are saved to `reference_basket_index.csv`.

This isolates genuine competitor rate changes from risk-mix shift — raw monthly averages
are misleading when the observed quote mix shifts month to month.

## Configuration

The main customization surface is `configs/config.example.yml`.

Key fields to adapt:

- `data.input_path`: input CSV, Parquet, or Excel file.
- `data.date_column`: quote or observation date used for time splits.
- `data.competitor_columns` or `data.competitor_column_regex`: competitor price fields.
- `data.target.top_n`: the primary average top-N competitor price target.
- `data.categorical_columns` and `data.numeric_columns`: model feature columns.
- `features.top_ns`: top-N market price aggregates to engineer (e.g. `[3, 5]`).
- `split`: time-based fractions or explicit `train_end_date` / `validation_end_date` cutoffs.
- `evaluation`: primary deployment gates (`d2_min`, `rmse_max`) and advisory actuarial thresholds (`mape_max`, `gini_min`, `mean_bias_pct_max`).
- `model.backend`: `sklearn` for local training or `h2o` for MOJO export.
- `individual_competitor_models.enabled`: set to `true` to train one model per competitor column in addition to the aggregated target model.

## Data Expectations

Each row should represent a comparable quote profile or market observation. Typical columns:

- Quote ID and quote date.
- Risk, product, coverage, channel, and geography fields.
- Own premium, if available.
- Competitor premiums, either labeled or unlabeled.
- Optional conversion, renewal, cancellation, or demand outcomes.

Competitor premiums must share coverage, limits, excess/deductible, annualisation, fees,
taxes, and payment basis. Configure those fields under `data.comparability_columns`. The
default `missing_panel_policy: complete` keeps a fixed panel and avoids changing target
composition when a low-priced competitor does not quote. Challenger targets (`min`, `median`,
and `softmin`) can be compared across governed runs by downstream demand value.

The pipeline creates robust aggregated targets, with `avg_top_3_competitor_premium`
as the recommended default. Individual competitor prices can be noisy; aggregated
top-N targets are usually more stable for optimization and monitoring.

## Interactive Dashboard

Install the visualisation dependencies:

```powershell
python -m pip install -e ".[plots]"
```

Launch the dashboard from the project root:

```powershell
streamlit run scripts/dashboard.py
```

Or point it directly at a run directory:

```powershell
streamlit run scripts/dashboard.py -- --output-dir output/run_example
```

The dashboard auto-detects all run directories under `output/` and exposes a selector
in the sidebar. Every chart responds to four interactive filters: date range,
split (train/validation/test), competitor columns, and segment (region, channel,
coverage type, vehicle segment).

The eight tabs map directly to the visual spec in `Docs/objective.md`:

1. **Market Overview** — competitor premium trends, own-to-market ratio, conversion elasticity, market price index heatmap, mix-adjusted competitor price index
2. **Competitor-Level** — individual price curves, missing quote rates, aggressiveness heatmap, per-competitor actual vs predicted (requires `individual_competitor_models.enabled: true`)
3. **Market Components** — actual vs predicted over time, own rank distribution, price gap by segment, dispersion trend
4. **Model Performance** — KPI tiles (D², Gini, MAPE, Bias%, RMSLE), actual vs predicted scatter, residuals over time, decile lift chart, feature importance, Optuna tuning history
5. **Monitoring** — data quality KPIs, D²/Gini/MAPE/Bias% by sample, prediction distribution drift, segment-level MAPE deterioration
6. **Pricing Action** — segment positioning table with recommended action, opportunity map, segments ranked by pricing uncertainty
7. **Profile Explorer** — raw-data drill-down with categorical multiselects and numeric range sliders; observed own vs competitor prices, rank distribution, conversion rate, and missing quote rate for any market segment
8. **Demand Handoff** — rolling-origin anchor coverage, incremental demand-signal diagnostics, relative-price views, artifact integrity, QA gates, and filtered handoff export

## Monitoring

After a training run, use the drift monitor against a new data extract:

```powershell
competitor-pricing-ai monitor --config configs/config.example.yml
```

The monitor computes feature drift using PSI diagnostics and writes `monitoring_report.md`
plus `monitoring_metrics.json`. Retraining is recommended when D² drop, RMSE increase,
Gini drop, or MAPE increase exceeds the configured thresholds.
