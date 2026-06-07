# Competitor Pricing Analysis System

Configurable AI/ML system for reverse engineering car insurance competitor pricing
strategies and turning market prices into pricing-actionable intelligence.

The system is designed around the Earnix-style lifecycle in `Docs/earnix_competitors_summary.md`:

1. Validate and align competitor quote data.
2. Engineer aggregated market features such as average top-N competitor premium.
3. Use strict time-based train/validation/test splits.
4. Train explainable ML models for competitor market components.
5. Export scoring artifacts, including optional H2O MOJO exports.
6. Generate QA, feature-importance, and business insight reports.
7. Monitor drift and refresh need.

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

- `model.joblib` for the default sklearn backend.
- `metrics.json` with out-of-time validation and test metrics (R², RMSE, RMSLE, MAE, MAPE, Bias%, Gini).
- `lift_table_test.json` — decile lift table for the out-of-time test set.
- `feature_importance.csv`.
- `predictions_validation.csv` and `predictions_test.csv`.
- `qa_checklist.json`.
- `business_report.md`.
- `run_config_resolved.yml`.

## Backends

Four model backends are available. Set `model.backend` in the config:

| Backend | Install | Best for | Gamma loss | ONNX export | Notes |
|---|---|---|---|---|---|
| `sklearn` | built-in | Default, zero deps | `loss: gamma` | via skl2onnx | `HistGradientBoostingRegressor` |
| `catboost` | `pip install -e "[catboost]"` | Many categoricals, production | `Tweedie:variance_power=2` | native | No OrdinalEncoder needed; native `.cbm` save |
| `lightgbm` | `pip install -e "[lightgbm]"` | Speed, large datasets | `objective: gamma` | — | Fastest training |
| `h2o` | `pip install -e "[h2o]"` | MOJO export, enterprise | `distribution: gamma` | — | Produces MOJO for Earnix/H2O deployment |

H2O does **not** support CatBoost — they are entirely separate frameworks.

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

The ONNX file can be scored in any ONNX-compatible runtime (Python `onnxruntime`, Java, C++, Earnix ONNX import). If `target_transform: log1p` is configured, the ONNX model outputs log-scale values — apply `expm1` after scoring to recover the original premium scale.

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

The H2O backend trains an H2O GBM by default and saves the MOJO under the run output
folder when the installed H2O version supports MOJO export for that model.

## Configuration

The main customization surface is `configs/config.example.yml`.

Key fields to adapt:

- `data.input_path`: input CSV, Parquet, or Excel file.
- `data.date_column`: quote or observation date used for time splits.
- `data.competitor_columns` or `data.competitor_column_regex`: competitor price fields.
- `data.target.top_n`: the primary average top-N competitor price target.
- `data.categorical_columns` and `data.numeric_columns`: model feature columns.
- `features.top_ns`: top-N market price aggregates to engineer.
- `split`: time-based validation and test fractions.
- `evaluation`: primary deployment gates (`r2_min`, `rmse_max`) and advisory actuarial thresholds (`mape_max`, `gini_min`, `mean_bias_pct_max`).
- `model.backend`: `sklearn` for local training or `h2o` for MOJO export.
- `individual_competitor_models.enabled`: set to `true` to train one model per competitor column in addition to the aggregated target model.

## Data Expectations

Each row should represent a comparable quote profile or market observation. Typical
columns:

- Quote ID and quote date.
- Risk, product, coverage, channel, and geography fields.
- Own premium, if available.
- Competitor premiums, either labeled or unlabeled.
- Optional conversion, renewal, cancellation, or demand outcomes.

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

The dashboard auto-detects all run directories under `output/` and exposes a selector in the sidebar. Every chart responds to four interactive filters:

| Filter | Controls |
|---|---|
| **Date range** | Slider — narrows all time-series charts |
| **Splits** | train / validation / test — shown/hidden per chart |
| **Competitors** | Which competitor columns appear in Section 2 |
| **Segments** | region, channel, coverage type, vehicle segment |

The six tabs map directly to the visual spec in `Docs/objective.md`:

1. **Market Overview** — competitor premium trends, own-to-market ratio, conversion elasticity, market price index heatmap
2. **Competitor-Level** — individual price curves, missing quote rates, aggressiveness heatmap, per-competitor actual vs predicted (requires `individual_competitor_models.enabled: true`)
3. **Market Components** — actual vs predicted over time, own rank distribution, price gap by segment, dispersion trend
4. **Model Performance** — KPI tiles, actual vs predicted scatter, residuals over time, decile lift chart, feature importance
5. **Monitoring** — R²/Gini/MAPE/Bias% by sample, prediction distribution drift, segment-level MAPE deterioration
6. **Pricing Action** — segment positioning table with recommended action, opportunity map, segments ranked by pricing uncertainty

The dashboard reads from the run output directory and requires no additional pipeline changes — `market_data.csv` is written automatically by every training run.

## Monitoring

After a training run, use the drift monitor against a new data extract:

```powershell
competitor-pricing-ai monitor --config configs/config.example.yml
```

The monitor computes feature drift using PSI-like diagnostics and writes
`monitoring_report.md` plus `monitoring_metrics.json`. Retraining is triggered
when R2 drop, RMSE increase, Gini drop, or MAPE increase exceeds the configured
thresholds.
