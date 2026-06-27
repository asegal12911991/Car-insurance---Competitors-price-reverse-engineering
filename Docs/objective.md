# Objective — Competitor Pricing Intelligence System

## Core Objective

Reverse engineer competitor pricing strategies from observed market quotes and
convert the output into actionable, data-driven pricing decisions for car insurance.

The primary deliverable is a leakage-safe, frozen `market_anchor` input for a separately
governed demand model and optimisation process. This project has no direct Earnix dependency;
it produces historical and batch-scored handoff files plus optional ONNX artifacts.

---

## Core Capabilities

| Capability | Description |
|---|---|
| Data ingestion & validation | Load quote files, detect competitor columns by name or regex, validate panel completeness, coerce types |
| Market feature engineering | Build `avg_top_N`, dispersion, own-to-market ratios, rank, temporal features, and segment aggressiveness |
| Time-based splitting | Chronological train / validation / out-of-time test splits; no future leakage by construction |
| Multi-backend ML modelling | sklearn (default), CatBoost, LightGBM, H2O — all with Gamma / Tweedie loss appropriate for positive right-skewed insurance prices |
| Hyperparameter tuning | Optuna TPE search using the validation set as the objective (no k-fold; standard k-fold would leak future prices into past predictions) |
| Mix-adjusted reference basket | Fixed synthetic portfolio applied to the trained model each month; isolates genuine rate changes from risk-mix shift in monthly observations |
| Individual competitor models | Optional per-competitor sklearn models to predict each competitor's price separately, enabling exact own-price rank computation |
| Business intelligence reporting | `business_report.md` with model performance, QA status, feature drivers, and governance notes |
| Monitoring | PSI-based feature drift detection and performance degradation alerts against configured thresholds |
| Interactive dashboard | 7-tab Streamlit application covering market overview through raw-data profile exploration |
| Demand-model handoff | Rolling-origin historical anchors, frozen relative-price fields, batch scoring, provenance hashes |

## Downstream Acceptance Boundary

This project can test whether the anchor adds out-of-time conversion signal when a conversion
column is present. It cannot approve demand elasticity or an optimised tariff. Downstream
governance must validate calibration, elasticity sign and magnitude, profit impact, and the
stability of recommended prices. The anchor stays fixed while candidate own price changes.

---

## Business Context

**Primary mission:** reverse engineer competitor pricing strategies to enable dynamic,
data-driven pricing decisions.

**Key business targets:** competitor price benchmarking, primarily average top-N competitor
prices aggregated across the cheapest competitors, because top-N is more stable than
individual competitor prices and more directly useful for optimisation.

**Success metrics:**

- Strong actuarial discrimination — Gini coefficient ≥ 0.30
- Low scale-invariant error — MAPE ≤ 15%
- No systematic bias — mean bias % within ±5%
- High Gamma explained deviance — D² ≥ 0.75 (primary deployment gate)
- Bounded absolute error — RMSE < 60 (secondary deployment gate)

---

## Technical Objectives

### Evaluation thresholds

| Metric | Threshold | Type | Rationale |
|---|---|---|---|
| D² (Gamma deviance) | ≥ 0.75 | Blocking gate | Correct goodness-of-fit for positive right-skewed prices |
| RMSE | < 60 | Blocking gate | Absolute error floor |
| MAPE | ≤ 15% | Advisory | Scale-invariant; interpretable as % pricing error |
| Gini | ≥ 0.30 | Advisory | Actuarial rank-ordering discrimination |
| Mean bias % | ≤ ±5% | Advisory | Systematic over/under-pricing tolerance |

> **Note:** R² is retained as a legacy reference metric only. D² (Gamma deviance explained)
> is the correct goodness-of-fit measure for multiplicative pricing models.

### Modelling requirements

- Time-based data splits (no random shuffling)
- Gamma or Tweedie loss function — matches the multiplicative, positive distribution of insurance prices
- Optuna hyperparameter tuning with the out-of-time validation set as objective
- Permutation feature importance on the validation set
- Decile lift table on the out-of-time test set
- MOJO export supported via H2O backend (optional)

### Model backends

| Backend | Install | Notes |
|---|---|---|
| `sklearn` | built-in (default) | HistGradientBoostingRegressor, Gamma loss |
| `catboost` | `pip install -e "[catboost]"` | Native categoricals, Tweedie/Gamma loss |
| `lightgbm` | `pip install -e "[lightgbm]"` | Fastest training, Gamma loss |
| `h2o` | `pip install -e "[h2o]"` | MOJO export for production deployment |

### Mix-adjusted reference basket

A fixed portfolio of synthetic risk profiles (categorical levels × numeric percentiles,
up to 2,000 profiles) is applied to the trained model at each calendar month.
This produces a mix-adjusted price index — trends in the index reflect genuine competitor
rate changes rather than shifting risk profile mix in the observed monthly data.

---

## Dashboard — 7 Tabs

### Tab 1 · Market Overview
*High-level market position at a glance.*

- Average top-3 / top-5 competitor premium over time
- Own premium vs average top-3 competitor premium
- Own-to-market ratio over time by split
- Conversion rate vs own-to-market ratio
- Competitor premium distribution by month (box plot fallback)
- Market price index heatmap by segment × segment
- **Mix-adjusted competitor price index** — model applied to fixed reference basket monthly; orange line vs blue dashed raw observed average

### Tab 2 · Competitor-Level
*Per-competitor behaviour and individual model results.*

- All competitor price curves over time
- Missing quote rate by competitor over time
- Competitor aggressiveness heatmap by segment
- Individual model metrics table (R², RMSE, MAPE, Gini, Bias%)
- Actual vs predicted scatter per competitor
- Derived top-3 from individual models vs aggregated model vs actual

### Tab 3 · Market Components
*Core actuarial pricing view.*

- Actual vs predicted `avg_top_3_competitor_premium` over time (all splits)
- Own price rank distribution over time (stacked %)
- Price gap to market by segment over time
- Competitor premium dispersion (std) over time

### Tab 4 · Model Performance
*Beyond a single accuracy number.*

- Headline KPIs: Gini, MAPE, Bias%, RMSLE, R² (legacy) for train / validation / test
- Actual vs predicted scatter (test set)
- Mean residual over time (all splits)
- Decile lift chart and pred/actual ratio by decile
- Top-15 permutation feature importance
- Residuals by segment over time
- **Optuna tuning results** — trial history chart (value vs trial number with running-best overlay), best hyperparameters table

### Tab 5 · Monitoring
*Operational health and retraining triggers.*

- Data quality KPIs: total rows, rows with enough competitors, date range
- D² and Gini by sample (bar chart)
- MAPE% and Bias% by sample
- Prediction distribution drift over time (mean ± std band, test set)
- Segment-level MAPE deterioration over time

### Tab 6 · Pricing Action
*Closes the loop from model output to business decision.*

- Segment positioning table with recommended action (reduce / hold / increase)
- Opportunity map: price gap vs own rank per segment
- Segments ranked by model MAPE% (highest = most pricing uncertainty)

### Tab 7 · Profile Explorer
*Raw-data drill-down for any market segment.*

- Categorical multiselects + numeric range sliders (independent of sidebar)
- Live row count showing how many quotes match
- Own premium vs each competitor over time (observed monthly averages)
- Own-to-market ratio over time (colour-coded dots)
- Own rank distribution over time (stacked bar)
- Conversion rate over time
- Missing quote rate by competitor
- Monthly summary table (expandable)

---

## Quality Assurance

- Automated QA checklist written to `qa_checklist.json` after every run
- **Blocking gates** (must all pass before deployment is approved):
  - Data contract valid (sufficient rows with enough competitor quotes)
  - Time-based split confirmed
  - D² ≥ d2_min on both validation and test sets
  - RMSE ≤ rmse_max on both validation and test sets
  - Model export file present (serialised model; MOJO if H2O; ONNX if requested)
  - Runtime within the configured budget
- **Advisory checks** (logged and flagged but non-blocking):
  - MAPE ≤ mape_max, Gini ≥ gini_min, Bias% ≤ mean_bias_pct_max on validation set
  - Leakage review: own premium circularity check (high correlation with target triggers warning)
- Monitoring: PSI threshold alerts, performance degradation thresholds
- Governance: human review gate required before pricing actions are deployed

---

## Operational Parameters

- Full pipeline completes within the configured `model.max_runtime_seconds` (default 3,600 s)
- Memory allocation controlled via `model.memory_gb` (default 8 GB)
- All artefacts written to a versioned `output/<run_name>/` directory
- Reproducibility guaranteed via `project.random_seed`
