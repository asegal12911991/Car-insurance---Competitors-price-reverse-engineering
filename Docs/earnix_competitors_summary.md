---
output:
  pdf_document: default
  html_document: default
---
# Summary of `EarnixCompetitors.pptx`

**Purpose of this summary:** This presentation explains how competitor price information can be collected, transformed, modeled, monitored, and incorporated into insurance pricing and optimization workflows. The notes below are adapted for a project to create an **advanced competitor pricing intelligence AI** focused on reverse-engineering competitor pricing strategies using machine learning.

**Extraction note:** The deck appears to be largely image-based, so the source content was read using OCR from slide renders. Some UI screenshots and tiny table values may contain OCR noise; the strategic and modeling concepts are clear enough for project design.

---

## 1. Executive Summary

The presentation’s core message is that competitor pricing signals should not be treated as simple static reference variables. They are market components that must be engineered, modeled, forecasted, validated, monitored, and then incorporated into demand models and optimization workflows.

For a competitor pricing intelligence AI, the most important takeaway is that the system should reverse-engineer competitor behavior through a structured modeling pipeline:

1. **Collect competitor data** from aggregators, scraping, or internal market observations.
2. **Normalize and align the data** with the insurer’s demand model population, time period, and feature definitions.
3. **Engineer market components** such as average top-N competitor price, rank, relative price ratio, number of competitors, and market-position indicators.
4. **Select modeling targets** that are stable enough, price-actionable, and forecastable.
5. **Train machine-learning models** to estimate or forecast competitor price components.
6. **Feed modeled competitor signals into demand, cancellation, renewal, and optimization models.**
7. **Monitor drift and update frequently**, potentially every 2-4 weeks in dynamic markets.

The deck strongly supports using aggregated competitor targets - especially **average price of the top three/five cheapest competitors** - rather than modeling individual competitor prices directly, because aggregated targets are typically more stable and less noisy.

---

## 2. Slide-by-Slide Summary

### Slide 1 - Agenda

The deck covers:

- Competitor data sources, requirements, and considerations.
- Approaches to building competitor models.
- Expected effects of competitor models on pricing and demand models.
- Frequency of model updates.
- Demo screenshots showing the workflow in Earnix.

### Slide 2 - What competitor data may look like

Competitor price data may come from:

- Purchased aggregator feeds.
- Web scraping.
- Structured market scans.

The data can be structured as:

- One column per competitor, such as `Comp_1`, `Comp_2`, `Comp_3`.
- Aggregated columns, such as average price of the five lowest competitors.
- Labeled or unlabeled competitor premiums.

The deck recommends aggregation, such as **average price of the three cheapest competitors**, to reduce error and variance in the modeling target.

### Slide 3 - Consistency with the demand model

Competitor data must be consistent with the demand model along three dimensions:

- **Column values:** Numeric and nominal features should use the same units and definitions.
- **Missing values:** Missing-value handling should match the demand model’s treatment.
- **Time period and population:** The competitor dataset should align with the demand model’s observation period and profile distribution.

For the AI project, this implies that competitor intelligence should have a strict data-contract layer: every competitor quote observation must be mapped to the same risk, product, geography, and time definitions used by internal pricing models.

### Slide 4 - Analyze more than one market component

The presentation recommends analyzing multiple market features, not only a single competitor price variable. Relevant market components include:

- Rank or market position.
- Indicator for being among the cheapest competitors.
- Average price of the three cheapest competitors, excluding own price.
- Ratio between own premium and the lowest or average competitor premium.
- Number of competitors in the market sample.

**Checks:**
- Distribution of competitor prices.
- Correlation between own premium and market components.
- Conversion rate versus market components.

For a reverse-engineering AI, this is a crucial design principle: competitor strategy should be inferred from a **feature set of market behavior**, not from one target variable alone.

### Slides 5-6 - Choosing which target to model

The deck highlights three criteria for selecting modeling targets:

1. **Stability over time:** Stable targets are preferable.
2. **Ability to predict price-change or elasticity effects:** Targets should be useful for pricing and optimization decisions.
3. **Ability to predict future trends:** Targets must remain forecastable from deployment until the next refresh.

The presentation warns that rank and binary market-position indicators may be less desirable for optimization because they are discrete, less smooth, and may not continuously influence elasticity.

Recommended targets for this project:

- Primary: `Average Top 3 competitor premium`, `Average Top 5 competitor premium`, or `market price index`.
- Secondary: price ratio, competitor dispersion, cheapest competitor premium, rank, market-position indicators.
- Avoid relying on rank alone as the core modeled target.

### Slide 7 - Using GenAI and agentic coding

The deck suggests using GenAI-based agentic coding to accelerate:

- Data analysis.
- Model training.
- Model monitoring.

It also references production-ready model exports, including:

- H2O MOJO files.
- ONNX models.

For the competitor pricing intelligence AI, this supports an agentic ML workflow where the AI can generate analysis scripts, train models, test data splits, produce monitoring dashboards, and export models for deployment.

### Slide 8 - Prompting guidance for an AI modeling agent

The deck provides a prompt-design structure for AI-assisted competitor modeling. It includes:

- **Core objective:** Build an advanced AI system that reverse-engineers competitor pricing strategies using machine learning.
- **Technical objectives:** Deliver models with target accuracy (R² >= 0.75, RMSE, MAPE <= 15%, Gini >= 0.30, mean bias within ±5%), using time-based splits, H2O, ML methods, MOJO exports, feature-importance analysis, and decile lift tables.
- **Business guardrails:** Protect data privacy, ensure regulatory compliance, respect price bounds, and follow ethical competitive-intelligence practices.
- **Operational efficiency:** Complete the full analysis cycle within a defined runtime and memory budget, with automated QA checkpoints.
- **Quality assurance:** Include validation protocols, drift detection, error handling, and monitoring procedures.

This slide is especially relevant to the project: it is effectively a system prompt blueprint for an autonomous competitor-pricing ML agent.

### Slide 9 - Modeling approaches

Competitor market components can be modeled using:

- Models built inside Earnix Price-It.
- GBM or appropriate regression models such as GLM/GAM.
- H2O models built in R or Python.
- DataRobot models.
- Models built in other platforms and imported as ONNX.

The implication is that the architecture should be model-framework-agnostic, while preserving deployment compatibility.

### Slide 10 - Incorporating competitor models into pricing

Competitor signals can be used directly or transformed before entering pricing models. For example:

- Model the average of the three lowest competitor prices.
- Insert it into the demand model as a ratio between own price and competitor prices.
- Interact the market feature with channel, product, or geography.
- Use competitor models in behavioral models such as cancellation-rate models.
- Use competitor model output in optimization, for example to set individual constraints.

For the AI project, the output should not only be a competitor price forecast. It should produce **pricing-actionable features** that can be consumed by demand, elasticity, retention, cancellation, and optimization systems.

### Slide 11 - Expected effects on the demand model

Adding competitor-model features is expected to affect the demand model by:

- Changing model coefficients, especially coefficients capturing price effect.
- Increasing estimated elasticity because the model now accounts for competition.
- Changing out-of-time predictions due to forecasted competitor behavior.
- Improving goodness-of-fit statistics.

This means the project should evaluate competitor intelligence not only by target-model accuracy, but also by downstream uplift in demand-model fit, elasticity credibility, and optimization performance.

### Slide 12 - Model update frequency

Update frequency depends on:

- Market competitiveness.
- Predictive ability over time.
- External market changes.
- Regulatory changes.
- Economic changes.

The slide suggests that updates every **2-4 weeks** may be necessary in some cases. It also notes that automation can be helpful by cleansing, importing, and updating competitor data.

For the AI project, update frequency should be a monitored quantity, not a fixed assumption. The system should estimate degradation curves and trigger retraining when drift or forecast decay exceeds thresholds.

### Slides 13-17 - Demo: competitor data and exploratory analysis

The screenshots show an Earnix project containing competitor information and engineered fields such as:

- Average driver age.
- Average top-three competitor price.
- Average vehicle age.
- Individual competitor prices.
- Driver count.
- First-position indicator.
- Own premium.
- Rank.
- Ratio of own premium to competitor premium.

The demo includes charts such as:

- Average top-three competitor price over time.
- Average top-three competitor price versus own premium.
- Conversion rate versus the own-price-to-competitor-price ratio.

These screenshots reinforce the importance of exploratory analysis before model training. The AI system should automatically generate these diagnostics and use them to detect market regimes, competitor cycles, and own-price positioning.

### Slide 18 - Out-of-time test sample

The slide shows code for splitting data into train, test, and validation by time period. The test set is explicitly treated as an **out-of-time sample**.

This is essential for competitor strategy modeling. Random splits would overstate performance if competitor behavior is temporally correlated. The AI system should use rolling-origin, forward-chaining, or strict time-based validation.

### Slide 19 - Model training and feature importance

The demo shows a Gradient Boosting Machine model for `Avg Top 3`, apparently with a Gamma distribution and 3,000 trees. The dependent column is `Avg Top 3`. Feature importance is displayed, with the strongest feature appearing to be an average-premium-related variable, followed by driver age and policy attributes.

Project implication:

- Use GBM-style models as a strong baseline.
- Require feature-importance and explainability outputs.
- Check whether own premium or internal price variables create leakage, circularity, or strategic feedback loops.

### Slides 20-21 - Demand model comparison with and without competitor information

The demo compares demand model versions without and with competitor information. It shows changes in coefficients, statistics, and elasticity-related outputs.

The presentation’s later conclusion states that after including competitor information:

- The significance of the premium decreased.
- Model fit improved.
- Elasticity increased.

Interpretation: Some apparent price effect in the original model was likely capturing market context. Once competitor conditions are represented directly, the own-premium effect becomes more economically meaningful and the model better captures competitive sensitivity.

### Slide 22 - Summary of model impact

The key stated results are:

- Premium significance decreased.
- Model fit improved.
- Elasticity increased.

For the AI project, this should become a formal evaluation criterion: competitor-intelligence features should improve behavioral model quality while producing plausible elasticity changes.

### Slide 23 - Optimization impact

The final slide shows optimized-frontier analysis and compares demand against total written margin. The deck’s point is that adding competitor information affects optimization results, not just predictive model fit.

For the AI project, the final success metric should be business decision quality: better frontier analysis, more robust price recommendations, improved volume-margin trade-offs, and more realistic constraints under competitive market conditions.

---

## 3. Main Lessons for an Advanced Competitor Pricing Intelligence AI

### 3.1 Competitor data should be modeled as market behavior, not just scraped prices

A strong competitor-pricing intelligence system should infer latent competitor strategy from observed price surfaces. Useful targets include:

- Market price index.
- Average top-N competitor price.
- Cheapest competitor premium.
- Price rank.
- Relative price ratio.
- Competitor price dispersion.
- Segment-level market aggressiveness.
- Time-varying competitor movement.

### 3.2 Aggregated competitor targets are often superior to individual competitor targets

The deck recommends average top-three or top-five competitor prices because these reduce noise and target variance. This is especially relevant when:

- Competitor labels are unstable or unavailable.
- Web-scraped prices are incomplete.
- Quote panels vary by profile.
- Individual competitor behavior is highly idiosyncratic.

### 3.3 Time-aware validation is mandatory

Because competitor pricing is dynamic, the system should not rely on random validation splits. Recommended validation methods:

- Out-of-time holdout.
- Rolling-origin backtesting.
- Forward-chaining cross-validation.
- Stability by month, channel, product, geography, and competitor cluster.

### 3.4 Downstream impact matters more than standalone accuracy

A competitor model with high R² is not sufficient. Actuarially, the more meaningful tests are:

- **MAPE** (scale-invariant %-error): a £20 error on a £100 policy is far more damaging than on a £2,000 policy; RMSE masks this.
- **Gini coefficient**: measures how well the model rank-orders risks by price level — the standard actuarial discrimination test.
- **Mean bias %**: detects systematic over- or under-pricing across all risks; RMSE hides bias by squaring errors.
- **RMSLE**: penalises proportional errors equally, natural for the multiplicative GLM-style pricing structure.
- **Decile lift table**: shows whether pred/actual ratios are well-calibrated across the full pricing distribution, not just on average.

Beyond standalone accuracy, it should also improve:

- Demand-model fit.
- Elasticity plausibility.
- Conversion prediction.
- Cancellation or renewal prediction.
- Optimization frontier quality.
- Price recommendation robustness.

### 3.5 Monitoring should drive refresh frequency

The presentation suggests possible updates every 2-4 weeks, but the AI system should make refresh frequency data-driven using:

- Prediction drift.
- Feature drift.
- Competitor price-index drift.
- Degradation in out-of-time performance.
- Changes in rank/position stability.
- External triggers such as regulation or macroeconomic shocks.

---

## 4. Proposed AI System Blueprint Based on the Presentation

### 4.1 Inputs

- Own quote data and policy attributes.
- Competitor quote observations from aggregators or scraping.
- Product, channel, geography, and time variables.
- Demand/conversion outcomes.
- Renewal, cancellation, and retention outcomes where available.
- External market, regulatory, and macroeconomic signals.

### 4.2 Data quality and alignment layer

The AI should validate:

- Feature unit consistency.
- Missing-value rules.
- Quote-date alignment.
- Comparable profile distribution.
- Product and coverage comparability.
- Competitor panel completeness.
- Outlier and scraping-error detection.

### 4.3 Feature engineering layer

Core engineered features:

- `avg_top_3_competitor_premium`
- `avg_top_5_competitor_premium`
- `min_competitor_premium`
- `own_premium / avg_top_3_competitor_premium`
- `own_premium / min_competitor_premium`
- `rank`
- `top_3_indicator`
- `competitor_count`
- `competitor_premium_std`
- `market_price_index`
- `price_gap_to_market`
- `segment_market_aggressiveness`

Recommended interactions:

- Market ratio x channel.
- Market ratio x product.
- Market ratio x geography.
- Market ratio x customer tenure or lifecycle stage.
- Market ratio x regulatory region.

### 4.4 Modeling layer

Baseline model families:

- GBM / XGBoost / LightGBM / H2O GBM.
- GLM/GAM for transparent benchmark models.
- Quantile regression for uncertainty bands.
- Hierarchical models for geography/product segmentation.
- Time-series or panel models for competitor movement.
- Multi-task models for related market components.

Deployment formats:

- MOJO if using H2O.
- ONNX for platform portability.
- Native scoring services for Python-based models.

### 4.5 Explainability layer

The AI should automatically report:

- Global feature importance.
- Segment-level feature importance.
- Partial dependence or accumulated local effects.
- Monotonicity and reasonableness checks.
- Leakage diagnostics, especially where own premium is used to predict competitor prices.
- Drift and stability diagnostics.

### 4.6 Integration into pricing and optimization

Competitor intelligence outputs should feed:

- Demand models.
- Conversion models.
- Cancellation and retention models.
- Price optimization constraints.
- Frontier analysis.
- Scenario simulation.
- Segment-specific pricing strategy recommendations.

Example usage:

```text
Modeled target: Average Top 3 competitor price
Pricing feature: Own premium / modeled Average Top 3 competitor price
Optimization use: constrain price increases where own premium becomes materially above market
Monitoring use: trigger retraining if top-three forecast error deteriorates over recent weeks
```

---

## 5. Recommended Project Requirements Derived from the Deck

### Functional requirements

1. Ingest competitor data from aggregator files, scraping pipelines, or internal quote studies.
2. Standardize competitor data to match internal pricing-model definitions.
3. Create aggregated competitor-market features.
4. Train ML models for competitor market components using time-based splits.
5. Produce model explainability and QA reports.
6. Export production-ready models.
7. Monitor performance, drift, and refresh need.
8. Integrate outputs into demand and optimization workflows.

### Non-functional requirements

1. Strong auditability and reproducibility.
2. Explicit ethical competitive-intelligence guardrails.
3. Data privacy controls.
4. Regulatory compliance checks.
5. Runtime and memory limits for automated analysis.
6. Human review gates before pricing deployment.
7. Robust error handling for missing or degraded competitor feeds.

### Key success metrics

- Out-of-time R² / RMSE / RMSLE / MAE / MAPE for competitor target forecasts.
- Gini coefficient — actuarial discrimination of low- vs high-price risks.
- Mean bias % — no systematic over- or under-pricing beyond ±5%.
- Decile lift table — pred/actual ratio near 1.0 within each pricing band.
- Stability of feature importance over time.
- Demand model lift after adding competitor features.
- Improvement in conversion or retention prediction.
- Change in elasticity estimates and economic plausibility.
- Optimization frontier improvement.
- Drift-adjusted refresh frequency (R2 drop, RMSE/MAPE increase, Gini drop).

---

## 6. Risks and Watchouts

1. **Data leakage:** Own premium can be highly correlated with market prices but may create circularity if used naively.
2. **Scraping error:** Small collection errors can distort rank and cheapest-price features.
3. **Competitor panel instability:** Competitor availability may vary across profiles and time.
4. **Rank instability:** Rank can change abruptly with small price changes, making it less suitable as a primary optimization target.
5. **Regulatory sensitivity:** Competitor intelligence must remain compliant and avoid inappropriate coordination.
6. **Overfitting to current market regimes:** Market behavior can shift after regulation, inflation, distribution changes, or competitor repricing.
7. **Optimization misuse:** A good competitor forecast does not automatically imply a good price decision; downstream business constraints remain essential.

---

## 7. Bottom Line

The deck provides a practical framework for competitor-price modeling in insurance pricing. For the proposed advanced competitor pricing intelligence AI, the strongest design direction is:

> Build a time-aware, explainable, production-ready ML system that forecasts aggregated competitor market components, transforms them into pricing-actionable relative-position features, validates their downstream impact on demand and elasticity, and continuously monitors when the market has moved enough to require retraining.

The presentation’s most useful project-specific principle is that reverse-engineering competitor pricing strategy should be treated as a full pricing intelligence lifecycle: **data alignment -> market feature engineering -> target selection -> time-aware ML -> explainability -> demand/optimization integration -> monitoring and refresh.**
