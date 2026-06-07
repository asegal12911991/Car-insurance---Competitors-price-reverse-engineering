"""Interactive Competitor Pricing Intelligence Dashboard.

Launch from the project root:
    streamlit run scripts/dashboard.py

Pass a specific run directory on the command line:
    streamlit run scripts/dashboard.py -- --output-dir output/run_example
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml


# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Competitor Pricing Intelligence",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded",
)

SPLIT_COLOURS = {"train": "#636EFA", "validation": "#EF553B", "test": "#00CC96"}
SEGMENT_COLUMNS = ["region", "channel", "coverage_type", "vehicle_segment"]  # fallback / priority order


def detect_segment_columns(data: dict) -> list[str]:
    """Return filterable categorical columns from market_data, ordered by priority."""
    market = data.get("market")
    if market is None:
        return SEGMENT_COLUMNS

    # Columns to exclude from segment detection
    exclude = set()
    exclude.update(data.get("run_config", {}).get("data", {}).get("id_columns", []))
    exclude.add(data.get("date_column", ""))
    exclude.add(data.get("own_premium_column", ""))
    exclude.add(data.get("conversion_column") or "")
    exclude.update(data.get("competitor_columns", []))
    exclude.update(data.get("feature_metadata", {}).get("market_component_columns", []))
    exclude.update(data.get("feature_metadata", {}).get("temporal_columns", []))
    exclude.add(data.get("target_column", ""))
    exclude.add("_split")

    candidates = []
    for col in market.columns:
        if col in exclude:
            continue
        n_unique = market[col].nunique(dropna=True)
        if market[col].dtype == object or market[col].dtype.name == "category" or (2 <= n_unique <= 30):
            candidates.append((col, n_unique))

    # Sort: prefer SEGMENT_COLUMNS order first, then by cardinality
    priority = {c: i for i, c in enumerate(SEGMENT_COLUMNS)}
    candidates.sort(key=lambda x: (priority.get(x[0], len(SEGMENT_COLUMNS)), x[1]))
    return [c for c, _ in candidates]

# ──────────────────────────────────────────────────────────────────────
# Cached data loaders
# ──────────────────────────────────────────────────────────────────────

@st.cache_data
def _read_json(path: str) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def _read_yml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@st.cache_data
def load_run(output_dir_str: str) -> dict:
    """Load every available artefact from a training run output directory."""
    output_dir = Path(output_dir_str)
    d: dict = {"dir": output_dir}

    def jload(fname: str, key: str) -> None:
        p = output_dir / fname
        if p.exists():
            d[key] = _read_json(str(p))

    def cload(fname: str, key: str) -> None:
        p = output_dir / fname
        if p.exists():
            d[key] = _read_csv(str(p))

    jload("metrics.json", "metrics")
    jload("data_quality.json", "data_quality")
    jload("feature_metadata.json", "feature_metadata")
    jload("lift_table_test.json", "lift_table")
    jload("tuning_results.json", "tuning_results")
    jload("model_features.json", "model_features")
    cload("reference_basket_index.csv", "basket_index")

    yml_p = output_dir / "run_config_resolved.yml"
    if yml_p.exists():
        d["run_config"] = _read_yml(str(yml_p))

    cload("market_data.csv", "market")
    cload("predictions_train.csv", "pred_train")
    cload("predictions_validation.csv", "pred_val")
    cload("predictions_test.csv", "pred_test")
    cload("reference_features.csv", "reference")
    cload("feature_importance.csv", "feature_importance")

    # Resolve key metadata
    d["competitor_columns"] = (
        d.get("data_quality", {}).get("competitor_columns")
        or d.get("feature_metadata", {}).get("competitor_columns")
        or []
    )
    d["target_column"] = d.get("feature_metadata", {}).get("target_column", "")
    d["date_column"] = (d.get("run_config") or {}).get("data", {}).get("date_column", "quote_date")
    d["own_premium_column"] = (d.get("run_config") or {}).get("data", {}).get("own_premium_column", "own_premium")
    d["conversion_column"] = (d.get("run_config") or {}).get("data", {}).get("conversion_column")

    # Individual competitor model artefacts
    individual: dict = {}
    for comp_col in d["competitor_columns"]:
        safe = comp_col.replace(" ", "_")
        entry: dict = {}
        for fname, key in [
            (f"metrics_{safe}.json", "metrics"),
            (f"lift_table_test_{safe}.json", "lift_table"),
        ]:
            p = output_dir / fname
            if p.exists():
                entry[key] = _read_json(str(p))
        for fname, key in [
            (f"predictions_test_{safe}.csv", "predictions_test"),
            (f"feature_importance_{safe}.csv", "feature_importance"),
        ]:
            p = output_dir / fname
            if p.exists():
                entry[key] = _read_csv(str(p))
        if entry:
            individual[comp_col] = entry
    d["individual"] = individual
    return d


# ──────────────────────────────────────────────────────────────────────
# Filter helpers
# ──────────────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame | None,
    date_col: str,
    date_from,
    date_to,
    segment_filters: dict[str, list],
    splits: list[str] | None = None,
) -> pd.DataFrame | None:
    if df is None or df.empty:
        return df
    if date_col and date_col in df.columns:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        df = df[(dates >= pd.Timestamp(date_from)) & (dates <= pd.Timestamp(date_to))]
    for col, vals in segment_filters.items():
        if col in df.columns and vals:
            df = df[df[col].isin(vals)]
    if splits is not None and "_split" in df.columns:
        df = df[df["_split"].isin(splits)]
    return df if not df.empty else None


def add_month(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col in df.columns:
        df = df.copy()
        df["_month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
        df = df[df["_month"].notna()]
    return df


# ──────────────────────────────────────────────────────────────────────
# Tiny chart helpers
# ──────────────────────────────────────────────────────────────────────

def no_data(msg: str = "No data available for this selection.") -> None:
    st.info(msg, icon="ℹ️")


def diagonal_line(fig: go.Figure, series: pd.Series) -> go.Figure:
    mn, mx = float(series.min()), float(series.max())
    fig.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                  line=dict(dash="dash", color="grey", width=1))
    return fig


def hline_zero(fig: go.Figure) -> go.Figure:
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    return fig


def hline_at_market(fig: go.Figure) -> go.Figure:
    """Horizontal reference at y=1.0 for ratio time-series charts."""
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", line_width=1)
    fig.add_annotation(
        text="at market", xref="paper", yref="y",
        x=0.01, y=1.0, showarrow=False,
        font=dict(color="grey", size=11),
        bgcolor="rgba(255,255,255,0.6)",
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📊 Pricing Intelligence")

    # Auto-detect run directories
    candidates: list[str] = []
    for pattern in ["*/market_data.csv", "*/metrics.json"]:
        for p in sorted(Path("output").glob(pattern), reverse=True) if Path("output").exists() else []:
            s = str(p.parent)
            if s not in candidates:
                candidates.append(s)

    # Allow override via CLI: streamlit run dashboard.py -- --output-dir PATH
    cli_default = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output-dir" and i + 1 < len(sys.argv):
            cli_default = sys.argv[i + 1]
            break

    if candidates:
        default_idx = 0
        if cli_default and cli_default in candidates:
            default_idx = candidates.index(cli_default)
        run_dir_str = st.selectbox("Run directory", candidates, index=default_idx)
    else:
        run_dir_str = st.text_input("Run output directory",
                                    value=cli_default or "output/run_example")

    output_dir = Path(run_dir_str)
    if not output_dir.exists():
        st.error(f"Directory not found: {output_dir}")
        st.stop()

    if st.button("🔄 Reload data"):
        st.cache_data.clear()

    with st.spinner("Loading run data…"):
        data = load_run(str(output_dir))

    if not data.get("metrics") and data.get("market") is None:
        st.error("No recognisable run data found in this directory.")
        st.stop()

    dq = data.get("data_quality", {})
    st.success(f"**{output_dir.name}**")
    if dq:
        st.caption(f"{dq.get('rows', '?'):,} rows · {dq.get('date_min', '?')} → {dq.get('date_max', '?')}")

    st.divider()

    # ── Date range ──
    market_raw = data.get("market")
    date_col = data["date_column"]
    if market_raw is not None and date_col in market_raw.columns:
        dates_all = pd.to_datetime(market_raw[date_col], errors="coerce").dropna()
        min_d, max_d = dates_all.min().date(), dates_all.max().date()
    else:
        try:
            min_d = pd.Timestamp(dq.get("date_min", "2024-01-01")).date()
            max_d = pd.Timestamp(dq.get("date_max", "2025-12-31")).date()
        except Exception:
            from datetime import date as _date
            min_d, max_d = _date(2024, 1, 1), _date(2025, 12, 31)

    dr = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        date_from, date_to = dr
    elif isinstance(dr, (list, tuple)) and len(dr) == 1:
        date_from = date_to = dr[0]
    else:
        date_from, date_to = min_d, max_d

    # ── Split selector ──
    splits_in_data = (
        sorted(market_raw["_split"].dropna().unique().tolist())
        if market_raw is not None and "_split" in market_raw.columns
        else ["train", "validation", "test"]
    )
    selected_splits = st.multiselect("Splits", splits_in_data, default=splits_in_data)

    # ── Competitor selector ──
    comp_cols: list[str] = data["competitor_columns"]
    selected_comps = st.multiselect("Competitors", comp_cols, default=comp_cols) if comp_cols else []

    # ── Segment filters (auto-detected from market_data.csv) ──
    st.divider()
    st.subheader("Segment filters")
    detected_segments = detect_segment_columns(data)
    segment_filters: dict[str, list] = {}
    if market_raw is not None:
        for seg in detected_segments:
            if seg in market_raw.columns:
                vals = sorted(market_raw[seg].dropna().unique().tolist())
                sel = st.multiselect(seg.replace("_", " ").title(), vals, default=vals)
                segment_filters[seg] = sel

# ──────────────────────────────────────────────────────────────────────
# Filtered DataFrames (common across all tabs)
# ──────────────────────────────────────────────────────────────────────

_fkw = dict(date_col=date_col, date_from=date_from, date_to=date_to,
            segment_filters=segment_filters)

market_f = apply_filters(market_raw, splits=selected_splits, **_fkw)

# Predictions don't have _split; gate by which CSV to include
pred_train_f  = apply_filters(data.get("pred_train"),  splits=None, **_fkw) if "train"      in selected_splits else None
pred_val_f    = apply_filters(data.get("pred_val"),    splits=None, **_fkw) if "validation" in selected_splits else None
pred_test_f   = apply_filters(data.get("pred_test"),   splits=None, **_fkw) if "test"       in selected_splits else None
reference_f   = apply_filters(data.get("reference"),   splits=None, **_fkw)

target_col    = data["target_column"]
own_col       = data["own_premium_column"]
conv_col      = data["conversion_column"]

# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1 · Market Overview",
    "2 · Competitor-Level",
    "3 · Market Components",
    "4 · Model Performance",
    "5 · Monitoring",
    "6 · Pricing Action",
    "7 · Profile Explorer",
])

# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Market Overview
# ══════════════════════════════════════════════════════════════════════
with tab1:
    if market_f is None:
        no_data("market_data.csv not found. Re-run the training pipeline to generate it.")
    else:
        mkt_comp_cols = [
            c for c in data.get("feature_metadata", {}).get("market_component_columns", [])
            if c.startswith("avg_top_") and c in market_f.columns
        ]

        col_a, col_b = st.columns(2)

        # 1a — Average competitor premiums over time
        with col_a:
            if mkt_comp_cols:
                m = add_month(market_f, date_col)
                long = (m.groupby("_month")[mkt_comp_cols].mean().reset_index()
                         .melt("_month", var_name="Series", value_name="Premium"))
                fig = px.line(long, x="_month", y="Premium", color="Series",
                              title="Average Competitor Premiums Over Time",
                              labels={"_month": ""})
                st.plotly_chart(fig, use_container_width=True)
            else:
                no_data("No avg_top_N market component columns found.")

        # 1b — Own vs market premium
        with col_b:
            if target_col in market_f.columns and own_col in market_f.columns:
                m = add_month(market_f, date_col)
                comp = (m.groupby("_month")[[own_col, target_col]].mean().reset_index()
                         .melt("_month", var_name="Series", value_name="Premium"))
                fig = px.line(comp, x="_month", y="Premium", color="Series",
                              title=f"Own Premium vs {target_col}",
                              labels={"_month": ""})
                st.plotly_chart(fig, use_container_width=True)

        col_c, col_d = st.columns(2)

        # 1c — Own-to-market ratio over time
        with col_c:
            ratio_col = next((c for c in market_f.columns if "own_to_" in c and "ratio" in c), None)
            if ratio_col:
                m = add_month(market_f, date_col)
                group_by = ["_month", "_split"] if "_split" in m.columns else ["_month"]
                agg = m.groupby(group_by, observed=True)[ratio_col].mean().reset_index()
                color = "_split" if "_split" in agg.columns else None
                fig = px.line(agg, x="_month", y=ratio_col, color=color,
                              color_discrete_map=SPLIT_COLOURS,
                              title="Own-to-Market Ratio Over Time",
                              labels={"_month": "", ratio_col: "Ratio"})
                fig = hline_at_market(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                no_data("own_to_market ratio column not found — configure own_premium_column.")

        # 1d — Conversion rate vs own-to-market ratio (or competitor box plot fallback)
        with col_d:
            if conv_col and conv_col in market_f.columns and ratio_col:
                cd = market_f[[ratio_col, conv_col]].dropna()
                cd = cd[pd.to_numeric(cd[conv_col], errors="coerce").notna()].copy()
                cd[conv_col] = pd.to_numeric(cd[conv_col])
                cd["_bin"] = pd.cut(cd[ratio_col], bins=25)
                agg = (cd.groupby("_bin", observed=True)[conv_col].mean()
                         .reset_index()
                         .assign(_mid=lambda x: x["_bin"].apply(lambda b: b.mid)))
                fig = px.scatter(agg, x="_mid", y=conv_col,
                                 title="Conversion Rate vs Own-to-Market Ratio",
                                 labels={"_mid": "Own / Market ratio", conv_col: "Conversion rate"})
                fig.add_vline(x=1.0, line_dash="dash", line_color="grey")
                st.plotly_chart(fig, use_container_width=True)
            else:
                avail = [c for c in selected_comps if c in market_f.columns]
                if avail:
                    m = add_month(market_f, date_col)
                    long = (m[["_month"] + avail].melt("_month", var_name="Competitor", value_name="Premium")
                              .dropna())
                    long["Month"] = long["_month"].dt.strftime("%Y-%m")
                    fig = px.box(long, x="Month", y="Premium", color="Competitor",
                                 title="Competitor Premium Distribution by Month",
                                 labels={"Month": ""})
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)

        # 1e — Mix-adjusted competitor price index
        basket_idx = data.get("basket_index")
        if basket_idx is not None and not basket_idx.empty:
            basket_idx = basket_idx.copy()
            basket_idx["_month"] = pd.to_datetime(basket_idx["quote_month"], errors="coerce")
            basket_idx = basket_idx.dropna(subset=["_month"])

            fig = go.Figure()
            # IQR band
            fig.add_trace(go.Scatter(
                x=basket_idx["_month"], y=basket_idx["p75_prediction"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=basket_idx["_month"], y=basket_idx["p25_prediction"],
                mode="lines", fill="tonexty",
                fillcolor="rgba(255,127,14,0.15)",
                line=dict(width=0), name="IQR (p25–p75)",
            ))
            # Mix-adjusted mean
            fig.add_trace(go.Scatter(
                x=basket_idx["_month"], y=basket_idx["mean_prediction"],
                mode="lines", name="Mix-adjusted index",
                line=dict(color="#FF7F0E", width=2),
            ))
            # Raw observed average for comparison
            if target_col in market_f.columns:
                raw_m = add_month(market_f, date_col)
                raw_avg = raw_m.groupby("_month")[target_col].mean().reset_index()
                fig.add_trace(go.Scatter(
                    x=raw_avg["_month"], y=raw_avg[target_col],
                    mode="lines", name="Raw observed average",
                    line=dict(color="#636EFA", dash="dot", width=2),
                ))
            fig.update_layout(
                title="Mix-Adjusted Competitor Price Index vs Raw Average",
                xaxis_title="",
                yaxis_title="Avg top-3 competitor premium",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"**Mix-adjusted** (orange): model applied to a fixed {basket_idx['n_profiles'].iloc[0]:,}-profile "
                "reference basket each month — isolates genuine rate changes from risk-mix shift.  "
                "**Raw average** (blue dashed): mean of observed quotes, which varies with monthly profile mix."
            )

        # 1f — Market price index heatmap
        avail_segs = [s for s in detected_segments if s in market_f.columns]
        mpi_col = "market_price_index" if "market_price_index" in market_f.columns else target_col
        if len(avail_segs) >= 2 and mpi_col in market_f.columns:
            hc1, hc2 = st.columns(2)
            seg_x = hc1.selectbox("Heatmap rows", avail_segs,
                                   index=0, key="hmap_row")
            seg_y_opts = [s for s in avail_segs if s != seg_x]
            seg_y = hc2.selectbox("Heatmap columns", seg_y_opts,
                                   index=0, key="hmap_col")
            heat = market_f.groupby([seg_x, seg_y], observed=True)[mpi_col].mean().unstack(seg_y)
            fig = px.imshow(heat, color_continuous_scale="RdYlGn_r", aspect="auto",
                            title=f"Market Price Index — {seg_x.replace('_',' ').title()} × {seg_y.replace('_', ' ').title()}",
                            labels={"color": "Index"})
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Competitor-Level
# ══════════════════════════════════════════════════════════════════════
with tab2:
    individual = data.get("individual", {})
    avail_comps = [c for c in selected_comps if market_f is not None and c in market_f.columns]

    if not avail_comps and not individual:
        no_data("No competitor columns or individual models found.")
    else:
        col_a, col_b = st.columns(2)

        # 2a — All competitor price curves
        with col_a:
            if avail_comps and market_f is not None:
                m = add_month(market_f, date_col)
                long = (m.groupby("_month")[avail_comps].mean().reset_index()
                         .melt("_month", var_name="Competitor", value_name="Premium"))
                fig = px.line(long, x="_month", y="Premium", color="Competitor",
                              title="Competitor Price Curves Over Time",
                              labels={"_month": ""})
                st.plotly_chart(fig, use_container_width=True)

        # 2b — Missing quote rate by competitor over time
        with col_b:
            if avail_comps and market_f is not None and date_col in market_f.columns:
                m = add_month(market_f, date_col)
                miss_rows = []
                for c in avail_comps:
                    monthly = m.groupby("_month")[c].apply(lambda s: s.isna().mean() * 100).reset_index()
                    monthly.columns = ["_month", "missing_pct"]
                    monthly["Competitor"] = c
                    miss_rows.append(monthly)
                miss_long = pd.concat(miss_rows, ignore_index=True)
                fig = px.line(miss_long, x="_month", y="missing_pct", color="Competitor",
                              title="Missing Quote Rate by Competitor Over Time (%)",
                              labels={"_month": "", "missing_pct": "Missing (%)"})
                fig.add_hline(y=0, line_dash="dot", line_color="lightgrey", line_width=1)
                st.plotly_chart(fig, use_container_width=True)

        # 2c — Competitor aggressiveness heatmap
        avail_segs_2 = [s for s in detected_segments if market_f is not None and s in market_f.columns]
        seg_col = st.selectbox("Group aggressiveness by", avail_segs_2, key="agg_seg") if avail_segs_2 else None
        if seg_col and avail_comps and market_f is not None:
            market_mean = market_f[avail_comps].stack().mean()
            if market_mean and market_mean > 0:
                rel = (market_f.groupby(seg_col, observed=True)[avail_comps].mean() / market_mean).round(3)
                fig = px.imshow(rel, color_continuous_scale="RdYlGn_r", aspect="auto",
                                title=f"Competitor Aggressiveness by {seg_col.replace('_',' ').title()} (relative to market mean)",
                                labels={"color": "Relative price"})
                st.plotly_chart(fig, use_container_width=True)

        # 2d — Individual model actual vs predicted + metrics table
        shown_ind = [c for c in selected_comps if c in individual]
        if shown_ind:
            st.subheader("Individual Model Results")

            # Metrics comparison table
            rows = []
            for c in shown_ind:
                m = individual[c].get("metrics", {}).get("test", {})
                rows.append({
                    "Competitor": c,
                    "R²": round(m.get("r2", float("nan")), 4),
                    "RMSE": round(m.get("rmse", float("nan")), 2),
                    "MAPE%": round(m.get("mape", float("nan")), 2),
                    "Gini": round(m.get("gini", float("nan")), 4),
                    "Bias%": round(m.get("mean_bias_pct", float("nan")), 2),
                    "N (test)": m.get("n", ""),
                })
            st.dataframe(pd.DataFrame(rows).set_index("Competitor"), use_container_width=True)

            # Actual vs predicted scatter per competitor
            comp_choice = st.selectbox("Show scatter for", shown_ind)
            entry = individual[comp_choice]
            p_df = entry.get("predictions_test")
            if p_df is not None and comp_choice in p_df.columns and "prediction" in p_df.columns:
                p_df_f = apply_filters(p_df, date_col=date_col, date_from=date_from,
                                       date_to=date_to, segment_filters={})
                fig = px.scatter(p_df_f, x=comp_choice, y="prediction", opacity=0.35,
                                 title=f"{comp_choice}: Actual vs Predicted (test set)",
                                 labels={comp_choice: "Actual", "prediction": "Predicted"})
                all_vals = pd.concat([p_df_f[comp_choice], p_df_f["prediction"]]).dropna()
                fig = diagonal_line(fig, all_vals)
                st.plotly_chart(fig, use_container_width=True)

            # Derived top-3 from individual models vs aggregated model
            if len(shown_ind) >= 3 and pred_test_f is not None and target_col in (pred_test_f.columns if pred_test_f is not None else []):
                st.subheader("Derived Top-3 (individual models) vs Aggregated Model vs Actual")
                id_cols = (data.get("run_config") or {}).get("data", {}).get("id_columns", [])
                id_col = id_cols[0] if id_cols else None

                comp_pred_frames = {}
                for c in shown_ind:
                    pf = individual[c].get("predictions_test")
                    if pf is not None and "prediction" in pf.columns:
                        if id_col and id_col in pf.columns:
                            comp_pred_frames[c] = pf.set_index(id_col)["prediction"]
                        else:
                            comp_pred_frames[c] = pf["prediction"].reset_index(drop=True)

                if len(comp_pred_frames) >= 3:
                    preds_mat = pd.DataFrame(comp_pred_frames)
                    sorted_mat = np.sort(preds_mat.values, axis=1)[:, :3]
                    derived = pd.Series(np.nanmean(sorted_mat, axis=1), name="derived_top3")

                    base = pred_test_f[[date_col, target_col, "prediction"]].copy() if date_col in pred_test_f.columns else pred_test_f[[target_col, "prediction"]].copy()
                    base = base.iloc[:len(derived)].copy()
                    base["derived_top3"] = derived.values

                    if date_col in base.columns:
                        base = add_month(base, date_col)
                        agg = base.groupby("_month")[[target_col, "prediction", "derived_top3"]].mean().reset_index()
                        long = agg.melt("_month", var_name="Series", value_name="Premium")
                        rename = {target_col: "Actual", "prediction": "Aggregated model",
                                  "derived_top3": "Derived from individual models"}
                        long["Series"] = long["Series"].map(rename)
                        fig = px.line(long, x="_month", y="Premium", color="Series",
                                      title="Actual vs Aggregated Model vs Derived Top-3 Over Time",
                                      labels={"_month": ""})
                        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — Market Components
# ══════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)

    # 3a — Actual vs predicted over time (all splits)
    with col_a:
        frames = []
        for name, pf in [("train", pred_train_f), ("validation", pred_val_f), ("test", pred_test_f)]:
            if pf is not None and target_col in pf.columns and date_col in pf.columns:
                pf = pf.copy()
                pf["_label"] = name
                frames.append(pf)
        if frames:
            all_p = pd.concat(frames, ignore_index=True)
            m = add_month(all_p, date_col)
            agg = m.groupby(["_month", "_label"])[[target_col, "prediction"]].mean().reset_index()
            fig = go.Figure()
            for split, grp in agg.groupby("_label"):
                clr = SPLIT_COLOURS.get(split, "grey")
                fig.add_trace(go.Scatter(x=grp["_month"], y=grp[target_col],
                                         name=f"Actual ({split})", line=dict(color=clr)))
                fig.add_trace(go.Scatter(x=grp["_month"], y=grp["prediction"],
                                         name=f"Predicted ({split})", line=dict(color=clr, dash="dot")))
            fig.update_layout(title=f"Actual vs Predicted {target_col}", xaxis_title="", yaxis_title="Premium")
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data()

    # 3b — Own rank distribution over time (stacked %)
    with col_b:
        rank_col = "rank_own_premium"
        if market_f is not None and rank_col in market_f.columns and date_col in market_f.columns:
            m = add_month(market_f, date_col)
            m["_rank"] = m[rank_col].round().astype("Int64").astype(str)
            monthly_counts = (
                m.groupby(["_month", "_rank"], observed=True)
                 .size()
                 .reset_index(name="n")
            )
            monthly_totals = monthly_counts.groupby("_month")["n"].transform("sum")
            monthly_counts["share_pct"] = (monthly_counts["n"] / monthly_totals * 100).round(1)
            monthly_counts["Month"] = monthly_counts["_month"].dt.strftime("%Y-%m")

            # Order ranks so rank 1 is at the bottom of the stack
            rank_order = sorted(monthly_counts["_rank"].unique(), key=lambda x: int(x))
            fig = px.bar(
                monthly_counts, x="Month", y="share_pct", color="_rank",
                category_orders={"_rank": rank_order},
                title="Own Price Rank Distribution Over Time",
                labels={"share_pct": "Share of quotes (%)", "Month": "", "_rank": "Rank"},
                color_discrete_sequence=px.colors.sequential.Blues_r[:len(rank_order)],
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(barmode="stack", legend_title="Rank (1 = cheapest)")
            st.plotly_chart(fig, use_container_width=True)
        elif market_f is not None and rank_col in market_f.columns:
            # Fallback: static distribution if no date column
            counts = market_f[rank_col].round().value_counts().sort_index()
            fig = px.bar(x=counts.index.astype(int), y=counts.values,
                         title="Own Price Rank Distribution",
                         labels={"x": "Rank (1 = cheapest)", "y": "Quote count"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data("rank_own_premium not available — set own_premium_column in config.")

    col_c, col_d = st.columns(2)

    # 3c — Price gap to market by segment
    with col_c:
        gap_col = next((c for c in (market_f.columns if market_f is not None else [])
                        if "price_gap" in c), None)
        avail_segs_3c = [s for s in detected_segments if market_f is not None and s in market_f.columns]
        seg_for_gap = st.selectbox("Group by", avail_segs_3c, key="gap_seg") if avail_segs_3c else None
        if gap_col and seg_for_gap and date_col in market_f.columns:
            m = add_month(market_f, date_col)
            agg = (m.groupby(["_month", seg_for_gap], observed=True)[gap_col]
                    .mean()
                    .reset_index()
                    .rename(columns={gap_col: "gap"}))
            fig = px.line(agg, x="_month", y="gap", color=seg_for_gap,
                          title=f"Price Gap to Market Over Time by {seg_for_gap.replace('_',' ').title()}",
                          labels={"_month": "", "gap": "Gap (own − market)"})
            fig.add_hline(y=0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data("price_gap column or segment column not found.")

    # 3d — Competitor premium dispersion over time
    with col_d:
        std_col = "competitor_premium_std"
        if market_f is not None and std_col in market_f.columns:
            m = add_month(market_f, date_col)
            agg = m.groupby("_month")[std_col].mean().reset_index()
            fig = px.line(agg, x="_month", y=std_col,
                          title="Competitor Premium Dispersion (Std) Over Time",
                          labels={"_month": "", std_col: "Std dev"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data("competitor_premium_std not available — enable add_competitor_distribution.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ══════════════════════════════════════════════════════════════════════
with tab4:
    metrics = data.get("metrics")
    if metrics:
        # ── Headline KPIs: actuarial metrics first, R² demoted to gate check ──
        samples = [s for s in ["train", "validation", "test"] if s in metrics]
        for sample in samples:
            m = metrics[sample]
            gini      = m.get("gini", float("nan"))
            mape      = m.get("mape", float("nan"))
            bias_pct  = m.get("mean_bias_pct", float("nan"))
            rmsle     = m.get("rmsle", float("nan"))
            r2        = m.get("r2", float("nan"))
            n         = m.get("n", 0)

            st.markdown(f"**{sample.title()}** &nbsp; <sub>{n:,} obs</sub>", unsafe_allow_html=True)
            kc = st.columns(5)
            kc[0].metric("Gini",   f"{gini:.3f}"  if not np.isnan(gini)     else "—",
                         help="Actuarial discrimination (rank-ordering). Higher = better.")
            kc[1].metric("MAPE",   f"{mape:.2f}%" if not np.isnan(mape)     else "—",
                         help="Mean absolute % error. Scale-invariant; directly interpretable as pricing accuracy.")
            kc[2].metric("Bias%",  f"{bias_pct:+.2f}%" if not np.isnan(bias_pct) else "—",
                         help="Systematic over/under-pricing. Target: within ±5%.")
            kc[3].metric("RMSLE",  f"{rmsle:.4f}" if not np.isnan(rmsle)    else "—",
                         help="Log-scale error; natural for multiplicative pricing models.")
            kc[4].metric("R² (legacy)", f"{r2:.4f}" if not np.isnan(r2) else "—",
                         help="Kept for reference only. D² (Gamma deviance) is the correct goodness-of-fit measure for positive right-skewed prices.")

        with st.expander("Full metrics table"):
            rows = []
            for s in ["train", "validation", "test"]:
                if s in metrics:
                    m = metrics[s]
                    rows.append({"Sample": s, "Gini": m.get("gini"), "MAPE%": m.get("mape"),
                                 "Bias%": m.get("mean_bias_pct"), "RMSLE": m.get("rmsle"),
                                 "D²": m.get("d2"), "MAE": m.get("mae"), "RMSE": m.get("rmse"),
                                 "R² (legacy)": m.get("r2"), "N": m.get("n")})
            st.dataframe(pd.DataFrame(rows).set_index("Sample").round(4), use_container_width=True)
        st.divider()

    col_a, col_b = st.columns(2)

    # 4a — Actual vs predicted scatter (test)
    with col_a:
        if pred_test_f is not None and target_col in pred_test_f.columns:
            color_col = date_col if date_col in pred_test_f.columns else None
            fig = px.scatter(pred_test_f, x=target_col, y="prediction", opacity=0.3,
                             color=color_col,
                             title="Actual vs Predicted — Test Set",
                             labels={target_col: "Actual", "prediction": "Predicted"})
            all_vals = pd.concat([pred_test_f[target_col], pred_test_f["prediction"]]).dropna()
            fig = diagonal_line(fig, all_vals)
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data()

    # 4b — Residuals over time
    with col_b:
        res_frames = []
        for name, pf in [("train", pred_train_f), ("validation", pred_val_f), ("test", pred_test_f)]:
            if pf is not None and "residual" in pf.columns and date_col in pf.columns:
                pf = pf.copy()
                pf["_label"] = name
                res_frames.append(pf)
        if res_frames:
            all_r = pd.concat(res_frames, ignore_index=True)
            m = add_month(all_r, date_col)
            agg = m.groupby(["_month", "_label"])["residual"].mean().reset_index()
            fig = px.line(agg, x="_month", y="residual", color="_label",
                          color_discrete_map=SPLIT_COLOURS,
                          title="Mean Residual Over Time",
                          labels={"_month": "", "residual": "Mean residual", "_label": "Split"})
            fig = hline_zero(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data()

    col_c, col_d = st.columns(2)

    # 4c — Decile lift chart
    with col_c:
        lt = data.get("lift_table")
        if lt:
            lt_df = pd.DataFrame(lt)
            fig = go.Figure([
                go.Bar(name="Mean actual",    x=lt_df["quantile"], y=lt_df["mean_actual"],    marker_color="#636EFA"),
                go.Bar(name="Mean predicted", x=lt_df["quantile"], y=lt_df["mean_predicted"], marker_color="#EF553B"),
            ])
            fig.update_layout(barmode="group",
                              title="Decile Lift: Mean Actual vs Predicted",
                              xaxis_title="Decile (1 = lowest predicted)",
                              yaxis_title="Premium")
            st.plotly_chart(fig, use_container_width=True)

            # Pred/actual ratio line
            fig2 = px.line(lt_df, x="quantile", y="pred_actual_ratio",
                           title="Pred / Actual Ratio by Decile (ideal = 1.0)",
                           labels={"quantile": "Decile", "pred_actual_ratio": "Ratio"})
            fig2.add_hline(y=1.0, line_dash="dash", line_color="grey")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            no_data("lift_table_test.json not found.")

    # 4d — Feature importance
    with col_d:
        fi = data.get("feature_importance")
        if fi is not None and not fi.empty:
            top15 = fi.nlargest(15, "importance_mean")
            error_col = "importance_std" if "importance_std" in fi.columns else None
            fig = px.bar(top15, x="importance_mean", y="feature", orientation="h",
                         error_x=error_col,
                         title="Top 15 Feature Importance (permutation)",
                         labels={"importance_mean": "Importance", "feature": ""})
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            no_data()

    # 4e — Optuna tuning results
    tuning = data.get("tuning_results")
    if tuning:
        st.divider()
        st.subheader("Hyperparameter Tuning (Optuna)")

        metric_name = tuning.get("metric", "metric")
        direction   = tuning.get("direction", "minimize")
        best_value  = tuning.get("best_value")
        n_trials    = tuning.get("n_trials", 0)
        n_completed = tuning.get("n_completed", 0)
        backend     = tuning.get("backend", "")

        tk = st.columns(4)
        tk[0].metric("Backend",     backend)
        tk[1].metric("Trials (completed)", f"{n_completed}/{n_trials}")
        tk[2].metric(f"Best {metric_name}", f"{best_value:.4f}" if best_value is not None else "—",
                     help=f"Direction: {direction}")
        tk[3].metric("Metric",      metric_name)

        # Best params table
        best_params = tuning.get("best_params", {})
        if best_params:
            with st.expander("Best hyperparameters", expanded=True):
                st.dataframe(
                    pd.DataFrame(
                        [{"Parameter": k, "Value": v} for k, v in best_params.items()]
                    ).set_index("Parameter"),
                    use_container_width=True,
                )

        # Trial history chart
        all_trials = tuning.get("all_trials", [])
        if all_trials:
            trials_df = pd.DataFrame([
                {"Trial": t["number"], metric_name: t["value"]}
                for t in all_trials if t.get("value") is not None
            ])
            if not trials_df.empty:
                # Running best overlay
                if direction == "minimize":
                    trials_df["best_so_far"] = trials_df[metric_name].cummin()
                else:
                    trials_df["best_so_far"] = trials_df[metric_name].cummax()

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=trials_df["Trial"], y=trials_df[metric_name],
                    mode="markers", name="Trial value",
                    marker=dict(color="#636EFA", opacity=0.5, size=6),
                ))
                fig.add_trace(go.Scatter(
                    x=trials_df["Trial"], y=trials_df["best_so_far"],
                    mode="lines", name="Best so far",
                    line=dict(color="#EF553B", width=2),
                ))
                fig.update_layout(
                    title=f"Optuna Trial History — {metric_name} ({direction})",
                    xaxis_title="Trial number",
                    yaxis_title=metric_name,
                )
                st.plotly_chart(fig, use_container_width=True)

    # 4f — Residuals by segment
    avail_segs_4f = [s for s in detected_segments if reference_f is not None and s in reference_f.columns]
    seg_for_resid = st.selectbox("Residuals by segment", avail_segs_4f, key="resid_seg") if avail_segs_4f else None
    if seg_for_resid and reference_f is not None and "residual" in reference_f.columns and date_col in reference_f.columns:
        m = add_month(reference_f, date_col)
        agg = (m.groupby(["_month", seg_for_resid], observed=True)["residual"]
                .mean()
                .reset_index())
        fig = px.line(agg, x="_month", y="residual", color=seg_for_resid,
                      title=f"Mean Residual Over Time by {seg_for_resid.replace('_',' ').title()} — Test Set",
                      labels={"_month": "", "residual": "Mean residual"})
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — Monitoring
# ══════════════════════════════════════════════════════════════════════
with tab5:
    metrics = data.get("metrics")
    dq = data.get("data_quality", {})

    # Data quality KPIs
    if dq:
        st.subheader("Data Quality")
        kpi = st.columns(4)
        kpi[0].metric("Total rows", f"{dq.get('rows', '?'):,}")
        kpi[1].metric("Rows with enough competitors",
                      f"{dq.get('rows_with_enough_competitors', '?'):,}")
        kpi[2].metric("Date min", str(dq.get("date_min", "?")))
        kpi[3].metric("Date max", str(dq.get("date_max", "?")))
        st.divider()

    col_a, col_b = st.columns(2)

    # 5a — R² and Gini by sample
    with col_a:
        if metrics:
            perf_rows = []
            for s in ["train", "validation", "test"]:
                if s in metrics:
                    m = metrics[s]
                    perf_rows.append({"Sample": s,
                                      "D²": m.get("d2", m.get("r2", 0)),
                                      "Gini": m.get("gini", 0)})
            df_p = pd.DataFrame(perf_rows)
            fig = px.bar(df_p, x="Sample", y=["D²", "Gini"], barmode="group",
                         title="D² (Gamma) and Gini by Sample",
                         labels={"value": "Score", "variable": "Metric"})
            st.plotly_chart(fig, use_container_width=True)

    # 5b — MAPE% and Bias%
    with col_b:
        if metrics:
            perf_rows2 = []
            for s in ["train", "validation", "test"]:
                if s in metrics:
                    m = metrics[s]
                    perf_rows2.append({"Sample": s, "MAPE%": m.get("mape", 0),
                                       "Bias%": m.get("mean_bias_pct", 0)})
            df_p2 = pd.DataFrame(perf_rows2)
            fig = px.bar(df_p2, x="Sample", y=["MAPE%", "Bias%"], barmode="group",
                         title="MAPE% and Bias% by Sample",
                         labels={"value": "%", "variable": "Metric"})
            fig = hline_zero(fig)
            st.plotly_chart(fig, use_container_width=True)

    # 5c — Prediction distribution drift (output shift)
    if pred_test_f is not None and "prediction" in pred_test_f.columns and date_col in pred_test_f.columns:
        m = add_month(pred_test_f, date_col)
        agg = m.groupby("_month")["prediction"].agg(["mean", "std"]).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=agg["_month"], y=agg["mean"] + agg["std"],
                                 mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=agg["_month"], y=agg["mean"] - agg["std"],
                                 mode="lines", fill="tonexty", fillcolor="rgba(0,204,150,0.15)",
                                 line=dict(width=0), name="±1 std"))
        fig.add_trace(go.Scatter(x=agg["_month"], y=agg["mean"],
                                 mode="lines", name="Mean prediction",
                                 line=dict(color="#00CC96", width=2)))
        fig.update_layout(title="Prediction Distribution Drift — Test Set",
                          xaxis_title="", yaxis_title="Predicted premium")
        st.plotly_chart(fig, use_container_width=True)

    # 5d — Segment MAPE (deterioration map)
    avail_segs_5d = [s for s in detected_segments if reference_f is not None and s in reference_f.columns]
    seg_mon = st.selectbox("MAPE deterioration by segment", avail_segs_5d, key="mon_seg") if avail_segs_5d else None
    if seg_mon and reference_f is not None and "absolute_error" in reference_f.columns and target_col in reference_f.columns and date_col in reference_f.columns:
        rc = reference_f.copy()
        rc["_mape"] = rc["absolute_error"] / rc[target_col].replace(0, np.nan) * 100
        m = add_month(rc, date_col)
        agg = m.groupby(["_month", seg_mon], observed=True)["_mape"].mean().reset_index()
        fig = px.line(agg, x="_month", y="_mape", color=seg_mon,
                      title=f"MAPE% Over Time by {seg_mon.replace('_',' ').title()} — Test Set",
                      labels={"_month": "", "_mape": "MAPE%"})
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 6 — Pricing Action
# ══════════════════════════════════════════════════════════════════════
with tab6:
    seg_avail = [s for s in detected_segments if market_f is not None and s in market_f.columns]

    if not seg_avail:
        no_data("No segment columns found in market data.")
    else:
        seg_choice = st.selectbox("Group by segment", seg_avail, key="action_seg")

        ratio_col = next((c for c in (market_f.columns if market_f is not None else [])
                          if "own_to_" in c and "ratio" in c), None)
        rank_col = "rank_own_premium" if market_f is not None and "rank_own_premium" in market_f.columns else None
        gap_col = next((c for c in (market_f.columns if market_f is not None else [])
                        if "price_gap" in c), None)

        agg_cols = [c for c in [own_col, target_col, ratio_col, rank_col, gap_col]
                    if c and market_f is not None and c in market_f.columns]
        if agg_cols:
            pos = market_f.groupby(seg_choice, observed=True)[agg_cols].mean().round(3)
            if ratio_col in pos.columns:
                def _action(r: float) -> str:
                    if r > 1.10:
                        return "⬇ Reduce — above market"
                    if r < 0.90:
                        return "⬆ Opportunity — below market"
                    return "✓ Hold — at market"
                pos["Recommended action"] = pos[ratio_col].map(_action)
            st.subheader("Segment Positioning Table")
            st.dataframe(pos, use_container_width=True)

        col_a, col_b = st.columns(2)

        # 6a — Opportunity map: price gap vs own rank
        with col_a:
            if gap_col and rank_col and market_f is not None:
                agg2 = market_f.groupby(seg_choice, observed=True)[[gap_col, rank_col]].mean().reset_index()
                fig = px.scatter(agg2, x=gap_col, y=rank_col, text=seg_choice,
                                 color=gap_col, color_continuous_scale="RdYlGn_r",
                                 title="Opportunity Map: Price Gap vs Own Rank",
                                 labels={gap_col: "Price gap (own − market)",
                                         rank_col: "Avg own rank (lower = cheaper)"})
                fig.add_vline(x=0, line_dash="dash", line_color="grey")
                fig.update_traces(textposition="top center")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                no_data("price_gap or rank_own_premium column not available.")

        # 6b — Segments ranked by MAPE (model uncertainty = pricing risk)
        with col_b:
            if (reference_f is not None and seg_choice in reference_f.columns
                    and "absolute_error" in reference_f.columns
                    and target_col in reference_f.columns):
                rc = reference_f.copy()
                rc["_mape"] = rc["absolute_error"] / rc[target_col].replace(0, np.nan) * 100
                agg3 = rc.groupby(seg_choice, observed=True)["_mape"].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(agg3, x=seg_choice, y="_mape",
                             title="Segments by Model MAPE% (highest = most pricing uncertainty)",
                             labels={"_mape": "MAPE%", seg_choice: ""},
                             color="_mape", color_continuous_scale="Oranges")
                fig.update_coloraxes(showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                no_data()


# ══════════════════════════════════════════════════════════════════════
# TAB 7 — Profile Explorer  (raw data only)
# ══════════════════════════════════════════════════════════════════════
with tab7:
    if market_raw is None:
        no_data("market_data.csv not found — run the pipeline first.")
    else:
        mf             = data.get("model_features", {})
        cat_cols_model = mf.get("categorical_columns", [])
        all_num_cols   = mf.get("numeric_columns", [])
        temporal_names = set(data.get("feature_metadata", {}).get("temporal_columns", []))
        base_num_cols  = [c for c in all_num_cols if c not in temporal_names]

        comp_cols_avail = [c for c in data.get("competitor_columns", []) if c in market_raw.columns]

        # ── Segment filters ───────────────────────────────────────────
        st.subheader("Segment Filters")
        cat_cols_present = [c for c in cat_cols_model if c in market_raw.columns]

        pe_cat_filters: dict[str, list] = {}
        if cat_cols_present:
            cat_wcols = st.columns(min(len(cat_cols_present), 4))
            for i, col in enumerate(cat_cols_present):
                opts = sorted(market_raw[col].dropna().unique().tolist())
                sel = cat_wcols[i % 4].multiselect(
                    col.replace("_", " ").title(), opts, default=opts, key=f"pe7_cat_{col}"
                )
                pe_cat_filters[col] = sel if sel else opts

        num_filters: dict[str, tuple] = {}
        num_cols_present = [c for c in base_num_cols if c in market_raw.columns]
        if num_cols_present:
            num_wcols = st.columns(min(len(num_cols_present), 4))
            for i, col in enumerate(num_cols_present):
                arr = pd.to_numeric(market_raw[col], errors="coerce").dropna()
                mn, mx = float(arr.min()), float(arr.max())
                lo, hi = num_wcols[i % 4].slider(
                    col.replace("_", " ").title(),
                    min_value=mn, max_value=mx, value=(mn, mx),
                    key=f"pe7_num_{col}",
                )
                num_filters[col] = (lo, hi)

        # ── Apply filters ─────────────────────────────────────────────
        seg = market_raw.copy()
        for col, vals in pe_cat_filters.items():
            seg = seg[seg[col].isin(vals)]
        for col, (lo, hi) in num_filters.items():
            numeric_col = pd.to_numeric(seg[col], errors="coerce")
            seg = seg[(numeric_col >= lo) & (numeric_col <= hi)]

        n_rows = len(seg)
        pct = 100 * n_rows / len(market_raw) if len(market_raw) > 0 else 0
        st.caption(f"**{n_rows:,} rows** match ({pct:.1f}% of dataset)")

        if n_rows == 0:
            no_data("No rows match the current filters — widen the selection.")
        else:
            seg = add_month(seg, date_col)
            monthly_n = seg.groupby("_month").size().reset_index(name="n")

            st.divider()

            # ── Chart A: own premium + each competitor over time ─────
            price_series = [c for c in [own_col] + comp_cols_avail if c and c in seg.columns]
            if price_series:
                agg_prices = (
                    seg.groupby("_month")[price_series]
                    .mean()
                    .reset_index()
                    .merge(monthly_n, on="_month")
                    .melt(id_vars=["_month", "n"], var_name="Series", value_name="Premium")
                )
                fig_p = px.line(
                    agg_prices, x="_month", y="Premium", color="Series",
                    title="Own Premium vs Competitor Premiums Over Time (segment average)",
                    labels={"_month": "", "Premium": "Avg premium (£)"},
                )
                st.plotly_chart(fig_p, use_container_width=True)

            col_a, col_b = st.columns(2)

            # ── Chart B: own-to-market ratio ─────────────────────────
            with col_a:
                ratio_col_raw = next(
                    (c for c in seg.columns if "own_to_" in c and "ratio" in c), None
                )
                if ratio_col_raw:
                    agg_ratio = seg.groupby("_month")[ratio_col_raw].mean().reset_index()
                    fig_r = go.Figure()
                    fig_r.add_hrect(
                        y0=0.95, y1=1.05, fillcolor="rgba(0,204,150,0.08)",
                        line_width=0, annotation_text="±5%", annotation_position="top left",
                    )
                    fig_r.add_trace(go.Scatter(
                        x=agg_ratio["_month"], y=agg_ratio[ratio_col_raw],
                        mode="lines+markers",
                        marker=dict(
                            color=agg_ratio[ratio_col_raw].apply(
                                lambda r: "#EF553B" if r > 1.05 else ("#636EFA" if r < 0.95 else "#00CC96")
                            ),
                            size=9,
                        ),
                        line=dict(color="grey", width=1),
                        showlegend=False,
                    ))
                    fig_r = hline_at_market(fig_r)
                    fig_r.update_layout(
                        title="Own-to-Market Ratio Over Time",
                        xaxis_title="", yaxis_title="Ratio",
                    )
                    st.plotly_chart(fig_r, use_container_width=True)
                else:
                    no_data("own_to_market ratio column not found.")

            # ── Chart C: own rank distribution ───────────────────────
            with col_b:
                rank_col_raw = "rank_own_premium"
                if rank_col_raw in seg.columns:
                    seg["_rank_str"] = (
                        seg[rank_col_raw].round().astype("Int64").astype(str)
                    )
                    rank_counts = (
                        seg.groupby(["_month", "_rank_str"], observed=True)
                        .size()
                        .reset_index(name="n")
                    )
                    totals = rank_counts.groupby("_month")["n"].transform("sum")
                    rank_counts["share_pct"] = (rank_counts["n"] / totals * 100).round(1)
                    rank_counts["Month"] = rank_counts["_month"].dt.strftime("%Y-%m")
                    rank_order = sorted(rank_counts["_rank_str"].unique(), key=lambda x: int(x))
                    fig_rk = px.bar(
                        rank_counts, x="Month", y="share_pct", color="_rank_str",
                        category_orders={"_rank_str": rank_order},
                        title="Own Price Rank Distribution Over Time",
                        labels={"share_pct": "Share (%)", "Month": "", "_rank_str": "Rank"},
                        color_discrete_sequence=px.colors.sequential.Blues_r[:len(rank_order)],
                    )
                    fig_rk.update_xaxes(tickangle=45)
                    fig_rk.update_layout(barmode="stack", legend_title="Rank (1 = cheapest)")
                    st.plotly_chart(fig_rk, use_container_width=True)
                else:
                    no_data("rank_own_premium not available.")

            col_c, col_d = st.columns(2)

            # ── Chart D: conversion rate ──────────────────────────────
            with col_c:
                if conv_col and conv_col in seg.columns:
                    conv_num = pd.to_numeric(seg[conv_col], errors="coerce")
                    seg_conv = seg.copy()
                    seg_conv["_conv"] = conv_num
                    agg_conv = seg_conv.groupby("_month")["_conv"].mean().reset_index()
                    fig_conv = px.line(
                        agg_conv, x="_month", y="_conv",
                        title="Conversion Rate Over Time",
                        labels={"_month": "", "_conv": "Conversion rate"},
                        markers=True,
                    )
                    st.plotly_chart(fig_conv, use_container_width=True)
                else:
                    no_data("Conversion column not available.")

            # ── Chart E: missing quote rate per competitor ────────────
            with col_d:
                if comp_cols_avail:
                    miss_rows = []
                    for c in comp_cols_avail:
                        monthly = (
                            seg.groupby("_month")[c]
                            .apply(lambda s: s.isna().mean() * 100)
                            .reset_index()
                        )
                        monthly.columns = ["_month", "missing_pct"]
                        monthly["Competitor"] = c
                        miss_rows.append(monthly)
                    miss_long = pd.concat(miss_rows, ignore_index=True)
                    fig_miss = px.line(
                        miss_long, x="_month", y="missing_pct", color="Competitor",
                        title="Missing Quote Rate by Competitor (%)",
                        labels={"_month": "", "missing_pct": "Missing (%)"},
                    )
                    fig_miss.add_hline(y=0, line_dash="dot", line_color="lightgrey", line_width=1)
                    st.plotly_chart(fig_miss, use_container_width=True)
                else:
                    no_data("No competitor columns found.")

            # ── Monthly summary table ─────────────────────────────────
            with st.expander("Monthly summary table"):
                sum_cols = (
                    [own_col] if own_col and own_col in seg.columns else []
                ) + [c for c in comp_cols_avail] + (
                    [ratio_col_raw] if ratio_col_raw and ratio_col_raw in seg.columns else []
                ) + (
                    [rank_col_raw] if rank_col_raw in seg.columns else []
                )
                if sum_cols:
                    tbl = seg.groupby("_month")[sum_cols].mean().round(2)
                    tbl["n_quotes"] = seg.groupby("_month").size()
                    tbl.index = tbl.index.strftime("%Y-%m")
                    st.dataframe(tbl, use_container_width=True)
