"""Competitor availability and target-composition diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.reporting import write_json


def save_panel_diagnostics(
    data: pd.DataFrame,
    competitor_columns: list[str],
    config: PipelineConfig,
    output_dir: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    date_column = config.data.date_column
    target = config.data.target.name
    working = data.copy()
    working["quote_month"] = (
        pd.to_datetime(working[date_column], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    working = working.dropna(subset=["quote_month"])

    coverage_rows = []
    for month, month_frame in working.groupby("quote_month"):
        for competitor in competitor_columns:
            coverage_rows.append({
                "quote_month": month,
                "competitor": competitor,
                "quote_rate": float(month_frame[competitor].notna().mean()),
                "rows": int(len(month_frame)),
            })
    coverage = pd.DataFrame(coverage_rows)

    eligibility = (
        working.groupby("quote_month")[target]
        .apply(lambda series: series.notna().mean())
        .reset_index(name="target_eligibility_rate")
    )
    eligibility["rows"] = working.groupby("quote_month").size().to_numpy()

    composition = (
        working.dropna(subset=["top_n_competitor_signature"])
        .groupby(["quote_month", "top_n_competitor_signature"], observed=True)
        .agg(rows=(target, "size"), mean_target=(target, "mean"))
        .reset_index()
    )
    if not composition.empty:
        composition["monthly_share"] = composition["rows"] / composition.groupby(
            "quote_month"
        )["rows"].transform("sum")

    complete_mask = working[competitor_columns].notna().all(axis=1)
    complete_mean = working.loc[complete_mask, target].mean()
    incomplete_mean = working.loc[~complete_mask, target].mean()
    incomplete_bias_pct = (
        float((incomplete_mean / complete_mean - 1) * 100)
        if pd.notna(complete_mean) and complete_mean != 0 and pd.notna(incomplete_mean)
        else None
    )
    summary = {
        "missing_panel_policy": config.data.target.missing_panel_policy,
        "minimum_monthly_competitor_coverage": (
            float(coverage["quote_rate"].min()) if not coverage.empty else None
        ),
        "minimum_monthly_target_eligibility": (
            float(eligibility["target_eligibility_rate"].min())
            if not eligibility.empty else None
        ),
        "complete_panel_rate": float(complete_mask.mean()),
        "complete_panel_mean_target": (
            float(complete_mean) if pd.notna(complete_mean) else None
        ),
        "incomplete_panel_mean_target": (
            float(incomplete_mean) if pd.notna(incomplete_mean) else None
        ),
        "incomplete_vs_complete_target_bias_pct": incomplete_bias_pct,
        "distinct_top_n_compositions": int(
            working["top_n_competitor_signature"].nunique(dropna=True)
        ),
    }

    coverage_path = output_dir / "competitor_coverage_by_month.csv"
    eligibility_path = output_dir / "target_eligibility_by_month.csv"
    composition_path = output_dir / "target_panel_composition.csv"
    summary_path = output_dir / "panel_diagnostics.json"
    coverage.to_csv(coverage_path, index=False)
    eligibility.to_csv(eligibility_path, index=False)
    composition.to_csv(composition_path, index=False)
    write_json(summary, summary_path)
    return ({
        "competitor_coverage_by_month": str(coverage_path),
        "target_eligibility_by_month": str(eligibility_path),
        "target_panel_composition": str(composition_path),
        "panel_diagnostics": str(summary_path),
    }, summary)
