"""Market component and model feature engineering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from competitor_pricing_ai.config import PipelineConfig
from competitor_pricing_ai.data import detect_competitor_columns


TARGET_DERIVED_PREFIXES = (
    "avg_top_",
    "min_competitor",
    "max_competitor",
    "median_competitor",
    "softmin_competitor",
    "competitor_premium_",
    "market_price_index",
    "own_to_",
    "price_gap_",
    "rank_",
    "top_",
    "segment_market_aggressiveness",
)


@dataclass
class FeatureMetadata:
    competitor_columns: list[str]
    target_column: str
    market_component_columns: list[str]
    temporal_columns: list[str]
    leakage_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "competitor_columns": self.competitor_columns,
            "target_column": self.target_column,
            "market_component_columns": self.market_component_columns,
            "temporal_columns": self.temporal_columns,
            "leakage_warnings": self.leakage_warnings,
        }


def engineer_market_features(
    df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, FeatureMetadata]:
    """Create targets and market intelligence fields from competitor premium columns.

    The target and competitor-derived market components are useful outputs and report fields.
    They are excluded from the default model feature set to avoid target leakage.
    """

    engineered = df.copy()
    competitor_columns = detect_competitor_columns(engineered, config)
    competitor_prices = engineered[competitor_columns].apply(pd.to_numeric, errors="coerce")
    competitor_prices = competitor_prices.mask(competitor_prices <= 0)
    engineered[competitor_columns] = competitor_prices

    market_columns: list[str] = []
    top_ns = sorted(set(config.features.top_ns + [config.data.target.top_n]))
    for top_n in top_ns:
        column = f"avg_top_{top_n}_competitor_premium"
        engineered[column] = row_top_n_mean(competitor_prices, top_n)
        if config.data.target.missing_panel_policy == "complete":
            engineered.loc[competitor_prices.isna().any(axis=1), column] = np.nan
        market_columns.append(column)

    canonical_target = f"avg_top_{config.data.target.top_n}_competitor_premium"
    target_column = config.data.target.name
    target_values = {
        "avg_top_n": engineered[canonical_target],
        "min": competitor_prices.min(axis=1, skipna=True),
        "median": competitor_prices.median(axis=1, skipna=True),
        "softmin": row_softmin(
            competitor_prices, config.data.target.softmin_temperature
        ),
    }[config.data.target.aggregation]
    engineered[target_column] = target_values
    if config.data.target.missing_panel_policy == "complete":
        engineered.loc[competitor_prices.isna().any(axis=1), target_column] = np.nan
    if target_column not in market_columns:
        market_columns.append(target_column)

    if config.features.add_competitor_distribution:
        distribution = {
            "min_competitor_premium": competitor_prices.min(axis=1, skipna=True),
            "max_competitor_premium": competitor_prices.max(axis=1, skipna=True),
            "median_competitor_premium": competitor_prices.median(axis=1, skipna=True),
            "softmin_competitor_premium": row_softmin(
                competitor_prices, config.data.target.softmin_temperature
            ),
            "competitor_premium_std": competitor_prices.std(axis=1, skipna=True),
            "competitor_count": competitor_prices.notna().sum(axis=1),
        }
        for column, values in distribution.items():
            engineered[column] = values
            market_columns.append(column)
        target_median = engineered[target_column].dropna().median()
        engineered["market_price_index"] = (
            engineered[target_column] / target_median if pd.notna(target_median) else np.nan
        )
        market_columns.append("market_price_index")

    if config.features.add_relative_position and config.data.own_premium_column:
        own_column = config.data.own_premium_column
        if own_column in engineered.columns:
            own_premium = pd.to_numeric(engineered[own_column], errors="coerce")
            ratio_column = f"own_to_{target_column}_ratio"
            gap_column = f"price_gap_to_{target_column}"
            engineered[ratio_column] = safe_divide(own_premium, engineered[target_column])
            engineered[gap_column] = own_premium - engineered[target_column]
            engineered["own_to_min_competitor_ratio"] = safe_divide(
                own_premium, engineered.get("min_competitor_premium")
            )
            engineered["rank_own_premium"] = rank_own_price(own_premium, competitor_prices)
            engineered[f"top_{config.data.target.top_n}_indicator"] = (
                engineered["rank_own_premium"].le(config.data.target.top_n).astype("Int64")
            )
            market_columns.extend(
                [
                    ratio_column,
                    gap_column,
                    "own_to_min_competitor_ratio",
                    "rank_own_premium",
                    f"top_{config.data.target.top_n}_indicator",
                ]
            )

    segment_config = config.features.add_segment_aggressiveness
    if segment_config.enabled and segment_config.segment_columns:
        segment_columns = [column for column in segment_config.segment_columns if column in engineered.columns]
        if segment_columns:
            global_target_mean = engineered[target_column].mean()
            segment_column = "segment_market_aggressiveness"
            group_mean = engineered.groupby(segment_columns, dropna=False)[target_column].transform("mean")
            engineered[segment_column] = group_mean.fillna(global_target_mean) / global_target_mean
            market_columns.append(segment_column)

    temporal_columns = []
    if config.features.add_temporal_features:
        temporal_columns = add_temporal_features(engineered, config.data.date_column)

    leakage_warnings = build_leakage_warnings(engineered, config, target_column, competitor_columns)

    metadata = FeatureMetadata(
        competitor_columns=competitor_columns,
        target_column=target_column,
        market_component_columns=sorted(set(market_columns)),
        temporal_columns=temporal_columns,
        leakage_warnings=leakage_warnings,
    )
    return engineered, metadata


def row_top_n_mean(values: pd.DataFrame, top_n: int) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    valid_counts = np.sum(~np.isnan(arr), axis=1)
    sorted_values = np.sort(arr, axis=1)
    top_slice = sorted_values[:, :top_n]
    means = np.nansum(top_slice, axis=1) / np.maximum(
        np.sum(~np.isnan(top_slice), axis=1), 1
    )
    means[valid_counts < top_n] = np.nan
    return pd.Series(means, index=values.index)


def row_softmin(values: pd.DataFrame, temperature: float) -> pd.Series:
    """Smooth minimum, approaching the minimum as temperature approaches zero."""
    arr = values.to_numpy(dtype=float)
    all_missing = np.all(np.isnan(arr), axis=1)
    minimum = np.min(np.where(np.isnan(arr), np.inf, arr), axis=1)
    safe_minimum = np.where(all_missing, 0.0, minimum)
    shifted = np.exp(-(arr - safe_minimum[:, None]) / temperature)
    mean_shifted = np.nansum(shifted, axis=1) / np.maximum(
        np.sum(~np.isnan(shifted), axis=1), 1
    )
    result = safe_minimum - temperature * np.log(np.maximum(mean_shifted, 1e-12))
    result[all_missing] = np.nan
    return pd.Series(result, index=values.index)


def safe_divide(numerator: pd.Series, denominator: pd.Series | None) -> pd.Series:
    if denominator is None:
        return pd.Series(np.nan, index=numerator.index)
    denominator = pd.to_numeric(denominator, errors="coerce").replace(0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / denominator


def rank_own_price(own_premium: pd.Series, competitor_prices: pd.DataFrame) -> pd.Series:
    combined = competitor_prices.copy()
    combined["__own_premium__"] = pd.to_numeric(own_premium, errors="coerce")
    return combined.rank(axis=1, method="min", ascending=True)["__own_premium__"]


def add_temporal_features(df: pd.DataFrame, date_column: str) -> list[str]:
    dates = pd.to_datetime(df[date_column], errors="coerce")
    created = {
        "quote_year": dates.dt.year,
        "quote_month": dates.dt.month,
        "quote_quarter": dates.dt.quarter,
        "quote_weekofyear": dates.dt.isocalendar().week.astype("Int64"),
        "quote_dayofweek": dates.dt.dayofweek,
        "quote_time_days": (dates - pd.Timestamp("2000-01-01")).dt.days,
        "quote_month_sin": np.sin(2 * np.pi * dates.dt.month / 12),
        "quote_month_cos": np.cos(2 * np.pi * dates.dt.month / 12),
        "quote_dayofweek_sin": np.sin(2 * np.pi * dates.dt.dayofweek / 7),
        "quote_dayofweek_cos": np.cos(2 * np.pi * dates.dt.dayofweek / 7),
    }
    for column, values in created.items():
        df[column] = values
    return list(created.keys())


def select_model_features(
    df: pd.DataFrame, config: PipelineConfig, metadata: FeatureMetadata
) -> tuple[list[str], list[str], list[str]]:
    excluded = set(config.data.id_columns)
    excluded.update(config.data.leakage_columns)
    excluded.update(metadata.competitor_columns)
    excluded.update(metadata.market_component_columns)
    excluded.add(metadata.target_column)
    excluded.add(config.data.date_column)
    if config.data.own_premium_column:
        excluded.add(config.data.own_premium_column)
    if config.data.conversion_column:
        excluded.add(config.data.conversion_column)
    if config.data.weight_column:
        excluded.add(config.data.weight_column)

    categorical = [
        column
        for column in config.data.categorical_columns
        if column in df.columns and column not in excluded
    ]
    numeric = [
        column for column in config.data.numeric_columns if column in df.columns and column not in excluded
    ]
    numeric.extend(
        column for column in metadata.temporal_columns if column in df.columns and column not in excluded
    )

    deduped_numeric = []
    for column in numeric:
        if column not in deduped_numeric and column not in categorical:
            deduped_numeric.append(column)

    feature_columns = categorical + deduped_numeric
    if not feature_columns:
        raise ValueError("No model feature columns remain after leakage exclusions")
    return feature_columns, categorical, deduped_numeric


def build_leakage_warnings(
    df: pd.DataFrame,
    config: PipelineConfig,
    target_column: str,
    competitor_columns: list[str],
) -> list[str]:
    warnings: list[str] = []
    model_candidates = set(config.data.numeric_columns + config.data.categorical_columns)
    competitor_overlap = sorted(model_candidates.intersection(competitor_columns))
    if competitor_overlap:
        warnings.append(
            "Competitor premium columns were configured as model features and will be excluded: "
            + ", ".join(competitor_overlap)
        )

    if config.data.own_premium_column and config.data.own_premium_column in model_candidates:
        own = pd.to_numeric(df[config.data.own_premium_column], errors="coerce")
        target = pd.to_numeric(df[target_column], errors="coerce")
        valid_pairs = own.notna() & target.notna()
        if valid_pairs.sum() < 3:
            return warnings
        correlation = own[valid_pairs].corr(target[valid_pairs])
        if pd.notna(correlation) and abs(correlation) >= 0.85:
            warnings.append(
                "Own premium is highly correlated with the competitor target "
                f"(corr={correlation:.3f}); review circularity before deployment."
            )
    return warnings
