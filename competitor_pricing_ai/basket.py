"""Fixed reference basket for mix-adjusted competitor price indexing.

Why: observed monthly data mixes genuine rate changes with risk-mix shift
(different profiles submitted each month). Applying the trained model to a
*fixed* synthetic portfolio each month isolates true rate movement.
"""

from __future__ import annotations

from itertools import product as iproduct
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from competitor_pricing_ai.features import add_temporal_features
from competitor_pricing_ai.models import predict_with_bundle

if TYPE_CHECKING:
    from competitor_pricing_ai.config import PipelineConfig
    from competitor_pricing_ai.models import ModelTrainingResult
    from competitor_pricing_ai.splits import SplitResult


def build_reference_basket(
    data: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns_base: list[str],
    max_size: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a fixed grid of profiles covering the rating space.

    Categoricals: every unique value seen in training data.
    Numerics: [p25, p50, p75] of the training distribution.
    Cross-product is sampled to max_size when larger.
    """
    cat_levels: dict[str, list] = {
        col: sorted(data[col].dropna().unique().tolist())
        for col in categorical_columns
        if col in data.columns
    }
    num_grids: dict[str, list] = {}
    for col in numeric_columns_base:
        if col not in data.columns:
            continue
        arr = data[col].dropna().to_numpy()
        # deduplicate so sparse distributions don't explode the grid
        num_grids[col] = sorted(
            set(round(float(np.percentile(arr, p)), 1) for p in [25, 50, 75])
        )

    all_keys = list(cat_levels) + list(num_grids)
    all_values = list(cat_levels.values()) + list(num_grids.values())
    if not all_keys:
        return pd.DataFrame()

    rows = [dict(zip(all_keys, combo)) for combo in iproduct(*all_values)]
    basket = pd.DataFrame(rows)

    if len(basket) > max_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(basket), size=max_size, replace=False)
        basket = basket.iloc[sorted(idx)].reset_index(drop=True)

    return basket


def compute_basket_index(
    basket: pd.DataFrame,
    bundle: dict[str, Any],
    months: pd.DatetimeIndex,
    date_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Apply model to the fixed basket at every month in *months*.

    Returns one row per month with mean/p25/p75 of predicted prices.
    Because the basket profiles never change, any trend reflects rate
    movements rather than mix shift.
    """
    records = []
    for dt in months:
        monthly = basket.copy()
        monthly[date_column] = dt
        add_temporal_features(monthly, date_column)

        preds = predict_with_bundle(bundle, monthly)
        preds = np.maximum(preds, 0.0)

        records.append(
            {
                "quote_month": dt.to_period("M").to_timestamp(),
                "mean_prediction": float(np.mean(preds)),
                "p25_prediction": float(np.percentile(preds, 25)),
                "p75_prediction": float(np.percentile(preds, 75)),
                "n_profiles": len(preds),
            }
        )

    return pd.DataFrame(records)


def generate_basket_artefacts(
    training_result: "ModelTrainingResult",
    split: "SplitResult",
    config: "PipelineConfig",
    output_dir: Path,
) -> dict[str, str]:
    """Build basket + index, save CSVs, return artefact paths."""
    import joblib

    # H2O saves a directory, not a joblib bundle — skip gracefully
    if training_result.backend == "h2o":
        return {}

    bundle = joblib.load(training_result.model_path)

    basket = build_reference_basket(
        data=split.train,
        categorical_columns=training_result.categorical_columns,
        numeric_columns_base=config.data.numeric_columns,
        random_state=config.project.random_seed,
    )
    if basket.empty:
        return {}

    all_dates = pd.to_datetime(
        pd.concat([split.train, split.validation, split.test])[config.data.date_column],
        errors="coerce",
    ).dropna()
    months = pd.date_range(
        start=all_dates.min().to_period("M").to_timestamp(),
        end=all_dates.max().to_period("M").to_timestamp(),
        freq="MS",
    )

    index_df = compute_basket_index(
        basket=basket,
        bundle=bundle,
        months=months,
        date_column=config.data.date_column,
        feature_columns=training_result.feature_columns,
    )

    basket_path = output_dir / "reference_basket.csv"
    index_path = output_dir / "reference_basket_index.csv"
    basket.to_csv(basket_path, index=False)
    index_df.to_csv(index_path, index=False)

    return {
        "reference_basket": str(basket_path),
        "reference_basket_index": str(index_path),
    }
