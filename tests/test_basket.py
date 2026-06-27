from __future__ import annotations

import pandas as pd

from competitor_pricing_ai.basket import build_reference_basket, compute_basket_index


def test_build_reference_basket_is_a_cross_product_of_levels_and_quantiles() -> None:
    data = pd.DataFrame(
        {
            "region": ["north", "north", "south", "south"],
            "driver_age": [20, 30, 40, 50],
        }
    )
    basket = build_reference_basket(
        data, categorical_columns=["region"], numeric_columns_base=["driver_age"]
    )

    assert set(basket["region"].unique()) == {"north", "south"}
    # 2 categorical levels x up to 3 numeric quantiles
    assert len(basket) == 2 * basket["driver_age"].nunique()
    assert len(basket) <= 6


def test_build_reference_basket_is_empty_when_no_configured_columns_exist() -> None:
    data = pd.DataFrame({"unrelated": [1, 2, 3]})
    basket = build_reference_basket(data, categorical_columns=["region"], numeric_columns_base=["driver_age"])
    assert basket.empty


def test_build_reference_basket_caps_at_max_size() -> None:
    data = pd.DataFrame(
        {
            "region": [f"region_{i}" for i in range(10)],
            "vehicle_class": [f"class_{i}" for i in range(10)],
        }
    )
    basket = build_reference_basket(
        data, categorical_columns=["region", "vehicle_class"], numeric_columns_base=[], max_size=20
    )
    assert len(basket) == 20


class _ConstantModel:
    """Stand-in model whose prediction is a fixed offset from a numeric feature."""

    def predict(self, X: pd.DataFrame) -> "pd.Series":
        return X["driver_age"].to_numpy() * 10.0


def test_compute_basket_index_returns_one_row_per_month_with_fixed_basket() -> None:
    basket = pd.DataFrame({"driver_age": [20, 30, 40]})
    bundle = {
        "model": _ConstantModel(),
        "backend": "sklearn",
        "feature_columns": ["driver_age"],
    }
    months = pd.date_range("2025-01-01", periods=3, freq="MS")

    index_df = compute_basket_index(
        basket, bundle, months, date_column="quote_date", feature_columns=["driver_age"]
    )

    assert len(index_df) == 3
    # Same fixed basket every month -> identical predictions -> no spurious "trend"
    assert index_df["mean_prediction"].nunique() == 1
    assert index_df["n_profiles"].eq(3).all()
