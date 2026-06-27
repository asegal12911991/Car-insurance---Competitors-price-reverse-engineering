from __future__ import annotations

import numpy as np

from competitor_pricing_ai.metrics import gini_coefficient


def test_normalized_gini_is_one_for_perfect_ordering() -> None:
    actual = np.array([100.0, 110.0, 130.0, 200.0])
    assert gini_coefficient(actual, actual) == 1.0


def test_normalized_gini_is_invariant_to_prediction_scale() -> None:
    actual = np.array([100.0, 110.0, 130.0, 200.0])
    assert gini_coefficient(actual, actual * 7) == 1.0
