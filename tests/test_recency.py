from __future__ import annotations

import pandas as pd

from competitor_pricing_ai.config import DataConfig, PipelineConfig, ProjectConfig
from competitor_pricing_ai.models import get_training_weight


def test_recency_weight_favors_recent_observations() -> None:
    frame = pd.DataFrame({
        "quote_date": pd.to_datetime(["2025-01-01", "2025-03-01"]),
    })
    config = PipelineConfig(
        project=ProjectConfig(),
        data=DataConfig(
            input_path="unused.csv",
            date_column="quote_date",
            competitor_columns=["a", "b", "c"],
        ),
    )
    weights = get_training_weight(frame, config, as_of_date="2025-03-01")
    assert weights[1] > weights[0]
