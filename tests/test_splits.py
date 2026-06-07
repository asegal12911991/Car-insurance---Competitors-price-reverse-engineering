from __future__ import annotations

import pandas as pd

from competitor_pricing_ai.config import DataConfig, PipelineConfig, ProjectConfig, SplitConfig
from competitor_pricing_ai.splits import time_based_split


def test_time_based_split_is_chronological() -> None:
    frame = pd.DataFrame(
        {
            "quote_date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "avg_top_3_competitor_premium": range(20),
        }
    )
    config = PipelineConfig(
        project=ProjectConfig(),
        data=DataConfig(
            input_path="unused.csv",
            date_column="quote_date",
            competitor_columns=["comp_a", "comp_b", "comp_c"],
        ),
        split=SplitConfig(validation_fraction=0.2, test_fraction=0.2),
    )

    split = time_based_split(frame, config)

    assert len(split.train) == 12
    assert len(split.validation) == 4
    assert len(split.test) == 4
    assert split.train["quote_date"].max() < split.validation["quote_date"].min()
    assert split.validation["quote_date"].max() < split.test["quote_date"].min()
