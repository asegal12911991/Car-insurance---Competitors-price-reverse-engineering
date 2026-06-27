from __future__ import annotations

import pytest

from competitor_pricing_ai.config import ConfigError, load_config, validate_config


def test_example_config_loads() -> None:
    config = load_config("configs/config.example.yml")

    assert config.project.name == "car_insurance_competitor_pricing"
    assert config.data.target.top_n == 3
    assert config.model.backend == "sklearn"


def test_own_premium_cannot_be_a_competitor_model_feature() -> None:
    config = load_config("configs/config.example.yml")
    config.data.numeric_columns.append(config.data.own_premium_column)
    with pytest.raises(ConfigError, match="cannot be competitor-model features"):
        validate_config(config)
