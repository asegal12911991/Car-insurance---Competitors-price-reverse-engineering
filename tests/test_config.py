from __future__ import annotations

from competitor_pricing_ai.config import load_config


def test_example_config_loads() -> None:
    config = load_config("configs/config.example.yml")

    assert config.project.name == "car_insurance_competitor_pricing"
    assert config.data.target.top_n == 3
    assert config.model.backend == "sklearn"
