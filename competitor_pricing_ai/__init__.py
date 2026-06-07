"""Competitor pricing intelligence toolkit."""

from competitor_pricing_ai.config import load_config
from competitor_pricing_ai.pipeline import run_training_pipeline

__all__ = ["load_config", "run_training_pipeline"]

__version__ = "0.1.0"
