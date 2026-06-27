"""Command-line interface."""

from __future__ import annotations

import argparse
import sys

from competitor_pricing_ai.config import ConfigError, load_config
from competitor_pricing_ai.monitoring import run_monitoring
from competitor_pricing_ai.pipeline import run_training_pipeline
from competitor_pricing_ai.scoring import score_market_anchor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="competitor-pricing-ai",
        description="Competitor pricing intelligence pipeline for car insurance pricing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run the full training pipeline")
    train_parser.add_argument("--config", required=True, help="Path to YAML configuration")

    validate_parser = subparsers.add_parser("validate-config", help="Validate a YAML configuration")
    validate_parser.add_argument("--config", required=True, help="Path to YAML configuration")

    monitor_parser = subparsers.add_parser("monitor", help="Run drift and performance monitoring")
    monitor_parser.add_argument("--config", required=True, help="Path to YAML configuration")

    score_parser = subparsers.add_parser(
        "score", help="Create frozen market-anchor features for a quote batch"
    )
    score_parser.add_argument("--config", required=True, help="Path to YAML configuration")
    score_parser.add_argument("--input", required=True, help="CSV, Parquet, or Excel quote file")
    score_parser.add_argument("--output", required=True, help="Destination CSV")
    score_parser.add_argument("--model", default=None, help="Optional model.joblib path")

    args = parser.parse_args(argv)

    try:
        if args.command == "validate-config":
            config = load_config(args.config)
            print(f"Configuration is valid: {config.project.name}")
            return 0

        if args.command == "train":
            result = run_training_pipeline(args.config)
            print(f"Training complete. Output directory: {result.output_dir}")
            t = result.metrics["test"]
            print(
                "Test metrics: "
                f"D²={t.get('d2', float('nan')):.4f}, "
                f"Gini={t.get('gini', float('nan')):.4f}, "
                f"MAPE={t.get('mape', float('nan')):.2f}%, "
                f"Bias%={t.get('mean_bias_pct', float('nan')):+.2f}%, "
                f"RMSE={t['rmse']:.2f}"
            )
            print(f"QA overall passed: {result.qa_checklist['overall_passed']}")
            return 0

        if args.command == "monitor":
            metrics = run_monitoring(args.config)
            print("Monitoring complete.")
            print(
                "Retrain recommended: "
                f"{metrics['refresh_recommendation']['retrain_recommended']}"
            )
            return 0

        if args.command == "score":
            path = score_market_anchor(
                args.config, args.input, args.output, model_path=args.model
            )
            print(f"Scoring complete. Output: {path}")
            return 0

    except (ConfigError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
