"""Generate synthetic car-insurance competitor pricing data for local testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/competitor_quotes.csv")
    parser.add_argument("--current-output", default=None)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = generate(args.rows, args.seed, drift=False)
    write_csv(data, args.output)
    print(f"Wrote {len(data):,} rows to {args.output}")

    if args.current_output:
        current = generate(max(1000, args.rows // 3), args.seed + 1, drift=True)
        write_csv(current, args.current_output)
        print(f"Wrote {len(current):,} current rows to {args.current_output}")
    return 0


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def generate(rows: int, seed: int, drift: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start_date = np.datetime64("2024-01-01") if not drift else np.datetime64("2025-07-01")
    dates = start_date + rng.integers(0, 520 if not drift else 90, size=rows).astype("timedelta64[D]")

    channels = rng.choice(["aggregator", "direct", "broker"], size=rows, p=[0.55, 0.30, 0.15])
    regions = rng.choice(["center", "north", "south", "jerusalem"], size=rows, p=[0.45, 0.20, 0.22, 0.13])
    vehicle_segments = rng.choice(["small", "family", "suv", "luxury"], size=rows, p=[0.25, 0.42, 0.25, 0.08])
    coverage = rng.choice(["tpl", "comprehensive"], size=rows, p=[0.20, 0.80])

    driver_age = np.clip(rng.normal(42, 13, size=rows), 18, 85)
    vehicle_age = np.clip(rng.gamma(2.0, 3.0, size=rows), 0, 20)
    claim_count = rng.poisson(0.35, size=rows)

    region_factor = map_values(regions, {"center": 1.06, "north": 0.96, "south": 1.00, "jerusalem": 1.10})
    channel_factor = map_values(channels, {"aggregator": 0.97, "direct": 1.03, "broker": 1.01})
    vehicle_factor = map_values(vehicle_segments, {"small": 0.86, "family": 1.00, "suv": 1.14, "luxury": 1.42})
    coverage_factor = map_values(coverage, {"tpl": 0.58, "comprehensive": 1.0})
    young_driver = np.maximum(0, 30 - driver_age) * 7.0
    old_vehicle = vehicle_age * 8.5
    claims = claim_count * 85
    seasonality = 18 * np.sin((pd.to_datetime(dates).month.to_numpy() / 12) * 2 * np.pi)
    trend_days = (dates - dates.min()).astype("timedelta64[D]").astype(float)
    trend = trend_days * (0.035 if not drift else 0.055)

    market_base = (
        620 * region_factor * channel_factor * vehicle_factor * coverage_factor
        + young_driver
        + old_vehicle
        + claims
        + seasonality
        + trend
    )
    if drift:
        market_base *= np.where(regions == "center", 1.05, 1.015)

    competitor_offsets = {
        "comp_a_premium": -32,
        "comp_b_premium": -12,
        "comp_c_premium": 8,
        "comp_d_premium": 28,
        "comp_e_premium": 52,
    }
    data = {
        "quote_id": [f"Q{seed}-{i:06d}" for i in range(rows)],
        "quote_date": pd.to_datetime(dates).strftime("%Y-%m-%d"),
        "channel": channels,
        "region": regions,
        "vehicle_segment": vehicle_segments,
        "coverage_type": coverage,
        "driver_age": np.round(driver_age, 1),
        "vehicle_age": np.round(vehicle_age, 1),
        "claim_count_3y": claim_count,
    }

    for column, offset in competitor_offsets.items():
        noise = rng.normal(0, 24, size=rows)
        data[column] = np.round(np.maximum(120, market_base + offset + noise), 2)

    top_three = np.sort(np.column_stack([data[column] for column in competitor_offsets]), axis=1)[:, :3]
    avg_top_three = top_three.mean(axis=1)
    own_markup = map_values(channels, {"aggregator": 1.00, "direct": 1.04, "broker": 1.02})
    own_noise = rng.normal(0, 30, size=rows)
    data["own_premium"] = np.round(avg_top_three * own_markup + 18 + own_noise, 2)
    ratio = data["own_premium"] / avg_top_three
    conversion_logit = 2.2 - 4.0 * (ratio - 1.0) - 0.18 * claim_count
    conversion_probability = 1 / (1 + np.exp(-conversion_logit))
    data["converted"] = rng.binomial(1, conversion_probability)

    frame = pd.DataFrame(data)
    missing_mask = rng.random((rows, len(competitor_offsets))) < 0.015
    for idx, column in enumerate(competitor_offsets):
        frame.loc[missing_mask[:, idx], column] = np.nan
    return frame.sort_values("quote_date").reset_index(drop=True)


def map_values(values: np.ndarray, mapping: dict[str, float]) -> np.ndarray:
    return np.array([mapping[value] for value in values])


if __name__ == "__main__":
    raise SystemExit(main())
