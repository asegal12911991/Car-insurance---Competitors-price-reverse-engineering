"""Model evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import d2_tweedie_score, mean_absolute_error, r2_score

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - compatibility for older sklearn
    root_mean_squared_error = None


def regression_metrics(y_true, y_pred, weights=None) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not mask.any():
        raise ValueError("No finite values are available for metric calculation")

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    w = np.asarray(weights, dtype=float)[mask] if weights is not None else None

    if root_mean_squared_error is None:
        rmse = float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w)))
    else:
        rmse = float(root_mean_squared_error(y_true, y_pred, sample_weight=w))

    mae = float(mean_absolute_error(y_true, y_pred, sample_weight=w))

    nonzero = y_true != 0
    mape = (
        float(
            np.average(
                np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]),
                weights=w[nonzero] if w is not None else None,
            )
            * 100
        )
        if nonzero.any()
        else float("nan")
    )

    mean_bias = float(np.average(y_pred - y_true, weights=w))
    mean_bias_pct = (
        float(
            np.average(
                (y_pred[nonzero] - y_true[nonzero]) / y_true[nonzero],
                weights=w[nonzero] if w is not None else None,
            )
            * 100
        )
        if nonzero.any()
        else float("nan")
    )

    # RMSLE: proportional errors, natural for multiplicative pricing models
    rmsle = float(
        np.sqrt(
            np.average(
                (np.log1p(np.maximum(y_pred, 0)) - np.log1p(np.maximum(y_true, 0))) ** 2,
                weights=w,
            )
        )
    )

    # D² (Gamma deviance) — proper goodness-of-fit for positive right-skewed prices.
    # power=2 selects the Gamma family. Clamp predictions to avoid log(0).
    try:
        d2 = float(d2_tweedie_score(y_true, np.maximum(y_pred, 1e-6), power=2))
    except Exception:
        d2 = float("nan")

    return {
        "d2": d2,
        "r2": float(r2_score(y_true, y_pred, sample_weight=w)),  # kept for backward compatibility
        "rmse": rmse,
        "rmsle": rmsle,
        "mae": mae,
        "mape": mape,
        "mean_bias": mean_bias,
        "mean_bias_pct": mean_bias_pct,
        "gini": gini_coefficient(y_true, y_pred),
        "n": int(mask.sum()),
    }


def gini_coefficient(y_true, y_pred) -> float:
    """Actuarial Gini: how well predicted prices discriminate actual price levels.

    Ranges 0 (random) to 1 (perfect rank ordering). Computed as
    2 * AUC(Lorenz curve) - 1 where observations are ordered by predicted value.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or y_true.sum() == 0:
        return float("nan")
    order = np.argsort(y_pred)
    sorted_actual = y_true[order]
    cumulative_actual = np.cumsum(sorted_actual) / sorted_actual.sum()
    cumulative_pop = np.arange(1, len(sorted_actual) + 1) / len(sorted_actual)
    lorenz_area = float(np.trapz(cumulative_actual, cumulative_pop))
    # Lorenz curve for a good model bows BELOW the 45° line → area < 0.5 → 1 - 2*area > 0
    return round(1 - 2 * lorenz_area, 6)


def lift_table(y_true, y_pred, n_quantiles: int = 10, weights=None) -> list[dict]:
    """Decile lift table — standard actuarial model validation output.

    Sorts by predicted value into n_quantiles buckets and reports mean
    predicted vs mean actual. pred_actual_ratio near 1.0 indicates
    well-calibrated predictions within that pricing band.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    w = np.asarray(weights, dtype=float)[mask] if weights is not None else np.ones(len(y_true))

    order = np.argsort(y_pred)
    y_true, y_pred, w = y_true[order], y_pred[order], w[order]

    edges = np.percentile(y_pred, np.linspace(0, 100, n_quantiles + 1))
    bucket_ids = np.clip(np.digitize(y_pred, edges[1:-1]), 0, n_quantiles - 1)

    overall_mean = float(np.average(y_true, weights=w))
    rows = []
    for i in range(n_quantiles):
        idx = bucket_ids == i
        if not idx.any():
            continue
        wi = w[idx]
        mean_pred = float(np.average(y_pred[idx], weights=wi))
        mean_actual = float(np.average(y_true[idx], weights=wi))
        rows.append({
            "quantile": i + 1,
            "n": int(idx.sum()),
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_actual, 4),
            "lift": round(mean_actual / overall_mean, 4) if overall_mean else float("nan"),
            "pred_actual_ratio": round(mean_pred / mean_actual, 4) if mean_actual else float("nan"),
        })
    return rows
