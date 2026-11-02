"""Evaluation helpers shared by the hybrid training script.

Every baseline consumes training-window statistics only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    truth = np.asarray(y_true, dtype="float64").reshape(-1)
    prediction = np.asarray(y_pred, dtype="float64").reshape(-1)
    if truth.size != prediction.size:
        raise ValueError("y_true and y_pred must have equal length")
    mask = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[mask]
    prediction = prediction[mask]
    n = int(truth.size)
    result: dict[str, Any] = {
        "n": n,
        "mae": None,
        "rmse": None,
        "r2": None,
        "mape_pct": None,
    }
    if n == 0:
        return result
    error = truth - prediction
    result["mae"] = float(np.mean(np.abs(error)))
    result["rmse"] = float(np.sqrt(np.mean(error**2)))
    if n >= 2 and np.ptp(truth) > 0:
        denominator = float(np.sum((truth - truth.mean()) ** 2))
        result["r2"] = float(1.0 - np.sum(error**2) / denominator)
    nonzero = np.abs(truth) > 1e-12
    if nonzero.any():
        result["mape_pct"] = float(
            100.0 * np.mean(np.abs(error[nonzero] / truth[nonzero]))
        )
    return result


def persistence_baseline(
    frame: pd.DataFrame, current_column: str = "current_soh_pct"
) -> np.ndarray:
    if current_column not in frame.columns:
        raise KeyError(f"current column {current_column!r} is not present")
    return pd.to_numeric(frame[current_column], errors="coerce").to_numpy(dtype="float64")


def mean_fade_baseline(
    train_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    *,
    entity_column: str,
    order_column: str,
    current_column: str = "current_soh_pct",
    target_column: str = "target_soh_pct",
) -> np.ndarray:
    missing = [
        column
        for column in (entity_column, order_column, current_column, target_column)
        if column not in train_frame.columns or column not in eval_frame.columns
    ]
    if missing:
        raise KeyError(f"mean-fade baseline is missing columns: {missing}")
    ordered = train_frame.sort_values(
        [entity_column, order_column], kind="mergesort"
    )
    fade = (
        pd.to_numeric(ordered[current_column], errors="coerce")
        - pd.to_numeric(ordered[target_column], errors="coerce")
    ).to_numpy(dtype="float64")
    fade = fade[np.isfinite(fade)]
    step = float(np.mean(fade)) if fade.size else 0.0
    current = pd.to_numeric(eval_frame[current_column], errors="coerce").to_numpy(
        dtype="float64"
    )
    return current - step


__all__ = ["mean_fade_baseline", "persistence_baseline", "regression_metrics"]
