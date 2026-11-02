"""Pre-threshold and post-threshold RUL evaluation.

Post-threshold rows have RUL == 0. Pooling them with pre-threshold rows hides
large pre-EOL errors behind trivial zeros, so the two regimes are reported
separately and are never pooled into one error metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

BUCKET_LABELS = ("1-5", "6-10", "11-30", "31+")


class RULEvaluationError(ValueError):
    """Raised when RUL evaluation inputs are not comparable."""


def _metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    if len(truth) == 0:
        return {"n": 0, "mae": None, "rmse": None, "bias": None}
    error = predictions - truth
    return {
        "n": int(len(truth)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "bias": float(np.mean(error)),
    }


def _bucket_label(predicted: float) -> str | None:
    if not np.isfinite(predicted) or predicted < 1:
        return None
    if predicted <= 5:
        return "1-5"
    if predicted <= 10:
        return "6-10"
    if predicted <= 30:
        return "11-30"
    return "31+"


def evaluate_rul(
    truth: Any,
    predictions: Any,
    *,
    near_eol_threshold: float = 5.0,
) -> dict[str, Any]:
    """Evaluate RUL predictions with pre/post-threshold separation.

    ``near_eol`` is the subset of pre-threshold rows whose true RUL is at or
    below ``near_eol_threshold``. Horizon buckets stratify pre-threshold rows
    by predicted RUL so calibration can be inspected without mixing in
    post-threshold zeros.
    """
    truth_array = np.asarray(truth, dtype="float64").reshape(-1)
    prediction_array = np.asarray(predictions, dtype="float64").reshape(-1)
    if truth_array.shape != prediction_array.shape:
        raise RULEvaluationError(
            f"truth and predictions differ in length: "
            f"{truth_array.shape[0]} vs {prediction_array.shape[0]}"
        )
    valid = np.isfinite(truth_array) & np.isfinite(prediction_array)
    truth_array = truth_array[valid]
    prediction_array = prediction_array[valid]

    pre = truth_array > 0
    post = truth_array <= 0
    near = pre & (truth_array <= float(near_eol_threshold))

    buckets: dict[str, dict[str, Any]] = {
        label: {"n": 0, "mae": None, "rmse": None, "bias": None}
        for label in BUCKET_LABELS
    }
    grouped: dict[str, list[int]] = {label: [] for label in BUCKET_LABELS}
    for index in np.flatnonzero(pre):
        label = _bucket_label(float(prediction_array[index]))
        if label is not None:
            grouped[label].append(int(index))
    for label, indices in grouped.items():
        selected = np.asarray(indices, dtype="int64")
        buckets[label] = _metrics(truth_array[selected], prediction_array[selected])

    return {
        "n": int(valid.sum()),
        "near_eol_threshold": float(near_eol_threshold),
        "counts": {
            "pre_threshold": int(pre.sum()),
            "post_threshold": int(post.sum()),
            "near_eol": int(near.sum()),
        },
        "overall": _metrics(truth_array, prediction_array),
        "pre_threshold": _metrics(truth_array[pre], prediction_array[pre]),
        "post_threshold": _metrics(truth_array[post], prediction_array[post]),
        "near_eol": _metrics(truth_array[near], prediction_array[near]),
        "buckets": buckets,
        "pooling_rule": (
            "Pre-threshold (RUL > 0) and post-threshold (RUL == 0) rows are "
            "reported separately and are never pooled into a single error metric."
        ),
    }


__all__ = ["BUCKET_LABELS", "RULEvaluationError", "evaluate_rul"]
