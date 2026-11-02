"""Metrics for regression and classification tasks.

Regression metrics are reused from the existing hybrid evaluation module so
that every Member 2 report uses the same definitions. Classification metrics
include balanced accuracy because the failure and tier targets are imbalanced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from member2_modeling.hybrid.evaluation import regression_metrics


def regression_report(y_true: Any, y_pred: Any, residual_q90: float | None = None) -> dict[str, Any]:
    result = regression_metrics(y_true, y_pred)
    truth = np.asarray(y_true, dtype="float64")
    prediction = np.asarray(y_pred, dtype="float64")
    error = truth - prediction
    result["bias"] = float(np.mean(error)) if error.size else None
    result["median_abs_error"] = (
        float(np.median(np.abs(error))) if error.size else None
    )
    result["residual_q90"] = residual_q90
    return result


def classification_report(
    y_true: Any,
    y_pred: Any,
    *,
    probabilities: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    truth = np.asarray(y_true).astype(str)
    prediction = np.asarray(y_pred).astype(str)
    result: dict[str, Any] = {
        "n": int(truth.size),
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(truth, prediction, average="weighted", zero_division=0)
        ),
        "log_loss": None,
        "brier": None,
        "roc_auc": None,
        "per_class": {},
        "confusion_matrix": None,
    }
    label_list = labels if labels is not None else sorted(set(truth) | set(prediction))
    precision = precision_score(
        truth, prediction, labels=label_list, average=None, zero_division=0
    )
    recall = recall_score(
        truth, prediction, labels=label_list, average=None, zero_division=0
    )
    result["per_class"] = {
        label: {
            "n": int(np.sum(truth == label)),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
        }
        for index, label in enumerate(label_list)
    }
    matrix = confusion_matrix(truth, prediction, labels=label_list)
    result["confusion_matrix"] = {
        "labels": label_list,
        "matrix": matrix.astype(int).tolist(),
    }
    if probabilities is not None:
        probability = np.asarray(probabilities, dtype="float64")
        if probability.ndim == 2 and probability.shape[1] == 2:
            positive = probability[:, 1]
            truth_binary = (truth == label_list[1]).astype(int)
            result["brier"] = float(brier_score_loss(truth_binary, positive))
            if len(np.unique(truth_binary)) == 2:
                result["roc_auc"] = float(roc_auc_score(truth_binary, positive))
        try:
            result["log_loss"] = float(
                log_loss(truth, probability, labels=label_list)
            )
        except ValueError:
            result["log_loss"] = None
    return result


def positive_rate(y_true: Any, positive_label: str) -> float:
    truth = np.asarray(y_true).astype(str)
    return float(np.mean(truth == positive_label)) if truth.size else 0.0


__all__ = [
    "classification_report",
    "positive_rate",
    "regression_metrics",
    "regression_report",
]
