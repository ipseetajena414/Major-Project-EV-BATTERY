"""Permutation feature importance for the frozen snapshot models.

SHAP is not installed and dependency pins may not change, so importance is
estimated by permuting one transformed feature at a time on a bounded sample of
the frozen test split.  Permutation importance describes model reliance, not
causation, and correlated features share credit arbitrarily.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from member2_modeling.analytics.sequential import WORKSPACE

SNAPSHOT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "snapshot_ml"
REGISTRY_PATH = SNAPSHOT_DIR / "model_registry.json"
MAX_ROWS = 300
REPEATS = 1
TOP_K = 15

CAVEAT = (
    "Permutation importance measures model reliance on a feature in this test "
    "sample. It is not a causal effect and is unstable when features are "
    "correlated."
)


def _metric(task_type: str, truth: np.ndarray, prediction: np.ndarray) -> float:
    if task_type == "regression":
        return float(np.mean(np.abs(truth - prediction)))
    return float(np.mean(truth.astype(str) == prediction.astype(str)))


def _importance_one_task(
    task_id: str,
    task: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from member1_pipeline import loader

    artifact_dir = SNAPSHOT_DIR / task["artifact_dir"]
    predictions_path = artifact_dir / "test_predictions.parquet"
    if not predictions_path.is_file():
        return [], {"task_id": task_id, "status": "skipped", "reason": "no test predictions"}
    frame = loader.load_dataset(task["dataset_id"])
    test = pd.read_parquet(predictions_path)
    if "snapshot_row_id" not in test.columns:
        return [], {
            "task_id": task_id,
            "status": "skipped",
            "reason": "test predictions have no snapshot_row_id",
        }
    if "source_row_id" in frame.columns:
        merged = frame.merge(
            test[["snapshot_row_id", "observed"]],
            left_on="source_row_id",
            right_on="snapshot_row_id",
            how="inner",
        )
    else:
        selected = frame.iloc[test["snapshot_row_id"].to_numpy()].copy()
        selected["observed"] = test["observed"].to_numpy()
        merged = selected
    if merged.empty:
        return [], {"task_id": task_id, "status": "skipped", "reason": "no join"}

    features = list(task.get("numeric_features", [])) + list(
        task.get("categorical_features", [])
    )
    features = [column for column in features if column in merged.columns]
    if not features:
        return [], {"task_id": task_id, "status": "skipped", "reason": "no features"}

    sample = merged.sample(n=min(MAX_ROWS, len(merged)), random_state=0)
    pipeline = joblib.load(artifact_dir / "feature_pipeline.joblib")
    estimator = joblib.load(artifact_dir / "model.joblib")
    matrix = np.asarray(pipeline.transform(sample[features]))
    names = list(pipeline.feature_names_out())
    if matrix.shape[1] != len(names):
        names = [f"feature_{index}" for index in range(matrix.shape[1])]

    truth = (
        sample["observed"].to_numpy(dtype="float64")
        if task["task_type"] == "regression"
        else sample["observed"].astype(str).to_numpy()
    )
    baseline_prediction = estimator.predict(matrix)
    baseline = _metric(task["task_type"], truth, baseline_prediction)

    results: list[dict[str, Any]] = []
    for column in range(matrix.shape[1]):
        deltas = []
        for _ in range(REPEATS):
            shuffled = matrix[:, column].copy()
            rng.shuffle(shuffled)
            permuted = matrix.copy()
            permuted[:, column] = shuffled
            prediction = estimator.predict(permuted)
            permuted_metric = _metric(task["task_type"], truth, prediction)
            if task["task_type"] == "regression":
                deltas.append(permuted_metric - baseline)
            else:
                deltas.append(baseline - permuted_metric)
        results.append(
            {
                "task_id": task_id,
                "dataset_id": task["dataset_id"],
                "feature": names[column].split("=")[0],
                "transformed_feature": names[column],
                "importance": float(np.mean(deltas)),
                "metric": "mae_increase" if task["task_type"] == "regression" else "accuracy_drop",
                "n_rows": int(len(sample)),
                "repeats": REPEATS,
                "method": "permutation_transformed_feature",
                "caveat": CAVEAT,
            }
        )

    aggregated = (
        pd.DataFrame(results)
        .groupby(["task_id", "dataset_id", "feature", "metric", "method", "caveat"], as_index=False)
        .agg(importance=("importance", "sum"), n_rows=("n_rows", "max"), repeats=("repeats", "max"))
    )
    aggregated = aggregated.sort_values("importance", ascending=False).head(TOP_K)
    summary = {
        "task_id": task_id,
        "status": "ok",
        "baseline_metric": baseline,
        "metric": "mae" if task["task_type"] == "regression" else "accuracy",
        "n_rows": int(len(sample)),
        "features_evaluated": int(len(names)),
        "top_features": aggregated.head(5)[["feature", "importance"]].to_dict("records"),
        "shap_available": False,
        "method_note": (
            "SHAP is not installed and dependency pins are frozen, so "
            "permutation importance is used instead."
        ),
    }
    return aggregated.to_dict("records"), summary


def build_feature_importance() -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_id, task in registry["tasks"].items():
        try:
            task_rows, summary = _importance_one_task(task_id, task, rng)
        except Exception as exc:  # noqa: BLE001 - report, never fabricate importance
            task_rows, summary = [], {
                "task_id": task_id,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        rows.extend(task_rows)
        summaries.append(summary)
    frame = pd.DataFrame(rows)
    report = {
        "method": "permutation importance on transformed features",
        "shap_available": False,
        "dependency_pins_changed": False,
        "caveat": CAVEAT,
        "tasks": summaries,
    }
    return frame, report


__all__ = ["CAVEAT", "build_feature_importance"]
