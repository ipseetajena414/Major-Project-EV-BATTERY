"""Model comparison on frozen test results and RUL validation folds.

Snapshot test metrics are read from the frozen model registry and are never
recomputed.  RUL candidates come from the leave-one-cell-out study.  A model is
only marked selected if the recorded selection protocol chose it; nothing here
re-tunes against test data.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from member2_modeling.analytics.sequential import WORKSPACE
from member2_modeling.rul.service import METRICS_PATH

SNAPSHOT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "snapshot_ml"
REGISTRY_PATH = SNAPSHOT_DIR / "model_registry.json"


def _snapshot_rows(registry: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for task_id, task in registry["tasks"].items():
        metrics = task.get("test_metrics", {})
        selected = task.get("selected_model", {}).get("name", "unknown")
        task_type = task["task_type"]
        if task_type == "regression":
            primary, secondary = "mae", "rmse"
            selected_value = metrics.get("mae")
            selected_secondary = metrics.get("rmse")
            baseline_name, baseline_value, baseline_secondary = None, None, None
            for name, block in task.get("baselines", {}).items():
                value = block.get("mae")
                if value is None:
                    continue
                if baseline_value is None or value < baseline_value:
                    baseline_name = name
                    baseline_value = value
                    baseline_secondary = block.get("rmse")
        else:
            primary, secondary = "balanced_accuracy", "macro_f1"
            selected_value = metrics.get("balanced_accuracy")
            selected_secondary = metrics.get("macro_f1")
            baseline_name, baseline_value, baseline_secondary = None, None, None
            for name, block in task.get("baselines", {}).items():
                value = block.get("balanced_accuracy")
                if value is None:
                    continue
                if baseline_value is None or value > baseline_value:
                    baseline_name = name
                    baseline_value = value
                    baseline_secondary = block.get("macro_f1")

        rows.append(
            {
                "domain": "snapshot",
                "task_id": task_id,
                "dataset_id": task["dataset_id"],
                "task_type": task_type,
                "model_id": selected,
                "model_role": "selected",
                "primary_metric": primary,
                "primary_value": selected_value,
                "secondary_metric": secondary,
                "secondary_value": selected_secondary,
                "selected": True,
                "frozen_test": True,
                "readiness": task.get("status"),
                "note": "selected on the validation split; frozen test result",
            }
        )
        if baseline_name is not None:
            rows.append(
                {
                    "domain": "snapshot",
                    "task_id": task_id,
                    "dataset_id": task["dataset_id"],
                    "task_type": task_type,
                    "model_id": baseline_name,
                    "model_role": "best_baseline",
                    "primary_metric": primary,
                    "primary_value": baseline_value,
                    "secondary_metric": secondary,
                    "secondary_value": baseline_secondary,
                    "selected": False,
                    "frozen_test": True,
                    "readiness": task.get("status"),
                    "note": "best simple baseline on the same frozen test rows",
                }
            )
        delta = (
            None
            if selected_value is None or baseline_value is None
            else (
                float(selected_value - baseline_value)
                if task_type == "regression"
                else float(selected_value - baseline_value)
            )
        )
        summaries.append(
            {
                "task_id": task_id,
                "dataset_id": task["dataset_id"],
                "task_type": task_type,
                "selected_model": selected,
                "selected_primary": selected_value,
                "primary_metric": primary,
                "best_baseline": baseline_name,
                "best_baseline_primary": baseline_value,
                "delta_vs_baseline": delta,
                "delta_direction": (
                    "lower_is_better" if task_type == "regression" else "higher_is_better"
                ),
                "selection_protocol": "validation split only",
                "test_usage": "single evaluation after configuration freeze",
            }
        )
    return rows, summaries


def _rul_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    selection = metrics["selection"]
    rows: list[dict[str, Any]] = []
    for model_id, block in metrics["candidates"].items():
        rows.append(
            {
                "domain": "rul",
                "task_id": "rul_cycles_to_eol",
                "dataset_id": "NASA_Ames_PCoE",
                "task_type": "regression",
                "model_id": model_id,
                "model_role": (
                    "selected" if model_id == selection["selected_model"] else "candidate"
                ),
                "primary_metric": "pre_threshold_mae_cycles",
                "primary_value": block["pre_threshold"]["mae"],
                "secondary_metric": "pre_threshold_rmse_cycles",
                "secondary_value": block["pre_threshold"]["rmse"],
                "selected": model_id == selection["selected_model"],
                "frozen_test": False,
                "readiness": "EXPERIMENTAL",
                "note": (
                    f"LOCO scored rows {block['n_scored']}/{block['n_records']}; "
                    f"90% interval coverage {block['coverage']}"
                ),
            }
        )
    summary = {
        "task_id": "rul_cycles_to_eol",
        "selected_model": selection["selected_model"],
        "best_single_model": selection.get("best_single_model"),
        "selection_protocol": selection["selection_protocol"],
        "selection_metric_mae_cycles": selection["selection_metric"],
        "n_cells": metrics["n_cells"],
        "note": (
            "RUL has no protected test split; leave-one-cell-out is the only "
            "independent evidence and it covers three cells."
        ),
    }
    return rows, summary


def build_model_comparison() -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    snapshot_rows, snapshot_summary = _snapshot_rows(registry)
    rul_rows, rul_summary = _rul_rows()
    frame = pd.DataFrame(snapshot_rows + rul_rows)
    report = {
        "registry_version": registry.get("registry_version"),
        "snapshot": snapshot_summary,
        "rul": rul_summary,
        "policy": {
            "snapshot_test": "frozen; never used for selection",
            "rul_test": "none; leave-one-cell-out only",
            "range": "calculated estimate; no measured range target",
        },
        "limitations": [
            "Snapshot improvements over baselines are not evidence of causal "
            "battery behaviour.",
            "RUL model ranking is based on 59-78 scored origins from three cells.",
        ],
    }
    return frame, report


__all__ = ["build_model_comparison"]
