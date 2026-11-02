"""Uncertainty reporting for snapshot, range and RUL outputs.

Regression intervals are split residual intervals (q90 of validation residuals)
and their empirical coverage is measured on the frozen test split.  Classification
probabilities are explicitly uncalibrated.  RUL intervals are cross-fitted
conformal absolute-residual intervals with measured coverage on held-out cells.
Nothing here is presented as a guaranteed or calibrated probability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from member2_modeling.analytics.sequential import WORKSPACE
from member2_modeling.rul.service import METRICS_PATH

SNAPSHOT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "snapshot_ml"
REGISTRY_PATH = SNAPSHOT_DIR / "model_registry.json"


def _registry() -> dict[str, Any]:
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def build_uncertainty_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    registry = _registry()
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    for task_id, task in registry["tasks"].items():
        artifact_dir = SNAPSHOT_DIR / task["artifact_dir"]
        predictions_path = artifact_dir / "test_predictions.parquet"
        task_type = task["task_type"]
        if task_type == "regression":
            frame = pd.read_parquet(predictions_path)
            coverage = width = None
            if {"interval_90_low", "interval_90_high"}.issubset(frame.columns):
                covered = (frame["observed"] >= frame["interval_90_low"]) & (
                    frame["observed"] <= frame["interval_90_high"]
                )
                coverage = float(covered.mean())
                width = float(
                    (frame["interval_90_high"] - frame["interval_90_low"]).mean()
                )
            uncertainty = task.get("uncertainty", {})
            rows.append(
                {
                    "domain": "snapshot",
                    "task_id": task_id,
                    "dataset_id": task["dataset_id"],
                    "task_type": task_type,
                    "method": uncertainty.get("kind"),
                    "nominal_coverage": 0.90,
                    "empirical_coverage": coverage,
                    "mean_interval_width": width,
                    "calibration_status": uncertainty.get("calibration_status"),
                    "n": int(len(frame)),
                    "note": (
                        "Split residual interval; coverage is measured on the "
                        "frozen test split and is not guaranteed for new data."
                    ),
                }
            )
            details[task_id] = {
                "uncertainty": uncertainty,
                "empirical_coverage": coverage,
                "mean_interval_width": width,
                "n_test": int(len(frame)),
            }
        else:
            metrics = task.get("test_metrics", {})
            uncertainty = task.get("uncertainty", {})
            rows.append(
                {
                    "domain": "snapshot",
                    "task_id": task_id,
                    "dataset_id": task["dataset_id"],
                    "task_type": task_type,
                    "method": uncertainty.get("kind"),
                    "nominal_coverage": None,
                    "empirical_coverage": None,
                    "mean_interval_width": None,
                    "calibration_status": uncertainty.get(
                        "calibration_status", "uncalibrated_class_probabilities"
                    ),
                    "n": int(metrics.get("n", 0)),
                    "note": (
                        "Class probabilities are not calibrated; do not interpret "
                        "them as calibrated probabilities."
                    ),
                }
            )
            details[task_id] = {
                "uncertainty": uncertainty,
                "brier": metrics.get("brier"),
                "log_loss": metrics.get("log_loss"),
                "roc_auc": metrics.get("roc_auc"),
                "n_test": int(metrics.get("n", 0)),
                "calibration_status": "uncalibrated_class_probabilities",
            }

    rul_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for model_id, block in rul_metrics["candidates"].items():
        rows.append(
            {
                "domain": "rul",
                "task_id": "rul_cycles_to_eol",
                "dataset_id": "NASA_Ames_PCoE",
                "task_type": "regression",
                "method": "cross_fitted_conformal_abs_residual_q90",
                "nominal_coverage": 0.90,
                "empirical_coverage": block.get("coverage"),
                "mean_interval_width": block.get("mean_interval_width"),
                "calibration_status": "empirical_on_held_out_cells",
                "n": block.get("n_scored", 0),
                "note": (
                    "Coverage is measured on leave-one-cell-out origins from "
                    "three cells; it is not a guarantee for new cells."
                ),
            }
        )

    frame = pd.DataFrame(rows)
    report = {
        "generated_from": {
            "snapshot_registry": str(REGISTRY_PATH),
            "rul_metrics": str(METRICS_PATH),
        },
        "snapshot_regression": {
            "method": "validation residual q90 around the point estimate",
            "nominal_coverage": 0.90,
            "calibration_status": "empirical_validation_residuals",
            "warning": (
                "Empirical coverage on the frozen test split does not guarantee "
                "coverage for new vehicles or future conditions."
            ),
        },
        "snapshot_classification": {
            "method": "raw model class probabilities",
            "calibration_status": "uncalibrated_class_probabilities",
            "warning": (
                "Probabilities are uncalibrated confidence scores. They are not "
                "calibrated probabilities and have no coverage guarantee."
            ),
        },
        "rul": {
            "method": "cross-fitted conformal absolute-residual q90",
            "nominal_coverage": 0.90,
            "calibration_status": "empirical_on_held_out_cells",
            "warning": (
                "Only three NMC cells are available; coverage is unstable and "
                "no VALIDATED status is claimed."
            ),
        },
        "range": {
            "method": "fixed consumption scenarios (0.15, 0.18, 0.22 kWh/km)",
            "calibration_status": "assumption_scenario_spread",
            "validated_accuracy": False,
            "warning": (
                "Scenario spread reflects assumed consumption, not statistical "
                "uncertainty, and the dataset has no measured range target."
            ),
        },
        "tasks": details,
    }
    return frame, report


__all__ = ["build_uncertainty_summary"]
