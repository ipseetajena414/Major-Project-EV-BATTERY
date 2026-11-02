"""Inference interface for the registered snapshot models.

The API validates task, dataset, feature, chemistry and numeric-range
compatibility before serving a prediction. It never substitutes another
dataset's model and never hides an OOD or unsupported-chemistry input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .registry import load_registry
from .training import MODEL_VERSION


class InferenceInputError(ValueError):
    """Raised when an input cannot be scored safely by the requested model."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class SnapshotInferenceAPI:
    def __init__(self, registry_root: Path | str) -> None:
        self.root = Path(registry_root)
        self.registry = load_registry(self.root)
        self._cache: dict[str, dict[str, Any]] = {}

    def tasks(self) -> dict[str, dict[str, Any]]:
        return {
            task_id: {
                "dataset_id": entry["dataset_id"],
                "target": entry["target"],
                "task_type": entry["task_type"],
                "status": entry["status"],
                "unit": entry["unit"],
                "model_id": entry["model_id"],
            }
            for task_id, entry in self.registry["tasks"].items()
        }

    def _load(self, task_id: str) -> dict[str, Any]:
        if task_id in self._cache:
            return self._cache[task_id]
        entry = self.registry["tasks"].get(task_id)
        if entry is None:
            raise KeyError(
                f"Unknown task {task_id!r}; registered tasks: {sorted(self.registry['tasks'])}"
            )
        directory = self.root / entry["artifact_dir"]
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        schema = json.loads((directory / "schema.json").read_text(encoding="utf-8"))
        loaded = {
            "entry": entry,
            "config": config,
            "schema": schema,
            "pipeline": joblib.load(directory / "feature_pipeline.joblib"),
            "model": joblib.load(directory / "model.joblib"),
        }
        self._cache[task_id] = loaded
        return loaded

    def info(self, task_id: str) -> dict[str, Any]:
        loaded = self._load(task_id)
        entry = loaded["entry"]
        return {
            "task_id": task_id,
            "dataset_id": entry["dataset_id"],
            "target": entry["target"],
            "task_type": entry["task_type"],
            "status": entry["status"],
            "unit": entry["unit"],
            "model_id": entry["model_id"],
            "model_version": entry["model_version"],
            "selected_model": entry["selected_model"],
            "required_features": entry["required_features"],
            "chemistry_column": entry["chemistry_column"],
            "supported_chemistry": sorted(
                loaded["schema"]["categorical_vocabularies"].get(
                    entry["chemistry_column"], []
                )
            ),
            "test_metrics": entry["test_metrics"],
            "uncertainty": entry["uncertainty"],
            "limitations": entry["limitations"],
            "production_ready": False,
        }

    def _prepare(
        self,
        task_id: str,
        records: list[dict[str, Any]] | pd.DataFrame,
        *,
        allow_unseen_chemistry: bool,
        allow_missing_values: bool,
    ) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
        loaded = self._load(task_id)
        config = loaded["config"]
        entry = loaded["entry"]
        frame = (
            records.copy()
            if isinstance(records, pd.DataFrame)
            else pd.DataFrame(list(records))
        )
        if frame.empty:
            raise InferenceInputError("No input records were supplied.")

        required = list(entry["required_features"])
        absent = [column for column in required if column not in frame.columns]
        if absent:
            raise InferenceInputError(
                f"Missing required features for {task_id}: {absent}. "
                "The model was trained on this exact schema and will not guess them."
            )

        forbidden_present = [
            column
            for column in [entry["target"], *config["forbidden_columns"]]
            if column in frame.columns and column not in required
        ]
        warnings: list[str] = []
        if forbidden_present:
            warnings.append(
                "Ignored target/leakage columns present in the input: "
                + ", ".join(sorted(set(forbidden_present)))
            )

        chemistry_column = entry["chemistry_column"]
        supported = set(
            loaded["schema"]["categorical_vocabularies"].get(chemistry_column, [])
        )
        unseen: list[str] = []
        if chemistry_column in frame.columns:
            values = frame[chemistry_column].astype("string").fillna("__MISSING__")
            unseen = sorted(set(values.astype(str)) - supported - {"__MISSING__"})
        else:
            warnings.append(
                f"Chemistry column {chemistry_column!r} not supplied; the model will "
                "treat it as missing."
            )
        if unseen:
            message = (
                f"Unsupported chemistry for {task_id}: {unseen}. "
                f"Training vocabulary: {sorted(supported)}."
            )
            if not allow_unseen_chemistry:
                raise InferenceInputError(message)
            warnings.append(message + " Prediction proceeds with explicit opt-in.")

        missing_cells = int(frame[required].isna().sum().sum())
        if missing_cells and not allow_missing_values:
            counts = frame[required].isna().sum()
            offenders = {
                column: int(count)
                for column, count in counts.items()
                if int(count) > 0
            }
            raise InferenceInputError(
                f"Missing values in required features: {offenders}. "
                "Set allow_missing_values=True to apply training-split medians."
            )
        if missing_cells:
            warnings.append(
                f"{missing_cells} missing feature cells were filled with training-split "
                "medians (numeric) or the missing category (categorical)."
            )

        ood: dict[str, list[float]] = {}
        statistics = config.get("numeric_statistics", {})
        for column in entry["numeric_features"]:
            stats = statistics.get(column)
            if not stats:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            observed = values.dropna()
            if observed.empty:
                continue
            low, high = stats["min"], stats["max"]
            out_of_range = int(((observed < low) | (observed > high)).sum())
            if out_of_range:
                ood[column] = [float(observed.min()), float(observed.max())]
        if ood:
            warnings.append(
                "Inputs outside the training numeric range were detected: "
                + ", ".join(sorted(ood))
            )

        validation = {
            "rows": int(len(frame)),
            "required_feature_count": len(required),
            "missing_cells_filled": missing_cells,
            "unseen_chemistry": unseen,
            "out_of_training_range": ood,
            "ignored_columns": sorted(set(forbidden_present)),
            "warnings": warnings,
        }
        return frame, validation, loaded

    def predict(
        self,
        task_id: str,
        records: list[dict[str, Any]] | pd.DataFrame,
        *,
        allow_unseen_chemistry: bool = False,
        allow_missing_values: bool = True,
    ) -> dict[str, Any]:
        frame, validation, loaded = self._prepare(
            task_id,
            records,
            allow_unseen_chemistry=allow_unseen_chemistry,
            allow_missing_values=allow_missing_values,
        )
        entry = loaded["entry"]
        pipeline = loaded["pipeline"]
        model = loaded["model"]
        features = pipeline.transform(frame[entry["required_features"]])

        output: dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": entry["dataset_id"],
            "target": entry["target"],
            "task_type": entry["task_type"],
            "unit": entry["unit"],
            "model_id": entry["model_id"],
            "model_version": entry["model_version"],
            "selected_model": entry["selected_model"],
            "status": entry["status"],
            "production_ready": False,
            "input_validation": validation,
            "predictions": [],
            "provenance": entry["dataset_provenance"],
        }

        if entry["task_type"] == "regression":
            prediction = np.asarray(model.predict(features), dtype="float64").reshape(-1)
            residual = entry["uncertainty"].get("residual_q90")
            bounds = {
                "remaining_capacity": (0.0, None),
                "state_of_health": (0.0, 100.0),
                "state_of_health_pct": (0.0, 100.0),
                "discharge_capacity_ah": (0.0, None),
            }.get(entry["target"])
            for index, value in enumerate(prediction):
                row: dict[str, Any] = {
                    "row": index,
                    "prediction": float(value),
                }
                if residual is not None:
                    row["interval_90"] = [
                        float(value - residual),
                        float(value + residual),
                    ]
                if bounds is not None:
                    low, high = bounds
                    violated = value < low or (high is not None and value > high)
                    row["physical_bounds_violated"] = bool(violated)
                output["predictions"].append(row)
            output["uncertainty"] = loaded["entry"]["uncertainty"]
        else:
            encoded = np.asarray(model.predict(features), dtype=int).reshape(-1)
            classes = list(entry["classes"])
            probability = np.asarray(model.predict_proba(features), dtype="float64")
            for index, code in enumerate(encoded):
                output["predictions"].append(
                    {
                        "row": index,
                        "predicted_class": classes[int(code)],
                        "probabilities": {
                            label: float(probability[index, position])
                            for position, label in enumerate(classes)
                        },
                    }
                )
            output["uncertainty"] = loaded["entry"]["uncertainty"]
        return _jsonable(output)


__all__ = ["InferenceInputError", "MODEL_VERSION", "SnapshotInferenceAPI"]
