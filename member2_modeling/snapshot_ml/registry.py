"""Versioned model registry with artifact checksums.

The registry is the single source of truth for which Member 2 task models
exist, how they were trained, what they may be used for, and whether their
files still match the recorded hashes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import sklearn

from .specs import TASKS, UNSUPPORTED_TASKS

REGISTRY_VERSION = "2.0.0"
REGISTRY_FILENAME = "model_registry.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_registry(results: list[dict[str, Any]], output_root: Path | str) -> dict[str, Any]:
    import lightgbm
    import numpy
    import pandas
    import pyarrow
    import torch

    root = Path(output_root)
    entries: dict[str, Any] = {}
    for result in results:
        task_id = result["task_id"]
        artifact_dir = Path(result["artifact_dir"])
        files = {
            name: {
                "path": str((artifact_dir / name).relative_to(root).as_posix()),
                "sha256": sha256_file(artifact_dir / name),
            }
            for name in (
                "model.joblib",
                "feature_pipeline.joblib",
                "config.json",
                "metrics.json",
                "schema.json",
                "test_predictions.parquet",
            )
            if (artifact_dir / name).is_file()
        }
        metrics = result["metrics"]
        entries[task_id] = {
            "model_id": f"member2::{task_id}::{REGISTRY_VERSION}",
            "model_version": REGISTRY_VERSION,
            "dataset_id": metrics["dataset_id"],
            "target": metrics["target"],
            "task_type": metrics["task_type"],
            "status": metrics["status"],
            "production_ready": False,
            "selected_model": metrics["selected_model"],
            "artifact_dir": str(artifact_dir.relative_to(root).as_posix()),
            "files": files,
            "required_features": result["config"]["feature_columns"],
            "numeric_features": result["config"]["numeric_features"],
            "categorical_features": result["config"]["categorical_features"],
            "chemistry_column": result["config"]["chemistry_column"],
            "entity_column": result["config"]["entity_column"],
            "unit": result["config"]["target"]["unit"],
            "classes": result["config"]["classes"],
            "split_sha256": result["config"]["split"]["sha256"],
            "entity_overlap": result["config"]["split"]["entity_overlap"],
            "test_metrics": {**metrics["test"], "per_chemistry": metrics["per_chemistry"]},
            "baselines": metrics["baselines"],
            "feature_importance": metrics["feature_importance"],
            "uncertainty": metrics["uncertainty"],
            "dataset_provenance": result["config"]["dataset_provenance"],
            "limitations": result["config"]["limitations"],
        }

    registry = {
        "registry_version": REGISTRY_VERSION,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "pandas": pandas.__version__,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "torch": torch.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "policy": {
            "split_unit": "entity (vehicle_id / cell_id / battery_id)",
            "preprocessing_fit": "training split only",
            "model_selection": "validation split only",
            "test_usage": "single evaluation after configuration freeze",
            "sequence_models": "not used on snapshot data",
            "rul": "unavailable on active data",
            "range": "calculated estimate only; no measured range target",
        },
        "tasks": entries,
        "supported_task_specs": {
            task_id: spec.to_dict() for task_id, spec in TASKS.items()
        },
        "unsupported_tasks": UNSUPPORTED_TASKS,
    }
    return _jsonable(registry)


def write_registry(registry: dict[str, Any], output_root: Path | str) -> Path:
    root = Path(output_root)
    path = root / REGISTRY_FILENAME
    path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_registry(output_root: Path | str) -> dict[str, Any]:
    path = Path(output_root) / REGISTRY_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Model registry not found: {path}. Run member2_modeling.snapshot_ml.run first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_registry(output_root: Path | str) -> dict[str, Any]:
    root = Path(output_root)
    registry = load_registry(root)
    problems: list[str] = []
    checked = 0
    for task_id, entry in registry["tasks"].items():
        artifact_dir = root / entry["artifact_dir"]
        for name, record in entry["files"].items():
            path = artifact_dir / name
            if not path.is_file():
                problems.append(f"{task_id}: missing {name}")
                continue
            actual = sha256_file(path)
            if actual != record["sha256"]:
                problems.append(f"{task_id}: checksum mismatch for {name}")
            checked += 1
    return {
        "ok": not problems,
        "checked_files": checked,
        "problems": problems,
        "tasks": len(registry["tasks"]),
    }


__all__ = [
    "REGISTRY_FILENAME",
    "REGISTRY_VERSION",
    "build_registry",
    "load_registry",
    "sha256_file",
    "verify_registry",
    "write_registry",
]
