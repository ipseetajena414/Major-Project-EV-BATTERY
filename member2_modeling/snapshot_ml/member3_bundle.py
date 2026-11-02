"""Proposed Member 3 bundle built from the new snapshot models.

This module never writes to ``member3/`` and never overwrites the existing
``artifacts/member3_exports/`` bundle. Output goes to a new versioned directory.
The existing Member 3 adapter expects cell-cycle rows with ``cycle_index``; the
snapshot datasets have no cycle history, so that mismatch is reported instead of
being hidden behind a fabricated cycle index.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from member1_pipeline import loader

from .range_estimate import estimate_range
from .registry import load_registry, sha256_file, verify_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "member2_modeling" / "artifacts" / "snapshot_ml"
DEFAULT_OUTPUT = REPO_ROOT / "member2_modeling" / "artifacts" / "member3_exports_v2"

SOH_COLUMNS = [
    "snapshot_row_id",
    "entity_id",
    "dataset_id",
    "chemistry",
    "observed_soh_pct",
    "predicted_soh_pct",
    "interval_90_low",
    "interval_90_high",
    "unit",
    "model_id",
    "model_status",
    "validation_status",
    "split",
    "quality_flag",
]

FAILURE_COLUMNS = [
    "snapshot_row_id",
    "entity_id",
    "dataset_id",
    "chemistry",
    "observed_failure",
    "predicted_failure_probability",
    "predicted_class",
    "model_id",
    "model_status",
    "validation_status",
    "split",
    "quality_flag",
]

RANGE_COLUMNS = [
    "snapshot_row_id",
    "entity_id",
    "chemistry",
    "battery_capacity_kwh",
    "soh_pct",
    "soh_source",
    "range_km",
    "range_km_optimistic",
    "range_km_conservative",
    "unit",
    "kind",
    "is_measured",
    "validated_accuracy",
    "validation_status",
    "consumption_kwh_per_km",
]


class BundleError(RuntimeError):
    """Raised when a bundle cannot be produced or validated."""


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _soh_rows(
    predictions: pd.DataFrame,
    *,
    dataset_id: str,
    model_id: str,
    status: str,
) -> list[dict[str, Any]]:
    rows = []
    for record in predictions.to_dict("records"):
        low = record.get("interval_90_low")
        high = record.get("interval_90_high")
        rows.append(
            {
                "snapshot_row_id": record.get("snapshot_row_id", record.get("source_row_id")),
                "entity_id": record.get("entity_id"),
                "dataset_id": dataset_id,
                "chemistry": record.get("chemistry"),
                "observed_soh_pct": record.get("observed"),
                "predicted_soh_pct": record.get("prediction"),
                "interval_90_low": low,
                "interval_90_high": high,
                "unit": "%",
                "model_id": model_id,
                "model_status": status,
                "validation_status": "grouped_holdout_evaluated_not_production_validated",
                "split": "test",
                "quality_flag": record.get("quality_flag"),
            }
        )
    return rows


def _failure_rows(
    predictions: pd.DataFrame, *, model_id: str, status: str
) -> list[dict[str, Any]]:
    rows = []
    for record in predictions.to_dict("records"):
        probability = None
        for key, value in record.items():
            if str(key).startswith("probability_1"):
                probability = value
        rows.append(
            {
                "snapshot_row_id": record.get(
                    "snapshot_row_id", record.get("source_row_id")
                ),
                "entity_id": record.get("entity_id"),
                "dataset_id": "EV_FAILURE_PREDICTION",
                "chemistry": record.get("chemistry"),
                "observed_failure": record.get("observed"),
                "predicted_failure_probability": probability,
                "predicted_class": record.get("prediction"),
                "model_id": model_id,
                "model_status": status,
                "validation_status": "grouped_holdout_evaluated_not_production_validated",
                "split": "test",
                "quality_flag": record.get("quality_flag"),
            }
        )
    return rows


def _range_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    ev = loader.load_dataset(
        "EV_FAILURE_PREDICTION",
        columns=[
            "source_row_id",
            "vehicle_id",
            "battery_capacity_kwh",
            "discharge_efficiency",
        ],
    )
    merged = predictions.merge(
        ev, left_on="entity_id", right_on="vehicle_id", how="left", suffixes=("", "_ev")
    )
    rows = []
    for record in merged.to_dict("records"):
        capacity = record.get("battery_capacity_kwh")
        soh = record.get("prediction")
        if capacity is None or not np.isfinite(capacity) or not np.isfinite(soh):
            continue
        efficiency = record.get("discharge_efficiency")
        estimate = estimate_range(
            capacity,
            soh,
            discharge_efficiency_pct=100.0
            if efficiency is None or not np.isfinite(efficiency)
            else efficiency,
        )
        rows.append(
            {
                "snapshot_row_id": record.get(
                    "snapshot_row_id", record.get("source_row_id")
                ),
                "entity_id": record.get("entity_id"),
                "chemistry": record.get("chemistry"),
                "battery_capacity_kwh": capacity,
                "soh_pct": soh,
                "soh_source": "ev_soh_model_prediction",
                "range_km": estimate["range_km"],
                "range_km_optimistic": estimate["range_km_scenarios"]["optimistic"],
                "range_km_conservative": estimate["range_km_scenarios"]["conservative"],
                "unit": "km",
                "kind": "calculated_estimate",
                "is_measured": False,
                "validated_accuracy": False,
                "validation_status": "not_validated_no_range_target",
                "consumption_kwh_per_km": estimate["consumption_kwh_per_km"],
            }
        )
    return rows


def validate_bundle(directory: Path | str) -> dict[str, Any]:
    directory = Path(directory)
    problems: list[str] = []
    checked: dict[str, int] = {}
    for name, columns in (
        ("ev_soh_predictions.csv", SOH_COLUMNS),
        ("solid_state_soh_predictions.csv", SOH_COLUMNS),
        ("ev_failure_predictions.csv", FAILURE_COLUMNS),
        ("ev_range_estimate.csv", RANGE_COLUMNS),
    ):
        path = directory / name
        if not path.is_file():
            problems.append(f"missing {name}")
            continue
        frame = pd.read_csv(path)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            problems.append(f"{name}: missing columns {missing}")
        checked[name] = int(len(frame))
        if not frame.empty and "snapshot_row_id" in frame.columns:
            if frame["snapshot_row_id"].isna().any():
                problems.append(f"{name}: snapshot_row_id contains null values")
            if frame["snapshot_row_id"].duplicated().any():
                problems.append(f"{name}: snapshot_row_id contains duplicate values")
        if name.endswith("soh_predictions.csv") and not frame.empty:
            soh = pd.to_numeric(frame["predicted_soh_pct"], errors="coerce")
            if soh.isna().any() or ((soh < 0) | (soh > 100)).any():
                problems.append(f"{name}: predicted SOH outside 0-100 or non-finite")
        if name == "ev_failure_predictions.csv" and not frame.empty:
            probability = pd.to_numeric(
                frame["predicted_failure_probability"], errors="coerce"
            )
            if probability.isna().any() or ((probability < 0) | (probability > 1)).any():
                problems.append(f"{name}: failure probability outside 0-1")
        if name == "ev_range_estimate.csv" and not frame.empty:
            if not (frame["kind"] == "calculated_estimate").all():
                problems.append(f"{name}: range rows must be labelled calculated_estimate")
            if frame["is_measured"].astype(str).str.lower().isin(["true", "1"]).any():
                problems.append(f"{name}: range rows claim measured values")
    return {"ok": not problems, "rows": checked, "problems": problems}


def build_bundle(
    source_root: Path | str = DEFAULT_SOURCE,
    output: Path | str = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    source = Path(source_root)
    destination = Path(output)
    if destination.resolve() == (REPO_ROOT / "member2_modeling" / "artifacts" / "member3_exports").resolve():
        raise BundleError("Refusing to overwrite the existing Member 3 export bundle.")
    destination.mkdir(parents=True, exist_ok=True)

    registry = load_registry(source)
    verification = verify_registry(source)
    if not verification["ok"]:
        raise BundleError(f"Registry checksum verification failed: {verification['problems']}")

    def predictions(task_id: str) -> pd.DataFrame:
        entry = registry["tasks"][task_id]
        frame = pd.read_parquet(source / entry["artifact_dir"] / "test_predictions.parquet")
        frame = frame.rename(columns={entry["entity_column"]: "entity_id"})
        frame = frame.rename(columns={entry["chemistry_column"]: "chemistry"})
        return frame

    ev_soh = registry["tasks"]["ev_soh"]
    solid_soh = registry["tasks"]["solid_soh"]
    failure = registry["tasks"]["ev_failure"]

    _write_csv(
        destination / "ev_soh_predictions.csv",
        SOH_COLUMNS,
        _soh_rows(
            predictions("ev_soh"),
            dataset_id="EV_FAILURE_PREDICTION",
            model_id=ev_soh["model_id"],
            status=ev_soh["status"],
        ),
    )
    _write_csv(
        destination / "solid_state_soh_predictions.csv",
        SOH_COLUMNS,
        _soh_rows(
            predictions("solid_soh"),
            dataset_id="SOLID_STATE_TELEMETRY_2026",
            model_id=solid_soh["model_id"],
            status=solid_soh["status"],
        ),
    )
    _write_csv(
        destination / "ev_failure_predictions.csv",
        FAILURE_COLUMNS,
        _failure_rows(
            predictions("ev_failure"),
            model_id=failure["model_id"],
            status=failure["status"],
        ),
    )
    _write_csv(
        destination / "ev_range_estimate.csv",
        RANGE_COLUMNS,
        _range_rows(predictions("ev_soh")),
    )

    contract = {
        "contract_version": "2.0.0-snapshot-draft",
        "production_ready": False,
        "datasets": ["EV_FAILURE_PREDICTION", "SOLID_STATE_TELEMETRY_2026"],
        "files": {
            "ev_soh_predictions.csv": {
                "columns": SOH_COLUMNS,
                "unit": "%",
                "status": ev_soh["status"],
                "model_id": ev_soh["model_id"],
            },
            "solid_state_soh_predictions.csv": {
                "columns": SOH_COLUMNS,
                "unit": "%",
                "status": solid_soh["status"],
                "model_id": solid_soh["model_id"],
            },
            "ev_failure_predictions.csv": {
                "columns": FAILURE_COLUMNS,
                "unit": "failure probability",
                "status": failure["status"],
                "model_id": failure["model_id"],
            },
            "ev_range_estimate.csv": {
                "columns": RANGE_COLUMNS,
                "unit": "km",
                "kind": "calculated_estimate",
                "is_measured": False,
                "validated_accuracy": False,
                "accuracy_claim": "none",
            },
        },
        "rul": {
            "available": False,
            "reason": "No active Member 1 dataset has an observed EOL trajectory.",
        },
        "member3_adapter": {
            "existing_adapter": "member3/integration/member2_range_adapter.py",
            "required_columns": ["cycle_index", "cell_id", "chemistry", "soh_pct"],
            "compatible": False,
            "reason": (
                "The snapshot datasets have no cycle_index. Emitting a fake cycle index "
                "would fabricate temporal structure, so this bundle is proposed as a new "
                "snapshot contract rather than passed to the cycle adapter."
            ),
        },
        "limitations": [
            "Snapshot predictions are grouped-holdout estimates, not temporal forecasts.",
            "EV capacity targets are algebraically derived from the health percentage.",
            "Synthetic tasks are excluded from this bundle because they are RESEARCH_ONLY.",
            "Range is a formula estimate with no measured ground truth.",
        ],
    }
    (destination / "SNAPSHOT_PREDICTION_CONTRACT.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    provenance = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_registry": str((source / "model_registry.json").relative_to(REPO_ROOT)),
        "registry_sha256": sha256_file(source / "model_registry.json"),
        "registry_verification": verification,
        "datasets": {
            dataset_id: {
                "processed_path": loader.load_registry()["active_datasets"][dataset_id][
                    "processed_path"
                ],
                "processed_sha256": loader.load_registry()["active_datasets"][dataset_id][
                    "processed_sha256"
                ],
            }
            for dataset_id in ("EV_FAILURE_PREDICTION", "SOLID_STATE_TELEMETRY_2026")
        },
        "models": {
            task_id: {
                "model_id": registry["tasks"][task_id]["model_id"],
                "artifact_dir": registry["tasks"][task_id]["artifact_dir"],
                "files": registry["tasks"][task_id]["files"],
                "test_metrics": registry["tasks"][task_id]["test_metrics"],
            }
            for task_id in ("ev_soh", "solid_soh", "ev_failure")
        },
        "member3_modified": False,
    }
    (destination / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    validation = validate_bundle(destination)
    (destination / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "COMPATIBILITY.md").write_text(
        "\n".join(
            [
                "# Member 3 compatibility notes",
                "",
                "Existing adapter: `member3/integration/member2_range_adapter.py`.",
                "",
                "It requires `cycle_index`, `cell_id`, `chemistry` and `soh_pct`.",
                "The active Member 1 datasets are snapshots and contain no cycle history,",
                "so no `cycle_index` exists. This bundle therefore does not claim adapter",
                "compatibility and does not fabricate a cycle index.",
                "",
                "Bundle validation status: "
                + ("OK" if validation["ok"] else "FAILED"),
                "",
                "Production readiness: false. Snapshot models are grouped-holdout",
                "experimental models, and the range file is a formula estimate.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "output": str(destination),
        "validation": validation,
        "contract": contract,
    }


__all__ = [
    "BundleError",
    "build_bundle",
    "validate_bundle",
]


if __name__ == "__main__":
    result = build_bundle()
    print(json.dumps(result["validation"], indent=2))
