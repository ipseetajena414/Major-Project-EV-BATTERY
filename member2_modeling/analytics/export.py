"""Assemble the machine-readable Member 2 analytics package.

Outputs are written only under ``member2_modeling/artifacts/analytics_v1``.
Member 3 and Member 5 directories are never modified; Member 3 receives a
schema and a template instead of fabricated rows.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from member2_modeling.analytics.anomaly import (
    detect_sequential_anomalies,
    detect_snapshot_anomalies,
)
from member2_modeling.analytics.comparison import build_model_comparison
from member2_modeling.analytics.degradation import (
    build_degradation_summary,
    build_trajectory_export,
)
from member2_modeling.analytics.explain import build_feature_importance
from member2_modeling.analytics.health_status import build_health_status
from member2_modeling.analytics.sequential import (
    WORKSPACE,
    load_cycles,
    sequential_support,
    source_sha256,
)
from member2_modeling.analytics.uncertainty import build_uncertainty_summary
from member2_modeling.rul.labels import EOL_PCT
from member2_modeling.rul.pipeline import build_rul_artifacts
from member2_modeling.rul.service import RUL_DIR

ANALYTICS_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "analytics_v1"
SNAPSHOT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "snapshot_ml"
REGISTRY_PATH = SNAPSHOT_DIR / "model_registry.json"
INVENTORY_PATH = SNAPSHOT_DIR / "dataset_inventory.json"
SCHEMA_VERSION = "analytics_v1"

RUL_FILES = (
    "rul_predictions.csv",
    "rul_current_predictions.csv",
    "rul_censored_projections.csv",
    "rul_unavailable.csv",
    "rul_per_battery.csv",
    "rul_degradation_trajectories.csv",
    "rul_eol_trajectories.csv",
    "rul_metrics.json",
    "rul_model_registry.json",
    "rul_provenance.json",
)


def _sha256(path: Path) -> str:
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _registry() -> dict[str, Any]:
    return _read_json(REGISTRY_PATH)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


# ── prediction tables ───────────────────────────────────────────────────


def _snapshot_predictions(registry: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_id, task in registry["tasks"].items():
        path = SNAPSHOT_DIR / task["artifact_dir"] / "test_predictions.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        entity_column = task.get("entity_column")
        chemistry_column = task.get("chemistry_column")
        probability_columns = [
            column for column in frame.columns if column.startswith("probability_")
        ]
        for record in frame.to_dict("records"):
            probabilities = {
                column.removeprefix("probability_"): _number(record[column])
                for column in probability_columns
            }
            probability = (
                max(
                    (value for value in probabilities.values() if value is not None),
                    default=None,
                )
                if probabilities
                else None
            )
            observed = record.get("observed")
            prediction = record.get("prediction")
            residual = (
                _number(observed) - _number(prediction)
                if task["task_type"] == "regression"
                else None
            )
            rows.append(
                {
                    "domain": "snapshot",
                    "record_scope": "frozen_test",
                    "dataset_id": task["dataset_id"],
                    "task_id": task_id,
                    "target": task["target"],
                    "task_type": task["task_type"],
                    "model_id": task["selected_model"]["name"],
                    "model_version": task.get("model_version"),
                    "record_id": record.get("snapshot_row_id"),
                    "entity_id": (
                        record.get(entity_column) if entity_column else None
                    ),
                    "chemistry": (
                        record.get(chemistry_column) if chemistry_column else None
                    ),
                    "split": record.get("split", "test"),
                    "observed": observed,
                    "prediction": prediction,
                    "residual": residual,
                    "abs_residual": None if residual is None else abs(residual),
                    "lower": record.get("interval_90_low"),
                    "upper": record.get("interval_90_high"),
                    "predicted_label": (
                        prediction if task["task_type"] != "regression" else None
                    ),
                    "probability": probability,
                    "probabilities_json": (
                        json.dumps(probabilities, sort_keys=True)
                        if probabilities
                        else None
                    ),
                    "quality_flag": record.get("quality_flag"),
                    "prediction_status": "frozen_test_prediction",
                    "validation_status": task.get("status"),
                }
            )
    return pd.DataFrame(rows)


def _rul_prediction_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path = RUL_DIR / "rul_predictions.csv"
    if path.is_file():
        frame = pd.read_csv(path)
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "domain": "rul",
                    "record_scope": "leave_one_cell_out",
                    "dataset_id": record.get("dataset_id"),
                    "task_id": "rul_cycles_to_eol",
                    "target": "estimated_rul_cycles",
                    "task_type": "regression",
                    "model_id": record.get("model_id"),
                    "model_version": record.get("model_version"),
                    "record_id": (
                        f"{record.get('battery_id')}@{int(record.get('current_cycle'))}"
                    ),
                    "entity_id": record.get("battery_id"),
                    "chemistry": record.get("chemistry"),
                    "split": record.get("validation_protocol"),
                    "observed": record.get("actual_rul_cycles"),
                    "prediction": record.get("predicted_rul_cycles"),
                    "residual": record.get("residual_cycles"),
                    "abs_residual": (
                        None
                        if pd.isna(record.get("residual_cycles"))
                        else abs(record.get("residual_cycles"))
                    ),
                    "lower": record.get("lower_uncertainty_bound"),
                    "upper": record.get("upper_uncertainty_bound"),
                    "predicted_label": None,
                    "probability": None,
                    "probabilities_json": None,
                    "quality_flag": None,
                    "prediction_status": record.get("prediction_status"),
                    "validation_status": record.get("validation_status"),
                }
            )
    censored_path = RUL_DIR / "rul_censored_projections.csv"
    if censored_path.is_file():
        frame = pd.read_csv(censored_path)
        for record in frame.to_dict("records"):
            rows.append(
                {
                    "domain": "rul",
                    "record_scope": "censored_projection",
                    "dataset_id": record.get("dataset_id"),
                    "task_id": "rul_cycles_to_eol",
                    "target": "estimated_rul_cycles",
                    "task_type": "regression",
                    "model_id": record.get("model_id"),
                    "model_version": record.get("model_version"),
                    "record_id": (
                        f"{record.get('battery_id')}@{int(record.get('current_cycle'))}"
                    ),
                    "entity_id": record.get("battery_id"),
                    "chemistry": record.get("chemistry"),
                    "split": "no_observed_eol",
                    "observed": None,
                    "prediction": record.get("estimated_rul_cycles"),
                    "residual": None,
                    "abs_residual": None,
                    "lower": record.get("lower_uncertainty_bound"),
                    "upper": record.get("upper_uncertainty_bound"),
                    "predicted_label": None,
                    "probability": None,
                    "probabilities_json": None,
                    "quality_flag": None,
                    "prediction_status": record.get("prediction_status"),
                    "validation_status": record.get("validation_status"),
                }
            )
    return pd.DataFrame(rows)


# ── metric tables ───────────────────────────────────────────────────────


def _model_metrics(registry: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = {"snapshot": {}, "rul": {}, "range": {}}
    metric_fields = (
        "n",
        "mae",
        "rmse",
        "bias",
        "r2",
        "mape_pct",
        "median_abs_error",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "brier",
        "log_loss",
        "roc_auc",
    )
    for task_id, task in registry["tasks"].items():
        metrics_path = SNAPSHOT_DIR / task["artifact_dir"] / "metrics.json"
        task_metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
        nested["snapshot"][task_id] = {
            "dataset_id": task["dataset_id"],
            "task_type": task["task_type"],
            "target": task["target"],
            "unit": task["unit"],
            "readiness": task["status"],
            "selected_model": task["selected_model"],
            "test": task.get("test_metrics"),
            "baselines": task.get("baselines"),
            "uncertainty": task.get("uncertainty"),
            "per_chemistry": task_metrics.get("per_chemistry"),
        }
        selected = task.get("test_metrics", {})
        row = {
            "domain": "snapshot",
            "task_id": task_id,
            "dataset_id": task["dataset_id"],
            "task_type": task["task_type"],
            "model_id": task["selected_model"]["name"],
            "model_role": "selected",
            "readiness": task["status"],
            "frozen_test": True,
        }
        row.update({field: selected.get(field) for field in metric_fields})
        row["coverage"] = None
        rows.append(row)
        for baseline_name, baseline in task.get("baselines", {}).items():
            row = {
                "domain": "snapshot",
                "task_id": task_id,
                "dataset_id": task["dataset_id"],
                "task_type": task["task_type"],
                "model_id": baseline_name,
                "model_role": "baseline",
                "readiness": task["status"],
                "frozen_test": True,
            }
            row.update({field: baseline.get(field) for field in metric_fields})
            row["coverage"] = None
            rows.append(row)

    rul_metrics_path = RUL_DIR / "rul_metrics.json"
    if rul_metrics_path.is_file():
        rul_metrics = _read_json(rul_metrics_path)
        nested["rul"] = rul_metrics
        for model_id, block in rul_metrics["candidates"].items():
            rows.append(
                {
                    "domain": "rul",
                    "task_id": "rul_cycles_to_eol",
                    "dataset_id": "NASA_Ames_PCoE",
                    "task_type": "regression",
                    "model_id": model_id,
                    "model_role": (
                        "selected"
                        if model_id == rul_metrics["selection"]["selected_model"]
                        else "candidate"
                    ),
                    "readiness": "EXPERIMENTAL",
                    "frozen_test": False,
                    "n": block["n_scored"],
                    "mae": block["pre_threshold"]["mae"],
                    "rmse": block["pre_threshold"]["rmse"],
                    "bias": block["pre_threshold"]["bias"],
                    "r2": None,
                    "mape_pct": None,
                    "median_abs_error": None,
                    "accuracy": None,
                    "balanced_accuracy": None,
                    "macro_f1": None,
                    "weighted_f1": None,
                    "brier": None,
                    "log_loss": None,
                    "roc_auc": None,
                    "coverage": block.get("coverage"),
                }
            )
        nested["range"] = {
            "kind": "calculated_estimate",
            "validated_accuracy": False,
            "consumption_scenarios_kwh_per_km": {
                "optimistic": 0.15,
                "central": 0.18,
                "conservative": 0.22,
            },
        }
    frame = pd.DataFrame(rows)
    return frame, nested


def _classification_results(registry: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    for task_id, task in registry["tasks"].items():
        metrics = task.get("test_metrics", {})
        if task["task_type"] == "regression":
            continue
        labels = metrics.get("confusion_matrix", {}).get("labels", [])
        per_class = metrics.get("per_class", {})
        for label in labels:
            block = per_class.get(label, {})
            rows.append(
                {
                    "task_id": task_id,
                    "dataset_id": task["dataset_id"],
                    "class_label": label,
                    "n": block.get("n"),
                    "precision": block.get("precision"),
                    "recall": block.get("recall"),
                    "accuracy": metrics.get("accuracy"),
                    "balanced_accuracy": metrics.get("balanced_accuracy"),
                    "macro_f1": metrics.get("macro_f1"),
                    "weighted_f1": metrics.get("weighted_f1"),
                    "brier": metrics.get("brier"),
                    "log_loss": metrics.get("log_loss"),
                    "roc_auc": metrics.get("roc_auc"),
                    "calibration_status": task.get("uncertainty", {}).get(
                        "calibration_status", "uncalibrated_class_probabilities"
                    ),
                }
            )
        matrices[task_id] = {
            "dataset_id": task["dataset_id"],
            "labels": labels,
            "matrix": metrics.get("confusion_matrix", {}).get("matrix"),
            "n": metrics.get("n"),
            "per_chemistry": metrics.get("per_chemistry"),
        }
    return pd.DataFrame(rows), matrices


def _chemistry_metrics(registry: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = (
        "n",
        "mae",
        "rmse",
        "bias",
        "r2",
        "mape_pct",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "brier",
        "log_loss",
        "roc_auc",
    )
    for task_id, task in registry["tasks"].items():
        metrics_path = SNAPSHOT_DIR / task["artifact_dir"] / "metrics.json"
        task_metrics = _read_json(metrics_path) if metrics_path.is_file() else {}
        per_chemistry = task.get("test_metrics", {}).get("per_chemistry")
        source = "registry_test_metrics"
        if not per_chemistry:
            per_chemistry = task_metrics.get("per_chemistry") or {}
            source = "task_metrics"
        for chemistry, block in per_chemistry.items():
            row = {
                "task_id": task_id,
                "dataset_id": task["dataset_id"],
                "task_type": task["task_type"],
                "chemistry": chemistry,
                "metric_source": source,
                "selected_model": task["selected_model"]["name"],
            }
            row.update({field: block.get(field) for field in fields})
            rows.append(row)
    return pd.DataFrame(rows)


# ── schemas and templates ───────────────────────────────────────────────

COMBINED_COLUMNS = (
    "battery_id",
    "cell_id",
    "dataset_id",
    "chemistry",
    "cycle_index",
    "soh_pct",
    "soh_source",
    "range_km",
    "usable_energy_kwh",
    "consumption_kwh_per_km",
    "vehicle_config_id",
    "rul_cycles",
    "rul_lower_cycles",
    "rul_upper_cycles",
    "rul_eol_threshold_pct",
    "rul_status",
    "anomaly_flag",
    "anomaly_score",
    "health_band",
    "member2_model_id",
    "member3_model_id",
    "member2_artifact_sha256",
    "member3_artifact_sha256",
    "provenance_json",
)


def _combined_schema() -> dict[str, Any]:
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Member 2 + Member 3 combined battery analytics row",
        "description": (
            "One row joins a Member 2 battery-state estimate to an optional "
            "Member 3 vehicle/range result. Member 2 never fabricates Member 3 "
            "fields: range_km, usable_energy_kwh, consumption_kwh_per_km, "
            "vehicle_config_id and member3_model_id may be null when no "
            "Member 3 result exists."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "battery_id",
            "dataset_id",
            "chemistry",
            "cycle_index",
            "soh_pct",
            "soh_source",
            "member2_model_id",
            "provenance_json",
        ],
        "properties": {
            "battery_id": {"type": "string", "minLength": 1},
            "cell_id": {"type": ["string", "null"]},
            "dataset_id": {"type": "string", "minLength": 1},
            "chemistry": {
                "type": "string",
                "enum": ["NMC", "LFP", "Na-ion", "LMO", "LTO", "NCA", "UNKNOWN"],
            },
            "cycle_index": {"type": "integer", "minimum": 0},
            "soh_pct": {"type": "number", "minimum": 0, "maximum": 110},
            "soh_source": {
                "type": "string",
                "enum": ["observed", "predicted", "calculated"],
            },
            "range_km": {
                "type": ["number", "null"],
                "minimum": 0,
                "description": "Member 3 output only; null when unavailable.",
            },
            "usable_energy_kwh": {"type": ["number", "null"], "minimum": 0},
            "consumption_kwh_per_km": {"type": ["number", "null"], "minimum": 0},
            "vehicle_config_id": {"type": ["string", "null"]},
            "rul_cycles": {
                "type": ["number", "null"],
                "minimum": 0,
                "description": "Member 2 experimental RUL estimate; null when UNAVAILABLE.",
            },
            "rul_lower_cycles": {"type": ["number", "null"], "minimum": 0},
            "rul_upper_cycles": {"type": ["number", "null"], "minimum": 0},
            "rul_eol_threshold_pct": {"type": ["number", "null"]},
            "rul_status": {
                "type": ["string", "null"],
                "enum": [
                    "VALIDATED",
                    "EXPERIMENTAL",
                    "EXPERIMENTAL_CENSORED",
                    "UNAVAILABLE",
                    None,
                ],
            },
            "anomaly_flag": {"type": ["boolean", "null"]},
            "anomaly_score": {"type": ["number", "null"]},
            "health_band": {
                "type": ["string", "null"],
                "enum": ["GOOD", "FAIR", "POOR", "CRITICAL", None],
            },
            "member2_model_id": {"type": "string"},
            "member3_model_id": {"type": ["string", "null"]},
            "member2_artifact_sha256": {"type": ["string", "null"]},
            "member3_artifact_sha256": {"type": ["string", "null"]},
            "provenance_json": {
                "type": "string",
                "description": "JSON-encoded provenance object for the row.",
            },
        },
        "x-notes": [
            "Member 3 owns range_km and vehicle configuration fields.",
            "Member 2 owns SOH, RUL, anomaly and health-band fields.",
            "A null range_km means Member 3 has not produced a result; it is "
            "never imputed.",
            "rul_status=VALIDATED is reserved and currently unused.",
        ],
    }


def _member3_soh_sequences() -> pd.DataFrame:
    cycles = load_cycles()
    selected = cycles[cycles["chemistry"].isin(["NMC", "LFP"])].copy()
    return pd.DataFrame(
        {
            "cell_id": selected["cell_id"],
            "cycle_index": selected["cycle_id"],
            "chemistry": selected["chemistry"],
            "soh_pct": selected["soh_pct"],
            "dataset_id": selected["dataset_id"],
            "source_sha256": source_sha256(),
            "is_observed": True,
            "soh_source": "observed",
        }
    )


# ── handover document ───────────────────────────────────────────────────


def _handover_markdown(
    registry: dict[str, Any],
    rul_summary: dict[str, Any],
) -> str:
    selection = rul_summary.get("selection", {})
    return f"""# Member 5 Analytics Handover

Generated from frozen Member 2 artifacts. No Member 3 or Member 5 directory was
modified. All RUL values are experimental and not approved for operational use.

## 1. What is in this package

| File | Purpose |
|---|---|
| `prediction_results.csv` | Frozen test predictions plus RUL leave-one-cell-out predictions |
| `model_metrics.csv` / `model_metrics.json` | Selected models and baselines with frozen metrics |
| `regression_residuals.csv` | Per-record residuals for snapshot regression and RUL |
| `classification_results.csv` | Per-class precision/recall and overall classification metrics |
| `confusion_matrices.json` | Confusion matrices, including per-chemistry blocks |
| `chemistry_metrics.csv` | Metrics split by chemistry |
| `dataset_summary.json` | Dataset inventory and RUL support matrix |
| `model_comparison.csv` | Selected model versus best baseline on the same rows |
| `prediction_provenance.json` | Source hashes, artifact hashes, execution metadata |
| `feature_importance.csv` | Permutation importance (no SHAP; dependency pins frozen) |
| `uncertainty_summary.csv/json` | Interval coverage and calibration status |
| `anomaly_flags.csv` | Statistical telemetry outliers; no fault labels |
| `health_status.csv` | Informational SOH bands |
| `degradation_trajectories.csv` | Observed SOH rows plus BATTERY_5-only forecasts, clearly separated; NMC EOL forecasts live in `rul_eol_trajectories.csv` |
| `degradation_summary.csv/json` | Per-cell fade rates and support status |
| `rul_*.csv/json` | Full RUL artifacts copied from `rul_v1`, including per-battery metrics and EOL trajectories |
| `member3_combined_analytics_schema.json` | JSON Schema for joined Member 2 + Member 3 rows |
| `member3_results_template.csv` | Header-only template; no fabricated Member 3 rows |
| `member3_soh_sequences.csv` | Observed NMC/LFP cycles in Member 3 adapter columns |
| `ANALYTICS_MANIFEST.json` | File hashes and sizes for this package |

## 2. Recommended Member 5 workflow

```python
import pandas as pd

base = pd.read_csv("member2_modeling/artifacts/analytics_v1/prediction_results.csv")
rul = base[(base.domain == "rul") & (base.record_scope == "leave_one_cell_out")]
selected = rul[rul.model_id == "{selection.get('selected_model', 'median_fade_constant')}"]

# Actual versus predicted RUL for Member 5 error plots.
error = selected.dropna(subset=["observed", "prediction"]).copy()
error["residual"] = error["prediction"] - error["observed"]

# Historical SOH only; forecast rows are flagged separately.
traj = pd.read_csv("member2_modeling/artifacts/analytics_v1/degradation_trajectories.csv")
observed = traj[traj.record_type == "observed"]
forecast = traj[traj.record_type == "forecast"]
```

## 3. Confidence and status rules

- Snapshot regression: 90% split-residual intervals. Empirical coverage on the
  frozen test split is reported in `uncertainty_summary.csv`.
- Snapshot classification: probabilities are **uncalibrated**. Do not label them
  calibrated.
- RUL: {selection.get('selection_protocol', 'leave-one-cell-out')}; selected model
  `{selection.get('selected_model')}` with pre-threshold MAE
  {selection.get('selection_metric'):.3f} cycles. Only three NMC cells exist.
- LFP: censored projection only, no observed EOL.
- Snapshot datasets and Na-ion: RUL UNAVAILABLE.
- Anomaly flags are statistical outliers, not confirmed faults.
- Health bands are informational, not a safety diagnosis.

## 4. Joining Member 3 results

Member 3 owns `range_km`, `usable_energy_kwh`, `consumption_kwh_per_km`,
`vehicle_config_id` and `member3_model_id`. Join on `(cell_id, cycle_index)` or
on `(battery_id, cycle_index)` when identifiers agree. Never impute a missing
range; leave it null. The schema forbids additional properties so that
downstream columns stay stable.

## 5. Reproducibility

Rebuild this package with:

```bash
.venv/Scripts/python.exe -m member2_modeling.analytics.run_analytics --rebuild-rul
```

The RUL pipeline reads the legacy `master_table.parquet` read-only and never
re-runs or bypasses the protected PADM test evaluation sentinel.
"""


# ── assembly ────────────────────────────────────────────────────────────


def build_analytics_package(
    out_dir: Path | str = ANALYTICS_DIR,
    *,
    rebuild_rul: bool = False,
    eol_pct: float = EOL_PCT,
) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if rebuild_rul or not (RUL_DIR / "rul_summary.json").is_file():
        build_rul_artifacts(eol_pct=eol_pct)

    rul_summary = _read_json(RUL_DIR / "rul_summary.json")
    registry = _registry()

    predictions = pd.concat(
        [_snapshot_predictions(registry), _rul_prediction_rows()],
        ignore_index=True,
    )
    predictions.to_csv(destination / "prediction_results.csv", index=False)

    model_metrics, nested_metrics = _model_metrics(registry)
    model_metrics.to_csv(destination / "model_metrics.csv", index=False)
    _write_json(destination / "model_metrics.json", nested_metrics)

    residuals = predictions.dropna(subset=["residual"]).copy()
    residuals = residuals[
        [
            "domain",
            "record_scope",
            "dataset_id",
            "task_id",
            "model_id",
            "record_id",
            "entity_id",
            "chemistry",
            "split",
            "observed",
            "prediction",
            "residual",
            "abs_residual",
            "prediction_status",
            "validation_status",
        ]
    ]
    residuals.to_csv(destination / "regression_residuals.csv", index=False)

    classification, matrices = _classification_results(registry)
    classification.to_csv(destination / "classification_results.csv", index=False)
    _write_json(destination / "confusion_matrices.json", matrices)

    chemistry = _chemistry_metrics(registry)
    chemistry.to_csv(destination / "chemistry_metrics.csv", index=False)

    comparison, comparison_report = build_model_comparison()
    comparison.to_csv(destination / "model_comparison.csv", index=False)

    sequential = sequential_support(eol_pct=eol_pct)
    inventory = _read_json(INVENTORY_PATH) if INVENTORY_PATH.is_file() else {}
    dataset_summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_inventory": inventory,
        "sequential_source": {
            "path": str(WORKSPACE / "master_table.parquet"),
            "sha256": source_sha256(),
            "cycles": int(len(load_cycles())),
            "cells": int(
                load_cycles()[["dataset_id", "cell_id"]].drop_duplicates().shape[0]
            ),
        },
        "rul_support": sequential["datasets"],
        "rul_cells": sequential["cells"],
        "snapshot_sequences_available": False,
        "statement": (
            "Active Member 1 datasets are snapshots and remain UNAVAILABLE for "
            "RUL. Genuine sequences come only from the legacy cycle-level table."
        ),
    }
    _write_json(destination / "dataset_summary.json", dataset_summary)

    uncertainty, uncertainty_report = build_uncertainty_summary()
    uncertainty.to_csv(destination / "uncertainty_summary.csv", index=False)
    _write_json(destination / "uncertainty_summary.json", uncertainty_report)

    anomalies, anomaly_summary = detect_sequential_anomalies()
    anomalies.to_csv(destination / "anomaly_flags.csv", index=False)
    snapshot_anomalies: dict[str, Any] = {}
    for dataset_id in ("EV_FAILURE_PREDICTION", "SOLID_STATE_TELEMETRY_2026"):
        try:
            rows, summary = detect_snapshot_anomalies(dataset_id)
            snapshot_anomalies[dataset_id] = summary
            if not rows.empty:
                rows.to_csv(
                    destination / f"anomaly_flags_{dataset_id.lower()}.csv",
                    index=False,
                )
        except Exception as exc:  # noqa: BLE001
            snapshot_anomalies[dataset_id] = {
                "dataset_id": dataset_id,
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    _write_json(
        destination / "anomaly_summary.json",
        {
            "sequential": anomaly_summary,
            "snapshots": snapshot_anomalies,
            "confirmed_fault_labels_available": False,
        },
    )

    health, health_report = build_health_status(eol_pct=eol_pct)
    health.to_csv(destination / "health_status.csv", index=False)
    _write_json(destination / "health_status_summary.json", health_report)

    trajectories = build_trajectory_export()
    trajectories.to_csv(destination / "degradation_trajectories.csv", index=False)
    degradation, degradation_report = build_degradation_summary(eol_pct=eol_pct)
    degradation.to_csv(destination / "degradation_summary.csv", index=False)
    _write_json(destination / "degradation_summary.json", degradation_report)

    importance, importance_report = build_feature_importance()
    importance.to_csv(destination / "feature_importance.csv", index=False)
    _write_json(destination / "feature_importance_summary.json", importance_report)

    combined_schema = _combined_schema()
    _write_json(
        destination / "member3_combined_analytics_schema.json", combined_schema
    )
    pd.DataFrame(columns=list(COMBINED_COLUMNS)).to_csv(
        destination / "member3_results_template.csv", index=False
    )
    _member3_soh_sequences().to_csv(
        destination / "member3_soh_sequences.csv", index=False
    )

    for filename in RUL_FILES:
        source = RUL_DIR / filename
        if source.is_file():
            (destination / filename).write_bytes(source.read_bytes())

    _write_json(
        destination / "prediction_provenance.json",
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "registry_version": registry.get("registry_version"),
            "environment": registry.get("environment"),
            "policy": registry.get("policy"),
            "python": sys.version,
            "platform": platform.platform(),
            "sources": {
                task_id: task.get("dataset_provenance")
                for task_id, task in registry["tasks"].items()
            },
            "rul_source": _read_json(RUL_DIR / "rul_provenance.json"),
            "record_traceability": (
                "Snapshot predictions carry snapshot_row_id; RUL predictions "
                "carry battery_id and the source cycle id."
            ),
            "no_fabrication": (
                "No Member 3 range output was created. No RUL was produced for "
                "snapshot or Na-ion data. Censored and unavailable statuses are "
                "explicit."
            ),
            "feature_schemas": {
                task_id: {
                    "required_features": task.get("required_features"),
                    "numeric_features": task.get("numeric_features"),
                    "categorical_features": task.get("categorical_features"),
                }
                for task_id, task in registry["tasks"].items()
            },
        },
    )

    handover = _handover_markdown(
        registry,
        _read_json(RUL_DIR / "rul_metrics.json"),
    )
    (destination / "MEMBER5_ANALYTICS_HANDOVER.md").write_text(
        handover, encoding="utf-8"
    )

    manifest_files = []
    for path in sorted(destination.glob("*")):
        if path.is_file() and path.name != "ANALYTICS_MANIFEST.json":
            manifest_files.append(
                {
                    "name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_json(
        destination / "ANALYTICS_MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "entrypoint": "member2_modeling.analytics.run_analytics",
            "files": manifest_files,
            "member_directories_modified": False,
            "rul_approved_for_use": False,
            "combined_schema": "member3_combined_analytics_schema.json",
            "handover": "MEMBER5_ANALYTICS_HANDOVER.md",
        },
    )

    return {
        "out_dir": str(destination),
        "files": len(manifest_files) + 1,
        "prediction_rows": int(len(predictions)),
        "snapshot_prediction_rows": int(
            (predictions["domain"] == "snapshot").sum()
        ),
        "rul_prediction_rows": int((predictions["domain"] == "rul").sum()),
        "anomalies_flagged": int(
            anomalies["is_statistical_anomaly"].sum() if len(anomalies) else 0
        ),
        "health_cells": int(len(health)),
        "degradation_cells": int(len(degradation)),
        "feature_importance_rows": int(len(importance)),
        "rul_selected_model": rul_summary.get("selected_model"),
        "comparison": comparison_report,
    }


__all__ = [
    "ANALYTICS_DIR",
    "COMBINED_COLUMNS",
    "SCHEMA_VERSION",
    "build_analytics_package",
]
