"""Custom battery input API for Member 2 models.

The service validates custom records, routes them to the correct dataset and
target model, and returns provenance and readiness metadata. It never invents
sequential history and never accepts leakage columns. RUL is served only from
genuine sequential evidence, always marked experimental and never approved for
operational use.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from member2_modeling.health import TabularHealthModel
from member2_modeling.health.capacity import (
    capacity_semantics,
    derive_capacity_from_soh,
)
from member2_modeling.hybrid import HybridSOHModel

from .input import (
    CustomInputError,
    coerce_categorical,
    coerce_numeric,
    is_missing,
    normalize_records,
)

WORKSPACE = Path(__file__).resolve().parents[2]
HEALTH_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "health"
HEALTH_REPORT = HEALTH_DIR / "health_model_evaluation.json"
COMPATIBILITY_REPORT = (
    WORKSPACE
    / "member2_modeling"
    / "artifacts"
    / "reports"
    / "member2_dataset_compatibility.json"
)
HYBRID_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "hybrid_nmc_legacy"
HYBRID_MANIFEST = HYBRID_DIR / "hybrid_manifest.json"
HYBRID_MODEL_DIR = HYBRID_DIR / "hybrid_model"

SOH_TARGETS = {
    "EV_FAILURE_PREDICTION": "state_of_health",
    "SOLID_STATE_TELEMETRY_2026": "state_of_health_pct",
}
CAPACITY_TARGETS = {"remaining_capacity", "discharge_capacity_ah"}

SNAPSHOT_WARNING = (
    "Snapshot models estimate i.i.d. generalization on independent records; "
    "they are not temporal or vehicle-level forecasts."
)
SYNTHETIC_WARNING = (
    "Synthetic labels are not measured experimental truth; results are RESEARCH_ONLY."
)
SOLID_STATE_WARNING = (
    "Solid-state records mix electrolyte and cathode metadata and are not assumed "
    "to be genuine solid-state cells."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


class BatteryIntelligenceAPI:
    """Single, batch and sequence inference over the real Member 2 artifacts."""

    def __init__(self) -> None:
        if not HEALTH_REPORT.is_file():
            raise FileNotFoundError(
                f"Health model evaluation not found: {HEALTH_REPORT}. "
                "Run member2_modeling.health.train_health_models first."
            )
        self._health_report = json.loads(HEALTH_REPORT.read_text(encoding="utf-8"))
        self._compatibility = json.loads(
            COMPATIBILITY_REPORT.read_text(encoding="utf-8")
        )
        self._routes: dict[tuple[str, str], dict[str, Any]] = {}
        self._models: dict[tuple[str, str], TabularHealthModel] = {}
        self._hybrid: HybridSOHModel | None = None
        self._hybrid_config: dict[str, Any] | None = None
        self._hybrid_manifest: dict[str, Any] | None = None
        self._rul = None

    # ── metadata ─────────────────────────────────────────────────────────

    def _route(self, dataset_id: str, target: str) -> dict[str, Any]:
        key = (dataset_id, target)
        if key in self._routes:
            return self._routes[key]
        datasets = self._health_report["datasets"]
        if dataset_id not in datasets:
            raise KeyError(
                f"Unknown dataset_id {dataset_id!r}; expected one of {sorted(datasets)}"
            )
        targets = datasets[dataset_id]["targets"]
        if target not in targets:
            raise KeyError(
                f"Unknown target {target!r} for {dataset_id}; expected one of "
                f"{sorted(targets)}"
            )
        entry = targets[target]
        artifact_dir = WORKSPACE / entry["artifact_dir"]
        schema = json.loads(
            (artifact_dir / "model_schema.json").read_text(encoding="utf-8")
        )
        config = json.loads(
            (artifact_dir / "model_config.json").read_text(encoding="utf-8")
        )
        provenance = datasets[dataset_id]["provenance"]
        chemistry_labels = self._compatibility["datasets"][dataset_id][
            "compatibility"
        ]["chemistry_labels"]
        evaluation = entry["evaluation"]
        route = {
            "dataset_id": dataset_id,
            "target": target,
            "artifact_dir": artifact_dir,
            "feature_columns": list(schema["feature_columns"]),
            "numeric_features": list(schema["numeric_features"]),
            "categorical_features": list(schema["categorical_features"]),
            "chemistry_column": schema.get("chemistry_column"),
            "forbidden_columns": sorted(
                {*config.get("forbidden_columns", ()), target}
            ),
            "allowed_chemistry": list(chemistry_labels),
            "unit": str(entry["unit"]).split(" (")[0],
            "unit_detail": entry["unit"],
            "readiness": entry["readiness"],
            "label_origin": entry["label_origin"],
            "target_kind": entry["target_kind"],
            "schema_version": schema.get("schema_version", "unknown"),
            "evaluation": evaluation,
            "provenance": provenance,
        }
        self._routes[key] = route
        return route

    def available_routes(self) -> list[dict[str, Any]]:
        routes = []
        for dataset_id, dataset in self._health_report["datasets"].items():
            for target in dataset["targets"]:
                route = self._route(dataset_id, target)
                routes.append(
                    {
                        "dataset_id": dataset_id,
                        "target": target,
                        "unit": route["unit"],
                        "readiness": route["readiness"],
                        "label_origin": route["label_origin"],
                        "feature_count": len(route["feature_columns"]),
                    }
                )
        return routes

    def requirements(self, dataset_id: str, target: str) -> dict[str, Any]:
        route = self._route(dataset_id, target)
        return {
            "dataset_id": dataset_id,
            "target": target,
            "feature_columns": route["feature_columns"],
            "numeric_features": route["numeric_features"],
            "categorical_features": route["categorical_features"],
            "chemistry_column": route["chemistry_column"],
            "forbidden_columns": route["forbidden_columns"],
            "allowed_chemistry": route["allowed_chemistry"],
            "unit": route["unit"],
            "readiness": route["readiness"],
            "label_origin": route["label_origin"],
            "target_kind": route["target_kind"],
            "feature_count": len(route["feature_columns"]),
        }

    def hybrid_requirements(self) -> dict[str, Any]:
        config = self._load_hybrid_config()
        manifest = self._load_hybrid_manifest()
        return {
            "feature_columns": list(config["feature_columns"]),
            "sequence_length": int(config["sequence_length"]),
            "entity_column": config["entity_column"],
            "order_column": config["order_column"],
            "target": config["target_column"],
            "unit": "%",
            "readiness": "EXPERIMENTAL",
            "production_ready": False,
            "benchmark_warning": manifest.get("benchmark_warning"),
        }

    # ── loading ──────────────────────────────────────────────────────────

    def _load_health(self, route: dict[str, Any]) -> TabularHealthModel:
        key = (route["dataset_id"], route["target"])
        if key not in self._models:
            self._models[key] = TabularHealthModel.load(route["artifact_dir"])
        return self._models[key]

    def _load_hybrid_config(self) -> dict[str, Any]:
        if self._hybrid_config is None:
            path = HYBRID_MODEL_DIR / "hybrid_config.json"
            if not path.is_file():
                raise FileNotFoundError(f"Hybrid model config not found: {path}")
            self._hybrid_config = json.loads(path.read_text(encoding="utf-8"))
        return self._hybrid_config

    def _load_hybrid_manifest(self) -> dict[str, Any]:
        if self._hybrid_manifest is None:
            if not HYBRID_MANIFEST.is_file():
                raise FileNotFoundError(f"Hybrid manifest not found: {HYBRID_MANIFEST}")
            self._hybrid_manifest = json.loads(
                HYBRID_MANIFEST.read_text(encoding="utf-8")
            )
        return self._hybrid_manifest

    def _load_hybrid(self) -> HybridSOHModel:
        if self._hybrid is None:
            self._hybrid = HybridSOHModel.load(HYBRID_MODEL_DIR)
        return self._hybrid

    # ── health prediction ────────────────────────────────────────────────

    def _prepare_health(
        self,
        records: list[dict[str, Any]],
        route: dict[str, Any],
        chemistry: str | None,
    ) -> tuple[pd.DataFrame, dict[int, dict[str, Any]], set[str], set[str], int]:
        feature_columns = route["feature_columns"]
        numeric = set(route["numeric_features"])
        chemistry_column = route["chemistry_column"]
        allowed = set(route["allowed_chemistry"])
        columns = list(feature_columns)
        if chemistry_column and chemistry_column not in columns:
            columns.append(chemistry_column)

        rows: list[dict[str, Any]] = []
        row_status: dict[int, dict[str, Any]] = {}
        missing_union: set[str] = set()
        ignored: set[str] = set()

        for row_index, record in enumerate(records):
            errors: list[str] = []
            leaked = sorted(
                key
                for key, value in record.items()
                if key in route["forbidden_columns"] and not is_missing(value)
            )
            if leaked:
                errors.append(
                    f"Target/leakage fields cannot be supplied as inputs: {leaked}"
                )

            supplied_chemistry = chemistry
            if "chemistry" in record and not is_missing(record["chemistry"]):
                supplied_chemistry = str(record["chemistry"])
            if supplied_chemistry is not None:
                supplied_chemistry = str(supplied_chemistry)
                if supplied_chemistry not in allowed:
                    errors.append(
                        f"Unsupported chemistry {supplied_chemistry!r} for "
                        f"{route['dataset_id']}; allowed: {sorted(allowed)}"
                    )

            row: dict[str, Any] = {}
            for column in feature_columns:
                if column == chemistry_column and supplied_chemistry is not None:
                    row[column] = supplied_chemistry
                elif column in record and not is_missing(record[column]):
                    try:
                        if column in numeric:
                            row[column] = coerce_numeric(record[column], column)
                        else:
                            row[column] = coerce_categorical(record[column], column)
                    except CustomInputError as error:
                        errors.append(str(error))
                        row[column] = None
                else:
                    row[column] = None
            if chemistry_column and chemistry_column not in feature_columns:
                row[chemistry_column] = supplied_chemistry

            known = set(feature_columns) | {chemistry_column, "chemistry"}
            for key in record:
                if key not in known and key not in route["forbidden_columns"]:
                    ignored.add(key)

            missing = [column for column in feature_columns if row.get(column) is None]
            missing_union.update(missing)
            if errors:
                row_status[row_index] = {
                    "row": row_index,
                    "status": "error",
                    "error": "; ".join(errors),
                }
            else:
                rows.append(row)
                row_status[row_index] = {
                    "row": row_index,
                    "status": "ok",
                    "missing_feature_count": len(missing),
                    "missing_features": missing,
                }

        frame = pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)
        return frame, row_status, missing_union, ignored, len(rows)

    @staticmethod
    def _clip_prediction(target: str, value: float) -> tuple[float, bool]:
        if not math.isfinite(value):
            raise CustomInputError("model produced a non-finite prediction")
        if target in {"state_of_health", "state_of_health_pct"}:
            clipped = min(100.0, max(0.0, value))
        elif target in CAPACITY_TARGETS:
            clipped = max(0.0, value)
        else:
            clipped = value
        return float(round(clipped, 4)), clipped != value

    def _route_warnings(self, route: dict[str, Any]) -> list[str]:
        warnings = [SNAPSHOT_WARNING]
        if route["readiness"] == "RESEARCH_ONLY" or route["label_origin"] == "synthetic":
            warnings.append(SYNTHETIC_WARNING)
        if route["dataset_id"] == "SOLID_STATE_TELEMETRY_2026":
            warnings.append(SOLID_STATE_WARNING)
        if route["unit"] == "kWh":
            warnings.append(
                "Capacity is measured pack energy in kWh, not cell capacity in Ah."
            )
        return warnings

    def _provenance(self, route: dict[str, Any]) -> dict[str, Any]:
        evaluation = route["evaluation"]
        return {
            "model_artifact": route["artifact_dir"].relative_to(WORKSPACE).as_posix(),
            "model_type": "tabular_lightgbm_regressor",
            "schema_version": route["schema_version"],
            "target_column": route["target"],
            "feature_count": len(route["feature_columns"]),
            "source_file": route["provenance"]["source_file"],
            "source_sha256": route["provenance"]["source_sha256"],
            "processed_sha256": route["provenance"]["processed_sha256"],
            "training_rows": int(evaluation["counts"]["rows"]),
            "test_mae": float(evaluation["model"]["mae"]),
            "test_n": int(evaluation["model"]["n"]),
            "split_protocol": evaluation["split_protocol"],
            "read_only": True,
        }

    def predict(
        self,
        dataset_id: str,
        target: str,
        records: Any,
        *,
        chemistry: str | None = None,
    ) -> dict[str, Any]:
        route = self._route(dataset_id, target)
        normalized = normalize_records(records)
        frame, row_status, missing, ignored, valid_count = self._prepare_health(
            normalized, route, chemistry
        )

        predictions: dict[int, dict[str, Any]] = {}
        if valid_count:
            model = self._load_health(route)
            values = model.predict(frame)
            valid_rows = [
                index
                for index in range(len(normalized))
                if row_status[index]["status"] == "ok"
            ]
            for row_index, value in zip(valid_rows, values):
                try:
                    clipped_value, was_clipped = self._clip_prediction(
                        route["target"], float(value)
                    )
                except CustomInputError as error:
                    row_status[row_index] = {
                        "row": row_index,
                        "status": "error",
                        "error": str(error),
                    }
                    continue
                predictions[row_index] = {
                    "row": row_index,
                    "status": "ok",
                    "prediction": clipped_value,
                    "clipped": was_clipped,
                    "missing_feature_count": row_status[row_index][
                        "missing_feature_count"
                    ],
                }

        output_rows = []
        for row_index in range(len(normalized)):
            if row_index in predictions:
                output_rows.append(predictions[row_index])
            else:
                output_rows.append(row_status[row_index])

        ok_count = sum(row["status"] == "ok" for row in output_rows)
        if ok_count == len(normalized):
            status = "ok"
        elif ok_count == 0:
            status = "error"
        else:
            status = "partial"

        warnings = self._route_warnings(route)
        if missing:
            warnings.append(
                f"{len(missing)} required feature(s) were not supplied and use "
                f"training-fitted imputation: {sorted(missing)}"
            )
        if ignored:
            warnings.append(f"Ignored unrecognized fields: {sorted(ignored)}")

        feature_count = len(route["feature_columns"])
        if valid_count and feature_count:
            present = sum(
                feature_count - row_status[index]["missing_feature_count"]
                for index in range(len(normalized))
                if row_status[index]["status"] == "ok"
            )
            completeness = present / (valid_count * feature_count)
        else:
            completeness = 0.0

        return {
            "status": status,
            "task": "regression",
            "dataset_id": dataset_id,
            "target": target,
            "unit": route["unit"],
            "prediction_kind": "model_prediction",
            "capacity_kind": "predicted" if target in CAPACITY_TARGETS else None,
            "readiness": route["readiness"],
            "label_origin": route["label_origin"],
            "target_kind": route["target_kind"],
            "production_ready": False,
            "completeness": round(float(completeness), 4),
            "missing_features": sorted(missing),
            "warnings": warnings,
            "predictions": output_rows,
            "provenance": self._provenance(route),
        }

    # ── algebraic capacity ───────────────────────────────────────────────

    def predict_capacity_from_soh(
        self,
        dataset_id: str,
        records: Any,
        *,
        chemistry: str | None = None,
    ) -> dict[str, Any]:
        semantics = capacity_semantics(dataset_id)
        algebraic = semantics["algebraic"]
        if algebraic is None:
            return {
                "status": "unsupported",
                "dataset_id": dataset_id,
                "prediction_kind": "algebraic",
                "reason": (
                    "No rated nominal capacity column exists for this dataset, so "
                    "capacity cannot be derived algebraically from SOH."
                ),
                "warnings": semantics["notes"],
            }
        soh_target = SOH_TARGETS[dataset_id]
        normalized = normalize_records(records)
        soh_result = self.predict(
            dataset_id, soh_target, normalized, chemistry=chemistry
        )

        predictions: dict[int, dict[str, Any]] = {}
        for row in soh_result["predictions"]:
            row_index = int(row["row"])
            if row["status"] != "ok":
                predictions[row_index] = row
                continue
            source_column = algebraic["source_column"]
            supplied = normalized[row_index].get(source_column)
            if is_missing(supplied):
                predictions[row_index] = {
                    "row": row_index,
                    "status": "error",
                    "error": (
                        f"{source_column} is required to derive capacity "
                        "algebraically from SOH."
                    ),
                }
                continue
            try:
                rated = coerce_numeric(supplied, source_column)
                derived = derive_capacity_from_soh(
                    row["prediction"],
                    rated,
                    unit=algebraic["unit"],
                    scope=algebraic["scope"],
                    is_pack_energy=algebraic["is_pack_energy"],
                )
            except (CustomInputError, ValueError) as error:
                predictions[row_index] = {
                    "row": row_index,
                    "status": "error",
                    "error": str(error),
                }
                continue
            predictions[row_index] = {
                "row": row_index,
                "status": "ok",
                "capacity": float(round(derived["capacity"], 4)),
                "soh_pct": derived["soh_pct"],
                "rated_capacity": derived["rated_capacity"],
                "unit": derived["unit"],
                "scope": derived["scope"],
                "kind": derived["kind"],
                "basis": derived["basis"],
                "is_measured": derived["is_measured"],
                "is_predicted": derived["is_predicted"],
                "is_pack_energy": derived["is_pack_energy"],
            }

        output_rows = [predictions[index] for index in range(len(normalized))]
        ok_count = sum(row["status"] == "ok" for row in output_rows)
        if ok_count == len(normalized):
            status = "ok"
        elif ok_count == 0:
            status = "error"
        else:
            status = "partial"

        return {
            "status": status,
            "dataset_id": dataset_id,
            "soh_target": soh_target,
            "prediction_kind": "algebraic",
            "capacity_kind": "algebraic",
            "unit": algebraic["unit"],
            "warnings": [
                *soh_result["warnings"],
                "Algebraic capacity is SOH x rated capacity; it is not a "
                "measurement and not independently validated.",
            ],
            "predictions": output_rows,
            "provenance": soh_result["provenance"],
        }

    # ── RUL ──────────────────────────────────────────────────────────────

    def _rul_service(self):
        from member2_modeling.rul.service import RULService

        if getattr(self, "_rul", None) is None:
            self._rul = RULService()
        return self._rul

    def rul_metadata(self) -> dict[str, Any]:
        return self._rul_service().metadata()

    def rul_support(self) -> dict[str, Any]:
        return self._rul_service().support_matrix()

    def rul_models(self) -> list[str]:
        return self._rul_service().available_models()

    def predict_rul(
        self,
        dataset_id: str = "LEGACY_NMC_SEQUENTIAL",
        *,
        chemistry: str | None = None,
        cell_id: str | None = None,
        history: Any = None,
        model: str = "auto",
        eol_threshold_pct: float = 80.0,
    ) -> dict[str, Any]:
        """Return experimental RUL for a genuine sequence or stored artifact.

        ``history`` accepts a list of ``{"cycle_index": int, "soh_pct": float}``
        rows. When omitted, the frozen retrospective predictions for the
        requested sequential dataset are returned. Snapshot datasets and
        insufficient sequences return ``status="unavailable"``.
        """
        service = self._rul_service()
        if history is not None:
            return service.predict_from_history(
                history,
                battery_id=cell_id or "CUSTOM",
                chemistry=chemistry or "NMC",
                model=model,
                eol_threshold_pct=eol_threshold_pct,
            )
        result = service.dataset_predictions(dataset_id)
        if chemistry is not None:
            result["chemistry"] = chemistry
        return result

    # ── sequence prediction ──────────────────────────────────────────────

    def predict_sequence(
        self,
        records: Any,
        *,
        chemistry: str = "NMC",
        cell_id: str = "CUSTOM",
    ) -> dict[str, Any]:
        chem = str(chemistry).upper()
        if chem != "NMC":
            return {
                "status": "unsupported",
                "chemistry": chemistry,
                "reason": (
                    "The hybrid sequence model is trained on legacy NMC histories "
                    f"only; chemistry {chemistry!r} is not supported. NMC is required."
                ),
                "predictions": [],
            }
        normalized = normalize_records(records)
        config = self._load_hybrid_config()
        manifest = self._load_hybrid_manifest()
        features = list(config["feature_columns"])
        entity_column = config["entity_column"]
        order_column = config["order_column"]
        sequence_length = int(config["sequence_length"])

        rows: list[dict[str, Any]] = []
        keys: set[tuple[str, int]] = set()
        missing_union: set[str] = set()
        for row_index, record in enumerate(normalized):
            row: dict[str, Any] = {}
            for column in features:
                if column in record and not is_missing(record[column]):
                    try:
                        row[column] = coerce_numeric(record[column], column)
                    except CustomInputError as error:
                        raise CustomInputError(
                            f"Record {row_index}: {error}"
                        ) from error
                else:
                    row[column] = np.nan
                    missing_union.add(column)
            entity = record.get(entity_column, cell_id)
            if is_missing(entity):
                raise CustomInputError(
                    f"Record {row_index}: {entity_column} must not be empty"
                )
            order_value = record.get(order_column, row_index)
            if isinstance(order_value, (bool, np.bool_)):
                raise CustomInputError(
                    f"Record {row_index}: {order_column} must be an integer"
                )
            try:
                order = int(order_value)
            except (TypeError, ValueError) as error:
                raise CustomInputError(
                    f"Record {row_index}: {order_column} must be an integer, "
                    f"got {order_value!r}"
                ) from error
            key = (str(entity), order)
            if key in keys:
                raise CustomInputError(
                    f"Duplicate {entity_column}/{order_column} pair: {key}"
                )
            keys.add(key)
            row[entity_column] = str(entity)
            row[order_column] = order
            rows.append(row)

        frame = pd.DataFrame(rows, columns=[*features, entity_column, order_column])
        model = self._load_hybrid()
        output = model.predict(frame)

        predictions = []
        for row_index, record in enumerate(output.to_dict(orient="records")):
            fused, clipped = self._clip_prediction(
                "state_of_health", float(record["fused_pred"])
            )
            sequence_value = record["sequence_pred"]
            available = bool(record["sequence_available"])
            predictions.append(
                {
                    "row": row_index,
                    "cell_id": str(record[entity_column]),
                    "cycle_index": int(record[order_column]),
                    "prediction": fused,
                    "clipped": clipped,
                    "tabular_prediction": float(round(float(record["tabular_pred"]), 4)),
                    "sequence_prediction": (
                        None
                        if not np.isfinite(sequence_value)
                        else float(round(float(sequence_value), 4))
                    ),
                    "sequence_available": available,
                }
            )

        warnings = [
            "Hybrid readiness is EXPERIMENTAL. Persistence and training-mean-fade "
            "baselines outperform the fused model on the chronological test split.",
            "Rows without enough causal history use the learned tabular-only fallback.",
        ]
        if len(normalized) < sequence_length:
            warnings.append(
                f"Sequence history is shorter than {sequence_length} ordered records; "
                "every row uses the learned tabular-only fallback."
            )
        if missing_union:
            warnings.append(
                f"{len(missing_union)} sequence feature(s) were not supplied and use "
                f"training-fitted values: {sorted(missing_union)}"
            )

        return {
            "status": "ok",
            "task": "future_cycle_soh",
            "chemistry": chem,
            "unit": "%",
            "readiness": "EXPERIMENTAL",
            "production_ready": False,
            "sequence_length": sequence_length,
            "warnings": warnings,
            "predictions": predictions,
            "provenance": {
                "model_artifact": HYBRID_MODEL_DIR.relative_to(WORKSPACE).as_posix(),
                "manifest": HYBRID_MANIFEST.relative_to(WORKSPACE).as_posix(),
                "manifest_sha256": _sha256(HYBRID_MANIFEST),
                "training_source": manifest["training_dataset"]["source_file"],
                "training_source_sha256": manifest["training_dataset"][
                    "source_sha256"
                ],
                "sequence_length": sequence_length,
                "readiness": "EXPERIMENTAL",
                "production_ready": False,
                "benchmark_warning": manifest.get("benchmark_warning"),
                "read_only": True,
            },
        }


__all__ = ["BatteryIntelligenceAPI"]
