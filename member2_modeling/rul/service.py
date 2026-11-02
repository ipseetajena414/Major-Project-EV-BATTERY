"""Serving layer for experimental RUL estimates.

The deployed artifact is the selected candidate from the leave-one-cell-out
study.  Custom histories are accepted only when cycle ids and SOH values are
supplied together; the service never invents a cycle index.  Every response is
marked experimental and not approved for operational use.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from member2_modeling.analytics.sequential import WORKSPACE, source_sha256
from member2_modeling.rul.availability import rul_availability
from member2_modeling.rul.forecast import (
    FittedModel,
    RULForecastError,
    RULHistory,
    build_rul_record,
    make_history,
    predict_candidate,
    projected_curve_interval,
)
from member2_modeling.rul.labels import EOL_PCT

RUL_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "rul_v1"
BUNDLE_PATH = RUL_DIR / "rul_model.joblib"
REGISTRY_PATH = RUL_DIR / "rul_model_registry.json"
METRICS_PATH = RUL_DIR / "rul_metrics.json"
CURRENT_PATH = RUL_DIR / "rul_current_predictions.csv"
CENSORED_PATH = RUL_DIR / "rul_censored_projections.csv"
UNAVAILABLE_PATH = RUL_DIR / "rul_unavailable.csv"

REQUIRED_FIELDS = (
    "battery_id",
    "current_cycle",
    "current_soh_pct",
    "eol_threshold_pct",
    "predicted_eol_cycle",
    "estimated_rul_cycles",
    "lower_uncertainty_bound",
    "upper_uncertainty_bound",
    "model_id",
    "model_version",
    "chemistry",
    "prediction_status",
    "validation_status",
    "provenance",
)


class RULServiceError(ValueError):
    """Raised when an RUL request cannot be answered without fabrication."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def _records_from_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    frame = pd.read_csv(path)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        cleaned = {key: _clean(value) for key, value in row.items()}
        warnings = cleaned.get("warnings")
        if isinstance(warnings, str):
            try:
                parsed = ast.literal_eval(warnings)
                cleaned["warnings"] = list(parsed) if parsed else []
            except (ValueError, SyntaxError):
                cleaned["warnings"] = [warnings]
        cleaned["provenance"] = {
            "artifact": str(path.relative_to(WORKSPACE)),
            "source_sha256": source_sha256(),
            "retrospective": True,
        }
        records.append(cleaned)
    return records


class RULService:
    """Loads the frozen RUL artifact and answers prediction requests."""

    def __init__(self, artifact_dir: Path | str = RUL_DIR) -> None:
        self._dir = Path(artifact_dir)
        self._bundle: dict[str, Any] | None = None
        self._registry: dict[str, Any] | None = None

    # ── metadata ─────────────────────────────────────────────────────────

    def _load_bundle(self) -> dict[str, Any]:
        if self._bundle is None:
            path = self._dir / "rul_model.joblib"
            if not path.is_file():
                raise RULServiceError(
                    f"RUL artifact not found: {path}. Run "
                    "member2_modeling.rul.pipeline.build_rul_artifacts first."
                )
            self._bundle = joblib.load(path)
        return self._bundle

    def _load_registry(self) -> dict[str, Any]:
        if self._registry is None:
            path = self._dir / "rul_model_registry.json"
            if not path.is_file():
                raise RULServiceError(f"RUL registry not found: {path}")
            self._registry = json.loads(path.read_text(encoding="utf-8"))
        return self._registry

    def metadata(self) -> dict[str, Any]:
        bundle = self._load_bundle()
        registry = self._load_registry()
        return {
            "selected_model": bundle["model_id"],
            "model_version": bundle["model_version"],
            "trained_cells": bundle["trained_cells"],
            "chemistry": bundle["chemistry"],
            "eol_threshold_pct": bundle["eol_threshold_pct"],
            "calibration": bundle["calibration"],
            "selection": registry["selection"],
            "approved_for_use": False,
            "production_ready": False,
        }

    def metrics(self) -> dict[str, Any]:
        if not METRICS_PATH.is_file():
            raise RULServiceError(f"RUL metrics not found: {METRICS_PATH}")
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    def support_matrix(self) -> dict[str, Any]:
        return self.metrics()["support"]

    def available_models(self) -> list[str]:
        return sorted(self._load_bundle()["candidates"])

    def current_predictions(self) -> list[dict[str, Any]]:
        records = _records_from_csv(self._dir / "rul_current_predictions.csv")
        for record in records:
            record["provenance"].update(
                {
                    "artifact_sha256": _sha256_text(
                        (self._dir / "rul_current_predictions.csv").read_text(
                            encoding="utf-8"
                        )
                    ),
                    "note": (
                        "Retrospective out-of-fold record at a historical origin; "
                        "not a live fleet prediction."
                    ),
                }
            )
        return records

    def censored_predictions(self) -> list[dict[str, Any]]:
        return _records_from_csv(self._dir / "rul_censored_projections.csv")

    def unavailable_predictions(self) -> list[dict[str, Any]]:
        return _records_from_csv(self._dir / "rul_unavailable.csv")

    # ── prediction ───────────────────────────────────────────────────────

    def predict_from_history(
        self,
        history: Iterable[dict[str, Any]] | RULHistory,
        *,
        battery_id: str = "CUSTOM",
        dataset_id: str = "CUSTOM_SEQUENCE",
        chemistry: str = "NMC",
        model: str = "auto",
        eol_threshold_pct: float = EOL_PCT,
    ) -> dict[str, Any]:
        parsed = self._parse_history(history)
        if parsed.current_soh <= float(eol_threshold_pct):
            return self._post_threshold_response(
                parsed, battery_id, dataset_id, chemistry, model, eol_threshold_pct
            )
        bundle = self._load_bundle()
        model_name = bundle["model_id"] if model == "auto" else model
        candidates = bundle.get("candidates", {})
        if model_name not in candidates:
            raise RULServiceError(
                f"Unknown RUL model {model_name!r}; available: {sorted(candidates)}"
            )
        fitted = FittedModel.from_payload(candidates[model_name])
        point = predict_candidate(fitted, parsed, float(eol_threshold_pct))
        quantiles = bundle.get("calibration", {}).get("all_models", {})
        quantile = quantiles.get(model_name)
        lower = upper = None
        if point is not None and quantile is not None:
            lower = max(0.0, point - float(quantile))
            upper = point + float(quantile)

        cross_chemistry = str(chemistry).upper() != "NMC"
        validation_status = (
            "NOT_VALIDATED_FOR_CHEMISTRY"
            if cross_chemistry
            else "EXPERIMENTAL_LOCO_3_CELLS"
        )
        prediction_status = (
            "experimental_cross_chemistry"
            if cross_chemistry
            else "experimental_custom_history"
        )
        if point is None:
            prediction_status = "censored_no_estimate"
        history_hash = _sha256_text(
            "|".join(
                f"{int(cycle)}:{float(soh):.6f}"
                for cycle, soh in zip(parsed.cycle_ids, parsed.soh_pct)
            )
        )
        warnings = [
            "Custom history is treated as user-supplied data; its sensors and "
            "calibration are not verified by Member 2.",
        ]
        if cross_chemistry:
            warnings.append(
                f"The selected model was trained on NMC cells and is not "
                f"validated for chemistry {chemistry!r}."
            )
        if quantile is None:
            warnings.append(
                "No calibration residual was available, so no uncertainty "
                "interval is reported."
            )
        record = build_rul_record(
            battery_id=battery_id,
            history=parsed,
            eol_pct=float(eol_threshold_pct),
            point=point,
            lower=lower,
            upper=upper,
            model_id=model_name,
            chemistry=chemistry,
            dataset_id=dataset_id,
            prediction_status=prediction_status,
            validation_status=validation_status,
            provenance={
                "source": "user_supplied_history",
                "history_sha256": history_hash,
                "trained_cells": bundle["trained_cells"],
                "training_chemistry": bundle["chemistry"],
                "source_sha256": source_sha256(),
                "artifact": str(BUNDLE_PATH.relative_to(WORKSPACE)),
            },
            interval_method=(
                "cross_fitted_conformal_abs_residual_q90"
                if quantile is not None
                else None
            ),
            interval_coverage_target=0.90 if quantile is not None else None,
            warnings=warnings,
        )
        return self._envelope(
            dataset_id=dataset_id,
            chemistry=chemistry,
            readiness=(
                "EXPERIMENTAL_CROSS_CHEMISTRY"
                if cross_chemistry
                else "EXPERIMENTAL"
            ),
            validation_status=validation_status,
            model_name=model_name,
            predictions=[record],
            warnings=warnings,
            eol_threshold_pct=float(eol_threshold_pct),
        )

    def _post_threshold_response(
        self,
        history: RULHistory,
        battery_id: str,
        dataset_id: str,
        chemistry: str,
        model: str,
        eol_threshold_pct: float,
    ) -> dict[str, Any]:
        bundle = self._load_bundle()
        model_name = bundle["model_id"] if model == "auto" else model
        record = build_rul_record(
            battery_id=battery_id,
            history=history,
            eol_pct=float(eol_threshold_pct),
            point=0.0,
            lower=0.0,
            upper=0.0,
            model_id=model_name,
            chemistry=chemistry,
            dataset_id=dataset_id,
            prediction_status="post_threshold",
            validation_status="EXPERIMENTAL_LOCO_3_CELLS",
            provenance={
                "source": "user_supplied_history",
                "note": "Current SOH is already at or below the threshold.",
            },
            interval_method="none",
            warnings=[
                "The supplied history is already at or below the EOL threshold; "
                "remaining useful life is reported as zero cycles."
            ],
        )
        return self._envelope(
            dataset_id=dataset_id,
            chemistry=chemistry,
            readiness="EXPERIMENTAL",
            validation_status="EXPERIMENTAL_LOCO_3_CELLS",
            model_name=model_name,
            predictions=[record],
            warnings=record["warnings"],
            eol_threshold_pct=float(eol_threshold_pct),
        )

    def _envelope(
        self,
        *,
        dataset_id: str,
        chemistry: str,
        readiness: str,
        validation_status: str,
        model_name: str,
        predictions: list[dict[str, Any]],
        warnings: list[str],
        eol_threshold_pct: float,
    ) -> dict[str, Any]:
        bundle = self._load_bundle()
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "chemistry": chemistry,
            "approved_for_use": False,
            "production_ready": False,
            "unit": "observed_cycles",
            "readiness": readiness,
            "validation_status": validation_status,
            "eol_threshold_pct": float(eol_threshold_pct),
            "model": {
                "model_id": model_name,
                "model_version": bundle["model_version"],
                "trained_cells": bundle["trained_cells"],
                "selection": self._load_registry()["selection"],
            },
            "predictions": predictions,
            "warnings": [
                "RUL is experimental and not approved for operational use.",
                *warnings,
            ],
        }

    def dataset_predictions(self, dataset_id: str) -> dict[str, Any]:
        availability = rul_availability(dataset_id)
        if availability["status"] == "UNAVAILABLE":
            return {
                "status": "unavailable",
                "dataset_id": dataset_id,
                "chemistry": None,
                "approved_for_use": False,
                "unit": availability["unit"],
                "reason": availability["reason"],
                "predictions": [],
                "warnings": [
                    "No genuine sequential end-of-life evidence exists for this "
                    "dataset; no RUL is fabricated."
                ],
            }
        if availability["status"] == "EXPERIMENTAL_CENSORED":
            predictions = [
                record
                for record in self.censored_predictions()
                if record.get("dataset_id") == dataset_id
            ]
            return {
                "status": "ok",
                "dataset_id": dataset_id,
                "chemistry": "LFP",
                "approved_for_use": False,
                "production_ready": False,
                "unit": availability["unit"],
                "readiness": "EXPERIMENTAL_CENSORED",
                "validation_status": "EXPERIMENTAL_CENSORED_NO_EOL",
                "eol_threshold_pct": availability["eol_threshold_pct"],
                "model": self._load_registry()["selection"],
                "predictions": predictions,
                "warnings": [
                    "No EOL crossing was observed; these are censored projections "
                    "with no validated accuracy.",
                ],
            }
        predictions = self.current_predictions()
        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "chemistry": "NMC",
            "approved_for_use": False,
            "production_ready": False,
            "unit": availability["unit"],
            "readiness": "EXPERIMENTAL",
            "validation_status": "EXPERIMENTAL_LOCO_3_CELLS",
            "eol_threshold_pct": availability["eol_threshold_pct"],
            "model": self._load_registry()["selection"],
            "predictions": predictions,
            "warnings": [
                "Three NMC cells only; estimates are experimental and not "
                "approved for operational use.",
            ],
        }

    # ── parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_history(
        history: Iterable[dict[str, Any]] | RULHistory,
    ) -> RULHistory:
        if isinstance(history, RULHistory):
            return history
        rows = list(history)
        if not rows:
            raise RULServiceError("history must contain at least 8 cycles")
        cycle_ids: list[int] = []
        soh_values: list[float] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RULServiceError(f"history row {index} must be an object")
            cycle = row.get("cycle_index", row.get("cycle_id"))
            soh = row.get("soh_pct")
            if cycle is None:
                raise RULServiceError(
                    f"history row {index} is missing cycle_index/cycle_id"
                )
            if soh is None:
                raise RULServiceError(f"history row {index} is missing soh_pct")
            try:
                cycle_value = int(cycle)
                soh_value = float(soh)
            except (TypeError, ValueError) as exc:
                raise RULServiceError(
                    f"history row {index} has non-numeric cycle or SOH"
                ) from exc
            if not np.isfinite(soh_value):
                raise RULServiceError(f"history row {index} has non-finite SOH")
            cycle_ids.append(cycle_value)
            soh_values.append(soh_value)
        try:
            return make_history(cycle_ids, soh_values)
        except RULForecastError as exc:
            raise RULServiceError(str(exc)) from exc


def best_model_comparison() -> pd.DataFrame:
    """Tabular comparison of candidates from the frozen RUL evaluation."""
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    rows = []
    for name, block in metrics["candidates"].items():
        rows.append(
            {
                "model_id": name,
                "n_records": block["n_records"],
                "n_scored": block["n_scored"],
                "n_unavailable_predictions": block["n_unavailable_predictions"],
                "pre_threshold_mae_cycles": block["pre_threshold"]["mae"],
                "pre_threshold_rmse_cycles": block["pre_threshold"]["rmse"],
                "pre_threshold_bias_cycles": block["pre_threshold"]["bias"],
                "near_eol_mae_cycles": block["near_eol"]["mae"],
                "coverage": block["coverage"],
                "mean_interval_width": block["mean_interval_width"],
                "selected": name == metrics["selection"]["selected_model"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "pre_threshold_mae_cycles", na_position="last"
    )


def censored_interval_details(history: RULHistory, eol_pct: float = EOL_PCT) -> dict[str, Any]:
    _, _, details = projected_curve_interval(history, eol_pct)
    return details


__all__ = [
    "BUNDLE_PATH",
    "CENSORED_PATH",
    "CURRENT_PATH",
    "METRICS_PATH",
    "REGISTRY_PATH",
    "REQUIRED_FIELDS",
    "RUL_DIR",
    "RULService",
    "RULServiceError",
    "UNAVAILABLE_PATH",
    "best_model_comparison",
    "censored_interval_details",
]
