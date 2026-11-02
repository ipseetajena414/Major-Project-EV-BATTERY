"""Build RUL evaluation, predictions and provenance artifacts.

Primary validation is leave-one-cell-out over the three NASA NMC cells that
actually cross the 80% SOH threshold.  A within-cell chronological split is
reported as secondary evidence only.  The LFP cell that never reaches 80% is
exported as a censored projection, and datasets without genuine sequences are
exported as explicitly UNAVAILABLE.

The protected PADM test evaluation and sentinel are never read for fitting,
re-run, or bypassed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from member2_modeling.analytics.sequential import (
    MIN_HISTORY_CYCLES,
    WORKSPACE,
    cell_histories,
    cell_summary,
    load_cycles,
    sequential_support,
    source_sha256,
    support_status,
)
from member2_modeling.rul.evaluation import evaluate_rul
from member2_modeling.rul.forecast import (
    CANDIDATE_NAMES,
    MODEL_DESCRIPTIONS,
    MODEL_VERSION,
    FittedModel,
    Origin,
    RULForecastError,
    RULHistory,
    build_rul_record,
    conformal_quantile,
    fit_candidate,
    fit_padm_with_future,
    make_history,
    predict_candidate,
    projected_curve_interval,
)
from member2_modeling.rul.labels import EOL_PCT

ARTIFACT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "rul_v1"
NMC_DATASET = "NASA_Ames_PCoE"
LFP_DATASET = "LFP_MultiStat_B5"

PRIMARY_CELLS = ("B0005", "B0006", "B0018")
ORIGIN_STEP = 2
CALIBRATION_COVERAGE = 0.90
MIN_TRAIN_ORIGINS = 10

SUPPORT_LIMITATIONS = (
    "Only three NMC cells provide a genuine observed EOL crossing.",
    "One NMC cell (B0018) returns above 80% for several cycles after its first "
    "crossing; the first-crossing convention is audited separately.",
    "The 80% SOH threshold is a project assumption, not a manufacturer limit.",
    "Cycle ids in the source table are contiguous for these cells, so predicted "
    "EOL cycle ids are reported in source-cycle space.",
    "No LFP or Na-ion cell reaches 80% SOH, so no supervised RUL validation is "
    "possible for those chemistries.",
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
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _cell_data(dataset_id: str) -> dict[str, dict[str, Any]]:
    histories = cell_histories(dataset_id)
    return {
        cell_id: {
            **history,
            "history": make_history(history["cycle_ids"], history["soh"]),
        }
        for cell_id, history in histories.items()
    }


def _samples_for_cells(
    cells: list[str],
    data: dict[str, dict[str, Any]],
    *,
    eol_pct: float,
) -> tuple[list[Origin], list[tuple[RULHistory, float]]]:
    origins: list[Origin] = []
    padm_samples: list[tuple[RULHistory, float]] = []
    for cell_id in cells:
        entry = data.get(cell_id)
        if entry is None:
            continue
        history: RULHistory = entry["history"]
        soh = history.soh_pct
        below = soh < float(eol_pct)
        if not bool(below.any()):
            continue
        crossing = int(np.argmax(below))
        if crossing < MIN_HISTORY_CYCLES:
            continue
        eol_cycle = int(history.cycle_ids[crossing])
        for index in range(MIN_HISTORY_CYCLES - 1, crossing, ORIGIN_STEP):
            prefix = history.truncated(index)
            origins.append(
                Origin(
                    cell_id=cell_id,
                    dataset_id=entry["dataset_id"],
                    chemistry=entry["chemistry"],
                    history=prefix,
                    target_rul=float(eol_cycle - int(prefix.cycle_ids[-1])),
                    target_eol_cycle=eol_cycle,
                    crossing_position=crossing,
                )
            )
            if index + 1 < history.n:
                padm_samples.append(
                    (prefix, float(soh[index + 1] - soh[index]))
                )
    return origins, padm_samples


def _fit_all(
    origins: list[Origin],
    padm_samples: list[tuple[RULHistory, float]],
    eol_pct: float,
) -> dict[str, FittedModel]:
    fitted: dict[str, FittedModel] = {}
    for name in CANDIDATE_NAMES:
        try:
            if name == "padm_ar":
                fitted[name] = fit_padm_with_future(padm_samples)
            else:
                fitted[name] = fit_candidate(name, origins, eol_pct)
        except (RULForecastError, ValueError):
            continue
    return fitted


def _predict_all(
    models: dict[str, FittedModel],
    origins: list[Origin],
    eol_pct: float,
) -> dict[str, list[float | None]]:
    return {
        name: [predict_candidate(model, origin.history, eol_pct) for origin in origins]
        for name, model in models.items()
    }


def _calibration(
    train_cells: list[str],
    data: dict[str, dict[str, Any]],
    eol_pct: float,
) -> tuple[dict[str, float | None], dict[str, dict[str, Any]]]:
    """Cross-fitted absolute-residual quantiles inside the training cells."""
    residuals: dict[str, list[float]] = {}
    details: dict[str, dict[str, Any]] = {}
    for target_cell in train_cells:
        fit_cells = [cell for cell in train_cells if cell != target_cell]
        if not fit_cells:
            continue
        origins, padm_samples = _samples_for_cells(fit_cells, data, eol_pct=eol_pct)
        models = _fit_all(origins, padm_samples, eol_pct)
        target_origins, _ = _samples_for_cells(
            [target_cell], data, eol_pct=eol_pct
        )
        predictions = _predict_all(models, target_origins, eol_pct)
        for name, values in predictions.items():
            bucket = residuals.setdefault(name, [])
            for origin, value in zip(target_origins, values):
                if value is not None and origin.target_rul is not None:
                    bucket.append(float(value - origin.target_rul))
    quantiles = {
        name: conformal_quantile(values, CALIBRATION_COVERAGE)
        for name, values in residuals.items()
    }
    for name, values in residuals.items():
        array = np.asarray(values, dtype="float64")
        details[name] = {
            "n_residuals": int(array.size),
            "median_abs_residual": (
                float(np.median(np.abs(array))) if array.size else None
            ),
            "q90_abs_residual": quantiles[name],
            "coverage_target": CALIBRATION_COVERAGE,
        }
    return quantiles, details


def _loco_evaluation(
    data: dict[str, dict[str, Any]],
    cells: list[str],
    eol_pct: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    calibration_details: dict[str, dict[str, Any]] = {}
    for held_out in cells:
        train_cells = [cell for cell in cells if cell != held_out]
        if not train_cells:
            continue
        train_origins, train_padm = _samples_for_cells(
            train_cells, data, eol_pct=eol_pct
        )
        if len(train_origins) < MIN_TRAIN_ORIGINS:
            continue
        models = _fit_all(train_origins, train_padm, eol_pct)
        quantiles, details = _calibration(train_cells, data, eol_pct)
        for name, item in details.items():
            calibration_details.setdefault(name, {})[held_out] = item

        held_origins, _ = _samples_for_cells([held_out], data, eol_pct=eol_pct)
        for name, model in models.items():
            quantile = quantiles.get(name)
            for origin in held_origins:
                point = predict_candidate(model, origin.history, eol_pct)
                lower = upper = None
                if point is not None and quantile is not None:
                    lower = max(0.0, point - quantile)
                    upper = point + quantile
                truth = origin.target_rul
                records.append(
                    {
                        "validation_protocol": "leave_one_cell_out",
                        "fold": f"held_out::{held_out}",
                        "battery_id": held_out,
                        "dataset_id": origin.dataset_id,
                        "chemistry": origin.chemistry,
                        "model_id": name,
                        "model_version": MODEL_VERSION,
                        "origin_position": origin.origin_position,
                        "current_cycle": int(origin.history.current_cycle),
                        "current_soh_pct": float(origin.history.current_soh),
                        "eol_threshold_pct": float(eol_pct),
                        "actual_eol_cycle": origin.target_eol_cycle,
                        "actual_rul_cycles": truth,
                        "predicted_rul_cycles": point,
                        "predicted_eol_cycle": (
                            None
                            if point is None
                            else int(round(origin.history.current_cycle + point))
                        ),
                        "lower_uncertainty_bound": lower,
                        "upper_uncertainty_bound": upper,
                        "interval_coverage_target": (
                            CALIBRATION_COVERAGE if quantile is not None else None
                        ),
                        "interval_method": (
                            "cross_fitted_conformal_abs_residual_q90"
                            if quantile is not None
                            else None
                        ),
                        "residual_cycles": (
                            None if point is None else float(point - truth)
                        ),
                        "interval_covered": (
                            None
                            if point is None or lower is None
                            else bool(lower <= truth <= upper)
                        ),
                        "prediction_status": (
                            "post_threshold"
                            if origin.history.current_soh <= float(eol_pct)
                            else "experimental_loco_evaluation"
                        ),
                        "validation_status": "EXPERIMENTAL_LOCO_3_CELLS",
                        "training_cells": ";".join(train_cells),
                        "source_sha256": source_sha256(),
                    }
                )
    return records, calibration_details


def _chronological_evaluation(
    data: dict[str, dict[str, Any]],
    cells: list[str],
    eol_pct: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for cell_id in cells:
        entry = data.get(cell_id)
        if entry is None:
            continue
        origins, padm_samples = _samples_for_cells([cell_id], data, eol_pct=eol_pct)
        if len(origins) < MIN_TRAIN_ORIGINS:
            continue
        cut = max(MIN_TRAIN_ORIGINS, int(round(len(origins) * 0.6)))
        train_origins = origins[:cut]
        test_origins = origins[cut:]
        if not test_origins:
            continue
        train_padm = padm_samples[:cut]
        models = _fit_all(train_origins, train_padm, eol_pct)
        for name, model in models.items():
            for origin in test_origins:
                point = predict_candidate(model, origin.history, eol_pct)
                records.append(
                    {
                        "validation_protocol": "within_cell_chronological",
                        "fold": f"within::{cell_id}",
                        "battery_id": cell_id,
                        "dataset_id": origin.dataset_id,
                        "chemistry": origin.chemistry,
                        "model_id": name,
                        "model_version": MODEL_VERSION,
                        "origin_position": origin.origin_position,
                        "current_cycle": int(origin.history.current_cycle),
                        "current_soh_pct": float(origin.history.current_soh),
                        "eol_threshold_pct": float(eol_pct),
                        "actual_rul_cycles": origin.target_rul,
                        "predicted_rul_cycles": point,
                        "residual_cycles": (
                            None
                            if point is None
                            else float(point - origin.target_rul)
                        ),
                        "prediction_status": (
                            "post_threshold"
                            if origin.history.current_soh <= float(eol_pct)
                            else "experimental_chronological_evaluation"
                        ),
                        "validation_status": "EXPERIMENTAL_WITHIN_CELL",
                        "n_train_origins": len(train_origins),
                        "n_test_origins": len(test_origins),
                        "source_sha256": source_sha256(),
                    }
                )
    return records


def _metric_block(records: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in records if row.get("predicted_rul_cycles") is not None]
    truth = np.asarray([row["actual_rul_cycles"] for row in usable], dtype="float64")
    prediction = np.asarray(
        [row["predicted_rul_cycles"] for row in usable], dtype="float64"
    )
    failures = len(records) - len(usable)
    if truth.size == 0:
        return {
            "n_records": len(records),
            "n_scored": 0,
            "n_unavailable_predictions": failures,
            "pre_threshold": {"n": 0, "mae": None, "rmse": None, "bias": None},
            "coverage": None,
            "mean_interval_width": None,
        }
    evaluated = evaluate_rul(truth, prediction)
    covered = [
        row["interval_covered"]
        for row in usable
        if row.get("interval_covered") is not None
    ]
    widths = [
        row["upper_uncertainty_bound"] - row["lower_uncertainty_bound"]
        for row in usable
        if row.get("upper_uncertainty_bound") is not None
        and row.get("lower_uncertainty_bound") is not None
    ]
    return {
        "n_records": len(records),
        "n_scored": int(truth.size),
        "n_unavailable_predictions": failures,
        "pre_threshold": evaluated["pre_threshold"],
        "overall": evaluated["overall"],
        "near_eol": evaluated["near_eol"],
        "buckets": evaluated["buckets"],
        "coverage": float(np.mean(covered)) if covered else None,
        "coverage_n": len(covered),
        "mean_interval_width": float(np.mean(widths)) if widths else None,
    }


def _summarize(
    records: list[dict[str, Any]],
    cells: list[str],
) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for name in CANDIDATE_NAMES:
        subset = [row for row in records if row["model_id"] == name]
        if not subset:
            continue
        per_battery = {
            cell: _metric_block([row for row in subset if row["battery_id"] == cell])
            for cell in cells
        }
        by_model[name] = {
            **_metric_block(subset),
            "description": MODEL_DESCRIPTIONS.get(name),
            "per_battery": per_battery,
        }
    return by_model


def _select_model(by_model: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    eligible = {
        name: block
        for name, block in by_model.items()
        if block["pre_threshold"]["mae"] is not None
        and block["n_scored"] >= 0.9 * block["n_records"]
    }
    if not eligible:
        raise RULForecastError("No RUL candidate produced scorable pre-threshold rows")

    best_name, best_block = min(
        eligible.items(),
        key=lambda item: (
            float(item[1]["pre_threshold"]["mae"]),
            float(item[1]["pre_threshold"]["rmse"] or item[1]["pre_threshold"]["mae"]),
            CANDIDATE_NAMES.index(item[0]),
        ),
    )
    best_mae = float(best_block["pre_threshold"]["mae"])
    tolerance = max(0.10, 0.05 * best_mae)
    simpler = [
        (name, block)
        for name, block in eligible.items()
        if CANDIDATE_NAMES.index(name) < CANDIDATE_NAMES.index(best_name)
        and float(block["pre_threshold"]["mae"]) <= best_mae + tolerance
    ]
    if simpler:
        selected_name, selected = min(
            simpler, key=lambda item: CANDIDATE_NAMES.index(item[0])
        )
        rule = (
            "prefer the simplest candidate whose pre-threshold MAE is within "
            f"{tolerance:.2f} cycles of the best candidate"
        )
    else:
        selected_name, selected = best_name, best_block
        rule = "lowest pooled leave-one-cell-out pre-threshold MAE"
    return selected_name, {
        "selected_model": selected_name,
        "best_single_model": best_name,
        "best_single_model_mae": best_mae,
        "selection_protocol": rule,
        "selection_metric": selected["pre_threshold"]["mae"],
        "selection_rmse": selected["pre_threshold"]["rmse"],
        "simplicity_tolerance_cycles": tolerance,
        "complexity_tiebreak": list(CANDIDATE_NAMES),
        "eligible_models": sorted(eligible),
    }


def _final_bundle(
    data: dict[str, dict[str, Any]],
    cells: list[str],
    selected_name: str,
    eol_pct: float,
) -> dict[str, Any]:
    origins, padm_samples = _samples_for_cells(cells, data, eol_pct=eol_pct)
    models = _fit_all(origins, padm_samples, eol_pct)
    if selected_name not in models:
        raise RULForecastError(f"Selected model {selected_name!r} could not be fitted")
    quantiles, details = _calibration(cells, data, eol_pct)
    bundle = {
        "model": models[selected_name].to_payload(),
        "model_id": selected_name,
        "model_version": MODEL_VERSION,
        "candidates": {
            name: model.to_payload() for name, model in models.items()
        },
        "calibration": {
            "coverage_target": CALIBRATION_COVERAGE,
            "q90_abs_residual": quantiles.get(selected_name),
            "details": details.get(selected_name),
            "all_models": quantiles,
        },
        "trained_cells": list(cells),
        "chemistry": next(iter(data.values()))["chemistry"],
        "eol_threshold_pct": float(eol_pct),
    }
    return bundle


def _reporting_records(
    loco_records: list[dict[str, Any]],
    data: dict[str, dict[str, Any]],
    selected_name: str,
    eol_pct: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for cell_id, entry in data.items():
        soh = entry["history"].soh_pct
        below = soh < float(eol_pct)
        if not bool(below.any()):
            continue
        crossing = int(np.argmax(below))
        if crossing < MIN_HISTORY_CYCLES:
            continue
        target_index = int(round(crossing * 0.5))
        subset = [
            row
            for row in loco_records
            if row["battery_id"] == cell_id
            and row["model_id"] == selected_name
            and row["origin_position"] <= target_index
        ]
        if not subset:
            continue
        chosen = max(subset, key=lambda row: row["origin_position"])
        record = build_rul_record(
            battery_id=cell_id,
            history=entry["history"].truncated(chosen["origin_position"]),
            eol_pct=eol_pct,
            point=chosen["predicted_rul_cycles"],
            lower=chosen["lower_uncertainty_bound"],
            upper=chosen["upper_uncertainty_bound"],
            model_id=selected_name,
            chemistry=entry["chemistry"],
            dataset_id=entry["dataset_id"],
            prediction_status="retrospective_experimental",
            validation_status="EXPERIMENTAL_LOCO_3_CELLS",
            provenance={
                "source": str(WORKSPACE / "master_table.parquet"),
                "source_sha256": source_sha256(),
                "validation_protocol": "leave_one_cell_out",
                "fold": chosen["fold"],
                "reporting_origin_rule": "midpoint of pre-EOL origin span",
                "actual_rul_cycles": chosen["actual_rul_cycles"],
                "actual_eol_cycle": chosen["actual_eol_cycle"],
                "interval_method": "cross_fitted_conformal_abs_residual_q90",
                "interval_coverage_target": CALIBRATION_COVERAGE,
            },
            interval_method="cross_fitted_conformal_abs_residual_q90",
            interval_coverage_target=CALIBRATION_COVERAGE,
            warnings=[
                "This record is a retrospective out-of-fold estimate at a "
                "historical origin, not a live fleet prediction.",
                "Validation uses only three NMC cells.",
            ],
        )
        output.append(record)
    return output


def _censored_record(
    cell_id: str,
    entry: dict[str, Any],
    eol_pct: float,
) -> dict[str, Any]:
    history: RULHistory = entry["history"]
    linear = predict_candidate(
        FittedModel(name="linear_extrapolation", kind="within_history", estimator=None),
        history,
        eol_pct,
    )
    lower, upper, interval = projected_curve_interval(history, eol_pct)
    point = linear
    record = build_rul_record(
        battery_id=cell_id,
        history=history,
        eol_pct=eol_pct,
        point=point,
        lower=lower,
        upper=upper,
        model_id="linear_extrapolation",
        chemistry=entry["chemistry"],
        dataset_id=entry["dataset_id"],
        prediction_status="censored_projection",
        validation_status="EXPERIMENTAL_CENSORED_NO_EOL",
        provenance={
            "source": str(WORKSPACE / "master_table.parquet"),
            "source_sha256": source_sha256(),
            "validation_protocol": "none_no_observed_eol",
            "censored": True,
            "observed_cycles_without_eol": history.n,
            "interval_details": interval,
        },
        interval_method=interval.get("method"),
        interval_coverage_target=None,
        warnings=[
            "No EOL crossing was observed for this cell; the projection is an "
            "extrapolation far beyond the observed history.",
            "The interval is a curve-family spread, not a calibrated confidence "
            "interval.",
        ],
    )
    record["provenance"]["censored_observed_cycles"] = history.n
    return record


def _unavailable_records(
    eol_pct: float,
) -> list[dict[str, Any]]:
    cycles = load_cycles()
    records: list[dict[str, Any]] = []
    summary = cell_summary(eol_pct=eol_pct)
    for row in summary.to_dict("records"):
        status = support_status(row)
        if status["status"] in {"EXPERIMENTAL", "EXPERIMENTAL_CENSORED"}:
            continue
        dataset_id = str(row["dataset_id"])
        cell_id = str(row["cell_id"])
        group = cycles.loc[
            (cycles["dataset_id"] == dataset_id) & (cycles["cell_id"] == cell_id)
        ].sort_values("cycle_id")
        if group.empty:
            continue
        soh = group["soh_pct"].to_numpy(dtype="float64")
        records.append(
            {
                "battery_id": cell_id,
                "dataset_id": dataset_id,
                "chemistry": str(row["chemistry"]),
                "current_cycle": int(group["cycle_id"].iloc[-1]),
                "current_soh_pct": float(soh[-1]),
                "eol_threshold_pct": float(eol_pct),
                "predicted_eol_cycle": None,
                "estimated_rul_cycles": None,
                "lower_uncertainty_bound": None,
                "upper_uncertainty_bound": None,
                "model_id": None,
                "model_version": None,
                "prediction_status": "unavailable_insufficient_or_no_eol_history",
                "validation_status": "UNAVAILABLE",
                "provenance": {
                    "source": str(WORKSPACE / "master_table.parquet"),
                    "source_sha256": source_sha256(),
                    "observed_cycles": int(len(soh)),
                    "reason": status["reason"],
                },
            }
        )
    return records


def _observed_trajectories() -> pd.DataFrame:
    cycles = load_cycles()
    return pd.DataFrame(
        {
            "record_type": "observed",
            "is_forecast": False,
            "is_measured": True,
            "battery_id": cycles["cell_id"],
            "dataset_id": cycles["dataset_id"],
            "chemistry": cycles["chemistry"],
            "cycle_id": cycles["cycle_id"],
            "cycle_position": cycles["cycle_position"],
            "soh_pct": cycles["soh_pct"],
            "model_id": None,
            "validated": True,
        }
    )


def _curve_forecast(
    cell_id: str,
    entry: dict[str, Any],
    eol_pct: float,
    model_name: str,
    max_future: int = 500,
) -> list[dict[str, Any]]:
    history: RULHistory = entry["history"]
    soh = history.soh_pct
    x = np.arange(soh.size, dtype="float64")
    if model_name == "exponential_extrapolation":
        values = np.log(soh)
        target = float(np.log(eol_pct))
    else:
        values = soh
        target = float(eol_pct)
    denominator = float(np.sum((x - x.mean()) ** 2))
    if denominator <= 0:
        return []
    slope = float(np.sum((x - x.mean()) * (values - values.mean())) / denominator)
    if not np.isfinite(slope) or slope >= -1e-9:
        return []
    intercept = float(values.mean()) - slope * float(x.mean())
    x_eol = (target - intercept) / slope
    steps = int(np.ceil(x_eol - (history.n - 1)))
    if steps <= 0:
        return []
    steps = min(steps, max_future)
    future_positions = np.arange(history.n, history.n + steps)
    future_soh = intercept + slope * future_positions
    if model_name == "exponential_extrapolation":
        future_soh = np.exp(future_soh)
    last_cycle = int(history.cycle_ids[-1])
    rows = []
    for offset, (position, value) in enumerate(
        zip(future_positions, future_soh), start=1
    ):
        rows.append(
            {
                "record_type": "forecast",
                "is_forecast": True,
                "is_measured": False,
                "battery_id": cell_id,
                "dataset_id": entry["dataset_id"],
                "chemistry": entry["chemistry"],
                "cycle_id": last_cycle + offset,
                "cycle_position": int(position),
                "soh_pct": float(max(0.0, value)),
                "model_id": model_name,
                "validated": False,
            }
        )
    return rows


def build_rul_artifacts(
    out_dir: Path | str = ARTIFACT_DIR,
    *,
    eol_pct: float = EOL_PCT,
) -> dict[str, Any]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    data = _cell_data(NMC_DATASET)
    cells = [cell for cell in PRIMARY_CELLS if cell in data]
    if len(cells) < 3:
        raise RULForecastError(
            f"Expected three NMC cells, found {cells}; LOCO validation unavailable"
        )

    loco_records, calibration_details = _loco_evaluation(data, cells, eol_pct)
    by_model = _summarize(loco_records, cells)
    selected_name, selection = _select_model(by_model)

    chronological = _chronological_evaluation(data, cells, eol_pct)
    chronological_by_model = {
        name: _metric_block([row for row in chronological if row["model_id"] == name])
        for name in CANDIDATE_NAMES
        if any(row["model_id"] == name for row in chronological)
    }
    chronological_per_battery = {
        name: {
            cell: _metric_block(
                [
                    row
                    for row in chronological
                    if row["model_id"] == name and row["battery_id"] == cell
                ]
            )
            for cell in cells
        }
        for name in chronological_by_model
    }

    bundle = _final_bundle(data, cells, selected_name, eol_pct)
    joblib.dump(bundle, destination / "rul_model.joblib")

    loco_frame = pd.DataFrame(loco_records)
    loco_frame.to_csv(destination / "rul_predictions.csv", index=False)

    reporting = _reporting_records(loco_records, data, selected_name, eol_pct)
    reporting_frame = pd.DataFrame(reporting)
    reporting_frame.to_csv(destination / "rul_current_predictions.csv", index=False)

    lfp_data = _cell_data(LFP_DATASET)
    censored: list[dict[str, Any]] = []
    for cell_id, entry in lfp_data.items():
        if entry["history"].n >= MIN_HISTORY_CYCLES:
            censored.append(_censored_record(cell_id, entry, eol_pct))
    censored_frame = pd.DataFrame(censored)
    censored_frame.to_csv(destination / "rul_censored_projections.csv", index=False)

    unavailable = _unavailable_records(eol_pct)
    unavailable_frame = pd.DataFrame(unavailable)
    unavailable_frame.to_csv(destination / "rul_unavailable.csv", index=False)

    per_battery_rows = []
    for name, block in by_model.items():
        for cell_id, metrics in block["per_battery"].items():
            per_battery_rows.append(
                {
                    "model_id": name,
                    "battery_id": cell_id,
                    "n_records": metrics["n_records"],
                    "n_scored": metrics["n_scored"],
                    "mae_cycles": metrics["pre_threshold"]["mae"],
                    "rmse_cycles": metrics["pre_threshold"]["rmse"],
                    "bias_cycles": metrics["pre_threshold"]["bias"],
                    "coverage": metrics["coverage"],
                    "mean_interval_width": metrics["mean_interval_width"],
                }
            )
    per_battery = pd.DataFrame(per_battery_rows).sort_values(
        ["model_id", "battery_id"]
    )
    per_battery.to_csv(destination / "rul_per_battery.csv", index=False)

    trajectory_rows: list[dict[str, Any]] = []
    observed = _observed_trajectories()
    trajectory_rows.append(observed)
    trajectory_rows.append(
        pd.DataFrame(
            [
                {
                    "record_type": "observed",
                    "is_forecast": False,
                    "is_measured": True,
                    "battery_id": cell_id,
                    "dataset_id": entry["dataset_id"],
                    "chemistry": entry["chemistry"],
                    "cycle_id": int(entry["history"].cycle_ids[-1]),
                    "cycle_position": int(entry["history"].n - 1),
                    "soh_pct": float(entry["history"].current_soh),
                    "model_id": None,
                    "validated": True,
                }
                for cell_id, entry in lfp_data.items()
            ]
        )
    )
    for cell_id, entry in data.items():
        for model_name in ("linear_extrapolation", "exponential_extrapolation"):
            forecast_rows = _curve_forecast(cell_id, entry, eol_pct, model_name)
            if forecast_rows:
                trajectory_rows.append(pd.DataFrame(forecast_rows))
    for cell_id, entry in lfp_data.items():
        for model_name in ("linear_extrapolation", "exponential_extrapolation"):
            forecast_rows = _curve_forecast(
                cell_id, entry, eol_pct, model_name, max_future=500
            )
            if forecast_rows:
                trajectory_rows.append(pd.DataFrame(forecast_rows))
    trajectories = pd.concat(trajectory_rows, ignore_index=True)
    trajectories.to_csv(destination / "rul_degradation_trajectories.csv", index=False)

    eol_rows = []
    for row in loco_records:
        if row["predicted_rul_cycles"] is None:
            continue
        eol_rows.append(
            {
                "battery_id": row["battery_id"],
                "dataset_id": row["dataset_id"],
                "chemistry": row["chemistry"],
                "model_id": row["model_id"],
                "validation_protocol": row["validation_protocol"],
                "origin_cycle": row["current_cycle"],
                "actual_eol_cycle": row["actual_eol_cycle"],
                "predicted_eol_cycle": row["predicted_eol_cycle"],
                "lower_eol_cycle": row["current_cycle"]
                + (row["lower_uncertainty_bound"] or 0.0),
                "upper_eol_cycle": row["current_cycle"]
                + (row["upper_uncertainty_bound"] or 0.0),
                "residual_cycles": row["residual_cycles"],
            }
        )
    pd.DataFrame(eol_rows).to_csv(
        destination / "rul_eol_trajectories.csv", index=False
    )

    code_paths = [
        WORKSPACE / "member2_modeling" / "analytics" / "sequential.py",
        WORKSPACE / "member2_modeling" / "rul" / "forecast.py",
        WORKSPACE / "member2_modeling" / "rul" / "pipeline.py",
        WORKSPACE / "member2_modeling" / "rul" / "labels.py",
        WORKSPACE / "member2_modeling" / "rul" / "evaluation.py",
    ]
    source_path = Path(str(WORKSPACE / "master_table.parquet"))
    registry = {
        "registry_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_model": selected_name,
        "model_version": MODEL_VERSION,
        "model_id": f"member2_rul::{selected_name}::{MODEL_VERSION}",
        "approved_for_use": False,
        "production_ready": False,
        "task": "remaining_useful_life",
        "unit": "observed_cycles",
        "eol_threshold_pct": float(eol_pct),
        "target_definition": (
            "Observed cycles from the prediction origin until the first cycle "
            "whose SOH is strictly below the configured threshold."
        ),
        "trained_cells": list(cells),
        "validation_protocol": "leave-one-cell-out",
        "candidates": {
            name: {
                "description": MODEL_DESCRIPTIONS.get(name),
                "metrics": by_model.get(name, {}).get("pre_threshold"),
                "coverage": by_model.get(name, {}).get("coverage"),
            }
            for name in by_model
        },
        "selection": selection,
        "calibration": bundle["calibration"],
        "artifacts": {
            name: {"sha256": _sha256(destination / name)}
            for name in (
                "rul_predictions.csv",
                "rul_current_predictions.csv",
                "rul_censored_projections.csv",
                "rul_per_battery.csv",
                "rul_degradation_trajectories.csv",
                "rul_eol_trajectories.csv",
                "rul_model.joblib",
            )
            if (destination / name).is_file()
        },
        "source": {
            "path": str(source_path),
            "sha256": source_sha256(),
        },
        "code": {
            str(path.relative_to(WORKSPACE)): _sha256(path)
            for path in code_paths
            if path.is_file()
        },
        "limitations": list(SUPPORT_LIMITATIONS),
    }
    _write_json(destination / "rul_model_registry.json", registry)

    metrics = {
        "generated_at": registry["generated_at"],
        "eol_threshold_pct": float(eol_pct),
        "unit": "observed_cycles",
        "primary_protocol": "leave_one_cell_out",
        "secondary_protocol": "within_cell_chronological",
        "n_cells": len(cells),
        "n_loco_records": len(loco_records),
        "candidates": by_model,
        "chronological_candidates": chronological_by_model,
        "chronological_per_battery": chronological_per_battery,
        "selection": selection,
        "calibration": {
            name: details for name, details in calibration_details.items()
        },
        "support": sequential_support(eol_pct=eol_pct),
        "comparability_warning": (
            "MAE is only directly comparable between models with the same "
            "n_scored. PADM-AR and last-delta persistence return no estimate at "
            "some origins; n_unavailable_predictions records those failures."
        ),
        "insufficient_sample_statement": (
            "There are three independent NMC cells and one of them crosses EOL "
            "after 44 pre-EOL cycles. Error bars are correspondingly wide and "
            "no VALIDATED status is claimed."
        ),
        "limitations": list(SUPPORT_LIMITATIONS),
    }
    _write_json(destination / "rul_metrics.json", metrics)

    provenance = {
        "generated_at": registry["generated_at"],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source": registry["source"],
        "code": registry["code"],
        "selection": selection,
        "validation": {
            "primary": "leave-one-cell-out over cells",
            "cells": list(cells),
            "secondary": "within-cell chronological 60/40 split",
            "origin_step_cycles": ORIGIN_STEP,
            "min_history_cycles": MIN_HISTORY_CYCLES,
            "causal_rule": (
                "Every prediction uses only cycles at or before the origin; "
                "future SOH is never an input."
            ),
        },
        "excluded": {
            "protected_padm_test_evaluation": (
                "Read-only and not re-run; not used for fitting or selection."
            ),
            "snapshot_datasets": (
                "EV, solid-state and synthetic snapshots are UNAVAILABLE for RUL; "
                "no sequences were manufactured."
            ),
            "na_ion": (
                "25 cycles per cell with no genuine crossing; UNAVAILABLE."
            ),
        },
        "no_fabrication": (
            "All predictions derive from observed cycle-level SOH in the source "
            "table. Censored projections are labelled and carry no validated "
            "accuracy."
        ),
    }
    _write_json(destination / "rul_provenance.json", provenance)

    summary = {
        "out_dir": str(destination),
        "selected_model": selected_name,
        "selection_metric_mae_cycles": selection["selection_metric"],
        "n_loco_records": len(loco_records),
        "n_current_records": len(reporting),
        "n_censored_records": len(censored),
        "n_unavailable_records": len(unavailable),
        "support": sequential_support(eol_pct=eol_pct)["datasets"],
        "candidates": {
            name: {
                "pre_threshold_mae_cycles": block["pre_threshold"]["mae"],
                "pre_threshold_rmse_cycles": block["pre_threshold"]["rmse"],
                "coverage": block["coverage"],
                "n_scored": block["n_scored"],
            }
            for name, block in by_model.items()
        },
    }
    _write_json(destination / "rul_summary.json", summary)
    return summary


__all__ = [
    "ARTIFACT_DIR",
    "build_rul_artifacts",
]
