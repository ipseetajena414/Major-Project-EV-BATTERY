#!/usr/bin/env python3
"""File-based adapter from Member 2 SOH predictions to Member 3 range estimates.

Member 2 writes a prediction CSV; this module validates it, builds an SOH
trajectory and passes it to ``RangeCalculator.estimate_trajectory()``. It does
not import Member 2 code, load model artifacts or add deep-learning
dependencies to the range calculator.

Prediction CSV contract (one row per forecast):
  required: cycle_index, cell_id, chemistry, soh_pct
  optional: quality_flag, source (predicted|observed), model_name,
            model_status, split, current_observed_soh_pct,
            observed_next_soh_pct, unit

SOH must be a percentage on 1 < SOH <= 100. Values in (0, 1] are rejected as
ambiguous fractions. Only NMC and LFP are accepted, matching the chemistries
Member 2 actually trained.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

_MEMBER3_ROOT = Path(__file__).resolve().parents[1]
if str(_MEMBER3_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMBER3_ROOT))

from range_calculator import RangeCalculator, VehicleConfig  # noqa: E402

REQUIRED_COLUMNS = ("cycle_index", "cell_id", "chemistry", "soh_pct")
SUPPORTED_CHEMISTRIES = frozenset({"NMC", "LFP"})
SOH_MIN_EXCLUSIVE = 1.0
SOH_MAX_INCLUSIVE = 100.0
DEFAULT_QUALITY_FLAG = "MEMBER2_PREDICTION_UNVERIFIED"
BASELINE_QUALITY_FLAG = "PERSISTENCE_BASELINE_FROM_OBSERVED_SOH"
SOH_FIELDS = ("soh_pct", "current_observed_soh_pct")

RANGE_NOTE = (
    "Experimental estimate: Member 2 model is research-stage and the vehicle "
    "dynamics profile is partially sourced. Not a validated real-world range."
)


class PredictionArtifactError(ValueError):
    """Raised when the Member 2 prediction artifact is missing or malformed."""


class TrajectoryValidationError(ValueError):
    """Raised when a trajectory cannot be mapped safely to the range calculator."""


def _require_file(path: str | Path, label: str) -> Path:
    p = Path(path)
    if not p.is_file():
        raise PredictionArtifactError(
            f"{label} not found: {p}. Generate it with the real Member 2 inference "
            "pipeline (see member3/integration/README.md) rather than substituting "
            "synthetic values."
        )
    return p


def _parse_int(value: Any, field: str, row_number: int) -> int:
    if isinstance(value, bool):
        raise PredictionArtifactError(f"row {row_number}: {field} must be an integer")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise PredictionArtifactError(
            f"row {row_number}: {field} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(parsed) or parsed != int(parsed):
        raise PredictionArtifactError(f"row {row_number}: {field} must be an integer")
    return int(parsed)


def _parse_soh(value: Any, row_number: int, field: str = "soh_pct") -> float:
    try:
        soh = float(value)
    except (TypeError, ValueError) as exc:
        raise TrajectoryValidationError(
            f"row {row_number}: {field} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(soh):
        raise TrajectoryValidationError(f"row {row_number}: {field} is not finite")
    if 0 < soh <= SOH_MIN_EXCLUSIVE:
        raise TrajectoryValidationError(
            f"row {row_number}: {field}={soh} looks like a fraction (<= 1.0); "
            "expected percentage points on the 0-100 scale"
        )
    if soh <= 0:
        raise TrajectoryValidationError(
            f"row {row_number}: {field}={soh} must be a positive percentage on the 0-100 scale"
        )
    if soh > SOH_MAX_INCLUSIVE:
        raise TrajectoryValidationError(
            f"row {row_number}: {field}={soh} exceeds the Member 2 contract "
            f"bound of {SOH_MAX_INCLUSIVE}"
        )
    return soh


def load_prediction_artifact(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate a Member 2 prediction CSV. Returns normalised rows."""
    artifact = _require_file(path, "Member 2 prediction artifact")
    with artifact.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise PredictionArtifactError(
                f"{artifact}: missing required columns {missing}; "
                f"found {fieldnames}"
            )
        raw_rows = list(reader)
        if "unit" in fieldnames:
            units = {row["unit"].strip() for row in raw_rows if row.get("unit")}
            if units - {"%"}:
                raise PredictionArtifactError(
                    f"{artifact}: unsupported SOH unit(s) {sorted(units)}; only '%' is accepted"
                )
        rows = []
        for row_number, row in enumerate(raw_rows, start=2):
            cycle_index = _parse_int(row["cycle_index"], "cycle_index", row_number)
            cell_id = (row.get("cell_id") or "").strip()
            if not cell_id:
                raise PredictionArtifactError(f"row {row_number}: cell_id is empty")
            chemistry = (row.get("chemistry") or "").strip().upper()
            if chemistry not in SUPPORTED_CHEMISTRIES:
                raise PredictionArtifactError(
                    f"row {row_number}: chemistry {chemistry!r} is not supported by "
                    f"Member 2 predictions; expected one of {sorted(SUPPORTED_CHEMISTRIES)}"
                )
            source = (row.get("source") or "predicted").strip().lower()
            if source not in {"predicted", "observed"}:
                raise PredictionArtifactError(
                    f"row {row_number}: source must be 'predicted' or 'observed', got {source!r}"
                )
            item: dict[str, Any] = {
                "cycle_index": cycle_index,
                "cell_id": cell_id,
                "chemistry": chemistry,
                "soh_pct": _parse_soh(row["soh_pct"], row_number),
                "quality_flag": (row.get("quality_flag") or DEFAULT_QUALITY_FLAG).strip(),
                "source": source,
                "model_name": row.get("model_name") or None,
                "model_status": row.get("model_status") or None,
                "split": row.get("split") or None,
                "current_observed_soh_pct": (
                    _parse_soh(row["current_observed_soh_pct"], row_number, "current_observed_soh_pct")
                    if row.get("current_observed_soh_pct")
                    else None
                ),
                "observed_next_soh_pct": (
                    _parse_soh(row["observed_next_soh_pct"], row_number, "observed_next_soh_pct")
                    if row.get("observed_next_soh_pct")
                    else None
                ),
            }
            rows.append(item)

    if not rows:
        raise PredictionArtifactError(f"{artifact}: contains no prediction rows")

    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["cell_id"], row["cycle_index"])
        if key in seen:
            raise PredictionArtifactError(
                f"{artifact}: duplicate cycle_index {row['cycle_index']} for cell {row['cell_id']}"
            )
        seen.add(key)

    last_by_cell: dict[str, int] = {}
    for row in rows:
        previous = last_by_cell.get(row["cell_id"])
        if previous is not None and row["cycle_index"] <= previous:
            raise PredictionArtifactError(
                f"{artifact}: cycle_index must be strictly increasing per cell; "
                f"cell {row['cell_id']} goes {previous} -> {row['cycle_index']}. Sort the "
                "export chronologically before loading."
            )
        last_by_cell[row["cell_id"]] = row["cycle_index"]
    return rows


def build_soh_trajectory(
    rows: Iterable[dict[str, Any]],
    soh_field: str = "soh_pct",
    quality_flag_override: str | None = None,
) -> list[dict[str, Any]]:
    """Convert validated rows into ``estimate_trajectory`` input dicts."""
    if soh_field not in SOH_FIELDS:
        raise TrajectoryValidationError(
            f"soh_field must be one of {list(SOH_FIELDS)}, got {soh_field!r}"
        )
    trajectory = []
    for row in rows:
        value = row.get(soh_field)
        if value is None:
            raise TrajectoryValidationError(
                f"cell {row['cell_id']} cycle {row['cycle_index']}: column {soh_field!r} "
                "is required for this mode but missing"
            )
        flag = quality_flag_override
        if flag is None:
            if soh_field == "current_observed_soh_pct":
                flag = BASELINE_QUALITY_FLAG
            else:
                flag = row.get("quality_flag") or DEFAULT_QUALITY_FLAG
        trajectory.append(
            {
                "cycle_index": row["cycle_index"],
                "soh_pct": float(value),
                "cell_id": row["cell_id"],
                "quality_flag": flag,
            }
        )
    return trajectory


def estimate_range_trajectory(
    calculator: RangeCalculator,
    cycle: list[dict[str, float]],
    rows: list[dict[str, Any]],
    ambient_c: float | None = None,
    hvac_power_kw: float | None = None,
    soh_field: str = "soh_pct",
    quality_flag_override: str | None = None,
) -> list[dict[str, Any]]:
    """Run the validated trajectory through the physics engine, merging metadata."""
    trajectory = build_soh_trajectory(rows, soh_field, quality_flag_override)
    estimates = calculator.estimate_trajectory(
        cycle=cycle,
        soh_trajectory=trajectory,
        ambient_c=ambient_c,
        hvac_power_kw=hvac_power_kw,
    )
    if len(estimates) != len(rows):
        raise TrajectoryValidationError(
            f"range calculator returned {len(estimates)} rows for {len(rows)} inputs"
        )
    merged = []
    for row, traj, estimate in zip(rows, trajectory, estimates):
        merged.append(
            {
                "cycle_index": row["cycle_index"],
                "cell_id": row["cell_id"],
                "chemistry": row["chemistry"],
                "soh_pct": round(traj["soh_pct"], 6),
                "soh_source": "observed_anchor" if soh_field == "current_observed_soh_pct" else row["source"],
                "quality_flag": traj["quality_flag"],
                "estimated_range_km": round(estimate["estimated_range_km"], 6),
                "consumption_kwh_per_100km": round(estimate["consumption_kwh_per_100km"], 6),
                "current_observed_soh_pct": row.get("current_observed_soh_pct"),
                "observed_next_soh_pct": row.get("observed_next_soh_pct"),
                "split": row.get("split"),
                "model_name": row.get("model_name"),
                "model_status": row.get("model_status"),
                "vehicle_name": estimate["vehicle_name"],
                "estimate_quality_status": estimate["quality_status"],
                "ambient_c": ambient_c,
                "hvac_power_kw": estimate["hvac_power_kw"],
                "notes": RANGE_NOTE,
            }
        )
    return merged


def export_results_csv(
    results: list[dict[str, Any]],
    out_path: str | Path,
    vehicle_config_path: str | Path | None = None,
    cycle_path: str | Path | None = None,
) -> Path:
    """Write the joined trajectory + range estimate table."""
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "cycle_index", "cell_id", "chemistry", "soh_pct", "soh_source",
        "quality_flag", "estimated_range_km", "consumption_kwh_per_100km",
        "current_observed_soh_pct", "observed_next_soh_pct", "split",
        "model_name", "model_status", "vehicle_name", "estimate_quality_status",
        "ambient_c", "hvac_power_kw", "vehicle_config", "drive_cycle",
        "generated_at_utc", "notes",
    ]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            enriched = dict(row)
            enriched["vehicle_config"] = str(vehicle_config_path) if vehicle_config_path else None
            enriched["drive_cycle"] = str(cycle_path) if cycle_path else None
            enriched["generated_at_utc"] = generated
            writer.writerow(enriched)
    return destination


def run_pipeline(
    predictions_path: str | Path,
    vehicle_config_path: str | Path,
    cycle_path: str | Path,
    out_path: str | Path,
    ambient_c: float | None = None,
    hvac_power_kw: float | None = None,
    soh_field: str = "soh_pct",
    quality_flag_override: str | None = None,
) -> dict[str, Any]:
    """Validated end-to-end: prediction CSV -> SOH trajectory -> range CSV."""
    rows = load_prediction_artifact(predictions_path)
    config_path = _require_file(vehicle_config_path, "Vehicle configuration")
    cycle_file = _require_file(cycle_path, "Drive cycle")
    config = VehicleConfig.from_json(config_path)
    calculator = RangeCalculator(config)
    cycle = calculator.load_cycle(cycle_file)
    results = estimate_range_trajectory(
        calculator, cycle, rows, ambient_c, hvac_power_kw, soh_field, quality_flag_override
    )
    destination = export_results_csv(results, out_path, config_path, cycle_file)
    ranges = [r["estimated_range_km"] for r in results]
    return {
        "rows": len(results),
        "output_csv": str(destination),
        "vehicle_name": config.vehicle_name,
        "vehicle_config": str(config_path),
        "drive_cycle": str(cycle_file),
        "soh_field": soh_field,
        "soh_min": min(r["soh_pct"] for r in results),
        "soh_max": max(r["soh_pct"] for r in results),
        "range_min_km": min(ranges),
        "range_max_km": max(ranges),
        "soh_source": results[0]["soh_source"],
        "model_name": results[0].get("model_name"),
        "estimate_quality_status": sorted({r["estimate_quality_status"] for r in results}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Member 2 SOH predictions -> Member 3 range estimates")
    parser.add_argument("--predictions", required=True, help="Member 2 prediction CSV")
    parser.add_argument("--vehicle-config", required=True, help="vehicle profile JSON")
    parser.add_argument("--cycle", required=True, help="drive-cycle CSV")
    parser.add_argument("--out", required=True, help="output CSV path")
    parser.add_argument("--ambient-c", type=float, default=None)
    parser.add_argument("--hvac-kw", type=float, default=None)
    parser.add_argument("--soh-field", choices=SOH_FIELDS, default="soh_pct")
    parser.add_argument("--quality-flag", default=None, help="override the per-row quality flag")
    args = parser.parse_args(argv)

    try:
        summary = run_pipeline(
            args.predictions, args.vehicle_config, args.cycle, args.out,
            args.ambient_c, args.hvac_kw, args.soh_field, args.quality_flag,
        )
    except (PredictionArtifactError, TrajectoryValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
