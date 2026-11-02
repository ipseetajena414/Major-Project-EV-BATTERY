"""Formula-based EV range estimate.

The EV dataset has rated pack energy and SOH but no measured range or measured
energy-per-kilometre target. Range is therefore a *calculated estimate* under an
explicit consumption assumption. It must never be reported as validated range
accuracy.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

DEFAULT_CONSUMPTION_KWH_PER_KM = 0.18
CONSUMPTION_SCENARIOS = {
    "optimistic": 0.15,
    "central": DEFAULT_CONSUMPTION_KWH_PER_KM,
    "conservative": 0.22,
}


class RangeEstimateError(ValueError):
    """Raised for invalid range-estimate inputs."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RangeEstimateError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RangeEstimateError(f"{name} must be finite")
    return number


def estimate_range(
    capacity_kwh: float,
    soh_pct: float,
    *,
    discharge_efficiency_pct: float = 100.0,
    consumption_kwh_per_km: float = DEFAULT_CONSUMPTION_KWH_PER_KM,
) -> dict[str, Any]:
    rated = _finite(capacity_kwh, "capacity_kwh")
    soh = _finite(soh_pct, "soh_pct")
    efficiency = _finite(discharge_efficiency_pct, "discharge_efficiency_pct")
    consumption = _finite(consumption_kwh_per_km, "consumption_kwh_per_km")
    if rated <= 0:
        raise RangeEstimateError("capacity_kwh must be positive")
    if not 0 <= soh <= 100:
        raise RangeEstimateError("soh_pct must be within 0-100")
    if not 0 < efficiency <= 100:
        raise RangeEstimateError("discharge_efficiency_pct must be within (0, 100]")
    if consumption <= 0:
        raise RangeEstimateError("consumption_kwh_per_km must be positive")

    usable_kwh = rated * (soh / 100.0) * (efficiency / 100.0)
    scenarios = {
        name: usable_kwh / rate for name, rate in CONSUMPTION_SCENARIOS.items()
    }
    return {
        "range_km": usable_kwh / consumption,
        "range_km_scenarios": scenarios,
        "usable_energy_kwh": usable_kwh,
        "rated_capacity_kwh": rated,
        "soh_pct": soh,
        "discharge_efficiency_pct": efficiency,
        "consumption_kwh_per_km": consumption,
        "unit": "km",
        "kind": "calculated_estimate",
        "is_measured": False,
        "validated_accuracy": False,
        "validation_status": "not_validated_no_range_target",
        "assumptions": [
            "Consumption is an assumed constant, not a measured value from the dataset.",
            f"Central consumption assumption: {DEFAULT_CONSUMPTION_KWH_PER_KM} kWh/km.",
            "Usable energy = rated pack energy x SOH x discharge efficiency.",
            "No terrain, speed, HVAC, payload or temperature correction is applied.",
        ],
        "limitations": [
            "The dataset contains no measured driving-range or energy-per-km target.",
            "Scenario spread reflects an assumption range, not statistical uncertainty.",
        ],
    }


def estimate_range_from_record(
    record: dict[str, Any],
    *,
    soh_pct: float | None = None,
) -> dict[str, Any]:
    """Estimate range from an EV snapshot record.

    ``soh_pct`` may be supplied (for example from the ev_soh model). Otherwise
    the record's own SOH label is used when present.
    """
    if "battery_capacity_kwh" not in record:
        raise RangeEstimateError("battery_capacity_kwh is required")
    capacity = record["battery_capacity_kwh"]
    if soh_pct is None:
        for candidate in ("state_of_health", "battery_health_percent"):
            value = record.get(candidate)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                soh_pct = value
                break
    if soh_pct is None:
        raise RangeEstimateError(
            "soh_pct is required when the record has no SOH label"
        )
    efficiency = record.get("discharge_efficiency")
    if efficiency is None or (isinstance(efficiency, float) and math.isnan(efficiency)):
        efficiency = 100.0
    result = estimate_range(
        capacity,
        soh_pct,
        discharge_efficiency_pct=efficiency,
    )
    result["estimate_source"] = (
        "model_predicted_soh" if "state_of_health" not in record else "record_soh_label"
    )
    return result


def estimate_range_batch(
    records: Iterable[dict[str, Any]],
    soh_values: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    rows = list(records)
    soh_list = None if soh_values is None else list(soh_values)
    if soh_list is not None and len(soh_list) != len(rows):
        raise RangeEstimateError("soh_values must have the same length as records")
    return [
        estimate_range_from_record(
            record, soh_pct=None if soh_list is None else soh_list[index]
        )
        for index, record in enumerate(rows)
    ]


__all__ = [
    "CONSUMPTION_SCENARIOS",
    "DEFAULT_CONSUMPTION_KWH_PER_KM",
    "RangeEstimateError",
    "estimate_range",
    "estimate_range_batch",
    "estimate_range_from_record",
]
