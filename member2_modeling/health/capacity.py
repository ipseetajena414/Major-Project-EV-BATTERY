"""Capacity semantics for each processed dataset.

Capacity is never just a number: it is a measured value, a model prediction,
or an algebraic derivation from SOH and a rated capacity. The unit and scope
matter because the EV dataset stores pack energy in kWh while the cell
datasets store capacity in Ah.
"""

from __future__ import annotations

import math
from typing import Any


class HealthModelError(ValueError):
    """Raised for invalid health-model inputs or capacity derivations."""


CAPACITY_SPECS: dict[str, dict[str, Any]] = {
    "EV_FAILURE_PREDICTION": {
        "measured_column": "remaining_capacity",
        "measured_unit": "kWh",
        "scope": "vehicle_pack_energy",
        "is_pack_energy": True,
        "predicted_capacity_available": True,
        "algebraic": {
            "basis": "soh_pct_times_rated_capacity",
            "source_column": "battery_capacity_kwh",
            "unit": "kWh",
            "scope": "vehicle_pack_energy",
            "is_pack_energy": True,
        },
        "notes": [
            "remaining_capacity is derived pack energy in kWh (algebraically from battery_capacity_kwh and battery_health_percent), not cell Ah.",
            "A capacity model trained on remaining_capacity predicts pack energy.",
            "SOH x battery_capacity_kwh is an algebraic estimate, not a measurement.",
        ],
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "measured_column": None,
        "measured_unit": None,
        "scope": "cell_nominal_capacity",
        "is_pack_energy": False,
        "predicted_capacity_available": False,
        "algebraic": {
            "basis": "soh_pct_times_rated_capacity",
            "source_column": "nominal_capacity_ah",
            "unit": "Ah",
            "scope": "cell_nominal_capacity",
            "is_pack_energy": False,
        },
        "notes": [
            "No measured capacity column exists; only nominal_capacity_ah and SOH.",
            "Capacity can only be derived algebraically from nominal capacity and SOH.",
        ],
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "measured_column": "discharge_capacity_ah",
        "measured_unit": "Ah",
        "scope": "cell_discharge_capacity",
        "is_pack_energy": False,
        "predicted_capacity_available": True,
        "algebraic": None,
        "notes": [
            "discharge_capacity_ah is a synthetic measured label, not experimental truth.",
            "No rated nominal capacity column exists, so no algebraic derivation is exposed.",
        ],
    },
}


def capacity_semantics(dataset_id: str) -> dict[str, Any]:
    if dataset_id not in CAPACITY_SPECS:
        raise HealthModelError(f"Unknown dataset_id for capacity semantics: {dataset_id!r}")
    spec = CAPACITY_SPECS[dataset_id]
    return {
        "dataset_id": dataset_id,
        "measured_column": spec["measured_column"],
        "measured_unit": spec["measured_unit"],
        "scope": spec["scope"],
        "is_pack_energy": spec["is_pack_energy"],
        "predicted_capacity_available": spec["predicted_capacity_available"],
        "algebraic": None if spec["algebraic"] is None else dict(spec["algebraic"]),
        "notes": list(spec["notes"]),
    }


def derive_capacity_from_soh(
    soh_pct: float,
    rated_capacity: float,
    *,
    unit: str,
    scope: str,
    is_pack_energy: bool = False,
) -> dict[str, Any]:
    if isinstance(soh_pct, bool) or not isinstance(soh_pct, (int, float)):
        raise HealthModelError("soh_pct must be numeric")
    if isinstance(rated_capacity, bool) or not isinstance(rated_capacity, (int, float)):
        raise HealthModelError("rated_capacity must be numeric")
    soh = float(soh_pct)
    rated = float(rated_capacity)
    if not math.isfinite(soh) or not 0.0 <= soh <= 100.0:
        raise HealthModelError("soh_pct must be finite and within 0-100")
    if not math.isfinite(rated) or rated <= 0.0:
        raise HealthModelError("rated_capacity must be finite and positive")
    if not unit or not scope:
        raise HealthModelError("unit and scope are required")
    return {
        "capacity": rated * soh / 100.0,
        "unit": unit,
        "scope": scope,
        "kind": "algebraic",
        "basis": "soh_pct_times_rated_capacity",
        "rated_capacity": rated,
        "soh_pct": soh,
        "is_measured": False,
        "is_predicted": False,
        "is_pack_energy": bool(is_pack_energy),
        "validation_status": "not_independently_validated",
    }


__all__ = [
    "HealthModelError",
    "capacity_semantics",
    "derive_capacity_from_soh",
]
