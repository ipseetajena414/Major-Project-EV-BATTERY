"""Adapter over Member 3's real ``RangeCalculator`` (``member3/range_calculator.py``)."""
from __future__ import annotations
import json
from typing import Any

from .paths import VEHICLE_DIR, CYCLE_DIR, ensure_repo_on_path

ensure_repo_on_path()


class RangeBlocked(ValueError):
    """Raised when range cannot be computed because the SOH input is not allowed."""


def list_vehicles() -> list[str]:
    return sorted(p.name for p in VEHICLE_DIR.glob("*.json"))


def list_cycles() -> list[str]:
    return sorted(p.name for p in CYCLE_DIR.glob("*.csv"))


CYCLE_NOTES = {
    "udds_city.csv": "UDDS urban schedule (converted from member3/drive_cycles/source/UDDS.txt)",
    "hwfet_highway.csv": "HWFET highway schedule (converted from member3/drive_cycles/source/HWFET.txt)",
    "example_cycle.csv": "Tiny smoke-test cycle, not a standard cycle",
}


def vehicle_info(name: str) -> dict[str, Any]:
    cfg = json.loads((VEHICLE_DIR / name).read_text())
    prov = cfg.get("provenance", {})
    return {"name": cfg.get("vehicle_name"), "pack_kwh": cfg.get("nominal_pack_energy_kwh"),
            "rated_range_km_reference": cfg.get("rated_range_km_reference"),
            "provenance_status": prov.get("status", "UNKNOWN"),
            "illustrative": "ILLUSTRATIVE" in str(prov.get("status", "")).upper() or name.startswith("illustrative"),
            "note": prov.get("note", ""), "assumptions": prov.get("assumptions", [])}


def _calc(vehicle: str):
    from member3.range_calculator import RangeCalculator, VehicleConfig
    return RangeCalculator(VehicleConfig.from_json(VEHICLE_DIR / vehicle))


def estimate(vehicle: str, cycle: str, soh: dict[str, Any], ambient_c: float | None,
             hvac_power_kw: float | None = None, quality_flag: str | None = None) -> dict[str, Any]:
    """Run ``RangeCalculator.estimate``. ``soh`` must be the dict returned by ``soh_for_range``.

    Speed units are already m/s in the cycle CSVs and SOH is passed as a percentage (the
    calculator rejects fractions <= 1.0), so no unit conversion happens here.
    """
    if not soh or not soh.get("allowed"):
        raise RangeBlocked((soh or {}).get("reason", "SOH not allowed"))
    from member3.range_calculator import RangeCalculator
    calc = _calc(vehicle)
    pts = RangeCalculator.load_cycle(CYCLE_DIR / cycle)
    out = calc.estimate(pts, soh_pct=soh["soh_pct"], ambient_c=ambient_c, hvac_power_kw=hvac_power_kw,
                        quality_flag=quality_flag)
    info = vehicle_info(vehicle)
    out["_applied_soh_source"] = soh.get("label")
    out["_vehicle_provenance_status"] = info["provenance_status"]
    out["_cycle"] = cycle
    out["_cycle_note"] = CYCLE_NOTES.get(cycle, "Custom/unknown cycle")
    return out
