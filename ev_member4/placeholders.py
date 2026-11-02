"""Model layer for the simulation: the two functions graph.py calls every block.

predict_soh        -> HYPOTHETICAL degradation simulation (assumed coefficients in config.CHEMISTRIES).
estimate_range_km  -> Member 3 RangeCalculator (using sourced vehicle config + standard drive cycle).

ARCHITECTURAL PRINCIPLE:
Member 4 does NOT independently predict SOH, RUL, capacity, or range.
When actual Member 2 predictions are provided, they are ingested with intact provenance.
When missing, SOH and RUL remain explicitly None / UNAVAILABLE.
Hypothetical multi-year lifecycle curves are labeled as hypothetical comparative simulations.
"""
import os
import sys
import warnings
from typing import Optional

try:
    from . import config
except (ImportError, ValueError):
    import config

FEATURE_ORDER = [
    "mean_temp_c", "max_c_rate", "target_soc", "min_soc",
    "hours_above_80soc_per_cycle", "fast_charge_fraction", "n_cycles",
]


# ------------------------------------------------------------ degradation
def predict_soh(chemistry: str, current_soh: Optional[float], features: dict) -> Optional[float]:
    """Hypothetical degradation calculation for scenario comparison only.

    DISCLAIMER: This is NOT a trained machine learning model or an empirical battery prediction.
    It computes comparative degradation decay from assumed sensitivity factors in config.CHEMISTRIES.
    If current_soh is None (unmeasured/missing), returns None without fabricating values.
    """
    if current_soh is None:
        return None

    p = config.CHEMISTRIES[chemistry]
    t = features["mean_temp_c"]

    temp_factor = 1 + p["k_temp"] * max(0.0, t - 25) / 10 + p["k_cold"] * max(0.0, 10 - t) / 10
    crate_factor = 1 + p["k_crate"] * max(0.0, features["max_c_rate"] - 1.0)
    soc_factor = (1 + p["k_soc_high"] * min(features["hours_above_80soc_per_cycle"] / 6.0, 1.5)
                  + p["k_soc_low"] * max(0.0, 20 - features["min_soc"]) / 20)
    knee = 1 + 0.8 * max(0.0, 85 - current_soh) / 15       # fade speeds up late in life

    fade = features["n_cycles"] * p["base_fade"] * temp_factor * crate_factor * soc_factor * knee
    return max(0.0, current_soh - fade)


# ------------------------------------------------------------ range
_RC = {
    "calc": None,
    "cycle": None,
    "limitations": None,
    "failed": False,
    "vehicle_path": None,
    "cycle_path": None,
}


def _load_member3(vehicle_config_path: str = None, drive_cycle_path: str = None):
    """Load Member 3's RangeCalculator. Re-loads if paths change.

    The repo-level `member3` package wins over the vendored copy if installed.
    """
    v_path = vehicle_config_path or config.VEHICLE_CONFIG
    c_path = drive_cycle_path or config.DRIVE_CYCLE

    if (_RC["calc"] is not None and not _RC["failed"]
            and _RC["vehicle_path"] == v_path and _RC["cycle_path"] == c_path):
        return _RC["calc"]

    try:
        if config.VENDOR_DIR not in sys.path:
            sys.path.append(config.VENDOR_DIR)
        from member3.range_calculator import RangeCalculator, VehicleConfig
        calc = RangeCalculator(VehicleConfig.from_json(v_path))
        _RC["calc"] = calc
        _RC["cycle"] = RangeCalculator.load_cycle(c_path)
        _RC["limitations"] = calc.estimate(_RC["cycle"], soh_pct=100.0, ambient_c=25.0)["limitations"]
        _RC["vehicle_path"] = v_path
        _RC["cycle_path"] = c_path
        _RC["failed"] = False
    except Exception as exc:
        _RC["failed"] = True
        warnings.warn(f"Member 3 range engine unavailable ({type(exc).__name__}: {exc}); using the simple placeholder.")
    return _RC["calc"]


def _placeholder_range(soh: float, ambient_temp_c: float, chemistry: str) -> float:
    p = config.CHEMISTRIES[chemistry]
    hot = 0.006 * max(0.0, ambient_temp_c - 30)
    cold = p["cold_range_loss"] * max(0.0, 15 - ambient_temp_c)
    return config.RATED_RANGE_KM * (soh / 100.0) * max(0.4, 1 - hot - cold) * 0.95


def estimate_range_km(soh: Optional[float], ambient_temp_c: float, chemistry: str) -> Optional[float]:
    """Real-world range in km for this SOH and ambient temperature using Member 3's physics engine.

    If SOH is None (unmeasured/missing), range cannot be calculated and returns None.
    """
    if soh is None:
        return None

    if config.RANGE_BACKEND == "member3":
        calc = _load_member3()
        if calc is not None:
            soh_in = min(max(soh, 1.01), 120.0)
            return float(calc.estimate(_RC["cycle"], soh_pct=float(soh_in), ambient_c=float(ambient_temp_c))["estimated_range_km"])
    return _placeholder_range(soh, ambient_temp_c, chemistry)


# ------------------------------------------------------------ provenance
def provenance() -> dict:
    """Written into results.json so no reader mistakes hypothetical numbers for ML predictions."""
    using_m3 = config.RANGE_BACKEND == "member3" and _load_member3() is not None
    calc = _RC["calc"] if using_m3 else None

    if calc is not None:
        cfg = getattr(calc, "c", getattr(calc, "config", None))
        v_prov = getattr(cfg, "provenance", {}) or {}
        v_status = v_prov.get("status", "PARTIALLY_SOURCED_UNCALIBRATED")
        range_meta = {
            "source": "member3.range_calculator.RangeCalculator",
            "vehicle_name": cfg.vehicle_name if cfg else None,
            "vehicle_config_file": os.path.basename(config.VEHICLE_CONFIG),
            "nominal_pack_energy_kwh": cfg.nominal_pack_energy_kwh if cfg else None,
            "usable_energy_fraction": cfg.usable_energy_fraction if cfg else None,
            "rated_range_km_reference": cfg.rated_range_km_reference if cfg else None,
            "drive_cycle": os.path.basename(config.DRIVE_CYCLE),
            "engine_limitations": _RC["limitations"],
            "status": v_status,
            "vehicle_provenance": v_prov,
        }
    else:
        range_meta = {
            "source": "simple placeholder formula",
            "vehicle_name": None,
            "vehicle_config_file": None,
            "drive_cycle": None,
            "engine_limitations": None,
            "status": "illustrative: fallback formula",
        }

    return {
        "degradation": {
            "is_machine_learning_prediction": False,
            "type": "HYPOTHETICAL_SIMULATION",
            "source": "assumed comparative coefficients (config.CHEMISTRIES)",
            "status": "illustrative_scenario_model",
            "disclaimer": config.HYPOTHETICAL_SIMULATION_DISCLAIMER,
            "why_not_member2": "Member 2's schema marks temperature, C-rate and SOC features as BLOCKED in the master table; "
                               "its models predict next-record SOH from 29 discharge-curve statistics rather than charging stress.",
        },
        "range": range_meta,
        "end_of_life_soh_pct": config.END_OF_LIFE_SOH,
    }
