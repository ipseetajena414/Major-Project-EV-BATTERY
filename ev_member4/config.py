"""Central config. Change numbers here, not inside the agents.

STATUS OF EACH MODEL (also written into results.json as "model_provenance"):
  * Range      -> Member 3 RangeCalculator with sourced reference vehicle configs (e.g. Tata Curvv.ev,
                  Nexon.ev, Mahindra XEV 9e, Tesla Model Y) and EPA standard drive cycles (UDDS/HWFET).
  * Degradation-> ASSUMED coefficients below (not learned from data). Member 2's model cannot supply this:
                  it needs discharge-curve statistics, and temperature / C-rate / SOC are BLOCKED in its feature schema.
"""
import json
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------- LLM ----------------
USE_LLM = bool(os.getenv("GOOGLE_API_KEY"))   # on automatically when a key is in .env; otherwise templated text
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")    # Google AI Studio model name
LLM_EVERY_N_BLOCKS = 20              # only call the LLM every N blocks to save free-tier quota

# ---------------- Range engine (Member 3) ----------------
RANGE_BACKEND = os.getenv("EV_RANGE_BACKEND", "member3")  # "member3" = RangeCalculator, "placeholder" = fallback
VENDOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
MEMBER3_DIR = os.path.join(VENDOR_DIR, "member3")
VEHICLE_CONFIGS_DIR = os.path.join(MEMBER3_DIR, "vehicle_configs")
DRIVE_CYCLES_DIR = os.path.join(MEMBER3_DIR, "drive_cycles")


def resolve_vehicle_config(name_or_path: str = None) -> str:
    """Resolve vehicle config path from a filename, relative path, or absolute path.

    Defaults to EV_VEHICLE_CONFIG env var, or tata_curvv_ev_55kwh_partial_reference.json.
    """
    target = name_or_path or os.getenv("EV_VEHICLE_CONFIG", "tata_curvv_ev_55kwh_partial_reference.json")
    if os.path.isabs(target) and os.path.exists(target):
        return target
    in_configs = os.path.join(VEHICLE_CONFIGS_DIR, target)
    if os.path.exists(in_configs):
        return in_configs
    in_m3 = os.path.join(MEMBER3_DIR, target)
    if os.path.exists(in_m3):
        return in_m3
    fallback = os.path.join(MEMBER3_DIR, "vehicle_config.json")
    if os.path.exists(fallback):
        return fallback
    return target


def resolve_drive_cycle(name_or_path: str = None) -> str:
    """Resolve drive cycle path from a filename, relative path, or absolute path.

    Defaults to EV_DRIVE_CYCLE env var, or udds_city.csv (EPA UDDS urban cycle).
    """
    target = name_or_path or os.getenv("EV_DRIVE_CYCLE", "udds_city.csv")
    if os.path.isabs(target) and os.path.exists(target):
        return target
    in_cycles = os.path.join(DRIVE_CYCLES_DIR, target)
    if os.path.exists(in_cycles):
        return in_cycles
    fallback = os.path.join(DRIVE_CYCLES_DIR, "synthetic_example_cycle.csv")
    if os.path.exists(fallback):
        return fallback
    return target


VEHICLE_CONFIG = resolve_vehicle_config()
DRIVE_CYCLE = resolve_drive_cycle()


def set_active_vehicle(name_or_path: str) -> str:
    """Set the active vehicle configuration and update rated range."""
    global VEHICLE_CONFIG, RATED_RANGE_KM
    VEHICLE_CONFIG = resolve_vehicle_config(name_or_path)
    RATED_RANGE_KM = _get_rated_range(VEHICLE_CONFIG)
    return VEHICLE_CONFIG


def set_active_drive_cycle(name_or_path: str) -> str:
    """Set the active drive cycle."""
    global DRIVE_CYCLE
    DRIVE_CYCLE = resolve_drive_cycle(name_or_path)
    return DRIVE_CYCLE


def _get_rated_range(cfg_path: str) -> float:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            ref = data.get("rated_range_km_reference")
            if ref and float(ref) > 0:
                return float(ref)
    except Exception:
        pass
    return 502.0


def list_available_vehicles() -> list:
    """List all available vehicle profiles in the vehicle configuration library."""
    vehicles = []
    if os.path.isdir(VEHICLE_CONFIGS_DIR):
        for f in sorted(os.listdir(VEHICLE_CONFIGS_DIR)):
            if f.endswith(".json"):
                fpath = os.path.join(VEHICLE_CONFIGS_DIR, f)
                try:
                    with open(fpath, "r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    vehicles.append({
                        "file": f,
                        "path": fpath,
                        "vehicle_name": data.get("vehicle_name", f),
                        "nominal_pack_energy_kwh": data.get("nominal_pack_energy_kwh"),
                        "rated_range_km_reference": data.get("rated_range_km_reference"),
                        "status": data.get("provenance", {}).get("status", "UNKNOWN"),
                    })
                except Exception:
                    continue
    return vehicles


def list_available_drive_cycles() -> list:
    """List all available drive cycle traces."""
    cycles = []
    if os.path.isdir(DRIVE_CYCLES_DIR):
        for f in sorted(os.listdir(DRIVE_CYCLES_DIR)):
            if f.endswith(".csv"):
                cycles.append({
                    "file": f,
                    "path": os.path.join(DRIVE_CYCLES_DIR, f),
                })
    return cycles


# ---------------- BMS rules (from the blueprint) ----------------
BMS_LIMITS = {
    "max_temp_c": 35.0,                      # thermal cutoff
    "max_c_rate": 1.5,                       # C-rate ceiling
    "max_soc": 80.0,                         # upper SOC bound
    "min_soc": 20.0,                         # lower SOC bound
    "refuse_fast_charge_above_ambient_c": 42.0,
    "fallback_c_rate": 0.3,                  # slow AC charging when DCFC is refused
}

# ---------------- Lifecycle ----------------
END_OF_LIFE_SOH = 70.0        # blueprint says 70-80%; agree on one number with your team
RATED_RANGE_KM = _get_rated_range(VEHICLE_CONFIG)
CYCLES_PER_YEAR = 300
CYCLES_PER_BLOCK = 30         # one graph loop = 30 cycles of identical behaviour
MAX_YEARS = 20

# ---------------- Simple thermal model ----------------
CELL_HEATING_C_PER_C_RATE = 4.0   # cell temp rise (deg C) per 1C of charge rate
PRECONDITION_STRENGTH = 0.6       # fraction of the gap to 25 deg C closed by pre-conditioning
HIGH_SOC_DWELL_HOURS = 6.0        # hours parked at >80% SOC per cycle when charging above 80%

# ---------------- Driver persona (aggressive) ----------------
DRIVER = {
    "requested_c_rates": [2.0, 2.5, 3.0],   # cycles through these, deterministic
    "target_soc": 100.0,
    "min_soc": 5.0,
}

# ---------------- Chemistry placeholder coefficients ----------------
# base_fade: % SOH lost per cycle at ~25C, ~1C, 20-80% SOC
CHEMISTRIES = {
    "NMC":        dict(base_fade=0.012, k_temp=0.35, k_cold=0.30, k_crate=0.50,
                       k_soc_high=0.50, k_soc_low=0.20, cold_range_loss=0.012),
    "LFP":        dict(base_fade=0.008, k_temp=0.25, k_cold=0.50, k_crate=0.25,
                       k_soc_high=0.15, k_soc_low=0.10, cold_range_loss=0.014),
    "Sodium-ion": dict(base_fade=0.012, k_temp=0.20, k_cold=0.10, k_crate=0.30,
                       k_soc_high=0.20, k_soc_low=0.10, cold_range_loss=0.007),
    "Solid-State": dict(base_fade=0.006, k_temp=0.15, k_cold=0.20, k_crate=0.15,
                        k_soc_high=0.20, k_soc_low=0.10, cold_range_loss=0.009),
}

# ---------------- Scenarios ----------------
SCENARIOS = {
    "unrestricted":     dict(enforce_temp=False, enforce_c_rate=False, enforce_soc=False, precondition=False),
    "bms_supervised":   dict(enforce_temp=True,  enforce_c_rate=True,  enforce_soc=False, precondition=False),
    "soc_20_80":        dict(enforce_temp=False, enforce_c_rate=False, enforce_soc=True,  precondition=False),
    "precondition_only": dict(enforce_temp=False, enforce_c_rate=False, enforce_soc=False, precondition=True),
    "full_protection":  dict(enforce_temp=True,  enforce_c_rate=True,  enforce_soc=True,  precondition=True),
}

# ---------------- Climates: ambient = base + amp * sin(season) ----------------
CLIMATES = {
    "hot":       dict(base=33.0, amp=8.0),
    "temperate": dict(base=20.0, amp=10.0),
    "cold":      dict(base=2.0,  amp=8.0),
}
