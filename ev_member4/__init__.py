"""ev_member4: Multi-Agent BMS Simulation & Preservation Advisory Sandbox.

Member 4 Package Interface for Member 5 (Streamlit Dashboard) & Standalone CLI.
"""
import os
import sys

# Ensure package directory and repo root are accessible on sys.path
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from .run_simulation import (
    run_simulation,
    run_all,
    simulate_bms_negotiation,
    set_active_vehicle,
    set_active_drive_cycle,
    list_available_vehicles,
    list_available_drive_cycles,
)
from .config import (
    BMS_LIMITS,
    RULES_CATALOG,
    CHEMISTRIES,
    SCENARIOS,
    CLIMATES,
    resolve_vehicle_config,
    resolve_drive_cycle,
)
from .bms import evaluate, predict_cell_temp
from .agents import driver_propose, bms_explain, driver_respond_to_bms, bms_final_verify

__all__ = [
    "run_simulation",
    "run_all",
    "simulate_bms_negotiation",
    "set_active_vehicle",
    "set_active_drive_cycle",
    "list_available_vehicles",
    "list_available_drive_cycles",
    "BMS_LIMITS",
    "RULES_CATALOG",
    "CHEMISTRIES",
    "SCENARIOS",
    "CLIMATES",
    "resolve_vehicle_config",
    "resolve_drive_cycle",
    "evaluate",
    "predict_cell_temp",
    "driver_propose",
    "bms_explain",
    "driver_respond_to_bms",
    "bms_final_verify",
]
