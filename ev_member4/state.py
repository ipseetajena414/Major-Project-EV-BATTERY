from typing import Any, Dict, List, TypedDict

import config


class SimState(TypedDict):
    # Setup
    chemistry: str
    scenario_name: str
    scenario: Dict[str, Any]
    climate: Dict[str, float]

    # Battery
    soc: float
    soh: float
    cell_temp_c: float
    ambient_temp_c: float
    range_km: float

    # Driver's request this block
    requested_c_rate: float
    requested_target_soc: float
    requested_min_soc: float
    driver_message: str

    # BMS decision this block
    bms_decision: str          # "approve" | "modify" | "reject"
    bms_reason: str
    bms_message: str
    applied_c_rate: float
    applied_target_soc: float
    applied_min_soc: float

    # Accumulated stress
    cycle_count: int
    thermal_stress_hours: float
    fast_charge_count: int
    hours_above_80soc: float
    decision_counts: Dict[str, int]

    # Output
    history: List[dict]        # one record per block, for plotting / Member 5
    done: bool


def initial_state(chemistry: str, scenario_name: str, climate_name: str) -> SimState:
    return SimState(
        chemistry=chemistry,
        scenario_name=scenario_name,
        scenario=config.SCENARIOS[scenario_name],
        climate=config.CLIMATES[climate_name],
        soc=50.0,
        soh=100.0,
        cell_temp_c=25.0,
        ambient_temp_c=25.0,
        range_km=config.RATED_RANGE_KM,
        requested_c_rate=0.0,
        requested_target_soc=0.0,
        requested_min_soc=0.0,
        driver_message="",
        bms_decision="",
        bms_reason="",
        bms_message="",
        applied_c_rate=0.0,
        applied_target_soc=0.0,
        applied_min_soc=0.0,
        cycle_count=0,
        thermal_stress_hours=0.0,
        fast_charge_count=0,
        hours_above_80soc=0.0,
        decision_counts={"approve": 0, "modify": 0, "reject": 0},
        history=[],
        done=False,
    )
