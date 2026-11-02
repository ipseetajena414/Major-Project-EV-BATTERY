from typing import Any, Dict, List, Optional, TypedDict

try:
    from . import config
except (ImportError, ValueError):
    import config


class SimState(TypedDict):
    # Setup
    chemistry: str
    scenario_name: str
    scenario: Dict[str, Any]
    climate: Dict[str, float]
    bms_limits: Dict[str, float]
    is_hypothetical_simulation: bool

    # Battery State
    soc: float
    soh: Optional[float]                 # None when unmeasured/missing (strictly no fabrication)
    cell_temp_c: float
    ambient_temp_c: float
    range_km: Optional[float]            # None when SOH is unavailable

    # Member 2 ML Predictions (explicitly separated)
    member2_predictions: Optional[Dict[str, Any]]
    member2_soh: Optional[float]
    member2_soh_status: str             # "AVAILABLE" | "UNAVAILABLE"
    member2_rul: Optional[float]
    member2_rul_status: str             # "AVAILABLE" | "UNAVAILABLE"

    # Negotiation Round 1
    requested_c_rate: float
    requested_target_soc: float
    requested_min_soc: float
    driver_message: str
    bms_decision: str                   # "approve" | "modify" | "reject"
    bms_reason: str
    bms_message: str
    bms_rules_triggered: List[str]

    # Negotiation Round 2 (Counter-proposal & Final BMS Check)
    negotiation_round: int
    driver_counter_action: Optional[Dict[str, float]]
    driver_counter_message: Optional[str]
    final_decision: str
    final_reason: str
    final_bms_message: str
    applied_c_rate: float
    applied_target_soc: float
    applied_min_soc: float
    final_rules_triggered: List[str]

    # Accumulated Stress
    cycle_count: int
    thermal_stress_hours: float
    fast_charge_count: int
    hours_above_80soc: float
    decision_counts: Dict[str, int]

    # Complete Transcript History
    history: List[dict]
    done: bool


def initial_state(
    chemistry: str,
    scenario_name: str,
    climate_name: str = "hot",
    battery_state: Optional[Dict[str, Any]] = None,
    member2_predictions: Optional[Dict[str, Any]] = None,
    bms_limits: Optional[Dict[str, Any]] = None,
    is_hypothetical_simulation: bool = False,
    hypothetical_initial_soh: float = 100.0,
) -> SimState:
    """Initialize simulation state with strict handling of missing SOH/RUL.

    If battery_state lacks SOH and member2_predictions is None, SOH remains None (UNAVAILABLE).
    Only explicit hypothetical lifecycle comparisons set an assumed initial SOH.
    """
    bat = battery_state or {}
    m2 = member2_predictions or {}
    limits = config.get_bms_limits(bms_limits)

    m2_soh = m2.get("soh_pct")
    m2_rul = m2.get("rul_cycles")
    m2_soh_status = "AVAILABLE" if m2_soh is not None else "UNAVAILABLE"
    m2_rul_status = "AVAILABLE" if m2_rul is not None else "UNAVAILABLE"

    # Determine initial SOH strictly without fabrication
    if is_hypothetical_simulation:
        soh = float(hypothetical_initial_soh)
    elif "soh_pct" in bat and bat["soh_pct"] is not None:
        soh = float(bat["soh_pct"])
    elif m2_soh is not None:
        soh = float(m2_soh)
    else:
        soh = None

    soc = float(bat.get("soc_pct", 50.0))
    cell_temp = float(bat.get("cell_temp_c", 25.0))
    ambient = float(config.CLIMATES.get(climate_name, config.CLIMATES["hot"])["base"])

    return SimState(
        chemistry=chemistry,
        scenario_name=scenario_name,
        scenario=config.SCENARIOS[scenario_name],
        climate=config.CLIMATES[climate_name],
        bms_limits=limits,
        is_hypothetical_simulation=is_hypothetical_simulation,
        soc=soc,
        soh=soh,
        cell_temp_c=cell_temp,
        ambient_temp_c=ambient,
        range_km=config.RATED_RANGE_KM if soh is not None else None,
        member2_predictions=member2_predictions,
        member2_soh=m2_soh,
        member2_soh_status=m2_soh_status,
        member2_rul=m2_rul,
        member2_rul_status=m2_rul_status,
        requested_c_rate=0.0,
        requested_target_soc=0.0,
        requested_min_soc=0.0,
        driver_message="",
        bms_decision="",
        bms_reason="",
        bms_message="",
        bms_rules_triggered=[],
        negotiation_round=1,
        driver_counter_action=None,
        driver_counter_message=None,
        final_decision="",
        final_reason="",
        final_bms_message="",
        applied_c_rate=0.0,
        applied_target_soc=0.0,
        applied_min_soc=0.0,
        final_rules_triggered=[],
        cycle_count=int(bat.get("cycle_count", 0)),
        thermal_stress_hours=0.0,
        fast_charge_count=0,
        hours_above_80soc=0.0,
        decision_counts={"approve": 0, "modify": 0, "reject": 0, "compromise_approved": 0},
        history=[],
        done=False,
    )
