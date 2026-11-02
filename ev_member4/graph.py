"""LangGraph loop with genuine multi-turn negotiation:

driver_proposes (Round 1) -> bms_evaluates (Round 1)
       -> driver_responds (Round 2) -> bms_final_verifies (Final Safety Check)
       -> apply_action -> update_battery_state -> check_end_of_life
                                                       |
               +---------------------------------------+ (not done)
"""
import math
from langgraph.graph import END, START, StateGraph

try:
    from . import agents, bms, config
    from .placeholders import estimate_range_km, predict_soh
    from .state import SimState
except (ImportError, ValueError):
    import agents, bms, config
    from placeholders import estimate_range_km, predict_soh
    from state import SimState


def _block_index(state: SimState) -> int:
    return state["cycle_count"] // config.CYCLES_PER_BLOCK


def _ambient(state: SimState) -> float:
    year_frac = state["cycle_count"] / config.CYCLES_PER_YEAR
    c = state["climate"]
    return c["base"] + c["amp"] * math.sin(2 * math.pi * year_frac)


def driver_proposes(state: SimState) -> dict:
    """Round 1: Driver persona proposes initial aggressive charging parameters."""
    ambient = _ambient(state)
    p = agents.driver_propose(_block_index(state), ambient, state["soh"])
    return {
        "ambient_temp_c": ambient,
        "requested_c_rate": p["c_rate"],
        "requested_target_soc": p["target_soc"],
        "requested_min_soc": p["min_soc"],
        "driver_message": p["message"],
        "negotiation_round": 1,
    }


def bms_evaluates(state: SimState) -> dict:
    """Round 1: BMS supervisor evaluates proposal against deterministic safety rules."""
    limits = state.get("bms_limits") or config.BMS_LIMITS
    d = bms.evaluate(
        state["requested_c_rate"],
        state["requested_target_soc"],
        state["requested_min_soc"],
        state["ambient_temp_c"],
        state["scenario"],
        limits,
    )
    bms_msg = agents.bms_explain(_block_index(state), d, state["requested_c_rate"], round_num=1)
    return {
        "bms_decision": d["decision"],
        "bms_reason": d["reason"],
        "bms_message": bms_msg,
        "bms_rules_triggered": d["rules_triggered"],
        "applied_c_rate": d["c_rate"],
        "applied_target_soc": d["target_soc"],
        "applied_min_soc": d["min_soc"],
    }


def driver_responds(state: SimState) -> dict:
    """Round 2: Driver reviews BMS envelope and either accepts or proposes a compromise."""
    if state["bms_decision"] == "approve":
        # Direct approval: no counter-proposal needed
        return {
            "negotiation_round": 1,
            "driver_counter_action": None,
            "driver_counter_message": "Proposal approved; no counter-proposal needed.",
        }

    # Evaluate counter-proposal or acceptance
    bms_eval = {
        "decision": state["bms_decision"],
        "reason": state["bms_reason"],
        "c_rate": state["applied_c_rate"],
        "target_soc": state["applied_target_soc"],
        "min_soc": state["applied_min_soc"],
        "rules_triggered": state["bms_rules_triggered"],
    }
    requested = {
        "c_rate": state["requested_c_rate"],
        "target_soc": state["requested_target_soc"],
        "min_soc": state["requested_min_soc"],
    }
    resp = agents.driver_respond_to_bms(2, bms_eval, requested, state["ambient_temp_c"])
    return {
        "negotiation_round": 2,
        "driver_counter_action": {
            "action": resp["action"],
            "c_rate": resp["c_rate"],
            "target_soc": resp["target_soc"],
            "min_soc": resp["min_soc"],
        },
        "driver_counter_message": resp["message"],
    }


def bms_final_verifies(state: SimState) -> dict:
    """Final deterministic verification of driver's response / counter-proposal.

    CRITICAL ARCHITECTURAL SAFETY GUARANTEE:
    No action executes without passing this final deterministic safety check.
    If the driver counter-proposes an unsafe compromise, BMS strictly overrides it.
    """
    limits = state.get("bms_limits") or config.BMS_LIMITS
    driver_counter = state.get("driver_counter_action")

    if not driver_counter or state["negotiation_round"] == 1:
        # Approval directly in round 1
        final_decision = state["bms_decision"]
        final_reason = state["bms_reason"]
        final_rules = state["bms_rules_triggered"]
        final_c = state["applied_c_rate"]
        final_target = state["applied_target_soc"]
        final_floor = state["applied_min_soc"]
        final_msg = state["bms_message"]
    else:
        # Re-evaluate driver's round 2 counter-proposal deterministically
        verif = agents.bms_final_verify(
            driver_counter,
            state["ambient_temp_c"],
            state["scenario"],
            limits,
        )
        final_decision = verif["final_decision"]
        final_reason = verif["final_reason"]
        final_rules = verif["rules_triggered"]
        final_c = verif["c_rate"]
        final_target = verif["target_soc"]
        final_floor = verif["min_soc"]
        final_msg = verif["final_bms_message"]

    counts = dict(state["decision_counts"])
    counts[final_decision] = counts.get(final_decision, 0) + 1

    return {
        "final_decision": final_decision,
        "final_reason": final_reason,
        "final_bms_message": final_msg,
        "applied_c_rate": final_c,
        "applied_target_soc": final_target,
        "applied_min_soc": final_floor,
        "final_rules_triggered": final_rules,
        "decision_counts": counts,
    }


def apply_action(state: SimState) -> dict:
    cell_temp = bms.predict_cell_temp(
        state["ambient_temp_c"],
        state["applied_c_rate"],
        state["scenario"].get("precondition", False),
    )
    return {"cell_temp_c": cell_temp, "soc": state["applied_target_soc"]}


def update_battery_state(state: SimState) -> dict:
    n = config.CYCLES_PER_BLOCK
    c = state["applied_c_rate"]
    target = state["applied_target_soc"]
    floor = state["applied_min_soc"]
    temp = state["cell_temp_c"]

    charge_hours = max(0.0, (target - floor) / 100.0 / max(0.1, c)) if c > 0 else 0.0
    dwell = config.HIGH_SOC_DWELL_HOURS if target > 80 else 0.0
    above_80 = dwell + (max(0.0, target - 80) / 100.0 / max(0.1, c)) if c > 0 else 0.0

    features = {
        "mean_temp_c": temp,
        "max_c_rate": c,
        "target_soc": target,
        "min_soc": floor,
        "hours_above_80soc_per_cycle": above_80,
        "fast_charge_fraction": 1.0 if c > 1.5 else 0.0,
        "n_cycles": n,
    }

    # SOH degradation calculation
    if state["is_hypothetical_simulation"] or state["soh"] is not None:
        new_soh = predict_soh(state["chemistry"], state["soh"], features)
        range_km = estimate_range_km(new_soh, state["ambient_temp_c"], state["chemistry"])
    else:
        new_soh = None
        range_km = None

    cycle_count = state["cycle_count"] + n

    # Complete Transcript Record with all required fields
    record = {
        "cycle": cycle_count,
        "year": round(cycle_count / config.CYCLES_PER_YEAR, 3),
        "negotiation_round": state["negotiation_round"],
        "driver_message": state["driver_message"],
        "bms_message": state["final_bms_message"] or state["bms_message"],
        "requested_action": {
            "c_rate": state["requested_c_rate"],
            "target_soc": state["requested_target_soc"],
            "min_soc": state["requested_min_soc"],
        },
        "applied_action": {
            "c_rate": c,
            "target_soc": target,
            "min_soc": floor,
        },
        "decision": state["final_decision"],
        "reason": state["final_reason"],
        "rules_triggered": state["final_rules_triggered"],
        "ambient_c": round(state["ambient_temp_c"], 1),
        "cell_temp_c": round(temp, 1),
        "soh": round(new_soh, 3) if new_soh is not None else None,
        "range_km": round(range_km, 1) if range_km is not None else None,
        "counter_proposal": state.get("driver_counter_action"),
    }

    return {
        "soh": new_soh,
        "range_km": range_km,
        "cycle_count": cycle_count,
        "thermal_stress_hours": state["thermal_stress_hours"] + (n * charge_hours if temp > 35 else 0.0),
        "fast_charge_count": state["fast_charge_count"] + (n if c > 1.5 else 0),
        "hours_above_80soc": state["hours_above_80soc"] + n * above_80,
        "history": state["history"] + [record],
    }


def check_end_of_life(state: SimState) -> dict:
    if not state.get("is_hypothetical_simulation", False):
        # Interactive / single session negotiation completes after 1 step
        return {"done": True}

    max_cycles = config.MAX_YEARS * config.CYCLES_PER_YEAR
    eol_reached = (state["soh"] is not None and state["soh"] <= config.END_OF_LIFE_SOH)
    return {"done": eol_reached or state["cycle_count"] >= max_cycles}


def _route(state: SimState) -> str:
    return END if state["done"] else "driver_proposes"


def build_graph():
    g = StateGraph(SimState)
    g.add_node("driver_proposes", driver_proposes)
    g.add_node("bms_evaluates", bms_evaluates)
    g.add_node("driver_responds", driver_responds)
    g.add_node("bms_final_verifies", bms_final_verifies)
    g.add_node("apply_action", apply_action)
    g.add_node("update_battery_state", update_battery_state)
    g.add_node("check_end_of_life", check_end_of_life)

    g.add_edge(START, "driver_proposes")
    g.add_edge("driver_proposes", "bms_evaluates")
    g.add_edge("bms_evaluates", "driver_responds")
    g.add_edge("driver_responds", "bms_final_verifies")
    g.add_edge("bms_final_verifies", "apply_action")
    g.add_edge("apply_action", "update_battery_state")
    g.add_edge("update_battery_state", "check_end_of_life")
    g.add_conditional_edges("check_end_of_life", _route, {END: END, "driver_proposes": "driver_proposes"})
    return g.compile()
