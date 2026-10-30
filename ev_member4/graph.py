"""LangGraph loop:

driver_proposes -> bms_evaluates -> apply_action -> update_battery_state -> check_end_of_life
                       ^                                                         |
                       +------------------ (not end of life) --------------------+
Each loop iteration simulates CYCLES_PER_BLOCK cycles of identical behaviour.
"""
import math

from langgraph.graph import END, START, StateGraph

import agents
import bms
import config
from placeholders import estimate_range_km, predict_soh
from state import SimState


def _block_index(state: SimState) -> int:
    return state["cycle_count"] // config.CYCLES_PER_BLOCK


def _ambient(state: SimState) -> float:
    year_frac = state["cycle_count"] / config.CYCLES_PER_YEAR
    c = state["climate"]
    return c["base"] + c["amp"] * math.sin(2 * math.pi * year_frac)


def driver_proposes(state: SimState) -> dict:
    ambient = _ambient(state)
    p = agents.driver_propose(_block_index(state), ambient, state["soh"])
    return {
        "ambient_temp_c": ambient,
        "requested_c_rate": p["c_rate"],
        "requested_target_soc": p["target_soc"],
        "requested_min_soc": p["min_soc"],
        "driver_message": p["message"],
    }


def bms_evaluates(state: SimState) -> dict:
    d = bms.evaluate(state["requested_c_rate"], state["requested_target_soc"],
                     state["requested_min_soc"], state["ambient_temp_c"], state["scenario"])
    counts = dict(state["decision_counts"])
    counts[d["decision"]] += 1
    return {
        "bms_decision": d["decision"],
        "bms_reason": d["reason"],
        "bms_message": agents.bms_explain(_block_index(state), d, state["requested_c_rate"]),
        "applied_c_rate": d["c_rate"],
        "applied_target_soc": d["target_soc"],
        "applied_min_soc": d["min_soc"],
        "decision_counts": counts,
    }


def apply_action(state: SimState) -> dict:
    cell_temp = bms.predict_cell_temp(state["ambient_temp_c"], state["applied_c_rate"],
                                      state["scenario"]["precondition"])
    return {"cell_temp_c": cell_temp, "soc": state["applied_target_soc"]}


def update_battery_state(state: SimState) -> dict:
    n = config.CYCLES_PER_BLOCK
    c, target, floor = state["applied_c_rate"], state["applied_target_soc"], state["applied_min_soc"]
    temp = state["cell_temp_c"]

    charge_hours = max(0.0, (target - floor) / 100.0 / c)
    dwell = config.HIGH_SOC_DWELL_HOURS if target > 80 else 0.0
    above_80 = dwell + max(0.0, target - 80) / 100.0 / c

    features = {
        "mean_temp_c": temp,
        "max_c_rate": c,
        "target_soc": target,
        "min_soc": floor,
        "hours_above_80soc_per_cycle": above_80,
        "fast_charge_fraction": 1.0 if c > 1.5 else 0.0,
        "n_cycles": n,
    }
    new_soh = predict_soh(state["chemistry"], state["soh"], features)   # <- Member 2 plugs in here
    range_km = estimate_range_km(new_soh, state["ambient_temp_c"], state["chemistry"])  # <- Member 3

    cycle_count = state["cycle_count"] + n
    record = {
        "cycle": cycle_count,
        "year": round(cycle_count / config.CYCLES_PER_YEAR, 3),
        "soh": round(new_soh, 3),
        "range_km": round(range_km, 1),
        "ambient_c": round(state["ambient_temp_c"], 1),
        "cell_temp_c": round(temp, 1),
        "requested_c_rate": state["requested_c_rate"],
        "applied_c_rate": c,
        "soc_window": f"{floor:.0f}-{target:.0f}",
        "decision": state["bms_decision"],
        "reason": state["bms_reason"],
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
    max_cycles = config.MAX_YEARS * config.CYCLES_PER_YEAR
    return {"done": state["soh"] <= config.END_OF_LIFE_SOH or state["cycle_count"] >= max_cycles}


def _route(state: SimState) -> str:
    return END if state["done"] else "driver_proposes"


def build_graph():
    g = StateGraph(SimState)
    g.add_node("driver_proposes", driver_proposes)
    g.add_node("bms_evaluates", bms_evaluates)
    g.add_node("apply_action", apply_action)
    g.add_node("update_battery_state", update_battery_state)
    g.add_node("check_end_of_life", check_end_of_life)

    g.add_edge(START, "driver_proposes")
    g.add_edge("driver_proposes", "bms_evaluates")
    g.add_edge("bms_evaluates", "apply_action")
    g.add_edge("apply_action", "update_battery_state")
    g.add_edge("update_battery_state", "check_end_of_life")
    g.add_conditional_edges("check_end_of_life", _route, {END: END, "driver_proposes": "driver_proposes"})
    return g.compile()
