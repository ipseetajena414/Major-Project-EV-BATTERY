"""BMS safety rules. Pure Python: decisions are deterministic, the LLM only explains them.

ENGINEERING NOTE ON THERMAL ESTIMATION:
Cell temperature prediction uses a lumped-parameter heuristic (ambient + k * C-rate).
This is an uncalibrated engineering approximation for software simulation. It has NOT been
experimentally validated against calorimeter test benches or certified under automotive safety
standards (such as ISO 6469, SAE J1772, or UL 2580).
"""
import os
import sys

try:
    from . import config
except (ImportError, ValueError):
    import config


def predict_cell_temp(ambient_c: float, c_rate: float, precondition: bool = False) -> float:
    """Heuristic cell temperature estimation based on ambient and charge rate.

    Ambient rise: ambient_c + CELL_HEATING_C_PER_C_RATE * c_rate.
    Pre-conditioning: closes PRECONDITION_STRENGTH fraction of the gap to 25 deg C.
    """
    raw = ambient_c + config.CELL_HEATING_C_PER_C_RATE * max(0.0, c_rate)
    if precondition:
        raw += (25.0 - raw) * config.PRECONDITION_STRENGTH
    return round(raw, 2)


def evaluate(c_rate: float, target_soc: float, min_soc: float,
             ambient_c: float, scenario: dict, bms_limits: dict = None) -> dict:
    """Evaluate driver charge request deterministically against configured BMS safety constraints.

    Returns decision ('approve' | 'modify' | 'reject'), applied limits, rules triggered, and reason.
    """
    lim = config.get_bms_limits(bms_limits)
    reasons = []
    rules_triggered = []
    refused = False
    precondition = scenario.get("precondition", False)

    # 1. State of Charge (SOC) Bounds
    if scenario.get("enforce_soc", False):
        if target_soc > lim["max_soc"]:
            reasons.append(f"SOC target {target_soc:.0f}% capped to {lim['max_soc']:.0f}%")
            target_soc = lim["max_soc"]
            rules_triggered.append("RULE_SOC_MAX_CEILING")
        if min_soc < lim["min_soc"]:
            reasons.append(f"discharge floor {min_soc:.0f}% raised to {lim['min_soc']:.0f}%")
            min_soc = lim["min_soc"]
            rules_triggered.append("RULE_SOC_MIN_FLOOR")

    # 2. Maximum C-rate Kinetic Limit
    if scenario.get("enforce_c_rate", False) and c_rate > lim["max_c_rate"]:
        reasons.append(f"{c_rate:.1f}C exceeds ceiling, limited to {lim['max_c_rate']}C")
        c_rate = lim["max_c_rate"]
        rules_triggered.append("RULE_CRATE_CEILING")

    # 3. Thermal Protection & Dynamic Derating
    if scenario.get("enforce_temp", False):
        # A. High Ambient Fast-Charge Lockout
        if ambient_c >= lim["refuse_fast_charge_above_ambient_c"]:
            reasons.append(f"ambient {ambient_c:.0f}C too hot, fast charge refused")
            c_rate = lim["fallback_c_rate"]
            refused = True
            rules_triggered.append("RULE_THERMAL_LOCKOUT")
        else:
            # B. Core Cell Temperature Derating
            start = c_rate
            while (predict_cell_temp(ambient_c, c_rate, precondition) > lim["max_temp_c"]
                   and c_rate > lim["fallback_c_rate"] + 1e-9):
                c_rate = round(c_rate - 0.1, 2)
            if c_rate < start:
                reasons.append(f"cell temp would exceed {lim['max_temp_c']:.0f}C, rate cut to {c_rate}C")
                rules_triggered.append("RULE_CELL_OVERHEAT_DERATE")

        # C. Critical Thermal Fallback Verification
        # If even fallback C-rate cannot prevent critical temperature rise, completely abort charging
        predicted_temp = predict_cell_temp(ambient_c, c_rate, precondition)
        crit_temp = lim.get("critical_shutdown_temp_c", 50.0)
        if predicted_temp > crit_temp:
            reasons.append(f"predicted cell temp ({predicted_temp:.1f}C) exceeds critical limit ({crit_temp:.0f}C); charging aborted")
            c_rate = 0.0
            refused = True
            rules_triggered.append("RULE_CATASTROPHIC_SHUTDOWN")

    final_temp = predict_cell_temp(ambient_c, c_rate, precondition)
    decision = "reject" if refused else ("modify" if reasons else "approve")

    return {
        "decision": decision,
        "reason": "; ".join(reasons) if reasons else "Within all safety boundaries",
        "c_rate": c_rate,
        "target_soc": target_soc,
        "min_soc": min_soc,
        "rules_triggered": rules_triggered,
        "predicted_cell_temp_c": final_temp,
        "safe_envelope": {
            "max_permitted_c_rate": lim["max_c_rate"] if scenario.get("enforce_c_rate") else 3.0,
            "max_permitted_target_soc": lim["max_soc"] if scenario.get("enforce_soc") else 100.0,
            "min_permitted_floor_soc": lim["min_soc"] if scenario.get("enforce_soc") else 0.0,
        },
    }
