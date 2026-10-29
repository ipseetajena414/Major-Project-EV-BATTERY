"""BMS safety rules. Pure Python: decisions are deterministic, the LLM only explains them."""
import config


def predict_cell_temp(ambient_c: float, c_rate: float, precondition: bool) -> float:
    raw = ambient_c + config.CELL_HEATING_C_PER_C_RATE * c_rate
    if precondition:
        raw += (25.0 - raw) * config.PRECONDITION_STRENGTH
    return raw


def evaluate(c_rate: float, target_soc: float, min_soc: float,
             ambient_c: float, scenario: dict) -> dict:
    lim = config.BMS_LIMITS
    reasons = []
    refused = False

    if scenario["enforce_soc"]:
        if target_soc > lim["max_soc"]:
            reasons.append(f"SOC target {target_soc:.0f}% capped to {lim['max_soc']:.0f}%")
            target_soc = lim["max_soc"]
        if min_soc < lim["min_soc"]:
            reasons.append(f"discharge floor {min_soc:.0f}% raised to {lim['min_soc']:.0f}%")
            min_soc = lim["min_soc"]

    if scenario["enforce_c_rate"] and c_rate > lim["max_c_rate"]:
        reasons.append(f"{c_rate:.1f}C exceeds ceiling, limited to {lim['max_c_rate']}C")
        c_rate = lim["max_c_rate"]

    if scenario["enforce_temp"]:
        if ambient_c >= lim["refuse_fast_charge_above_ambient_c"]:
            reasons.append(f"ambient {ambient_c:.0f}C too hot, fast charge refused")
            c_rate = lim["fallback_c_rate"]
            refused = True
        else:
            start = c_rate
            while (predict_cell_temp(ambient_c, c_rate, scenario["precondition"]) > lim["max_temp_c"]
                   and c_rate > lim["fallback_c_rate"] + 1e-9):
                c_rate = round(c_rate - 0.1, 2)
            if c_rate < start:
                reasons.append(f"cell temp would exceed {lim['max_temp_c']:.0f}C, rate cut to {c_rate}C")

    decision = "reject" if refused else ("modify" if reasons else "approve")
    return {
        "decision": decision,
        "reason": "; ".join(reasons) if reasons else "within all limits",
        "c_rate": c_rate,
        "target_soc": target_soc,
        "min_soc": min_soc,
    }
