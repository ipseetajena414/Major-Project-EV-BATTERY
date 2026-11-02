"""Driver and BMS agents.

The NUMBERS are decided by Python (driver profile in config.DRIVER, rules in bms.py).
The LLM, when enabled, only writes the in-character message / explanation.
With USE_LLM = False everything runs offline with templated text.
"""
import os
import sys
from typing import Optional

try:
    from . import bms, config
except (ImportError, ValueError):
    import bms, config

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI
        load_dotenv()
        key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        _llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=key)
    return _llm


def _extract_text(res) -> str:
    if hasattr(res, "text") and res.text:
        return res.text.strip()
    if isinstance(res.content, str):
        return res.content.strip()
    if isinstance(res.content, list):
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in res.content]
        return "".join(parts).strip()
    return str(res.content).strip()


def _say(prompt: str, fallback: str, block_index: int, use_llm: Optional[bool] = None) -> str:
    """Call Gemini only every N blocks, and fall back to the template on any error."""
    llm_enabled = config.USE_LLM if use_llm is None else bool(use_llm)
    if not llm_enabled or block_index % config.LLM_EVERY_N_BLOCKS != 0:
        return fallback
    try:
        res = _get_llm().invoke(prompt)
        return _extract_text(res)
    except Exception as exc:  # quota, bad key, retired model name, network...
        return f"{fallback} [LLM unavailable: {type(exc).__name__}]"


def driver_propose(block_index: int, ambient_c: float, soh: Optional[float],
                   persona: str = "aggressive", driver_config: dict = None,
                   use_llm: Optional[bool] = None) -> dict:
    """Driver proposes initial charging parameters.

    If SOH is None (missing/unmeasured), SOH is explicitly reported as unavailable and never fabricated.
    """
    cfg = driver_config or config.DRIVER
    rates = cfg.get("requested_c_rates", [2.0, 2.5, 3.0])
    c_rate = rates[block_index % len(rates)]
    target = cfg.get("target_soc", 100.0)
    floor = cfg.get("min_soc", 5.0)

    soh_str = f"{soh:.1f}%" if soh is not None else "UNAVAILABLE"
    soh_prompt_str = f"battery health {soh:.0f}%" if soh is not None else "battery health unmeasured"

    fallback = (f"Ambient is {ambient_c:.0f}C (battery health {soh_str}) and I'm in a hurry: "
                f"charge {floor:.0f}% -> {target:.0f}% at {c_rate}C.")
    prompt = (f"You are an impatient EV driver persona ({persona}). In ONE sentence, demand a fast charge. "
              f"Facts: ambient {ambient_c:.0f}C, {soh_prompt_str}, "
              f"you want {floor:.0f}% to {target:.0f}% at {c_rate}C.")

    return {
        "c_rate": c_rate,
        "target_soc": target,
        "min_soc": floor,
        "persona": persona,
        "message": _say(prompt, fallback, block_index, use_llm=use_llm),
    }


def bms_explain(block_index: int, decision: dict, requested_c: float, round_num: int = 1,
                use_llm: Optional[bool] = None) -> str:
    """BMS supervisor's explanation of an evaluation decision."""
    rules_text = f" [Rules: {', '.join(decision.get('rules_triggered', []))}]" if decision.get("rules_triggered") else ""
    fallback = (f"ROUND {round_num} {decision['decision'].upper()}: {decision['reason']}{rules_text}. "
                f"Offered {decision['c_rate']}C, {decision['min_soc']:.0f}-{decision['target_soc']:.0f}% SOC.")
    prompt = ("You are a strict battery management system supervisor. In ONE sentence, explain this decision to the driver. "
              f"Driver asked {requested_c}C. Decision: {decision['decision']}. Reason: {decision['reason']}. "
              f"Permitted rate: {decision['c_rate']}C.")
    return _say(prompt, fallback, block_index, use_llm=use_llm)


def driver_respond_to_bms(round_num: int, bms_eval: dict, requested: dict,
                          ambient_c: float, driver_config: dict = None) -> dict:
    """Driver persona evaluates BMS Round 1 feedback and submits counter-proposal or acceptance."""
    cfg = driver_config or config.DRIVER
    decision = bms_eval["decision"]
    offered_c = bms_eval["c_rate"]
    concession_step = cfg.get("concession_c_rate_step", 0.3)
    min_acceptable = cfg.get("min_acceptable_c_rate", 1.0)

    if decision == "approve":
        msg = f"Accepting BMS approval: charging at {offered_c}C up to {bms_eval['target_soc']:.0f}% SOC."
        return {
            "action": "accept",
            "c_rate": offered_c,
            "target_soc": bms_eval["target_soc"],
            "min_soc": bms_eval["min_soc"],
            "message": msg,
        }

    if decision == "modify":
        # Driver checks if compromise counter-offer is possible
        diff = requested["c_rate"] - offered_c
        if diff > concession_step and offered_c >= min_acceptable:
            compromise_c = round(offered_c + 0.2, 2)
            msg = (f"BMS offered {offered_c}C, but I am on a tight schedule. "
                   f"Counter-proposing compromise at {compromise_c}C with target {bms_eval['target_soc']:.0f}% SOC.")
            return {
                "action": "counter",
                "c_rate": compromise_c,
                "target_soc": bms_eval["target_soc"],
                "min_soc": bms_eval["min_soc"],
                "message": msg,
            }
        else:
            msg = f"Accepting BMS safety modification: charging at {offered_c}C, {bms_eval['min_soc']:.0f}-{bms_eval['target_soc']:.0f}% SOC."
            return {
                "action": "accept",
                "c_rate": offered_c,
                "target_soc": bms_eval["target_soc"],
                "min_soc": bms_eval["min_soc"],
                "message": msg,
            }

    # If decision == "reject"
    if offered_c > 0.0:
        msg = f"Fast charging was refused. Reluctantly accepting slow charging at {offered_c}C."
        return {
            "action": "accept",
            "c_rate": offered_c,
            "target_soc": bms_eval["target_soc"],
            "min_soc": bms_eval["min_soc"],
            "message": msg,
        }
    else:
        msg = "Charging completely refused due to critical thermal safety. Session aborted."
        return {
            "action": "abort",
            "c_rate": 0.0,
            "target_soc": bms_eval["target_soc"],
            "min_soc": bms_eval["min_soc"],
            "message": msg,
        }


def bms_final_verify(driver_response: dict, ambient_c: float, scenario: dict,
                     bms_limits: dict = None, initial_bms_eval: dict = None) -> dict:
    """Final deterministic safety check on the driver's response / counter-proposal.

    CRITICAL ARCHITECTURAL GUARANTEE:
    No action executes without passing this deterministic safety check.
    If the driver counter-proposes an unsafe compromise, BMS strictly overrides to safe limits.
    """
    lim = config.get_bms_limits(bms_limits)

    if driver_response.get("action") == "abort":
        return {
            "final_decision": "reject",
            "final_reason": "Charging aborted due to thermal safety",
            "c_rate": 0.0,
            "target_soc": driver_response.get("target_soc", 80.0),
            "min_soc": driver_response.get("min_soc", 20.0),
            "rules_triggered": ["RULE_CATASTROPHIC_SHUTDOWN"],
            "final_bms_message": "Charging session aborted safely (0.0C).",
        }

    # Re-evaluate the driver's proposed action deterministically
    req_c = driver_response["c_rate"]
    req_target = driver_response["target_soc"]
    req_floor = driver_response["min_soc"]

    eval_result = bms.evaluate(req_c, req_target, req_floor, ambient_c, scenario, lim)

    if driver_response.get("action") == "counter":
        if eval_result["c_rate"] == req_c and eval_result["decision"] in ("approve", "modify"):
            final_decision = "compromise_approved"
            final_reason = f"Driver compromise of {req_c}C evaluated as safe under active constraints"
            bms_msg = f"Compromise approved: {req_c}C satisfies thermal and current limits."
        else:
            final_decision = "modify"
            final_reason = f"Driver counter-proposal ({req_c}C) exceeded safety limit; overridden to {eval_result['c_rate']}C"
            bms_msg = f"Counter-proposal {req_c}C rejected for safety. Enforcing hard limit of {eval_result['c_rate']}C."
    else:
        final_decision = eval_result["decision"]
        final_reason = eval_result["reason"]
        bms_msg = f"Final action verified safe at {eval_result['c_rate']}C ({eval_result['min_soc']:.0f}-{eval_result['target_soc']:.0f}% SOC)."

    return {
        "final_decision": final_decision,
        "final_reason": final_reason,
        "c_rate": eval_result["c_rate"],
        "target_soc": eval_result["target_soc"],
        "min_soc": eval_result["min_soc"],
        "rules_triggered": eval_result["rules_triggered"],
        "final_bms_message": bms_msg,
    }
