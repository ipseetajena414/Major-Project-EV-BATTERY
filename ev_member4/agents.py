"""Driver and BMS agents.

The NUMBERS are decided by Python (driver profile in config.DRIVER, rules in bms.py).
The LLM, when enabled, only writes the in-character message / explanation.
With USE_LLM = False everything runs offline with templated text.
"""
import config

_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        import os
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


def _say(prompt: str, fallback: str, block_index: int) -> str:
    """Call Gemini only every N blocks, and fall back to the template on any error."""
    if not config.USE_LLM or block_index % config.LLM_EVERY_N_BLOCKS != 0:
        return fallback
    try:
        res = _get_llm().invoke(prompt)
        return _extract_text(res)
    except Exception as exc:  # quota, bad key, retired model name, network...
        return f"{fallback} [LLM unavailable: {type(exc).__name__}]"


def driver_propose(block_index: int, ambient_c: float, soh: float) -> dict:
    """Aggressive driver: always wants a fast charge to full."""
    rates = config.DRIVER["requested_c_rates"]
    c_rate = rates[block_index % len(rates)]
    target, floor = config.DRIVER["target_soc"], config.DRIVER["min_soc"]
    fallback = (f"Ambient is {ambient_c:.0f}C and I'm in a hurry: "
                f"charge {floor:.0f}% -> {target:.0f}% at {c_rate}C.")
    prompt = ("You are an impatient EV driver persona. In ONE sentence, demand a fast charge. "
              f"Facts: ambient {ambient_c:.0f}C, battery health {soh:.0f}%, "
              f"you want {floor:.0f}% to {target:.0f}% at {c_rate}C.")
    return {"c_rate": c_rate, "target_soc": target, "min_soc": floor,
            "message": _say(prompt, fallback, block_index)}


def bms_explain(block_index: int, decision: dict, requested_c: float) -> str:
    """BMS supervisor's explanation of a decision that bms.evaluate already made."""
    fallback = (f"{decision['decision'].upper()}: {decision['reason']}. "
                f"Applied {decision['c_rate']}C, {decision['min_soc']:.0f}-{decision['target_soc']:.0f}% SOC.")
    prompt = ("You are a strict battery management system. In ONE sentence, explain this decision to the driver. "
              f"Driver asked {requested_c}C. Decision: {decision['decision']}. Reason: {decision['reason']}. "
              f"Applied {decision['c_rate']}C.")
    return _say(prompt, fallback, block_index)
