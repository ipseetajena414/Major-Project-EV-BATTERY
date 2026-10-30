"""Adapter over Member 4 (``ev_member4``). Safety decisions stay in Member 4's deterministic ``bms.py``."""
from __future__ import annotations
from typing import Any

from .paths import M4_DIR, ensure_repo_on_path

ensure_repo_on_path()

_ERR: str | None = None
_M: dict[str, Any] = {}


def _load():
    global _ERR
    if _M or _ERR:
        return
    import sys
    try:
        if str(M4_DIR) not in sys.path:
            sys.path.insert(0, str(M4_DIR))
        import config, bms, run_simulation  # Member 4's flat modules
        _M.update(config=config, bms=bms, run=run_simulation)
    except Exception as exc:
        _ERR = f"{type(exc).__name__}: {exc}"


def availability() -> tuple[bool, str | None]:
    _load()
    return (not _ERR), _ERR


def options() -> dict[str, list[str]]:
    _load()
    c = _M["config"]
    return {"chemistries": list(c.CHEMISTRIES), "scenarios": list(c.SCENARIOS), "climates": list(c.CLIMATES)}


def llm_enabled() -> bool:
    _load()
    return bool(_M["config"].USE_LLM)


def evaluate_request(c_rate: float, target_soc: float, min_soc: float, ambient_c: float, scenario: str) -> dict[str, Any]:
    """One deterministic BMS decision (scenario input, not a measurement)."""
    _load()
    c, bms = _M["config"], _M["bms"]
    d = bms.evaluate(c_rate, target_soc, min_soc, ambient_c, c.SCENARIOS[scenario])
    d["requested"] = {"c_rate": c_rate, "target_soc": target_soc, "min_soc": min_soc, "ambient_c": ambient_c}
    d["limits"] = dict(c.BMS_LIMITS)
    return d


def simulate(chemistry: str, scenario: str, climate: str, vehicle: str | None = None, cycle: str | None = None):
    """Full lifecycle simulation. Degradation coefficients are Member 4 PLACEHOLDERS, not learned from data."""
    _load()
    return _M["run"].run_simulation(chemistry, scenario, climate, vehicle=vehicle, cycle=cycle)


def compare_scenarios(chemistry: str, climate: str, vehicle: str | None = None, cycle: str | None = None) -> list[dict]:
    _load()
    return [simulate(chemistry, s, climate, vehicle, cycle) for s in _M["config"].SCENARIOS]
