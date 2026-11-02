"""Run the Driver-vs-BMS simulation and provide Member 5 Dashboard API.

    python run_simulation.py                          # hot climate, default vehicle & cycle
    python run_simulation.py temperate                # temperate climate
    python run_simulation.py all                      # all climates
    python run_simulation.py hot --vehicle tata_nexon_ev_45kwh_partial_reference.json
    python run_simulation.py --list-vehicles          # view available vehicles
    python run_simulation.py --list-cycles            # view available drive cycles

Member 5 (Streamlit Dashboard) Import:
    from ev_member4 import run_simulation, simulate_bms_negotiation, run_all
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Path bootstrapping for both package-level and standalone executions
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_PKG_DIR)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

try:
    from . import agents, bms, config, placeholders, state
    from .config import (
        set_active_vehicle,
        set_active_drive_cycle,
        list_available_vehicles,
        list_available_drive_cycles,
        RULES_CATALOG,
        get_bms_limits,
    )
    from .graph import build_graph
    from .placeholders import provenance
    from .state import initial_state
except (ImportError, ValueError):
    import agents, bms, config, placeholders, state
    from config import (
        set_active_vehicle,
        set_active_drive_cycle,
        list_available_vehicles,
        list_available_drive_cycles,
        RULES_CATALOG,
        get_bms_limits,
    )
    from graph import build_graph
    from placeholders import provenance
    from state import initial_state

SCHEMA_VERSION = "2.0.0"
_GRAPH = build_graph()


def simulate_bms_negotiation(
    battery_state: Optional[Dict[str, Any]] = None,
    member2_predictions: Optional[Dict[str, Any]] = None,
    driver_request: Optional[Dict[str, Any]] = None,
    requested_action: Optional[Dict[str, Any]] = None,
    climate: str = "hot",
    vehicle: Optional[str] = None,
    cycle: Optional[str] = None,
    scenario: str = "bms_supervised",
    bms_limits: Optional[Dict[str, Any]] = None,
    bms_limits_override: Optional[Dict[str, Any]] = None,
    use_llm: Optional[bool] = None,
    **kwargs,
) -> dict:
    """Single-turn / interactive BMS negotiation endpoint for Member 5's Streamlit Dashboard.

    Accepts:
      - battery_state: dict with 'soc_pct'/'soc', 'soh_pct'/'soh', 'cell_temp_c'/'temp_c', etc.
      - member2_predictions: dict with 'soh_pct', 'rul_cycles', model provenance
      - driver_request / requested_action: dict with 'c_rate'/'charge_rate_c', 'target_soc', 'min_soc', 'persona'
      - bms_limits / bms_limits_override: optional overrides for BMS safety thresholds
      - use_llm: optional toggle (True for live Gemini, False for fast offline templates)

    Returns complete transcript, rules triggered, applied actions, and explicit SOH/RUL statuses.
    """
    if vehicle:
        set_active_vehicle(vehicle)
    if cycle:
        set_active_drive_cycle(cycle)

    limits = get_bms_limits(bms_limits or bms_limits_override)
    bat = dict(battery_state or {})
    m2 = dict(member2_predictions or {})
    d_req = dict(driver_request or requested_action or {})

    ambient = float(config.CLIMATES.get(climate, config.CLIMATES["hot"])["base"])
    m2_soh = m2.get("soh_pct", m2.get("soh"))
    m2_rul = m2.get("rul_cycles", m2.get("rul"))

    # Determine known SOH without fabrication
    current_soh = bat.get("soh_pct", bat.get("soh", m2_soh))
    current_soc = bat.get("soc_pct", bat.get("soc", 50.0))
    current_temp = bat.get("cell_temp_c", bat.get("temp_c", 25.0))

    # 1. Driver Request (Round 1)
    req_c = float(d_req.get("c_rate", d_req.get("charge_rate_c", config.DRIVER["requested_c_rates"][0])))
    req_target = float(d_req.get("target_soc", config.DRIVER["target_soc"]))
    req_floor = float(d_req.get("min_soc", config.DRIVER["min_soc"]))
    persona = d_req.get("persona", "aggressive")

    d_msg = d_req.get("message")
    if not d_msg:
        p = agents.driver_propose(0, ambient, current_soh, persona=persona, use_llm=use_llm)
        d_msg = p["message"]

    scen = config.SCENARIOS.get(scenario, config.SCENARIOS["bms_supervised"])

    # 2. BMS Evaluation (Round 1)
    r1_eval = bms.evaluate(req_c, req_target, req_floor, ambient, scen, limits)
    r1_msg = agents.bms_explain(0, r1_eval, req_c, round_num=1, use_llm=use_llm)

    # 3. Driver Counter-Proposal / Acceptance (Round 2)
    driver_resp = agents.driver_respond_to_bms(
        round_num=2,
        bms_eval=r1_eval,
        requested={"c_rate": req_c, "target_soc": req_target, "min_soc": req_floor},
        ambient_c=ambient,
    )

    # 4. BMS Final Verification (Deterministic Safety Guarantee)
    verif = agents.bms_final_verify(driver_resp, ambient, scen, limits)

    # Real-world range calculation via Member 3 physics engine
    range_km = placeholders.estimate_range_km(current_soh, ambient, "NMC") if current_soh is not None else None

    transcript = [
        {
            "round": 1,
            "speaker": f"Driver ({persona})",
            "message": d_msg,
            "action": {"c_rate": req_c, "target_soc": req_target, "min_soc": req_floor},
        },
        {
            "round": 1,
            "speaker": "BMS Safety Supervisor",
            "message": r1_msg,
            "decision": r1_eval["decision"],
            "reason": r1_eval["reason"],
            "rules_triggered": r1_eval["rules_triggered"],
            "safe_envelope": r1_eval["safe_envelope"],
        },
        {
            "round": 2,
            "speaker": f"Driver ({persona})",
            "message": driver_resp["message"],
            "action_type": driver_resp["action"],
            "counter_action": {
                "c_rate": driver_resp["c_rate"],
                "target_soc": driver_resp["target_soc"],
                "min_soc": driver_resp["min_soc"],
            },
        },
        {
            "round": 2,
            "speaker": "BMS Safety Supervisor (Final Verification)",
            "message": verif["final_bms_message"],
            "decision": verif["final_decision"],
            "reason": verif["final_reason"],
            "rules_triggered": verif["rules_triggered"],
            "applied_action": {
                "c_rate": verif["c_rate"],
                "target_soc": verif["target_soc"],
                "min_soc": verif["min_soc"],
            },
        },
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "SUCCESS",
        "climate": climate,
        "vehicle": os.path.basename(config.VEHICLE_CONFIG),
        "drive_cycle": os.path.basename(config.DRIVE_CYCLE),
        "ambient_temp_c": ambient,
        "battery_state": {
            "soc_pct": current_soc,
            "soh_pct": current_soh,
            "cell_temp_c": current_temp,
            "range_km": range_km,
        },
        "member2_predictions": member2_predictions,
        "member2_soh_pct": m2_soh,
        "member2_soh_status": "AVAILABLE" if m2_soh is not None else "UNAVAILABLE",
        "member2_rul_cycles": m2_rul,
        "member2_rul_status": "AVAILABLE" if m2_rul is not None else "UNAVAILABLE",
        "driver_proposal": {
            "round": 1,
            "message": d_msg,
            "action": "charge_fast" if req_c > 1.0 else "charge_normal",
            "charge_rate_c": req_c,
            "target_soc": req_target,
        },
        "bms_evaluation": {
            "round": 1,
            "message": r1_msg,
            "decision": r1_eval["decision"],
            "safe_charge_rate_c": r1_eval["c_rate"],
            "safe_target_soc": r1_eval["target_soc"],
            "rules_triggered": r1_eval["rules_triggered"],
        },
        "driver_counter": {
            "round": 2,
            "message": driver_resp["message"],
            "action": driver_resp["action"],
            "charge_rate_c": driver_resp["c_rate"],
            "target_soc": driver_resp["target_soc"],
        },
        "final_applied_action": {
            "round": 2,
            "action": "charge_fast" if verif["c_rate"] > 1.0 else ("cutoff" if verif["c_rate"] == 0.0 else "charge_normal"),
            "charge_rate_c": verif["c_rate"],
            "target_soc": verif["target_soc"],
            "decision": verif["final_decision"],
            "rules_triggered": verif["rules_triggered"],
            "reason": verif["final_reason"],
        },
        "requested_action": {"c_rate": req_c, "target_soc": req_target, "min_soc": req_floor},
        "applied_action": {
            "c_rate": verif["c_rate"],
            "target_soc": verif["target_soc"],
            "min_soc": verif["min_soc"],
        },
        "decision": verif["final_decision"],
        "reason": verif["final_reason"],
        "rules_triggered": verif["rules_triggered"],
        "rules_catalog": RULES_CATALOG,
        "negotiation_transcript": transcript,
        "transcript": transcript,
        "provenance": {
            "is_hypothetical_simulation": False,
            "soh_prediction_source": (m2.get("provenance") or "Member 2 Model") if m2_soh is not None else "UNAVAILABLE (No prediction provided)",
            "safety_rules_version": SCHEMA_VERSION,
        },
        "recommendations": (
            "Keep daily charge targets at or below 80% SOC and avoid DC fast charging above 35°C cell temperature. "
            "Pre-condition the battery pack in extreme hot or cold ambient conditions."
        ),
    }


def run_simulation(
    chemistry: str,
    scenario: str,
    climate: str = "hot",
    vehicle: Optional[str] = None,
    cycle: Optional[str] = None,
    battery_state: Optional[Dict[str, Any]] = None,
    member2_predictions: Optional[Dict[str, Any]] = None,
    bms_limits: Optional[Dict[str, Any]] = None,
    is_hypothetical_simulation: bool = True,
    hypothetical_initial_soh: float = 100.0,
) -> dict:
    """Run simulation loop across multiple cycles with genuine 2-round negotiation.

    For multi-year comparative curves, is_hypothetical_simulation=True calculates degradation
    via assumed comparative coefficients (clearly labeled).
    """
    if vehicle:
        set_active_vehicle(vehicle)
    if cycle:
        set_active_drive_cycle(cycle)

    blocks = config.MAX_YEARS * config.CYCLES_PER_YEAR // config.CYCLES_PER_BLOCK
    st = initial_state(
        chemistry=chemistry,
        scenario_name=scenario,
        climate_name=climate,
        battery_state=battery_state,
        member2_predictions=member2_predictions,
        bms_limits=bms_limits,
        is_hypothetical_simulation=is_hypothetical_simulation,
        hypothetical_initial_soh=hypothetical_initial_soh,
    )

    final = _GRAPH.invoke(st, config={"recursion_limit": blocks * 7 + 100})
    eol_reached = (final["soh"] is not None and final["soh"] <= config.END_OF_LIFE_SOH)
    years = final["cycle_count"] / config.CYCLES_PER_YEAR

    m2 = member2_predictions or {}
    m2_soh = m2.get("soh_pct")
    m2_rul = m2.get("rul_cycles")

    return {
        "schema_version": SCHEMA_VERSION,
        "chemistry": chemistry,
        "scenario": scenario,
        "climate": climate,
        "vehicle": os.path.basename(config.VEHICLE_CONFIG),
        "drive_cycle": os.path.basename(config.DRIVE_CYCLE),
        "years_to_eol": round(years, 2) if is_hypothetical_simulation else None,
        "eol_reached": eol_reached,
        "final_soh": round(final["soh"], 2) if final["soh"] is not None else None,
        "final_range_km": round(final["range_km"], 1) if final["range_km"] is not None else None,
        "thermal_stress_hours": round(final["thermal_stress_hours"], 1),
        "fast_charge_count": final["fast_charge_count"],
        "bms_decisions": final["decision_counts"],
        "member2_predictions": member2_predictions,
        "member2_soh_pct": m2_soh,
        "member2_soh_status": "AVAILABLE" if m2_soh is not None else "UNAVAILABLE",
        "member2_rul_cycles": m2_rul,
        "member2_rul_status": "AVAILABLE" if m2_rul is not None else "UNAVAILABLE",
        "is_hypothetical_simulation": is_hypothetical_simulation,
        "disclaimer": config.HYPOTHETICAL_SIMULATION_DISCLAIMER if is_hypothetical_simulation else None,
        "negotiation_transcript": final["history"],
        "history": final["history"],
    }


def run_all(climate: str = "hot", vehicle: Optional[str] = None, cycle: Optional[str] = None) -> list:
    if vehicle:
        set_active_vehicle(vehicle)
    if cycle:
        set_active_drive_cycle(cycle)

    prov = provenance()
    v_name = prov["range"].get("vehicle_name", os.path.basename(config.VEHICLE_CONFIG))
    c_name = prov["range"].get("drive_cycle", os.path.basename(config.DRIVE_CYCLE))
    print(f"Vehicle: {v_name} | Cycle: {c_name} | Rated ref: {config.RATED_RANGE_KM} km", flush=True)

    total = len(config.CHEMISTRIES) * len(config.SCENARIOS)
    results = []
    idx = 1
    for chem in config.CHEMISTRIES:
        for scen in config.SCENARIOS:
            print(f"[{idx}/{total}] Simulating {chem} | {scen} ({climate})...", flush=True)
            res = run_simulation(chem, scen, climate)
            print(f"    -> Complete: EOL in {res['years_to_eol']} yrs | Final SOH: {res['final_soh']}%", flush=True)
            results.append(res)
            idx += 1
    return results


def make_guidelines(results: list) -> str:
    """Rule-based guideline text built from simulation numbers (LLM polish optional)."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "negotiation_transcript")} for r in results])
    lines = [
        "Battery Preservation Guidelines (from simulation findings):",
        "DISCLAIMER: Guidelines derived from hypothetical degradation comparison and BMS safety enforcement.",
    ]
    for chem, g in df.groupby("chemistry", sort=False):
        base = g[g.scenario == "unrestricted"].iloc[0]
        best = g.sort_values("years_to_eol", ascending=False).iloc[0]
        gain = best.years_to_eol - base.years_to_eol
        lines.append(f"- {chem}: unrestricted fast charging reaches end-of-life in "
                     f"{base.years_to_eol:.1f} yrs; best strategy '{best.scenario}' gives "
                     f"{best.years_to_eol:.1f} yrs ({gain:+.1f} yrs).")
    lines += [
        f"- Keep SOC between {config.BMS_LIMITS['min_soc']:.0f}% and {config.BMS_LIMITS['max_soc']:.0f}% for daily use.",
        f"- Avoid DC fast charging when cell temperature would exceed {config.BMS_LIMITS['max_temp_c']:.0f}C; "
        f"cap charge rate at {config.BMS_LIMITS['max_c_rate']}C.",
        "- Pre-condition the pack before fast charging in hot or cold weather.",
    ]
    text = "\n".join(lines)

    if config.USE_LLM:
        try:
            from agents import _get_llm, _extract_text
            res = _get_llm().invoke(
                "Rewrite these EV battery findings as a short, friendly list of advice for drivers. "
                "Keep every number unchanged.\n\n" + text)
            text = _extract_text(res)
        except Exception as exc:
            text += f"\n[LLM polish unavailable: {type(exc).__name__}]"
    return text


def export(results: list, climate: str, out_dir: str = None) -> Path:
    out = Path(out_dir or f"results/{climate}")
    out.mkdir(parents=True, exist_ok=True)

    summary = [{k: v for k, v in r.items() if k not in ("history", "negotiation_transcript")} for r in results]
    curves = {}
    for r in results:
        curves.setdefault(r["chemistry"], {})[r["scenario"]] = [
            {"year": h["year"], "soh": h["soh"], "range_km": h["range_km"]} for h in r["history"]
        ]

    # Full negotiation transcript from first run for dashboard visualization
    sample_transcript = results[0].get("negotiation_transcript", []) if results else []

    payload = {
        "schema_version": SCHEMA_VERSION,
        "climate": climate,
        "model_provenance": provenance(),
        "rules_catalog": RULES_CATALOG,
        "vehicle_config": os.path.basename(config.VEHICLE_CONFIG),
        "drive_cycle": os.path.basename(config.DRIVE_CYCLE),
        "summary": summary,
        "hypothetical_simulation_curves": curves,
        "sample_negotiation_transcript": sample_transcript[:10],
        "guidelines": make_guidelines(results),
        "disclaimer": config.HYPOTHETICAL_SIMULATION_DISCLAIMER,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, (chem, scens) in zip(axes.flat, curves.items()):
        for name, pts in scens.items():
            ax.plot([p["year"] for p in pts], [p["soh"] for p in pts], label=name)
        ax.axhline(config.END_OF_LIFE_SOH, color="grey", ls="--", lw=0.8)
        ax.set_title(chem)
        ax.set_xlabel("Years")
        ax.set_ylabel("SOH (%)")
    axes[0, 0].legend(fontsize=8)
    v_label = provenance()["range"].get("vehicle_name") or os.path.basename(config.VEHICLE_CONFIG)
    fig.suptitle(f"Hypothetical SOH vs Years by Charging Strategy ({climate} climate) — {v_label}")
    fig.tight_layout()
    fig.savefig(out / "soh_curves.png", dpi=130)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Driver persona vs BMS safety supervisor simulation")
    parser.add_argument("climate", nargs="?", default="hot",
                         choices=["hot", "temperate", "cold", "all"],
                         help="Target climate (hot, temperate, cold, or all)")
    parser.add_argument("-v", "--vehicle", type=str, default=None,
                        help="Vehicle config filename or path")
    parser.add_argument("-c", "--cycle", type=str, default=None,
                        help="Drive cycle CSV filename or path")
    parser.add_argument("--list-vehicles", action="store_true",
                        help="List all available vehicle configurations and exit")
    parser.add_argument("--list-cycles", action="store_true",
                        help="List all available drive cycles and exit")
    parser.add_argument("-o", "--out-dir", type=str, default=None,
                        help="Custom output directory")
    args = parser.parse_args()

    if args.list_vehicles:
        print("\nAvailable vehicle configurations:")
        for v in list_available_vehicles():
            print(f"  * {v['file']:45s} | Pack: {v['nominal_pack_energy_kwh']:4.1f} kWh | Ref: {str(v['rated_range_km_reference']):5s} km | {v['status']}")
        print()
        return

    if args.list_cycles:
        print("\nAvailable drive cycles:")
        for c in list_available_drive_cycles():
            print(f"  * {c['file']}")
        print()
        return

    if args.vehicle:
        set_active_vehicle(args.vehicle)
    if args.cycle:
        set_active_drive_cycle(args.cycle)

    climates = list(config.CLIMATES) if args.climate == "all" else [args.climate]
    for climate in climates:
        results = run_all(climate)
        out = export(results, climate, out_dir=args.out_dir)
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "bms_decisions", "negotiation_transcript")} for r in results])
        print(df.to_string(index=False))
        print("\n" + make_guidelines(results))
        print(f"\nSaved: {out}/results.json, summary.csv, soh_curves.png\n")


if __name__ == "__main__":
    main()
