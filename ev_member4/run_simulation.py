"""Run the Driver-vs-BMS simulation.

    python run_simulation.py                          # hot climate, default vehicle & cycle
    python run_simulation.py temperate                # temperate climate
    python run_simulation.py all                      # all climates
    python run_simulation.py hot --vehicle tata_nexon_ev_45kwh_partial_reference.json
    python run_simulation.py --list-vehicles          # view available vehicles
    python run_simulation.py --list-cycles            # view available drive cycles

Member 5 can also just import:
    from run_simulation import run_simulation, run_all, set_active_vehicle, set_active_drive_cycle
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

import config
from config import set_active_vehicle, set_active_drive_cycle, list_available_vehicles, list_available_drive_cycles
from graph import build_graph
from placeholders import provenance
from state import initial_state

_GRAPH = build_graph()


def run_simulation(chemistry: str, scenario: str, climate: str = "hot",
                   vehicle: str = None, cycle: str = None) -> dict:
    """One chemistry + one scenario + one climate -> summary and full history."""
    if vehicle:
        set_active_vehicle(vehicle)
    if cycle:
        set_active_drive_cycle(cycle)

    blocks = config.MAX_YEARS * config.CYCLES_PER_YEAR // config.CYCLES_PER_BLOCK
    final = _GRAPH.invoke(initial_state(chemistry, scenario, climate),
                          config={"recursion_limit": blocks * 5 + 50})
    eol_reached = final["soh"] <= config.END_OF_LIFE_SOH
    years = final["cycle_count"] / config.CYCLES_PER_YEAR
    return {
        "chemistry": chemistry,
        "scenario": scenario,
        "climate": climate,
        "vehicle": os.path.basename(config.VEHICLE_CONFIG),
        "drive_cycle": os.path.basename(config.DRIVE_CYCLE),
        "years_to_eol": round(years, 2),
        "eol_reached": eol_reached,          # False -> still above EOL after MAX_YEARS
        "final_soh": round(final["soh"], 2),
        "final_range_km": round(final["range_km"], 1),
        "thermal_stress_hours": round(final["thermal_stress_hours"], 1),
        "fast_charge_count": final["fast_charge_count"],
        "bms_decisions": final["decision_counts"],
        "history": final["history"],
    }


def run_all(climate: str = "hot", vehicle: str = None, cycle: str = None) -> list:
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
    """Rule-based guideline text built from the numbers (LLM polish is optional)."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in results])
    lines = ["Battery preservation guidelines (from simulation results):"]
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

    summary = [{k: v for k, v in r.items() if k != "history"} for r in results]
    curves = {}
    for r in results:
        curves.setdefault(r["chemistry"], {})[r["scenario"]] = [
            {"year": h["year"], "soh": h["soh"], "range_km": h["range_km"]} for h in r["history"]]

    payload = {
        "climate": climate,
        "model_provenance": provenance(),
        "vehicle_config": os.path.basename(config.VEHICLE_CONFIG),
        "drive_cycle": os.path.basename(config.DRIVE_CYCLE),
        "summary": summary,
        "curves": curves,
        "guidelines": make_guidelines(results),
    }      # <- hand this JSON to Member 5
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
    fig.suptitle(f"SOH vs years by charging strategy ({climate} climate) — {v_label}")
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
                        help="Vehicle config filename or path (e.g. tata_curvv_ev_55kwh_partial_reference.json)")
    parser.add_argument("-c", "--cycle", type=str, default=None,
                        help="Drive cycle CSV filename or path (e.g. udds_city.csv, hwfet_highway.csv)")
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
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("history", "bms_decisions")} for r in results])
        print(df.to_string(index=False))
        print("\n" + make_guidelines(results))
        print(f"\nSaved: {out}/results.json, summary.csv, soh_curves.png\n")


if __name__ == "__main__":
    main()
