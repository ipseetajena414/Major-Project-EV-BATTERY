"""
Member 1 - dataset coverage matrix (evidence-based status per dataset).

Reads the inventory, cycle-candidate summaries and telemetry summaries that were
produced from the actual archives. Writes AUDIT/MEMBER1_DATASET_COVERAGE.csv.

Run:
    python AUDIT/member1_coverage.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
OUT = AUDIT_DIR / "candidates"

COLUMNS = [
    "dataset_id",
    "chemistry",
    "cell_id",
    "status",
    "target_status",
    "source_verified",
    "workbooks",
    "cycle_tabs",
    "audited_cycles",
    "labelled_cycles",
    "candidate_cycles",
    "candidate_rows",
    "temperature_measured",
    "included_in_candidate_master",
    "evidence",
    "key_limitations",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    inv = pd.read_csv(AUDIT_DIR / "MEMBER1_SOURCE_INVENTORY.csv").set_index("dataset_id")
    cyc = load_json(OUT / "member1_cycle_candidates_summary.json")
    tel = {tag: load_json(OUT / f"telemetry_{tag}_summary.json") for tag in ["B1", "B2", "B3", "B4", "B5", "B6"]}
    manifest = pd.read_csv(REPO_ROOT / "master_table_candidate_manifest.csv").set_index("dataset_id")

    rows = []

    def add(dataset_id, chemistry, cell, status, target_status, temp, evidence, limits):
        src = inv.loc[dataset_id]
        tag = dataset_id.split("_")[-1]
        c = cyc.get(tag, {})
        t = tel.get(tag, {})
        candidate_cycles = int(manifest.loc[dataset_id, "cell_cycles"]) if dataset_id in manifest.index else None
        candidate_rows = int(manifest.loc[dataset_id, "rows"]) if dataset_id in manifest.index else None
        rows.append(
            {
                "dataset_id": dataset_id,
                "chemistry": chemistry,
                "cell_id": cell,
                "status": status,
                "target_status": target_status,
                "source_verified": bool(src["opened"]),
                "workbooks": src["workbooks"],
                "cycle_tabs": src["cycle_sheets"],
                "audited_cycles": src["cycle_records"],
                "labelled_cycles": src["label_records"],
                "candidate_cycles": candidate_cycles,
                "candidate_rows": candidate_rows,
                "temperature_measured": temp,
                "included_in_candidate_master": True,
                "evidence": evidence,
                "key_limitations": limits,
            }
        )

    add(
        "LFP_MultiStat_B1", "LFP", "BATTERY_1", "CONDITIONAL", "CONDITIONAL", "NO",
        "19 workbooks, 1,460 Step2_Cycle tabs, 1,456 computed capacity/SOH labels; raw re-read for sampled telemetry; corrected (source_file, worksheet) join",
        "110 cycle offsets underdetermined (Cycle_0200 30, Cycle_1300 80); 4 partial-discharge labels blocked; I(A/cm2) header with area=0; 20 ASCII cycle exports in Cycle_1300 (23-26 Sep) cover the 1201-1220 gap but are unlabelled and not integrated (export index lists 21 cycles)",
    )
    add(
        "LFP_MultiStat_B2", "LFP", "BATTERY_2", "CONDITIONAL", "CONDITIONAL", "NO",
        "20 workbooks verified byte-identical to manifest; 1,506 cycles, 1,496 analysis-eligible, 10 invalid; q_discharge/Q-V exports match; SOH as 0-1 fraction",
        "Channel Un1Ch8->Un1Ch2 at cycle 999; cycles 999-1015 need exclusion or channel covariate; 160 provenance-inferred cycles (historical); no temperature",
    )
    add(
        "LFP_MultiStat_B3", "LFP", "BATTERY_3", "CONDITIONAL", "CONDITIONAL", "NO",
        "25 workbooks, 1,979 cycle tabs, 1,973 Qdis labels; missing ids 201-221 confirmed; sample 1.165491 Ah reproduced; 4 instrument channels",
        "Physical cell identity per channel undocumented; I(A/cm2) unit unresolved; 8 quality-X and 3 quality-B cycles; SOH denominator nominal 1.25 Ah",
    )
    add(
        "LFP_MultiStat_B4", "LFP", "BATTERY_4", "CONDITIONAL", "BLOCKED", "NO",
        "21 workbooks, 1,577 cycle tabs audited for coverage; 5 previously unaudited checkpoints (150,200,250,300,350) added; telemetry sampled",
        "No audited capacity/SOH; current unit and charge/discharge boundaries unresolved; capacity/SOH intentionally null; cycle ids reconstructed from sheet order",
    )
    add(
        "LFP_MultiStat_B5", "LFP", "BATTERY_5", "CONDITIONAL", "CONDITIONAL", "NO",
        "17 workbooks, 1,498 real cycles (1,491 Step2 + 7 renamed); all workbooks sampled (canonical had 4); cycle-40 partial 0.5428382 Ah reproduced",
        "capacity_ground_truth=NO for all cycles; Qdis>Qchg in 1,488 of 1,497 comparable cycles; 6 partial discharges; old audit's 1,499 rows include a misclassified File List tab",
    )
    add(
        "LFP_MultiStat_B6", "LFP", "BATTERY_6", "CONDITIONAL", "CONDITIONAL", "NO",
        "19 workbooks, 1,702 cycles with validated chronology; 1,697 conditional discharge labels; header and File List timestamps reconciled",
        "Chronology reconstructed (no global cycle id in source); current unit underdetermined; no temperature",
    )
    add(
        "NASA_Ames_PCoE", "NMC", "B0005;B0006;B0018", "VERIFIED", "VERIFIED", "YES",
        "48-cell archive verified; 3 target cells with 468 discharge cycles; measured voltage/current/temperature and direct capacity",
        "Rated capacity 2.0 Ah; protocol differs from LFP cells; only 3 of 48 cells used in canonical build",
    )
    add(
        "Zenodo_IonTech_NaIon", "Na-ion", "4 canonical cells", "VERIFIED", "VERIFIED", "NO",
        "65 cell pickles verified; canonical build uses 4 cells x 25 cycles = 100 cycles; direct discharge capacity and per-cell nominal capacity",
        "Temperature not measured; only 4 of 65 cells and 25 cycles per cell in canonical build; chemistry differs from LFP",
    )

    with open(AUDIT_DIR / "MEMBER1_DATASET_COVERAGE.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(pd.DataFrame(rows)[["dataset_id", "status", "target_status", "candidate_cycles", "candidate_rows"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
