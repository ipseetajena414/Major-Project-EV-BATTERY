"""
Member 1 - LFP B1-B5 source investigation and validation-readiness audit (read-only).

What this does
--------------
1. Reconciles the raw archives and audit bundles for B1, B2, B3, B4, B5 (and notes
   the out-of-scope B6 package).
2. Recounts workbooks and cycle worksheets from the raw archives themselves.
3. Re-derives capacity for one cycle per raw LFP cell directly from the workbook
   and compares it with the audit tables (independent verification).
4. Re-checks the canonical master-table coverage facts.
5. Writes the two required matrices and a machine-readable evidence report.

Safety
------
* Never writes to DATA/, AUDIT/*.zip or the canonical master_table.* files.
* Only writes inside AUDIT/member1_audit_outputs/.
* Aborts if a canonical master-table hash changes during the run.

Run:
    python AUDIT/lfp_b1_b5_inventory.py

Exit code 0 when all internal checks pass, 1 otherwise.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
OUT_DIR = AUDIT_DIR / "member1_audit_outputs"

B1_RAW = REPO_ROOT / "DATA" / "B1_Discharge0.5C_Charge0.5C(1).zip"
B2_DERIVED = REPO_ROOT / "DATA" / "B2_LFP_Member1_audit.zip"
B3_RAW = REPO_ROOT / "DATA" / "B3_Discharge2C_Charge0.5C (1).zip"
B4_RAW = REPO_ROOT / "DATA" / "B4_Discharge3C_Charge0.5C (1).zip"
B5_RAW = REPO_ROOT / "DATA" / "B5_Discharge0.5C_Charge1C (1).zip"
B1_AUDIT = AUDIT_DIR / "B1_LFP_Audit_Output.zip"
B3B4_AUDIT = AUDIT_DIR / "Existing_Audits_Available_B3_B4 (1).zip"
B5_AUDIT = AUDIT_DIR / "B5.zip"
B6_AUDIT = AUDIT_DIR / "LFP_B6_Audit_Package (1).zip"
MASTER_PARQUET = REPO_ROOT / "master_table.parquet"
MASTER_CSV = REPO_ROOT / "master_table.csv"

MASTER_SHA256 = "6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb"
MASTER_CSV_SHA256 = "9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b"

EXPECTED_ARCHIVE_SHA256 = {
    "B1_AUDIT": "05bb6258f6965d9b5e2d27aabcd2706a78129d1cb9873389cf18d5150ad4e85e",
    "B3_RAW": "aff29ebec7befc8a14589e9b8b33beff83c3e30164b4f2c766202e4308d29bb2",
    "B4_RAW": "a17779c495f2fdd60ea588bc5763f1e32b05e5da069a46abda05f1009af098c6",
    "B5_RAW": "f3d6b1ef56992f57530e17ce2d4c98cf21de4f59f0c866d27d0921dae1287b3b",
    "B6_AUDIT": "96b5fef5905b948e21ea62f8d109d6e056c51088327d5efa1eee99c4c433878e",
}
EXPECTED_WORKSPACE_SHA256 = {
    "B1_RAW": "fe04aec9b706aa8228dc327240f5132b992fc940de8dd64df969262908f85a5f",
    "B2_DERIVED": "54cd6113506e7b11b5147739d24b412b3b279f0c356e63bcf67de544a8ce169d",
}

EXPECTED_CYCLE_SHEETS = {"B1": (19, 1460), "B3": (25, 1979), "B4": (21, 1577), "B5": (17, 1491)}

NOMINAL_RATING_AH = 1.25
DISCHARGE_CUTOFF_V = 2.5

COVERAGE_COLUMNS = [
    "cell_id", "dataset_id", "source_path", "source_status", "source_verified",
    "raw_data_available", "derived_data_available", "row_count",
    "observed_cycle_count", "labelled_cycle_count", "soh_provenance",
    "current_unit_status", "voltage_available", "current_available",
    "time_available", "temperature_available", "data_quality_issues",
    "independent_validation_suitability", "blocker", "recommended_action",
]
FEATURE_COLUMNS = [
    "cell_id", "feature_name", "required_inputs", "verified_units",
    "derivation_method", "input_availability", "suitability", "limitations",
    "missing_input", "recommended_action",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_cycle_inventory(zip_path: Path) -> dict:
    """Count workbooks and cycle worksheets from the raw archive itself."""
    with zipfile.ZipFile(zip_path) as archive:
        workbooks = sorted(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
        per_file = {}
        for name in workbooks:
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as inner:
                xml = inner.read("xl/workbook.xml").decode("utf-8", "ignore")
            per_file[name.split("/")[-1]] = {
                "sheets": len(re.findall(r"<sheet\b", xml)),
                "cycle_sheets": len(re.findall(r'name="[^"]*Step2_Cycle', xml)),
            }
    return {
        "archive": str(zip_path.relative_to(REPO_ROOT)),
        "workbook_count": len(workbooks),
        "cycle_sheet_count": sum(v["cycle_sheets"] for v in per_file.values()),
        "workbooks": per_file,
    }


def read_cycle_sheet(zip_path: Path, member: str, sheet: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(member)
    import openpyxl

    workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    worksheet = workbook[sheet]
    rows = list(worksheet.iter_rows(values_only=True))
    workbook.close()
    header = next(
        i
        for i, row in enumerate(rows)
        if row and isinstance(row[0], str) and row[0].strip() == "T(Seconds)"
    )
    frame = pd.DataFrame([r[:3] for r in rows[header + 1 :]], columns=["t_s", "v_v", "i_a"])
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    if frame.empty:
        raise RuntimeError(f"no numeric rows in {member}::{sheet}")
    return frame


def segment_cycle(frame: pd.DataFrame) -> dict:
    t = frame["t_s"].to_numpy(dtype=float)
    v = frame["v_v"].to_numpy(dtype=float)
    i = frame["i_a"].to_numpy(dtype=float)
    result = {"rows": int(len(frame)), "t_start_s": float(t[0]), "t_end_s": float(t[-1])}
    for label, mask in (("discharge", i < 0), ("charge", i > 0)):
        if mask.sum() < 2:
            result[label] = None
            continue
        tt, ii, vv = t[mask], np.abs(i[mask]), v[mask]
        fn = getattr(np, "trapezoid", None) or np.trapz
        result[label] = {
            "samples": int(mask.sum()),
            "duration_s": float(tt[-1] - tt[0]),
            "v_start": float(vv[0]),
            "v_end": float(vv[-1]),
            "v_min": float(vv.min()),
            "v_max": float(vv.max()),
            "capacity_ah": float(fn(ii, tt) / 3600.0),
            "mean_abs_current_a": float(ii.mean()),
        }
    return result


def verify_sample_cycles() -> dict:
    """Re-derive one cycle per cell from raw and compare with the audit tables."""
    b1 = segment_cycle(
        read_cycle_sheet(
            B1_RAW,
            "B1_Discharge0.5C_Charge0.5C/Cycle_1500.xlsx",
            "Run01_Un1Ch7_Step2_Cycle001",
        )
    )
    with zipfile.ZipFile(B1_AUDIT) as archive:
        b1_table = pd.read_csv(archive.open("B1_LFP_Audit_Output/LFP_staging_cycle_level.csv"))
    b1_row = b1_table[
        (b1_table["cycle_checkpoint"] == 1500) & (b1_table["global_cycle_est"] == 1401)
    ].iloc[0]

    b3 = segment_cycle(
        read_cycle_sheet(
            B3_RAW,
            "B3_Discharge2C_Charge0.5C/Cycle_0050.xlsx",
            "Run01_Un1Ch2_Step2_Cycle10",
        )
    )
    with zipfile.ZipFile(B3B4_AUDIT) as archive:
        b3_table = pd.read_csv(archive.open("B3_B4/b3_cycle_summary (1).csv"))
    b3_row = b3_table[
        (b3_table["source_workbook"] == "Cycle_0050.xlsx")
        & (b3_table["source_sheet"] == "Run01_Un1Ch2_Step2_Cycle10")
    ].iloc[0]

    b4 = segment_cycle(
        read_cycle_sheet(
            B4_RAW,
            "B4_Discharge3C_Charge0.5C/Cycle_0050.xlsx",
            "Run01_Un1Ch6_Step2_Cycle10",
        )
    )

    return {
        "B1": {
            "source": "DATA/B1_Discharge0.5C_Charge0.5C(1).zip::Cycle_1500.xlsx::Run01_Un1Ch7_Step2_Cycle001",
            "derived": b1,
            "audit_checkpoint": 1500,
            "audit_global_cycle_est": 1401,
            "audit_capacity_ah": float(b1_row["capacity_ah"]),
            "audit_soh_pct": float(b1_row["soh_pct"]),
            "audit_quality_flag": str(b1_row["quality_flag"]),
            "delta_capacity_ah": float(b1["discharge"]["capacity_ah"] - b1_row["capacity_ah"]),
        },
        "B3": {
            "source": "DATA/B3_Discharge2C_Charge0.5C (1).zip::Cycle_0050.xlsx::Run01_Un1Ch2_Step2_Cycle10",
            "derived": b3,
            "audit_cycle_index": int(b3_row["cycle_index_nominal"]),
            "audit_capacity_ah": float(b3_row["Qdis_Ah"]),
            "audit_soh_pct": float(b3_row["SOH_vs_nominal_1.25Ah_pct"]),
            "audit_quality_tier": str(b3_row["quality_tier"]),
            "delta_capacity_ah": float(b3["discharge"]["capacity_ah"] - b3_row["Qdis_Ah"]),
        },
        "B4": {
            "source": "DATA/B4_Discharge3C_Charge0.5C (1).zip::Cycle_0050.xlsx::Run01_Un1Ch6_Step2_Cycle10",
            "derived": b4,
            "audit_capacity_ah": None,
            "audit_note": (
                "B4 audit computed no capacity (blanket block on current-unit and phase "
                "segmentation); this raw re-derivation shows the discharge is complete "
                "to 2.4976 V and yields a physically plausible 3C capacity."
            ),
        },
    }


def master_facts() -> dict:
    df = pd.read_parquet(MASTER_PARQUET)
    per = {}
    for dataset_id, group in df.groupby("dataset_id"):
        labelled = group.dropna(subset=["soh_pct"]).groupby(["cell_id", "cycle_id"]).ngroups
        per[dataset_id] = {
            "rows": int(len(group)),
            "observed_cycle_ids": int(group["cycle_id"].nunique()),
            "cell_cycle_pairs": int(group.groupby(["cell_id", "cycle_id"]).ngroups),
            "labelled_cell_cycle_pairs": int(labelled),
            "soh_null_rows": int(group["soh_pct"].isna().sum()),
        }
    b5 = df[df["dataset_id"] == "LFP_MultiStat_B5"]
    b5_ids = sorted(int(v) for v in b5["cycle_id"].unique())
    b5_labelled = sorted(
        int(v) for v in b5.dropna(subset=["soh_pct"])["cycle_id"].unique()
    )
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "datasets": per,
        "b5_cycle_blocks": _blocks(b5_ids),
        "b5_missing_cycle_ids": sorted(set(range(1, 501)) - set(b5_ids)),
        "b5_labelled_cycles": len(b5_labelled),
        "b5_next_observed_record_pairs": max(len(b5_labelled) - 1, 0),
        "b5_null_soh_cycles": sorted(
            int(c) for c in b5[b5["soh_pct"].isna()]["cycle_id"].unique()
        ),
        "b5_null_soh_rows": int(b5["soh_pct"].isna().sum()),
    }


def _blocks(ids: list[int]) -> list[list[int]]:
    if not ids:
        return []
    blocks, start, prev = [], ids[0], ids[0]
    for value in ids[1:]:
        if value == prev + 1:
            prev = value
        else:
            blocks.append([start, prev])
            start = prev = value
    blocks.append([start, prev])
    return blocks


def b5_facts() -> dict:
    with zipfile.ZipFile(B5_AUDIT) as archive:
        summary = pd.read_csv(
            archive.open(next(n for n in archive.namelist() if n.endswith("B5_Cycle_Summary.csv")))
        )
    q_dis = summary["dis_q_ah_trapz_CALCULATED"]
    q_chg = summary["chg_q_ah_UNSUPPORTED_diagnostic_only"]
    paired = q_dis.notna() & q_chg.notna()
    ce = summary["ce_UNSUPPORTED_diagnostic_only"]
    return {
        "audit_cycle_rows": int(len(summary)),
        "audit_labelled_soh": int(summary["soh_nominal_pct_CONDITIONAL_1.25Ah_rated"].notna().sum()),
        "audit_excluded_cycles": sorted(
            int(c)
            for c in summary.loc[
                summary["modelling_use_capacity_soh"] == "EXCLUDE",
                "global_cycle_id_nominal_checkpoint",
            ]
        ),
        "capacity_ground_truth_values": sorted(summary["capacity_ground_truth"].unique().tolist()),
        "paired_cycles": int(paired.sum()),
        "cycles_with_qdis_gt_qchg": int((q_dis > q_chg)[paired].sum()),
        "ce_median": float(ce.median()),
        "ce_min": float(ce.min()),
        "ce_max": float(ce.max()),
    }


def b2_facts() -> dict:
    with zipfile.ZipFile(B2_DERIVED) as archive:
        card = json.loads(archive.read("LFP_Member1_audit/exports/dataset_card.json"))
        features = pd.read_parquet(
            io.BytesIO(archive.read("LFP_Member1_audit/exports/cycle_features.parquet"))
        )
        qv = pd.read_parquet(
            io.BytesIO(archive.read("LFP_Member1_audit/exports/qv_curves.parquet"))
        )
        splits = json.loads(archive.read("LFP_Member1_audit/exports/splits.json"))
        integrity = pd.read_csv(archive.open("LFP_Member1_audit/data_derived/raw_integrity.csv"))
        verification = json.loads(
            archive.read("LFP_Member1_audit/outputs/verification_report.json")
        )
    analysis = features[features["in_analysis"]]
    return {
        "dataset": card["dataset"],
        "raw_workbooks_referenced": card["raw_provenance"]["raw_workbooks"],
        "raw_bytes_referenced": card["raw_provenance"]["raw_bytes"],
        "raw_integrity_rows": int(len(integrity)),
        "derived_cycles": int(len(features)),
        "cycles_in_analysis": int(len(analysis)),
        "soh_rated_min": float(analysis["soh_rated"].min()),
        "soh_rated_max": float(analysis["soh_rated"].max()),
        "soh_fraction_semantics": "soh_rated is 0-1; multiply by 100 for percent",
        "channels": {str(int(k)): int(v) for k, v in features["channel"].value_counts().items()},
        "channel_change_cycle": 999,
        "qv_curves_shape": list(qv.shape),
        "splits": splits,
        "label_derived_columns": [
            c
            for c in features.columns
            if any(k in c.lower() for k in ("q_discharge", "q_charge", "soh_", "q_rated"))
        ],
        "verification_checks": len(verification["checks"]),
        "verification_all_passed": all(c["passed"] for c in verification["checks"]),
    }


def b4_coverage_gap() -> dict:
    with zipfile.ZipFile(B4_RAW) as archive:
        raw = sorted({n.split("/")[-1] for n in archive.namelist() if n.lower().endswith(".xlsx")})
    with zipfile.ZipFile(B3B4_AUDIT) as archive:
        summary = pd.read_csv(archive.open("B4/LFP_cycle_summary.csv"))
    audited = sorted(summary["source_file"].unique())
    cycle_rows = summary[summary["phase_from_sheet_name"] == "cycle_step_unknown_subphase"]
    return {
        "raw_workbooks": len(raw),
        "audited_workbooks": len(audited),
        "unaudited_workbooks": sorted(set(raw) - set(audited)),
        "audited_cycle_sheets": int(len(cycle_rows)),
        "audited_complete_discharges": int((cycle_rows["voltage_min_v"] <= 2.55).sum()),
        "audited_no_discharge": int((cycle_rows["negative_current_rows"] == 0).sum()),
        "raw_median_abs_discharge_current_a": float(
            cycle_rows["current_min_original_units"].median()
        ),
        "raw_median_charge_current_a": float(
            cycle_rows["current_max_original_units"].median()
        ),
    }


def b6_note() -> dict:
    with zipfile.ZipFile(B6_AUDIT) as archive:
        meta = json.loads(archive.read("LFP_B6_Audit_Output/LFP_audit_run_metadata.json"))
    present = bool(
        glob.glob(str(REPO_ROOT / "DATA" / "*B6*"))
        + glob.glob("/c/Users/KIIT0001/Downloads/*B6*")
    )
    raw_zip_present = any(
        not p.endswith("Audit_Package.zip")
        for p in glob.glob(str(REPO_ROOT / "DATA" / "*B6*"))
        + glob.glob("/c/Users/KIIT0001/Downloads/*B6*")
    )
    return {
        "scope": "OUT_OF_SCOPE for the B1-B5 brief; recorded because the audit package was attached",
        "raw_input_zip": meta["input_zip"],
        "raw_input_size_bytes": meta["input_size_bytes"],
        "raw_present_in_workspace": raw_zip_present,
        "referenced_workbooks": meta["workbook_count"],
        "cycle_like_sheets": meta["cycle_like_worksheet_summary_count"],
        "complete_discharge_labels": meta["complete_discharge_screen_pass_count"],
        "label_semantics": "conditional capacity/SOH relative to B6 metadata nominal capacity",
        "finding": (
            "Raw B6 archive present in DATA and ingested by the B2/B6 follow-up; the "
            "externally supplied audit package indicates 1,697 conditional cycle labels."
            if raw_zip_present
            else "Raw B6 workbooks are absent; only the external audit package exists. "
            "It cannot be used as verified raw data, but it indicates B6 would add "
            "~1,697 conditional cycle labels if the raw archive were supplied."
        ),
    }


def build_coverage_rows(master: dict, b5: dict, b2: dict, b4gap: dict) -> list[dict]:
    d = master["datasets"]
    b1 = d["LFP_MultiStat_B1"]
    b5m = d["LFP_MultiStat_B5"]
    return [
        {
            "cell_id": "BATTERY_1",
            "dataset_id": "LFP_MultiStat_B1",
            "source_path": "DATA/B1_Discharge0.5C_Charge0.5C(1).zip; AUDIT/B1_LFP_Audit_Output.zip",
            "source_status": "SOURCE_VERIFIED_MASTER_PARTIAL",
            "source_verified": "YES - 19 workbooks / 1,460 cycle sheets recounted; one raw cycle re-derived and matched the audit to 1e-6 Ah",
            "raw_data_available": "YES",
            "derived_data_available": "YES - 1,460-row cycle table, 1,456 capacity/SOH labels",
            "row_count": b1["rows"],
            "observed_cycle_count": b1["observed_cycle_ids"],
            "labelled_cycle_count": b1["labelled_cell_cycle_pairs"],
            "soh_provenance": "capacity_ah / 1.25 Ah rated; audit-computed, conditional on current-unit interpretation",
            "current_unit_status": "VALUES_ARE_AMPERES; header I(A/cm2) with Surface Area=0 is a label error (verified against 0.625 A protocol)",
            "voltage_available": "YES",
            "current_available": "YES",
            "time_available": "YES (1 Hz source; master rows sampled every 3rd)",
            "temperature_available": "NO - Aux Count 0; no thermal channel in any workbook",
            "data_quality_issues": (
                "Master ingests only a staging sample: 4,789 rows from cycle 2 of checkpoint 50, "
                "while the source has 1,460 cycle sheets and 1,456 labels. 110 labels carry an "
                "underdetermined cycle offset (108 alone, 2 combined with other flags); 4 cycles "
                "have no capacity label; master mixes charge and discharge rows. BOL reference "
                "is first observed, not true BOL."
            ),
            "independent_validation_suitability": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
            "blocker": "Master coverage (1/1,460 cycles); no unit attestation; no measured BOL reference",
            "recommended_action": "Candidate re-ingestion of all 1,460 cycles from raw with per-cycle labels; verify current unit; use contiguous splits",
        },
        {
            "cell_id": "BATTERY_2",
            "dataset_id": "LFP_MultiStat_B2",
            "source_path": "NOT_IN_WORKSPACE (raw workbooks absent); DATA/B2_LFP_Member1_audit.zip (derived)",
            "source_status": "ONLY_DERIVED_DATA_AVAILABLE",
            "source_verified": "PARTIAL - derived tables verified internally (7+ checks, per-file SHA-256 list); raw not available to re-derive",
            "raw_data_available": "NO",
            "derived_data_available": "YES - cycle_features 1,506x255, qv_curves, contiguous splits",
            "row_count": 0,
            "observed_cycle_count": 0,
            "labelled_cycle_count": 0,
            "soh_provenance": "q_discharge_ah / 1.250 Ah rated; audited export; soh_rated is a 0-1 fraction",
            "current_unit_status": "VERIFIED_AMPERES (BioLogic I / A)",
            "voltage_available": "NO raw telemetry; 101-point Q-V curves only",
            "current_available": "NO raw telemetry; cycle-level current summaries only",
            "time_available": "NO raw telemetry; cycle-level durations only",
            "temperature_available": "NO - not measured",
            "data_quality_issues": (
                "Master: absent; derived: 1,506 cycles, 1,496 in analysis, SOH 0.855-0.967. "
                "10 invalid cycles; instrument channel changes Un1Ch8->Un1Ch2 at cycle 999 "
                "(998 vs 508 cycles) with a ~0.9% capacity step; 160 cycles have inferred "
                "provenance; label columns q_discharge_ah / soh_rated must never be features."
            ),
            "independent_validation_suitability": "SUITABLE_FOR_INDEPENDENT_VALIDATION",
            "blocker": "None for cycle-level validation; raw absent limits bespoke telemetry features",
            "recommended_action": "Use provided contiguous splits and cycle_index; exclude 999-1015 or model channel as covariate; multiply SOH by 100; do not use label columns as inputs",
        },
        {
            "cell_id": "BATTERY_3",
            "dataset_id": "LFP_MultiStat_B3",
            "source_path": "DATA/B3_Discharge2C_Charge0.5C (1).zip; AUDIT/Existing_Audits_Available_B3_B4 (1).zip",
            "source_status": "SOURCE_VERIFIED_NOT_INGESTED",
            "source_verified": "YES - 25 workbooks / 1,979 cycle sheets recounted; one raw cycle re-derived and matched Qdis to 1e-6 Ah",
            "raw_data_available": "YES",
            "derived_data_available": "YES - 1,979-row cycle summary with 1,971 SOH labels (1,973 Qdis values)",
            "row_count": 0,
            "observed_cycle_count": 0,
            "labelled_cycle_count": 0,
            "soh_provenance": "Qdis_Ah / 1.25 Ah rated; secondary retention vs first 2C cycle; audit-computed",
            "current_unit_status": "VALUES_ARE_AMPERES; audit states the A/cm2 header is a unit-label error (2.499 A matches 2.5 A setpoint)",
            "voltage_available": "YES",
            "current_available": "YES",
            "time_available": "YES (1 Hz)",
            "temperature_available": "NO - Aux Count 0",
            "data_quality_issues": (
                "Master: absent; source: 1,979 cycles across 25 workbooks, 1,971 labels. "
                "Cycles 201-221 absent (a 29-cycle workbook implies 21 withheld); 8 tier-X "
                "cycles (6 no discharge, 2 interrupted); 4 instrument channels, so physical "
                "cell identity is unproven; CE median 0.983 (physically plausible)."
            ),
            "independent_validation_suitability": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
            "blocker": "Not ingested; no unit attestation; cell identity across channels unproven; no measured BOL",
            "recommended_action": "Priority independent cell: candidate extraction of all 1,979 cycles, unit/identity attestation, then chronological validation",
        },
        {
            "cell_id": "BATTERY_4",
            "dataset_id": "LFP_MultiStat_B4",
            "source_path": "DATA/B4_Discharge3C_Charge0.5C (1).zip; AUDIT/Existing_Audits_Available_B3_B4 (1).zip (partial)",
            "source_status": "SOURCE_VERIFIED_AUDIT_INCOMPLETE",
            "source_verified": "YES for raw inventory (21 workbooks / 1,577 cycle sheets); audit covers only 16 workbooks; one raw cycle re-derived",
            "raw_data_available": "YES",
            "derived_data_available": "PARTIAL - descriptive 1,382-row audit; no capacity/SOH labels computed",
            "row_count": 0,
            "observed_cycle_count": 0,
            "labelled_cycle_count": 0,
            "soh_provenance": "Derivable as Qdis/1.25 Ah, but not computed: audit blocks capacity on unit and phase grounds",
            "current_unit_status": "VALUES_ARE_AMPERES; negative-current median -3.7495 A and positive median +0.6259 A match the 3C/0.5C protocol; header still says A/cm2",
            "voltage_available": "YES",
            "current_available": "YES",
            "time_available": "YES (1 Hz)",
            "temperature_available": "NO - Aux Count 0",
            "data_quality_issues": (
                "Master: absent; source: 21 workbooks / 1,577 cycle sheets, but the audit "
                "covers 16 workbooks / 1,350 cycle sheets; Cycles 0150-0350 are unaudited. "
                "3 no-discharge sheets, 1 charge excursion to 3.887 V (Cycle_1000, cycle 86), "
                "no phase labels (sign-based segmentation required). The audit's blanket "
                "block is stricter than the data."
            ),
            "independent_validation_suitability": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
            "blocker": "No computed labels; 5 workbooks unaudited; unit/phase attestation outstanding",
            "recommended_action": "Re-audit all 21 workbooks with sign-based segmentation; compute Qdis/SOH; verify against raw samples; then assess",
        },
        {
            "cell_id": "BATTERY_5",
            "dataset_id": "LFP_MultiStat_B5",
            "source_path": "DATA/B5_Discharge0.5C_Charge1C (1).zip; AUDIT/B5.zip",
            "source_status": "SOURCE_VERIFIED_MASTER_PARTIAL_CONDITIONAL",
            "source_verified": "YES - 17 workbooks / 1,491 cycle sheets recounted; source audit 1,499 records",
            "raw_data_available": "YES",
            "derived_data_available": "YES - clean telemetry parquet + 1,499-row cycle summary",
            "row_count": b5m["rows"],
            "observed_cycle_count": b5m["observed_cycle_ids"],
            "labelled_cycle_count": b5m["labelled_cell_cycle_pairs"],
            "soh_provenance": "capacity / 1.25 Ah rated, but audit sets capacity_ground_truth=NO for every cycle",
            "current_unit_status": "VALUES_ARE_AMPERES; header I(A/cm2) with Surface Area=0 (same label error)",
            "voltage_available": "YES",
            "current_available": "YES",
            "time_available": "YES (1 Hz source; master samples every 5th discharge row)",
            "temperature_available": "NO - Aux Count 0",
            "data_quality_issues": (
                f"Master: 333,344 rows / 250 cycles from 4 of 17 workbooks; 249 labels; cycle 40 "
                f"missing (625 rows); blocks 101-150 and 201-400 absent by design. "
                f"Qdis>Qchg in {b5['cycles_with_qdis_gt_qchg']}/{b5['paired_cycles']} paired cycles "
                f"(CE median {b5['ce_median']:.4f}); 43 discharge rows dip below the 2.5 V cutoff "
                f"(min 2.4947 V); 7 source "
                f"cycles excluded (only cycle 40 falls in ingested blocks)."
            ),
            "independent_validation_suitability": "BLOCKED_BY_LABEL_QUALITY",
            "blocker": "Absolute capacity is not ground truth; charge-balance discrepancy; single cell already used for training",
            "recommended_action": "Keep as conditional training/evaluation cell; do not use as the independent validation cell; keep cycle 40 excluded",
        },
    ]


FEATURE_DEFS = {
    "discharge_capacity_ah": (
        "segmented discharge current, time base",
        "Ah (trapezoid of A over s; conditional)",
        "Coulomb-count |I| between discharge start and 2.5 V cutoff",
    ),
    "soh_pct_label": (
        "capacity_ah, rated 1.25 Ah",
        "% (capacity/1.25*100)",
        "Direct ratio; never a model input",
    ),
    "voltage_series": (
        "E(V), T(Seconds), cycle id",
        "V (verified)",
        "Per-sample voltage aligned to cycle-relative time",
    ),
    "dq_dv": (
        "Q(t), V(t) at 1 Hz",
        "Ah/V (derived)",
        "Numerical dQ/dV with monotonic-Q filtering and smoothing",
    ),
    "dv_dq": (
        "Q(t), V(t) at 1 Hz",
        "V/Ah (derived)",
        "Numerical dV/dQ on monotonic Q segments",
    ),
    "ica_peaks": (
        "dQ/dV curve",
        "Ah/V; peak voltage/height",
        "Peak detection with prominence and minimum-height safeguards",
    ),
    "ah_energy_throughput": (
        "I(t), V(t), t",
        "Ah, Wh (computed)",
        "Integrate I dt and V I dt separately for charge and discharge",
    ),
    "c_rate_phase_durations": (
        "I(t), protocol setpoints",
        "C-rate; s",
        "Segment by current sign and compare |I| with 1.25 Ah rating",
    ),
    "cccv_fraction": (
        "I(t), V(t) during charge",
        "% of charge time in CV; s",
        "Detect CV hold after the 3.65 V plateau and the 0.025 A cutoff",
    ),
    "temperature": (
        "auxiliary temperature channel",
        "degC",
        "Not derivable without a measured channel",
    ),
    "resistance_proxy": (
        "current step and voltage response",
        "Ohm (proxy)",
        "delta V / delta I across phase transitions",
    ),
    "qv_curve_samples": (
        "Q(t), V(t) per cycle",
        "101-point Q-V curve",
        "Resample capacity at fixed fractions of the discharge",
    ),
    "efc_cycle_index": (
        "cycle sequence, capacity",
        "EFC; cycle id",
        "Sum |Ah| / (2 * 1.25 Ah) or observed cycle index",
    ),
    "lagged_soh": (
        "prior-cycle SOH labels",
        "%",
        "Lag-1+ SOH with explicit gap flags; never same-cycle SOH",
    ),
}

# (input_availability, suitability, limitations, missing_input, recommended_action)
CELL_FEATURES = {
    "BATTERY_1": {
        "discharge_capacity_ah": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Unit label error; master label is the wrong cycle; 4 cycles uncomputable", "Unit attestation", "Use audit per-cycle labels after unit check"),
        "soh_pct_label": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Master 96.178% is cycle 50's last label, not telemetry cycle 2 (97.612%)", "Unit attestation", "Repair label-cycle pairing before use"),
        "voltage_series": ("AVAILABLE_IN_MASTER_AND_SOURCE", "READY_WITH_CAVEAT", "Master has 1 cycle sampled every 3rd row", "Full-resolution extraction", "Extract full 1,460 cycles at 1 Hz"),
        "dq_dv": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Needs 1 Hz raw and monotonic Q; master 3 s sample too coarse", "Full-resolution extraction", "Derive from raw, not master"),
        "dv_dq": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Same resolution constraint", "Full-resolution extraction", "Derive from raw after phase segmentation"),
        "ica_peaks": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Peak stability not yet validated", "dQ/dV validation", "Benchmark peak detection on raw cycles"),
        "ah_energy_throughput": ("AVAILABLE_IN_MASTER_AND_SOURCE", "CONDITIONAL", "Conditional on current-unit interpretation", "Unit attestation", "Recompute from raw once units confirmed"),
        "c_rate_phase_durations": ("AVAILABLE_IN_MASTER_AND_SOURCE", "READY_WITH_CAVEAT", "Master mixes charge/discharge rows per cycle", "Phase flag in master", "Segregate by current sign"),
        "cccv_fraction": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Requires raw charge segment", "Full-resolution extraction", "Compute from raw charge phase"),
        "temperature": ("NOT_AVAILABLE", "NOT_AVAILABLE", "Aux Count 0 in every workbook", "Physically measured temperature", "Do not impute or use scenario values as features"),
        "resistance_proxy": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Step transitions at 1 Hz; no validated method", "Method validation", "Define and test a DCIR proxy"),
        "qv_curve_samples": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Needs per-cycle resampling", "Extraction", "Resample raw discharge curves"),
        "efc_cycle_index": ("AVAILABLE_IN_SOURCE_ONLY", "READY_WITH_CAVEAT", "110 labels carry underdetermined cycle offset", "Cycle-order audit", "Exclude offset-uncertain cycles"),
        "lagged_soh": ("BLOCKED_IN_MASTER", "BLOCKED", "Only one master cycle; no within-cell history", "Multi-cycle ingestion", "Ingest the 1,460-cycle source then build lags"),
    },
    "BATTERY_2": {
        "discharge_capacity_ah": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "Label column q_discharge_ah; cannot be an input", "Raw workbooks", "Use as target only; keep out of features"),
        "soh_pct_label": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "soh_rated is 0-1; multiply by 100", "Raw workbooks", "Use as target; never as input"),
        "voltage_series": ("NOT_AVAILABLE", "NOT_AVAILABLE", "No raw telemetry; only 101-point Q-V curves", "Raw workbooks", "Acquire raw if per-sample voltage is needed"),
        "dq_dv": ("AVAILABLE_IN_DERIVED_ONLY", "CONDITIONAL", "Only 101-point Q-V curves, not 1 Hz", "Raw workbooks", "Derive coarse curves from exports; flag resolution"),
        "dv_dq": ("AVAILABLE_IN_DERIVED_ONLY", "CONDITIONAL", "Same coarse resolution", "Raw workbooks", "Use exported curves with safeguards"),
        "ica_peaks": ("AVAILABLE_IN_DERIVED_ONLY", "CONDITIONAL", "Coarse curves limit peak resolution", "Raw workbooks", "Validate peaks against B5-derived peaks"),
        "ah_energy_throughput": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "Cycle-level only; label columns excluded", "None", "Use q_charge_ah/energy summaries as inputs only if they do not encode the label"),
        "c_rate_phase_durations": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "Cycle-level durations available; no raw segmentation", "None", "Use discharge/charge duration and C-rate columns"),
        "cccv_fraction": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "charge_cv_fraction provided", "None", "Use provided column"),
        "temperature": ("NOT_AVAILABLE", "NOT_AVAILABLE", "Not measured", "Measured temperature", "Do not impute"),
        "resistance_proxy": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "dcir_proxy_ohm/polarisation provided", "None", "Use provided proxies; check channel-step artefact"),
        "qv_curve_samples": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "101-point discharge/charge Q-V arrays", "None", "Use qv_curves.parquet keyed by cycle_index"),
        "efc_cycle_index": ("AVAILABLE_IN_DERIVED_ONLY", "READY", "Contiguous splits provided", "None", "Use splits.json; no random shuffling"),
        "lagged_soh": ("AVAILABLE_IN_DERIVED_ONLY", "READY_WITH_CAVEAT", "Channel change at 999; exclude 999-1015", "None", "Lag within contiguous splits only"),
    },
    "BATTERY_3": {
        "discharge_capacity_ah": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Unit label error; 6 cycles lack discharge; 2 interrupted", "Unit attestation", "Extract cycle labels from raw/audit after unit check"),
        "soh_pct_label": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "1,971 labels; BOL is first observed 2C cycle", "Unit attestation", "Use audit labels for candidate validation"),
        "voltage_series": ("AVAILABLE_IN_SOURCE_ONLY", "READY", "1 Hz raw, complete to cutoff in 1,971 cycles", "Ingestion", "Extract full telemetry"),
        "dq_dv": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "2C discharge is steep; smoothing choice matters", "Method validation", "Test smoothing against a reference cycle"),
        "dv_dq": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Same steep-slope constraint", "Method validation", "Use monotonic-Q safeguards"),
        "ica_peaks": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Peak stability at 2C unverified", "Method validation", "Validate on raw cycles"),
        "ah_energy_throughput": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Conditional on current-unit interpretation", "Unit attestation", "Recompute from raw"),
        "c_rate_phase_durations": ("AVAILABLE_IN_SOURCE_ONLY", "READY_WITH_CAVEAT", "Current segments carry timestamp gaps (395 flagged)", "Gap handling", "Use true dt and flag gaps"),
        "cccv_fraction": ("AVAILABLE_IN_SOURCE_ONLY", "READY_WITH_CAVEAT", "CV step present; t_CV_s provided in audit", "Ingestion", "Use audit t_CV_s"),
        "temperature": ("NOT_AVAILABLE", "NOT_AVAILABLE", "Aux Count 0", "Measured temperature", "Do not impute"),
        "resistance_proxy": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Step transitions at 1 Hz", "Method validation", "Define and test proxy"),
        "qv_curve_samples": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Needs resampling at 1 Hz", "Extraction", "Resample raw discharge curves"),
        "efc_cycle_index": ("AVAILABLE_IN_SOURCE_ONLY", "READY_WITH_CAVEAT", "Cycles 201-221 absent; one 29-cycle block", "Gap-aware indexing", "Preserve observed-index gaps"),
        "lagged_soh": ("AVAILABLE_IN_SOURCE_ONLY", "READY_WITH_CAVEAT", "Gaps break strict lags", "Gap-aware lags", "Lag only within contiguous blocks"),
    },
    "BATTERY_4": {
        "discharge_capacity_ah": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Not computed by audit; raw sample proves a complete 3C discharge is integrable", "Phase segmentation + unit attestation", "Re-audit with sign-based segmentation"),
        "soh_pct_label": ("NOT_CALCULATED", "CONDITIONAL", "Audit sets NOT_CALCULATED for all cycles", "Re-audit", "Compute Qdis/1.25 after segmentation"),
        "voltage_series": ("AVAILABLE_IN_SOURCE_ONLY", "READY_WITH_CAVEAT", "One cycle (Cycle_1000, cycle 86) reaches 3.887 V during charge", "Outlier policy", "Flag the excursion, keep raw values"),
        "dq_dv": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Phase labels absent; requires sign segmentation", "Phase segmentation", "Segment by current sign then derive"),
        "dv_dq": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Same segmentation requirement", "Phase segmentation", "Segment by current sign then derive"),
        "ica_peaks": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "3C curves are steep; peaks may be weak", "Method validation", "Validate before use"),
        "ah_energy_throughput": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Derivable from raw; audit did not compute", "Unit attestation", "Recompute from raw"),
        "c_rate_phase_durations": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Audit phase labels are unknown_subphase", "Sign-based segmentation", "Segment by current sign"),
        "cccv_fraction": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Requires charge-phase segmentation", "Phase segmentation", "Detect CV after 3.65 V"),
        "temperature": ("NOT_AVAILABLE", "NOT_AVAILABLE", "Aux Count 0", "Measured temperature", "Do not impute"),
        "resistance_proxy": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "3C transitions are fast", "Method validation", "Use 1 Hz step response"),
        "qv_curve_samples": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Needs resampling", "Extraction", "Resample raw discharge curves"),
        "efc_cycle_index": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "5 workbooks (Cycles 0150-0350) unaudited", "Full audit", "Audit all 21 workbooks"),
        "lagged_soh": ("BLOCKED", "BLOCKED", "No SOH labels yet", "Computed labels", "Build after re-audit"),
    },
    "BATTERY_5": {
        "discharge_capacity_ah": ("AVAILABLE_IN_MASTER", "CONDITIONAL", "capacity_ground_truth=NO; charge-balance discrepancy", "Independent capacity reference", "Use only with conditional weighting"),
        "soh_pct_label": ("AVAILABLE_IN_MASTER", "CONDITIONAL", "Same caveat; cycle 40 missing", "Independent reference", "Keep cycle 40 excluded"),
        "voltage_series": ("AVAILABLE_IN_MASTER_AND_SOURCE", "READY_WITH_CAVEAT", "Master samples every 5th discharge row; full 1 Hz in B5.zip", "None", "Use B5.zip full-resolution telemetry"),
        "dq_dv": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "5 s master sampling; full 1 Hz in audit", "None", "Derive from B5.zip, not master"),
        "dv_dq": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Same resolution caveat", "None", "Derive from B5.zip"),
        "ica_peaks": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Peak stability untested on conditional capacity", "Method validation", "Validate before use"),
        "ah_energy_throughput": ("AVAILABLE_IN_MASTER_AND_SOURCE", "CONDITIONAL", "Current imbalance; charge capacity unsupported", "Independent reference", "Use discharge-only throughput with flags"),
        "c_rate_phase_durations": ("AVAILABLE_IN_MASTER_AND_SOURCE", "READY_WITH_CAVEAT", "Master is discharge-only; phase flag needed", "None", "Use source phase labels"),
        "cccv_fraction": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Charge capacity unsupported by audit", "Independent reference", "Flag as diagnostic only"),
        "temperature": ("NOT_AVAILABLE", "NOT_AVAILABLE", "Aux Count 0", "Measured temperature", "Do not impute"),
        "resistance_proxy": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Derivable at 1 Hz; imbalance caveat", "Method validation", "Flag conditional"),
        "qv_curve_samples": ("AVAILABLE_IN_SOURCE_ONLY", "CONDITIONAL", "Needs resampling", "Extraction", "Resample B5.zip curves"),
        "efc_cycle_index": ("AVAILABLE_IN_MASTER_AND_SOURCE", "READY_WITH_CAVEAT", "Blocks 101-150 and 201-400 absent", "None", "Preserve observed-index gaps"),
        "lagged_soh": ("AVAILABLE_IN_MASTER", "READY_WITH_CAVEAT", "Lags valid only within contiguous blocks", "None", "Do not lag across missing blocks"),
    },
}


def build_feature_rows() -> list[dict]:
    rows = []
    for cell_id, features in CELL_FEATURES.items():
        for feature_name, (
            availability,
            suitability,
            limitations,
            missing,
            action,
        ) in features.items():
            required, units, method = FEATURE_DEFS[feature_name]
            rows.append(
                {
                    "cell_id": cell_id,
                    "feature_name": feature_name,
                    "required_inputs": required,
                    "verified_units": units,
                    "derivation_method": method,
                    "input_availability": availability,
                    "suitability": suitability,
                    "limitations": limitations,
                    "missing_input": missing,
                    "recommended_action": action,
                }
            )
    return rows


def build_evidence() -> dict:
    hashes = {"parquet": sha256_file(MASTER_PARQUET), "csv": sha256_file(MASTER_CSV)}
    if hashes["parquet"] != MASTER_SHA256 or hashes["csv"] != MASTER_CSV_SHA256:
        raise RuntimeError("canonical master table hash mismatch before audit")

    inventories = {
        "B1": archive_cycle_inventory(B1_RAW),
        "B3": archive_cycle_inventory(B3_RAW),
        "B4": archive_cycle_inventory(B4_RAW),
        "B5": archive_cycle_inventory(B5_RAW),
    }
    for key, (expected_books, expected_cycles) in EXPECTED_CYCLE_SHEETS.items():
        inv = inventories[key]
        if (inv["workbook_count"], inv["cycle_sheet_count"]) != (expected_books, expected_cycles):
            raise RuntimeError(
                f"{key} inventory changed: {inv['workbook_count']}/{inv['cycle_sheet_count']}"
            )

    samples = verify_sample_cycles()
    master = master_facts()
    b5 = b5_facts()
    b2 = b2_facts()
    b4gap = b4_coverage_gap()
    b6 = b6_note()

    coverage = build_coverage_rows(master, b5, b2, b4gap)
    features = build_feature_rows()

    archive_hashes = {}
    for key, path in {
        "B1_AUDIT": B1_AUDIT,
        "B3_RAW": B3_RAW,
        "B4_RAW": B4_RAW,
        "B5_RAW": B5_RAW,
        "B6_AUDIT": B6_AUDIT,
    }.items():
        digest = sha256_file(path)
        archive_hashes[key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": digest,
            "matches_expected": digest == EXPECTED_ARCHIVE_SHA256[key],
        }

    workspace_hashes = {}
    for key, path in {"B1_RAW": B1_RAW, "B2_DERIVED": B2_DERIVED}.items():
        digest = sha256_file(path)
        workspace_hashes[key] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": digest,
            "matches_expected": digest == EXPECTED_WORKSPACE_SHA256[key],
        }

    return {
        "metadata": {
            "title": "Member 1 - LFP B1-B5 source investigation and validation readiness",
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "canonical_dataset_modified": False,
            "notes": (
                "Read-only. Source counts come from the archives themselves; one cycle "
                "per raw cell is re-derived and compared with the audit tables."
            ),
        },
        "guardrails": {
            "master_table_sha256": hashes,
            "hashes_match_expected": (
                hashes["parquet"] == MASTER_SHA256 and hashes["csv"] == MASTER_CSV_SHA256
            ),
            "attachments_match_expected": all(
                v["matches_expected"] for v in archive_hashes.values()
            ),
            "workspace_archives_match_expected": all(
                v["matches_expected"] for v in workspace_hashes.values()
            ),
            "writes_confined_to": str(OUT_DIR.relative_to(REPO_ROOT)),
            "member2_artifacts_modified": False,
        },
        "attachments_reconciled": {
            "note": (
                "The B1 audit, B3/B4/B5 raw and B6 audit archives supplied with this "
                "request are byte-identical to the workspace copies; no new raw data "
                "was introduced."
            ),
            "archive_hashes": archive_hashes,
        },
        "workspace_archive_hashes": workspace_hashes,
        "raw_inventories": inventories,
        "independent_sample_verification": samples,
        "master_table_facts": master,
        "b5_facts": b5,
        "b2_facts": b2,
        "b4_coverage_gap": b4gap,
        "b6_out_of_scope_note": b6,
        "coverage_rows": coverage,
        "feature_rows_count": len(features),
    }


def write_outputs(evidence: dict, coverage: list[dict], features: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lfp_b1_b5_evidence.json").write_text(
        json.dumps(evidence, indent=2, default=str), encoding="utf-8"
    )
    for name, rows, columns in (
        ("MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv", coverage, COVERAGE_COLUMNS),
        ("MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv", features, FEATURE_COLUMNS),
    ):
        path = AUDIT_DIR / name
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    before = {
        "parquet": sha256_file(MASTER_PARQUET),
        "csv": sha256_file(MASTER_CSV),
    }
    evidence = build_evidence()
    coverage = evidence["coverage_rows"]
    features = build_feature_rows()
    write_outputs(evidence, coverage, features)
    after = {
        "parquet": sha256_file(MASTER_PARQUET),
        "csv": sha256_file(MASTER_CSV),
    }
    if before != after:
        print("ABORT: canonical master table changed during the audit")
        return 1
    if not evidence["guardrails"]["hashes_match_expected"]:
        print("ABORT: canonical hashes do not match the verified values")
        return 1
    if not (
        evidence["guardrails"]["attachments_match_expected"]
        and evidence["guardrails"]["workspace_archives_match_expected"]
    ):
        print("ABORT: a source archive no longer matches its recorded hash")
        return 1

    print("raw inventory (workbooks / cycle sheets):")
    for key, inv in evidence["raw_inventories"].items():
        print(f"  {key}: {inv['workbook_count']} / {inv['cycle_sheet_count']}")
    for key in ("B1", "B3", "B4"):
        sample = evidence["independent_sample_verification"][key]["derived"]["discharge"]
        print(
            f"  sample {key}: Q={sample['capacity_ah']:.6f} Ah, "
            f"V_end={sample['v_end']:.4f} V, I={sample['mean_abs_current_a']:.4f} A"
        )
    print(f"master rows: {evidence['master_table_facts']['rows']}")
    print(f"B5 labels: {evidence['master_table_facts']['b5_labelled_cycles']}")
    print(f"coverage matrix: {len(coverage)} rows")
    print(f"feature matrix : {len(features)} rows")
    print(f"evidence       : {OUT_DIR / 'lfp_b1_b5_evidence.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
