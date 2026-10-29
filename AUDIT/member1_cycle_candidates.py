"""
Member 1 - cycle-level candidate exports and B2 leakage/channel manifests.

Writes (AUDIT/candidates/):
    B1_candidate_cycle_labels.csv
    B3_candidate_cycle_labels.csv
    B4_candidate_cycle_labels.csv
    B5_candidate_cycle_labels.csv
    B2_channel_transition_mask.csv
    B2_channel_transition_summary.json
and (AUDIT/):
    MEMBER1_FEATURE_TARGET_MANIFEST.csv
    MEMBER1_TARGET_DEFINITIONS.csv

Read-only on raw archives and existing audit packages.

Run:
    python AUDIT/member1_cycle_candidates.py
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
DATA = REPO_ROOT / "DATA"
OUT = AUDIT_DIR / "candidates"
OUT.mkdir(exist_ok=True)


def b1_export():
    with zipfile.ZipFile(AUDIT_DIR / "B1_LFP_Audit_Output.zip") as z:
        df = pd.read_csv(io.BytesIO(z.read("B1_LFP_Audit_Output/LFP_capacity_soh_feasibility.csv")))
    keys = df["source_file"].astype(str) + "::" + df["worksheet"].astype(str)
    assert keys.is_unique, "B1 (source_file, worksheet) must be unique"
    out = pd.DataFrame(
        {
            "dataset_id": "LFP_MultiStat_B1",
            "cell_id": "BATTERY_1",
            "source_file": df["source_file"],
            "worksheet": df["worksheet"],
            "checkpoint_label": df["checkpoint_label"],
            "cycle_id_within_file": df["cycle_id_within_file"],
            "cycle_id": df["global_cycle_est"],
            "cycle_offset_confidence": df["cycle_offset_confidence"],
            "capacity_ah": df["discharge_capacity_ah"],
            "soh_pct_nominal_1.25Ah": df["soh_nominal_pct"],
            "retention_bol_pct": df["retention_bol_pct"],
            "capacity_status": df["capacity_status"],
            "soh_status": df["soh_status"],
            "discharge_segment_complete": df["discharge_segment_complete"],
            "current_total_status": df["total_current_status"],
            "feasibility_overall": df["feasibility_overall"],
            "blockers": df["blockers"],
        }
    )
    out["label_status"] = np.where(out["capacity_ah"].notna(), "CONDITIONAL", "BLOCKED")
    out.loc[out["cycle_offset_confidence"].ne("VERIFIED"), "label_status"] = (
        "CONDITIONAL_OFFSET_UNDERDETERMINED"
    )
    out.loc[out["capacity_ah"].isna(), "label_status"] = "BLOCKED_PARTIAL_DISCHARGE"
    out["modelling_use"] = np.where(
        out["capacity_ah"].notna() & out["cycle_offset_confidence"].eq("VERIFIED"),
        "CONDITIONAL_SOH_TARGET",
        "BLOCKED",
    )
    out.to_csv(OUT / "B1_candidate_cycle_labels.csv", index=False)
    return {
        "cycles": int(len(out)),
        "capacity_non_null": int(out["capacity_ah"].notna().sum()),
        "blocked_partial": int(out["capacity_ah"].isna().sum()),
        "offset_underdetermined": int(out["cycle_offset_confidence"].ne("VERIFIED").sum()),
        "missing_cycle_ids": "151-170 (Cycle_0200 has 30 tabs), 1201-1220 (Cycle_1300 has 80 tabs)",
        "cycle_id_min": int(out["cycle_id"].min()),
        "cycle_id_max": int(out["cycle_id"].max()),
    }


def b3_export():
    with zipfile.ZipFile(AUDIT_DIR / "Existing_Audits_Available_B3_B4 (1).zip") as z:
        df = pd.read_csv(io.BytesIO(z.read("B3_B4/b3_cycle_summary (1).csv")))
    out = pd.DataFrame(
        {
            "dataset_id": "LFP_MultiStat_B3",
            "cell_id": "BATTERY_3",
            "source_file": df["source_workbook"],
            "worksheet": df["source_sheet"],
            "channel": df["channel"],
            "sheet_cycle_no": df["sheet_cycle_no"],
            "cycle_id": df["cycle_index_nominal"],
            "cycle_index_confidence": df["cycle_index_confidence"],
            "capacity_ah": df["Qdis_Ah"],
            "soh_pct_nominal_1.25Ah": df["SOH_vs_nominal_1.25Ah_pct"],
            "soh_pct_vs_initial_2C": df["SOH_vs_initial_2C_pct"],
            "charge_capacity_ah": df["Qchg_Ah"],
            "coulombic_efficiency": df["coulombic_eff"],
            "quality_tier": df["quality_tier"],
            "flags": df["flags"],
            "n_samples": df["n_samples"],
            "n_timestamp_gaps": df["n_timestamp_gaps"],
        }
    )
    out["label_status"] = np.where(out["capacity_ah"].notna(), "CONDITIONAL", "BLOCKED")
    out["modelling_use"] = np.where(
        out["quality_tier"].eq("A") & out["capacity_ah"].notna(),
        "CONDITIONAL_SOH_TARGET",
        "BLOCKED_OR_EXCLUDE",
    )
    out.to_csv(OUT / "B3_candidate_cycle_labels.csv", index=False)
    ids = set(out["cycle_id"].dropna().astype(int))
    missing = [i for i in range(1, max(ids) + 1) if i not in ids]
    return {
        "cycles": int(len(out)),
        "capacity_non_null": int(out["capacity_ah"].notna().sum()),
        "quality_tiers": out["quality_tier"].value_counts().to_dict(),
        "channels": out["channel"].value_counts().to_dict(),
        "missing_cycle_ids": missing,
        "capacity_range": [float(out["capacity_ah"].min()), float(out["capacity_ah"].max())],
        "soh_range": [
            float(out["soh_pct_nominal_1.25Ah"].min()),
            float(out["soh_pct_nominal_1.25Ah"].max()),
        ],
    }


def b4_export():
    """Every cycle tab is listed; capacity/SOH stay null because units/segmentation are unresolved."""
    zip_path = DATA / "B4_Discharge3C_Charge0.5C (1).zip"
    rows = []
    with zipfile.ZipFile(zip_path) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
        for member in members:
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as inner:
                xml = inner.read("xl/workbook.xml").decode("utf-8", "ignore")
            names = re.findall(r'<sheet[^>]*name="([^"]*)"', xml)
            step2 = [n for n in names if "Step2_Cycle" in n]
            checkpoint = int(re.search(r"Cycle_(\d+)", member).group(1))
            n = len(step2)
            for k, sheet in enumerate(step2, start=1):
                cycle_id = checkpoint - n + k
                rows.append(
                    {
                        "dataset_id": "LFP_MultiStat_B4",
                        "cell_id": "BATTERY_4",
                        "source_file": Path(member).name,
                        "worksheet": sheet,
                        "checkpoint_label": checkpoint,
                        "cycle_id": cycle_id,
                        "cycle_id_confidence": "RECONSTRUCTED_FROM_SHEET_ORDER",
                        "capacity_ah": None,
                        "soh_pct_nominal_1.25Ah": None,
                        "label_status": "BLOCKED_NO_AUDITED_TARGET",
                        "modelling_use": "TELEMETRY_ONLY_NO_TARGET",
                        "blockers": (
                            "I(A/cm2) header conflicts with protocol amperes; surface area=0; "
                            "charge/discharge segmentation and BOL reference unverified"
                        ),
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "B4_candidate_cycle_labels.csv", index=False)
    return {
        "cycles": int(len(out)),
        "workbooks": int(out["source_file"].nunique()),
        "capacity_non_null": 0,
        "old_audit_coverage": "1,382 rows from 16 checkpoints; 1,350 sheets (historical)",
        "new_coverage": f"{len(out)} tabs across {out['source_file'].nunique()} workbooks",
    }


def b5_export():
    with zipfile.ZipFile(AUDIT_DIR / "B5.zip") as z:
        df = pd.read_csv(io.BytesIO(z.read("B5_Cycle_Summary.csv")))
    dropped = df[df["worksheet"].astype(str).str.lower() == "file list"]
    df = df[df["worksheet"].astype(str).str.lower() != "file list"].copy()
    out = pd.DataFrame(
        {
            "dataset_id": "LFP_MultiStat_B5",
            "cell_id": "BATTERY_5",
            "source_file": df["source_file"],
            "worksheet": df["worksheet"],
            "sheet_class": df["sheet_class"],
            "checkpoint": df["checkpoint"],
            "cycle_id": df["global_cycle_id_nominal_checkpoint"],
            "global_id_status": df["global_id_status"],
            "discharge_capacity_ah_conditional": df["dis_q_ah_trapz_CALCULATED"],
            "charge_capacity_ah_unsupported_diagnostic": df["chg_q_ah_UNSUPPORTED_diagnostic_only"],
            "soh_pct_nominal_1.25Ah_conditional": df["soh_nominal_pct_CONDITIONAL_1.25Ah_rated"],
            "ce_unsupported_diagnostic": df["ce_UNSUPPORTED_diagnostic_only"],
            "capacity_ground_truth": df["capacity_ground_truth"],
            "capacity_status": df["capacity_status"],
            "cycle_status": df["cycle_status"],
            "dis_complete": df["dis_complete"],
            "quality_flags": df["quality_flags"],
            "modelling_use_capacity_soh": df["modelling_use_capacity_soh"],
            "modelling_use_charge_capacity_ce": df["modelling_use_charge_capacity_ce"],
        }
    )
    out["partial_discharge"] = ~out["dis_complete"].astype(str).str.lower().isin(["true", "1", "yes"])
    out.to_csv(OUT / "B5_candidate_cycle_labels.csv", index=False)
    return {
        "cycles_exported": int(len(out)),
        "audit_rows_dropped_as_non_cycle": int(len(dropped)),
        "dropped_rows": dropped[["source_file", "worksheet", "global_cycle_id_nominal_checkpoint"]]
        .astype(str)
        .to_dict("records"),
        "capacity_ground_truth_yes": int(out["capacity_ground_truth"].eq("YES").sum()),
        "partial_discharge_cycles": int(out["partial_discharge"].sum()),
        "cycle40": out[out["cycle_id"].eq(40)][
            ["worksheet", "discharge_capacity_ah_conditional", "dis_complete", "quality_flags"]
        ].astype(str).to_dict("records"),
        "capacity_range": [
            float(out["discharge_capacity_ah_conditional"].min()),
            float(out["discharge_capacity_ah_conditional"].max()),
        ],
    }


def b2_manifests():
    with zipfile.ZipFile(DATA / "B2_LFP_Member1_audit.zip") as z:
        df = pd.read_parquet(io.BytesIO(z.read("LFP_Member1_audit/exports/cycle_features.parquet")))
    mask = df[
        ["cycle_index", "file_id", "sheet", "channel", "in_analysis", "valid_cycle", "provenance"]
    ].copy()
    mask["transition_window_999_1015"] = mask["cycle_index"].between(999, 1015)
    mask["recommended_treatment"] = np.where(
        mask["transition_window_999_1015"], "EXCLUDE_PRIMARY__OR_CHANNEL_COVARIATE_SENSITIVITY", "KEEP"
    )
    mask.to_csv(OUT / "B2_channel_transition_mask.csv", index=False)

    valid = df[df["in_analysis"]].copy()
    trans = valid["cycle_index"].between(999, 1015)
    excl = valid[~trans]
    n_train = int(round(len(excl) * 0.70))
    n_val = int(round(len(excl) * 0.15))
    summary = {
        "total_feature_rows": int(len(df)),
        "in_analysis": int(len(valid)),
        "invalid_cycles": int((~df["in_analysis"]).sum()),
        "channel_counts": df["channel"].value_counts().to_dict(),
        "transition_cycle_first_channel2": int(df.loc[df["channel"].eq(2.0), "cycle_index"].min()),
        "transition_window_cycles": int(trans.sum()) + int(
            df["cycle_index"].between(999, 1015).sum() - trans.sum()
        ),
        "treatment_exclusion": {
            "cycles": int(len(excl)),
            "removed_eligible": int(trans.sum()),
            "chronological_70_15_15": [n_train, n_val, len(excl) - n_train - n_val],
            "note": "cycles 999-1015 removed; 999 is already invalid, 1000-1015 are eligible",
        },
        "treatment_channel_covariate": {
            "cycles": int(len(valid)),
            "chronological_70_15_15": [
                int(round(len(valid) * 0.70)),
                int(round(len(valid) * 0.15)),
                len(valid) - int(round(len(valid) * 0.70)) - int(round(len(valid) * 0.15)),
            ],
            "note": "retain all valid cycles and pass instrument channel as a covariate",
        },
        "recommendation": "EXCLUDE primary; report channel-covariate sensitivity. Member 2 decides.",
    }
    (OUT / "B2_channel_transition_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    target_cols = {
        "q_discharge_ah": ("TARGET", "Same-cycle discharge capacity in Ah; never an input for a capacity/SOH target."),
        "soh_rated": ("TARGET", "q_discharge_ah / 1.25 Ah as a 0-1 fraction; multiply by 100 for percent."),
        "soh_initial_ref": ("TARGET_DERIVED_LEAKAGE", "Normalised by the first valid cycle; leaks trajectory-wide information."),
        "soh_slope_per_100cyc": ("TARGET_DERIVED_LEAKAGE", "Fitted over a cycle window; contains future-cycle information."),
        "q_fade_rate_ah_per_100cyc": ("TARGET_DERIVED_LEAKAGE", "Fitted capacity fade rate; contains future-cycle information."),
        "coulombic_efficiency": ("TARGET_DERIVED", "q_discharge_ah / q_charge_ah, so it embeds the target."),
    }
    metadata = {
        "cycle_index", "file_id", "sheet", "session_in_workbook", "session_global",
        "timestamp", "channel", "provenance", "valid_cycle", "in_analysis", "ce_reliable",
    }
    qc = {"max_overlap_frac"}
    rows = []
    for col in df.columns:
        if col in target_cols:
            role, note = target_cols[col]
            rows.append((col, role, "YES", note))
        elif col in metadata:
            rows.append((col, "METADATA", "NO", "Identity/provenance; not a model input."))
        elif col in qc:
            rows.append((col, "QUALITY_CONTROL", "NO", "Validity/overlap screen."))
        elif col in {"charge_c_rate", "discharge_c_rate"}:
            rows.append((col, "PERMITTED_INPUT", "NO", "Protocol setpoint; known before the cycle."))
        elif col.startswith(("discharge_qv_", "charge_qv_")):
            rows.append((col, "PERMITTED_INPUT", "NO", "Same-cycle measured Q-V curve point."))
        elif col in {"v_upper_v", "v_lower_v", "q_rated_ah"}:
            rows.append((col, "PERMITTED_INPUT", "NO", "Protocol/nominal constants."))
        else:
            rows.append(
                (col, "CONDITIONAL_INPUT", "REVIEW",
                 "Same-cycle summary statistic; confirm it cannot reconstruct q_discharge_ah.")
            )
    manifest = pd.DataFrame(rows, columns=["column", "role", "same_cycle_leakage", "notes"])
    manifest.to_csv(AUDIT_DIR / "MEMBER1_FEATURE_TARGET_MANIFEST.csv", index=False)

    targets = pd.DataFrame(
        [
            ("LFP_MultiStat_B1", "soh_pct_nominal_1.25Ah", "capacity / 1.25 Ah x 100", "%", "CONDITIONAL",
             "110 cycles have underdetermined offsets; current header underdetermined"),
            ("LFP_MultiStat_B2", "soh_rated", "q_discharge_ah / 1.25 Ah (0-1 fraction)", "fraction", "CONDITIONAL",
             "Multiply by 100 for percent; 10 invalid cycles; channel transition at 999"),
            ("LFP_MultiStat_B3", "SOH_vs_nominal_1.25Ah_pct", "Qdis_Ah / 1.25 Ah x 100", "%", "CONDITIONAL",
             "Channel/cell identity undocumented; 21 missing cycle ids; 8 quality-X cycles"),
            ("LFP_MultiStat_B4", "capacity_ah", "not derived", "Ah", "BLOCKED",
             "Current unit and phase boundaries unresolved; no audited target"),
            ("LFP_MultiStat_B5", "soh_nominal_pct_CONDITIONAL_1.25Ah_rated", "capacity / 1.25 Ah x 100", "%",
             "CONDITIONAL", "capacity_ground_truth=NO; Qdis>Qchg in 1,488 cycles; cycle 40 partial"),
            ("LFP_MultiStat_B6", "soh_nominal_pct_conditional", "capacity / 1.25 Ah x 100", "%", "CONDITIONAL",
             "Chronology reconstructed from headers/File List; current unit underdetermined"),
            ("NASA_Ames_PCoE", "soh_pct", "capacity / 2.0 Ah x 100", "%", "VERIFIED",
             "Direct measured capacity; measured temperature available"),
            ("Zenodo_IonTech_NaIon", "soh_pct", "capacity / nominal_capacity_in_Ah x 100", "%", "VERIFIED",
             "Nominal capacity per cell; temperature not measured"),
        ],
        columns=["dataset_id", "target_name", "formula", "units", "status", "notes"],
    )
    targets.to_csv(AUDIT_DIR / "MEMBER1_TARGET_DEFINITIONS.csv", index=False)
    return summary


def main() -> int:
    report = {
        "B1": b1_export(),
        "B3": b3_export(),
        "B4": b4_export(),
        "B5": b5_export(),
        "B2": b2_manifests(),
    }
    (OUT / "member1_cycle_candidates_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
