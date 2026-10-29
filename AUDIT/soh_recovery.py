"""
Member 1 - Missing / questionable SOH label recovery audit (read-only).

Scope
-----
* Enumerate every (dataset, cell, cycle) label in the canonical master table.
* Investigate every MISSING SOH label and every label whose cycle pairing or
  definition is questionable.
* Recover a value ONLY from an authoritative, traceable source whose definition,
  units and calculation are documented; record source, method and evidence.
* Never interpolate, forward-fill or fabricate. Never modify the master table.

Outputs (written only inside AUDIT/member1_audit_outputs/):
    soh_recovery_log.csv    one row per cycle (819) with classification,
                            adopted values, usable-before/after and evidence.
    soh_recovery_report.json machine-readable summary, incl. usable-cycle delta.

Run:
    python AUDIT/soh_recovery.py

Exit code 0 when the canonical files are unchanged and the audit is consistent,
1 otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
sys.path.insert(0, str(AUDIT_DIR))

import member1_audit as base  # noqa: E402

OUT_DIR = AUDIT_DIR / "member1_audit_outputs"
B5_AUDIT_ZIP = AUDIT_DIR / "B5.zip"
B5_RAW_ZIP = REPO_ROOT / "DATA" / "B5_Discharge0.5C_Charge1C (1).zip"
B5_RAW_INNER = "B5_Discharge0.5C_Charge1C/Cycle_0050.xlsx"
B5_RAW_SHEET = "Run01_Un1Ch1_Step2_Cycle40"
B1_AUDIT_ZIP = AUDIT_DIR / "B1_LFP_Audit_Output.zip"
B1_CYCLE_LEVEL = "B1_LFP_Audit_Output/LFP_staging_cycle_level.csv"
B1_SAMPLE_ROWS = "B1_LFP_Audit_Output/LFP_staging_sample_raw_rows.csv"
B1_FEASIBILITY = "B1_LFP_Audit_Output/LFP_capacity_soh_feasibility.csv"

B5_PARTIAL_CYCLE = 40
DISCHARGE_CUTOFF_V = 2.5


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


def load_b5_summary() -> pd.DataFrame:
    with zipfile.ZipFile(B5_AUDIT_ZIP) as archive:
        name = next(n for n in archive.namelist() if n.endswith("B5_Cycle_Summary.csv"))
        return pd.read_csv(archive.open(name))


def load_b1_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(B1_AUDIT_ZIP) as archive:
        cycle_level = pd.read_csv(archive.open(B1_CYCLE_LEVEL))
        sample = pd.read_csv(archive.open(B1_SAMPLE_ROWS))
        feasibility = pd.read_csv(archive.open(B1_FEASIBILITY))
    return cycle_level, sample, feasibility


def raw_b5_cycle40_evidence() -> dict:
    """Recompute the discharge segment of B5 cycle 40 from the raw workbook."""
    if not B5_RAW_ZIP.exists():
        return {"available": False, "reason": f"raw archive not found: {B5_RAW_ZIP.name}"}
    with zipfile.ZipFile(B5_RAW_ZIP) as archive:
        raw = archive.read(B5_RAW_INNER)
    import openpyxl

    workbook = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
    worksheet = workbook[B5_RAW_SHEET]
    rows = list(worksheet.iter_rows(values_only=True))
    workbook.close()

    meta: dict[str, str] = {}
    header_index = None
    for index, row in enumerate(rows):
        if not row or not isinstance(row[0], str):
            continue
        line = row[0].strip()
        if line.startswith("Exp Title:"):
            meta["exp_title"] = line.split(":", 1)[1].strip()
            if len(row) > 1 and row[1] is not None:
                meta["protocol"] = str(row[1]).strip()
        elif line.startswith("I Name:"):
            meta["current_header"] = line.split(":", 1)[1].strip()
        if line == "T(Seconds)":
            header_index = index
            break
    if header_index is None:
        return {"available": False, "reason": "T(Seconds) header row not found"}

    block = rows[header_index + 1 :]
    frame = pd.DataFrame([r[:3] for r in block], columns=["t_s", "v_v", "i_a"])
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna()
    discharge = frame[frame["i_a"] < 0]
    if discharge.empty:
        return {"available": False, "reason": "no negative-current rows in worksheet"}

    t = discharge["t_s"].to_numpy(dtype=float)
    i_abs = discharge["i_a"].abs().to_numpy(dtype=float)
    hold = float(np.sum(i_abs[:-1] * np.diff(t)) / 3600.0)
    return {
        "available": True,
        "source_archive": B5_RAW_ZIP.name,
        "source_member": B5_RAW_INNER,
        "source_sha256": sha256_bytes(raw),
        "worksheet": B5_RAW_SHEET,
        "protocol_metadata": meta,
        "numeric_rows": int(len(frame)),
        "discharge_rows": int(len(discharge)),
        "discharge_t_start_s": float(t[0]),
        "discharge_t_end_s": float(t[-1]),
        "discharge_duration_s": float(t[-1] - t[0]),
        "voltage_start_v": float(discharge["v_v"].iloc[0]),
        "voltage_end_v": float(discharge["v_v"].iloc[-1]),
        "voltage_min_v": float(discharge["v_v"].min()),
        "samples_at_or_below_cutoff": int((discharge["v_v"] <= DISCHARGE_CUTOFF_V).sum()),
        "fraction_at_or_below_cutoff": float((discharge["v_v"] <= DISCHARGE_CUTOFF_V).mean()),
        "mean_abs_current_a": float(i_abs.mean()),
        "partial_capacity_trapz_ah": _trapezoid(i_abs, t) / 3600.0,
        "partial_capacity_samplehold_ah": hold,
        "record_ends_at_last_discharge_row": bool(
            abs(float(frame["t_s"].iloc[-1]) - float(t[-1])) < 1e-9
        ),
    }


def b5_cycle40_evidence(summary: pd.DataFrame) -> dict:
    record = summary[summary["global_cycle_id_nominal_checkpoint"] == B5_PARTIAL_CYCLE]
    if len(record) != 1:
        raise RuntimeError(f"expected exactly 1 summary row for B5 cycle 40, got {len(record)}")
    row = record.iloc[0]
    neighbours = summary[
        (summary["checkpoint"] == row["checkpoint"]) & (summary["dis_complete"] == True)  # noqa: E712
    ]["dis_q_ah_trapz_CALCULATED"].dropna()
    reference_ah = float(neighbours.median())
    raw = raw_b5_cycle40_evidence()
    partial_ah = float(row["dis_q_ah_trapz_CALCULATED"])
    return {
        "target": "LFP_MultiStat_B5::BATTERY_5::cycle_40",
        "summary_row_count_for_cycle": int(len(record)),
        "cycle_status": str(row["cycle_status"]),
        "discharge_complete": bool(row["dis_complete"]),
        "discharge_rows_audit": int(row["dis_rows"]),
        "discharge_i_mean_a": float(row["dis_i_mean_a"]),
        "discharge_v_start_v": float(row["dis_v_start"]),
        "discharge_v_end_v": float(row["dis_v_end"]),
        "partial_capacity_trapz_ah": partial_ah,
        "partial_capacity_samplehold_ah": float(row["dis_q_ah_samplehold_CALCULATED_sensitivity"]),
        "legacy_proxy_ah": float(row["reconcile_old_proxy_ah"]),
        "conditional_capacity_ah": None
        if pd.isna(row["discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth"])
        else float(row["discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth"]),
        "conditional_soh_pct": None
        if pd.isna(row["soh_nominal_pct_CONDITIONAL_1.25Ah_rated"])
        else float(row["soh_nominal_pct_CONDITIONAL_1.25Ah_rated"]),
        "capacity_ground_truth": str(row["capacity_ground_truth"]),
        "capacity_status": str(row["capacity_status"]),
        "modelling_use_capacity_soh": str(row["modelling_use_capacity_soh"]),
        "summary_content_sha1": str(row["content_sha1"]),
        "neighbouring_full_cycle_median_ah": reference_ah,
        "partial_fraction_of_neighbour_median": partial_ah / reference_ah,
        "raw_workbook": raw,
        "recovered_capacity_ah": None,
        "recovered_soh_pct": None,
        "recovery_status": "NOT_RECOVERABLE",
        "reason": (
            "No complete discharge exists for this cycle. The raw worksheet's "
            "discharge segment ends at "
            f"{raw['voltage_end_v']:.4f} V after {raw['discharge_duration_s']:.0f} s "
            f"({raw['discharge_rows']} rows) with 0 samples at or below the "
            f"{DISCHARGE_CUTOFF_V} V cutoff; the workbook record ends there. "
            "The audit records DISCHARGE_INTERRUPTED, dis_complete=False, "
            "capacity_ground_truth=NO, modelling_use_capacity_soh=EXCLUDE and a "
            "NaN conditional capacity/SOH. No second measurement of cycle 40 "
            "exists (1 summary row). The 0.5428 Ah value is partial throughput, "
            "not a cell capacity, so SOH cannot be derived without inventing the "
            "missing ~55% of the discharge."
        ),
        "what_would_make_it_recoverable": (
            "A repeat/complete discharge for this cycle (or the original lab "
            "record of the full 0.625 A CC discharge to the 2.5 V cutoff) with "
            "the same cell state and protocol; alternatively a documented, "
            "authoritative full-discharge capacity measurement for this exact "
            "cycle. None exists in the workspace."
        ),
    }


def b1_recovery_evidence(
    master: pd.DataFrame,
    cycle_level: pd.DataFrame,
    sample: pd.DataFrame,
    feasibility: pd.DataFrame,
) -> dict:
    checkpoint = int(sample["cycle_checkpoint"].iloc[0])
    est = int(sample["global_cycle_est"].iloc[0])
    if sample["cycle_checkpoint"].nunique() != 1 or sample["global_cycle_est"].nunique() != 1:
        raise RuntimeError("telemetry sample spans more than one checkpoint/cycle")
    row = feasibility[
        (feasibility["checkpoint_label"] == checkpoint)
        & (feasibility["global_cycle_est"] == est)
    ]
    if len(row) != 1:
        raise RuntimeError(f"expected exactly 1 feasibility row for B1 est={est}, got {len(row)}")
    row = row.iloc[0]
    cycle_row = cycle_level[
        (cycle_level["cycle_checkpoint"] == checkpoint)
        & (cycle_level["global_cycle_est"] == est)
    ].iloc[0]
    b1 = master[master["dataset_id"] == "LFP_MultiStat_B1"]
    master_cap = float(b1["capacity_ah"].iloc[0])
    master_soh = float(b1["soh_pct"].iloc[0])
    recovered_cap = float(row["discharge_capacity_ah"])
    recovered_soh = float(row["soh_nominal_pct"])
    last_cycle = cycle_level[cycle_level["cycle_checkpoint"] == checkpoint].iloc[-1]
    return {
        "target": f"LFP_MultiStat_B1::BATTERY_1::cycle_{int(b1['cycle_id'].iloc[0])}",
        "telemetry_checkpoint": checkpoint,
        "telemetry_global_cycle_est": est,
        "master_capacity_ah": master_cap,
        "master_soh_pct": master_soh,
        "recovered_capacity_ah": recovered_cap,
        "recovered_soh_pct": recovered_soh,
        "collapsed_checkpoint_last_cycle_est": int(last_cycle["global_cycle_est"]),
        "collapsed_checkpoint_last_cycle_capacity_ah": float(last_cycle["capacity_ah"]),
        "cycle_offset_confidence": str(row["cycle_offset_confidence"]),
        "discharge_segment_complete": bool(row["discharge_segment_complete"]),
        "capacity_status": str(row["capacity_status"]),
        "soh_status": str(row["soh_status"]),
        "feasibility_overall": str(row["feasibility_overall"]),
        "current_unit_status": str(row["total_current_status"]),
        "cycle_level_quality_flag": str(cycle_row["quality_flag"]),
        "cycle_level_formula_id": str(cycle_row["formula_id"]),
        "source_artifact": (
            "AUDIT/B1_LFP_Audit_Output.zip::" + B1_FEASIBILITY +
            f" (checkpoint_label={checkpoint}, global_cycle_est={est})"
        ),
        "recovery_status": "RECOVERED_WITH_CONDITIONS",
        "tier_after_recovery": "CONDITIONAL",
        "method": (
            "Direct lookup of the audited per-cycle computed value for the "
            "telemetry's VERIFIED global cycle index; no interpolation, no "
            "scaling, no use of neighbouring cycles. Replaces the label that the "
            "builder collapsed from the checkpoint's last cycle."
        ),
        "reason": (
            f"Telemetry rows carry global_cycle_est={est} (checkpoint {checkpoint}); "
            f"the audit computes {recovered_cap:.6f} Ah / {recovered_soh:.3f}% for "
            f"that cycle with a complete discharge, cycle_offset_confidence=VERIFIED "
            f"and feasibility={row['feasibility_overall']}. The master label "
            f"{master_cap:.6f} Ah belongs to est={int(last_cycle['global_cycle_est'])} "
            "of the same checkpoint, i.e. a different cycle. The recovered label is "
            "CONDITIONAL, not HIGH, because the workbook current header "
            "(I(A/cm2), Surface Area=0) leaves total amperes underdetermined."
        ),
    }


def classification_frame(
    master: pd.DataFrame,
    baseline: pd.DataFrame,
    b5: dict,
    b1: dict,
) -> pd.DataFrame:
    base_lookup = {
        (r.dataset_id, r.cell_id, int(r.cycle_id)): r
        for r in baseline.itertuples(index=False)
    }
    rows = []
    for (dataset_id, cell_id, cycle_id), group in master.groupby(
        ["dataset_id", "cell_id", "cycle_id"], sort=True
    ):
        cycle_id = int(cycle_id)
        before = base_lookup[(dataset_id, cell_id, cycle_id)]
        cap = group["capacity_ah"].iloc[0]
        soh = group["soh_pct"].iloc[0]
        tier_before = before.target_confidence

        state = "VALID_MEASURED"
        issue = "NONE"
        recovery = "NOT_REQUIRED"
        adopted_cap = None if pd.isna(cap) else float(cap)
        adopted_soh = None if pd.isna(soh) else float(soh)
        tier_after = tier_before
        source = ""
        locator = ""
        method = "Label present and traceable; no recovery needed."
        evidence = ""
        digest = ""

        if dataset_id == "NASA_Ames_PCoE":
            if not pd.isna(soh) and soh > 100.0:
                state = "VALID_DEFINITIONAL"
                issue = "SOH_ABOVE_100PCT_DUE_TO_RATED_BASELINE"
                evidence = (
                    "soh_pct = capacity_ah / 2.0 Ah manufacturer rating; a fresh "
                    "cell can exceed its rating, so >100% is definitional and "
                    "physically consistent, not a labelling error."
                )
                method = "Definition reviewed; value retained unchanged."
        elif dataset_id == "Zenodo_IonTech_NaIon":
            state = "VALID_DEFINITIONAL"
            issue = "SOH_VS_RATED_NOT_MEASURED_BOL"
            evidence = (
                "soh_pct = capacity_ah / 1.0 Ah rated nominal, not retention vs "
                "measured beginning-of-life; SOH starts below 100% because the "
                "measured BOL capacity is below the rating. Definitional, not missing."
            )
            method = "Definition reviewed; value retained unchanged."
        elif dataset_id == "LFP_MultiStat_B5":
            if cycle_id == B5_PARTIAL_CYCLE:
                state = "MISSING_NOT_RECOVERABLE"
                issue = "PARTIAL_DISCHARGE"
                recovery = "NOT_RECOVERABLE"
                adopted_cap = None
                adopted_soh = None
                tier_after = "INVALID"
                raw = b5["raw_workbook"]
                source = b5["source_artifact"] if "source_artifact" in b5 else (
                    f"{B5_RAW_ZIP.name}::{B5_RAW_INNER} + AUDIT/B5.zip::B5_Cycle_Summary.csv"
                )
                locator = "global_cycle_id_nominal_checkpoint=40; sheet " + B5_RAW_SHEET
                digest = raw.get("source_sha256", "") if raw.get("available") else ""
                method = (
                    "Investigated raw workbook + audit summary; no complete "
                    "discharge exists, so no value was adopted."
                )
                evidence = b5["reason"]
            else:
                state = "CONDITIONAL_SOURCE_CAVEAT"
                issue = "CHARGE_BALANCE_DISCREPANCY"
                evidence = (
                    "Capacity/SOH are source-computed conditional values "
                    "(Q_dis > Q_chg in ~99.7% of cycles); label present, not missing. "
                    "No recovery applies."
                )
                method = "Label present; conditional tier retained."
        elif dataset_id == "LFP_MultiStat_B1":
            state = "QUESTIONABLE_RECOVERED_WITH_CONDITIONS"
            issue = "LABEL_CYCLE_MISMATCH"
            recovery = "RECOVERED_WITH_CONDITIONS"
            adopted_cap = b1["recovered_capacity_ah"]
            adopted_soh = b1["recovered_soh_pct"]
            tier_after = "CONDITIONAL"
            source = b1["source_artifact"]
            locator = (
                f"checkpoint_label={b1['telemetry_checkpoint']}, "
                f"global_cycle_est={b1['telemetry_global_cycle_est']}"
            )
            method = b1["method"]
            evidence = b1["reason"]

        usable_before = tier_before in {"HIGH", "CONDITIONAL"}
        usable_after = tier_after in {"HIGH", "CONDITIONAL"}
        rows.append(
            {
                "dataset_id": dataset_id,
                "cell_id": cell_id,
                "chemistry": group["chemistry"].iloc[0],
                "cycle_id": cycle_id,
                "n_telemetry_rows": int(len(group)),
                "master_capacity_ah": None if pd.isna(cap) else float(cap),
                "master_soh_pct": None if pd.isna(soh) else float(soh),
                "label_state": state,
                "issue_type": issue,
                "recovery_status": recovery,
                "recovered_capacity_ah": b1["recovered_capacity_ah"]
                if dataset_id == "LFP_MultiStat_B1"
                else None,
                "recovered_soh_pct": b1["recovered_soh_pct"]
                if dataset_id == "LFP_MultiStat_B1"
                else None,
                "adopted_capacity_ah": adopted_cap,
                "adopted_soh_pct": adopted_soh,
                "target_confidence_before": tier_before,
                "target_confidence_after": tier_after,
                "usable_as_target_before": usable_before,
                "usable_as_target_after": usable_after,
                "source_artifact": source,
                "source_locator": locator,
                "method": method,
                "evidence": evidence,
                "source_sha256": digest,
            }
        )
    return pd.DataFrame(rows)


def build_report(master: pd.DataFrame, log: pd.DataFrame, b5: dict, b1: dict, hashes: dict) -> dict:
    state_counts = log["label_state"].value_counts().sort_index().to_dict()
    usable_before = int(log["usable_as_target_before"].sum())
    usable_after = int(log["usable_as_target_after"].sum())

    def tier_counts(col: str) -> dict:
        return {str(k): int(v) for k, v in log[col].value_counts().sort_index().items()}

    def per_dataset(col: str) -> dict:
        return {
            str(k): int(v)
            for k, v in log.groupby("dataset_id")[col].sum().sort_index().items()
        }

    recovered = log[log["recovery_status"] == "RECOVERED_WITH_CONDITIONS"]
    not_recoverable = log[log["recovery_status"] == "NOT_RECOVERABLE"]

    missing_soh = int(master["soh_pct"].isna().sum())
    return {
        "metadata": {
            "title": "Member 1 - Missing / questionable SOH label recovery audit",
            "project": "Autonomous Multi-Agent EV Battery Life & Range Predictor",
            "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
            "canonical_dataset_modified": False,
            "method_notes": (
                "Read-only. Every recovered value comes from a direct lookup in an "
                "authoritative audit table for the exact cycle; no interpolation, "
                "no forward/backward fill, no synthetic measurements."
            ),
        },
        "guardrails": {
            "master_table_parquet_sha256": hashes["parquet"],
            "master_table_csv_sha256": hashes["csv"],
            "parquet_hash_expected": base.EXPECTED_SHA256_PARQUET,
            "csv_hash_expected": base.EXPECTED_SHA256_CSV,
            "hashes_match_expected": (
                hashes["parquet"] == base.EXPECTED_SHA256_PARQUET
                and hashes["csv"] == base.EXPECTED_SHA256_CSV
            ),
            "interpolation_used": False,
            "forward_fill_used": False,
            "fabricated_values": 0,
            "writes_confined_to": str(OUT_DIR.relative_to(REPO_ROOT)),
        },
        "scope": {
            "cycles_enumerated": int(len(log)),
            "cycles_with_missing_soh_rows_in_master": missing_soh,
            "missing_soh_cycles": sorted(
                int(c)
                for c in master[master["soh_pct"].isna()]
                .loc[lambda d: d["dataset_id"] == "LFP_MultiStat_B5", "cycle_id"]
                .unique()
            ),
            "label_state_counts": {str(k): int(v) for k, v in state_counts.items()},
            "issue_type_counts": {
                str(k): int(v) for k, v in log["issue_type"].value_counts().items()
            },
            "recovered_cycles": int(len(recovered)),
            "not_recoverable_cycles": int(len(not_recoverable)),
        },
        "usable_cycle_counts": {
            "before": {
                "total_cycles": int(len(log)),
                "usable_as_target": usable_before,
                "by_dataset": per_dataset("usable_as_target_before"),
                "by_tier": tier_counts("target_confidence_before"),
            },
            "after": {
                "total_cycles": int(len(log)),
                "usable_as_target": usable_after,
                "by_dataset": per_dataset("usable_as_target_after"),
                "by_tier": tier_counts("target_confidence_after"),
            },
            "delta_usable": usable_after - usable_before,
            "high_confidence_before": int((log["target_confidence_before"] == "HIGH").sum()),
            "high_confidence_after": int((log["target_confidence_after"] == "HIGH").sum()),
            "explanation": (
                "The single B1 cycle moves from INVALID to CONDITIONAL because its "
                "label was repaired to the audited value for the telemetry's own "
                "cycle. No value is promoted to HIGH. B5 cycle 40 stays excluded."
            ),
        },
        "investigations": {
            "LFP_MultiStat_B5::BATTERY_5::cycle_40": b5,
            "LFP_MultiStat_B1::BATTERY_1::cycle_2": b1,
        },
        "definitional_cases": {
            "NASA_Ames_PCoE_soh_above_100pct": {
                "cycles": int(
                    (
                        (log["dataset_id"] == "NASA_Ames_PCoE")
                        & (log["issue_type"] == "SOH_ABOVE_100PCT_DUE_TO_RATED_BASELINE")
                    ).sum()
                ),
                "cells": ["B0006"],
                "action": "retained; soh is referenced to the 2.0 Ah rating, not measured BOL",
            },
            "Zenodo_IonTech_NaIon_soh_below_100pct_from_first_cycle": {
                "cycles": int(
                    (
                        (log["dataset_id"] == "Zenodo_IonTech_NaIon")
                        & (log["issue_type"] == "SOH_VS_RATED_NOT_MEASURED_BOL")
                    ).sum()
                ),
                "action": "retained; soh is referenced to the 1.0 Ah rating, not measured BOL",
            },
        },
        "coverage_extension_not_performed": {
            "note": (
                "B5 cycles 101-150 and 201-400 are absent from the master by "
                "ingestion design, not missing labels. The audit summary contains "
                "computed conditional values for them, so they could be ingested "
                "later, but that requires a deliberate coverage decision and a "
                "rebuild; it is outside this recovery audit and is NOT counted in "
                "the usable-cycle numbers."
            ),
            "candidate_cycles_in_source": 250,
            "counted_as_recovery": False,
        },
        "member2_eligibility_impact": {
            "usable_targets_before": usable_before,
            "usable_targets_after": usable_after,
            "change": "+1 conditional cycle (LFP_MultiStat_B1); no HIGH promotion",
            "high_confidence_targets": int((log["target_confidence_after"] == "HIGH").sum()),
            "conditional_targets": int((log["target_confidence_after"] == "CONDITIONAL").sum()),
            "excluded_targets": int((log["target_confidence_after"] == "INVALID").sum()),
            "guidance": (
                "B1 may be used only as a CONDITIONAL/auxiliary target with the "
                "current-unit caveat, never as ground truth and never for thermal "
                "features. B5 cycle 40 must stay excluded from capacity/SOH training."
            ),
        },
        "files": {
            "log": str((OUT_DIR / "soh_recovery_log.csv").relative_to(REPO_ROOT)),
            "report": str((OUT_DIR / "soh_recovery_report.json").relative_to(REPO_ROOT)),
        },
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = {
        "parquet": base.sha256_file(base.MASTER_PARQUET),
        "csv": base.sha256_file(base.MASTER_CSV),
    }
    for name, digest in hashes_before.items():
        expected = (
            base.EXPECTED_SHA256_PARQUET if name == "parquet" else base.EXPECTED_SHA256_CSV
        )
        if digest != expected:
            print(f"ABORT: master_table.{name} hash changed before recovery run: {digest}")
            return 1

    master = base.load_master()
    baseline = base.candidate_cycle_labels(master, base.run_audit()["checks"])
    summary = load_b5_summary()
    cycle_level, sample, feasibility = load_b1_tables()

    b5 = b5_cycle40_evidence(summary)
    b5["source_artifact"] = (
        f"{B5_RAW_ZIP.name}::{B5_RAW_INNER} + AUDIT/B5.zip::B5_Cycle_Summary.csv"
    )
    b1 = b1_recovery_evidence(master, cycle_level, sample, feasibility)

    log = classification_frame(master, baseline, b5, b1)
    if len(log) != 819:
        raise RuntimeError(f"expected 819 cycle rows, got {len(log)}")
    adopted = int((log["recovery_status"] == "RECOVERED_WITH_CONDITIONS").sum())
    delta = int(log["usable_as_target_after"].sum()) - int(log["usable_as_target_before"].sum())
    if delta != adopted:
        raise RuntimeError(f"usable delta {delta} does not match adopted recoveries {adopted}")

    hashes_after = {
        "parquet": base.sha256_file(base.MASTER_PARQUET),
        "csv": base.sha256_file(base.MASTER_CSV),
    }
    if hashes_after != hashes_before:
        print("ABORT: canonical master table changed during the recovery run")
        return 1

    report = build_report(master, log, b5, b1, hashes_after)
    log.to_csv(OUT_DIR / "soh_recovery_log.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    (OUT_DIR / "soh_recovery_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(f"master parquet sha256 : {hashes_after['parquet']} (unchanged)")
    print(f"cycles enumerated     : {len(log)}")
    print(f"recovered             : {adopted} (B1 label-cycle repair, CONDITIONAL)")
    print(f"not recoverable       : {(log['recovery_status'] == 'NOT_RECOVERABLE').sum()} (B5 cycle 40)")
    print(f"usable before -> after: {report['usable_cycle_counts']['before']['usable_as_target']} -> "
          f"{report['usable_cycle_counts']['after']['usable_as_target']}")
    print(f"log                   : {OUT_DIR / 'soh_recovery_log.csv'}")
    print(f"report                : {OUT_DIR / 'soh_recovery_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
