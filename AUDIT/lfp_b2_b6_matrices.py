"""
Member 1 - extend the B1-B5 coverage and feature matrices to B1-B6.

The earlier matrices (``MEMBER1_LFP_B1_B5_*``) remain untouched as the audit
snapshot they were. This script writes superseding B1-B6 matrices:

* coverage  -> AUDIT/MEMBER1_LFP_B1_B6_COVERAGE_MATRIX.csv
* features  -> AUDIT/MEMBER1_LFP_B1_B6_FEATURE_READINESS_MATRIX.csv

Every row carries ``change_vs_b1_b5_report`` so Member 2 can see exactly which
conclusions moved and why (B2 raw verified; B6 added; nothing silently rewritten).

Run:
    python AUDIT/lfp_b2_b6_matrices.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent

OLD_COVERAGE = AUDIT_DIR / "MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv"
OLD_FEATURES = AUDIT_DIR / "MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv"
NEW_COVERAGE = AUDIT_DIR / "MEMBER1_LFP_B1_B6_COVERAGE_MATRIX.csv"
NEW_FEATURES = AUDIT_DIR / "MEMBER1_LFP_B1_B6_FEATURE_READINESS_MATRIX.csv"
CHANGE_LOG = AUDIT_DIR / "MEMBER1_LFP_B1_B6_MATRIX_CHANGELOG.json"

COVERAGE_COLUMNS = [
    "cell_id",
    "dataset_id",
    "source_path",
    "source_status",
    "source_verified",
    "raw_data_available",
    "derived_data_available",
    "master_row_count",
    "master_cycle_count",
    "master_labelled_cycle_count",
    "source_cycle_records",
    "source_labelled_records",
    "soh_provenance",
    "current_unit_status",
    "voltage_available",
    "current_available",
    "time_available",
    "temperature_available",
    "data_quality_issues",
    "independent_validation_suitability",
    "blocker",
    "recommended_action",
    "change_vs_b1_b5_report",
]

FEATURE_COLUMNS = [
    "cell_id",
    "feature_name",
    "required_inputs",
    "verified_units",
    "derivation_method",
    "input_availability",
    "suitability",
    "limitations",
    "missing_input",
    "recommended_action",
    "change_vs_b1_b5_report",
]

B2_FEATURE_UPDATES = {
    "discharge_capacity_ah": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Label column q_discharge_ah; cannot be an input. Re-derived from raw at cycles 50 and 1506 to <1e-8 Ah.",
        "recommended_action": "Use as target only; raw is now available for bespoke telemetry features",
        "change_vs_b1_b5_report": "CHANGED: raw workbooks now present and 20/20 SHA-256 match the derived export; label unchanged",
    },
    "soh_pct_label": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "soh_rated is a 0-1 fraction (x100). Nominal-based; soh_initial_ref_pct is retention vs first valid cycle.",
        "recommended_action": "Use soh_rated_pct as the nominal target and soh_initial_ref_pct as the within-cell retention target; never as input",
        "change_vs_b1_b5_report": "CHANGED: raw verified; second target (soh_initial_ref) surfaced explicitly",
    },
    "voltage_series": {
        "input_availability": "AVAILABLE_IN_SOURCE_ONLY",
        "suitability": "READY_WITH_CAVEAT",
        "limitations": "1 Hz raw telemetry now available; instrument channel switches Un1Ch8->Un1Ch2 at cycle 999.",
        "missing_input": "",
        "recommended_action": "Re-derive per-sample series from raw when needed; exclude or flag cycles 999-1015",
        "change_vs_b1_b5_report": "CHANGED: NOT_AVAILABLE -> raw telemetry verified",
    },
    "dq_dv": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Raw 1 Hz available; exported 101-point Q-V curves trade resolution for size.",
        "missing_input": "",
        "recommended_action": "Use exported curves for speed; re-derive at 1 Hz from raw for peak analysis",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified; derived curves retained",
    },
    "dv_dq": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Raw 1 Hz available; derived curves remain coarse.",
        "missing_input": "",
        "recommended_action": "Use exported curves with safeguards or re-derive from raw",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "ica_peaks": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Coarse curves limit peak resolution; raw 1 Hz now permits sharper dQ/dV.",
        "missing_input": "",
        "recommended_action": "Re-derive dQ/dV from raw for peak work; validate against B5",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "ah_energy_throughput": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Cycle-level only; label columns excluded.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "c_rate_phase_durations": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Cycle-level durations available; raw sign segmentation now possible.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "cccv_fraction": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "charge_cv_fraction provided; raw charge profile available for re-derivation.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "resistance_proxy": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "dcir_proxy_ohm/polarisation provided; channel switch at cycle 999 confounds resistance.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified; channel caveat unchanged",
    },
    "qv_curve_samples": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "101-point discharge/charge Q-V arrays; raw allows higher resolution.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "efc_cycle_index": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Contiguous splits provided; global cycle_index 1..1506 is contiguous.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified",
    },
    "lagged_soh": {
        "input_availability": "AVAILABLE_IN_SOURCE_AND_DERIVED",
        "limitations": "Channel change at 999; exclude 999-1015 or model channel as covariate.",
        "change_vs_b1_b5_report": "CHANGED: raw telemetry verified; split caveat unchanged",
    },
}


def _b6_feature_rows() -> list[dict]:
    base = "BATTERY_6"
    common_missing = ""
    rows = [
        dict(
            feature_name="discharge_capacity_ah",
            required_inputs="segmented discharge current, time base",
            verified_units="Ah (trapezoid of assumed-A current over s; conditional)",
            derivation_method="Coulomb-count |I| between discharge start and 2.50 V cutoff",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="1,697 of 1,702 cycles pass the discharge screen; current header says A/cm2 with Surface Area=0.",
            missing_input=common_missing,
            recommended_action="Use the 1,697 conditional labels in candidates/B6_candidate_cycle_chronology.csv; do not fill the 5 failed cycles",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="soh_pct_label",
            required_inputs="capacity_ah, rated 1.25 Ah",
            verified_units="% (capacity/1.25*100)",
            derivation_method="Direct ratio; nominal-based, not measured BOL retention",
            input_availability="AVAILABLE_IN_AUDIT_CANDIDATE",
            suitability="CONDITIONAL",
            limitations="BOL reference unknown; labels start at 99.65% and end at 81.70%, median 87.99%.",
            missing_input="Validated beginning-of-life capacity",
            recommended_action="Use as nominal target only; never as an input; never treat as BOL retention",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="voltage_series",
            required_inputs="E(V), T(Seconds), cycle id",
            verified_units="V",
            derivation_method="Per-sample voltage at 1 Hz, segmented by current sign",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="READY_WITH_CAVEAT",
            limitations="Two instrument channels (Un1Ch1 201 cycles, Un1Ch6 1,485 cycles); session offsets at workbook boundaries.",
            missing_input=common_missing,
            recommended_action="Use candidate chronology to align cycles; model session or de-trend",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="dq_dv",
            required_inputs="Q(t), V(t) at 1 Hz",
            verified_units="Ah/V (derived)",
            derivation_method="Numerical dQ/dV with monotonic-Q filtering and smoothing",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="Requires phase segmentation; 2 material voltage excursions >20 mV.",
            missing_input=common_missing,
            recommended_action="Segment by current sign before differentiating; flag the 2 excursion cycles",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="dv_dq",
            required_inputs="Q(t), V(t) at 1 Hz",
            verified_units="V/Ah (derived)",
            derivation_method="Numerical dV/dQ on monotonic Q segments",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="Same phase and excursion caveats.",
            missing_input=common_missing,
            recommended_action="Use with monotonic-Q safeguards",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="ica_peaks",
            required_inputs="dQ/dV curve",
            verified_units="Ah/V; peak voltage/height",
            derivation_method="Peak detection with prominence and minimum-height safeguards",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="No independent reference peaks for B6; validate against B5/B2.",
            missing_input=common_missing,
            recommended_action="Validate peaks against a cell with known-good labels before use",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="ah_energy_throughput",
            required_inputs="I(t), V(t), t",
            verified_units="Ah, Wh (computed)",
            derivation_method="Integrate I dt and V I dt separately for charge and discharge",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="Current-unit conditional; charge capacity zero on one interrupted cycle.",
            missing_input=common_missing,
            recommended_action="Compute from raw after unit attestation; exclude the interrupted cycle",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="c_rate_phase_durations",
            required_inputs="I(t), protocol setpoints",
            verified_units="C-rate; s",
            derivation_method="Segment by current sign and compare |I| with 1.25 Ah rating",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="READY_WITH_CAVEAT",
            limitations="Protocol is 0.5C discharge / 2C charge throughout; setpoints known from headers.",
            missing_input=common_missing,
            recommended_action="Use phase durations and C-rate; 378 sheets carry a time-gap flag",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="cccv_fraction",
            required_inputs="I(t), V(t) during charge",
            verified_units="% of charge time in CV; s",
            derivation_method="Detect CV hold after the 3.65 V plateau and the 0.025 A cutoff",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="Requires correct charge-phase segmentation.",
            missing_input=common_missing,
            recommended_action="Derive from raw; validate on a few cycles against the protocol",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="temperature",
            required_inputs="auxiliary temperature channel",
            verified_units="degC",
            derivation_method="Not derivable without a measured channel",
            input_availability="NOT_AVAILABLE",
            suitability="NOT_AVAILABLE",
            limitations="No thermal channel in any B6 workbook (Aux Count 0).",
            missing_input="Measured temperature",
            recommended_action="Do not impute",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="resistance_proxy",
            required_inputs="current step and voltage response",
            verified_units="Ohm (proxy)",
            derivation_method="delta V / delta I across phase transitions",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="Channel switch Un1Ch1->Un1Ch6 at candidate index 202 (2025-05-16) confounds resistance.",
            missing_input=common_missing,
            recommended_action="Model channel as covariate or split at the switch",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="qv_curve_samples",
            required_inputs="Q(t), V(t) per cycle",
            verified_units="Q-V curve (resampled)",
            derivation_method="Resample capacity at fixed fractions of the discharge",
            input_availability="AVAILABLE_IN_SOURCE_ONLY",
            suitability="CONDITIONAL",
            limitations="Requires phase segmentation and candidate chronology.",
            missing_input=common_missing,
            recommended_action="Derive from raw after unit attestation",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="efc_cycle_index",
            required_inputs="cycle sequence, capacity",
            verified_units="EFC; candidate cycle id",
            derivation_method="Candidate chronology from instrument Date/Time fields; sum |Ah| / (2*1.25 Ah) for EFC",
            input_availability="AVAILABLE_IN_CANDIDATE_CHRONOLOGY",
            suitability="CONDITIONAL",
            limitations="Audit assigns no global_cycle_id; the candidate order is reconstructed, not audited. 1,702 observed records do not equal 1,700 filename checkpoints.",
            missing_input="Audited global cycle identifier",
            recommended_action="Use candidates/B6_candidate_cycle_chronology.csv and re-validate order before training",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="lagged_soh",
            required_inputs="prior-cycle SOH labels",
            verified_units="%",
            derivation_method="Lag-1+ SOH with explicit gap flags; never same-cycle SOH",
            input_availability="AVAILABLE_IN_CANDIDATE_CHRONOLOGY",
            suitability="READY_WITH_CAVEAT",
            limitations="Depends on the candidate chronology; session offsets at workbook boundaries.",
            missing_input="Audited global cycle identifier",
            recommended_action="Lag only within the candidate order; exclude the 5 unlabelled cycles",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
        dict(
            feature_name="cycle_start_timestamp",
            required_inputs="per-sheet Date/Time header or File List",
            verified_units="datetime (local, no timezone)",
            derivation_method="Instrument-recorded Date/Time; File List sub-second value preferred where indexed",
            input_availability="AVAILABLE_IN_CANDIDATE_CHRONOLOGY",
            suitability="READY_WITH_CAVEAT",
            limitations="1,336 sheets come from header only; header time has 1 s resolution and no timezone.",
            missing_input="Timezone-aware timestamp",
            recommended_action="Use for ordering and calendar-age features, not for absolute-time science",
            change_vs_b1_b5_report="NEW: B6 added",
        ),
    ]
    for row in rows:
        row["cell_id"] = base
    return rows


def build_coverage() -> tuple[list[dict], list[str]]:
    old = pd.read_csv(OLD_COVERAGE)
    rename = {
        "row_count": "master_row_count",
        "observed_cycle_count": "master_cycle_count",
        "labelled_cycle_count": "master_labelled_cycle_count",
    }
    rows: list[dict] = []
    notes: list[str] = []
    for record in old.to_dict(orient="records"):
        row = {rename.get(k, k): v for k, v in record.items()}
        row["source_cycle_records"] = None
        row["source_labelled_records"] = None
        row["change_vs_b1_b5_report"] = "Unchanged"

        if row["cell_id"] == "BATTERY_1":
            row["source_cycle_records"] = 1460
            row["source_labelled_records"] = 1456
        elif row["cell_id"] == "BATTERY_3":
            row["source_cycle_records"] = 1979
            row["source_labelled_records"] = 1971
        elif row["cell_id"] == "BATTERY_4":
            row["source_cycle_records"] = 1577
            row["source_labelled_records"] = 0
        elif row["cell_id"] == "BATTERY_5":
            row["source_cycle_records"] = 1499
            row["source_labelled_records"] = 1492
        elif row["cell_id"] == "BATTERY_2":
            row.update(
                {
                    "source_path": (
                        "DATA/B2_Discharge1C_Charge0.5C.zip (raw); "
                        "DATA/B2_LFP_Member1_audit.zip (derived)"
                    ),
                    "source_status": "SOURCE_VERIFIED_RAW_AND_DERIVED",
                    "source_verified": (
                        "YES - 20 workbooks / 1,506 cycle sheets; all 20 workbook SHA-256 match the derived "
                        "export manifest; cycles 50 and 1506 re-derived from raw to <1e-8 Ah"
                    ),
                    "raw_data_available": "YES - supplied archive, 20 xlsx",
                    "derived_data_available": "YES - cycle_features 1,506x255, qv_curves, contiguous splits, raw_integrity",
                    "source_cycle_records": 1506,
                    "source_labelled_records": 1502,
                    "soh_provenance": (
                        "q_discharge_ah / 1.25 Ah rated (nominal); soh_initial_ref = retention vs first valid cycle"
                    ),
                    "data_quality_issues": (
                        "Raw verified. 1,506 cycles, 1,502 capacity labels, 1,496 in analysis. 10 flagged "
                        "invalid (4 null capacity: 41/972/998/1185; incomplete: 101/151/999/1186/1411/1412). "
                        "Instrument channel switches Un1Ch8->Un1Ch2 at cycle 999 with a ~0.9% capacity step. "
                        "160 cycles have inferred provenance. Label columns must never be features."
                    ),
                    "independent_validation_suitability": "SUITABLE_FOR_INDEPENDENT_VALIDATION",
                    "blocker": "None for cycle-level validation; channel switch needs a covariate; no measured BOL",
                    "recommended_action": (
                        "Use candidates/B2_candidate_cycle_labels.csv (raw-verified), the provided "
                        "contiguous splits, and soh_initial_ref_pct as the within-cell target; exclude "
                        "or flag cycles 999-1015; never use label columns as inputs"
                    ),
                    "change_vs_b1_b5_report": (
                        "CHANGED: ONLY_DERIVED_DATA_AVAILABLE -> SOURCE_VERIFIED_RAW_AND_DERIVED; "
                        "raw telemetry features unlocked; suitability was already SUITABLE"
                    ),
                }
            )
            notes.append("B2: raw archive verified byte-identical; status upgraded")
        rows.append(row)

    rows.append(
        {
            "cell_id": "BATTERY_6",
            "dataset_id": "LFP_MultiStat_B6",
            "source_path": "DATA/B6_Discharge0.5C_Charge2C.zip (raw); AUDIT/LFP_B6_Audit_Package (1).zip (audit)",
            "source_status": "SOURCE_VERIFIED_NOT_INGESTED",
            "source_verified": (
                "YES - 19 workbooks / 1,702 cycle-like sheets; all 19 workbook SHA-256 match the audit "
                "manifest; cycles re-derived from raw at Cycle10 and Sheet10 to 1e-6 Ah"
            ),
            "raw_data_available": "YES - supplied archive, 19 xlsx",
            "derived_data_available": "YES - audit cycle summary 1,702x65; candidate chronology in AUDIT/candidates/",
            "master_row_count": 0,
            "master_cycle_count": 0,
            "master_labelled_cycle_count": 0,
            "source_cycle_records": 1702,
            "source_labelled_records": 1697,
            "soh_provenance": "capacity_ah / 1.25 Ah rated (nominal); 1,697 conditional labels, 5 screened out",
            "current_unit_status": (
                "CONDITIONAL_AMPERES; numeric values match 0.625 A discharge / 2.4 A charge setpoints but the "
                "header says I(A/cm2) with Surface Area=0 (1,701 verified-with-conditions, 1 underdetermined)"
            ),
            "voltage_available": "YES - 1 Hz",
            "current_available": "YES - 1 Hz, unit conditional",
            "time_available": "YES - 1 Hz; per-cycle reset, cumulative within session",
            "temperature_available": "NO - Aux Count 0; no thermal channel",
            "data_quality_issues": (
                "No global_cycle_id in the audit; chronology reconstructed from per-sheet Date/Time and validated "
                "(0 duplicates, 0 reversals, checkpoint blocks contiguous). 1686 named + 16 generic (unindexed) "
                "sheets; Cycle_0050 has 51 and Cycle_0700 has 101 records, so filename checkpoints are not exact "
                "cycle numbers. Two channels (Un1Ch1 201 cycles, Un1Ch6 1,485); 378 time-gap flags; 2 material "
                "voltage excursions >20 mV; 5 excluded cycles (short discharge, 2 terminal-voltage, 1 no negative "
                "current, Sheet1). Session-boundary capacity steps of 0.5-0.8 pp."
            ),
            "independent_validation_suitability": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
            "blocker": (
                "Current unit attestation; no measured BOL; candidate chronology is reconstructed, not audited; "
                "session offsets"
            ),
            "recommended_action": (
                "Use candidates/B6_candidate_cycle_chronology.csv; attest current unit against setpoints; "
                "treat workbook sessions as covariates or de-trend; re-validate chronology before training; "
                "do not fill the 5 unlabelled cycles"
            ),
            "change_vs_b1_b5_report": (
                "NEW: B6 added (previously noted as out-of-scope audit-only); raw verified, chronology reconstructed"
            ),
        }
    )
    notes.append("B6: added as a candidate independent cell with reconstructed chronology")
    return rows, notes


def build_features() -> tuple[list[dict], list[str]]:
    old = pd.read_csv(OLD_FEATURES)
    rows: list[dict] = []
    notes: list[str] = []
    for record in old.to_dict(orient="records"):
        row = dict(record)
        row["change_vs_b1_b5_report"] = "Unchanged"
        if row["cell_id"] == "BATTERY_2" and row["feature_name"] in B2_FEATURE_UPDATES:
            row.update(B2_FEATURE_UPDATES[row["feature_name"]])
        if pd.isna(row.get("missing_input")):
            row["missing_input"] = ""
        rows.append(row)
    notes.append(f"B2: {len(B2_FEATURE_UPDATES)} feature rows updated for raw availability")

    b6 = _b6_feature_rows()
    rows.extend(b6)
    notes.append(f"B6: {len(b6)} feature rows added")
    return rows, notes


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=csv.QUOTE_MINIMAL, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: ("" if row.get(c) is None else row.get(c)) for c in columns})


def main() -> int:
    coverage, coverage_notes = build_coverage()
    features, feature_notes = build_features()
    write_csv(NEW_COVERAGE, coverage, COVERAGE_COLUMNS)
    write_csv(NEW_FEATURES, features, FEATURE_COLUMNS)
    CHANGE_LOG.write_text(
        json.dumps(
            {
                "supersedes": [OLD_COVERAGE.name, OLD_FEATURES.name],
                "coverage_rows": len(coverage),
                "feature_rows": len(features),
                "cells": sorted({r["cell_id"] for r in coverage}),
                "changes": coverage_notes + feature_notes,
                "rule": "Earlier B1-B5 files are preserved unchanged; every changed/new row is marked.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"coverage rows: {len(coverage)} ({sorted({r['cell_id'] for r in coverage})})")
    print(f"feature rows: {len(features)}")
    for note in coverage_notes + feature_notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
