"""
Member 1 - Read-only data readiness audit for the EV battery master table.

Purpose
-------
Independently re-verify the canonical master table (structure, coverage, nulls,
duplicate keys, timing, cycle gaps, label constancy/provenance, quality flags,
current phase, sampling and label-leakage) and emit a machine-readable report.

Safety
------
* Never imports or executes master_table_build.py.
* Never writes to DATA/, AUDIT/*.zip or the canonical master_table.* files.
* Only writes inside AUDIT/member1_audit_outputs/.
* Pure read-only pandas operations on the canonical parquet.

Run:
    python AUDIT/member1_audit.py

Exit code 0 when every master-contract assertion passes, 1 otherwise.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MASTER_PARQUET = REPO_ROOT / "master_table.parquet"
MASTER_CSV = REPO_ROOT / "master_table.csv"
B5_AUDIT_ZIP = REPO_ROOT / "AUDIT" / "B5.zip"
B1_AUDIT_ZIP = REPO_ROOT / "AUDIT" / "B1_LFP_Audit_Output.zip"
OUT_DIR = Path(__file__).resolve().parent / "member1_audit_outputs"

# --- Verified contract (from 2026-10-27 build, re-checked 2026-10-29) ---------
EXPECTED_SHA256_PARQUET = "6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb"
EXPECTED_SHA256_CSV = "9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b"
EXPECTED_ROWS = 515_270
MASTER_SCHEMA = [
    "chemistry", "dataset_id", "cell_id", "protocol_id", "cycle_id", "time_sec",
    "voltage_v", "current_a", "temp_c", "capacity_ah", "soh_pct", "quality_flag",
    "source_file",
]
EXPECTED_DATASETS = {
    "LFP_MultiStat_B1": {"chemistry": "LFP", "rows": 4_789, "cells": 1, "cell_cycle_pairs": 1},
    "LFP_MultiStat_B5": {"chemistry": "LFP", "rows": 333_344, "cells": 1, "cell_cycle_pairs": 250},
    "NASA_Ames_PCoE": {"chemistry": "NMC", "rows": 135_436, "cells": 3, "cell_cycle_pairs": 468},
    "Zenodo_IonTech_NaIon": {"chemistry": "Na-ion", "rows": 41_701, "cells": 4, "cell_cycle_pairs": 100},
}
B5_EXPECTED_BLOCKS = [(1, 100), (151, 200), (401, 500)]
B5_EXPECTED_NULL_SOH_CYCLES = {40}
NOMINAL_CAPACITY_AH = {
    "LFP_MultiStat_B1": 1.25,
    "LFP_MultiStat_B5": 1.25,
    "NASA_Ames_PCoE": 2.00,
    "Zenodo_IonTech_NaIon": 1.00,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contiguous_blocks(ids: list[int]) -> list[tuple[int, int]]:
    if not ids:
        return []
    blocks, start, prev = [], ids[0], ids[0]
    for value in ids[1:]:
        if value == prev + 1:
            prev = value
        else:
            blocks.append((start, prev))
            start = prev = value
    blocks.append((start, prev))
    return blocks


def load_master(path: Path = MASTER_PARQUET) -> pd.DataFrame:
    return pd.read_parquet(path)


def check_schema(df: pd.DataFrame) -> dict:
    return {
        "columns": list(df.columns),
        "columns_match": list(df.columns) == MASTER_SCHEMA,
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }


def dataset_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for dataset_id, group in df.groupby("dataset_id"):
        summary[dataset_id] = {
            "chemistry": sorted(group["chemistry"].unique().tolist()),
            "protocol_id": sorted(group["protocol_id"].unique().tolist()),
            "rows": int(len(group)),
            "cells": sorted(group["cell_id"].unique().tolist()),
            "cell_count": int(group["cell_id"].nunique()),
            "cycle_id_count": int(group["cycle_id"].nunique()),
            "cell_cycle_pairs": int(group.groupby(["cell_id", "cycle_id"]).ngroups),
            "nulls": {c: int(group[c].isna().sum()) for c in df.columns if group[c].isna().any()},
            "rows_per_cycle_min": int(group.groupby(["cell_id", "cycle_id"]).size().min()),
            "rows_per_cycle_median": float(group.groupby(["cell_id", "cycle_id"]).size().median()),
            "rows_per_cycle_max": int(group.groupby(["cell_id", "cycle_id"]).size().max()),
        }
    return summary


def duplicate_and_timing_checks(df: pd.DataFrame) -> dict:
    result = {
        "duplicate_composite_keys": int(
            df.duplicated(subset=["dataset_id", "cell_id", "cycle_id", "time_sec"]).sum()
        ),
    }
    per_dataset = {}
    for dataset_id, group in df.groupby("dataset_id"):
        neg_dt = zero_dt = dup_times = 0
        for _, cycle in group.groupby(["cell_id", "cycle_id"]):
            times = cycle["time_sec"].to_numpy()
            diffs = np.diff(times)
            neg_dt += int((diffs < 0).sum())
            zero_dt += int((diffs == 0).sum())
            dup_times += int(len(times) - len(np.unique(times)))
        per_dataset[dataset_id] = {
            "negative_dt": neg_dt,
            "zero_dt": zero_dt,
            "duplicate_timestamps_within_cycles": dup_times,
        }
    result["per_dataset"] = per_dataset
    return result


def cycle_gap_report(df: pd.DataFrame) -> dict:
    report = {}
    for (dataset_id, cell_id), group in df.groupby(["dataset_id", "cell_id"]):
        ids = sorted(int(v) for v in group["cycle_id"].unique())
        expected_span = set(range(ids[0], ids[-1] + 1))
        report[f"{dataset_id}::{cell_id}"] = {
            "min": ids[0],
            "max": ids[-1],
            "observed_count": len(ids),
            "blocks": _contiguous_blocks(ids),
            "missing_ids": sorted(expected_span - set(ids)),
        }
    return report


def label_constancy(df: pd.DataFrame) -> dict:
    report = {}
    for dataset_id, group in df.groupby("dataset_id"):
        per_cycle = group.groupby(["cell_id", "cycle_id"]).agg(
            cap_unique=("capacity_ah", "nunique"),
            soh_unique=("soh_pct", "nunique"),
            cap_null_rows=("capacity_ah", lambda s: int(s.isna().sum())),
            soh_null_rows=("soh_pct", lambda s: int(s.isna().sum())),
        )
        report[dataset_id] = {
            "max_unique_capacity_per_cycle": int(per_cycle["cap_unique"].max()),
            "max_unique_soh_per_cycle": int(per_cycle["soh_unique"].max()),
            "cycles_with_all_capacity_null": int((per_cycle["cap_null_rows"] > 0).sum()),
            "cycles_with_all_soh_null": int((per_cycle["soh_null_rows"] > 0).sum()),
            "partial_label_cycles": sorted(
                {
                    int(cyc)
                    for (cell, cyc), row in per_cycle.iterrows()
                    if (row["cap_null_rows"] > 0) ^ (row["soh_null_rows"] > 0)
                }
            ),
        }
    return report


def label_provenance(df: pd.DataFrame) -> dict:
    report = {"nominal_rating_implied": {}, "soh_residual_max_abs": {}, "soh_semantics": {}}
    for dataset_id, group in df.groupby("dataset_id"):
        rated = NOMINAL_CAPACITY_AH[dataset_id]
        paired = group.dropna(subset=["capacity_ah", "soh_pct"])
        if len(paired):
            implied = paired["capacity_ah"] / paired["soh_pct"] * 100.0
            report["nominal_rating_implied"][dataset_id] = {
                "min": float(implied.min()),
                "median": float(implied.median()),
                "max": float(implied.max()),
            }
            residual = (
                paired["soh_pct"] - paired["capacity_ah"] / rated * 100.0
            ).abs()
            report["soh_residual_max_abs"][dataset_id] = float(residual.max())
        report["soh_semantics"][dataset_id] = (
            f"capacity_ah / {rated} Ah manufacturer rating * 100 "
            "(NOT capacity retention vs measured beginning-of-life)"
        )
    report["partial_labels"] = {
        dataset_id: sorted(
            {
                int(cyc)
                for cyc, c in df[df["dataset_id"] == dataset_id]
                .groupby("cycle_id")
                .agg(cap=("capacity_ah", lambda s: s.notna().all()),
                     soh=("soh_pct", lambda s: s.notna().all()))
                .iterrows()
                if bool(c["cap"]) ^ bool(c["soh"])
            }
        )
        for dataset_id in df["dataset_id"].unique()
    }
    return report


def quality_flags(df: pd.DataFrame) -> dict:
    return {str(k): int(v) for k, v in df["quality_flag"].value_counts().items()}


def current_phase(df: pd.DataFrame) -> dict:
    report = {}
    for dataset_id, group in df.groupby("dataset_id"):
        current = group["current_a"]
        report[dataset_id] = {
            "fraction_negative_discharge": float((current < 0).mean()),
            "fraction_positive_charge": float((current > 0).mean()),
            "fraction_zero": float((current == 0).mean()),
            "min_a": float(current.min()),
            "max_a": float(current.max()),
            "unique_values": int(current.nunique()),
            "interpretation": (
                "discharge-only telemetry"
                if (current > 0.05).mean() == 0
                else "mixed charge/discharge telemetry"
            ),
        }
    return report


def time_semantics(df: pd.DataFrame) -> dict:
    """Whether time_sec resets at each cycle or is cumulative source time."""
    report = {}
    for dataset_id, group in df.groupby("dataset_id"):
        per_cycle_min = group.groupby(["cell_id", "cycle_id"])["time_sec"].min()
        resets = float((per_cycle_min.abs() < 1e-6).mean())
        report[dataset_id] = {
            "cycle_start_median_s": float(per_cycle_min.median()),
            "fraction_cycles_starting_near_zero": resets,
            "resets_each_cycle": resets > 0.99,
            "note": (
                "within-cycle elapsed time"
                if resets > 0.99
                else "cumulative source/workbook time; NOT reset per cycle - "
                "subtract the per-cycle minimum before using time-to-event features"
            ),
        }
    return report


def temperature_coverage(df: pd.DataFrame) -> dict:
    report = {}
    for dataset_id, group in df.groupby("dataset_id"):
        nonnull = group["temp_c"].notna().sum()
        report[dataset_id] = {
            "rows": int(len(group)),
            "temp_nonnull_rows": int(nonnull),
            "temp_null_rows": int(len(group) - nonnull),
            "temp_min_c": None if nonnull == 0 else float(group["temp_c"].min()),
            "temp_max_c": None if nonnull == 0 else float(group["temp_c"].max()),
        }
    return report


def leakage_assessment(df: pd.DataFrame) -> dict:
    rows_per_cycle = df.groupby(["dataset_id", "cell_id", "cycle_id"]).size()
    return {
        "label_is_cycle_level": True,
        "broadcast_factor_median": float(rows_per_cycle.median()),
        "broadcast_factor_max": int(rows_per_cycle.max()),
        "note": (
            "capacity_ah and soh_pct are cycle-level scalars repeated on every "
            "telemetry row of the cycle. Treating telemetry rows as independent "
            "samples leaks the label; collapse to one row per (cell, cycle) for "
            "trajectory/regression targets."
        ),
    }


def b5_source_crosscheck(df: pd.DataFrame) -> dict | None:
    if not B5_AUDIT_ZIP.exists():
        return None
    with zipfile.ZipFile(B5_AUDIT_ZIP) as archive:
        summary = pd.read_csv(archive.open("B5_Cycle_Summary.csv"))
    b5 = df[df["dataset_id"] == "LFP_MultiStat_B5"]
    cycles = set(int(c) for c in b5["cycle_id"].unique())
    subset = summary[summary["global_cycle_id_nominal_checkpoint"].isin(cycles)]
    trapz = dict(zip(subset["global_cycle_id_nominal_checkpoint"], subset["dis_q_ah_trapz_CALCULATED"]))
    usable = dict(
        zip(
            subset["global_cycle_id_nominal_checkpoint"],
            subset["discharge_capacity_ah_CALCULATED_CONDITIONAL_not_ground_truth"],
        )
    )
    mismatch_trapz, mismatch_usable = [], []
    for cycle_id, group in b5.groupby("cycle_id"):
        cap = group["capacity_ah"].iloc[0]
        if not np.isclose(cap, trapz.get(cycle_id, np.nan), equal_nan=True):
            mismatch_trapz.append(int(cycle_id))
        if not np.isclose(cap, usable.get(cycle_id, np.nan), equal_nan=True):
            mismatch_usable.append(int(cycle_id))
    excluded = subset[subset["modelling_use_capacity_soh"] == "EXCLUDE"]
    return {
        "master_cycles": len(cycles),
        "master_capacity_equals_raw_trapz_cycles": len(cycles) - len(mismatch_trapz),
        "master_capacity_differs_from_usable_conditional_cycles": mismatch_usable,
        "source_excluded_cycles": sorted(int(c) for c in excluded["global_cycle_id_nominal_checkpoint"]),
        "source_excluded_status": sorted(excluded["cycle_status"].unique().tolist()),
        "source_capacity_ground_truth": sorted(subset["capacity_ground_truth"].unique().tolist()),
        "finding": (
            "The builder maps capacity_ah from the raw trapezoidal throughput "
            "column, which keeps the partial discharge value for interrupted "
            "cycles. Only cycle 40 is affected; the audit's usable-capacity column "
            "is NaN there and its soh is excluded."
        ),
    }


def b1_source_crosscheck(df: pd.DataFrame) -> dict | None:
    if not B1_AUDIT_ZIP.exists():
        return None
    with zipfile.ZipFile(B1_AUDIT_ZIP) as archive:
        cycle_level = pd.read_csv(
            archive.open("B1_LFP_Audit_Output/LFP_staging_cycle_level.csv")
        )
        sample = pd.read_csv(
            archive.open("B1_LFP_Audit_Output/LFP_staging_sample_raw_rows.csv")
        )
    b1 = df[df["dataset_id"] == "LFP_MultiStat_B1"]
    collapsed = dict(zip(cycle_level["cycle_checkpoint"], cycle_level["capacity_ah"]))
    raw_checkpoints = sorted(int(v) for v in sample["cycle_checkpoint"].unique())
    raw_est = sorted(int(v) for v in sample["global_cycle_est"].unique())
    checkpoint = raw_checkpoints[0] if raw_checkpoints else None
    per_cycle = cycle_level[cycle_level["cycle_checkpoint"] == checkpoint]
    same_est = per_cycle[per_cycle["global_cycle_est"].isin(raw_est)]
    return {
        "telemetry_checkpoints": raw_checkpoints,
        "telemetry_global_cycle_est": raw_est,
        "master_cycle_id": sorted(int(v) for v in b1["cycle_id"].unique()),
        "master_capacity_ah": float(b1["capacity_ah"].iloc[0]),
        "audit_capacity_at_telemetry_cycle_ah": (
            None if same_est.empty else float(same_est["capacity_ah"].iloc[0])
        ),
        "audit_capacity_last_cycle_of_checkpoint_ah": (
            None if per_cycle.empty else float(per_cycle["capacity_ah"].iloc[-1])
        ),
        "collapsed_lookup_value_ah": float(collapsed.get(checkpoint, np.nan)),
        "finding": (
            "The builder builds a dict(checkpoint -> capacity) from a table where "
            "each checkpoint spans many cycles. The lookup collapses to the LAST "
            "cycle of that checkpoint, so the sampled telemetry is labelled with a "
            "different cycle's capacity. B1 also mixes charge and discharge phases."
        ),
    }


def contract_assertions(df: pd.DataFrame, checksums: dict, checks: dict) -> list[dict]:
    assertions = []

    def add(name: str, passed: bool, detail) -> None:
        assertions.append({"assertion": name, "passed": bool(passed), "detail": detail})

    add("parquet_sha256_unchanged", checksums["parquet"] == EXPECTED_SHA256_PARQUET, checksums["parquet"])
    add("csv_sha256_unchanged", checksums["csv"] == EXPECTED_SHA256_CSV, checksums["csv"])
    add("row_count", len(df) == EXPECTED_ROWS, int(len(df)))
    add("schema_exact_13_columns", checks["schema"]["columns_match"], checks["schema"]["columns"])
    add(
        "no_duplicate_composite_keys",
        checks["duplicates_timing"]["duplicate_composite_keys"] == 0,
        checks["duplicates_timing"]["duplicate_composite_keys"],
    )
    for dataset_id, expected in EXPECTED_DATASETS.items():
        summary = checks["datasets"].get(dataset_id, {})
        add(
            f"{dataset_id}::rows",
            summary.get("rows") == expected["rows"],
            summary.get("rows"),
        )
        add(
            f"{dataset_id}::cell_cycle_pairs",
            summary.get("cell_cycle_pairs") == expected["cell_cycle_pairs"],
            summary.get("cell_cycle_pairs"),
        )
        add(
            f"{dataset_id}::chemistry",
            summary.get("chemistry") == [expected["chemistry"]],
            summary.get("chemistry"),
        )
    b5_blocks = checks["cycle_gaps"].get("LFP_MultiStat_B5::BATTERY_5", {}).get("blocks", [])
    add(
        "b5_cycle_blocks",
        [tuple(b) for b in b5_blocks] == B5_EXPECTED_BLOCKS,
        b5_blocks,
    )
    b5_null_soh = sorted(
        int(c)
        for c in df[(df["dataset_id"] == "LFP_MultiStat_B5") & (df["soh_pct"].isna())]["cycle_id"].unique()
    )
    add("b5_null_soh_cycles", set(b5_null_soh) == B5_EXPECTED_NULL_SOH_CYCLES, b5_null_soh)
    b5_partial = checks["label_provenance"]["partial_labels"].get("LFP_MultiStat_B5", [])
    add("b5_cycle40_partial_label", b5_partial == [40], b5_partial)
    for dataset_id in ("LFP_MultiStat_B1", "LFP_MultiStat_B5", "Zenodo_IonTech_NaIon"):
        cov = checks["temperature"][dataset_id]
        add(f"{dataset_id}::temperature_unmeasured", cov["temp_nonnull_rows"] == 0, cov["temp_nonnull_rows"])
    nasa = checks["temperature"]["NASA_Ames_PCoE"]
    add("nasa_temperature_measured", nasa["temp_null_rows"] == 0, nasa["temp_null_rows"])
    add(
        "no_time_reversals_or_duplicate_timestamps",
        all(
            v["negative_dt"] == 0
            and v["zero_dt"] == 0
            and v["duplicate_timestamps_within_cycles"] == 0
            for v in checks["duplicates_timing"]["per_dataset"].values()
        ),
        checks["duplicates_timing"]["per_dataset"],
    )
    return assertions


def candidate_cycle_labels(df: pd.DataFrame, checks: dict) -> pd.DataFrame:
    """One row per (dataset, cell, cycle). Candidate only; never overwrites master."""
    b5_bad = set(checks["label_provenance"]["partial_labels"].get("LFP_MultiStat_B5", []))
    rows = []
    for (dataset_id, cell_id, cycle_id), group in df.groupby(
        ["dataset_id", "cell_id", "cycle_id"], sort=True
    ):
        cap = group["capacity_ah"].iloc[0]
        soh = group["soh_pct"].iloc[0]
        cap_status = "VALID"
        soh_status = "VALID"
        target_confidence = "HIGH"
        label_cycle_mismatch = False
        notes = ""
        if dataset_id == "LFP_MultiStat_B5":
            cap_status = soh_status = "CONDITIONAL_NOMINAL_RATING"
            target_confidence = "CONDITIONAL"
            notes = "charge-balance discrepancy; capacity conditional, not ground truth"
            if int(cycle_id) in b5_bad:
                cap_status = "INVALID_PARTIAL_DISCHARGE"
                soh_status = "MISSING_PARTIAL_DISCHARGE"
                target_confidence = "INVALID"
                notes = "interrupted discharge; capacity is partial throughput; use excluded"
        elif dataset_id == "LFP_MultiStat_B1":
            target_confidence = "INVALID"
            cap_status = soh_status = "LABEL_CYCLE_MISMATCH"
            label_cycle_mismatch = True
            notes = (
                "telemetry sampled from checkpoint-50 early cycle; master label is "
                "the last cycle of that checkpoint. Audit per-cycle value differs."
            )
        elif dataset_id == "NASA_Ames_PCoE":
            cap_status = soh_status = "MEASURED"
            notes = "capacity from PCoE discharge test; soh vs 2.0 Ah rating (can exceed 100%)"
        elif dataset_id == "Zenodo_IonTech_NaIon":
            cap_status = soh_status = "MEASURED_EXPORTED"
            notes = "discharge capacity from pkl; soh vs 1.0 Ah rating; only first 25 cycles ingested"
        if pd.isna(cap):
            cap_status = "MISSING"
        if pd.isna(soh):
            soh_status = "MISSING"
        rows.append(
            {
                "dataset_id": dataset_id,
                "cell_id": cell_id,
                "chemistry": group["chemistry"].iloc[0],
                "cycle_id": int(cycle_id),
                "n_telemetry_rows": int(len(group)),
                "capacity_ah": None if pd.isna(cap) else float(cap),
                "soh_pct": None if pd.isna(soh) else float(soh),
                "soh_semantics": "capacity_fraction_of_manufacturer_rated_nominal",
                "capacity_label_status": cap_status,
                "soh_label_status": soh_status,
                "label_cycle_mismatch": label_cycle_mismatch,
                "target_confidence": target_confidence,
                "usable_as_ml_target": target_confidence in {"HIGH", "CONDITIONAL"},
                "notes": notes,
            }
        )
    return pd.DataFrame(rows)


def run_audit() -> dict:
    checksums = {
        "parquet": sha256_file(MASTER_PARQUET),
        "csv": sha256_file(MASTER_CSV),
    }
    df = load_master()
    checks = {
        "schema": check_schema(df),
        "datasets": dataset_summary(df),
        "duplicates_timing": duplicate_and_timing_checks(df),
        "cycle_gaps": cycle_gap_report(df),
        "label_constancy": label_constancy(df),
        "label_provenance": label_provenance(df),
        "quality_flags": quality_flags(df),
        "current_phase": current_phase(df),
        "time_semantics": time_semantics(df),
        "temperature": temperature_coverage(df),
        "leakage": leakage_assessment(df),
        "b5_source_crosscheck": b5_source_crosscheck(df),
        "b1_source_crosscheck": b1_source_crosscheck(df),
    }
    assertions = contract_assertions(df, checksums, checks)
    report = {
        "audit_timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "master_parquet": str(MASTER_PARQUET),
        "checksums_sha256": checksums,
        "global_rows": int(len(df)),
        "global_columns": int(df.shape[1]),
        "checks": checks,
        "assertions": assertions,
        "all_assertions_passed": all(a["passed"] for a in assertions),
    }
    return report


def write_outputs(report: dict, df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audit_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    labels = candidate_cycle_labels(df, report["checks"])
    labels.to_csv(OUT_DIR / "candidate_cycle_labels.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def main() -> int:
    report = run_audit()
    df = load_master()
    write_outputs(report, df)
    passed = sum(1 for a in report["assertions"] if a["passed"])
    total = len(report["assertions"])
    print(f"master_table.parquet sha256 : {report['checksums_sha256']['parquet']}")
    print(f"rows x columns              : {report['global_rows']} x {report['global_columns']}")
    print(f"assertions passed           : {passed}/{total}")
    for assertion in report["assertions"]:
        if not assertion["passed"]:
            print(f"  FAILED: {assertion['assertion']} -> {assertion['detail']}")
    print(f"report written              : {OUT_DIR / 'audit_report.json'}")
    print(f"candidate labels written    : {OUT_DIR / 'candidate_cycle_labels.csv'}")
    return 0 if report["all_assertions_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
