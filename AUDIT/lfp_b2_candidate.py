"""
Member 1 - B2 candidate cycle-label table (raw-verified, isolated candidate output).

The B2 raw workbooks were previously absent from the workspace, so B2 was usable
only through the derived export ``DATA/B2_LFP_Member1_audit.zip``. The supplied
``B2_Discharge1C_Charge0.5C.zip`` now provides the raw telemetry. This script

1. verifies the new archive against the 20 per-workbook SHA-256 values recorded in
   the derived export's ``raw_integrity.csv`` (byte identity, not just presence),
2. re-derives two cycle capacities from raw and compares them with the export,
3. projects the 255-column export down to a labelled candidate table with explicit
   provenance, preserving every label and flag unchanged.

No label is recomputed, filled or overwritten. Existing flags are carried through.

Safety
------
* Reads raw archive + derived export only; writes only inside ``AUDIT/candidates/``.
* Aborts if any input hash changes during the run.

Run:
    python AUDIT/lfp_b2_candidate.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfp_b1_b5_inventory import read_cycle_sheet, segment_cycle  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
OUT_DIR = AUDIT_DIR / "candidates"

B2_RAW = REPO_ROOT / "DATA" / "B2_Discharge1C_Charge0.5C.zip"
B2_DERIVED = REPO_ROOT / "DATA" / "B2_LFP_Member1_audit.zip"
B2_RAW_SHA256 = "b75c36d96abd2ae17858eda570dd6627ef06433bd88ac55faf73d688cbc10279"
B2_DERIVED_SHA256 = "54cd6113506e7b11b5147739d24b412b3b279f0c356e63bcf67de544a8ce169d"

RAW_MEMBER_DIR = "B2_Discharge1C_Charge0.5C"
RAW_INTEGRITY = "LFP_Member1_audit/data_derived/raw_integrity.csv"
CYCLE_FEATURES = "LFP_Member1_audit/exports/cycle_features.parquet"

CANDIDATE_COLUMNS = [
    "cycle_index",
    "source_file",
    "worksheet",
    "session_global",
    "timestamp",
    "channel",
    "provenance",
    "valid_cycle",
    "in_analysis",
    "q_rated_ah",
    "q_discharge_ah",
    "q_charge_ah",
    "soh_rated_pct",
    "soh_initial_ref_pct",
    "coulombic_efficiency",
    "ce_reliable",
    "discharge_reached_cutoff",
    "charge_cv_terminated",
    "cycle_duration_s",
    "discharge_duration_s",
    "charge_duration_s",
    "source_archive",
    "source_sha256",
    "source_table",
    "raw_manifest_verified",
    "label_status",
    "notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_member(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_workbooks() -> dict:
    with zipfile.ZipFile(B2_DERIVED) as derived:
        integrity = pd.read_csv(io.BytesIO(derived.read(RAW_INTEGRITY)))
    expected = dict(zip(integrity["name"], integrity["sha256"]))
    expected_bytes = dict(zip(integrity["name"], integrity["bytes"]))

    results = {}
    with zipfile.ZipFile(B2_RAW) as archive:
        members = {Path(n).name: n for n in archive.namelist() if n.lower().endswith(".xlsx")}
        missing = sorted(set(expected) - set(members))
        extra = sorted(set(members) - set(expected))
        for name, member in members.items():
            actual = sha256_member(archive, member)
            info = archive.getinfo(member)
            results[name] = {
                "sha256": actual,
                "sha256_matches_export": actual == expected.get(name),
                "bytes": info.file_size,
                "bytes_match_export": info.file_size == expected_bytes.get(name),
            }
    return {
        "workbooks": len(results),
        "all_hashes_match": all(r["sha256_matches_export"] for r in results.values()),
        "all_sizes_match": all(r["bytes_match_export"] for r in results.values()),
        "missing_from_raw": missing,
        "extra_in_raw": extra,
        "per_workbook": results,
    }


def spot_check_cycles(features: pd.DataFrame) -> list[dict]:
    checks = []
    for cycle_index in (50, 1506):
        row = features.loc[features["cycle_index"] == cycle_index].iloc[0]
        member = f"{RAW_MEMBER_DIR}/{row['file_id']}.xlsx"
        segments = segment_cycle(read_cycle_sheet(B2_RAW, member, row["sheet"]))
        raw_q = segments["discharge"]["capacity_ah"]
        table_q = float(row["q_discharge_ah"])
        checks.append(
            {
                "cycle_index": int(cycle_index),
                "source_file": row["file_id"],
                "worksheet": row["sheet"],
                "raw_q_discharge_ah": round(raw_q, 6),
                "table_q_discharge_ah": round(table_q, 6),
                "abs_diff_ah": round(abs(raw_q - table_q), 9),
                "matches": abs(raw_q - table_q) <= 1e-6,
            }
        )
    return checks


def build_candidate() -> tuple[pd.DataFrame, dict]:
    raw_hash = sha256_file(B2_RAW)
    derived_hash = sha256_file(B2_DERIVED)
    if raw_hash != B2_RAW_SHA256:
        raise RuntimeError(f"B2 raw archive hash mismatch: {raw_hash}")
    if derived_hash != B2_DERIVED_SHA256:
        raise RuntimeError(f"B2 derived archive hash mismatch: {derived_hash}")

    manifest = verify_raw_workbooks()
    if not manifest["all_hashes_match"] or manifest["missing_from_raw"] or manifest["extra_in_raw"]:
        raise RuntimeError("B2 raw archive does not match the derived export manifest")

    with zipfile.ZipFile(B2_DERIVED) as derived:
        features = pd.read_parquet(io.BytesIO(derived.read(CYCLE_FEATURES)))

    if len(features) != 1506 or not features["cycle_index"].is_monotonic_increasing:
        raise RuntimeError("B2 cycle_features structure changed")
    if features["cycle_index"].tolist() != list(range(1, 1507)):
        raise RuntimeError("B2 cycle_index is not contiguous 1..1506")

    checks = spot_check_cycles(features)
    if not all(c["matches"] for c in checks):
        raise RuntimeError("B2 raw re-derivation disagrees with the derived table")

    def status(row: pd.Series) -> str:
        if not bool(row["in_analysis"]):
            return "excluded_flagged"
        if pd.isna(row["soh_rated"]):
            return "missing_label"
        return "conditional_nominal_soh"

    candidate = pd.DataFrame(
        {
            "cycle_index": features["cycle_index"],
            "source_file": features["file_id"],
            "worksheet": features["sheet"],
            "session_global": features["session_global"],
            "timestamp": features["timestamp"],
            "channel": features["channel"],
            "provenance": features["provenance"],
            "valid_cycle": features["valid_cycle"],
            "in_analysis": features["in_analysis"],
            "q_rated_ah": features["q_rated_ah"],
            "q_discharge_ah": features["q_discharge_ah"],
            "q_charge_ah": features["q_charge_ah"],
            "soh_rated_pct": features["soh_rated"] * 100.0,
            "soh_initial_ref_pct": features["soh_initial_ref"] * 100.0,
            "coulombic_efficiency": features["coulombic_efficiency"],
            "ce_reliable": features["ce_reliable"],
            "discharge_reached_cutoff": features["discharge_reached_cutoff"],
            "charge_cv_terminated": features["charge_cv_terminated"],
            "cycle_duration_s": features["cycle_duration_s"],
            "discharge_duration_s": features["discharge_duration_s"],
            "charge_duration_s": features["charge_duration_s"],
        }
    )
    candidate["source_archive"] = "B2_Discharge1C_Charge0.5C.zip"
    candidate["source_sha256"] = raw_hash
    candidate["source_table"] = "DATA/B2_LFP_Member1_audit.zip::LFP_Member1_audit/exports/cycle_features.parquet"
    candidate["raw_manifest_verified"] = True
    candidate["label_status"] = features.apply(status, axis=1)
    candidate["notes"] = (
        "SOH nominal = q_discharge / 1.25 Ah rated x100; soh_initial_ref_pct is retention "
        "vs the first valid cycle. Cycle-level labels only. Un1Ch8->Un1Ch2 channel switch at "
        "cycle 999 is a measurement artefact; cycles 999-1015 are flagged."
    )

    evidence = {
        "metadata": {
            "title": "B2 candidate cycle labels, raw workbooks verified byte-identical",
            "canonical_dataset_modified": False,
            "raw_archive": str(B2_RAW),
            "raw_sha256": raw_hash,
            "derived_archive": str(B2_DERIVED.relative_to(REPO_ROOT)),
            "derived_sha256": derived_hash,
            "candidate_output": str((OUT_DIR / "B2_candidate_cycle_labels.csv").relative_to(REPO_ROOT)),
        },
        "raw_manifest": {k: v for k, v in manifest.items() if k != "per_workbook"},
        "spot_checks": checks,
        "rows": int(len(candidate)),
        "label_status_counts": candidate["label_status"].value_counts().to_dict(),
        "in_analysis": int(candidate["in_analysis"].sum()),
        "in_analysis_soh_rated_pct_range": [
            float(candidate.loc[candidate["in_analysis"], "soh_rated_pct"].min()),
            float(candidate.loc[candidate["in_analysis"], "soh_rated_pct"].max()),
        ],
        "in_analysis_soh_initial_ref_pct_range": [
            float(candidate.loc[candidate["in_analysis"], "soh_initial_ref_pct"].min()),
            float(candidate.loc[candidate["in_analysis"], "soh_initial_ref_pct"].max()),
        ],
        "excluded_outlier_note": (
            "The global minimum SOH (7.02%, cycle 1411) is an incomplete discharge already "
            "flagged valid_cycle=False / in_analysis=False; it is excluded from the ranges above."
        ),
        "limitations": [
            "SOH is nominal-capacity based, identical definition to B5/B6.",
            "No temperature channel; channel switch at cycle 999 is a measurement artefact.",
            "Labels are cycle-level and must be grouped by cycle_index before joining telemetry.",
        ],
    }
    return candidate, evidence


def write_outputs(candidate: pd.DataFrame, evidence: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "B2_candidate_cycle_labels.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for record in candidate[CANDIDATE_COLUMNS].to_dict(orient="records"):
            writer.writerow(
                {k: ("" if (isinstance(v, float) and pd.isna(v)) or v is None else v) for k, v in record.items()}
            )
    (OUT_DIR / "B2_candidate_summary.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")


def main() -> int:
    before = {"raw": sha256_file(B2_RAW), "derived": sha256_file(B2_DERIVED)}
    candidate, evidence = build_candidate()
    write_outputs(candidate, evidence)
    after = {"raw": sha256_file(B2_RAW), "derived": sha256_file(B2_DERIVED)}
    if before != after:
        print("ABORT: input archive changed during the run")
        return 1

    print(f"B2 workbooks verified: {evidence['raw_manifest']['workbooks']} (all hashes match: "
          f"{evidence['raw_manifest']['all_hashes_match']})")
    for check in evidence["spot_checks"]:
        print(f"  cycle {check['cycle_index']}: raw {check['raw_q_discharge_ah']} Ah vs table "
              f"{check['table_q_discharge_ah']} Ah (|diff| {check['abs_diff_ah']})")
    print(f"rows: {evidence['rows']}  labels: {evidence['label_status_counts']}")
    print(f"SOH rated range (in-analysis): {evidence['in_analysis_soh_rated_pct_range']}")
    print(f"SOH initial-ref range (in-analysis): {evidence['in_analysis_soh_initial_ref_pct_range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
