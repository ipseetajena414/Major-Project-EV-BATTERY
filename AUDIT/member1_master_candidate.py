"""
Member 1 - candidate master table assembly.

Preserves the canonical NASA NMC and Na-ion rows unchanged, replaces the LFP
sections with corrected, provenance-joined sampled telemetry, and writes a
versioned candidate. The canonical master_table.parquet/.csv are never modified.

Outputs (repo root):
    master_table_candidate.parquet
    master_table_candidate.csv
    master_table_candidate_validation.json
    master_table_candidate_manifest.csv

Run:
    python AUDIT/member1_master_candidate.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
OUT = AUDIT_DIR / "candidates"

CANONICAL_PARQUET = REPO_ROOT / "master_table.parquet"
CANONICAL_CSV = REPO_ROOT / "master_table.csv"
BASELINE_PARQUET_SHA = "6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb"
BASELINE_CSV_SHA = "9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b"

MASTER_SCHEMA = [
    "chemistry", "dataset_id", "cell_id", "protocol_id", "cycle_id", "time_sec",
    "voltage_v", "current_a", "temp_c", "capacity_ah", "soh_pct", "quality_flag",
    "source_file",
]
LFP_TAGS = ["B1", "B2", "B3", "B4", "B5", "B6"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    canonical = pd.read_parquet(CANONICAL_PARQUET)
    canonical_sha_before = sha256_file(CANONICAL_PARQUET)
    canonical_csv_sha_before = sha256_file(CANONICAL_CSV)

    keep = canonical[~canonical["dataset_id"].str.startswith("LFP_")].copy()
    dropped = canonical[canonical["dataset_id"].str.startswith("LFP_")]

    frames = [keep]
    for tag in LFP_TAGS:
        path = OUT / f"telemetry_{tag}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}; run member1_lfp_telemetry.py {tag} first")
        df = pd.read_parquet(path)
        frames.append(df[MASTER_SCHEMA])
        print(f"  {tag}: {len(df):,} rows, {df['cycle_id'].nunique():,} cycles")

    candidate = pd.concat(frames, ignore_index=True)[MASTER_SCHEMA]
    candidate = candidate.sort_values(
        ["dataset_id", "cell_id", "cycle_id", "time_sec"], kind="mergesort"
    ).reset_index(drop=True)

    cell_cycle = (
        candidate["dataset_id"] + "|" + candidate["cell_id"] + "|" + candidate["cycle_id"].astype(str)
    )

    candidate.to_parquet(REPO_ROOT / "master_table_candidate.parquet", index=False)
    candidate.to_csv(REPO_ROOT / "master_table_candidate.csv", index=False)

    per_dataset = (
        candidate.groupby("dataset_id")
        .agg(rows=("cycle_id", "size"), cells=("cell_id", "nunique"))
        .reset_index()
    )
    per_dataset["cell_cycles"] = cell_cycle.groupby(candidate["dataset_id"]).nunique().values
    label_cycles = (
        cell_cycle[candidate["capacity_ah"].notna()]
        .groupby(candidate.loc[candidate["capacity_ah"].notna(), "dataset_id"])
        .nunique()
        .rename("cell_cycles_with_capacity")
    )
    per_dataset = per_dataset.merge(label_cycles, on="dataset_id", how="left").fillna(
        {"cell_cycles_with_capacity": 0}
    )
    per_dataset["cell_cycles_with_capacity"] = per_dataset["cell_cycles_with_capacity"].astype(int)
    per_dataset["source"] = np.where(
        per_dataset["dataset_id"].isin(dropped["dataset_id"].unique()),
        "REBUILT_FROM_RAW",
        np.where(per_dataset["dataset_id"].str.startswith("LFP_"), "REBUILT_FROM_RAW", "PRESERVED_FROM_CANONICAL"),
    )
    per_dataset.to_csv(REPO_ROOT / "master_table_candidate_manifest.csv", index=False)

    dup = int(
        candidate.duplicated(subset=["dataset_id", "cell_id", "cycle_id", "time_sec"]).sum()
    )
    validation = {
        "candidate_rows": int(len(candidate)),
        "candidate_columns": list(candidate.columns),
        "schema_conformity": list(candidate.columns) == MASTER_SCHEMA,
        "per_dataset": per_dataset.to_dict("records"),
        "canonical_rows": int(len(canonical)),
        "canonical_rows_dropped_lfp": int(len(dropped)),
        "canonical_rows_preserved_non_lfp": int(len(keep)),
        "canonical_parquet_sha256_unchanged": sha256_file(CANONICAL_PARQUET) == canonical_sha_before == BASELINE_PARQUET_SHA,
        "canonical_csv_sha256_unchanged": sha256_file(CANONICAL_CSV) == canonical_csv_sha_before == BASELINE_CSV_SHA,
        "duplicate_keys": dup,
        "null_counts": {c: int(candidate[c].isna().sum()) for c in candidate.columns},
        "ranges": {
            "voltage_v": [float(candidate["voltage_v"].min()), float(candidate["voltage_v"].max())],
            "current_a": [float(candidate["current_a"].min()), float(candidate["current_a"].max())],
            "temp_c": [
                float(candidate["temp_c"].min()) if candidate["temp_c"].notna().any() else None,
                float(candidate["temp_c"].max()) if candidate["temp_c"].notna().any() else None,
            ],
            "capacity_ah": [
                float(candidate["capacity_ah"].min()),
                float(candidate["capacity_ah"].max()),
            ],
            "soh_pct": [float(candidate["soh_pct"].min()), float(candidate["soh_pct"].max())],
        },
        "impossible_values": {
            "voltage_lt_0_or_gt_6": int(((candidate["voltage_v"] < 0) | (candidate["voltage_v"] > 6)).sum()),
            "temp_lt_-50_or_gt_100": int(
                ((candidate["temp_c"] < -50) | (candidate["temp_c"] > 100)).sum()
            ),
            "soh_lt_0_or_gt_150": int(
                ((candidate["soh_pct"] < 0) | (candidate["soh_pct"] > 150)).sum()
            ),
            "capacity_lt_0": int((candidate["capacity_ah"] < 0).sum()),
        },
        "quality_flag_top": candidate["quality_flag"].value_counts().head(15).to_dict(),
        "candidate_parquet_sha256": sha256_file(REPO_ROOT / "master_table_candidate.parquet"),
        "candidate_csv_sha256": sha256_file(REPO_ROOT / "master_table_candidate.csv"),
    }
    (REPO_ROOT / "master_table_candidate_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: validation[k] for k in [
        "candidate_rows", "schema_conformity", "duplicate_keys",
        "canonical_parquet_sha256_unchanged", "canonical_csv_sha256_unchanged",
    ]}, indent=2))
    print(per_dataset.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
