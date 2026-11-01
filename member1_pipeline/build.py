"""Build the three processed Member 1 datasets and their audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import audit
from .cleaning import QUALITY_COLUMNS, clean_dataset, raw_path, read_raw, sha256_file
from .config import (
    DATASETS,
    LEGACY_DIR,
    PIPELINE_VERSION,
    PROCESSED_DIR,
    REPO_ROOT,
    dataset_spec,
)

REQUIRED_PROVENANCE = ["source_row_id", "dataset_id", "data_origin", "source_file", "quality_flag"]
REQUIRED_SCHEMA_COLUMNS = list(dict.fromkeys([*REQUIRED_PROVENANCE, *QUALITY_COLUMNS]))
FORBIDDEN_FABRICATED_COLUMNS = {
    "SOLID_STATE_TELEMETRY_2026": ["time_sec", "voltage_v", "current_a", "capacity_ah"],
    "SYNTHETIC_BATTERY_LIFECYCLE": ["soh_pct", "rul_cycles", "remaining_life_cycles"],
}


def _legacy_entry(name: str, filename: str, reason: str, data: bool = False) -> dict | None:
    path = LEGACY_DIR / filename
    if not path.exists():
        return None
    entry = {
        "id": name,
        "path": path.relative_to(LEGACY_DIR.parent.parent).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "status": "deprecated",
        "reason": reason,
    }
    if data:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        entry["rows"] = int(len(frame))
        entry["columns"] = int(frame.shape[1])
        entry["column_names"] = list(frame.columns)
    return entry


def legacy_artifacts() -> list[dict]:
    candidates = [
        (
            "LEGACY_MASTER_TABLE_8COL",
            "master_table_8col_fabricated.parquet",
            "Archived root master table: 8 columns without provenance; Na-ion voltage/current were constants and Na-ion SOH was derived from resistance.",
            True,
        ),
        (
            "LEGACY_MASTER_TABLE_4789_LFP_B1",
            "master_table_4789_LFP_B1_regression.csv",
            "Archived 4,789-row LFP B1-only output from a partial legacy build; the NASA and Na-ion archives required by master_table_build.py are absent.",
            True,
        ),
        (
            "LEGACY_BUILD_SCRIPT",
            "master_table_build.py",
            "Legacy master build script; reads DATA/archive (4).zip, DATA/NA-ion.zip and AUDIT/B5.zip, which are not present in this checkout. Never run blindly.",
            False,
        ),
        (
            "LEGACY_VALIDATION_JSON",
            "master_table_final_validation.json",
            "Validation report for the earlier 515,270-row 13-column master table; the referenced artifact is not present in this checkout.",
            False,
        ),
        (
            "LEGACY_VALIDATION_SUMMARY",
            "master_table_validation_report.json",
            "Validation summary for the earlier 515,270-row 13-column master table; superseded by the current data_quality_report.json.",
            False,
        ),
        (
            "LEGACY_STATUS_REPORT",
            "MEMBER1_STATUS_REPORT.md",
            "Legacy Member 1 status report describing the retired 515,270-row master table.",
            False,
        ),
        (
            "LEGACY_FINAL_VALIDATION_REPORT",
            "MEMBER1_FINAL_VALIDATION_REPORT.md",
            "Legacy Member 1 final validation report describing the retired 515,270-row master table.",
            False,
        ),
    ]
    entries = []
    for name, filename, reason, is_data in candidates:
        entry = _legacy_entry(name, filename, reason, data=is_data)
        if entry:
            entries.append(entry)
    entries.append(
        {
            "id": "LEGACY_MASTER_TABLE_13COL_515270",
            "path": "not present in this checkout",
            "status": "documented_only",
            "reason": (
                "The earlier 13-column 515,270-row table is referenced by the archived validation "
                "reports and exists on other Git branches, but it cannot be regenerated here because "
                "DATA/archive (4).zip and DATA/NA-ion.zip are absent. It is not the active dataset."
            ),
        }
    )
    return entries


def build_all(
    out_dir: Path = PROCESSED_DIR,
    write_reports: bool = True,
    include_legacy: bool = True,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_frames: dict[str, pd.DataFrame] = {}
    clean_frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict] = {}
    processed_paths: dict[str, Path] = {}
    raw_hashes_before: dict[str, str] = {}

    for dataset_id, spec in DATASETS.items():
        source = REPO_ROOT / spec["source_file"]
        if not source.exists():
            raise FileNotFoundError(f"Missing raw dataset: {source}")
        raw_hashes_before[dataset_id] = sha256_file(source)
        raw_frames[dataset_id] = read_raw(dataset_id)
        frame, summary = clean_dataset(dataset_id)
        clean_frames[dataset_id] = frame
        summaries[dataset_id] = summary
        destination = out_dir / spec["processed_filename"]
        frame.to_parquet(destination, index=False, compression="snappy")
        processed_paths[dataset_id] = destination

    inventory = audit.build_inventory(raw_frames, clean_frames)
    quality = audit.build_quality_report(summaries, clean_frames, processed_paths)
    dictionary = audit.build_data_dictionary(clean_frames)
    leakage = audit.build_leakage_audit(raw_frames, clean_frames)
    registry = audit.build_registry(
        inventory,
        clean_frames,
        processed_paths,
        legacy_artifacts() if include_legacy else [],
    )

    if write_reports:
        audit.write_json(out_dir / "dataset_inventory.json", inventory)
        audit.write_json(out_dir / "data_quality_report.json", quality)
        audit.write_data_dictionary(out_dir / "data_dictionary.csv", dictionary)
        audit.write_json(out_dir / "leakage_audit.json", leakage)
        audit.write_json(out_dir / "dataset_registry.json", registry)

    raw_hashes_after = {
        dataset_id: sha256_file(REPO_ROOT / spec["source_file"])
        for dataset_id, spec in DATASETS.items()
    }
    if raw_hashes_before != raw_hashes_after:
        raise RuntimeError("A raw dataset changed during the build")

    validation = check_artifacts(out_dir) if write_reports else validate_outputs(out_dir)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "out_dir": out_dir.as_posix(),
        "datasets": {
            dataset_id: {
                "rows": int(len(clean_frames[dataset_id])),
                "columns": int(clean_frames[dataset_id].shape[1]),
                "processed_file": processed_paths[dataset_id].name,
                "processed_sha256": sha256_file(processed_paths[dataset_id]),
            }
            for dataset_id in DATASETS
        },
        "raw_hashes_unchanged": True,
        "validation": validation,
    }


def validate_outputs(out_dir: Path = PROCESSED_DIR) -> dict:
    result = {"ok": True, "datasets": {}, "errors": []}
    for dataset_id, spec in DATASETS.items():
        path = out_dir / spec["processed_filename"]
        if not path.exists():
            result["ok"] = False
            result["errors"].append(f"Missing processed file: {path}")
            continue
        frame = pd.read_parquet(path)
        raw = read_raw(dataset_id)
        issues = []
        if len(frame) != len(raw):
            issues.append(f"row count {len(frame)} != raw {len(raw)}")

        required = list(
            dict.fromkeys(
                [
                    *REQUIRED_SCHEMA_COLUMNS,
                    *spec["targets"],
                    *spec["id_columns"],
                    spec["chemistry_column"],
                    *raw.columns,
                ]
            )
        )
        missing_columns = [column for column in required if column not in frame.columns]
        if missing_columns:
            issues.append(f"missing required column(s): {missing_columns}")

        if "dataset_id" in frame.columns:
            if frame["dataset_id"].isna().any():
                issues.append("dataset_id contains null values")
            identities = frame["dataset_id"].dropna().unique().tolist()
            if identities != [dataset_id]:
                issues.append(
                    f"dataset_id identity mismatch: expected {dataset_id!r}, "
                    f"found {identities[:5]!r}"
                )
        if "data_origin" in frame.columns:
            if frame["data_origin"].isna().any():
                issues.append("data_origin contains null values")
            origins = frame["data_origin"].dropna().unique().tolist()
            if origins != [spec["data_origin"]]:
                issues.append(
                    f"data_origin mismatch: expected {spec['data_origin']!r}, "
                    f"found {origins[:5]!r}"
                )
        if "source_row_id" in frame.columns:
            if frame["source_row_id"].tolist() != list(range(1, len(frame) + 1)):
                issues.append("source_row_id is not a 1-based row sequence")

        for forbidden in FORBIDDEN_FABRICATED_COLUMNS.get(dataset_id, []):
            if forbidden in frame.columns:
                issues.append(f"forbidden fabricated column present: {forbidden}")

        result["datasets"][dataset_id] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "issues": issues,
        }
        if issues:
            result["ok"] = False
            result["errors"].extend(f"{dataset_id}: {issue}" for issue in issues)
    return result


def _load_json(path: Path, errors: list[str]) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"Unreadable report {path.name}: {exc}")
        return {}


def check_artifacts(out_dir: Path = PROCESSED_DIR) -> dict:
    validation = validate_outputs(out_dir)
    required_reports = [
        "dataset_inventory.json",
        "data_quality_report.json",
        "data_dictionary.csv",
        "leakage_audit.json",
        "dataset_registry.json",
    ]
    missing = [name for name in required_reports if not (out_dir / name).exists()]
    validation["missing_reports"] = missing
    if missing:
        validation["ok"] = False
        validation["errors"].extend(f"Missing report: {name}" for name in missing)

    errors: list[str] = []
    inventory = _load_json(out_dir / "dataset_inventory.json", errors)
    registry = _load_json(out_dir / "dataset_registry.json", errors)
    active = registry.get("active_datasets", {}) if isinstance(registry, dict) else {}

    for dataset_id, spec in DATASETS.items():
        processed_path = out_dir / spec["processed_filename"]
        if not processed_path.exists():
            continue
        inv = inventory.get(dataset_id) if isinstance(inventory, dict) else None
        reg = active.get(dataset_id) if isinstance(active, dict) else None
        if not isinstance(inv, dict):
            errors.append(f"{dataset_id}: dataset_inventory.json entry missing")
            inv = {}
        if not isinstance(reg, dict):
            errors.append(f"{dataset_id}: dataset_registry.json entry missing")
            reg = {}

        frame = pd.read_parquet(processed_path)
        raw_hash = sha256_file(raw_path(dataset_id))
        processed_hash = sha256_file(processed_path)

        if inv.get("source_sha256") != raw_hash:
            errors.append(
                f"{dataset_id}: raw source hash in dataset_inventory.json does not "
                f"match {spec['source_file']}"
            )
        if reg.get("source_sha256") != raw_hash:
            errors.append(
                f"{dataset_id}: raw source hash in dataset_registry.json does not "
                f"match {spec['source_file']}"
            )
        if reg.get("processed_file") != spec["processed_filename"]:
            errors.append(
                f"{dataset_id}: registry processed_file does not match the "
                f"configured dataset identity"
            )
        if reg.get("processed_sha256") != processed_hash:
            errors.append(
                f"{dataset_id}: processed hash does not match dataset_registry.json"
            )
        if inv.get("processed_rows") != len(frame) or inv.get("processed_columns") != frame.shape[1]:
            errors.append(
                f"{dataset_id}: inventory shape does not match the processed file"
            )
        if reg.get("rows") != len(frame) or reg.get("columns") != frame.shape[1]:
            errors.append(
                f"{dataset_id}: registry shape does not match the processed file"
            )
        if inv.get("processed_column_names") != list(frame.columns):
            errors.append(
                f"{dataset_id}: column schema in dataset_inventory.json does not "
                f"match the processed file"
            )
        actual_dtypes = {column: str(dtype) for column, dtype in frame.dtypes.items()}
        recorded_dtypes = inv.get("processed_dtypes", {})
        if recorded_dtypes != actual_dtypes:
            changed = sorted(
                column
                for column in set(actual_dtypes) | set(recorded_dtypes)
                if actual_dtypes.get(column) != recorded_dtypes.get(column)
            )
            errors.append(f"{dataset_id}: dtype drift for {changed}")

    if errors:
        validation["ok"] = False
        validation["errors"].extend(errors)
    return validation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the active Member 1 processed datasets and audit reports."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR,
        help=f"Output directory (default: {PROCESSED_DIR})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate existing outputs without rebuilding them.",
    )
    args = parser.parse_args(argv)

    if args.check:
        result = check_artifacts(args.out_dir)
        print("VALIDATION OK" if result["ok"] else "VALIDATION FAILED")
        if result["errors"]:
            for error in result["errors"]:
                print(f"  - {error}")
        return 0 if result["ok"] else 1

    manifest = build_all(args.out_dir)
    print("Member 1 pipeline complete")
    for dataset_id, entry in manifest["datasets"].items():
        print(
            f"  {dataset_id}: {entry['rows']} rows x {entry['columns']} columns -> {entry['processed_file']}"
        )
    if not manifest["validation"]["ok"]:
        print("Validation failed:")
        for error in manifest["validation"]["errors"]:
            print(f"  - {error}")
        return 1
    print("Validation OK; raw hashes unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
