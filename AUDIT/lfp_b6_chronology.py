"""
Member 1 - B6 candidate cycle chronology (read-only on raw, isolated candidate output).

Why this exists
---------------
The B6 audit package deliberately assigns no global chronological cycle id
(``global_cycle_id`` is NaN for all 1,702 cycle-like sheets) because the raw
workbooks are split by filename checkpoint and reload the logger between files.
Per-cycle labels therefore cannot be ordered across the whole cell without a
reconstruction. This script rebuilds a *candidate* order from the timestamps the
instrument itself recorded:

* the ``Date:``/``Time:`` fields inside each cycle sheet header, and
* the ``Time`` column of each workbook's ``File List`` sheet (sub-second precision)
  for sheets that are indexed there.

The result is a candidate chronology, not an audited label. Cycle identity,
capacity and SOH values are copied unchanged from the B6 audit package so that no
label is manufactured. Rows whose discharge screen failed keep a null label.

Safety
------
* Reads the raw archive and audit package only; nothing is written outside
  ``AUDIT/candidates/``.
* Aborts if either input hash changes during the run.
* Never touches master_table.*, DATA/, or the production builder.

Run:
    python AUDIT/lfp_b6_chronology.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path

import openpyxl
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
OUT_DIR = AUDIT_DIR / "candidates"

B6_RAW = REPO_ROOT / "DATA" / "B6_Discharge0.5C_Charge2C.zip"
B6_AUDIT = AUDIT_DIR / "LFP_B6_Audit_Package (1).zip"
B6_CYCLE_SUMMARY = "LFP_B6_Audit_Output/LFP_cycle_summary.csv"

B6_RAW_SHA256 = "25fe5a2417d68334eed59667c162db6851e1970a2109e523caf91dd11484b97a"
B6_AUDIT_SHA256 = "96b5fef5905b948e21ea62f8d109d6e056c51088327d5efa1eee99c4c433878e"

NAMED_CYCLE_RE = re.compile(r"Step2_Cycle(\d+)$", re.IGNORECASE)
GENERIC_SHEET_RE = re.compile(r"^Sheet(\d+)$", re.IGNORECASE)
PROTOCOL_RE = re.compile(
    r"x(?P<repeat>\d+).*?Charge CC:\s*(?P<charge_cc>[\d.]+)\s*A.*?"
    r"CV:\s*(?P<charge_cv>[\d.]+)\s*V.*?"
    r"Discharge CC:\s*(?P<discharge_cc>[\d.]+)\s*A.*?"
    r"XV:\s*(?P<discharge_cutoff>[\d.]+)\s*V",
    re.IGNORECASE | re.DOTALL,
)

CANDIDATE_COLUMNS = [
    "candidate_global_index",
    "datetime_local",
    "datetime_source",
    "source_archive",
    "source_sha256",
    "source_file",
    "workbook_checkpoint",
    "worksheet",
    "sheet_type",
    "source_cycle_tab_number",
    "channel",
    "exp_type",
    "protocol_repeat",
    "charge_cc_a",
    "charge_cv_v",
    "discharge_cc_a",
    "discharge_cutoff_v",
    "sample_count",
    "discharge_capacity_ah_conditional",
    "charge_capacity_ah_conditional",
    "soh_nominal_pct_conditional",
    "soh_provenance",
    "complete_discharge_screen",
    "complete_discharge_screen_reason",
    "current_unit_assessment",
    "quality_flag",
    "calculation_status",
    "value_origin",
    "label_status",
    "notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cell_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def parse_sheet_header(ws) -> dict:
    """Read the MultiStat header block of one cycle sheet (first ~40 rows)."""
    date_str = time_str = exp_title = ""
    for row in ws.iter_rows(min_row=1, max_row=40, values_only=True):
        if not row:
            continue
        first = _cell_text(row[0])
        if not first:
            continue
        if first.startswith("Date:") and not date_str:
            date_str = first.split(":", 1)[1].strip()
        elif first.startswith("Time:") and not time_str:
            time_str = first.split(":", 1)[1].strip()
        elif first.startswith("Exp Title:") and not exp_title:
            exp_title = " ".join(_cell_text(c) for c in row if c is not None).strip()
    return {"date_str": date_str, "time_str": time_str, "exp_title": exp_title}


def parse_file_list(ws) -> dict[str, datetime]:
    """Map worksheet name -> logger start datetime from the File List sheet."""
    out: dict[str, datetime] = {}
    header_seen = False
    for row in ws.iter_rows(min_row=1, max_row=400, values_only=True):
        if not row:
            continue
        first = _cell_text(row[0])
        if first == "Name":
            header_seen = True
            continue
        if not header_seen or not first:
            continue
        when = row[6] if len(row) > 6 else None
        if isinstance(when, datetime):
            out[first] = when
    return out


def read_workbook(zf: zipfile.ZipFile, member: str) -> list[dict]:
    source_file = Path(member).name
    checkpoint = int(re.search(r"Cycle_(\d+)", source_file).group(1))
    with zf.open(member) as handle:
        wb = openpyxl.load_workbook(io.BytesIO(handle.read()), read_only=True, data_only=True)
    try:
        file_list = parse_file_list(wb["File List"]) if "File List" in wb.sheetnames else {}
        rows: list[dict] = []
        for sheet_order, name in enumerate(wb.sheetnames):
            named = NAMED_CYCLE_RE.search(name)
            generic = GENERIC_SHEET_RE.match(name)
            if not named and not generic:
                continue
            header = parse_sheet_header(wb[name])
            if "Charge Cycle" not in header["exp_title"]:
                continue
            date_str, time_str = header["date_str"], header["time_str"]
            header_dt = None
            if date_str and time_str:
                header_dt = datetime.strptime(f"{date_str} {time_str}", "%d/%m/%Y %H:%M:%S")
            file_dt = file_list.get(name)
            if file_dt is not None:
                when, source = file_dt.replace(microsecond=0), "file_list"
            elif header_dt is not None:
                when, source = header_dt, "sheet_header"
            else:
                when, source = None, "missing"
            protocol = PROTOCOL_RE.search(header["exp_title"])
            rows.append(
                {
                    "source_file": source_file,
                    "workbook_checkpoint": checkpoint,
                    "worksheet": name,
                    "sheet_order": sheet_order,
                    "sheet_type": "named_cycle_tab" if named else "generic_sheet_tab",
                    "tab_number": int((named or generic).group(1)),
                    "channel": name.split("_")[1] if name.startswith("Run") else "",
                    "exp_title": header["exp_title"],
                    "header_datetime": header_dt,
                    "file_list_datetime": file_dt,
                    "datetime_local": when,
                    "datetime_source": source,
                    "protocol_repeat": int(protocol.group("repeat")) if protocol else None,
                    "charge_cc_a": float(protocol.group("charge_cc")) if protocol else None,
                    "charge_cv_v": float(protocol.group("charge_cv")) if protocol else None,
                    "discharge_cc_a": float(protocol.group("discharge_cc")) if protocol else None,
                    "discharge_cutoff_v": float(protocol.group("discharge_cutoff")) if protocol else None,
                }
            )
        return rows
    finally:
        wb.close()


def reconcile_timestamps(rows: list[dict]) -> dict:
    """Check header vs. File List agreement for sheets indexed in both."""
    both = [r for r in rows if r["header_datetime"] and r["file_list_datetime"]]
    mismatch = [
        (r["source_file"], r["worksheet"], r["header_datetime"], r["file_list_datetime"].replace(microsecond=0))
        for r in both
        if r["header_datetime"] != r["file_list_datetime"].replace(microsecond=0)
    ]
    return {
        "sheets_with_both_sources": len(both),
        "sheets_with_header_only": sum(1 for r in rows if r["header_datetime"] and not r["file_list_datetime"]),
        "sheets_with_file_list_only": sum(1 for r in rows if r["file_list_datetime"] and not r["header_datetime"]),
        "sheets_without_any_timestamp": sum(1 for r in rows if not r["datetime_local"]),
        "header_vs_file_list_mismatches": len(mismatch),
        "mismatch_examples": mismatch[:10],
    }


def apply_candidate_order(rows: list[dict]) -> list[dict]:
    orderable = [r for r in rows if r["datetime_local"] is not None]
    unorderable = [r for r in rows if r["datetime_local"] is None]
    orderable.sort(key=lambda r: (r["datetime_local"], r["workbook_checkpoint"], r["tab_number"]))
    for i, row in enumerate(orderable, start=1):
        row["candidate_global_index"] = i
    for i, row in enumerate(unorderable, start=len(orderable) + 1):
        row["candidate_global_index"] = i
    return orderable + unorderable


def validate_order(rows: list[dict]) -> dict:
    ordered = [r for r in rows if r["datetime_local"] is not None]
    checkpoints = [r["workbook_checkpoint"] for r in ordered]
    times = [r["datetime_local"] for r in ordered]
    duplicate_times = len(times) - len(set(times))
    backwards = [
        (ordered[i - 1]["source_file"], ordered[i - 1]["worksheet"], ordered[i]["source_file"], ordered[i]["worksheet"])
        for i in range(1, len(ordered))
        if ordered[i]["datetime_local"] < ordered[i - 1]["datetime_local"]
    ]
    checkpoint_regressions = [
        (ordered[i - 1]["workbook_checkpoint"], ordered[i]["workbook_checkpoint"])
        for i in range(1, len(ordered))
        if ordered[i]["workbook_checkpoint"] < ordered[i - 1]["workbook_checkpoint"]
    ]
    # Does each filename checkpoint occupy one contiguous block in the candidate order?
    blocks: dict[int, list[int]] = {}
    for pos, r in enumerate(ordered):
        blocks.setdefault(r["workbook_checkpoint"], []).append(pos)
    non_contiguous = {
        cp: (positions[0], positions[-1], len(positions))
        for cp, positions in blocks.items()
        if positions[-1] - positions[0] + 1 != len(positions)
    }
    # Timestamps inside a workbook must never run backwards in tab order.
    within_book = 0
    for source_file in {r["source_file"] for r in ordered}:
        sheets = [r for r in ordered if r["source_file"] == source_file]
        sheets.sort(key=lambda r: (r["sheet_order"], r["tab_number"]))
        within_book += sum(
            1
            for a, b in zip(sheets, sheets[1:])
            if b["datetime_local"] < a["datetime_local"]
        )
    generic_positions = [
        (r["candidate_global_index"], r["source_file"], r["worksheet"], r["datetime_local"].isoformat())
        for r in ordered
        if r["sheet_type"] == "generic_sheet_tab"
    ]
    return {
        "orderable_sheets": len(ordered),
        "unorderable_sheets": len(rows) - len(ordered),
        "duplicate_timestamps": duplicate_times,
        "backwards_timestamps_in_candidate_order": len(backwards),
        "backwards_examples": backwards[:5],
        "checkpoint_regressions_in_candidate_order": len(checkpoint_regressions),
        "checkpoint_regression_examples": checkpoint_regressions[:5],
        "non_contiguous_checkpoint_blocks": non_contiguous,
        "within_workbook_tab_order_backwards": within_book,
        "generic_sheet_positions": generic_positions,
    }


def build_evidence() -> tuple[list[dict], dict]:
    raw_hash = sha256_file(B6_RAW)
    audit_hash = sha256_file(B6_AUDIT)
    if raw_hash != B6_RAW_SHA256:
        raise RuntimeError(f"B6 raw archive hash mismatch: {raw_hash}")
    if audit_hash != B6_AUDIT_SHA256:
        raise RuntimeError(f"B6 audit package hash mismatch: {audit_hash}")

    rows: list[dict] = []
    with zipfile.ZipFile(B6_RAW) as zf:
        members = sorted(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        for member in members:
            rows.extend(read_workbook(zf, member))

    timestamp_check = reconcile_timestamps(rows)
    ordered = apply_candidate_order(rows)
    order_check = validate_order(ordered)

    with zipfile.ZipFile(B6_AUDIT) as zf:
        summary = pd.read_csv(io.BytesIO(zf.read(B6_CYCLE_SUMMARY)))

    extracted = pd.DataFrame(ordered)
    merged = summary.merge(
        extracted[
            [
                "source_file",
                "worksheet",
                "candidate_global_index",
                "datetime_local",
                "datetime_source",
                "sheet_type",
                "channel",
                "exp_title",
                "protocol_repeat",
                "charge_cc_a",
                "charge_cv_v",
                "discharge_cc_a",
                "discharge_cutoff_v",
                "header_datetime",
                "file_list_datetime",
            ]
        ],
        on=["source_file", "worksheet"],
        how="outer",
        indicator=True,
    )
    merge_failures = merged.loc[merged["_merge"] != "both", ["source_file", "worksheet", "_merge"]]
    if len(merge_failures):
        raise RuntimeError(f"B6 audit/extraction keys diverged:\n{merge_failures.head(20)}")

    merged = merged.sort_values("candidate_global_index").reset_index(drop=True)
    merged["label_status"] = merged.apply(
        lambda r: "conditional_nominal_soh" if pd.notna(r["soh_nominal_pct_conditional"]) else "unlabelled_screen_failed",
        axis=1,
    )
    merged["soh_provenance"] = (
        "capacity_ah / 1.25 Ah manufacturer nominal x 100 (rated, not measured BOL retention)"
    )

    candidate = pd.DataFrame(
        {
            "candidate_global_index": merged["candidate_global_index"],
            "datetime_local": merged["datetime_local"].dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "datetime_source": merged["datetime_source"],
            "source_archive": "B6_Discharge0.5C_Charge2C.zip",
            "source_sha256": raw_hash,
            "source_file": merged["source_file"],
            "workbook_checkpoint": merged["cycle_checkpoint_from_filename"],
            "worksheet": merged["worksheet"],
            "sheet_type": merged["sheet_type"],
            "source_cycle_tab_number": merged["source_cycle_tab_number"],
            "channel": merged["channel"],
            "exp_type": "Charge Cycle",
            "protocol_repeat": merged["protocol_repeat"],
            "charge_cc_a": merged["charge_cc_a"],
            "charge_cv_v": merged["charge_cv_v"],
            "discharge_cc_a": merged["discharge_cc_a"],
            "discharge_cutoff_v": merged["discharge_cutoff_v"],
            "sample_count": merged["sample_count"],
            "discharge_capacity_ah_conditional": merged["discharge_capacity_ah_conditional"],
            "charge_capacity_ah_conditional": merged["charge_capacity_ah_conditional"],
            "soh_nominal_pct_conditional": merged["soh_nominal_pct_conditional"],
            "soh_provenance": merged["soh_provenance"],
            "complete_discharge_screen": merged["complete_discharge_screen"],
            "complete_discharge_screen_reason": merged["complete_discharge_screen_reason"],
            "current_unit_assessment": merged["current_unit_assessment"],
            "quality_flag": merged["quality_flag"],
            "calculation_status": merged["calculation_status"],
            "value_origin": merged["value_origin"],
            "label_status": merged["label_status"],
            "notes": merged["notes"],
        }
    )

    evidence = {
        "metadata": {
            "title": "B6 candidate cycle chronology reconstructed from instrument timestamps",
            "canonical_dataset_modified": False,
            "raw_archive": str(B6_RAW),
            "raw_sha256": raw_hash,
            "audit_package": str(B6_AUDIT.relative_to(REPO_ROOT)),
            "audit_sha256": audit_hash,
            "candidate_output": str((OUT_DIR / "B6_candidate_cycle_chronology.csv").relative_to(REPO_ROOT)),
        },
        "cycle_rows": int(len(candidate)),
        "source_files": int(candidate["source_file"].nunique()),
        "checkpoints": sorted(int(x) for x in candidate["workbook_checkpoint"].dropna().unique()),
        "timestamp_reconciliation": timestamp_check,
        "candidate_order_validation": order_check,
        "labels": {
            "conditional_nominal_soh": int((candidate["label_status"] == "conditional_nominal_soh").sum()),
            "unlabelled_screen_failed": int((candidate["label_status"] == "unlabelled_screen_failed").sum()),
            "soh_min_pct": float(candidate["soh_nominal_pct_conditional"].min()),
            "soh_median_pct": float(candidate["soh_nominal_pct_conditional"].median()),
            "soh_max_pct": float(candidate["soh_nominal_pct_conditional"].max()),
        },
        "interpretation_limits": [
            "Candidate order only: derived from logger Date/Time fields, not an audited global cycle id.",
            "SOH is capacity vs 1.25 Ah rated nominal, not retention against measured beginning-of-life.",
            "Current values are amperes under a blank A/cm2 header with Surface Area 0 (conditional).",
            "No temperature channel exists in these workbooks.",
        ],
    }
    return candidate, evidence


def write_outputs(candidate: pd.DataFrame, evidence: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "B6_candidate_cycle_chronology.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for record in candidate[CANDIDATE_COLUMNS].to_dict(orient="records"):
            writer.writerow(
                {k: ("" if (isinstance(v, float) and pd.isna(v)) or v is None else v) for k, v in record.items()}
            )
    (OUT_DIR / "B6_candidate_chronology_summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )


def main() -> int:
    before = {"raw": sha256_file(B6_RAW), "audit": sha256_file(B6_AUDIT)}
    candidate, evidence = build_evidence()
    write_outputs(candidate, evidence)
    after = {"raw": sha256_file(B6_RAW), "audit": sha256_file(B6_AUDIT)}
    if before != after:
        print("ABORT: input archive changed during the run")
        return 1

    checks = evidence["candidate_order_validation"]
    print(f"B6 cycle-like sheets: {evidence['cycle_rows']}")
    print(f"orderable: {checks['orderable_sheets']}, unorderable: {checks['unorderable_sheets']}")
    print(f"duplicate timestamps: {checks['duplicate_timestamps']}")
    print(f"backwards timestamps: {checks['backwards_timestamps_in_candidate_order']}")
    print(f"checkpoint regressions: {checks['checkpoint_regressions_in_candidate_order']}")
    print(f"non-contiguous checkpoint blocks: {len(checks['non_contiguous_checkpoint_blocks'])}")
    print(f"within-workbook tab-order backwards: {checks['within_workbook_tab_order_backwards']}")
    print(f"header vs file-list mismatches: {evidence['timestamp_reconciliation']['header_vs_file_list_mismatches']}")
    print(f"labels: {evidence['labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
