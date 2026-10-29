"""
Member 1 - LFP sample-telemetry extractor (B1-B6).

Reads each raw MultiStat workbook once, joins every Step2_Cycle tab to its
audited cycle label by (workbook, worksheet), and emits a stratified sample of
charge and discharge telemetry for each cycle. Labels are never broadcast to a
different cycle and no value is imputed.

Outputs (per dataset):
    AUDIT/candidates/telemetry_<TAG>.parquet
    AUDIT/candidates/telemetry_<TAG>_summary.json

Run one dataset at a time (they are independent):
    python AUDIT/member1_lfp_telemetry.py B1
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
DATA = REPO_ROOT / "DATA"
OUT = AUDIT_DIR / "candidates"
OUT.mkdir(exist_ok=True)

DATASETS = {
    "B1": {
        "zip": DATA / "B1_Discharge0.5C_Charge0.5C(1).zip",
        "cell": "BATTERY_1",
        "protocol": "LFP_0.5C_DISCHARGE_0.5C_CHARGE",
        "dataset_id": "LFP_MultiStat_B1",
    },
    "B2": {
        "zip": DATA / "B2_Discharge1C_Charge0.5C.zip",
        "cell": "BATTERY_2",
        "protocol": "LFP_1C_DISCHARGE_0.5C_CHARGE",
        "dataset_id": "LFP_MultiStat_B2",
    },
    "B3": {
        "zip": DATA / "B3_Discharge2C_Charge0.5C (1).zip",
        "cell": "BATTERY_3",
        "protocol": "LFP_2C_DISCHARGE_0.5C_CHARGE",
        "dataset_id": "LFP_MultiStat_B3",
    },
    "B4": {
        "zip": DATA / "B4_Discharge3C_Charge0.5C (1).zip",
        "cell": "BATTERY_4",
        "protocol": "LFP_3C_DISCHARGE_0.5C_CHARGE",
        "dataset_id": "LFP_MultiStat_B4",
    },
    "B5": {
        "zip": DATA / "B5_Discharge0.5C_Charge1C (1).zip",
        "cell": "BATTERY_5",
        "protocol": "LFP_0.5C_DISCHARGE_1C_CHARGE",
        "dataset_id": "LFP_MultiStat_B5",
    },
    "B6": {
        "zip": DATA / "B6_Discharge0.5C_Charge2C.zip",
        "cell": "BATTERY_6",
        "protocol": "LFP_0.5C_DISCHARGE_2C_CHARGE",
        "dataset_id": "LFP_MultiStat_B6",
    },
}

MASTER_COLUMNS = [
    "chemistry", "dataset_id", "cell_id", "protocol_id", "cycle_id", "time_sec",
    "voltage_v", "current_a", "temp_c", "capacity_ah", "soh_pct", "quality_flag",
    "source_file",
]
EXTRA_COLUMNS = ["source_sheet", "phase", "label_status", "channel", "cycle_confidence"]

BASE_FLAGS = ["TEMP_UNMEASURED", "CURRENT_UNIT_UNDERDETERMINED"]


def _norm(value) -> str:
    return Path(str(value)).name.strip().lower()


def _join_key(source_file, worksheet) -> str:
    return f"{_norm(source_file)}::{str(worksheet).strip().lower()}"


def b1_labels() -> pd.DataFrame:
    with zipfile.ZipFile(AUDIT_DIR / "B1_LFP_Audit_Output.zip") as z:
        df = pd.read_csv(io.BytesIO(z.read("B1_LFP_Audit_Output/LFP_capacity_soh_feasibility.csv")))
    out = pd.DataFrame(
        {
            "source_file": df["source_file"],
            "source_sheet": df["worksheet"],
            "cycle_id": df["global_cycle_est"],
            "capacity_ah": df["discharge_capacity_ah"],
            "soh_pct": df["soh_nominal_pct"],
            "cycle_confidence": df["cycle_offset_confidence"],
            "label_status": np.where(
                df["discharge_capacity_ah"].notna(), "CONDITIONAL", "BLOCKED_PARTIAL_DISCHARGE"
            ),
            "channel": df["source_file"].str.extract(r"Un1Ch(\d+)", expand=False),
        }
    )
    blockers = df["blockers"].fillna("")
    out["extra_flags"] = np.where(
        df["cycle_offset_confidence"].ne("VERIFIED"), "OFFSET_UNDERDETERMINED", ""
    )
    out.loc[df["discharge_capacity_ah"].isna(), "extra_flags"] = (
        out.loc[df["discharge_capacity_ah"].isna(), "extra_flags"] + ";PARTIAL_DISCHARGE"
    )
    out["extra_flags"] = out["extra_flags"].str.strip(";")
    return out


def b2_labels() -> pd.DataFrame:
    with zipfile.ZipFile(DATA / "B2_LFP_Member1_audit.zip") as z:
        df = pd.read_parquet(io.BytesIO(z.read("LFP_Member1_audit/exports/cycle_features.parquet")))
    ch = df["channel"].astype("float")
    transition = ch.ne(ch.shift(1)) & ch.notna() & ch.shift(1).notna()
    out = pd.DataFrame(
        {
            "source_file": df["file_id"].astype(str) + ".xlsx",
            "source_sheet": df["sheet"],
            "cycle_id": df["cycle_index"],
            "capacity_ah": df["q_discharge_ah"],
            "soh_pct": df["soh_rated"].astype(float) * 100.0,
            "cycle_confidence": df["provenance"],
            "label_status": np.where(df["in_analysis"], "CONDITIONAL", "INVALID_CYCLE"),
            "channel": df["channel"].astype(str),
        }
    )
    out["extra_flags"] = np.where(df["in_analysis"], "", "INVALID_CYCLE")
    out["extra_flags"] = out["extra_flags"].str.strip(";")
    out.loc[df["in_analysis"] & ~df["discharge_reached_cutoff"].fillna(True), "extra_flags"] = (
        out["extra_flags"] + ";DISCHARGE_CUTOFF_NOT_REACHED"
    ).str.strip(";")
    out.loc[transition, "extra_flags"] = (out["extra_flags"] + ";CHANNEL_TRANSITION").str.strip(";")
    return out


def b3_labels() -> pd.DataFrame:
    with zipfile.ZipFile(AUDIT_DIR / "Existing_Audits_Available_B3_B4 (1).zip") as z:
        df = pd.read_csv(io.BytesIO(z.read("B3_B4/b3_cycle_summary (1).csv")))
    out = pd.DataFrame(
        {
            "source_file": df["source_workbook"],
            "source_sheet": df["source_sheet"],
            "cycle_id": df["cycle_index_nominal"],
            "capacity_ah": df["Qdis_Ah"],
            "soh_pct": df["SOH_vs_nominal_1.25Ah_pct"],
            "cycle_confidence": df["cycle_index_confidence"],
            "label_status": np.where(df["Qdis_Ah"].notna(), "CONDITIONAL", "BLOCKED"),
            "channel": df["channel"],
        }
    )
    out["extra_flags"] = np.where(
        df["Qdis_Ah"].notna(), "", "NO_DISCHARGE_CAPACITY"
    )
    return out


def b4_labels() -> pd.DataFrame:
    """No audited labels exist; cycle identity is reconstructed, targets stay null."""
    return pd.DataFrame(
        columns=[
            "source_file", "source_sheet", "cycle_id", "capacity_ah", "soh_pct",
            "cycle_confidence", "label_status", "channel", "extra_flags",
        ]
    )


def b5_labels() -> pd.DataFrame:
    with zipfile.ZipFile(AUDIT_DIR / "B5.zip") as z:
        df = pd.read_csv(io.BytesIO(z.read("B5_Cycle_Summary.csv")))
    # the old audit mislabelled the Cycle_1200 "File List" metadata tab as a cycle
    df = df[df["worksheet"].astype(str).str.lower() != "file list"].copy()
    out = pd.DataFrame(
        {
            "source_file": df["source_file"],
            "source_sheet": df["worksheet"],
            "cycle_id": df["global_cycle_id_nominal_checkpoint"],
            "capacity_ah": df["dis_q_ah_trapz_CALCULATED"],
            "soh_pct": df["soh_nominal_pct_CONDITIONAL_1.25Ah_rated"],
            "cycle_confidence": df["global_id_status"],
            "label_status": np.where(
                df["capacity_ground_truth"].eq("YES"), "VERIFIED", "CONDITIONAL"
            ),
            "channel": df["source_file"].str.extract(r"Un1Ch(\d+)", expand=False),
        }
    )
    out["extra_flags"] = "CAPACITY_GROUND_TRUTH_NO"
    partial = df["dis_complete"].astype(str).str.lower().isin(["false", "0", "no"])
    out.loc[partial, "extra_flags"] = "PARTIAL_DISCHARGE;CAPACITY_GROUND_TRUTH_NO"
    qf = df["quality_flags"].fillna("").astype(str)
    for token in ["CHARGE_BALANCE_DISCREPANCY", "CURRENT_UNIT"]:
        hit = qf.str.contains(token, case=False, regex=False)
        out.loc[hit, "extra_flags"] = (out.loc[hit, "extra_flags"] + ";" + token).str.strip(";")
    return out


def b6_labels() -> pd.DataFrame:
    df = pd.read_csv(OUT / "B6_candidate_cycle_chronology.csv")
    out = pd.DataFrame(
        {
            "source_file": df["source_file"],
            "source_sheet": df["worksheet"],
            "cycle_id": df["candidate_global_index"],
            "capacity_ah": df["discharge_capacity_ah_conditional"],
            "soh_pct": df["soh_nominal_pct_conditional"],
            "cycle_confidence": df["label_status"],
            "label_status": np.where(
                df["discharge_capacity_ah_conditional"].notna(), "CONDITIONAL", "BLOCKED"
            ),
            "channel": df["channel"],
        }
    )
    out["extra_flags"] = np.where(
        df["complete_discharge_screen"].astype(str).str.lower().isin(["true", "1", "yes"]),
        "",
        "PARTIAL_DISCHARGE",
    )
    return out


LABEL_LOADERS = {
    "B1": b1_labels,
    "B2": b2_labels,
    "B3": b3_labels,
    "B4": b4_labels,
    "B5": b5_labels,
    "B6": b6_labels,
}


def sample_rows(times, volts, currents, per_phase=50):
    """Stratified sample of charge and discharge points, preserving order."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(volts, dtype=float)
    i = np.asarray(currents, dtype=float)
    rows = []
    for phase, mask in (("charge", i > 0), ("discharge", i < 0)):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        if idx.size > per_phase:
            pick = np.unique(np.linspace(0, idx.size - 1, per_phase).round().astype(int))
            idx = idx[pick]
        for k in idx:
            rows.append((float(t[k]), float(v[k]), float(i[k]), phase))
    return rows


NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
ROW_RE = re.compile(rb'<row r="(\d+)"[^>]*>(.*?)</row>', re.S)
CELL_RE = re.compile(rb'<c r="([ABC])\d+"[^>]*>\s*<v>([^<]+)</v>')


def parse_sheet(xml: bytes):
    """Numeric (time, voltage, current) rows from a MultiStat sheet, header skipped."""
    times, volts, currents = [], [], []
    for match in ROW_RE.finditer(xml):
        if int(match.group(1)) < 50:
            continue
        cells = dict(CELL_RE.findall(match.group(2)))
        if len(cells) != 3:
            continue
        try:
            t = float(cells[b"A"])
            v = float(cells[b"B"])
            i = float(cells[b"C"])
        except (ValueError, KeyError):
            continue
        if not (0.0 <= t <= 1e7 and 0.5 <= v <= 5.5 and -50.0 <= i <= 50.0):
            continue
        times.append(t)
        volts.append(v)
        currents.append(i)
    return times, volts, currents


def read_workbook(raw: bytes, include_generic: bool = False):
    """Yield (sheet_name, times, volts, currents) for cycle tabs.

    Step2_Cycle tabs are the standard processed cycles. Some workbooks also hold
    renamed raw EC-Lab exports on tabs named Sheet1, Sheet2, ...; those are real
    cycle tabs only when an audit label exists for them (B5/B6), so callers opt in.
    """
    with zipfile.ZipFile(io.BytesIO(raw)) as inner:
        workbook = ET.fromstring(inner.read("xl/workbook.xml"))
        rels = ET.fromstring(inner.read("xl/_rels/workbook.xml.rels"))
        targets = {r.get("Id"): r.get("Target") for r in rels}
        for sheet in workbook.iter(NS + "sheet"):
            name = sheet.get("name")
            is_step2 = "Step2_Cycle" in name
            is_generic = include_generic and re.fullmatch(r"Sheet\d+", name or "") is not None
            if not (is_step2 or is_generic):
                continue
            target = targets[sheet.get(REL_NS + "id")].lstrip("/")
            if not target.startswith("xl/"):
                target = "xl/" + target
            times, volts, currents = parse_sheet(inner.read(target))
            yield name, times, volts, currents


def process(tag: str) -> dict:
    cfg = DATASETS[tag]
    labels = LABEL_LOADERS[tag]()
    labels["key"] = [
        _join_key(a, b) for a, b in zip(labels["source_file"], labels["source_sheet"])
    ]
    label_map = labels.drop_duplicates("key").set_index("key")
    if label_map.index.duplicated().any():
        raise RuntimeError(f"{tag}: duplicate (source_file, worksheet) label keys")

    records = []
    used_keys = set()
    sheets_read = 0
    label_misses = []
    with zipfile.ZipFile(cfg["zip"]) as archive:
        members = sorted(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
        for member in members:
            sheet_records = list(
                read_workbook(archive.read(member), include_generic=tag in {"B5", "B6"})
            )
            n_sheets = len(sheet_records)
            match = re.search(r"Cycle_(\d+)", member)
            checkpoint = int(match.group(1)) if match else None
            for local_no, (sheet, times, volts, currents) in enumerate(sheet_records, start=1):
                sheets_read += 1
                if not times:
                    label_misses.append(f"{_norm(member)}::{sheet}|empty")
                    continue

                key = _join_key(member, sheet)
                if key in label_map.index:
                    lab = label_map.loc[key]
                    used_keys.add(key)
                    cycle_id = lab["cycle_id"]
                    cap = lab["capacity_ah"]
                    soh = lab["soh_pct"]
                    label_status = str(lab["label_status"])
                    cycle_conf = str(lab["cycle_confidence"])
                    channel = "" if pd.isna(lab["channel"]) else str(lab["channel"])
                    extra = "" if pd.isna(lab["extra_flags"]) else str(lab["extra_flags"])
                else:
                    # No audited label. Keep the cycle visible with a reconstructed id.
                    cycle_id = checkpoint - n_sheets + local_no if checkpoint else local_no
                    cap = np.nan
                    soh = np.nan
                    label_status = "UNLABELED"
                    cycle_conf = "RECONSTRUCTED_FROM_SHEET_ORDER"
                    channel = ""
                    extra = "NO_AUDITED_LABEL"
                    label_misses.append(f"{_norm(member)}::{sheet}|unlabelled")

                flags = list(BASE_FLAGS)
                if extra:
                    flags.extend(extra.split(";"))
                if label_status == "UNLABELED":
                    flags.append("UNLABELED")
                if tag != "B2" and cycle_conf not in (
                    "VERIFIED", "file_list_indexed", "high (contiguous block, sheet order)"
                ):
                    flags.append("CYCLE_ID_RECONSTRUCTED")
                qflag = ";".join(dict.fromkeys(f for f in flags if f))

                for t, v, i, phase in sample_rows(times, volts, currents):
                    records.append(
                        {
                            "chemistry": "LFP",
                            "dataset_id": cfg["dataset_id"],
                            "cell_id": cfg["cell"],
                            "protocol_id": cfg["protocol"],
                            "cycle_id": int(cycle_id) if pd.notna(cycle_id) else None,
                            "time_sec": t,
                            "voltage_v": v,
                            "current_a": i,
                            "temp_c": None,
                            "capacity_ah": float(cap) if pd.notna(cap) else None,
                            "soh_pct": float(soh) if pd.notna(soh) else None,
                            "quality_flag": qflag,
                            "source_file": Path(member).name,
                            "source_sheet": sheet,
                            "phase": phase,
                            "label_status": label_status,
                            "channel": channel,
                            "cycle_confidence": cycle_conf,
                        }
                    )

    df = pd.DataFrame.from_records(records, columns=MASTER_COLUMNS + EXTRA_COLUMNS)
    out_path = OUT / f"telemetry_{tag}.parquet"
    df.to_parquet(out_path, index=False)

    unused = sorted(set(label_map.index) - used_keys)
    summary = {
        "dataset": tag,
        "dataset_id": cfg["dataset_id"],
        "zip": str(cfg["zip"]),
        "workbooks_read": len(members),
        "sheets_read": sheets_read,
        "label_records": int(len(labels)),
        "labels_matched_to_sheets": int(len(used_keys)),
        "labels_without_raw_sheet": unused[:20],
        "labels_without_raw_sheet_count": len(unused),
        "label_misses": label_misses[:50],
        "label_miss_count": len(label_misses),
        "rows_written": int(len(df)),
        "cycles_in_output": int(df["cycle_id"].nunique()),
        "phase_counts": df["phase"].value_counts().to_dict(),
        "label_status_counts": df["label_status"].value_counts().to_dict(),
        "quality_flag_counts": df["quality_flag"].value_counts().head(20).to_dict(),
        "capacity_non_null_cycles": int(df.drop_duplicates("cycle_id")["capacity_ah"].notna().sum()),
        "soh_range": [
            float(df["soh_pct"].min()) if df["soh_pct"].notna().any() else None,
            float(df["soh_pct"].max()) if df["soh_pct"].notna().any() else None,
        ],
        "voltage_range": [float(df["voltage_v"].min()), float(df["voltage_v"].max())],
        "current_range": [float(df["current_a"].min()), float(df["current_a"].max())],
        "duplicate_cycle_time_rows": int(df.duplicated(subset=["cycle_id", "time_sec"]).sum()),
        "output": str(out_path),
    }
    (OUT / f"telemetry_{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in DATASETS:
        raise SystemExit(f"usage: {sys.argv[0]} {{{','.join(DATASETS)}}}")
    process(sys.argv[1])
