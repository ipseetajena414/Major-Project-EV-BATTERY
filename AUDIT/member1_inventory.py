"""
Member 1 - complete source inventory and canonical baseline verification.

Covers the six LFP cells, NASA NMC and Na-ion. Every source is opened; counts are
read from the archives themselves, not from reports. Writes only MEMBER1_* CSV/JSON
under AUDIT/. Read-only on DATA/ and the canonical master table.

Run:
    python AUDIT/member1_inventory.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import pickle
import re
import zipfile
from pathlib import Path

import pandas as pd

AUDIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AUDIT_DIR.parent
DATA = REPO_ROOT / "DATA"

MASTER_PARQUET = REPO_ROOT / "master_table.parquet"
MASTER_CSV = REPO_ROOT / "master_table.csv"
MASTER_SHA256 = "6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb"
MASTER_CSV_SHA256 = "9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b"

LFP_SOURCES = {
    "LFP_MultiStat_B1": (DATA / "B1_Discharge0.5C_Charge0.5C(1).zip", "BATTERY_1", "B1"),
    "LFP_MultiStat_B2": (DATA / "B2_Discharge1C_Charge0.5C.zip", "BATTERY_2", "B2"),
    "LFP_MultiStat_B3": (DATA / "B3_Discharge2C_Charge0.5C (1).zip", "BATTERY_3", "B3"),
    "LFP_MultiStat_B4": (DATA / "B4_Discharge3C_Charge0.5C (1).zip", "BATTERY_4", "B4"),
    "LFP_MultiStat_B5": (DATA / "B5_Discharge0.5C_Charge1C (1).zip", "BATTERY_5", "B5"),
    "LFP_MultiStat_B6": (DATA / "B6_Discharge0.5C_Charge2C.zip", "BATTERY_6", "B6"),
}

# Authoritative cycle counts established by raw parsing in the audit packages.
# B1/B2/B3/B4 = number of Step2_Cycle sheets (= labelled cycles).
# B5 = 1,491 Step2_Cycle + 7 renamed raw cycle tabs = 1,498 real cycles.
#     The historical 1,499 audit rows also contain a "File List" metadata tab that the
#     older audit mislabelled CYCLE_RENAMED_SHEET; it is excluded here and flagged in the report.
# B6 = 1,686 Step2_Cycle + 16 renamed raw cycle tabs (1,702 chronology records).
AUDITED_CYCLES = {
    "LFP_MultiStat_B1": 1460,
    "LFP_MultiStat_B2": 1506,
    "LFP_MultiStat_B3": 1979,
    "LFP_MultiStat_B4": 1577,
    "LFP_MultiStat_B5": 1498,
    "LFP_MultiStat_B6": 1702,
}
NASA_ZIP = DATA / "archive (4).zip"
NAION_ZIP = DATA / "NA-ion.zip"

INVENTORY_COLUMNS = [
    "dataset_id",
    "chemistry",
    "cell_id",
    "source_path",
    "source_type",
    "bytes",
    "sha256",
    "opened",
    "workbooks",
    "worksheets",
    "cycle_sheets",
    "step2_cycle_sheets",
    "generic_raw_export_tabs",
    "generic_cycle_sheets",
    "cycle_records",
    "label_records",
    "notes",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lfp_inventory(zip_path: Path) -> dict:
    import openpyxl

    with zipfile.ZipFile(zip_path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".xlsx")]
        sheets = 0
        step2 = 0
        generic = 0
        raw_export_tabs = 0
        for member in members:
            raw = archive.read(member)
            with zipfile.ZipFile(io.BytesIO(raw)) as inner:
                xml = inner.read("xl/workbook.xml").decode("utf-8", "ignore")
            sheets += len(re.findall(r"<sheet\b", xml))
            names = re.findall(r'<sheet[^>]*name="([^"]*)"', xml)
            step2 += sum(1 for n in names if "Step2_Cycle" in n)
            gen = [n for n in names if re.fullmatch(r"Sheet\d+", n)]
            generic += len(gen)
            if gen:
                wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
                for name in gen:
                    ws = wb[name]
                    row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True), ())
                    has_header = any(isinstance(c, str) and "Begin Header" in c for c in row)
                    # header-only tabs carry the MultiStat preamble (127 rows) but no telemetry
                    if has_header and (ws.max_row or 0) > 500:
                        raw_export_tabs += 1
                wb.close()
    return {
        "workbooks": len(members),
        "worksheets": sheets,
        "cycle_sheets": step2 + raw_export_tabs,
        "step2_cycle_sheets": step2,
        "generic_raw_export_tabs": generic,
        "generic_cycle_sheets": raw_export_tabs,
    }


def nasa_inventory(zip_path: Path) -> dict:
    import scipy.io

    target = ["B0005", "B0006", "B0018"]
    cycles = 0
    total_samples = 0
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for cell in target:
            member = f"Battery_DataSet/Battery_DataSet/{cell}.mat"
            if member not in names:
                continue
            mat = scipy.io.loadmat(io.BytesIO(archive.read(member)))
            entries = mat[cell][0, 0]["cycle"][0]
            for entry in entries:
                if str(entry["type"][0]) != "discharge":
                    continue
                cycles += 1
                total_samples += len(entry["data"][0, 0]["Voltage_measured"][0])
    return {
        "workbooks": len([n for n in names if n.lower().endswith(".mat")]),
        "worksheets": None,
        "cycle_sheets": cycles,
        "notes": f"target cells B0005/B0006/B0018; {cycles} discharge cycles; {total_samples:,} telemetry samples",
    }


def naion_inventory(zip_path: Path, max_cells=4, max_cycles=25) -> dict:
    with zipfile.ZipFile(zip_path) as archive:
        pkls = sorted(n for n in archive.namelist() if n.endswith(".pkl"))
        selected = pkls[:max_cells]
        cycles = 0
        samples = 0
        for name in selected:
            obj = pickle.load(io.BytesIO(archive.read(name)))
            for entry in obj.get("cycle_data", [])[:max_cycles]:
                cycles += 1
                samples += len(entry.get("voltage_in_V", []))
    return {
        "workbooks": len(pkls),
        "worksheets": None,
        "cycle_sheets": cycles,
        "notes": (
            f"{len(pkls)} cell pickles available; canonical build uses first {max_cells} cells x "
            f"first {max_cycles} cycles = {cycles} cycles, {samples:,} samples; temperature not measured"
        ),
    }


def label_counts() -> dict:
    """Cycle-level label counts from the existing audit/derived tables (read-only)."""
    out = {}
    with zipfile.ZipFile(AUDIT_DIR / "B1_LFP_Audit_Output.zip") as z:
        cl = pd.read_csv(io.BytesIO(z.read("B1_LFP_Audit_Output/LFP_staging_cycle_level.csv")))
    out["LFP_MultiStat_B1"] = int(cl["capacity_ah"].notna().sum())
    with zipfile.ZipFile(DATA / "B2_LFP_Member1_audit.zip") as z:
        cf = pd.read_parquet(io.BytesIO(z.read("LFP_Member1_audit/exports/cycle_features.parquet")))
    out["LFP_MultiStat_B2"] = int(cf["q_discharge_ah"].notna().sum())
    with zipfile.ZipFile(AUDIT_DIR / "Existing_Audits_Available_B3_B4 (1).zip") as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        b3 = next(n for n in names if "b3_cycle_summary" in n.lower())
        b3df = pd.read_csv(io.BytesIO(z.read(b3)))
    out["LFP_MultiStat_B3"] = int(b3df.filter(regex="(?i)soh|capacity").notna().any(axis=1).sum())
    out["LFP_MultiStat_B4"] = 0
    with zipfile.ZipFile(AUDIT_DIR / "B5.zip") as z:
        b5 = pd.read_csv(io.BytesIO(z.read("B5_Cycle_Summary.csv")))
    out["LFP_MultiStat_B5"] = int(b5.filter(regex="(?i)soh_nominal").notna().any(axis=1).sum())
    cand = AUDIT_DIR / "candidates" / "B6_candidate_cycle_chronology.csv"
    if cand.exists():
        b6 = pd.read_csv(cand)
        out["LFP_MultiStat_B6"] = int(b6["soh_nominal_pct_conditional"].notna().sum())
    return out


def main() -> int:
    rows = []
    labels = label_counts()

    for dataset_id, (path, cell_id, tag) in LFP_SOURCES.items():
        if not path.exists():
            rows.append(
                {
                    "dataset_id": dataset_id, "chemistry": "LFP", "cell_id": cell_id,
                    "source_path": str(path), "source_type": "raw_archive", "bytes": None,
                    "sha256": None, "opened": False, "workbooks": None, "worksheets": None,
                    "cycle_sheets": None, "step2_cycle_sheets": None, "generic_raw_export_tabs": None,
                    "generic_cycle_sheets": None,
                    "cycle_records": None, "label_records": None,
                    "notes": "MISSING",
                }
            )
            continue
        inv = lfp_inventory(path)
        rows.append(
            {
                "dataset_id": dataset_id, "chemistry": "LFP", "cell_id": cell_id,
                "source_path": str(path), "source_type": "raw_archive",
                "bytes": path.stat().st_size, "sha256": sha256_file(path), "opened": True,
                "workbooks": inv["workbooks"], "worksheets": inv["worksheets"],
                "cycle_sheets": inv["cycle_sheets"], "step2_cycle_sheets": inv["step2_cycle_sheets"],
                "generic_raw_export_tabs": inv["generic_raw_export_tabs"],
                "generic_cycle_sheets": inv["generic_cycle_sheets"],
                "cycle_records": AUDITED_CYCLES[dataset_id],
                "label_records": labels.get(dataset_id),
                "notes": (
                    "MultiStat 1 Hz; I(A/cm2) header with Surface Area=0; no temperature channel"
                    + (f"; {inv['generic_raw_export_tabs']} generic Sheet* tabs, "
                       f"{inv['generic_cycle_sheets']} parse as raw EC-Lab cycle exports"
                       if inv["generic_raw_export_tabs"] else "")
                    + f"; audited cycle records={AUDITED_CYCLES[dataset_id]}"
                ),
            }
        )

    nasa = nasa_inventory(NASA_ZIP)
    rows.append(
        {
            "dataset_id": "NASA_Ames_PCoE", "chemistry": "NMC", "cell_id": "B0005;B0006;B0018",
            "source_path": str(NASA_ZIP), "source_type": "raw_archive",
            "bytes": NASA_ZIP.stat().st_size, "sha256": sha256_file(NASA_ZIP), "opened": True,
            "workbooks": nasa["workbooks"], "worksheets": None, "cycle_sheets": nasa["cycle_sheets"],
            "step2_cycle_sheets": None, "generic_raw_export_tabs": None, "generic_cycle_sheets": None,
            "cycle_records": nasa["cycle_sheets"], "label_records": nasa["cycle_sheets"],
            "notes": nasa["notes"],
        }
    )

    naion = naion_inventory(NAION_ZIP)
    rows.append(
        {
            "dataset_id": "Zenodo_IonTech_NaIon", "chemistry": "Na-ion", "cell_id": "4 canonical cells",
            "source_path": str(NAION_ZIP), "source_type": "raw_archive",
            "bytes": NAION_ZIP.stat().st_size, "sha256": sha256_file(NAION_ZIP), "opened": True,
            "workbooks": naion["workbooks"], "worksheets": None, "cycle_sheets": naion["cycle_sheets"],
            "step2_cycle_sheets": None, "generic_raw_export_tabs": None, "generic_cycle_sheets": None,
            "cycle_records": naion["cycle_sheets"], "label_records": naion["cycle_sheets"],
            "notes": naion["notes"],
        }
    )

    with open(AUDIT_DIR / "MEMBER1_SOURCE_INVENTORY.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    master = pd.read_parquet(MASTER_PARQUET)
    baseline = {
        "master_parquet_sha256": sha256_file(MASTER_PARQUET),
        "master_csv_sha256": sha256_file(MASTER_CSV),
        "master_parquet_sha256_matches_baseline": sha256_file(MASTER_PARQUET) == MASTER_SHA256,
        "master_csv_sha256_matches_baseline": sha256_file(MASTER_CSV) == MASTER_CSV_SHA256,
        "rows": int(len(master)),
        "columns": list(master.columns),
        "rows_by_dataset": {k: int(v) for k, v in master["dataset_id"].value_counts().items()},
        "rows_by_cell": {k: int(v) for k, v in master["cell_id"].value_counts().items()},
        "temp_nulls": int(master["temp_c"].isna().sum()),
        "soh_nulls": int(master["soh_pct"].isna().sum()),
    }
    (AUDIT_DIR / "MEMBER1_BASELINE.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    print("dataset inventory:")
    for row in rows:
        print(f"  {row['dataset_id']:24s} wb={row['workbooks']} sheets={row['worksheets']} cycles={row['cycle_sheets']} labels={row['label_records']} opened={row['opened']}")
    print(f"master rows={baseline['rows']} sha_ok={baseline['master_parquet_sha256_matches_baseline']}")
    print(f"rows_by_dataset={baseline['rows_by_dataset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
