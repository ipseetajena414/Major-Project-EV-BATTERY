"""Build the RUL availability and target audit artifact.

The report audits the legacy NASA NMC histories under both crossing
conventions and references the existing leave-one-cell-out evidence without
modifying it. RUL is never presented as approved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from member2_modeling.data_integration.profiles import DATASET_IDS

from .availability import LEGACY_NMC_DATASET_ID, rul_availability
from .labels import CONVENTIONS, EOL_PCT, add_rul_labels

WORKSPACE = Path(__file__).resolve().parents[2]
LEGACY_TABLE = WORKSPACE / "master_table.parquet"
LEGACY_DATASET_LABEL = "NASA_Ames_PCoE"
EVIDENCE_PATH = (
    WORKSPACE
    / "member2_modeling"
    / "experiments"
    / "results"
    / "rul"
    / "nmc_rul_summary.json"
)

REPORT_JSON = "rul_availability.json"
REPORT_MD = "MEMBER2_RUL_TARGET_AUDIT.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _legacy_cycles() -> pd.DataFrame:
    frame = pd.read_parquet(
        LEGACY_TABLE,
        columns=["dataset_id", "cell_id", "cycle_id", "soh_pct"],
    )
    nmc = frame[frame["dataset_id"] == LEGACY_DATASET_LABEL]
    cycles = (
        nmc.groupby(["cell_id", "cycle_id"], sort=True)["soh_pct"]
        .first()
        .reset_index()
        .dropna(subset=["soh_pct"])
        .sort_values(["cell_id", "cycle_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    cycles["cycle_position"] = cycles.groupby("cell_id").cumcount()
    return cycles


def _audit(cycles: pd.DataFrame, convention: str) -> dict[str, Any]:
    labelled = add_rul_labels(cycles, convention)
    cells: list[dict[str, Any]] = []
    for cell_id, group in labelled.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_position", kind="mergesort")
        raw_crossing = group["crossing_position"].iloc[0]
        crossing = None if pd.isna(raw_crossing) else int(raw_crossing)
        crossing_cycle_id = None
        transient = 0
        if crossing is not None:
            crossing_cycle_id = int(
                group.loc[group["cycle_position"] == crossing, "cycle_id"].iloc[0]
            )
            later = group[group["cycle_position"] > crossing]
            transient = int((later["soh_pct"] >= EOL_PCT).sum())
        cells.append(
            {
                "cell_id": str(cell_id),
                "observed_cycles": int(len(group)),
                "crossing_position": crossing,
                "crossing_cycle_id": crossing_cycle_id,
                "pre_threshold_cycles": int((group["rul_cycles"] > 0).sum()),
                "post_threshold_cycles": int((group["rul_cycles"] == 0).sum()),
                "censored": crossing is None,
                "cycles_above_eol_after_crossing": transient,
            }
        )
    return {
        "convention": convention,
        "cells": cells,
        "pre_threshold_cycles": int(sum(cell["pre_threshold_cycles"] for cell in cells)),
        "post_threshold_cycles": int(sum(cell["post_threshold_cycles"] for cell in cells)),
        "censored_cells": [cell["cell_id"] for cell in cells if cell["censored"]],
    }


def _evidence() -> dict[str, Any]:
    summary = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    primary = next(
        row
        for row in summary["primary_pooled"]
        if row["model"] == "lgbm_pre_eol_fixed200"
        and row["scope"] == "pre_threshold"
        and row["held_out_cell"] == "ALL"
    )
    sustained = next(
        row
        for row in summary["sensitivity_sustained3_pooled"]
        if row["model"] == "lgbm_pre_eol_fixed200"
        and row["scope"] == "pre_threshold"
        and row["held_out_cell"] == "ALL"
    )
    near = {
        row["model"]: row
        for row in summary["primary_near_eol"]
        if row["held_out_cell"] == "ALL"
    }
    production = summary["production_split"]
    return {
        "path": EVIDENCE_PATH.relative_to(WORKSPACE).as_posix(),
        "sha256": _sha256(EVIDENCE_PATH),
        "read_only": True,
        "validation_protocol": summary["validation"],
        "primary_pre_eol_n": int(primary["n"]),
        "primary_pre_eol_mae": float(primary["mae"]),
        "primary_pre_eol_rmse": float(primary["rmse"]),
        "primary_pre_eol_r2": float(primary["r2"]),
        "sustained3_pre_eol_n": int(sustained["n"]),
        "sustained3_pre_eol_mae": float(sustained["mae"]),
        "near_eol_mae": {
            "lgbm_pre_eol_fixed200": float(near["lgbm_pre_eol_fixed200"]["mae"]),
            "last_fade_persistence": float(near["last_fade_persistence"]["mae"]),
        },
        "production_test_pre_threshold_rows": int(production["test_pre_eol_n"]),
        "shipped_model_in_sample": bool(summary["shipped_model_diagnostic"]["in_sample"]),
        "shipped_model_pre_eol_mae": float(
            summary["shipped_model_diagnostic"]["pre_mae_overall"]
        ),
        "caveats": list(summary["caveats"]),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Member 2 RUL Target Audit",
        "",
        "RUL is **never approved** for operational use in this project. Snapshot",
        "datasets are unavailable for RUL, and the legacy NMC evidence is",
        "pre-EOL, experimental, and limited to three cells.",
        "",
        f"End-of-life threshold: **{EOL_PCT:.1f}% SOH**, strict (`SOH < {EOL_PCT:.1f}`).",
        "",
        "## Availability",
        "",
        "| Dataset | Status | Approved | Unit |",
        "|---|---|---|---|",
    ]
    for dataset_id, record in report["availability"].items():
        lines.append(
            f"| `{dataset_id}` | {record['status']} | "
            f"{'yes' if record['approved_for_use'] else 'no'} | "
            f"{record['unit'] or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Legacy NMC crossing audit",
            "",
            "Positions are 0-based observed-record indices, not arithmetic cycle-ID",
            "differences.",
            "",
            "| Convention | Cell | Observed cycles | Crossing position | Crossing cycle | Pre | Post | Above EOL after crossing |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for convention, audit in report["legacy_target_audit"]["conventions"].items():
        for cell in audit["cells"]:
            lines.append(
                f"| {convention} | {cell['cell_id']} | {cell['observed_cycles']} | "
                f"{cell['crossing_position']} | {cell['crossing_cycle_id']} | "
                f"{cell['pre_threshold_cycles']} | {cell['post_threshold_cycles']} | "
                f"{cell['cycles_above_eol_after_crossing']} |"
            )
    evidence = report["existing_validation_evidence"]
    lines.extend(
        [
            "",
            "## Existing validation evidence (read-only reference)",
            "",
            f"Source: `{evidence['path']}`  ",
            f"SHA-256: `{evidence['sha256']}`  ",
            f"Protocol: {evidence['validation_protocol']}",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Pre-EOL MAE (first crossing, n={evidence['primary_pre_eol_n']}) | {evidence['primary_pre_eol_mae']:.4f} |",
            f"| Pre-EOL RMSE | {evidence['primary_pre_eol_rmse']:.4f} |",
            f"| Pre-EOL R² | {evidence['primary_pre_eol_r2']:.4f} |",
            f"| Sustained-3 pre-EOL MAE (n={evidence['sustained3_pre_eol_n']}) | {evidence['sustained3_pre_eol_mae']:.4f} |",
            f"| Near-EOL MAE, LightGBM | {evidence['near_eol_mae']['lgbm_pre_eol_fixed200']:.4f} |",
            f"| Near-EOL MAE, last-fade persistence | {evidence['near_eol_mae']['last_fade_persistence']:.4f} |",
            f"| Production test pre-threshold rows | {evidence['production_test_pre_threshold_rows']} |",
            f"| Shipped model in-sample pre-EOL MAE | {evidence['shipped_model_pre_eol_mae']:.4f} |",
            "",
            "Pre-threshold and post-threshold rows are reported separately and are",
            "never pooled. The production test split contains only post-threshold",
            "rows, so it cannot measure pre-EOL forecasting.",
            "",
            "## Caveats",
            "",
        ]
    )
    lines.extend(f"- {caveat}" for caveat in evidence["caveats"])
    lines.append("")
    return "\n".join(lines)


def build_rul_report(out_dir: Path | str | None = None) -> dict[str, Any]:
    cycles = _legacy_cycles()
    report = {
        "eol_threshold_pct": EOL_PCT,
        "threshold_rule": f"strict SOH < {EOL_PCT}",
        "conventions_supported": list(CONVENTIONS),
        "rul_approved": False,
        "availability": {
            **{dataset_id: rul_availability(dataset_id) for dataset_id in DATASET_IDS},
            LEGACY_NMC_DATASET_ID: rul_availability(LEGACY_NMC_DATASET_ID),
        },
        "legacy_target_audit": {
            "source": LEGACY_TABLE.name,
            "source_sha256": _sha256(LEGACY_TABLE),
            "dataset_label": LEGACY_DATASET_LABEL,
            "conventions": {
                convention: _audit(cycles, convention) for convention in CONVENTIONS
            },
        },
        "existing_validation_evidence": _evidence(),
    }
    if out_dir is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / REPORT_JSON).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (destination / REPORT_MD).write_text(_markdown(report), encoding="utf-8")
    return report


__all__ = ["REPORT_JSON", "REPORT_MD", "build_rul_report"]
