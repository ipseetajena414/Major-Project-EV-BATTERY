"""Read-only access to genuinely sequential battery data.

The only genuine cycle-by-cycle SOH trajectories in this repository live in the
legacy ``master_table.parquet`` export.  That table stores many within-cycle
telemetry rows per cycle, so this module aggregates each
``(dataset_id, cell_id, cycle_id)`` group to a single cycle-level state before
any modelling.  Row order is never treated as cycle order; cycles are ordered by
their actual ``cycle_id`` and position is recorded separately.

Active Member 1 snapshot datasets are deliberately *not* turned into sequences.
They are reported as UNAVAILABLE for RUL.
"""

from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from member2_modeling.data_integration.profiles import DATASET_IDS
from member2_modeling.rul.labels import EOL_PCT, crossing_position

WORKSPACE = Path(__file__).resolve().parents[2]
MASTER_TABLE = WORKSPACE / "master_table.parquet"

MIN_HISTORY_CYCLES = 10
MIN_EOL_CYCLES = 30
MIN_CENSORED_CYCLES = 50
LOCO_MIN_CELLS = 3

SEQUENTIAL_DATASET_IDS = (
    "NASA_Ames_PCoE",
    "LFP_MultiStat_B1",
    "LFP_MultiStat_B5",
    "Zenodo_IonTech_NaIon",
)

CONTROLLED_ASSUMPTIONS = (
    "SOH is the within-cycle first observed value; telemetry columns are cycle means.",
    "Cycle order follows the source cycle_id, not row order or file order.",
    "Cycle positions are 0-based positions in the observed history, not cycle ids.",
)


class SequentialDataError(RuntimeError):
    """Raised when genuine cycle-level data cannot be loaded or validated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@functools.lru_cache(maxsize=1)
def source_sha256() -> str:
    if not MASTER_TABLE.is_file():
        raise SequentialDataError(f"Legacy sequential table not found: {MASTER_TABLE}")
    return _sha256(MASTER_TABLE)


@functools.lru_cache(maxsize=1)
def _cycle_table() -> pd.DataFrame:
    if not MASTER_TABLE.is_file():
        raise SequentialDataError(
            f"Legacy sequential table not found: {MASTER_TABLE}. "
            "Cycle-level analytics require it; snapshots cannot substitute."
        )
    raw = pd.read_parquet(
        MASTER_TABLE,
        columns=[
            "dataset_id",
            "chemistry",
            "cell_id",
            "cycle_id",
            "soh_pct",
            "capacity_ah",
            "voltage_v",
            "current_a",
            "temp_c",
        ],
    )
    if raw.empty:
        raise SequentialDataError("The legacy sequential table is empty.")

    cycles = (
        raw.groupby(
            ["dataset_id", "chemistry", "cell_id", "cycle_id"],
            as_index=False,
            sort=True,
        )
        .agg(
            soh_pct=("soh_pct", "first"),
            capacity_ah=("capacity_ah", "mean"),
            voltage_mean=("voltage_v", "mean"),
            voltage_std=("voltage_v", "std"),
            current_mean=("current_a", "mean"),
            current_abs_mean=("current_a", lambda values: float(np.abs(values).mean())),
            temp_mean=("temp_c", "mean"),
            temp_max=("temp_c", "max"),
            n_rows=("soh_pct", "size"),
        )
        .dropna(subset=["soh_pct"])
        .reset_index(drop=True)
    )
    cycles["cycle_id"] = cycles["cycle_id"].astype("int64")
    cycles = cycles.sort_values(
        ["dataset_id", "cell_id", "cycle_id"], kind="mergesort"
    ).reset_index(drop=True)
    cycles["cycle_position"] = cycles.groupby(["dataset_id", "cell_id"]).cumcount()
    cycles["cycle_gap"] = cycles.groupby(["dataset_id", "cell_id"])["cycle_id"].diff()
    cycles["chemistry"] = cycles["chemistry"].astype(str)
    return cycles


def load_cycles() -> pd.DataFrame:
    """Return a copy of the cycle-level table (one row per observed cycle)."""
    return _cycle_table().copy()


def load_cell(cell_id: str, dataset_id: str | None = None) -> pd.DataFrame:
    cycles = _cycle_table()
    mask = cycles["cell_id"] == cell_id
    if dataset_id is not None:
        mask &= cycles["dataset_id"] == dataset_id
    selected = cycles.loc[mask].sort_values("cycle_id")
    if selected.empty:
        raise SequentialDataError(f"No cycle history found for cell {cell_id!r}")
    return selected.reset_index(drop=True).copy()


def cell_summary(eol_pct: float = EOL_PCT) -> pd.DataFrame:
    """Per-cell trajectory summary and observed threshold crossings."""
    rows: list[dict[str, Any]] = []
    for (dataset_id, cell_id), group in _cycle_table().groupby(
        ["dataset_id", "cell_id"], sort=True
    ):
        group = group.sort_values("cycle_id")
        soh = group["soh_pct"].to_numpy(dtype="float64")
        cycle_positions = group["cycle_position"].to_numpy(dtype="int64")
        first_position = crossing_position(soh, "first", eol_pct=eol_pct)
        sustained_position = crossing_position(soh, "sustained3", eol_pct=eol_pct)
        deltas = np.diff(soh)
        rows.append(
            {
                "dataset_id": dataset_id,
                "chemistry": str(group["chemistry"].iloc[0]),
                "cell_id": cell_id,
                "n_cycles": int(len(group)),
                "first_cycle_id": int(group["cycle_id"].iloc[0]),
                "last_cycle_id": int(group["cycle_id"].iloc[-1]),
                "contiguous_cycle_ids": bool(
                    (np.diff(group["cycle_id"].to_numpy()) == 1).all()
                ),
                "soh_first_pct": float(soh[0]),
                "soh_last_pct": float(soh[-1]),
                "soh_min_pct": float(np.nanmin(soh)),
                "soh_max_pct": float(np.nanmax(soh)),
                "starts_below_threshold": bool(soh[0] < float(eol_pct)),
                "total_fade_pct": float(soh[0] - np.nanmin(soh)),
                "mean_delta_pct": float(np.mean(deltas)) if deltas.size else None,
                "first_crossing_position": first_position,
                "sustained3_crossing_position": sustained_position,
                "pre_eol_cycles": int(
                    first_position if first_position is not None else len(group)
                ),
                "censored": first_position is None,
                "eol_threshold_pct": float(eol_pct),
            }
        )
    return pd.DataFrame(rows)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def support_status(summary_row: dict[str, Any]) -> dict[str, Any]:
    """Assign an honest RUL support level for one cell trajectory."""
    chemistry = str(summary_row["chemistry"])
    n_cycles = int(summary_row["n_cycles"])
    crossing = summary_row.get("first_crossing_position")
    crosses = None if _missing(crossing) else int(crossing)
    dataset_id = str(summary_row["dataset_id"])

    if n_cycles < MIN_HISTORY_CYCLES:
        return {
            "status": "UNAVAILABLE",
            "reason": (
                f"Only {n_cycles} observed cycles; at least {MIN_HISTORY_CYCLES} "
                "are required before any trajectory estimate is attempted."
            ),
        }
    if summary_row.get("starts_below_threshold"):
        return {
            "status": "UNAVAILABLE",
            "reason": (
                "The cell is already below the threshold in its first observed "
                "cycle, so no genuine end-of-life transition is observed."
            ),
        }
    if crosses is not None:
        pre_eol = int(summary_row["pre_eol_cycles"])
        if pre_eol < MIN_EOL_CYCLES:
            return {
                "status": "UNAVAILABLE",
                "reason": (
                    f"Observed crossing after only {pre_eol} pre-EOL cycles; "
                    f"at least {MIN_EOL_CYCLES} are required for an RUL target."
                ),
            }
        return {
            "status": "EXPERIMENTAL",
            "reason": (
                f"Genuine {chemistry} sequence with an observed {EOL_PCT:.0f}% "
                f"SOH crossing at cycle position {crosses}. Validation is "
                "leave-one-cell-out over a very small number of cells, so the "
                "estimate is experimental and not approved for operation."
            ),
        }
    if n_cycles < MIN_CENSORED_CYCLES:
        return {
            "status": "UNAVAILABLE",
            "reason": (
                f"No threshold crossing and only {n_cycles} observed cycles; at "
                f"least {MIN_CENSORED_CYCLES} are required before a censored "
                "projection is even attempted."
            ),
        }
    return {
        "status": "EXPERIMENTAL_CENSORED",
        "reason": (
            f"Genuine {chemistry} sequence but no observed {EOL_PCT:.0f}% SOH "
            "crossing within the available history. Only a censored projection "
            "can be produced, and it has no validated accuracy."
        ),
    }


def sequential_support(eol_pct: float = EOL_PCT) -> dict[str, Any]:
    """Support matrix for every dataset, including snapshot datasets."""
    summary = cell_summary(eol_pct=eol_pct)
    cells: list[dict[str, Any]] = []
    for record in summary.to_dict("records"):
        record.update(support_status(record))
        cells.append(record)

    datasets: dict[str, dict[str, Any]] = {}
    for dataset_id, group in summary.groupby("dataset_id", sort=True):
        statuses = [support_status(row) for row in group.to_dict("records")]
        statuses_set = {item["status"] for item in statuses}
        if "EXPERIMENTAL" in statuses_set:
            status = "EXPERIMENTAL"
        elif "EXPERIMENTAL_CENSORED" in statuses_set:
            status = "EXPERIMENTAL_CENSORED"
        else:
            status = "UNAVAILABLE"
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "chemistry": str(group["chemistry"].iloc[0]),
            "cells": int(len(group)),
            "observed_cycles": int(group["n_cycles"].sum()),
            "cells_with_observed_crossing": int(
                sum(1 for item in statuses if item["status"] == "EXPERIMENTAL")
            ),
            "status": status,
            "validation_protocol": (
                "leave-one-cell-out" if status == "EXPERIMENTAL" else "none"
            ),
            "approved_for_use": False,
            "reason": next(
                item["reason"] for item in statuses if item["status"] == status
            ),
        }

    for dataset_id in DATASET_IDS:
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "chemistry": None,
            "cells": 0,
            "observed_cycles": 0,
            "cells_with_observed_crossing": 0,
            "status": "UNAVAILABLE",
            "validation_protocol": "none",
            "approved_for_use": False,
            "reason": (
                "Active Member 1 dataset is a snapshot table with no observed "
                "chronological end-of-life trajectory. A sequence is never "
                "manufactured from independent snapshots."
            ),
        }

    return {
        "eol_threshold_pct": float(eol_pct),
        "min_history_cycles": MIN_HISTORY_CYCLES,
        "min_eol_cycles": MIN_EOL_CYCLES,
        "min_censored_cycles": MIN_CENSORED_CYCLES,
        "loco_min_cells": LOCO_MIN_CELLS,
        "cells": cells,
        "datasets": datasets,
        "assumptions": list(CONTROLLED_ASSUMPTIONS),
        "source": {
            "path": str(MASTER_TABLE),
            "sha256": source_sha256(),
            "grains": "one row per (dataset_id, cell_id, cycle_id)",
        },
    }


def cell_histories(
    dataset_id: str,
    *,
    min_history: int = MIN_HISTORY_CYCLES,
) -> dict[str, dict[str, Any]]:
    """Return validated per-cell histories for a sequential dataset."""
    cycles = _cycle_table()
    subset = cycles.loc[cycles["dataset_id"] == dataset_id]
    if subset.empty:
        raise SequentialDataError(f"No sequential data for dataset {dataset_id!r}")
    histories: dict[str, dict[str, Any]] = {}
    for cell_id, group in subset.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_id")
        soh = group["soh_pct"].to_numpy(dtype="float64")
        cycle_ids = group["cycle_id"].to_numpy(dtype="int64")
        if len(soh) < min_history:
            continue
        if not np.isfinite(soh).all():
            raise SequentialDataError(f"Non-finite SOH values in cell {cell_id!r}")
        histories[cell_id] = {
            "cell_id": cell_id,
            "dataset_id": dataset_id,
            "chemistry": str(group["chemistry"].iloc[0]),
            "cycle_ids": cycle_ids,
            "soh": soh,
            "positions": np.arange(len(soh), dtype="int64"),
        }
    return histories


__all__ = [
    "CONTROLLED_ASSUMPTIONS",
    "LOCO_MIN_CELLS",
    "MASTER_TABLE",
    "MIN_CENSORED_CYCLES",
    "MIN_EOL_CYCLES",
    "MIN_HISTORY_CYCLES",
    "SEQUENTIAL_DATASET_IDS",
    "SequentialDataError",
    "cell_histories",
    "cell_summary",
    "load_cell",
    "load_cycles",
    "sequential_support",
    "source_sha256",
    "support_status",
]
