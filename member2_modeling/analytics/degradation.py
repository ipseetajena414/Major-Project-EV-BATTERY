"""Degradation analytics from genuine chronological observations.

Observed SOH trajectories and forecasts are kept in separate rows and separate
columns.  No forecast is ever described as measured, and a projection is only
offered where the support matrix allows it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from member2_modeling.analytics.sequential import (
    cell_summary,
    load_cycles,
    support_status,
)
from member2_modeling.rul.labels import EOL_PCT
from member2_modeling.rul.service import RUL_DIR


def _rate_per_100(soh: np.ndarray, window: int | None = None) -> float | None:
    values = soh[-window:] if window is not None else soh
    if values.size < 2:
        return None
    positions = np.arange(values.size, dtype="float64")
    centered = positions - positions.mean()
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return None
    slope = float(np.sum(centered * (values - values.mean())) / denominator)
    return float(-slope * 100.0)


def build_degradation_summary(
    *,
    eol_pct: float = EOL_PCT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = cell_summary(eol_pct=eol_pct)
    cycles = load_cycles()
    rows: list[dict[str, Any]] = []
    for record in summary.to_dict("records"):
        status = support_status(record)
        cell_soh = cycles.loc[
            (cycles["dataset_id"] == record["dataset_id"])
            & (cycles["cell_id"] == record["cell_id"]),
            "soh_pct",
        ].to_numpy(dtype="float64")
        crossing = record["first_crossing_position"]
        if crossing is not None and (
            isinstance(crossing, float) and not np.isfinite(crossing)
        ):
            crossing = None
        rows.append(
            {
                "battery_id": record["cell_id"],
                "dataset_id": record["dataset_id"],
                "chemistry": record["chemistry"],
                "observed_cycles": int(record["n_cycles"]),
                "first_cycle_id": int(record["first_cycle_id"]),
                "last_cycle_id": int(record["last_cycle_id"]),
                "soh_first_pct": float(record["soh_first_pct"]),
                "soh_last_pct": float(record["soh_last_pct"]),
                "soh_min_pct": float(record["soh_min_pct"]),
                "total_fade_pct": float(record["total_fade_pct"]),
                "fade_rate_all_pct_per_100_cycles": _rate_per_100(cell_soh),
                "fade_rate_last_50_pct_per_100_cycles": _rate_per_100(cell_soh, 50),
                "eol_threshold_pct": float(eol_pct),
                "observed_crossing_position": (
                    None if crossing is None else int(crossing)
                ),
                "rul_support_status": status["status"],
                "eol_projection_available": status["status"]
                in {"EXPERIMENTAL", "EXPERIMENTAL_CENSORED"},
                "projection_artifact": (
                    str(RUL_DIR / "rul_current_predictions.csv")
                    if status["status"] == "EXPERIMENTAL"
                    else str(RUL_DIR / "rul_censored_projections.csv")
                    if status["status"] == "EXPERIMENTAL_CENSORED"
                    else None
                ),
                "history_is_observed": True,
            }
        )
    frame = pd.DataFrame(rows)
    report = {
        "eol_threshold_pct": float(eol_pct),
        "cells": int(len(frame)),
        "history_definition": (
            "Rows with record_type='observed' are measured cycle-level SOH. "
            "Rows with record_type='forecast' are model extrapolations and are "
            "never labelled measured."
        ),
        "trajectory_artifact": str(RUL_DIR / "rul_degradation_trajectories.csv"),
        "cells_with_eol_projection": int(frame["eol_projection_available"].sum()),
        "limitations": [
            "Cross-cell comparisons are affected by different protocols and "
            "different cycle counts.",
            "Fade rates are descriptive and do not control for temperature, "
            "depth of discharge or C-rate.",
            "Forecast trajectories are not validated measurements.",
        ],
    }
    return frame, report


def build_trajectory_export() -> pd.DataFrame:
    """Long-format observed and forecast SOH rows for Member 5 graphs."""
    source = RUL_DIR / "rul_degradation_trajectories.csv"
    if not source.is_file():
        cycles = load_cycles()
        return pd.DataFrame(
            {
                "battery_id": cycles["cell_id"],
                "dataset_id": cycles["dataset_id"],
                "chemistry": cycles["chemistry"],
                "cycle_index": cycles["cycle_id"],
                "cycle_position": cycles["cycle_position"],
                "soh_pct": cycles["soh_pct"],
                "record_type": "observed",
                "is_measured": True,
                "is_forecast": False,
                "model_id": None,
                "validated": True,
            }
        )
    frame = pd.read_csv(source)
    frame = frame.rename(columns={"cycle_id": "cycle_index"})
    return frame[
        [
            "battery_id",
            "dataset_id",
            "chemistry",
            "cycle_index",
            "cycle_position",
            "soh_pct",
            "record_type",
            "is_measured",
            "is_forecast",
            "model_id",
            "validated",
        ]
    ].copy()


__all__ = ["build_degradation_summary", "build_trajectory_export"]
