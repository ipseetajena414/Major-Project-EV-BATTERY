"""Na-ion-only cycle-level features built from audited telemetry.

Design rules enforced here:
  * One row per (cell_id, cycle_id). Cycle-level SOH/capacity labels are
    broadcast across telemetry rows in the source table and are never
    treated as independent observations.
  * The target is the next *observed* record, not a fixed physical cycle
    step. Na-ion cycle ids happen to be contiguous, so both coincide, but
    the code does not assume that.
  * Features come from measured voltage and elapsed time, from the sign of
    measured current (charge vs discharge) and from the cycle index. No
    current-derived Ah/EFC/C-rate, no SOC, no calendar age, no temperature
    and no paper-formula coefficients are produced.
  * `soh_pct` is used only as the current-state input and as the target.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[2]
PARQUET = REPO_ROOT / "master_table.parquet"

CHEMISTRY = "Na-ion"
DATASET_ID = "Zenodo_IonTech_NaIon"
PROTOCOL_ID = "NaIon_Standard_Cycling"

# Predeclared, non-redundant model inputs. Every one is V, seconds, a
# ratio of seconds, a cycle index or the current-state SOH label.
FEATURE_COLUMNS = [
    "current_soh_pct",
    "cycle_position",
    "chg_duration_frac",
    "chg_v_mean",
    "dis_v_mean",
    "chg_v_q50",
    "dis_v_q50",
    "chg_v_slope_norm",
    "dis_v_slope_norm",
    "chg_v_range",
    "dis_v_range",
    "transition_voltage",
]

# Sensitivity set used in the prototype, kept only to show that exact
# linear redundancy does not drive the conclusion. Not the primary model.
REDUNDANT_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "v_mean_diff",
    "v_q50_diff",
    "v_slope_diff",
]

FORBIDDEN_FEATURES = {
    "capacity_ah",
    "soh_pct",
    "temp_c",
    "temperature",
    "discharge_ah_integrated",
    "charge_ah_integrated",
    "ah_throughput",
    "efc",
    "c_rate",
    "soc",
    "calendar_age_days",
}

TELEMETRY_COLUMNS = [
    "chemistry",
    "dataset_id",
    "cell_id",
    "protocol_id",
    "cycle_id",
    "time_sec",
    "voltage_v",
    "current_a",
    "temp_c",
    "capacity_ah",
    "soh_pct",
    "quality_flag",
    "source_file",
]


# ── Provenance ───────────────────────────────────────────────────────────────


def feature_provenance() -> pd.DataFrame:
    rows = [
        (
            "current_soh_pct",
            "state",
            "SOH of the cycle at which the prediction is made",
            "soh_pct",
            "percent",
            "cycle-level label read at time t; the only label-derived input",
            "derived_from_cycle_label",
        ),
        (
            "cycle_position",
            "time",
            "0-based position of the cycle in that cell's record",
            "cycle_id",
            "cycles",
            "cumulative count of observed cycles per cell",
            "derived_from_measured_index",
        ),
        (
            "chg_duration_frac",
            "time",
            "charge-phase duration divided by total cycle duration",
            "time_sec, current_a",
            "fraction",
            "phase split by sign(current_a); duration = phase time span",
            "derived_from_measured",
        ),
        (
            "chg_v_mean",
            "voltage",
            "mean voltage while current_a > 0",
            "voltage_v, current_a",
            "V",
            "arithmetic mean over charge-phase rows",
            "derived_from_measured",
        ),
        (
            "dis_v_mean",
            "voltage",
            "mean voltage while current_a < 0",
            "voltage_v, current_a",
            "V",
            "arithmetic mean over discharge-phase rows",
            "derived_from_measured",
        ),
        (
            "chg_v_q50",
            "voltage",
            "median voltage while current_a > 0",
            "voltage_v, current_a",
            "V",
            "median over charge-phase rows",
            "derived_from_measured",
        ),
        (
            "dis_v_q50",
            "voltage",
            "median voltage while current_a < 0",
            "voltage_v, current_a",
            "V",
            "median over discharge-phase rows",
            "derived_from_measured",
        ),
        (
            "chg_v_slope_norm",
            "voltage",
            "fitted voltage change across the charge phase",
            "voltage_v, time_sec, current_a",
            "V",
            "OLS slope vs elapsed time inside phase, multiplied by phase span",
            "derived_from_measured",
        ),
        (
            "dis_v_slope_norm",
            "voltage",
            "fitted voltage change across the discharge phase",
            "voltage_v, time_sec, current_a",
            "V",
            "OLS slope vs elapsed time inside phase, multiplied by phase span",
            "derived_from_measured",
        ),
        (
            "chg_v_range",
            "voltage",
            "max minus min voltage during charge",
            "voltage_v, current_a",
            "V",
            "peak-to-peak over charge-phase rows",
            "derived_from_measured",
        ),
        (
            "dis_v_range",
            "voltage",
            "max minus min voltage during discharge",
            "voltage_v, current_a",
            "V",
            "peak-to-peak over discharge-phase rows",
            "derived_from_measured",
        ),
        (
            "transition_voltage",
            "voltage",
            "voltage at the final charge-phase row before discharge starts",
            "voltage_v, current_a, time_sec",
            "V",
            "last voltage with current_a > 0 in the cycle",
            "derived_from_measured",
        ),
        (
            "v_mean_diff",
            "voltage",
            "chg_v_mean minus dis_v_mean (sensitivity only)",
            "voltage_v, current_a",
            "V",
            "exact linear combination of retained features",
            "derived_from_measured",
        ),
        (
            "v_q50_diff",
            "voltage",
            "chg_v_q50 minus dis_v_q50 (sensitivity only)",
            "voltage_v, current_a",
            "V",
            "exact linear combination of retained features",
            "derived_from_measured",
        ),
        (
            "v_slope_diff",
            "voltage",
            "chg_v_slope_norm minus dis_v_slope_norm (sensitivity only)",
            "voltage_v, time_sec, current_a",
            "V",
            "exact linear combination of retained features",
            "derived_from_measured",
        ),
    ]
    table = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "family",
            "description",
            "source_columns",
            "units",
            "derivation",
            "trust_level",
        ],
    )
    table["used_in_primary_model"] = table["feature"].isin(FEATURE_COLUMNS)
    return table


# ── Loading and audit ────────────────────────────────────────────────────────


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_naion_raw(parquet: Path = PARQUET) -> pd.DataFrame:
    raw = pd.read_parquet(Path(parquet))
    missing = [c for c in TELEMETRY_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(f"master table is missing columns: {missing}")
    raw = raw[raw["chemistry"] == CHEMISTRY].copy()
    if raw.empty:
        raise ValueError("no Na-ion rows found in master table")
    raw["cell_id"] = raw["cell_id"].astype(str)
    raw["_row_order"] = np.arange(len(raw), dtype=np.int64)
    return raw.sort_values(
        ["cell_id", "cycle_id", "time_sec", "_row_order"], kind="mergesort"
    ).reset_index(drop=True)


def _cycle_groups(raw: pd.DataFrame):
    return raw.groupby(["cell_id", "cycle_id"], sort=True)


def label_broadcast_audit(raw: pd.DataFrame) -> dict:
    """Confirm cycle labels are constant within a cycle and are not row-level."""
    per_cycle = _cycle_groups(raw).agg(
        n_rows=("soh_pct", "size"),
        soh_unique=("soh_pct", "nunique"),
        cap_unique=("capacity_ah", "nunique"),
    )
    identity = (raw["capacity_ah"] - raw["soh_pct"] / 100.0).abs()
    return {
        "cycles_checked": int(len(per_cycle)),
        "cycles_with_nonconstant_soh": int((per_cycle["soh_unique"] != 1).sum()),
        "cycles_with_nonconstant_capacity": int((per_cycle["cap_unique"] != 1).sum()),
        "max_abs_capacity_minus_soh_over_100": float(identity.max()),
        "max_rows_per_cycle": int(per_cycle["n_rows"].max()),
    }


def duplicate_audit(raw: pd.DataFrame) -> dict:
    within = raw.duplicated(subset=["cell_id", "cycle_id", "time_sec"], keep=False)
    cross = (
        raw.duplicated(subset=["cell_id", "time_sec"], keep=False)
        & ~within
    )
    cross_times = (
        raw[cross]
        .groupby(["cell_id", "time_sec"])["cycle_id"]
        .nunique()
    )
    return {
        "rows": int(len(raw)),
        "within_cycle_duplicate_rows": int(within.sum()),
        "cross_cycle_shared_timestamp_rows": int(cross.sum()),
        "cross_cycle_shared_timestamps": int((cross_times > 1).sum()),
        "note": (
            "shared timestamps are cycle-boundary rows (end of cycle N equals "
            "start of cycle N+1); no duplicates exist inside a cycle"
        ),
    }


def phase_structure_audit(raw: pd.DataFrame) -> dict:
    failures = []
    zero_rows = 0
    transition_counts = []
    for (cell_id, cycle_id), group in _cycle_groups(raw):
        current = group["current_a"].to_numpy(dtype=float)
        zero_rows += int((current == 0).sum())
        nz = np.sign(current[np.sign(current) != 0]).astype(int)
        if len(nz) == 0:
            failures.append({"cell_id": cell_id, "cycle_id": int(cycle_id), "reason": "no nonzero current"})
            continue
        changes = int((np.diff(nz) != 0).sum())
        transition_counts.append(changes)
        if changes != 1:
            failures.append(
                {"cell_id": cell_id, "cycle_id": int(cycle_id), "reason": f"{changes} sign changes"}
            )
        elif nz[0] != 1:
            failures.append(
                {"cell_id": cell_id, "cycle_id": int(cycle_id), "reason": "discharge precedes charge"}
            )
    return {
        "cycles_checked": int(len(_cycle_groups(raw).groups)),
        "zero_current_rows": zero_rows,
        "sign_change_counts": sorted(set(transition_counts)),
        "failures": failures,
    }


def unit_validation(raw: pd.DataFrame) -> dict:
    """Integrate current over time per phase and compare with capacity_ah.

    This is a unit check only. The integrated values never enter a model.
    """
    dis_diff, chg_diff = [], []
    for _, group in _cycle_groups(raw):
        group = group.sort_values("time_sec", kind="mergesort")
        t = group["time_sec"].to_numpy(dtype=float)
        i = group["current_a"].to_numpy(dtype=float)
        cap = float(group["capacity_ah"].iloc[0])
        dt = np.diff(t, prepend=t[0])
        dt = np.clip(dt, 0.0, None)
        dis_ah = float(np.sum(np.clip(-i, 0.0, None) * dt) / 3600.0)
        chg_ah = float(np.sum(np.clip(i, 0.0, None) * dt) / 3600.0)
        dis_diff.append(dis_ah - cap)
        chg_diff.append(chg_ah - cap)
    dis_diff = np.asarray(dis_diff)
    chg_diff = np.asarray(chg_diff)
    return {
        "cycles_checked": int(len(dis_diff)),
        "discharge_ah_minus_capacity_ah": {
            "max_abs": float(np.abs(dis_diff).max()),
            "mean_abs": float(np.abs(dis_diff).mean()),
        },
        "charge_ah_minus_capacity_ah": {
            "max_abs": float(np.abs(chg_diff).max()),
            "mean_abs": float(np.abs(chg_diff).mean()),
        },
        "conclusion": (
            "integrated phase charge matches nominal 1 Ah capacity within "
            "~0.03 Ah, so current_a is amperes and capacity_ah is ampere-hours; "
            "the residual is consistent with rounding and boundary rows"
        ),
    }


def audit_naion(raw: pd.DataFrame, parquet: Path | None = None) -> dict:
    per_cell = (
        raw.groupby("cell_id", sort=True)
        .agg(
            rows=("voltage_v", "size"),
            cycles=("cycle_id", "nunique"),
            cycle_min=("cycle_id", "min"),
            cycle_max=("cycle_id", "max"),
            current_min=("current_a", "min"),
            current_max=("current_a", "max"),
            temp_non_null=("temp_c", "count"),
            voltage_min=("voltage_v", "min"),
            voltage_max=("voltage_v", "max"),
        )
        .reset_index()
    )
    soh_summary = []
    for cell_id, group in raw.groupby("cell_id", sort=True):
        series = (
            group.groupby("cycle_id", sort=True)["soh_pct"].first().to_numpy(dtype=float)
        )
        diffs = np.diff(series)
        soh_summary.append(
            {
                "cell_id": str(cell_id),
                "cycles": int(len(series)),
                "soh_first": float(series[0]),
                "soh_last": float(series[-1]),
                "soh_change_pp": float(series[-1] - series[0]),
                "decreasing_steps": int((diffs < 0).sum()),
                "increasing_steps": int((diffs > 0).sum()),
                "flat_steps": int((diffs == 0).sum()),
            }
        )

    monotonic_violations = 0
    rows_per_cycle = []
    for _, group in _cycle_groups(raw):
        t = group["time_sec"].to_numpy(dtype=float)
        monotonic_violations += int((np.diff(t) < 0).sum())
        rows_per_cycle.append(len(group))

    quality = raw["quality_flag"].fillna("NONE").value_counts().to_dict()
    return {
        "chemistry": CHEMISTRY,
        "dataset_id": DATASET_ID,
        "protocol_id": PROTOCOL_ID,
        "parquet": str(Path(parquet).resolve()) if parquet else None,
        "parquet_sha256": sha256(parquet) if parquet else None,
        "rows": int(len(raw)),
        "cells": sorted(raw["cell_id"].unique().tolist()),
        "n_cells": int(raw["cell_id"].nunique()),
        "n_cycles": int(len(_cycle_groups(raw).groups)),
        "rows_per_cell": {
            str(k): int(v) for k, v in raw.groupby("cell_id").size().items()
        },
        "per_cell": per_cell.to_dict(orient="records"),
        "soh_trajectories": soh_summary,
        "rows_per_cycle_min": int(min(rows_per_cycle)),
        "rows_per_cycle_max": int(max(rows_per_cycle)),
        "missing_values": {c: int(v) for c, v in raw.isna().sum().items()},
        "quality_flags": quality,
        "temperature_available": bool(raw["temp_c"].notna().any()),
        "time_monotonicity_violations": monotonic_violations,
        "label_broadcast": label_broadcast_audit(raw),
        "duplicates": duplicate_audit(raw),
        "phase_structure": phase_structure_audit(raw),
        "units": unit_validation(raw),
        "soh_change_max_pp": float(
            max(abs(s["soh_change_pp"]) for s in soh_summary)
        ),
        "eol_threshold_pct": 80.0,
        "cells_at_or_below_eol": sorted(
            s["cell_id"] for s in soh_summary if s["soh_last"] <= 80.0
        ),
    }


# ── Cycle-level features ─────────────────────────────────────────────────────


def phase_masks(current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    current = np.asarray(current, dtype=float)
    return current > 0, current < 0


def _assert_phase_structure(cell_id: str, cycle_id: int, t, current) -> None:
    chg, dis = phase_masks(current)
    if not chg.any() or not dis.any():
        raise ValueError(f"{cell_id} cycle {cycle_id}: missing charge or discharge phase")
    nz = np.sign(current[np.sign(current) != 0]).astype(int)
    changes = np.flatnonzero(np.diff(nz) != 0)
    if len(changes) != 1 or nz[0] != 1:
        raise ValueError(
            f"{cell_id} cycle {cycle_id}: expected exactly one charge->discharge transition"
        )
    if t[chg].max() > t[dis].min():
        raise ValueError(f"{cell_id} cycle {cycle_id}: phases overlap in time")


def _phase_stats(t: np.ndarray, v: np.ndarray) -> dict:
    span = float(t[-1] - t[0])
    if len(t) >= 2 and span > 0:
        slope = float(np.polyfit(t - t[0], v, 1)[0])
    else:
        slope = 0.0
    return {
        "duration_s": span,
        "v_mean": float(np.mean(v)),
        "v_q50": float(np.median(v)),
        "v_range": float(np.max(v) - np.min(v)),
        "v_slope_per_s": slope,
        "v_slope_norm": slope * span,
    }


def extract_cycle_features(raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (cell_id, cycle_id), group in _cycle_groups(raw):
        g = group.sort_values(["time_sec", "_row_order"], kind="mergesort")
        t = g["time_sec"].to_numpy(dtype=float)
        v = g["voltage_v"].to_numpy(dtype=float)
        i = g["current_a"].to_numpy(dtype=float)
        _assert_phase_structure(cell_id, int(cycle_id), t, i)

        chg, dis = phase_masks(i)
        chg_stats = _phase_stats(t[chg], v[chg])
        dis_stats = _phase_stats(t[dis], v[dis])
        cycle_span = float(t.max() - t.min())
        if cycle_span <= 0:
            raise ValueError(f"{cell_id} cycle {cycle_id}: non-positive cycle duration")

        soh_values = g["soh_pct"].dropna().unique()
        cap_values = g["capacity_ah"].dropna().unique()
        if len(soh_values) != 1 or len(cap_values) != 1:
            raise ValueError(
                f"{cell_id} cycle {cycle_id}: cycle labels are not constant"
            )

        records.append(
            {
                "cell_id": str(cell_id),
                "cycle_id": int(cycle_id),
                "n_rows": int(len(g)),
                "time_start": float(t.min()),
                "time_end": float(t.max()),
                "cycle_duration_s": cycle_span,
                "zero_current_rows": int((i == 0).sum()),
                "chg_duration_s": chg_stats["duration_s"],
                "dis_duration_s": dis_stats["duration_s"],
                "chg_duration_frac": chg_stats["duration_s"] / cycle_span,
                "transition_voltage": float(v[chg][-1]),
                "chg_v_mean": chg_stats["v_mean"],
                "dis_v_mean": dis_stats["v_mean"],
                "chg_v_q50": chg_stats["v_q50"],
                "dis_v_q50": dis_stats["v_q50"],
                "chg_v_slope_norm": chg_stats["v_slope_norm"],
                "dis_v_slope_norm": dis_stats["v_slope_norm"],
                "chg_v_range": chg_stats["v_range"],
                "dis_v_range": dis_stats["v_range"],
                "v_mean_diff": chg_stats["v_mean"] - dis_stats["v_mean"],
                "v_q50_diff": chg_stats["v_q50"] - dis_stats["v_q50"],
                "v_slope_diff": chg_stats["v_slope_norm"] - dis_stats["v_slope_norm"],
                "capacity_ah": float(cap_values[0]),
                "soh_pct": float(soh_values[0]),
                "quality_flag": str(g["quality_flag"].iloc[0])
                if pd.notna(g["quality_flag"].iloc[0])
                else "NONE",
            }
        )

    cycles = pd.DataFrame.from_records(records)
    cycles = cycles.sort_values(["cell_id", "cycle_id"], kind="mergesort").reset_index(drop=True)
    cycles["cycle_position"] = cycles.groupby("cell_id").cumcount()
    cycles["current_soh_pct"] = cycles["soh_pct"]
    return cycles


# ── Next-observed-record pairs ───────────────────────────────────────────────


def build_pairs(cycles: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_id", kind="mergesort").reset_index(drop=True)
        if len(group) < 2:
            continue
        current = group.iloc[:-1].copy().reset_index(drop=True)
        nxt = group.iloc[1:].copy().reset_index(drop=True)

        current["target_soh_pct"] = nxt["soh_pct"].to_numpy(dtype=float)
        current["target_cycle_id"] = nxt["cycle_id"].to_numpy(dtype=int)
        current["target_cycle_position"] = nxt["cycle_position"].to_numpy(dtype=int)
        current["prev_soh_pct"] = current["soh_pct"].shift(1)
        current["pair_index"] = np.arange(len(current), dtype=int)
        current["target_record_position"] = current["pair_index"] + 1
        current["cycle_gap"] = (
            current["target_cycle_id"] - current["cycle_id"]
        ).astype(int)
        current["is_gap_pair"] = current["cycle_gap"] > 1
        frames.append(current)
    if not frames:
        raise ValueError("no cell has at least two cycles")
    return pd.concat(frames, ignore_index=True)


def feature_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    matrix = frame[columns].astype(float)
    return matrix.replace([np.inf, -np.inf], np.nan)


def assert_no_forbidden_features(columns: list[str]) -> None:
    bad = [c for c in columns if c in FORBIDDEN_FEATURES or c.startswith("temp_")]
    if bad:
        raise ValueError(f"forbidden features requested: {bad}")
    if "soh_pct" in columns:
        raise ValueError("raw soh_pct must not be a feature; use current_soh_pct")
