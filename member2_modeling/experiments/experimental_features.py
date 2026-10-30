"""Isolated experimental features for Member 2.

Only deterministic, unit-verified features are computed. Nothing here writes
production artifacts or fabricates missing values.

Cleared for implementation:
  - NMC discharge Ah throughput (current_a verified in A)
  - NMC cumulative EFC (project adaptation of C-Eq10-12)
  - NMC time-weighted thermal exposure (temperature present)
  - NMC voltage at discharge-capacity fractions
  - NMC dV/dQ mid-range diagnostic (diagnostic only; not a model feature)

Blocked:
  - LFP current integrals and EFC (current units underdetermined)
  - LFP/Na-ion thermal exposure (temperature missing)
  - All calendar/cycle capacity-loss formulas (missing parameters/inputs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CYCLE_KEY = ["dataset_id", "cell_id", "protocol_id", "cycle_id"]

RATED_CAPACITY_AH = {"NMC": 2.0, "LFP": 1.25}
THERMAL_BAND_C = (20.0, 30.0)

# Features that require a unit-verified current measurement.
CURRENT_DERIVED_FEATURES = [
    "ah_throughput",
    "energy_wh",
    "avg_power_w",
    "i_mean",
    "i_std",
    "i_min",
    "i_max",
]

NMC_NEW_FEATURES = [
    "ah_discharge_trapz",
    "efc_cumulative",
    "thermal_hours_outside_20_30",
    "thermal_degree_hours_above_30",
    "thermal_degree_hours_below_20",
    "v_at_q20",
    "v_at_q50",
    "v_at_q80",
]

FEATURE_PROVENANCE = {
    "ah_discharge_trapz": {
        "unit": "Ah",
        "type": "telemetry-derived",
        "source": "trapezoidal integral of |current_a| over dt during discharge",
        "chemistries": ["NMC"],
        "requires": ["current_a verified in A", "time_sec"],
    },
    "efc_cumulative": {
        "unit": "equivalent full cycles",
        "type": "project adaptation of C-Eq10-12",
        "source": "cumulative observed discharge Ah / 2.0 Ah NMC rated capacity",
        "chemistries": ["NMC"],
        "requires": ["ah_discharge_trapz"],
    },
    "thermal_hours_outside_20_30": {
        "unit": "h",
        "type": "telemetry-derived",
        "source": "time-weighted exposure with midpoint temperature outside 20-30 C",
        "chemistries": ["NMC"],
        "requires": ["temp_c", "time_sec"],
    },
    "thermal_degree_hours_above_30": {
        "unit": "degC*h",
        "type": "telemetry-derived",
        "source": "time-weighted integral of max(T-30, 0)",
        "chemistries": ["NMC"],
        "requires": ["temp_c", "time_sec"],
    },
    "thermal_degree_hours_below_20": {
        "unit": "degC*h",
        "type": "telemetry-derived",
        "source": "time-weighted integral of max(20-T, 0)",
        "chemistries": ["NMC"],
        "requires": ["temp_c", "time_sec"],
    },
    "v_at_q20": {
        "unit": "V",
        "type": "telemetry-derived",
        "source": "voltage interpolated at 20% of cumulative discharge Ah",
        "chemistries": ["NMC"],
        "requires": ["current_a verified in A", "time_sec", "voltage_v"],
    },
    "v_at_q50": {
        "unit": "V",
        "type": "telemetry-derived",
        "source": "voltage interpolated at 50% of cumulative discharge Ah",
        "chemistries": ["NMC"],
        "requires": ["current_a verified in A", "time_sec", "voltage_v"],
    },
    "v_at_q80": {
        "unit": "V",
        "type": "telemetry-derived",
        "source": "voltage interpolated at 80% of cumulative discharge Ah",
        "chemistries": ["NMC"],
        "requires": ["current_a verified in A", "time_sec", "voltage_v"],
    },
}


def _sorted_telemetry(group: pd.DataFrame) -> pd.DataFrame:
    """Sort by time and drop duplicate timestamps without reordering provenance."""
    group = group.sort_values("time_sec", kind="mergesort")
    return group.drop_duplicates(subset=["time_sec"], keep="first")


def _discharge(group: pd.DataFrame) -> pd.DataFrame:
    return group[group["current_a"] < 0]


def _trapezoid_ah(t: np.ndarray, i: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    dt = np.diff(t)
    dt = np.clip(dt, 0.0, None)
    return float(np.sum(0.5 * (np.abs(i[:-1]) + np.abs(i[1:])) * dt) / 3600.0)


def throughput_features(group: pd.DataFrame, chemistry: str) -> dict:
    """Compute discharge-only Ah throughput when current units are verified."""
    out = {
        "ah_discharge_trapz": np.nan,
        "discharge_duration_s": np.nan,
        "discharge_mean_current_a": np.nan,
        "discharge_n_samples": 0,
        "throughput_unit_verified": False,
        "throughput_status": "BLOCKED_CURRENT_UNIT_UNDERDETERMINED",
    }
    if chemistry != "NMC":
        return out

    g = _sorted_telemetry(group)
    d = _discharge(g)
    if len(d) < 2:
        out["throughput_status"] = "INSUFFICIENT_DISCHARGE_SAMPLES"
        return out

    t = d["time_sec"].to_numpy(dtype=float)
    i = d["current_a"].to_numpy(dtype=float)
    dt = np.diff(t)
    out.update(
        {
            "ah_discharge_trapz": _trapezoid_ah(t, i),
            "discharge_duration_s": float(t[-1] - t[0]),
            "discharge_mean_current_a": float(np.mean(i)),
            "discharge_n_samples": int(len(d)),
            "throughput_unit_verified": True,
            "throughput_status": "OK_NMC_CURRENT_A_VERIFIED",
        }
    )
    return out


def thermal_features(group: pd.DataFrame, chemistry: str) -> dict:
    """Time-weighted thermal exposure; NMC only because temperature is measured."""
    out = {
        "thermal_hours_outside_20_30": np.nan,
        "thermal_degree_hours_above_30": np.nan,
        "thermal_degree_hours_below_20": np.nan,
        "thermal_time_weighted": True,
        "thermal_status": "BLOCKED_TEMPERATURE_UNMEASURED",
    }
    if chemistry != "NMC" or "temp_c" not in group.columns:
        return out

    g = _sorted_telemetry(group)
    if g["temp_c"].isna().all():
        return out

    t = g["time_sec"].to_numpy(dtype=float)
    temp = g["temp_c"].to_numpy(dtype=float)
    dt = np.clip(np.diff(t), 0.0, None)
    if len(dt) == 0:
        out["thermal_status"] = "INSUFFICIENT_SAMPLES"
        return out

    t_mid = 0.5 * (temp[:-1] + temp[1:])
    low, high = THERMAL_BAND_C
    outside = (t_mid < low) | (t_mid > high)
    out.update(
        {
            "thermal_hours_outside_20_30": float(np.sum(dt[outside]) / 3600.0),
            "thermal_degree_hours_above_30": float(
                np.sum(np.maximum(t_mid - high, 0.0) * dt) / 3600.0
            ),
            "thermal_degree_hours_below_20": float(
                np.sum(np.maximum(low - t_mid, 0.0) * dt) / 3600.0
            ),
            "thermal_status": "OK_NMC_TEMPERATURE_MEASURED",
        }
    )
    return out


def capacity_fraction_voltages(
    group: pd.DataFrame, chemistry: str, fractions=(0.2, 0.5, 0.8)
) -> dict:
    """Voltage interpolated at fixed fractions of the cycle's discharge Ah."""
    out = {f"v_at_q{int(f * 100)}": np.nan for f in fractions}
    out["capacity_fraction_status"] = "BLOCKED_CURRENT_UNIT_UNDERDETERMINED"
    if chemistry != "NMC":
        return out

    d = _discharge(_sorted_telemetry(group))
    if len(d) < 3:
        out["capacity_fraction_status"] = "INSUFFICIENT_DISCHARGE_SAMPLES"
        return out

    t = d["time_sec"].to_numpy(dtype=float)
    i = d["current_a"].to_numpy(dtype=float)
    v = d["voltage_v"].to_numpy(dtype=float)
    dt = np.clip(np.diff(t), 0.0, None)
    dq = 0.5 * (np.abs(i[:-1]) + np.abs(i[1:])) * dt / 3600.0
    q = np.concatenate([[0.0], np.cumsum(dq)])
    if q[-1] <= 0:
        out["capacity_fraction_status"] = "ZERO_DISCHARGE_CAPACITY"
        return out

    q_frac = q / q[-1]
    for f in fractions:
        if q_frac[0] <= f <= q_frac[-1]:
            out[f"v_at_q{int(f * 100)}"] = float(np.interp(f, q_frac, v))
        else:
            out[f"v_at_q{int(f * 100)}"] = np.nan
    out["capacity_fraction_status"] = "OK_NMC_CURRENT_A_VERIFIED"
    return out


def dqdv_diagnostic(group: pd.DataFrame, chemistry: str) -> dict:
    """Mid-range dV/dQ summary; diagnostic only, never used as a model feature."""
    out = {
        "dqdv_median_mid": np.nan,
        "dqdv_iqr_mid": np.nan,
        "dqdv_q_points": 0,
        "dqdv_status": "BLOCKED_CURRENT_UNIT_UNDERDETERMINED",
    }
    if chemistry != "NMC":
        return out

    d = _discharge(_sorted_telemetry(group))
    if len(d) < 30:
        out["dqdv_status"] = "INSUFFICIENT_DISCHARGE_SAMPLES"
        return out

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        out["dqdv_status"] = "SCIPY_UNAVAILABLE"
        return out

    t = d["time_sec"].to_numpy(dtype=float)
    i = d["current_a"].to_numpy(dtype=float)
    v = d["voltage_v"].to_numpy(dtype=float)
    dt = np.clip(np.diff(t), 0.0, None)
    dq = 0.5 * (np.abs(i[:-1]) + np.abs(i[1:])) * dt / 3600.0
    q = np.concatenate([[0.0], np.cumsum(dq)])
    if q[-1] <= 0:
        out["dqdv_status"] = "ZERO_DISCHARGE_CAPACITY"
        return out

    q_frac = q / q[-1]
    mask = (q_frac >= 0.2) & (q_frac <= 0.8)
    if mask.sum() < 20:
        out["dqdv_status"] = "INSUFFICIENT_MID_RANGE_POINTS"
        return out

    q_grid = np.linspace(q[mask][0], q[mask][-1], 60)
    v_grid = np.interp(q_grid, q[mask], v[mask])
    window = 11 if len(v_grid) >= 11 else max(5, (len(v_grid) // 2) * 2 + 1)
    if window >= 5:
        v_grid = savgol_filter(v_grid, window, 2)
    dv_dq = np.gradient(v_grid, q_grid)
    out.update(
        {
            "dqdv_median_mid": float(np.median(dv_dq)),
            "dqdv_iqr_mid": float(np.percentile(dv_dq, 75) - np.percentile(dv_dq, 25)),
            "dqdv_q_points": int(mask.sum()),
            "dqdv_status": "OK_NMC_DIAGNOSTIC_ONLY",
        }
    )
    return out


def compute_cycle_extra_features(telemetry: pd.DataFrame) -> pd.DataFrame:
    """Compute extra features per cycle without touching label columns."""
    records = []
    for keys, group in telemetry.groupby(CYCLE_KEY, sort=False):
        chemistry = str(group["chemistry"].iloc[0])
        row = {
            "dataset_id": keys[0],
            "cell_id": keys[1],
            "protocol_id": keys[2],
            "cycle_id": keys[3],
        }
        row.update(throughput_features(group, chemistry))
        row.update(thermal_features(group, chemistry))
        row.update(capacity_fraction_voltages(group, chemistry))
        row.update(dqdv_diagnostic(group, chemistry))
        records.append(row)
    return pd.DataFrame(records)


def add_cumulative_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative observed-record throughput/EFC; no reindexing across missing IDs."""
    out = feature_df.sort_values(["cell_id", "cycle_id"], kind="mergesort").copy()
    out["throughput_cumulative_ah"] = (
        out.groupby("cell_id")["ah_discharge_trapz"].cumsum()
    )
    rated = out["chemistry"].map(RATED_CAPACITY_AH)
    out["efc_cumulative"] = np.where(
        out["throughput_unit_verified"] & rated.notna(),
        out["throughput_cumulative_ah"] / rated,
        np.nan,
    )
    return out


def unit_safe_lfp_features(feature_cols: list[str]) -> list[str]:
    """Remove every feature that depends on an unverified LFP current unit."""
    return [c for c in feature_cols if c not in CURRENT_DERIVED_FEATURES]


def merge_extra_features(production_features: pd.DataFrame, extras: pd.DataFrame) -> pd.DataFrame:
    """Merge extra features onto the production cycle table by cycle key."""
    merge_cols = CYCLE_KEY + [c for c in extras.columns if c not in CYCLE_KEY]
    return production_features.merge(
        extras[merge_cols], on=CYCLE_KEY, how="left", validate="one_to_one"
    )
