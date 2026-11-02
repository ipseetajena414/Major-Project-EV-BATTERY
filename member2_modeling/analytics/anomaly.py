"""Statistical anomaly detection on telemetry.

These detectors flag unusual observations.  They do **not** diagnose faults:
the project has no verified fault labels, so no row is ever labelled a failure,
a defect, or a safety event.  Sequential flags use robust z-scores and an
IsolationForest per cell; snapshot flags use robust z-scores only.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from member2_modeling.analytics.sequential import load_cycles

ROBUST_Z_THRESHOLD = 4.0
SNAPSHOT_Z_THRESHOLD = 6.0
SENSOR_KEYWORDS = (
    "voltage",
    "current",
    "temp",
    "resist",
    "capacity",
    "soh",
    "efficiency",
)
SEQUENTIAL_FEATURES = (
    "voltage_mean",
    "voltage_std",
    "current_mean",
    "current_abs_mean",
    "temp_mean",
    "temp_max",
    "capacity_ah",
)

INTERPRETATION = (
    "Statistical outlier that may merit inspection. It is not a confirmed "
    "fault, failure, or safety event."
)


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values[finite] - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.nanstd(values[finite]))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return (values - median) / scale


def detect_sequential_anomalies(
    *,
    robust_z_threshold: float = ROBUST_Z_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cycles = load_cycles()
    rows: list[dict[str, Any]] = []
    flagged = 0
    for (dataset_id, cell_id), group in cycles.groupby(
        ["dataset_id", "cell_id"], sort=True
    ):
        group = group.sort_values("cycle_id").reset_index(drop=True)
        features = [
            column
            for column in SEQUENTIAL_FEATURES
            if column in group.columns
            and group[column].notna().sum() >= 3
            and float(group[column].std(skipna=True) or 0.0) > 0
        ]
        features = [
            column
            for column in features
            if pd.notna(group[column].std(skipna=True))
            and float(group[column].std(skipna=True)) > 0
        ]
        if not features:
            continue
        z_values = {column: _robust_z(group[column].to_numpy()) for column in features}
        z_matrix = np.column_stack([z_values[column] for column in features])
        safe_z = np.where(np.isfinite(z_matrix), np.abs(z_matrix), 0.0)
        z_max = np.max(safe_z, axis=1)
        z_argmax = np.argmax(safe_z, axis=1)

        matrix = group[features].to_numpy(dtype="float64", copy=True)
        medians = np.nanmedian(matrix, axis=0)
        inds = np.where(np.isnan(matrix))
        matrix[inds] = np.take(medians, inds[1])
        if matrix.shape[0] >= 20:
            scaled = StandardScaler().fit_transform(matrix)
            forest = IsolationForest(
                n_estimators=200, contamination="auto", random_state=0
            )
            forest.fit(scaled)
            forest_score = forest.decision_function(scaled)
            forest_flag = forest.predict(scaled) == -1
        else:
            forest_score = np.full(matrix.shape[0], np.nan)
            forest_flag = np.zeros(matrix.shape[0], dtype=bool)

        for index, row in group.iterrows():
            robust_flag = bool(z_max[index] >= robust_z_threshold)
            iforest_flag = bool(forest_flag[index]) if np.isfinite(
                forest_score[index]
            ) else False
            is_anomaly = robust_flag or iforest_flag
            flagged += int(is_anomaly)
            detail = {}
            for column_index, column in enumerate(features):
                z = float(z_values[column][index])
                if abs(z) >= robust_z_threshold:
                    detail[column] = {
                        "value": float(group[column].iloc[index]),
                        "robust_z": round(z, 3),
                    }
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "cell_id": cell_id,
                    "chemistry": str(row["chemistry"]),
                    "cycle_id": int(row["cycle_id"]),
                    "cycle_position": int(row["cycle_position"]),
                    "soh_pct": float(row["soh_pct"]),
                    "max_abs_robust_z": (
                        float(z_max[index]) if np.isfinite(z_max[index]) else None
                    ),
                    "max_z_feature": features[int(z_argmax[index])],
                    "robust_outlier": robust_flag,
                    "iforest_score": (
                        float(forest_score[index])
                        if np.isfinite(forest_score[index])
                        else None
                    ),
                    "iforest_outlier": iforest_flag,
                    "is_statistical_anomaly": is_anomaly,
                    "anomaly_score": (
                        float(z_max[index]) if np.isfinite(z_max[index]) else None
                    ),
                    "anomaly_details": json.dumps(detail, sort_keys=True),
                    "interpretation": INTERPRETATION,
                    "confirmed_fault": False,
                }
            )
    flags = pd.DataFrame(rows)
    summary = {
        "method": "robust_z + per_cell_isolation_forest",
        "robust_z_threshold": float(robust_z_threshold),
        "iforest_contamination": "auto",
        "iforest_min_cycles": 20,
        "cycles_scored": int(len(flags)),
        "cycles_flagged": int(flagged),
        "flagged_fraction": float(flagged / len(flags)) if len(flags) else 0.0,
        "cells": int(
            flags[["dataset_id", "cell_id"]].drop_duplicates().shape[0]
            if len(flags)
            else 0
        ),
        "confirmed_fault_labels_available": False,
        "interpretation": INTERPRETATION,
        "limitations": [
            "No verified fault labels exist in the repository, so precision, "
            "recall and fault-detection accuracy cannot be computed.",
            "Flags describe telemetry context, not a diagnosis.",
            "IsolationForest is unsupervised and its contamination is automatic.",
        ],
    }
    return flags, summary


def detect_snapshot_anomalies(
    dataset_id: str,
    *,
    z_threshold: float = SNAPSHOT_Z_THRESHOLD,
    max_rows: int = 300,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from member1_pipeline import loader

    frame = loader.load_dataset(dataset_id)
    numeric = frame.select_dtypes(include=[np.number])
    sensor_columns = [
        column
        for column in numeric.columns
        if any(keyword in column.lower() for keyword in SENSOR_KEYWORDS)
    ]
    sensor_columns = [
        column
        for column in sensor_columns
        if numeric[column].notna().sum() >= 10 and numeric[column].std() > 0
    ]
    if not sensor_columns:
        return pd.DataFrame(), {
            "dataset_id": dataset_id,
            "sensor_columns": [],
            "rows_scored": 0,
            "rows_flagged": 0,
            "note": "No sensor-like numeric columns were available.",
        }
    z = np.column_stack([_robust_z(numeric[column].to_numpy()) for column in sensor_columns])
    abs_z = np.abs(z)
    safe_abs_z = np.where(np.isfinite(abs_z), abs_z, 0.0)
    max_z = np.max(safe_abs_z, axis=1)
    argmax = np.argmax(safe_abs_z, axis=1)
    flagged = max_z >= z_threshold
    order = np.argsort(-np.nan_to_num(max_z, nan=-np.inf))
    selected = [index for index in order if flagged[index]][:max_rows]

    entity_column = "vehicle_id" if "vehicle_id" in frame.columns else None
    chemistry_column = (
        "battery_chemistry" if "battery_chemistry" in frame.columns else None
    )
    rows = []
    for index in selected:
        detail = {
            column: round(float(abs_z[index, column_index]), 3)
            for column_index, column in enumerate(sensor_columns)
            if abs_z[index, column_index] >= z_threshold
        }
        rows.append(
            {
                "dataset_id": dataset_id,
                "snapshot_row_id": (
                    int(frame["source_row_id"].iloc[index])
                    if "source_row_id" in frame.columns
                    else int(index)
                ),
                "entity_id": (
                    str(frame[entity_column].iloc[index]) if entity_column else None
                ),
                "chemistry": (
                    str(frame[chemistry_column].iloc[index])
                    if chemistry_column
                    else None
                ),
                "max_abs_robust_z": float(max_z[index]),
                "max_z_feature": sensor_columns[int(argmax[index])],
                "anomaly_details": json.dumps(detail, sort_keys=True),
                "is_statistical_anomaly": True,
                "interpretation": INTERPRETATION,
                "confirmed_fault": False,
            }
        )
    summary = {
        "dataset_id": dataset_id,
        "method": "robust_z_cross_sectional",
        "z_threshold": float(z_threshold),
        "sensor_columns": sensor_columns,
        "rows_scored": int(len(frame)),
        "rows_flagged": int(flagged.sum()),
        "flagged_fraction": float(flagged.mean()),
        "top_feature_share": {
            column: int(np.sum(flagged & (np.asarray(sensor_columns)[argmax] == column)))
            for column in sensor_columns
        },
        "confirmed_fault_labels_available": False,
        "interpretation": INTERPRETATION,
        "limitations": [
            "Snapshot records are not chronological, so this is a cross-sectional "
            "outlier screen rather than a temporal anomaly detector.",
            "No verified fault labels exist to score the detector.",
        ],
    }
    return pd.DataFrame(rows), summary


__all__ = [
    "INTERPRETATION",
    "ROBUST_Z_THRESHOLD",
    "SENSOR_KEYWORDS",
    "SNAPSHOT_Z_THRESHOLD",
    "detect_sequential_anomalies",
    "detect_snapshot_anomalies",
]
