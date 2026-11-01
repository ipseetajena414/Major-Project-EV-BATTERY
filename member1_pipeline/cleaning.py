"""Reproducible cleaning for the three active Member 1 datasets.

Raw files are only read. Every input row is retained; anomalies are recorded in
provenance and quality-flag columns instead of being silently dropped.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import REPO_ROOT, dataset_spec, normalize_chemistry

PROVENANCE_COLUMNS = [
    "source_row_id",
    "dataset_id",
    "data_origin",
    "source_file",
]

QUALITY_COLUMNS = [
    "chemistry_family",
    "missing_field_count",
    "missing_fields",
    "outlier_field_count",
    "outlier_fields",
    "invalid_field_count",
    "invalid_fields",
    "quality_flag",
]

INTEGER_COLUMNS = {
    "EV_FAILURE_PREDICTION": {
        "battery_failure": "Int8",
        "manufacturing_year": "Int16",
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "cycle_number": "Int32",
        "thermal_runaway_onset_c": "Int32",
        "projected_cycle_life_80pct_soh": "Int32",
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "cycle_count": "Int32",
    },
}

FLAG_ORDER = [
    "OK",
    "MISSING_FIELDS",
    "MISSING_BATTERY_CHEMISTRY",
    "MISSING_BOTH_SOH_LABELS",
    "MISSING_ONE_SOH_LABEL",
    "MISSING_CORE_MEASUREMENT",
    "MISSING_TARGET",
    "CHEMISTRY_UNMAPPED",
    "SOH_LABEL_DISAGREEMENT",
    "PROJECTED_RUL_BEFORE_CURRENT_CYCLE",
    "SOH_AT_OR_BELOW_80",
    "ID_REPEATED",
    "ID_METADATA_CONFLICT",
    "CATEGORY_OUT_OF_VOCABULARY",
    "TARGET_INVALID",
    "NON_INTEGER_VALUE",
    "NONFINITE_VALUE",
    "INVALID_VALUE",
    "OUTLIER_DETECTED",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_path(dataset_id: str) -> Path:
    spec = dataset_spec(dataset_id)
    return REPO_ROOT / spec["source_file"]


def read_raw(dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(raw_path(dataset_id))


def _coerce_types(
    df: pd.DataFrame, dataset_id: str
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str]]:
    spec = dataset_spec(dataset_id)
    source_columns = list(df.columns)
    coercions: dict[str, int] = {}
    dtypes: dict[str, str] = {}
    int_spec = INTEGER_COLUMNS.get(dataset_id, {})

    for column in source_columns:
        series = df[column]
        if column in int_spec or is_numeric_dtype(series):
            converted = pd.to_numeric(series, errors="coerce")
            changed = int((series.notna() & converted.isna()).sum())
            if changed:
                coercions[column] = changed
            if column in int_spec:
                fractional = int((converted.dropna() % 1 != 0).sum())
                if fractional:
                    coercions[f"{column}:fractional"] = fractional
                    # Keep the observed value; fractional entries are flagged
                    # NON_INTEGER_VALUE instead of being rounded.
                    converted = converted.astype("Float64")
                else:
                    converted = converted.astype(int_spec[column])
            else:
                converted = converted.astype("float64")
            df[column] = converted
        else:
            text = series.astype("string").str.strip()
            text = text.mask(text == "", pd.NA)
            df[column] = text
        dtypes[column] = str(df[column].dtype)
    return df, coercions, dtypes


def _add_flags(
    flags: list[list[str]],
    mask: pd.Series,
    token: str,
) -> None:
    if token not in FLAG_ORDER:
        FLAG_ORDER.append(token)
    for position in np.flatnonzero(mask.fillna(False).to_numpy()):
        flags[int(position)].append(token)


def _collect_fields(
    field_lists: list[list[str]],
    mask: pd.Series,
    label: str,
) -> None:
    for position in np.flatnonzero(mask.fillna(False).to_numpy()):
        field_lists[int(position)].append(label)


def _sorted_tokens(tokens: list[str]) -> list[str]:
    return sorted(set(tokens), key=lambda token: (FLAG_ORDER.index(token), token))


def _outlier_masks(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]:
    masks: dict[str, pd.Series] = {}
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
        spread = q3 - q1
        if spread <= 0:
            continue
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        masks[column] = values.notna() & ((values < low) | (values > high))
    return masks


def clean_dataset(dataset_id: str) -> tuple[pd.DataFrame, dict]:
    spec = dataset_spec(dataset_id)
    raw = read_raw(dataset_id)
    rows_in = len(raw)
    source_sha = sha256_file(raw_path(dataset_id))

    df = raw.copy()
    source_columns = list(df.columns)
    df, coercions, dtypes = _coerce_types(df, dataset_id)

    # Provenance and chemistry family are inserted before source columns so the
    # output is self-describing when datasets are concatenated downstream.
    chemistry_column = spec["chemistry_column"]
    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        chemistry_family = pd.Series(
            "Solid-State", index=df.index, dtype="string"
        )
    else:
        chemistry_family = df[chemistry_column].map(normalize_chemistry).astype("string")

    df.insert(0, "source_row_id", np.arange(1, rows_in + 1, dtype=np.int64))
    df.insert(1, "dataset_id", pd.Series(dataset_id, index=df.index, dtype="string"))
    df.insert(2, "data_origin", pd.Series(spec["data_origin"], index=df.index, dtype="string"))
    df.insert(3, "source_file", pd.Series(spec["source_file"], index=df.index, dtype="string"))
    df["chemistry_family"] = chemistry_family

    flags: list[list[str]] = [[] for _ in range(rows_in)]
    missing_lists: list[list[str]] = [[] for _ in range(rows_in)]
    outlier_lists: list[list[str]] = [[] for _ in range(rows_in)]
    invalid_lists: list[list[str]] = [[] for _ in range(rows_in)]

    # Declared integer fields that contain fractions keep their observed value
    # and are flagged rather than rounded.
    for column in INTEGER_COLUMNS.get(dataset_id, {}):
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask = values.notna() & ((values % 1) != 0)
        if mask.any():
            _add_flags(flags, mask, "NON_INTEGER_VALUE")
            _collect_fields(invalid_lists, mask, column)

    # Missing values are retained. Record the exact field list per row.
    for column in source_columns:
        mask = df[column].isna()
        if mask.any():
            _add_flags(flags, mask, "MISSING_FIELDS")
            _collect_fields(missing_lists, mask, column)

    _add_flags(flags, df["chemistry_family"].isna(), "MISSING_BATTERY_CHEMISTRY")
    known_chemistry = {
        "NMC", "LFP", "LMO", "NCA", "LTO", "Solid-State", "Sodium-ion", "Lithium-Sulfur"
    }
    mapped = df["chemistry_family"].notna()
    unmapped = mapped & ~df["chemistry_family"].isin(known_chemistry)
    _add_flags(flags, unmapped, "CHEMISTRY_UNMAPPED")

    # Generic valid ranges and categorical vocabularies.
    for column, (low, high) in spec.get("valid_ranges", {}).items():
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        mask = values.notna() & ((values < low) | (values > high))
        _add_flags(flags, mask, "INVALID_VALUE")
        _collect_fields(invalid_lists, mask, column)

    for column, allowed in spec.get("allowed_categories", {}).items():
        if column not in df.columns:
            continue
        mask = df[column].notna() & ~df[column].isin(allowed)
        _add_flags(flags, mask, "CATEGORY_OUT_OF_VOCABULARY")
        _collect_fields(invalid_lists, mask, column)

    for column in spec.get("outlier_columns", []):
        mask = _outlier_masks(df, [column]).get(column)
        if mask is not None and mask.any():
            _add_flags(flags, mask, "OUTLIER_DETECTED")
            _collect_fields(outlier_lists, mask, column)

    numeric_columns = [
        column for column in source_columns if is_numeric_dtype(df[column])
    ]
    for column in numeric_columns:
        values = df[column]
        nonfinite = values.notna() & np.isinf(
            values.to_numpy(dtype="float64", na_value=np.nan)
        )
        if nonfinite.any():
            mask = pd.Series(nonfinite, index=df.index)
            _add_flags(flags, mask, "NONFINITE_VALUE")
            _collect_fields(invalid_lists, mask, column)

    # Dataset-specific audit rules.
    if dataset_id == "EV_FAILURE_PREDICTION":
        soh_a = df["state_of_health"]
        soh_b = df["battery_health_percent"]
        both_missing = soh_a.isna() & soh_b.isna()
        one_missing = soh_a.isna() ^ soh_b.isna()
        _add_flags(flags, both_missing, "MISSING_BOTH_SOH_LABELS")
        _add_flags(flags, one_missing, "MISSING_ONE_SOH_LABEL")
        disagreement = (
            soh_a.notna()
            & soh_b.notna()
            & ((soh_a - soh_b).abs() > 1.0)
        )
        _add_flags(flags, disagreement, "SOH_LABEL_DISAGREEMENT")
        core_missing = (
            df["cycle_count"].isna()
            | df["pack_voltage"].isna()
            | df["remaining_capacity"].isna()
        )
        _add_flags(flags, core_missing, "MISSING_CORE_MEASUREMENT")
        _add_flags(flags, df["battery_failure"].isna(), "MISSING_TARGET")
        _add_flags(
            flags,
            df["battery_failure"].notna() & ~df["battery_failure"].isin([0, 1]),
            "TARGET_INVALID",
        )

    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        projected_before_current = (
            df["projected_cycle_life_80pct_soh"].notna()
            & (df["projected_cycle_life_80pct_soh"] < df["cycle_number"])
        )
        _add_flags(
            flags, projected_before_current, "PROJECTED_RUL_BEFORE_CURRENT_CYCLE"
        )
        _add_flags(
            flags,
            df["state_of_health_pct"].notna() & (df["state_of_health_pct"] <= 80),
            "SOH_AT_OR_BELOW_80",
        )
        _add_flags(flags, df["state_of_health_pct"].isna(), "MISSING_TARGET")

    if dataset_id == "SYNTHETIC_BATTERY_LIFECYCLE":
        _add_flags(flags, df["state_of_health"].isna(), "MISSING_TARGET")
        _add_flags(
            flags,
            df["state_of_health"].notna()
            & ~df["state_of_health"].isin(spec["allowed_categories"]["state_of_health"]),
            "TARGET_INVALID",
        )
        counts = df["battery_id"].map(df["battery_id"].value_counts())
        repeated = counts > 1
        _add_flags(flags, repeated, "ID_REPEATED")
        electrolyte_by_id = df.groupby("battery_id")["electrolyte_type"].nunique()
        conflicting_ids = electrolyte_by_id[electrolyte_by_id > 1].index
        conflict = df["battery_id"].isin(conflicting_ids)
        _add_flags(flags, conflict, "ID_METADATA_CONFLICT")

    # Generic identifier repetition checks for datasets without a specific rule.
    if dataset_id != "SYNTHETIC_BATTERY_LIFECYCLE":
        for column in spec["id_columns"]:
            counts = df[column].map(df[column].value_counts())
            _add_flags(flags, counts > 1, "ID_REPEATED")

    df["missing_field_count"] = [len(items) for items in missing_lists]
    df["missing_fields"] = [";".join(items) for items in missing_lists]
    df["outlier_field_count"] = [len(items) for items in outlier_lists]
    df["outlier_fields"] = [";".join(items) for items in outlier_lists]
    df["invalid_field_count"] = [len(items) for items in invalid_lists]
    df["invalid_fields"] = [";".join(items) for items in invalid_lists]
    quality = []
    for tokens in flags:
        ordered = _sorted_tokens(tokens) if tokens else ["OK"]
        quality.append(";".join(ordered))
    df["quality_flag"] = pd.Series(quality, dtype="string")

    for column in ["missing_fields", "outlier_fields", "invalid_fields"]:
        df[column] = df[column].astype("string")
    for column in ["missing_field_count", "outlier_field_count", "invalid_field_count"]:
        df[column] = df[column].astype("Int16")

    assert len(df) == rows_in, "Cleaning must never drop rows"

    flag_counts = dict(Counter(token for tokens in flags for token in set(tokens)))
    summary = {
        "dataset_id": dataset_id,
        "source_file": spec["source_file"],
        "source_sha256": source_sha,
        "rows_in": rows_in,
        "rows_out": int(len(df)),
        "rows_removed": rows_in - int(len(df)),
        "source_columns": int(len(source_columns)),
        "output_columns": int(df.shape[1]),
        "type_coercions": coercions,
        "dtypes": dtypes,
        "quality_flag_token_counts": flag_counts,
        "rows_with_any_quality_flag": int(sum(bool(tokens) for tokens in flags)),
        "rows_with_outliers": int(sum(bool(items) for items in outlier_lists)),
        "rows_with_invalid_fields": int(sum(bool(items) for items in invalid_lists)),
        "rows_with_missing_fields": int(sum(bool(items) for items in missing_lists)),
        "missing_total_cells": int(df[source_columns].isna().sum().sum()),
        "duplicate_full_rows": int(df[source_columns].duplicated().sum()),
        "chemistry_distribution": {
            str(key): int(value)
            for key, value in df["chemistry_family"].value_counts(dropna=True).items()
        },
        "chemistry_family_missing": int(df["chemistry_family"].isna().sum()),
    }
    return df, summary
