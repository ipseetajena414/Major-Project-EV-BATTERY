"""Inventory, quality, data-dictionary, leakage and registry report builders."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .config import (
    DATASETS,
    PIPELINE_VERSION,
    PROCESSED_DIR,
    dataset_spec,
    description_for,
    role_for,
    unit_for,
)
from .cleaning import (
    QUALITY_COLUMNS,
    PROVENANCE_COLUMNS,
    raw_path,
    sha256_file,
)

QUALITY_COLUMN_DESCRIPTIONS = {
    "source_row_id": ("provenance", "1-based row number in the raw source file."),
    "dataset_id": ("provenance", "Registry identifier for the source dataset."),
    "data_origin": ("provenance", "source_dataset or synthetic."),
    "source_file": ("provenance", "Raw source filename."),
    "chemistry_family": ("quality", "Canonical chemistry label; source label preserved separately."),
    "missing_field_count": ("quality", "Number of source fields missing in this row."),
    "missing_fields": ("quality", "Semicolon-separated source fields missing in this row."),
    "outlier_field_count": ("quality", "Number of numeric fields flagged by the IQR rule."),
    "outlier_fields": ("quality", "Semicolon-separated fields flagged as IQR outliers."),
    "invalid_field_count": ("quality", "Number of fields failing range or vocabulary checks."),
    "invalid_fields": ("quality", "Semicolon-separated fields failing range or vocabulary checks."),
    "quality_flag": ("quality", "Semicolon-separated machine-readable quality flags; OK when clean."),
}

FLAG_DEFINITIONS = {
    "OK": "No quality anomaly was detected for this row.",
    "MISSING_FIELDS": "At least one source field is missing; values are retained as null.",
    "MISSING_BATTERY_CHEMISTRY": "The chemistry source field is missing; chemistry_family is null.",
    "MISSING_BOTH_SOH_LABELS": "Both state_of_health and battery_health_percent are missing.",
    "MISSING_ONE_SOH_LABEL": "Exactly one of the two independent SOH labels is missing.",
    "MISSING_CORE_MEASUREMENT": "cycle_count, pack_voltage or remaining_capacity is missing.",
    "MISSING_TARGET": "The declared target for this dataset is missing.",
    "CHEMISTRY_UNMAPPED": "The source chemistry label is not in the canonical vocabulary.",
    "SOH_LABEL_DISAGREEMENT": "The two SOH labels differ by more than 1 percentage point.",
    "PROJECTED_RUL_BEFORE_CURRENT_CYCLE": "projected_cycle_life_80pct_soh is below cycle_number.",
    "SOH_AT_OR_BELOW_80": "SOH is at or below the 80 percent end-of-life threshold.",
    "ID_REPEATED": "The entity identifier occurs more than once in the dataset.",
    "ID_METADATA_CONFLICT": "A repeated battery_id has conflicting electrolyte labels.",
    "CATEGORY_OUT_OF_VOCABULARY": "A categorical value is outside the declared source vocabulary.",
    "TARGET_INVALID": "A target value is outside its declared domain.",
    "NON_INTEGER_VALUE": "A field declared as an integer contained a fractional value.",
    "NONFINITE_VALUE": "A numeric field contains positive or negative infinity.",
    "INVALID_VALUE": "A numeric field falls outside its physically plausible range.",
    "OUTLIER_DETECTED": "A numeric field falls outside the 1.5 IQR fence; the row is retained.",
}


def _round(value: object, digits: int = 6) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _numeric_summary(series: pd.Series) -> dict:
    values = pd.to_numeric(series, errors="coerce")
    non_null = values.dropna()
    if non_null.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "nulls": int(series.isna().sum()),
            "negative": 0,
            "zero": 0,
            "infinite": 0,
        }
    return {
        "count": int(non_null.size),
        "min": _round(non_null.min()),
        "max": _round(non_null.max()),
        "mean": _round(non_null.mean()),
        "nulls": int(series.isna().sum()),
        "negative": int((non_null < 0).sum()),
        "zero": int((non_null == 0).sum()),
        "infinite": int(np.isinf(non_null.to_numpy(dtype="float64")).sum()),
    }


def _value_label(value: object) -> str:
    return "<null>" if pd.isna(value) else str(value)


def _categorical_summary(series: pd.Series, max_values: int = 40) -> dict:
    counts = series.value_counts(dropna=False)
    if len(counts) > max_values:
        strings = series.dropna().astype(str)
        if not strings.empty and strings.str.len().mean() > 40:
            return {
                "unique": int(series.nunique(dropna=True)),
                "values_omitted": "high-cardinality free-text field",
            }
        return {
            "unique": int(series.nunique(dropna=True)),
            "top": {
                _value_label(key): int(value) for key, value in counts.head(10).items()
            },
        }
    return {_value_label(key): int(value) for key, value in counts.items()}


def _id_analysis(frame: pd.DataFrame, columns: list[str]) -> dict:
    analysis = {}
    for column in columns:
        if column not in frame.columns:
            continue
        counts = frame[column].value_counts(dropna=True)
        analysis[column] = {
            "unique": int(frame[column].nunique(dropna=True)),
            "nulls": int(frame[column].isna().sum()),
            "max_rows_per_id": int(counts.max()) if len(counts) else 0,
            "ids_with_multiple_rows": int((counts > 1).sum()),
            "example_ids": [str(value) for value in counts.index[:5]],
        }
    return analysis


def _sequence_analysis(frame: pd.DataFrame, dataset_id: str) -> dict:
    spec = dataset_spec(dataset_id)
    entities = spec["entity_columns"]
    counts = None
    entity = entities[0]
    if entity in frame.columns:
        counts = frame[entity].value_counts(dropna=True)
    return {
        "entity_columns": entities,
        "cycle_column": spec["cycle_column"],
        "cycle_is_sequence_index": spec["cycle_is_sequence_index"],
        "unique_entities": int(counts.size) if counts is not None else None,
        "rows_per_entity_max": int(counts.max()) if counts is not None and len(counts) else None,
        "entities_with_multiple_rows": int((counts > 1).sum()) if counts is not None else None,
        "chronological_history_possible": bool(spec["sequence_available"]),
        "reason": spec["sequence_reason"],
    }


def _target_summary(frame: pd.DataFrame, dataset_id: str) -> dict:
    spec = dataset_spec(dataset_id)
    targets = dict(spec["targets"])
    targets.update({key: {"task": "regression", "description": value} for key, value in spec["candidate_targets"].items()})
    summary = {}
    for column, meta in targets.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        entry = {
            "task": meta["task"],
            "description": meta["description"],
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
        }
        if is_numeric_dtype(series):
            entry["summary"] = _numeric_summary(series)
        else:
            entry["distribution"] = _categorical_summary(series)
        summary[column] = entry
    return summary


def _corr(frame: pd.DataFrame, left: str, right: str) -> float | None:
    pair = frame[[left, right]].dropna()
    if len(pair) < 3 or pair[left].nunique() < 2 or pair[right].nunique() < 2:
        return None
    return _round(pair[left].corr(pair[right]))


def _soh_evidence(frame: pd.DataFrame) -> dict:
    a = frame["state_of_health"]
    b = frame["battery_health_percent"]
    both = frame.dropna(subset=["state_of_health", "battery_health_percent"])
    diff = (both["state_of_health"] - both["battery_health_percent"]).abs()
    return {
        "state_of_health_missing": int(a.isna().sum()),
        "battery_health_percent_missing": int(b.isna().sum()),
        "both_present": int(len(both)),
        "only_state_of_health": int(a.notna().sum() - len(both)),
        "only_battery_health_percent": int(b.notna().sum() - len(both)),
        "exact_equal": int((diff < 1e-9).sum()),
        "abs_diff_gt_1": int((diff > 1.0).sum()),
        "max_abs_diff": _round(diff.max()) if len(diff) else None,
        "correlation": _corr(frame, "state_of_health", "battery_health_percent"),
    }


def _capacity_loss_evidence(frame: pd.DataFrame) -> dict:
    pairs = frame.dropna(subset=["capacity_loss_percent", "battery_health_percent"])
    diff = (pairs["capacity_loss_percent"] - (100 - pairs["battery_health_percent"])).abs()
    return {
        "paired_rows": int(len(pairs)),
        "exact_100_minus_battery_health_percent": int((diff < 1e-9).sum()),
        "within_0_1": int((diff < 0.1).sum()),
    }


def _predicted_rul_evidence(frame: pd.DataFrame) -> dict:
    sub = frame.dropna(
        subset=["predicted_remaining_life_cycles", "cycle_count", "state_of_health"]
    )
    if len(sub) < 10:
        return {"rows": int(len(sub)), "r2_from_cycle_and_soh": None}
    design = np.column_stack(
        [np.ones(len(sub)), sub["cycle_count"], sub["state_of_health"]]
    )
    target = sub["predicted_remaining_life_cycles"].to_numpy()
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    predictions = design @ coefficients
    total = ((target - target.mean()) ** 2).sum()
    residual = ((target - predictions) ** 2).sum()
    return {
        "rows": int(len(sub)),
        "r2_from_cycle_and_soh": _round(1 - residual / total) if total else None,
        "correlation_with_cycle_count": _corr(sub, "predicted_remaining_life_cycles", "cycle_count"),
        "correlation_with_soh": _corr(sub, "predicted_remaining_life_cycles", "state_of_health"),
        "correlation_with_aging_score": _corr(sub, "predicted_remaining_life_cycles", "aging_score"),
    }


def _solid_state_evidence(frame: pd.DataFrame) -> dict:
    tier_soh = frame.groupby("degradation_risk_tier")["state_of_health_pct"].agg(
        ["count", "mean", "min", "max"]
    )
    band = pd.cut(
        frame["state_of_health_pct"],
        [69.999, 80, 85, 92, 100.001],
        labels=["70-80", "80-85", "85-92", "92-100"],
    )
    crosstab = pd.crosstab(frame["degradation_risk_tier"], band)
    margin_delta = (
        frame["thermal_stability_margin_c"]
        - (frame["thermal_runaway_onset_c"] - frame["operating_temperature_c"])
    ).abs()
    projected_before = frame["projected_cycle_life_80pct_soh"] < frame["cycle_number"]
    return {
        "unique_cells": int(frame["cell_id"].nunique(dropna=True)),
        "rows_per_cell_max": int(frame["cell_id"].value_counts().max()),
        "tier_soh_summary": {
            str(index): {
                "count": int(row["count"]),
                "mean": _round(row["mean"]),
                "min": _round(row["min"]),
                "max": _round(row["max"]),
            }
            for index, row in tier_soh.iterrows()
        },
        "tier_vs_soh_band": {
            str(index): {str(column): int(row[column]) for column in crosstab.columns}
            for index, row in crosstab.iterrows()
        },
        "thermal_margin_equals_onset_minus_operating": int((margin_delta < 1e-9).sum()),
        "rationale_rows_mentioning_soh": int(
            frame["cot_electrochemical_risk_rationale"].str.contains(
                "State of Health", case=False, regex=False
            ).sum()
        ),
        "rationale_unique_values": int(frame["cot_electrochemical_risk_rationale"].nunique()),
        "projected_before_current_cycle": int(projected_before.sum()),
        "projected_before_current_cycle_pct": _round(projected_before.mean()),
        "confidence_range": [
            _round(frame["confidence_score"].min()),
            _round(frame["confidence_score"].max()),
        ],
        "confidence_correlation_with_soh": _corr(frame, "confidence_score", "state_of_health_pct"),
        "risk_score_correlations_with_soh": {
            column: _corr(frame, column, "state_of_health_pct")
            for column in [
                "dendrite_penetration_risk",
                "fast_charge_acceptance_score",
                "dendrite_mitigation_efficiency",
            ]
        },
    }


def _synthetic_evidence(frame: pd.DataFrame) -> dict:
    counts = frame["battery_id"].value_counts()
    repeated_ids = counts[counts > 1].index
    electrolyte_by_id = frame.groupby("battery_id")["electrolyte_type"].nunique()
    conflicting = electrolyte_by_id[electrolyte_by_id > 1].index
    label_summary = {}
    for label, group in frame.groupby("state_of_health"):
        label_summary[str(label)] = {
            "rows": int(len(group)),
            "capacity_ah_min": _round(group["discharge_capacity_ah"].min()),
            "capacity_ah_max": _round(group["discharge_capacity_ah"].max()),
            "resistance_mohm_mean": _round(group["internal_resistance_mohms"].mean()),
            "cycle_count_min": int(group["cycle_count"].min()),
            "cycle_count_max": int(group["cycle_count"].max()),
        }
    return {
        "unique_battery_ids": int(frame["battery_id"].nunique(dropna=True)),
        "repeated_ids": int(len(repeated_ids)),
        "rows_in_repeated_ids": int(counts[counts > 1].sum()),
        "conflicting_metadata_ids": int(len(conflicting)),
        "rows_with_conflicting_metadata": int(frame["battery_id"].isin(conflicting).sum()),
        "label_summary": label_summary,
        "numeric_soh_column_exists": bool(
            any("soh" in column.lower() and is_numeric_dtype(frame[column]) for column in frame.columns)
        ),
        "rul_column_exists": bool(any("rul" in column.lower() for column in frame.columns)),
    }


def build_inventory(
    raw_frames: dict[str, pd.DataFrame],
    clean_frames: dict[str, pd.DataFrame],
) -> dict:
    inventory = {}
    for dataset_id, raw in raw_frames.items():
        clean = clean_frames[dataset_id]
        spec = dataset_spec(dataset_id)
        numeric = {
            column: _numeric_summary(raw[column])
            for column in raw.columns
            if is_numeric_dtype(raw[column])
        }
        categorical = {
            column: _categorical_summary(raw[column])
            for column in raw.columns
            if not is_numeric_dtype(raw[column])
        }
        inventory[dataset_id] = {
            "dataset_id": dataset_id,
            "source_file": spec["source_file"],
            "source_sha256": sha256_file(raw_path(dataset_id)),
            "data_origin": spec["data_origin"],
            "record_grain": spec["record_grain"],
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "column_names": list(raw.columns),
            "dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
            "processed_rows": int(len(clean)),
            "processed_columns": int(clean.shape[1]),
            "processed_column_names": list(clean.columns),
            "processed_dtypes": {column: str(dtype) for column, dtype in clean.dtypes.items()},
            "missing_counts": {
                column: int(value)
                for column, value in raw.isna().sum().items()
                if value
            },
            "missing_total_cells": int(raw.isna().sum().sum()),
            "duplicate_full_rows": int(raw.duplicated().sum()),
            "id_analysis": _id_analysis(raw, spec["id_columns"]),
            "chemistry_distribution": {
                "source": _categorical_summary(raw[spec["chemistry_column"]]),
                "canonical": {
                    str(key): int(value)
                    for key, value in clean["chemistry_family"]
                    .value_counts(dropna=True)
                    .items()
                },
                "canonical_missing": int(clean["chemistry_family"].isna().sum()),
            },
            "numeric_ranges": numeric,
            "categorical_distributions": categorical,
            "target_summary": _target_summary(clean, dataset_id),
            "sequence_analysis": _sequence_analysis(raw, dataset_id),
            "suspicious_values": _suspicious_values(dataset_id, raw, clean),
        }
    return inventory


def _suspicious_values(dataset_id: str, raw: pd.DataFrame, clean: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if dataset_id == "EV_FAILURE_PREDICTION":
        findings.append("One row per vehicle_id and per battery_serial; no chronological cycle history.")
        findings.append("Negative ambient/minimum/maximum temperatures are physically valid, not errors.")
        findings.append("internal_resistance and remaining_capacity have no declared source units; remaining_capacity is not pack Ah derived from kWh and pack voltage.")
        findings.append("capacity_loss_percent is exactly 100 minus battery_health_percent where both exist.")
        findings.append("state_of_health and battery_health_percent are near-duplicates but not identical and have different missing counts.")
        findings.append("predicted_remaining_life_cycles is a model-provided prediction, not measured RUL.")
    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        findings.append("Exactly one row per cell_id; cycle_number is a snapshot, not a history.")
        findings.append("projected_cycle_life_80pct_soh is projected, and for some rows it is below the current cycle_number.")
        findings.append("degradation_risk_tier is deterministically aligned with SOH bands.")
        findings.append("thermal_stability_margin_c is exactly thermal_runaway_onset_c minus operating_temperature_c.")
        findings.append("cot_electrochemical_risk_rationale restates SOH, cycle number and risk tier; it is generated text.")
        findings.append("confidence_score and the enriched risk scores have undocumented derivation.")
    if dataset_id == "SYNTHETIC_BATTERY_LIFECYCLE":
        findings.append("Synthetic origin; not observational field data.")
        findings.append("state_of_health is categorical only; no numeric SOH or measured RUL exists.")
        findings.append("Some battery_id values repeat with different electrolyte labels and different cycle snapshots.")
        findings.append("Labels are strongly associated with discharge capacity, resistance and cycle count, consistent with generated thresholds.")
    return findings


def build_quality_report(
    summaries: dict[str, dict],
    clean_frames: dict[str, pd.DataFrame],
    processed_paths: dict[str, Path],
) -> dict:
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "raw_data_modified": False,
        "imputation_applied": False,
        "rows_dropped": 0,
        "flag_definitions": FLAG_DEFINITIONS,
        "datasets": {},
    }
    for dataset_id, summary in summaries.items():
        clean = clean_frames[dataset_id]
        spec = dataset_spec(dataset_id)
        flag_counts = clean["quality_flag"].value_counts().to_dict()
        report["datasets"][dataset_id] = {
            **summary,
            "processed_file": processed_paths[dataset_id].name,
            "processed_sha256": sha256_file(processed_paths[dataset_id]),
            "rows_after_cleaning": int(len(clean)),
            "rows_dropped": 0,
            "missing_values_retained": True,
            "duplicate_rows_retained": int(summary["duplicate_full_rows"]),
            "quality_flag_distribution": {
                str(key): int(value) for key, value in flag_counts.items()
            },
            "outlier_columns_checked": spec["outlier_columns"],
            "column_level_issues": _column_level_issues(dataset_id, clean),
            "cleaning_decisions": _cleaning_decisions(dataset_id),
        }
    return report


def _column_level_issues(dataset_id: str, clean: pd.DataFrame) -> list[dict]:
    if dataset_id == "EV_FAILURE_PREDICTION":
        return [
            {"column": "internal_resistance", "issue": "Unit not declared by the source.", "action": "Retained; role marked feature_candidate_unit_ambiguous."},
            {"column": "remaining_capacity", "issue": "Grain ambiguous; not equal to kWh * 1000 / pack_voltage.", "action": "Retained; role marked feature_candidate_unit_ambiguous."},
            {"column": "predicted_remaining_life_cycles", "issue": "Predicted field, not measured RUL.", "action": "Excluded from RUL targets and features."},
            {"column": "capacity_loss_percent", "issue": "Deterministic derivative of battery_health_percent.", "action": "Excluded from feature set."},
        ]
    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        return [
            {"column": "projected_cycle_life_80pct_soh", "issue": "Projected, not observed RUL; can be below the current cycle.", "action": "Excluded from features and RUL targets."},
            {"column": "degradation_risk_tier", "issue": "Derived risk label aligned with SOH.", "action": "Excluded as a feature; usable only as a risk-classification target."},
            {"column": "cot_electrochemical_risk_rationale", "issue": "Generated narrative containing SOH and tier.", "action": "Excluded."},
            {"column": "confidence_score", "issue": "Enriched confidence; calibration undocumented.", "action": "Excluded."},
            {"column": "thermal_stability_margin_c", "issue": "Exact deterministic derivative of onset and operating temperature.", "action": "Excluded; recompute from source fields if needed."},
        ]
    counts = clean["battery_id"].value_counts()
    repeated_rows = int(counts[counts > 1].sum())
    conflicting = int(
        clean.groupby("battery_id")["electrolyte_type"].nunique().gt(1).sum()
    )
    return [
        {"column": "state_of_health", "issue": "Categorical target only; no numeric SOH.", "action": "Do not convert to a percentage or invent RUL."},
        {"column": "battery_id", "issue": f"{repeated_rows} rows share a repeated identifier; {conflicting} identifiers have conflicting electrolyte labels.", "action": "Retained and flagged ID_REPEATED / ID_METADATA_CONFLICT."},
    ]


def _cleaning_decisions(dataset_id: str) -> list[str]:
    decisions = [
        "Raw file opened read-only; no row was deleted.",
        "Numeric fields coerced with errors=coerce; failed coercions are recorded in type_coercions.",
        "Missing values retained as null; no imputation.",
        "Outliers flagged with the 1.5 IQR rule and retained.",
        "Category values validated against the declared source vocabulary.",
        "Chemistry labels normalized into chemistry_family while the source label is preserved.",
        "Provenance columns (source_row_id, dataset_id, data_origin, source_file) added to every row.",
    ]
    if dataset_id == "EV_FAILURE_PREDICTION":
        decisions.append(
            "state_of_health and battery_health_percent are retained as separate, independently audited labels."
        )
    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        decisions.append(
            "No voltage_v, current_a or time_sec column is created; the source has no cell time series."
        )
    if dataset_id == "SYNTHETIC_BATTERY_LIFECYCLE":
        decisions.append(
            "state_of_health is kept categorical and is never converted to numeric SOH or RUL."
        )
    return decisions


def build_data_dictionary(clean_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_id, frame in clean_frames.items():
        source_columns = set(
            pd.read_csv(raw_path(dataset_id), nrows=0).columns
        )
        for column in frame.columns:
            if column in PROVENANCE_COLUMNS or column in QUALITY_COLUMNS:
                role, description = QUALITY_COLUMN_DESCRIPTIONS[column]
                unit = ""
            else:
                role = role_for(dataset_id, column, is_numeric_dtype(frame[column]))
                description = description_for(dataset_id, column)
                unit = unit_for(dataset_id, column)
            notes = ""
            if role.startswith("excluded"):
                notes = "Excluded from the model feature set; see leakage_audit.json."
            elif role.startswith("feature_candidate"):
                notes = "Usable only with review; see leakage_audit.json."
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "column_name": column,
                    "source_column": column if column in source_columns else "",
                    "dtype": str(frame[column].dtype),
                    "role": role,
                    "unit": unit,
                    "description": description,
                    "nullable": "yes" if frame[column].isna().any() else "no",
                    "missing_count": int(frame[column].isna().sum()),
                    "notes": notes,
                }
            )
    return pd.DataFrame(rows)


EXCLUDED_REASONS = {
    "EV_FAILURE_PREDICTION": {
        "predicted_remaining_life_cycles": "Predicted field, not measured RUL; excluded as both an input feature and a ground-truth RUL target.",
        "capacity_loss_percent": "Deterministic derivative: exactly 100 minus battery_health_percent where both are present.",
        "state_of_health": "Independent SOH label. If battery_health_percent is the target, this field must not also be a feature.",
        "battery_health_percent": "Independent SOH label. If state_of_health is the target, this field must not also be a feature.",
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "degradation_risk_tier": "Risk label derived from SOH and other signals; target leakage for SOH regression.",
        "cot_electrochemical_risk_rationale": "Generated narrative that restates SOH, cycle number and risk tier; text leakage.",
        "projected_cycle_life_80pct_soh": "Projected value, not observed ground-truth RUL; also inconsistent with the current cycle for some rows.",
        "confidence_score": "Enriched confidence score with undocumented calibration; may encode the target.",
        "thermal_stability_margin_c": "Exact deterministic derivative of thermal_runaway_onset_c and operating_temperature_c.",
        "dendrite_penetration_risk": "Enriched risk score with undocumented derivation; exclude for SOH modeling.",
        "fast_charge_acceptance_score": "Enriched risk score with undocumented derivation; exclude for SOH modeling.",
        "dendrite_mitigation_efficiency": "Enriched risk score with undocumented derivation; exclude for SOH modeling.",
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {},
}


def build_leakage_audit(
    raw_frames: dict[str, pd.DataFrame],
    clean_frames: dict[str, pd.DataFrame],
) -> dict:
    audit = {
        "pipeline_version": PIPELINE_VERSION,
        "global_rules": [
            "Identifiers are never model features.",
            "Declared targets and candidate targets are never model features for their own task.",
            "Predicted or projected fields are not measured ground truth and are excluded.",
            "Deterministically derived fields are excluded when they encode the target.",
            "Text rationales that restate target values are excluded.",
            "Enriched scores with undocumented derivation are excluded pending provenance.",
            "Categorical SOH labels are never converted to numeric percentages or RUL.",
            "No BiLSTM sequence model is supported by these three datasets; each has at most snapshot rows per entity.",
        ],
        "datasets": {},
    }
    for dataset_id, clean in clean_frames.items():
        raw = raw_frames[dataset_id]
        spec = dataset_spec(dataset_id)
        roles = {}
        for column in clean.columns:
            if column in PROVENANCE_COLUMNS:
                roles[column] = "provenance"
            elif column in QUALITY_COLUMNS:
                roles[column] = "quality"
            else:
                roles[column] = role_for(
                    dataset_id, column, is_numeric_dtype(clean[column])
                )
        excluded = []
        for column, reason in EXCLUDED_REASONS[dataset_id].items():
            role = roles.get(column, "excluded")
            excluded.append(
                {
                    "column": column,
                    "role": role,
                    "reason": reason,
                    "evidence": _field_evidence(dataset_id, raw, clean, column),
                }
            )
        audit["datasets"][dataset_id] = {
            "record_grain": spec["record_grain"],
            "sequence_available": spec["sequence_available"],
            "sequence_reason": spec["sequence_reason"],
            "targets": {
                name: {
                    "task": meta["task"],
                    "description": meta["description"],
                    "evidence": _field_evidence(dataset_id, raw, clean, name),
                }
                for name, meta in spec["targets"].items()
            },
            "candidate_targets": {
                name: {
                    "task": "regression",
                    "description": description,
                    "evidence": _field_evidence(dataset_id, raw, clean, name),
                }
                for name, description in spec["candidate_targets"].items()
            },
            "identifiers": {
                column: "Entity identifier; never a feature."
                for column in spec["id_columns"]
            },
            "features_allowed": [
                column for column, role in roles.items() if role == "feature"
            ],
            "features_requires_review": [
                column
                for column, role in roles.items()
                if role.startswith("feature_candidate")
            ],
            "excluded_fields": excluded,
            "evidence": _dataset_evidence(dataset_id, raw, clean),
        }
    return audit


def _field_evidence(
    dataset_id: str,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    column: str,
) -> dict:
    if dataset_id == "EV_FAILURE_PREDICTION":
        if column in {"state_of_health", "battery_health_percent"}:
            return _soh_evidence(clean)
        if column == "capacity_loss_percent":
            return _capacity_loss_evidence(clean)
        if column == "predicted_remaining_life_cycles":
            return _predicted_rul_evidence(clean)
        if column == "battery_failure":
            counts = clean[column].value_counts(dropna=False)
            return {
                "counts": {str(key): int(value) for key, value in counts.items()},
                "positive_rate": _round((clean[column] == 1).mean()),
            }
    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        evidence = _solid_state_evidence(clean)
        if column == "state_of_health_pct":
            return {
                "min": _round(clean[column].min()),
                "max": _round(clean[column].max()),
                "mean": _round(clean[column].mean()),
                "missing": int(clean[column].isna().sum()),
            }
        if column == "degradation_risk_tier":
            return {
                "tier_soh_summary": evidence["tier_soh_summary"],
                "tier_vs_soh_band": evidence["tier_vs_soh_band"],
            }
        if column == "cot_electrochemical_risk_rationale":
            return {
                "unique_values": evidence["rationale_unique_values"],
                "rows_mentioning_soh": evidence["rationale_rows_mentioning_soh"],
            }
        if column == "projected_cycle_life_80pct_soh":
            return {
                "projected_before_current_cycle": evidence["projected_before_current_cycle"],
                "projected_before_current_cycle_pct": evidence["projected_before_current_cycle_pct"],
            }
        if column == "confidence_score":
            return {
                "range": evidence["confidence_range"],
                "correlation_with_soh": evidence["confidence_correlation_with_soh"],
            }
        if column == "thermal_stability_margin_c":
            return {
                "exact_onset_minus_operating_rows": evidence["thermal_margin_equals_onset_minus_operating"],
                "total_rows": int(len(clean)),
            }
        if column in {"dendrite_penetration_risk", "fast_charge_acceptance_score", "dendrite_mitigation_efficiency"}:
            return {
                "correlation_with_soh": evidence["risk_score_correlations_with_soh"].get(column),
            }
    if dataset_id == "SYNTHETIC_BATTERY_LIFECYCLE":
        evidence = _synthetic_evidence(clean)
        if column == "state_of_health":
            return {
                "distribution": evidence["label_summary"],
                "numeric_soh_column_exists": evidence["numeric_soh_column_exists"],
                "rul_column_exists": evidence["rul_column_exists"],
            }
    series = clean[column]
    if is_numeric_dtype(series):
        return _numeric_summary(series)
    return _categorical_summary(series)


def _dataset_evidence(
    dataset_id: str,
    raw: pd.DataFrame,
    clean: pd.DataFrame,
) -> dict:
    if dataset_id == "EV_FAILURE_PREDICTION":
        implied_pack_ah = raw["battery_capacity_kwh"] * 1000 / raw["pack_voltage"]
        ratio = (raw["remaining_capacity"] / implied_pack_ah).dropna()
        return {
            "soh_label_comparison": _soh_evidence(clean),
            "capacity_loss_relation": _capacity_loss_evidence(clean),
            "predicted_rul": _predicted_rul_evidence(clean),
            "remaining_capacity_vs_implied_pack_ah": {
                "paired_rows": int(ratio.size),
                "ratio_mean": _round(ratio.mean()),
                "ratio_min": _round(ratio.min()),
                "ratio_max": _round(ratio.max()),
                "correlation": _corr(
                    pd.DataFrame(
                        {
                            "remaining_capacity": raw["remaining_capacity"],
                            "implied_pack_ah": implied_pack_ah,
                        }
                    ),
                    "remaining_capacity",
                    "implied_pack_ah",
                ),
            },
            "aging_score_correlation_with_soh": _corr(clean, "aging_score", "state_of_health"),
            "thermal_runaway_risk_correlation_with_failure": _corr(clean, "thermal_runaway_risk", "battery_failure"),
        }
    if dataset_id == "SOLID_STATE_TELEMETRY_2026":
        return _solid_state_evidence(clean)
    return _synthetic_evidence(clean)


def build_registry(
    inventory: dict,
    clean_frames: dict[str, pd.DataFrame],
    processed_paths: dict[str, Path],
    legacy_artifacts: list[dict] | None = None,
) -> dict:
    active = {}
    for dataset_id, frame in clean_frames.items():
        spec = dataset_spec(dataset_id)
        active[dataset_id] = {
            "source_file": spec["source_file"],
            "source_sha256": inventory[dataset_id]["source_sha256"],
            "processed_file": processed_paths[dataset_id].name,
            "processed_path": _relative_processed_path(processed_paths[dataset_id]),
            "processed_sha256": sha256_file(processed_paths[dataset_id]),
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "data_origin": spec["data_origin"],
            "record_grain": spec["record_grain"],
            "entity_columns": spec["entity_columns"],
            "cycle_column": spec["cycle_column"],
            "cycle_is_sequence_index": spec["cycle_is_sequence_index"],
            "sequence_available": spec["sequence_available"],
            "chemistry_column": spec["chemistry_column"],
            "chemistry_family_column": spec["chemistry_family_column"],
            "targets": {
                name: {"task": meta["task"], "description": meta["description"]}
                for name, meta in spec["targets"].items()
            },
            "candidate_targets": dict(spec["candidate_targets"]),
            "status": "active",
        }
    return {
        "registry_version": PIPELINE_VERSION,
        "active_datasets": active,
        "active_master_table": None,
        "master_table_policy": (
            "No unified cell-cycle master table is produced. The three active datasets have "
            "different grains and none contains a real cell-level voltage/current/temperature "
            "time series. Downstream code must load datasets separately through member1_pipeline.loader. "
            "Do not merge vehicle snapshots, cell snapshots and synthetic snapshots into one table."
        ),
        "legacy_datasets": legacy_artifacts or [],
    }


def _relative_processed_path(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(PROCESSED_DIR.resolve().parent.parent).as_posix()
    except ValueError:
        return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_data_dictionary(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
