"""Inventory of the Member 1 datasets as Member 2 actually consumes them.

Every value is read from the processed files, the Member 1 registry, the data
dictionary or the Member 2 task specs. Nothing is copied from old reports.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from member1_pipeline import loader
from member2_modeling.data_integration.profiles import DATASET_IDS, feature_columns_for_target

from .specs import ENTITY_COLUMNS, TASKS

RAW_LOCATIONS = {
    "EV_FAILURE_PREDICTION": "ev battery_failure prediction Dataset.csv",
    "SOLID_STATE_TELEMETRY_2026": "solid_state_ev_battery_telemetry_enriched_2026.csv",
    "SYNTHETIC_BATTERY_LIFECYCLE": "synthetic_battery_lifecycle.csv",
}

INTENDED_TASKS = {
    "EV_FAILURE_PREDICTION": [
        "state_of_health regression (ev_soh)",
        "remaining_capacity regression (ev_capacity, derived target)",
        "battery_failure classification (ev_failure)",
        "formula-based range estimate (not validated)",
    ],
    "SOLID_STATE_TELEMETRY_2026": [
        "state_of_health_pct regression (solid_soh)",
        "degradation_risk_tier classification (solid_tier)",
    ],
    "SYNTHETIC_BATTERY_LIFECYCLE": [
        "discharge_capacity_ah regression (synthetic_capacity)",
        "categorical state_of_health classification (synthetic_class)",
    ],
}


def _missing_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    summary = {}
    for column in columns:
        if column not in frame.columns:
            continue
        missing = int(frame[column].isna().sum())
        if missing:
            summary[column] = {
                "missing": missing,
                "missing_pct": round(100.0 * missing / len(frame), 4),
            }
    return dict(sorted(summary.items(), key=lambda item: -item[1]["missing"]))


def build_inventory() -> dict[str, Any]:
    registry = loader.load_registry()
    dictionary = loader.load_data_dictionary()
    entries: dict[str, Any] = {}
    for dataset_id in DATASET_IDS:
        registered = registry["active_datasets"][dataset_id]
        frame = loader.load_dataset(dataset_id)
        entity = ENTITY_COLUMNS[dataset_id]
        chemistry_column = registered["chemistry_column"]
        unit_map = (
            dictionary[dictionary["dataset_id"] == dataset_id]
            .set_index("column_name")["unit"]
            .dropna()
            .astype(str)
            .to_dict()
        )
        task_targets = {
            spec.target: {
                "task_id": spec.task_id,
                "task_type": spec.task_type,
                "feature_count": len(feature_columns_for_target(dataset_id, spec.target)),
                "unit": spec.unit,
                "label_origin": spec.label_origin,
                "status": spec.status,
            }
            for spec in TASKS.values()
            if spec.dataset_id == dataset_id
        }
        chemistry_counts = (
            frame[chemistry_column].astype(str).value_counts().sort_index().to_dict()
        )
        missing_by_target = {
            target: int(frame[target].isna().sum())
            for target in task_targets
            if target in frame.columns
        }
        entries[dataset_id] = {
            "dataset_id": dataset_id,
            "original_source": RAW_LOCATIONS[dataset_id],
            "member1_status": "active_integrated_dataset",
            "legacy": False,
            "raw_data_location": RAW_LOCATIONS[dataset_id],
            "processed_data_location": registered["processed_path"],
            "processed_sha256": registered["processed_sha256"],
            "source_sha256": registered["source_sha256"],
            "records": int(len(frame)),
            "columns": int(frame.shape[1]),
            "unique_entities": int(frame[entity].nunique()),
            "entity_column": entity,
            "entity_repeats": int((frame[entity].value_counts() > 1).sum()),
            "chemistry_column": chemistry_column,
            "chemistry_values": sorted(chemistry_counts),
            "chemistry_counts": chemistry_counts,
            "feature_columns": sorted(loader.feature_columns(dataset_id)),
            "feature_count": len(loader.feature_columns(dataset_id)),
            "target_columns": sorted(loader.target_columns(dataset_id)),
            "targets_used_by_member2": task_targets,
            "units": unit_map,
            "temporal_resolution": (
                "single snapshot per entity; cycle field is a scalar, not a time index"
            ),
            "sequential": False,
            "sequence_reason": (
                "No chronological cycle history; at most one snapshot per entity."
                " See DATA/processed/dataset_registry.json."
            ),
            "missing_by_target": missing_by_target,
            "missing_features": _missing_summary(
                frame, sorted(loader.feature_columns(dataset_id))
            ),
            "quality_flags": (
                frame["quality_flag"].astype(str).value_counts().head(10).to_dict()
                if "quality_flag" in frame.columns
                else {}
            ),
            "intended_tasks": INTENDED_TASKS[dataset_id],
            "usable_for": {
                "train_val_test": True,
                "split_unit": entity,
                "temporal_validation": False,
                "sequence_modelling": False,
                "rul": False,
            },
            "data_origin": registered["data_origin"],
            "record_grain": registered["record_grain"],
        }
    return entries


def inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Member 2 dataset inventory",
        "",
        "Generated from `DATA/processed/` and `member1_pipeline`; no legacy table is used.",
        "",
        "| Dataset | Rows | Entity | Unique | Chemistry values | Features | Member 2 targets |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for dataset_id, entry in inventory.items():
        lines.append(
            f"| `{dataset_id}` | {entry['records']:,} | `{entry['entity_column']}` | "
            f"{entry['unique_entities']:,} | {len(entry['chemistry_values'])} | "
            f"{entry['feature_count']} | "
            + ", ".join(f"`{target}`" for target in entry["targets_used_by_member2"])
            + " |"
        )
    lines.extend(["", "## Sequence and RUL availability", ""])
    for dataset_id, entry in inventory.items():
        lines.append(
            f"- `{dataset_id}`: sequential={entry['sequential']}; "
            f"temporal_validation allowed={entry['usable_for']['temporal_validation']}; "
            f"RUL allowed={entry['usable_for']['rul']}."
        )
    lines.extend(["", "## Missing labels", ""])
    for dataset_id, entry in inventory.items():
        missing = ", ".join(
            f"`{target}`={count}" for target, count in entry["missing_by_target"].items()
        )
        lines.append(f"- `{dataset_id}`: {missing or 'none'}")
    lines.append("")
    return "\n".join(lines)


__all__ = ["build_inventory", "inventory_markdown"]
