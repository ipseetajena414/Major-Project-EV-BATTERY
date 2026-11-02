"""Dataset compatibility and provenance layer for Member 2.

Member 1 publishes three snapshot datasets. This module inspects their real
schemas and reports what each dataset can and cannot support. It never infers
a chronological sequence from row order and never treats a projection or a
synthetic label as observed evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from member1_pipeline import loader

DATASET_IDS = (
    "EV_FAILURE_PREDICTION",
    "SOLID_STATE_TELEMETRY_2026",
    "SYNTHETIC_BATTERY_LIFECYCLE",
)

REPORT_JSON = "member2_dataset_compatibility.json"
REPORT_MD = "MEMBER2_DATASET_COMPATIBILITY_REPORT.md"

EV_HEALTH_LEAKAGE = (
    "state_of_health",
    "battery_health_percent",
    "capacity_loss_percent",
    "remaining_capacity",
    "predicted_remaining_life_cycles",
)

SOLID_HEALTH_LEAKAGE = (
    "state_of_health_pct",
    "degradation_risk_tier",
    "projected_cycle_life_80pct_soh",
    "cot_electrochemical_risk_rationale",
    "confidence_score",
)


def _target(
    task: str,
    definition: str,
    leakage: Iterable[str] = (),
    *,
    training_permitted: bool = True,
    readiness: str = "EXPERIMENTAL",
    label_origin: str = "source_dataset",
    rul_status: str = "UNAVAILABLE",
    rul_reason: str = "No observed sequential EOL history in this dataset.",
    unit: str | None = None,
) -> dict[str, Any]:
    return {
        "task": task,
        "definition": definition,
        "unit": unit,
        "leakage_columns": list(dict.fromkeys(leakage)),
        "training_permitted": training_permitted,
        "readiness": readiness,
        "label_origin": label_origin,
        "rul_status": rul_status,
        "rul_reason": rul_reason,
    }


TARGET_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "EV_FAILURE_PREDICTION": {
        "battery_failure": _target(
            "classification",
            "Binary battery-failure label supplied by the source dataset.",
            EV_HEALTH_LEAKAGE,
            readiness="EXPERIMENTAL",
        ),
        "state_of_health": _target(
            "regression",
            "SOH percentage; audited independently of battery_health_percent.",
            (
                "battery_health_percent",
                "capacity_loss_percent",
                "remaining_capacity",
                "predicted_remaining_life_cycles",
            ),
            unit="%",
        ),
        "battery_health_percent": _target(
            "regression",
            "SOH percentage; audited independently of state_of_health.",
            (
                "state_of_health",
                "capacity_loss_percent",
                "remaining_capacity",
                "predicted_remaining_life_cycles",
            ),
            unit="%",
        ),
        "remaining_capacity": _target(
            "regression",
            "Pack capacity derived algebraically as battery_capacity_kwh × "
            "battery_health_percent / 100 (rounded to 2 decimals); it is not "
            "an independent measurement. The derivation is verified only on "
            "rows where both battery_capacity_kwh and battery_health_percent "
            "are present.",
            (
                "state_of_health",
                "battery_health_percent",
                "capacity_loss_percent",
                "predicted_remaining_life_cycles",
            ),
            label_origin="derived_from_health_percent",
            unit="kWh (pack energy, not cell Ah)",
        ),
        "predicted_remaining_life_cycles": _target(
            "projection_not_rul",
            "Source-provided projection. It is not an observed RUL label.",
            training_permitted=False,
            readiness="UNAVAILABLE",
            rul_status="UNAVAILABLE",
            rul_reason="The field is a supplied projection, not an observed EOL trajectory.",
            unit="cycles (projected)",
        ),
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "state_of_health_pct": _target(
            "regression",
            "Candidate SOH percentage; mixed electrolyte/cathode metadata.",
            SOLID_HEALTH_LEAKAGE[1:],
            unit="%",
            rul_reason=(
                "projected_cycle_life_80pct_soh is a projection and cycle_number is "
                "not a chronological trajectory."
            ),
        ),
        "degradation_risk_tier": _target(
            "classification",
            "Provided risk tier derived from SOH and other signals.",
            SOLID_HEALTH_LEAKAGE,
        ),
        "projected_cycle_life_80pct_soh": _target(
            "projection_not_rul",
            "Projected cycle life at 80% SOH. Never an observed RUL label.",
            training_permitted=False,
            readiness="UNAVAILABLE",
            rul_status="UNAVAILABLE",
            rul_reason="Projected cycle life is not observed remaining useful life.",
            unit="cycles (projected)",
        ),
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "state_of_health": _target(
            "classification",
            "Categorical Healthy/Degraded/End_of_Life label.",
            ("discharge_capacity_ah",),
            label_origin="synthetic",
            readiness="RESEARCH_ONLY",
            rul_reason="Synthetic snapshot labels with no chronological history.",
        ),
        "discharge_capacity_ah": _target(
            "regression",
            "Measured discharge capacity in the synthetic campaign.",
            ("state_of_health",),
            label_origin="synthetic",
            readiness="RESEARCH_ONLY",
            unit="Ah (synthetic)",
            rul_reason="Synthetic snapshot records do not establish observed RUL.",
        ),
    },
}

EXCLUDED_ROLE_PREFIXES = ("excluded", "target", "provenance", "identifier", "quality")


class DatasetCompatibilityError(ValueError):
    """Raised when a requested target/feature combination is not defensible."""


def _registry() -> dict[str, Any]:
    return loader.load_registry()


def _inventory() -> dict[str, Any]:
    return loader.load_inventory()


def _dictionary() -> pd.DataFrame:
    return loader.load_data_dictionary()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is pd.NA or value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    return value


def _targets(dataset_id: str) -> dict[str, dict[str, Any]]:
    if dataset_id not in TARGET_SPECS:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}; expected one of {DATASET_IDS}")
    return TARGET_SPECS[dataset_id]


def target_spec(dataset_id: str, target: str) -> dict[str, Any]:
    targets = _targets(dataset_id)
    if target not in targets:
        raise KeyError(
            f"Target {target!r} is not defined for {dataset_id}; "
            f"available: {sorted(targets)}"
        )
    return json.loads(json.dumps(_jsonable(targets[target])))


def _base_features(dataset_id: str, dictionary: pd.DataFrame) -> list[str]:
    subset = dictionary[dictionary["dataset_id"] == dataset_id]
    columns = []
    for column, role in zip(subset["column_name"], subset["role"]):
        if str(role) == "feature":
            columns.append(str(column))
    return columns


def feature_columns_for_target(
    dataset_id: str,
    target: str,
    requested: Iterable[str] | None = None,
) -> list[str]:
    """Return leakage-safe feature columns in the dataset's stored order."""
    spec = target_spec(dataset_id, target)
    if not spec["training_permitted"]:
        raise DatasetCompatibilityError(
            f"{dataset_id} target {target!r} is not permitted for training: {spec['definition']}"
        )
    dictionary = _dictionary()
    base = _base_features(dataset_id, dictionary)
    removed = {target, *spec["leakage_columns"]}
    if dataset_id == "EV_FAILURE_PREDICTION":
        removed |= {
            "vehicle_id",
            "battery_serial",
            "source_row_id",
            "dataset_id",
            "data_origin",
            "source_file",
        }
    allowed = [column for column in base if column not in removed]

    if requested is not None:
        requested_list = list(dict.fromkeys(str(column) for column in requested))
        unknown = [column for column in requested_list if column not in allowed]
        if unknown:
            raise DatasetCompatibilityError(
                f"Requested columns are not leakage-safe features for "
                f"{dataset_id}/{target}: {unknown}. Target/leakage columns must "
                "never be used as inputs."
            )
        return requested_list
    return allowed


def _missingness(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        if missing:
            result[str(column)] = {
                "count": missing,
                "pct": round(100.0 * missing / len(frame), 4),
            }
    return result


def _identifier_analysis(frame: pd.DataFrame, entities: Iterable[str]) -> dict[str, Any]:
    analysis: dict[str, Any] = {}
    for column in entities:
        if column not in frame.columns:
            continue
        series = frame[column]
        repeated = int(len(series) - series.nunique(dropna=True))
        analysis[str(column)] = {
            "unique": int(series.nunique(dropna=True)),
            "repeated_rows": repeated,
            "nulls": int(series.isna().sum()),
        }
    return analysis


def profile_dataset(dataset_id: str) -> dict[str, Any]:
    """Profile one processed dataset using its real schema and registry data."""
    if dataset_id not in DATASET_IDS:
        raise KeyError(f"Unknown dataset_id {dataset_id!r}; expected one of {DATASET_IDS}")
    registry = _registry()
    inventory = _inventory()
    dictionary = _dictionary()
    frame = loader.load_dataset(dataset_id)
    entry = registry["active_datasets"][dataset_id]
    inv = inventory[dataset_id]
    targets = _targets(dataset_id)
    sequence_available = bool(entry["sequence_available"])
    if sequence_available:
        sequence_reason = "Registry declares chronological histories for this dataset."
    else:
        sequence_reason = (
            "Registry declares snapshot-only records. Row order and cycle identifiers "
            "are not treated as a chronological degradation trajectory."
        )

    chemistry_column = entry.get("chemistry_column")
    chemistry_family = entry.get("chemistry_family_column") or "chemistry_family"
    chemistry_labels: list[str] = []
    if chemistry_family in frame.columns:
        chemistry_labels = sorted(
            str(value)
            for value in frame[chemistry_family].dropna().unique().tolist()
        )
    elif chemistry_column and chemistry_column in frame.columns:
        chemistry_labels = sorted(
            str(value)
            for value in frame[chemistry_column].dropna().unique().tolist()
        )

    label_counts = {}
    for name, spec in targets.items():
        if name in frame.columns:
            series = frame[name]
            if spec["task"] == "classification":
                label_counts[name] = {
                    str(key): int(value)
                    for key, value in series.value_counts(dropna=False).items()
                }

    safe_features = {}
    for name, spec in targets.items():
        if spec["training_permitted"]:
            safe_features[name] = feature_columns_for_target(dataset_id, name)

    leakage_columns = sorted(
        {
            column
            for spec in targets.values()
            for column in spec["leakage_columns"]
        }
    )
    permissible_targets = [
        name for name, spec in targets.items() if spec["training_permitted"]
    ]
    return _jsonable(
        {
            "dataset_id": dataset_id,
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "source_file": entry["source_file"],
            "data_origin": entry["data_origin"],
            "record_grain": entry["record_grain"],
            "provenance": {
                "source_sha256": entry["source_sha256"],
                "processed_sha256": entry["processed_sha256"],
                "quality_report": "DATA/processed/data_quality_report.json",
            },
            "compatibility": {
                "available_columns": list(frame.columns),
                "candidate_targets": permissible_targets,
                "targets": targets,
                "label_counts": label_counts,
                "chemistry_labels": chemistry_labels,
                "identifier_analysis": _identifier_analysis(
                    frame, entry.get("entity_columns") or []
                ),
                "cycle_field": entry.get("cycle_column"),
                "cycle_is_sequence_index": bool(entry.get("cycle_is_sequence_index")),
                "sequence_available": sequence_available,
                "sequence_reason": sequence_reason,
                "missingness": _missingness(frame),
                "duplicate_full_rows": int(inv.get("duplicate_full_rows", 0)),
                "leakage_columns": leakage_columns,
                "safe_feature_columns": safe_features,
                "training_permitted_targets": permissible_targets,
                "useful_for": {
                    "inference": False,
                    "experimentation": True,
                    "augmentation": bool(entry["data_origin"] == "synthetic"),
                    "validation": bool(entry["data_origin"] == "source_dataset"),
                },
            },
        }
    )


def _compatibility_summary(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": profile["rows"],
        "columns": profile["columns"],
        "data_origin": profile["data_origin"],
        "record_grain": profile["record_grain"],
        "sequence_available": profile["compatibility"]["sequence_available"],
        "training_permitted_targets": profile["compatibility"][
            "training_permitted_targets"
        ],
        "leakage_columns": profile["compatibility"]["leakage_columns"],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Member 2 Dataset Compatibility Report",
        "",
        "Generated from `DATA/processed/` registry, inventory, data dictionary and",
        "the actual Parquet schemas. No row order is interpreted as battery history.",
        "",
        "**Active unified master table:** none (`active_master_table = null`).",
        "",
        "## Observed RUL availability",
        "",
        "Observed RUL is unavailable for all three datasets. The solid-state",
        "`projected_cycle_life_80pct_soh` column and the EV",
        "`predicted_remaining_life_cycles` column are projections, not observed",
        "end-of-life trajectories, and are never used as RUL labels.",
        "",
        "## Dataset summary",
        "",
        "| Dataset | Rows | Grain | Sequence data | Permitted targets |",
        "|---|---:|---|---|---|",
    ]
    for dataset_id, profile in report["datasets"].items():
        compat = profile["compatibility"]
        targets = ", ".join(compat["training_permitted_targets"]) or "none"
        lines.append(
            f"| `{dataset_id}` | {profile['rows']} | {profile['record_grain']} | "
            f"{'yes' if compat['sequence_available'] else 'no (snapshot-only)'} | {targets} |"
        )
    lines.extend(["", "## Target details", ""])
    for dataset_id, profile in report["datasets"].items():
        lines.append(f"### `{dataset_id}`")
        lines.append("")
        lines.append(
            f"Source: `{profile['source_file']}`  \n"
            f"SHA-256: `{profile['provenance']['source_sha256']}`  \n"
            f"Chemistry labels: {', '.join(profile['compatibility']['chemistry_labels']) or 'none'}"
        )
        lines.append("")
        lines.append("| Target | Task | Training | Readiness | RUL |")
        lines.append("|---|---|---|---|---|")
        for name, spec in profile["compatibility"]["targets"].items():
            lines.append(
                f"| `{name}` | {spec['task']} | "
                f"{'yes' if spec['training_permitted'] else 'no'} | "
                f"{spec['readiness']} | {spec['rul_status']} |"
            )
        lines.append("")
        lines.append(
            "Leakage-excluded columns: "
            + (", ".join(f"`{c}`" for c in profile["compatibility"]["leakage_columns"]) or "none")
        )
        lines.append("")
    lines.append(
        "Synthetic labels are not measured experimental ground truth. Solid-state "
        "records are mixed electrolyte/cathode metadata and are not assumed to be "
        "genuine solid-state cells."
    )
    lines.append("")
    return "\n".join(lines)


def build_compatibility_report(out_dir: Path | str | None = None) -> dict[str, Any]:
    """Build the compatibility report; optionally write JSON and Markdown."""
    registry = _registry()
    report = _jsonable(
        {
            "registry_version": registry.get("registry_version"),
            "active_master_table": registry.get("active_master_table"),
            "datasets": {
                dataset_id: profile_dataset(dataset_id)
                for dataset_id in DATASET_IDS
            },
            "summary": {
                dataset_id: _compatibility_summary(profile_dataset(dataset_id))
                for dataset_id in DATASET_IDS
            },
        }
    )
    if out_dir is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / REPORT_JSON).write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (destination / REPORT_MD).write_text(_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "DATASET_IDS",
    "DatasetCompatibilityError",
    "build_compatibility_report",
    "feature_columns_for_target",
    "profile_dataset",
    "target_spec",
]
