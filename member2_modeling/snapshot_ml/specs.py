"""Task registry for the active Member 1 snapshot datasets.

Every task below is backed by a label that actually exists in the processed
data. Nothing here creates a sequence, an RUL target or a range target. Tasks
that the data cannot support are listed in ``UNSUPPORTED_TASKS`` with the
evidence that rules them out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from member1_pipeline import loader
from member2_modeling.data_integration.profiles import (
    DATASET_IDS,
    target_spec,
)

CHEMISTRY_COLUMNS: dict[str, str] = {
    "EV_FAILURE_PREDICTION": "battery_chemistry",
    "SOLID_STATE_TELEMETRY_2026": "electrolyte_chemistry",
    "SYNTHETIC_BATTERY_LIFECYCLE": "electrolyte_type",
}

ENTITY_COLUMNS: dict[str, str] = {
    "EV_FAILURE_PREDICTION": "vehicle_id",
    "SOLID_STATE_TELEMETRY_2026": "cell_id",
    "SYNTHETIC_BATTERY_LIFECYCLE": "battery_id",
}

DATASET_GRAIN: dict[str, str] = {
    "EV_FAILURE_PREDICTION": "vehicle_pack_snapshot",
    "SOLID_STATE_TELEMETRY_2026": "cell_snapshot",
    "SYNTHETIC_BATTERY_LIFECYCLE": "synthetic_battery_snapshot",
}


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    dataset_id: str
    target: str
    task_type: str  # "regression" | "binary" | "multiclass"
    status: str
    unit: str | None
    label_origin: str
    notes: tuple[str, ...] = ()

    @property
    def entity_column(self) -> str:
        return ENTITY_COLUMNS[self.dataset_id]

    @property
    def chemistry_column(self) -> str:
        return CHEMISTRY_COLUMNS[self.dataset_id]

    def to_dict(self) -> dict[str, Any]:
        spec = target_spec(self.dataset_id, self.target)
        return {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "target": self.target,
            "task_type": self.task_type,
            "status": self.status,
            "unit": self.unit,
            "label_origin": self.label_origin,
            "entity_column": self.entity_column,
            "chemistry_column": self.chemistry_column,
            "record_grain": DATASET_GRAIN[self.dataset_id],
            "target_definition": spec["definition"],
            "readiness": spec["readiness"],
            "rul_status": spec["rul_status"],
            "rul_reason": spec["rul_reason"],
            "notes": list(self.notes),
        }


TASKS: dict[str, TaskSpec] = {
    "ev_soh": TaskSpec(
        task_id="ev_soh",
        dataset_id="EV_FAILURE_PREDICTION",
        target="state_of_health",
        task_type="regression",
        status="EXPERIMENTAL",
        unit="%",
        label_origin="source_label",
        notes=(
            "battery_health_percent, capacity_loss_percent, remaining_capacity and "
            "predicted_remaining_life_cycles are forbidden as features.",
            "The two supplied SOH labels disagree by up to 6.22 points; this model "
            "predicts state_of_health only.",
        ),
    ),
    "ev_capacity": TaskSpec(
        task_id="ev_capacity",
        dataset_id="EV_FAILURE_PREDICTION",
        target="remaining_capacity",
        task_type="regression",
        status="EXPERIMENTAL",
        unit="kWh (pack energy, not cell Ah)",
        label_origin="derived_from_health_percent",
        notes=(
            "remaining_capacity equals battery_capacity_kwh * battery_health_percent / 100 "
            "to within 0.005 kWh, so this is an algebraically derived target, not an "
            "independent pack-energy measurement.",
            "The derivation is verified only on rows where battery_capacity_kwh and "
            "battery_health_percent are both present; rows with a null input are "
            "not verified.",
            "Reported performance must not be described as measured-capacity accuracy.",
        ),
    ),
    "ev_failure": TaskSpec(
        task_id="ev_failure",
        dataset_id="EV_FAILURE_PREDICTION",
        target="battery_failure",
        task_type="binary",
        status="EXPERIMENTAL",
        unit="probability of failure (0/1 label)",
        label_origin="source_label",
        notes=(
            "All SOH labels, capacity loss, remaining capacity and predicted RUL are "
            "forbidden as features.",
            "The positive class is 1,384 / 20,000 (6.9%); balanced accuracy is the "
            "selection metric and accuracy alone is not accepted.",
        ),
    ),
    "solid_soh": TaskSpec(
        task_id="solid_soh",
        dataset_id="SOLID_STATE_TELEMETRY_2026",
        target="state_of_health_pct",
        task_type="regression",
        status="EXPERIMENTAL",
        unit="%",
        label_origin="source_label",
        notes=(
            "degradation_risk_tier, projected cycle life, risk rationale and enriched "
            "scores are forbidden as features.",
        ),
    ),
    "solid_tier": TaskSpec(
        task_id="solid_tier",
        dataset_id="SOLID_STATE_TELEMETRY_2026",
        target="degradation_risk_tier",
        task_type="multiclass",
        status="EXPERIMENTAL",
        unit="risk tier class",
        label_origin="derived_from_soh",
        notes=(
            "The tier is derived from SOH and other signals, but state_of_health_pct is "
            "excluded, so this is not a relabelled SOH shortcut.",
            "Four classes are imbalanced (1,682 to 3,758 rows); balanced accuracy and "
            "macro F1 are reported alongside accuracy.",
        ),
    ),
    "synthetic_capacity": TaskSpec(
        task_id="synthetic_capacity",
        dataset_id="SYNTHETIC_BATTERY_LIFECYCLE",
        target="discharge_capacity_ah",
        task_type="regression",
        status="RESEARCH_ONLY",
        unit="Ah (synthetic)",
        label_origin="synthetic_label",
        notes=(
            "Generated data, not experimental truth.",
            "Repeated battery_id values (22) are kept inside one split by the grouped "
            "splitter.",
        ),
    ),
    "synthetic_class": TaskSpec(
        task_id="synthetic_class",
        dataset_id="SYNTHETIC_BATTERY_LIFECYCLE",
        target="state_of_health",
        task_type="multiclass",
        status="RESEARCH_ONLY",
        unit="Healthy / Degraded / End_of_Life class",
        label_origin="synthetic_label",
        notes=(
            "discharge_capacity_ah is forbidden as a feature because the categorical "
            "label is generated from capacity-like signals.",
            "No numeric SOH or RUL is invented from this label.",
        ),
    ),
}

UNSUPPORTED_TASKS: dict[str, dict[str, str]] = {
    "next_cycle_soh": {
        "status": "UNAVAILABLE",
        "reason": (
            "All three active Member 1 datasets are snapshot tables. EV has one row per "
            "vehicle, solid-state one row per cell, and synthetic at most two conflicting "
            "snapshots per battery. No chronological cycle history exists."
        ),
        "evidence": "DATA/processed/dataset_registry.json sequence_available=false for all datasets",
    },
    "rul": {
        "status": "UNAVAILABLE_ON_ACTIVE_DATA",
        "reason": (
            "No active dataset contains an observed degradation trajectory to end of life. "
            "projected_cycle_life_80pct_soh and predicted_remaining_life_cycles are supplied "
            "projections, not observed RUL labels."
        ),
        "evidence": (
            "SOLID_STATE_TELEMETRY_2026 projected cycle life is a projection; "
            "EV predicted_remaining_life_cycles is a model output. The legacy NMC PADM "
            "experiment remains preserved but is not an active-data result."
        ),
    },
    "ev_range": {
        "status": "CALCULATED_ESTIMATE_ONLY",
        "reason": (
            "No measured driving-range target exists in the EV dataset. Range can only be "
            "calculated from rated pack energy, SOH and an assumed consumption rate; it "
            "cannot be validated for accuracy against ground truth."
        ),
        "evidence": "EV_FAILURE_PREDICTION has no range_km or measured consumption column",
    },
    "seq_chemistry_transfer": {
        "status": "UNAVAILABLE",
        "reason": (
            "Cross-chemistry transfer learning needs compatible feature schemas and enough "
            "entities per chemistry. The three snapshots use different grains and columns, "
            "so a single shared sequence model would be fabricated."
        ),
        "evidence": "EV 82 cols, solid-state 41 cols, synthetic 20 cols; grains differ",
    },
}


def task(task_id: str) -> TaskSpec:
    try:
        return TASKS[task_id]
    except KeyError as exc:
        raise KeyError(
            f"Unknown task_id {task_id!r}; available: {sorted(TASKS)}"
        ) from exc


def task_ids() -> list[str]:
    return list(TASKS)


def chemistry_values(dataset_id: str) -> list[str]:
    frame = loader.load_dataset(dataset_id, columns=[CHEMISTRY_COLUMNS[dataset_id]])
    values = frame[CHEMISTRY_COLUMNS[dataset_id]].dropna().astype(str).unique().tolist()
    return sorted(values)


def task_frame(spec: TaskSpec):
    """Load the dataset for a task. Member 1 data is opened read-only."""
    return loader.load_dataset(spec.dataset_id)


__all__ = [
    "CHEMISTRY_COLUMNS",
    "DATASET_GRAIN",
    "DATASET_IDS",
    "ENTITY_COLUMNS",
    "TASKS",
    "UNSUPPORTED_TASKS",
    "TaskSpec",
    "chemistry_values",
    "task",
    "task_frame",
    "task_ids",
]
