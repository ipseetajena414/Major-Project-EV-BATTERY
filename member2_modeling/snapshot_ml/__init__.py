"""Grouped, multi-model Member 2 pipeline for the active Member 1 datasets."""

from .inventory import build_inventory, inventory_markdown
from .inference import InferenceInputError, SnapshotInferenceAPI
from .range_estimate import (
    CONSUMPTION_SCENARIOS,
    DEFAULT_CONSUMPTION_KWH_PER_KM,
    RangeEstimateError,
    estimate_range,
    estimate_range_batch,
    estimate_range_from_record,
)
from .registry import (
    REGISTRY_VERSION,
    build_registry,
    load_registry,
    verify_registry,
    write_registry,
)
from .specs import TASKS, UNSUPPORTED_TASKS, TaskSpec, task, task_ids
from .training import MODEL_VERSION, TrainingError, train_task

__all__ = [
    "CONSUMPTION_SCENARIOS",
    "DEFAULT_CONSUMPTION_KWH_PER_KM",
    "InferenceInputError",
    "MODEL_VERSION",
    "REGISTRY_VERSION",
    "RangeEstimateError",
    "SnapshotInferenceAPI",
    "TASKS",
    "TaskSpec",
    "TrainingError",
    "UNSUPPORTED_TASKS",
    "build_inventory",
    "build_registry",
    "estimate_range",
    "estimate_range_batch",
    "estimate_range_from_record",
    "inventory_markdown",
    "load_registry",
    "task",
    "task_ids",
    "train_task",
    "verify_registry",
    "write_registry",
]
