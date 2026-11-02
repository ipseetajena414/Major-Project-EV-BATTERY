"""Member 2 data integration package."""

from .profiles import (
    DATASET_IDS,
    DatasetCompatibilityError,
    build_compatibility_report,
    feature_columns_for_target,
    profile_dataset,
    target_spec,
)

__all__ = [
    "DATASET_IDS",
    "DatasetCompatibilityError",
    "build_compatibility_report",
    "feature_columns_for_target",
    "profile_dataset",
    "target_spec",
]
