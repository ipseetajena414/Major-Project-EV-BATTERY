"""Dataset eligibility for the fused hybrid model.

A sequence branch requires real chronological histories. The three Member 1
snapshot datasets are eligible for tabular experimentation only; using their
row order as a sequence would invent history that does not exist.

The legacy NASA NMC table is the only sequential source in this repository.
It is used explicitly as labelled legacy research input, never as a Member 1
processed dataset.
"""

from __future__ import annotations

from typing import Any

from member1_pipeline import loader
from member2_modeling.data_integration.profiles import DATASET_IDS

LEGACY_NMC_DATASET_ID = "LEGACY_NMC_SEQUENTIAL"

SNAPSHOT_REASON = (
    "The Member 1 registry declares snapshot-only records. Row order and cycle "
    "identifiers are not treated as a chronological degradation trajectory, so "
    "no BiLSTM sequence branch or hybrid fusion can be fitted."
)

LEGACY_REASON = (
    "The legacy NASA NMC master table provides chronological per-cycle histories "
    "with measured SOH labels. The hybrid may be trained on it as explicitly "
    "labelled legacy research input; it is not a Member 1 processed dataset."
)


def hybrid_eligibility(dataset_id: str) -> dict[str, Any]:
    if dataset_id in DATASET_IDS:
        registry = loader.load_registry()
        entry = registry["active_datasets"][dataset_id]
        sequence_available = bool(entry.get("sequence_available"))
        if sequence_available:
            return {
                "dataset_id": dataset_id,
                "sequence_available": True,
                "hybrid_status": "EXPERIMENTAL",
                "eligible_components": ["tabular", "sequence", "fusion"],
                "reason": "The registry declares chronological histories for this dataset.",
            }
        return {
            "dataset_id": dataset_id,
            "sequence_available": False,
            "hybrid_status": "UNAVAILABLE",
            "eligible_components": ["tabular"],
            "reason": SNAPSHOT_REASON,
        }
    if dataset_id == LEGACY_NMC_DATASET_ID:
        return {
            "dataset_id": dataset_id,
            "sequence_available": True,
            "hybrid_status": "EXPERIMENTAL",
            "eligible_components": ["tabular", "sequence", "fusion"],
            "reason": LEGACY_REASON,
        }
    raise KeyError(
        f"Unknown dataset_id {dataset_id!r}; expected one of "
        f"{DATASET_IDS + (LEGACY_NMC_DATASET_ID,)}"
    )


__all__ = ["LEGACY_NMC_DATASET_ID", "hybrid_eligibility"]
