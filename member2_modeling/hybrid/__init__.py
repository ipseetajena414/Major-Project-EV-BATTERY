"""Fused LightGBM + BiLSTM battery-health models."""
from .branches import SequenceBranch, TabularBranch
from .eligibility import LEGACY_NMC_DATASET_ID, hybrid_eligibility
from .fusion import HybridError, OutOfFoldFusion
from .hybrid import HybridSOHModel

__all__ = [
    "HybridError",
    "HybridSOHModel",
    "LEGACY_NMC_DATASET_ID",
    "OutOfFoldFusion",
    "SequenceBranch",
    "TabularBranch",
    "hybrid_eligibility",
]
