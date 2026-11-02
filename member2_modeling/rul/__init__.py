"""RUL conventions, availability and rigorous pre/post-threshold evaluation."""
from .availability import (
    LEGACY_NMC_DATASET_ID,
    RULNotApprovedError,
    require_approved_rul,
    rul_availability,
)
from .evaluation import BUCKET_LABELS, RULEvaluationError, evaluate_rul
from .forecast import RULHistory, build_rul_record, make_history
from .labels import (
    CONVENTIONS,
    EOL_PCT,
    SUSTAINED_COUNT,
    RULTargetError,
    add_rul_labels,
    crossing_position,
)
from .report import build_rul_report

# `pipeline` and `service` import `analytics.sequential`, which itself imports
# `rul.labels`. Resolving them lazily breaks that package-init cycle for fresh
# `member2_modeling.analytics.*` imports while keeping the public names.
_LAZY_ATTRS = {
    "build_rul_artifacts": ".pipeline",
    "RULService": ".service",
}


def __getattr__(name: str):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "build_rul_artifacts",
    "build_rul_report",
    "BUCKET_LABELS",
    "CONVENTIONS",
    "EOL_PCT",
    "LEGACY_NMC_DATASET_ID",
    "RULEvaluationError",
    "RULHistory",
    "RULNotApprovedError",
    "RULService",
    "RULTargetError",
    "SUSTAINED_COUNT",
    "add_rul_labels",
    "build_rul_record",
    "crossing_position",
    "evaluate_rul",
    "make_history",
    "require_approved_rul",
    "rul_availability",
]
