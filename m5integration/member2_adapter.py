"""Thin adapter over Member 2 (SOH prediction). Reuses Member 2's own hardened contract builder.

Nothing is invented here: features come from Member 2's ``extract_cycle_features`` applied to
real rows of ``master_table.parquet``; the health result comes from ``build_health_result``.
"""
from __future__ import annotations
from typing import Any

from .paths import ROOT, M2_DIR, MASTER_TABLE, ensure_repo_on_path

ensure_repo_on_path()

# Chemistry states shown to the user. Supported means "a learned model exists" (still experimental).
SUPPORTED = ("NMC", "LFP")
UNSUPPORTED_REASONS = {
    "Na-ion": "No SOH model: only 100 cycles over 4 cells with minimal fade; excluded from training by Member 2.",
    "Sodium-ion": "No SOH model: only 100 cycles over 4 cells with minimal fade; excluded from training by Member 2.",
    "Solid-State": "No dataset and no trained SOH model exist for solid-state batteries.",
}
OK_STATUSES = ("ok", "ok_experimental")


def load_predictor():
    """Return ``(predictor, error_message)``."""
    try:
        from member2_modeling.predictor import Member2Predictor
        return Member2Predictor(str(M2_DIR)), None
    except Exception as exc:  # surfaced to the UI, never swallowed silently
        return None, f"{type(exc).__name__}: {exc}"


def feature_extraction_available() -> tuple[bool, str | None]:
    try:
        from member2_modeling import build_and_train  # noqa: F401  (imports torch at module level)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def list_cycles_index():
    """Return a DataFrame of (chemistry, dataset_id, cell_id, cycle_id, soh_pct, quality_flag), one row per
    (dataset, cell, cycle). Note: quality_flag here is the first row's flag; use ``member1_quality_flag`` for the
    full per-cycle value."""
    import pandas as pd
    cols = ["chemistry", "dataset_id", "cell_id", "cycle_id", "soh_pct", "quality_flag"]
    df = pd.read_parquet(MASTER_TABLE, columns=cols)
    return df.drop_duplicates(["dataset_id", "cell_id", "cycle_id"]).reset_index(drop=True)


def _cycle_rows(cell_id: str, cycle_id: int, dataset_id: str | None):
    """Read the parquet rows of one cycle. A supplied ``dataset_id`` is always applied as a filter.

    Returns ``(frame, error)``. With no ``dataset_id`` the selection must resolve to exactly one dataset,
    otherwise an error is returned instead of silently mixing datasets.
    """
    import pandas as pd
    filters = [("cell_id", "==", cell_id), ("cycle_id", "==", int(cycle_id))]
    if dataset_id is not None:
        filters.append(("dataset_id", "==", dataset_id))
    frame = pd.read_parquet(MASTER_TABLE, filters=filters)
    if frame.empty:
        where = f"dataset {dataset_id!r}, " if dataset_id is not None else ""
        return None, f"No rows for {where}cell {cell_id} cycle {cycle_id}."
    found = sorted(frame["dataset_id"].astype(str).unique())
    if len(found) > 1:
        return None, (f"Ambiguous selection: cell {cell_id} cycle {cycle_id} exists in several datasets {found}; "
                      "supply dataset_id.")
    return frame, None


def member1_quality_flag(cell_id: str, cycle_id: int, dataset_id: str | None = None) -> str | None:
    """Member 1's original ``quality_flag`` for one cycle, kept OUTSIDE the Member 2 contract.

    One distinct value is returned verbatim. If rows of the cycle carry several distinct values, they are all
    kept verbatim, sorted and joined with ' | ' (nothing is merged or reinterpreted). ``None`` if the cycle is
    not found or ambiguous.
    """
    frame, err = _cycle_rows(cell_id, cycle_id, dataset_id)
    if err:
        return None
    vals = sorted(frame["quality_flag"].astype(str).unique())
    return vals[0] if len(vals) == 1 else " | ".join(vals)


def health_for_cycle(predictor, cell_id: str, cycle_id: int, chemistry: str, dataset_id: str | None = None):
    """Build Member 2's v1.1.0-draft health contract for one real observed cycle.

    Returns ``(contract_or_None, error_or_None)``. ``dataset_id`` (when given) filters the parquet rows and must
    match; the contract's ``subject.dataset_id`` comes from the rows actually used. The contract schema is not
    changed; Member 1's quality flag is available separately via ``member1_quality_flag``.
    Missing SOC/temperature/C-rate features are never fabricated: Member 2's adapter reports them as blocked.
    """
    frame, err = _cycle_rows(cell_id, cycle_id, dataset_id)
    if err:
        return None, err
    from member2_modeling import build_and_train as bt
    from member2_modeling.integration_readiness import adapter as ia

    features = bt.extract_cycle_features(frame, frame["chemistry"].iloc[0])
    features["current_soh_pct"] = float(frame["soh_pct"].iloc[0])
    prov = {"cell_id": cell_id, "dataset_id": str(frame["dataset_id"].iloc[0]), "cycle_id": int(cycle_id)}
    return ia.build_health_result(predictor, features, chemistry, prov, include_capacity=True), None


def soh_for_range(contract: dict[str, Any] | None, source: str = "observed") -> dict[str, Any]:
    """Decide which SOH (percent) may be forwarded to Member 3.

    ``observed`` (default, the persistence baseline) or ``predicted`` (experimental, opt-in).
    A blocked/unavailable result yields ``allowed=False``; no SOH is silently substituted.
    """
    if not contract:
        return {"allowed": False, "soh_pct": None, "reason": "No Member 2 health result."}
    status = contract.get("status")
    soh = contract.get("soh") or {}
    if status not in OK_STATUSES:
        return {"allowed": False, "soh_pct": None, "reason": f"Member 2 status is '{status}'; range is blocked.",
                "errors": contract.get("errors")}
    key = "predicted_soh_pct" if source == "predicted" else "current_observed_soh_pct"
    value = soh.get(key)
    if value is None:
        return {"allowed": False, "soh_pct": None, "reason": f"{key} is not available."}
    return {"allowed": True, "soh_pct": float(value), "source": source,
            "label": "EXPERIMENTAL predicted SOH" if source == "predicted" else "Observed SOH (baseline anchor)",
            "status": status}


def rul_signal(contract: dict[str, Any] | None):
    """RUL is not approved for downstream use. Always returns None as a decision signal."""
    return None
