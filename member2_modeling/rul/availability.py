"""RUL availability per dataset.

No RUL product in this repository is approved for operational use.  The only
supervised RUL evidence comes from three legacy NASA NMC cells.  One LFP cell
has a genuine long sequence but never crosses the threshold, so it can only be
projected with censoring.  Snapshot datasets and the short Na-ion sequences are
UNAVAILABLE.
"""

from __future__ import annotations

from typing import Any

from member2_modeling.data_integration.profiles import DATASET_IDS

from .labels import CONVENTIONS, EOL_PCT

LEGACY_NMC_DATASET_ID = "LEGACY_NMC_SEQUENTIAL"
NASA_NMC_DATASET_ID = "NASA_Ames_PCoE"
LFP_CENSORED_DATASET_ID = "LFP_MultiStat_B5"
LFP_UNAVAILABLE_DATASET_ID = "LFP_MultiStat_B1"
NA_ION_DATASET_ID = "Zenodo_IonTech_NaIon"

SNAPSHOT_REASON = (
    "The dataset is a snapshot table with no observed chronological end-of-life "
    "trajectory. Row order and cycle identifiers do not establish RUL."
)

LEGACY_REASON = (
    "Only three NMC cells cross the project-defined 80% SOH threshold, one cell "
    "returns above 80% for several cycles after its first crossing, and the "
    "shipped RUL model was validated in-sample on pre-EOL rows. RUL remains "
    "EXPERIMENTAL and is never approved for operational use."
)

LFP_CENSORED_REASON = (
    "One LFP cell has 249 genuine cycles but never reaches 80% SOH. Only a "
    "censored projection with no validated accuracy is available."
)

NA_ION_REASON = (
    "Four Na-ion cells have 25 cycles each; one starts below the threshold and "
    "none shows a genuine end-of-life transition. Insufficient for RUL."
)

LFP_SHORT_REASON = (
    "Only one cycle is available for this LFP cell; no trajectory can be formed."
)


class RULNotApprovedError(RuntimeError):
    """Raised when code requests an RUL model that is not approved for use."""


def _record(dataset_id: str, status: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "status": status,
        "approved_for_use": False,
        "rul_cycles": None,
        "unit": "observed_cycles" if status != "UNAVAILABLE" else None,
        "eol_threshold_pct": EOL_PCT if status != "UNAVAILABLE" else None,
        "sequence_available": status != "UNAVAILABLE",
        "conventions": list(CONVENTIONS),
        "reason": reason,
        **extra,
    }


def rul_availability(dataset_id: str) -> dict[str, Any]:
    if dataset_id in DATASET_IDS:
        return _record(dataset_id, "UNAVAILABLE", SNAPSHOT_REASON)
    if dataset_id in {LEGACY_NMC_DATASET_ID, NASA_NMC_DATASET_ID}:
        return _record(
            dataset_id,
            "EXPERIMENTAL_PRE_EOL_ONLY",
            LEGACY_REASON,
            validation_protocol="leave-one-cell-out",
            cells=3,
        )
    if dataset_id == LFP_CENSORED_DATASET_ID:
        return _record(
            dataset_id,
            "EXPERIMENTAL_CENSORED",
            LFP_CENSORED_REASON,
            validation_protocol="none_no_observed_eol",
            cells=1,
        )
    if dataset_id == LFP_UNAVAILABLE_DATASET_ID:
        return _record(dataset_id, "UNAVAILABLE", LFP_SHORT_REASON, cells=1)
    if dataset_id == NA_ION_DATASET_ID:
        return _record(dataset_id, "UNAVAILABLE", NA_ION_REASON, cells=4)
    raise KeyError(
        f"Unknown dataset_id {dataset_id!r}; expected one of "
        f"{DATASET_IDS + (LEGACY_NMC_DATASET_ID, NASA_NMC_DATASET_ID, LFP_CENSORED_DATASET_ID, LFP_UNAVAILABLE_DATASET_ID, NA_ION_DATASET_ID)}"
    )


def require_approved_rul(dataset_id: str) -> dict[str, Any]:
    """Return the availability record only if RUL is approved; otherwise raise."""
    availability = rul_availability(dataset_id)
    raise RULNotApprovedError(
        f"RUL for {dataset_id!r} is not approved for use: {availability['reason']}"
    )


__all__ = [
    "LFP_CENSORED_DATASET_ID",
    "LFP_UNAVAILABLE_DATASET_ID",
    "LEGACY_NMC_DATASET_ID",
    "NA_ION_DATASET_ID",
    "NASA_NMC_DATASET_ID",
    "RULNotApprovedError",
    "require_approved_rul",
    "rul_availability",
]
