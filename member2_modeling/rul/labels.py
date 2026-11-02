"""RUL label conventions for observed sequential battery histories.

RUL is only meaningful when a cell crosses a defined end-of-life threshold in
observed data. The crossing convention must be explicit: a single transient dip
below the threshold and a sustained crossing produce different labels.

Rows are ordered per entity by the supplied order column, and RUL is measured
in observed records (positional distance), never in arithmetic cycle-ID gaps.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

EOL_PCT = 80.0
SUSTAINED_COUNT = 3
CONVENTIONS = ("first", "sustained3")


class RULTargetError(ValueError):
    """Raised when an RUL target cannot be defined defensibly."""


def crossing_position(
    soh: Any,
    convention: str = "first",
    *,
    eol_pct: float = EOL_PCT,
    sustained_count: int = SUSTAINED_COUNT,
) -> int | None:
    """Return the positional index of the EOL crossing, or None.

    The threshold is strict (``SOH < eol_pct``): a value exactly at the
    threshold has not crossed it. Null SOH values never count as a crossing.
    """
    if convention not in CONVENTIONS:
        raise RULTargetError(
            f"Unknown crossing convention {convention!r}; expected one of {CONVENTIONS}"
        )
    if sustained_count < 1:
        raise RULTargetError("sustained_count must be >= 1")
    values = pd.to_numeric(pd.Series(soh), errors="coerce").to_numpy(dtype="float64")
    below = values < float(eol_pct)
    if convention == "first":
        hits = np.flatnonzero(below)
        return int(hits[0]) if hits.size else None
    for start in range(len(values) - sustained_count + 1):
        if bool(below[start : start + sustained_count].all()):
            return int(start)
    return None


def add_rul_labels(
    frame: pd.DataFrame,
    convention: str = "first",
    *,
    entity_column: str = "cell_id",
    order_column: str = "cycle_position",
    target_column: str = "soh_pct",
    eol_pct: float = EOL_PCT,
    sustained_count: int = SUSTAINED_COUNT,
) -> pd.DataFrame:
    """Attach ``rul_cycles``, ``is_post_threshold`` and ``crossing_position``.

    The input frame is not mutated and its row order is preserved. Cells that
    never cross the threshold receive null RUL and null post-threshold flags:
    their RUL is censored, not zero.
    """
    if convention not in CONVENTIONS:
        raise RULTargetError(
            f"Unknown crossing convention {convention!r}; expected one of {CONVENTIONS}"
        )
    missing = [
        column
        for column in (entity_column, order_column, target_column)
        if column not in frame.columns
    ]
    if missing:
        raise RULTargetError(f"Missing RUL labelling columns: {missing}")

    labelled = frame.copy()
    rul = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    post = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    crossing = pd.Series(pd.NA, index=frame.index, dtype="Int64")

    ordered = frame.sort_values([entity_column, order_column], kind="mergesort")
    for _, group in ordered.groupby(entity_column, sort=False):
        position = crossing_position(
            group[target_column],
            convention,
            eol_pct=eol_pct,
            sustained_count=sustained_count,
        )
        if position is None:
            continue
        positions = np.arange(len(group))
        rul.loc[group.index] = np.where(
            positions <= position, float(position) - positions, 0.0
        )
        post.loc[group.index] = positions >= position
        crossing.loc[group.index] = position

    labelled["rul_cycles"] = rul
    labelled["is_post_threshold"] = post
    labelled["crossing_position"] = crossing
    return labelled


__all__ = [
    "CONVENTIONS",
    "EOL_PCT",
    "RULTargetError",
    "SUSTAINED_COUNT",
    "add_rul_labels",
    "crossing_position",
]
