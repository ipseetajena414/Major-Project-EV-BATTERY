"""Entity-grouped, target-stratified splits for snapshot records.

The split unit is the entity (vehicle, cell or synthetic battery), never the
row. This prevents the same entity from appearing in both training and test.
Stratification keeps the target distribution comparable across splits without
weakening group isolation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


class SplitError(ValueError):
    """Raised when a defensible grouped split cannot be produced."""


@dataclass
class GroupedSplit:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    protocol: str
    strata: str
    warning: str | None = None

    def indices(self) -> dict[str, np.ndarray]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def counts(self) -> dict[str, int]:
        return {key: int(value.size) for key, value in self.indices().items()}

    def digest(
        self, frame: pd.DataFrame, entity_column: str
    ) -> str:
        payload = {
            "protocol": self.protocol,
            "strata": self.strata,
            "train": sorted(frame.iloc[self.train][entity_column].astype(str).tolist()),
            "val": sorted(frame.iloc[self.val][entity_column].astype(str).tolist()),
            "test": sorted(frame.iloc[self.test][entity_column].astype(str).tolist()),
        }
        text = repr(payload).encode("utf-8")
        return hashlib.sha256(text).hexdigest()


def _regression_strata(values: np.ndarray) -> np.ndarray:
    if values.size < 20:
        raise SplitError("Too few labelled rows for a stratified grouped split.")
    for bins in (5, 4, 3, 2):
        try:
            labels = pd.qcut(
                pd.Series(values), q=bins, duplicates="drop", labels=False
            )
        except (ValueError, TypeError):
            continue
        if labels is None or labels.nunique() < 2:
            continue
        counts = labels.value_counts()
        if counts.min() >= 5:
            return labels.astype(int).astype(str).to_numpy()
    raise SplitError("Could not build at least two stable target strata.")


def _classification_strata(values: pd.Series) -> np.ndarray:
    labels = values.astype(str).to_numpy()
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        raise SplitError("Classification target has fewer than two classes.")
    if counts.min() < 5:
        raise SplitError(
            f"Smallest class has {int(counts.min())} rows; a grouped split would be unstable."
        )
    return labels


def make_grouped_split(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    entity_column: str,
    task_type: str,
    random_state: int = 42,
) -> GroupedSplit:
    """Return a 60/20/20 entity-disjoint split."""
    entities = frame[entity_column].astype(str).to_numpy()
    if len(np.unique(entities)) < 15:
        raise SplitError(
            f"Only {len(np.unique(entities))} entities; cannot isolate train/val/test."
        )
    if task_type == "regression":
        strata = _regression_strata(np.asarray(target, dtype="float64"))
        protocol = "entity_grouped_stratified_60_20_20"
    else:
        strata = _classification_strata(pd.Series(target))
        protocol = "entity_grouped_class_stratified_60_20_20"

    try:
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        folds = list(outer.split(frame, strata, groups=entities))
        pool_idx, test_idx = folds[0]
        inner = StratifiedGroupKFold(
            n_splits=4, shuffle=True, random_state=random_state + 1
        )
        inner_folds = list(
            inner.split(
                frame.iloc[pool_idx],
                strata[pool_idx],
                groups=entities[pool_idx],
            )
        )
        train_local, val_local = inner_folds[0]
        train_idx = pool_idx[train_local]
        val_idx = pool_idx[val_local]
        protocol = protocol + "_stratified_group_kfold"
        warning = None
    except ValueError as exc:
        outer = GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=random_state
        )
        pool_idx, test_idx = next(outer.split(frame, groups=entities))
        inner = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=random_state + 1
        )
        train_idx, val_idx = next(
            inner.split(frame.iloc[pool_idx], groups=entities[pool_idx])
        )
        train_idx = pool_idx[train_idx]
        val_idx = pool_idx[val_idx]
        protocol = "entity_grouped_shuffle_60_20_20"
        warning = f"Stratified grouped split unavailable ({exc}); used grouped shuffle."

    split = GroupedSplit(
        train=np.sort(train_idx),
        val=np.sort(val_idx),
        test=np.sort(test_idx),
        protocol=protocol,
        strata="target_quantile_bins" if task_type == "regression" else "target_class",
        warning=warning,
    )
    _verify_disjoint(split, entities)
    return split


def _verify_disjoint(split: GroupedSplit, entities: np.ndarray) -> None:
    train = set(entities[split.train])
    val = set(entities[split.val])
    test = set(entities[split.test])
    if train & val or train & test or val & test:
        raise SplitError("Group leakage: an entity appears in more than one split.")
    if not train or not val or not test:
        raise SplitError("A split is empty.")


def distribution(
    frame: pd.DataFrame, split: GroupedSplit, target_column: str
) -> dict[str, Any]:
    """Describe target distribution per split for the audit record."""
    result: dict[str, Any] = {}
    for name, indices in split.indices().items():
        values = pd.to_numeric(frame.iloc[indices][target_column], errors="coerce")
        finite = values[np.isfinite(values)]
        result[name] = {
            "rows": int(len(indices)),
            "mean": float(finite.mean()) if len(finite) else None,
            "min": float(finite.min()) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
            "std": float(finite.std(ddof=0)) if len(finite) else None,
        }
    return result


def class_distribution(
    frame: pd.DataFrame, split: GroupedSplit, target_column: str
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, indices in split.indices().items():
        counts = (
            frame.iloc[indices][target_column]
            .astype(str)
            .value_counts()
            .sort_index()
        )
        result[name] = {str(key): int(value) for key, value in counts.items()}
    return result


def entity_counts(
    frame: pd.DataFrame, split: GroupedSplit, entity_column: str
) -> dict[str, int]:
    return {
        name: int(frame.iloc[indices][entity_column].nunique())
        for name, indices in split.indices().items()
    }


__all__ = [
    "GroupedSplit",
    "SplitError",
    "class_distribution",
    "distribution",
    "entity_counts",
    "make_grouped_split",
]
