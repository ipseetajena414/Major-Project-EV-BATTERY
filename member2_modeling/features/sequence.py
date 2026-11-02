"""Causal sequence window construction for eligible sequential datasets.

Windows end at the target record. No future measurements are placed inside an
input window. The scaler is fitted on training rows only.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .tabular import FeatureSchemaError


class SequenceFeatureBuilder:
    def __init__(
        self,
        feature_columns: Iterable[str],
        target_column: str,
        entity_column: str,
        order_column: str,
        sequence_length: int,
    ) -> None:
        self.feature_columns = list(feature_columns)
        self.target_column = target_column
        self.entity_column = entity_column
        self.order_column = order_column
        if sequence_length < 1:
            raise FeatureSchemaError("sequence_length must be >= 1")
        self.sequence_length = int(sequence_length)
        if self.target_column in self.feature_columns:
            raise FeatureSchemaError(
                f"Target column {self.target_column!r} cannot appear in sequence features"
            )
        self._fitted = False
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame) -> "SequenceFeatureBuilder":
        missing = [
            column
            for column in [*self.feature_columns, self.entity_column, self.order_column]
            if column not in frame.columns
        ]
        if missing:
            raise FeatureSchemaError(f"Missing sequence columns: {missing}")
        values = frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        mean = values.mean(skipna=True).to_numpy(dtype="float64")
        scale = values.std(skipna=True, ddof=0).to_numpy(dtype="float64")
        mean = np.where(np.isfinite(mean), mean, 0.0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
        self._mean, self._scale = mean, scale
        self._fitted = True
        return self

    def transform(
        self, frame: pd.DataFrame, *, include_missing_target: bool = False
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if not self._fitted:
            raise FeatureSchemaError("SequenceFeatureBuilder must be fitted before transform().")
        missing = [
            column
            for column in [
                *self.feature_columns,
                *(() if include_missing_target else (self.target_column,)),
                self.entity_column,
                self.order_column,
            ]
            if column not in frame.columns
        ]
        if missing:
            raise FeatureSchemaError(f"Missing sequence columns: {missing}")

        windows: list[np.ndarray] = []
        targets: list[float] = []
        meta_rows: list[dict[str, Any]] = []

        ordered = frame.sort_values(
            [self.entity_column, self.order_column], kind="mergesort"
        )
        for entity, group in ordered.groupby(self.entity_column, sort=True):
            group = group.reset_index(drop=True)
            if self.target_column not in group.columns:
                group[self.target_column] = np.nan
            values = group[self.feature_columns].apply(pd.to_numeric, errors="coerce")
            values = values.fillna(pd.Series(self._mean, index=self.feature_columns))
            scaled = (values.to_numpy(dtype="float64") - self._mean) / self._scale
            for end in range(self.sequence_length - 1, len(group)):
                target = pd.to_numeric(
                    pd.Series([group.loc[end, self.target_column]]), errors="coerce"
                ).iloc[0]
                if pd.isna(target) and not include_missing_target:
                    continue
                start = end - self.sequence_length + 1
                windows.append(scaled[start : end + 1])
                targets.append(float(target) if pd.notna(target) else np.nan)
                meta_rows.append(
                    {
                        "entity": entity,
                        "target_cycle_id": group.loc[end, self.order_column],
                        "sequence_end_position": end,
                    }
                )

        if not windows:
            return (
                np.zeros((0, self.sequence_length, len(self.feature_columns))),
                np.zeros(0),
                pd.DataFrame(columns=["entity", "target_cycle_id", "sequence_end_position"]),
            )
        return (
            np.asarray(windows, dtype="float32"),
            np.asarray(targets, dtype="float32"),
            pd.DataFrame(meta_rows),
        )

    def fit_transform(
        self, frame: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        return self.fit(frame).transform(frame)

    def to_schema(self) -> dict[str, Any]:
        if not self._fitted:
            raise FeatureSchemaError("SequenceFeatureBuilder must be fitted before export.")
        return {
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "entity_column": self.entity_column,
            "order_column": self.order_column,
            "sequence_length": self.sequence_length,
            "scaler_mean": None if self._mean is None else self._mean.tolist(),
            "scaler_scale": None if self._scale is None else self._scale.tolist(),
            "causal": True,
        }
