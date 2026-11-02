"""Leakage-aware tabular feature pipeline.

The pipeline is fitted on training data only. Numeric imputation values and
categorical vocabularies are stored from the training frame; transforming a
holdout frame never consults holdout statistics.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

SCHEMA_VERSION = "2.0.0"
MISSING_CATEGORY = "__MISSING__"


class FeatureSchemaError(ValueError):
    """Raised when a feature schema is unsafe or not fitted."""


class TabularFeaturePipeline:
    def __init__(
        self,
        numeric_features: Iterable[str],
        categorical_features: Iterable[str],
        *,
        target: str | None = None,
        forbidden_columns: Iterable[str] = (),
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.target = target
        self.forbidden_columns = set(forbidden_columns)
        self._fitted = False
        self._medians: dict[str, float] = {}
        self._categories: dict[str, list[str]] = {}
        self._feature_names: list[str] = []

    @property
    def all_input_features(self) -> list[str]:
        return [*self.numeric_features, *self.categorical_features]

    def _validate_inputs(self) -> None:
        duplicated = len(self.all_input_features) != len(set(self.all_input_features))
        if duplicated:
            raise FeatureSchemaError("Feature names must be unique.")
        forbidden = set(self.forbidden_columns)
        if self.target is not None:
            forbidden.add(self.target)
        unsafe = sorted(forbidden & set(self.all_input_features))
        if unsafe:
            raise FeatureSchemaError(
                f"Target or leakage columns cannot be used as features: {unsafe}"
            )

    def fit(self, frame: pd.DataFrame) -> "TabularFeaturePipeline":
        self._validate_inputs()
        missing = [c for c in self.all_input_features if c not in frame.columns]
        if missing:
            raise FeatureSchemaError(f"Missing feature columns in training frame: {missing}")

        self._medians = {}
        self._categories = {}
        names: list[str] = []
        for column in self.numeric_features:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = values.median()
            self._medians[column] = float(median) if pd.notna(median) else 0.0
            names.append(column)
        for column in self.categorical_features:
            values = frame[column].astype("string").fillna(MISSING_CATEGORY)
            categories = sorted(str(value) for value in values.unique().tolist())
            self._categories[column] = categories
            names.extend(f"{column}={category}" for category in categories)
        self._feature_names = names
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise FeatureSchemaError("Pipeline must be fitted before transform().")
        missing = [c for c in self.all_input_features if c not in frame.columns]
        if missing:
            raise FeatureSchemaError(f"Missing feature columns at transform time: {missing}")

        blocks: list[np.ndarray] = []
        for column in self.numeric_features:
            values = pd.to_numeric(frame[column], errors="coerce").astype("float64")
            values = values.fillna(self._medians[column]).to_numpy(dtype="float64")
            blocks.append(values.reshape(-1, 1))
        for column in self.categorical_features:
            values = frame[column].astype("string").fillna(MISSING_CATEGORY)
            categories = self._categories[column]
            block = np.zeros((len(frame), len(categories)), dtype="float64")
            positions = {category: index for index, category in enumerate(categories)}
            for row, value in enumerate(values.tolist()):
                index = positions.get(str(value))
                if index is not None:
                    block[row, index] = 1.0
            blocks.append(block)
        if not blocks:
            return np.zeros((len(frame), 0), dtype="float64")
        return np.hstack(blocks)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    def feature_names_out(self) -> list[str]:
        if not self._fitted:
            raise FeatureSchemaError("Pipeline must be fitted before reading feature names.")
        return list(self._feature_names)

    def to_schema(self) -> dict[str, Any]:
        if not self._fitted:
            raise FeatureSchemaError("Pipeline must be fitted before exporting a schema.")
        return {
            "schema_version": SCHEMA_VERSION,
            "numeric_features": list(self.numeric_features),
            "categorical_features": list(self.categorical_features),
            "target": self.target,
            "forbidden_columns": sorted(self.forbidden_columns),
            "numeric_medians": dict(self._medians),
            "categorical_vocabularies": {
                key: list(values) for key, values in self._categories.items()
            },
            "feature_names": list(self._feature_names),
            "missing_category": MISSING_CATEGORY,
        }


def infer_feature_types(
    frame: pd.DataFrame, columns: Iterable[str]
) -> tuple[list[str], list[str]]:
    """Split columns into numeric and categorical using the real dtypes."""
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        if column not in frame.columns:
            raise FeatureSchemaError(f"Unknown column: {column}")
        if pd.api.types.is_numeric_dtype(frame[column]):
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical
