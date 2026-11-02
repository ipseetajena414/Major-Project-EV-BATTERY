"""Validation helpers for custom battery input records."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import pandas as pd


class CustomInputError(ValueError):
    """Raised when custom battery input cannot be validated."""


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        outcome = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(outcome) if isinstance(outcome, (bool, np.bool_)) else False


def normalize_records(records: Any) -> list[dict[str, Any]]:
    """Return a non-empty list of shallow-copied plain-dict records."""
    if isinstance(records, pd.DataFrame):
        items: Iterable[Any] = records.to_dict(orient="records")
    elif isinstance(records, Mapping):
        items = [records]
    else:
        try:
            items = list(records)
        except TypeError as error:
            raise CustomInputError(
                "records must be a mapping, a DataFrame, or an iterable of mappings"
            ) from error
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise CustomInputError(f"Record {index} must be a mapping of feature names to values")
        record: dict[str, Any] = {}
        for key, value in item.items():
            if not isinstance(key, str):
                raise CustomInputError(f"Record {index} has a non-string feature name: {key!r}")
            record[key] = value
        normalized.append(record)
    if not normalized:
        raise CustomInputError("At least one input record is required.")
    return normalized


def coerce_numeric(value: Any, field: str) -> float:
    if is_missing(value):
        raise CustomInputError(f"{field}: value is missing")
    if isinstance(value, (bool, np.bool_)):
        raise CustomInputError(f"{field}: expected a number, got a boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError as error:
            raise CustomInputError(f"{field}: expected a number, got {value!r}") from error
    else:
        raise CustomInputError(f"{field}: expected a number, got {type(value).__name__}")
    if not math.isfinite(number):
        raise CustomInputError(f"{field}: expected a finite number, got {value!r}")
    return number


def coerce_categorical(value: Any, field: str) -> str:
    if is_missing(value):
        raise CustomInputError(f"{field}: value is missing")
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return str(value)
    raise CustomInputError(
        f"{field}: expected a categorical scalar, got {type(value).__name__}"
    )


__all__ = [
    "CustomInputError",
    "coerce_categorical",
    "coerce_numeric",
    "is_missing",
    "normalize_records",
]
