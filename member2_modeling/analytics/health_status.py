"""Configurable informational battery-health bands.

Bands are a communication aid for analysts and students.  They are not a
certified safety diagnosis, not a warranty determination, and not an
operational dispatch rule.  Thresholds are explicit, validated, and can be
overridden by the caller.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np
import pandas as pd

from member2_modeling.analytics.sequential import cell_summary
from member2_modeling.rul.labels import EOL_PCT

DEFAULT_BANDS = (
    {"label": "GOOD", "min_soh_pct": 90.0},
    {"label": "FAIR", "min_soh_pct": 80.0},
    {"label": "POOR", "min_soh_pct": 70.0},
    {"label": "CRITICAL", "min_soh_pct": 0.0},
)

DISCLAIMER = (
    "Informational health status only. It is not a certified safety diagnosis "
    "and must not be used as the sole basis for operational decisions."
)


class HealthBandError(ValueError):
    """Raised when health bands are invalid or ambiguous."""


def validate_bands(bands: Iterable[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    selected = list(bands) if bands is not None else list(DEFAULT_BANDS)
    if not selected:
        raise HealthBandError("At least one health band is required")
    normalized: list[dict[str, Any]] = []
    seen = set()
    for band in selected:
        label = str(band.get("label", "")).strip()
        if not label:
            raise HealthBandError("Every band needs a non-empty label")
        if label in seen:
            raise HealthBandError(f"Duplicate health band label {label!r}")
        seen.add(label)
        try:
            threshold = float(band["min_soh_pct"])
        except (KeyError, TypeError, ValueError) as exc:
            raise HealthBandError(
                f"Band {label!r} needs a numeric min_soh_pct"
            ) from exc
        if not 0.0 <= threshold <= 100.0:
            raise HealthBandError(f"Band {label!r} threshold must be within 0-100")
        normalized.append({"label": label, "min_soh_pct": threshold})
    normalized.sort(key=lambda item: item["min_soh_pct"], reverse=True)
    thresholds = [item["min_soh_pct"] for item in normalized]
    if len(set(thresholds)) != len(thresholds):
        raise HealthBandError("Health band thresholds must be unique")
    if thresholds[-1] > 0.0:
        raise HealthBandError("The lowest band must start at 0% SOH")
    return normalized


def classify_soh(
    soh_pct: float,
    bands: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    value = float(soh_pct)
    if not np.isfinite(value):
        raise HealthBandError("soh_pct must be finite")
    normalized = validate_bands(bands)
    for band in normalized:
        if value >= band["min_soh_pct"]:
            return {
                "soh_pct": value,
                "band": band["label"],
                "band_min_soh_pct": band["min_soh_pct"],
                "bands": normalized,
            }
    return {
        "soh_pct": value,
        "band": normalized[-1]["label"],
        "band_min_soh_pct": normalized[-1]["min_soh_pct"],
        "bands": normalized,
    }


def build_health_status(
    bands: Iterable[dict[str, Any]] | None = None,
    *,
    eol_pct: float = EOL_PCT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized = validate_bands(bands)
    summary = cell_summary(eol_pct=eol_pct)
    rows = []
    for record in summary.to_dict("records"):
        classification = classify_soh(record["soh_last_pct"], normalized)
        rows.append(
            {
                "battery_id": record["cell_id"],
                "dataset_id": record["dataset_id"],
                "chemistry": record["chemistry"],
                "current_cycle": int(record["last_cycle_id"]),
                "current_soh_pct": float(record["soh_last_pct"]),
                "soh_min_pct": float(record["soh_min_pct"]),
                "health_band": classification["band"],
                "band_min_soh_pct": classification["band_min_soh_pct"],
                "informational_only": True,
                "certified_safety_diagnosis": False,
                "status_basis": "observed_cycle_level_soh",
                "bands_json": json.dumps(normalized),
                "disclaimer": DISCLAIMER,
            }
        )
    frame = pd.DataFrame(rows)
    counts = (
        frame["health_band"].value_counts().to_dict() if len(frame) else {}
    )
    report = {
        "bands": normalized,
        "band_definition": (
            "A cell is assigned the first band (highest min_soh_pct) whose "
            "threshold is less than or equal to its latest observed SOH."
        ),
        "counts": counts,
        "cells": int(len(frame)),
        "informational_only": True,
        "certified_safety_diagnosis": False,
        "disclaimer": DISCLAIMER,
        "limitations": [
            "Bands are project-defined and are not manufacturer limits.",
            "A single latest SOH value ignores rate of change and operating context.",
            "No cell-level safety event or field failure is inferred.",
        ],
    }
    return frame, report


__all__ = [
    "DEFAULT_BANDS",
    "DISCLAIMER",
    "HealthBandError",
    "build_health_status",
    "classify_soh",
    "validate_bands",
]
