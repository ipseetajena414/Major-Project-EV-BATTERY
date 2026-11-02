"""Tests for the advanced analytics package contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from member2_modeling.analytics.health_status import (
    DISCLAIMER,
    HealthBandError,
    classify_soh,
    validate_bands,
)
from member2_modeling.analytics.sequential import sequential_support

ANALYTICS_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "analytics_v1"
)
REQUIRED_FILES = (
    "model_metrics.csv",
    "model_metrics.json",
    "prediction_results.csv",
    "regression_residuals.csv",
    "classification_results.csv",
    "confusion_matrices.json",
    "chemistry_metrics.csv",
    "dataset_summary.json",
    "model_comparison.csv",
    "prediction_provenance.json",
    "member3_combined_analytics_schema.json",
    "member3_results_template.csv",
    "MEMBER5_ANALYTICS_HANDOVER.md",
    "ANALYTICS_MANIFEST.json",
)


def test_health_bands_are_configurable_and_validated() -> None:
    assert classify_soh(95.0)["band"] == "GOOD"
    assert classify_soh(85.0)["band"] == "FAIR"
    assert classify_soh(75.0)["band"] == "POOR"
    assert classify_soh(50.0)["band"] == "CRITICAL"

    custom = [
        {"label": "PASS", "min_soh_pct": 85.0},
        {"label": "FAIL", "min_soh_pct": 0.0},
    ]
    assert classify_soh(84.9, custom)["band"] == "FAIL"
    assert classify_soh(85.0, custom)["band"] == "PASS"

    with pytest.raises(HealthBandError, match="lowest band"):
        validate_bands([{"label": "ONLY", "min_soh_pct": 50.0}])
    with pytest.raises(HealthBandError, match="Duplicate"):
        validate_bands(
            [
                {"label": "A", "min_soh_pct": 0.0},
                {"label": "A", "min_soh_pct": 90.0},
            ]
        )


def test_health_status_is_informational_only() -> None:
    assert "not a certified safety diagnosis" in DISCLAIMER


def test_snapshot_datasets_remain_unavailable_for_rul() -> None:
    support = sequential_support()
    for dataset_id in (
        "EV_FAILURE_PREDICTION",
        "SOLID_STATE_TELEMETRY_2026",
        "SYNTHETIC_BATTERY_LIFECYCLE",
    ):
        dataset = support["datasets"][dataset_id]
        assert dataset["status"] == "UNAVAILABLE"
        assert dataset["approved_for_use"] is False
        assert dataset["cells_with_observed_crossing"] == 0

    assert support["datasets"]["NASA_Ames_PCoE"]["status"] == "EXPERIMENTAL"
    assert support["datasets"]["NASA_Ames_PCoE"]["approved_for_use"] is False
    assert support["datasets"]["LFP_MultiStat_B5"]["status"] == "EXPERIMENTAL_CENSORED"
    assert support["datasets"]["Zenodo_IonTech_NaIon"]["status"] == "UNAVAILABLE"


@pytest.mark.skipif(
    not (ANALYTICS_DIR / "ANALYTICS_MANIFEST.json").is_file(),
    reason="analytics_v1 package has not been generated",
)
def test_manifest_covers_required_files_and_hashes() -> None:
    manifest = json.loads((ANALYTICS_DIR / "ANALYTICS_MANIFEST.json").read_text())
    names = {entry["name"] for entry in manifest["files"]}
    for filename in REQUIRED_FILES:
        if filename == "ANALYTICS_MANIFEST.json":
            continue
        assert filename in names, filename
    assert manifest["rul_approved_for_use"] is False
    assert manifest["member_directories_modified"] is False


@pytest.mark.skipif(
    not (ANALYTICS_DIR / "member3_combined_analytics_schema.json").is_file(),
    reason="analytics_v1 package has not been generated",
)
def test_combined_schema_is_draft07_and_never_claims_member3_data() -> None:
    schema = json.loads(
        (ANALYTICS_DIR / "member3_combined_analytics_schema.json").read_text()
    )
    assert schema["$schema"].endswith("draft-07/schema#")
    properties = schema["properties"]
    for field in (
        "battery_id",
        "dataset_id",
        "chemistry",
        "cycle_index",
        "soh_pct",
        "rul_cycles",
        "anomaly_flag",
        "health_band",
    ):
        assert field in properties
    assert "rul_status" in properties
    assert "range_km" in properties
    assert "member 3" in properties["range_km"].get("description", "").lower()
