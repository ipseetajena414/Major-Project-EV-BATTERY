"""Tests for the Member 1 -> Member 2 dataset integration layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from member1_pipeline import loader
from member2_modeling.data_integration import (
    DATASET_IDS,
    DatasetCompatibilityError,
    build_compatibility_report,
    feature_columns_for_target,
    profile_dataset,
    target_spec,
)

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts"


def test_profiles_cover_all_three_processed_datasets() -> None:
    profiles = {dataset_id: profile_dataset(dataset_id) for dataset_id in DATASET_IDS}
    assert set(profiles) == {
        "EV_FAILURE_PREDICTION",
        "SOLID_STATE_TELEMETRY_2026",
        "SYNTHETIC_BATTERY_LIFECYCLE",
    }
    for profile in profiles.values():
        assert profile["rows"] > 0
        assert profile["columns"] > 0
        assert profile["data_origin"]
        assert profile["record_grain"]
        assert profile["provenance"]["source_sha256"]
        assert profile["compatibility"]["sequence_available"] is False


def test_sequence_availability_is_never_inferred_from_row_order() -> None:
    profile = profile_dataset("EV_FAILURE_PREDICTION")
    assert profile["compatibility"]["sequence_available"] is False
    assert profile["compatibility"]["sequence_reason"]
    assert profile["compatibility"]["cycle_is_sequence_index"] is False


def test_ev_failure_target_excludes_health_derived_leakage() -> None:
    spec = target_spec("EV_FAILURE_PREDICTION", "battery_failure")
    assert spec["task"] == "classification"
    assert spec["training_permitted"] is True
    for column in (
        "state_of_health",
        "battery_health_percent",
        "capacity_loss_percent",
        "remaining_capacity",
        "predicted_remaining_life_cycles",
    ):
        assert column in spec["leakage_columns"]


def test_ev_soh_target_does_not_double_count_duplicate_labels() -> None:
    spec = target_spec("EV_FAILURE_PREDICTION", "state_of_health")
    assert spec["task"] == "regression"
    for column in (
        "battery_health_percent",
        "capacity_loss_percent",
        "remaining_capacity",
        "predicted_remaining_life_cycles",
    ):
        assert column in spec["leakage_columns"]
    assert "battery_health_percent" not in feature_columns_for_target(
        "EV_FAILURE_PREDICTION", "state_of_health"
    )


def test_solid_state_projected_cycle_life_is_not_observed_rul() -> None:
    spec = target_spec("SOLID_STATE_TELEMETRY_2026", "state_of_health_pct")
    assert spec["task"] == "regression"
    assert spec["rul_status"] == "UNAVAILABLE"
    assert "projected_cycle_life_80pct_soh" in spec["leakage_columns"]
    assert "projected_cycle_life_80pct_soh" not in feature_columns_for_target(
        "SOLID_STATE_TELEMETRY_2026", "state_of_health_pct"
    )


def test_synthetic_labels_are_marked_synthetic_research_only() -> None:
    spec = target_spec("SYNTHETIC_BATTERY_LIFECYCLE", "state_of_health")
    assert spec["task"] == "classification"
    assert spec["label_origin"] == "synthetic"
    assert spec["readiness"] == "RESEARCH_ONLY"
    assert spec["rul_status"] == "UNAVAILABLE"


def test_synthetic_soh_and_capacity_labels_cannot_cross_leak() -> None:
    capacity_spec = target_spec("SYNTHETIC_BATTERY_LIFECYCLE", "discharge_capacity_ah")
    assert "state_of_health" in capacity_spec["leakage_columns"]
    assert "state_of_health" not in feature_columns_for_target(
        "SYNTHETIC_BATTERY_LIFECYCLE", "discharge_capacity_ah"
    )

    soh_spec = target_spec("SYNTHETIC_BATTERY_LIFECYCLE", "state_of_health")
    assert "discharge_capacity_ah" in soh_spec["leakage_columns"]
    assert "discharge_capacity_ah" not in feature_columns_for_target(
        "SYNTHETIC_BATTERY_LIFECYCLE", "state_of_health"
    )


def test_feature_selection_rejects_target_and_leakage_columns() -> None:
    with pytest.raises(DatasetCompatibilityError):
        feature_columns_for_target(
            "EV_FAILURE_PREDICTION", "battery_failure", requested=["battery_failure"]
        )
    with pytest.raises(DatasetCompatibilityError):
        feature_columns_for_target(
            "EV_FAILURE_PREDICTION", "battery_failure", requested=["state_of_health"]
        )
    columns = feature_columns_for_target("EV_FAILURE_PREDICTION", "battery_failure")
    assert "battery_failure" not in columns
    assert "state_of_health" not in columns
    assert "battery_chemistry" in columns
    assert "cycle_count" in columns


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(KeyError):
        target_spec("SOLID_STATE_TELEMETRY_2026", "no_such_target")


def test_remaining_capacity_is_described_as_derived_not_measured() -> None:
    spec = target_spec("EV_FAILURE_PREDICTION", "remaining_capacity")
    assert spec["label_origin"] == "derived_from_health_percent"
    definition = spec["definition"]
    assert "derived algebraically" in definition
    assert "not an independent measurement" in definition
    assert (
        "verified only on rows where both battery_capacity_kwh and "
        "battery_health_percent are present"
    ) in definition
    assert "measured remaining" not in definition.lower()

    registry = json.loads(
        (ARTIFACT_ROOT / "snapshot_ml" / "model_registry.json").read_text(
            encoding="utf-8"
        )
    )
    supported = registry["supported_task_specs"]["ev_capacity"]
    assert supported["target_definition"] == definition
    assert supported["label_origin"] == "derived_from_health_percent"
    assert "measured remaining" not in supported["target_definition"].lower()
    task_notes = " ".join(registry["tasks"]["ev_capacity"]["limitations"])
    assert (
        "verified only on rows where battery_capacity_kwh and "
        "battery_health_percent are both present"
    ) in task_notes
    assert "not an independent pack-energy measurement" in task_notes

    config = json.loads(
        (ARTIFACT_ROOT / "snapshot_ml" / "ev_capacity" / "config.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["target"]["definition"] == definition
    assert config["task"]["target_definition"] == definition

    report = build_compatibility_report()
    target = report["datasets"]["EV_FAILURE_PREDICTION"]["compatibility"]["targets"][
        "remaining_capacity"
    ]
    assert target["definition"] == definition
    assert target["label_origin"] == "derived_from_health_percent"

    health_config = json.loads(
        (
            ARTIFACT_ROOT
            / "health"
            / "EV_FAILURE_PREDICTION"
            / "remaining_capacity"
            / "model_config.json"
        ).read_text(encoding="utf-8")
    )
    assert health_config["target_kind"] == "derived_from_health_percent"
    health_report = json.loads(
        (ARTIFACT_ROOT / "health" / "health_model_evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    health_target = health_report["datasets"]["EV_FAILURE_PREDICTION"]["targets"][
        "remaining_capacity"
    ]
    assert health_target["label_origin"] == "derived_from_health_percent"
    assert health_target["target_kind"] == "derived_from_health_percent"


def test_remaining_capacity_relation_is_verified_only_with_both_inputs() -> None:
    frame = loader.load_ev_failure(
        columns=[
            "battery_capacity_kwh",
            "battery_health_percent",
            "remaining_capacity",
        ]
    )
    inputs_present = frame[
        ["battery_capacity_kwh", "battery_health_percent"]
    ].notna().all(axis=1)
    target_present = frame["remaining_capacity"].notna()

    verifiable = frame[inputs_present & target_present]
    assert not verifiable.empty
    derived = (
        verifiable["battery_capacity_kwh"]
        * verifiable["battery_health_percent"]
        / 100
    ).round(2)
    assert np.allclose(
        derived, verifiable["remaining_capacity"], rtol=0, atol=1e-9
    )

    # The relation is only assertable where both inputs exist; the dataset has
    # rows in both unverifiable states and they must stay unclaimed.
    assert (inputs_present & ~target_present).any()
    assert (~inputs_present & target_present).any()


def test_compatibility_report_writes_json_and_markdown(tmp_path) -> None:
    report = build_compatibility_report(out_dir=tmp_path)
    assert report["active_master_table"] is None
    assert set(report["datasets"]) == set(DATASET_IDS)
    json_path = tmp_path / "member2_dataset_compatibility.json"
    md_path = tmp_path / "MEMBER2_DATASET_COMPATIBILITY_REPORT.md"
    assert json_path.is_file() and md_path.is_file()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["datasets"]["SOLID_STATE_TELEMETRY_2026"]["compatibility"][
        "sequence_available"
    ] is False
    assert "Observed RUL" in md_path.read_text(encoding="utf-8")
