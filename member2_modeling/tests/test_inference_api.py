"""Tests for the custom battery input and prediction interface."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from member1_pipeline import loader
from member2_modeling.inference import BatteryIntelligenceAPI, CustomInputError


@pytest.fixture(scope="module")
def api() -> BatteryIntelligenceAPI:
    return BatteryIntelligenceAPI()


def _feature_record(api, dataset_id, target, row_number=0):
    frame = loader.load_dataset(dataset_id)
    row = frame.iloc[row_number]
    requirements = api.requirements(dataset_id, target)
    record = {}
    for column in requirements["feature_columns"]:
        value = row[column]
        if pd.isna(value):
            continue
        record[column] = value.item() if hasattr(value, "item") else value
    return record


def test_single_health_prediction_returns_bounded_soh_and_provenance(api) -> None:
    record = _feature_record(api, "EV_FAILURE_PREDICTION", "state_of_health")
    result = api.predict(
        "EV_FAILURE_PREDICTION", "state_of_health", [record], chemistry="NMC"
    )
    assert result["status"] == "ok"
    assert len(result["predictions"]) == 1
    prediction = result["predictions"][0]
    assert prediction["status"] == "ok"
    assert 0.0 <= prediction["prediction"] <= 100.0
    assert result["unit"] == "%"
    assert result["readiness"] == "EXPERIMENTAL"
    assert result["production_ready"] is False
    assert 0.0 < result["completeness"] <= 1.0
    assert result["provenance"]["source_sha256"]
    assert result["provenance"]["processed_sha256"]
    assert any("snapshot" in warning.lower() for warning in result["warnings"])


def test_batch_prediction_returns_one_row_per_input(api) -> None:
    record = _feature_record(api, "EV_FAILURE_PREDICTION", "state_of_health")
    result = api.predict(
        "EV_FAILURE_PREDICTION", "state_of_health", [record, record]
    )
    assert result["status"] == "ok"
    assert len(result["predictions"]) == 2
    assert [row["row"] for row in result["predictions"]] == [0, 1]


def test_missing_features_are_imputed_with_explicit_warning(api) -> None:
    result = api.predict(
        "EV_FAILURE_PREDICTION", "state_of_health", [{"cycle_count": 120.0}]
    )
    assert result["status"] == "ok"
    assert result["completeness"] < 0.2
    assert result["missing_features"]
    assert any("imput" in warning.lower() for warning in result["warnings"])


def test_target_and_leakage_fields_are_rejected(api) -> None:
    record = _feature_record(api, "EV_FAILURE_PREDICTION", "state_of_health")
    leaked = dict(record, state_of_health=50.0)
    result = api.predict("EV_FAILURE_PREDICTION", "state_of_health", [leaked])
    assert result["status"] == "error"
    assert result["predictions"][0]["status"] == "error"
    assert "leakage" in result["predictions"][0]["error"].lower()

    mixed = api.predict(
        "EV_FAILURE_PREDICTION", "state_of_health", [leaked, record]
    )
    assert mixed["status"] == "partial"
    assert mixed["predictions"][1]["status"] == "ok"


def test_invalid_numeric_and_unknown_chemistry_are_rejected(api) -> None:
    bad_number = api.predict(
        "EV_FAILURE_PREDICTION", "state_of_health", [{"cycle_count": "not-a-number"}]
    )
    assert bad_number["status"] == "error"
    assert "cycle_count" in bad_number["predictions"][0]["error"]

    record = _feature_record(api, "EV_FAILURE_PREDICTION", "state_of_health")
    bad_chemistry = api.predict(
        "EV_FAILURE_PREDICTION", "state_of_health", [record], chemistry="Unobtainium"
    )
    assert bad_chemistry["status"] == "error"
    assert "chemistry" in bad_chemistry["predictions"][0]["error"].lower()


def test_capacity_route_reports_predicted_pack_energy(api) -> None:
    record = _feature_record(api, "EV_FAILURE_PREDICTION", "remaining_capacity")
    result = api.predict("EV_FAILURE_PREDICTION", "remaining_capacity", [record])
    assert result["status"] == "ok"
    assert result["unit"] == "kWh"
    assert result["prediction_kind"] == "model_prediction"
    assert result["capacity_kind"] == "predicted"
    assert result["predictions"][0]["prediction"] >= 0.0
    assert any("kWh" in warning for warning in result["warnings"])


def test_synthetic_capacity_is_marked_research_only(api) -> None:
    record = _feature_record(
        api, "SYNTHETIC_BATTERY_LIFECYCLE", "discharge_capacity_ah"
    )
    result = api.predict(
        "SYNTHETIC_BATTERY_LIFECYCLE", "discharge_capacity_ah", [record]
    )
    assert result["status"] == "ok"
    assert result["readiness"] == "RESEARCH_ONLY"
    assert result["label_origin"] == "synthetic"
    assert any("synthetic" in warning.lower() for warning in result["warnings"])


def test_algebraic_capacity_is_never_labelled_measured(api) -> None:
    record = _feature_record(api, "EV_FAILURE_PREDICTION", "state_of_health")
    record["battery_capacity_kwh"] = 75.0
    result = api.predict_capacity_from_soh("EV_FAILURE_PREDICTION", [record])
    assert result["status"] == "ok"
    prediction = result["predictions"][0]
    assert prediction["kind"] == "algebraic"
    assert prediction["is_measured"] is False
    assert prediction["is_predicted"] is False
    assert prediction["unit"] == "kWh"
    assert math.isclose(
        prediction["capacity"],
        prediction["soh_pct"] * 75.0 / 100.0,
        rel_tol=1e-9,
    )


def test_solid_state_algebraic_capacity_uses_ah(api) -> None:
    record = _feature_record(
        api, "SOLID_STATE_TELEMETRY_2026", "state_of_health_pct"
    )
    record["nominal_capacity_ah"] = 1.2
    result = api.predict_capacity_from_soh("SOLID_STATE_TELEMETRY_2026", [record])
    assert result["status"] == "ok"
    assert result["predictions"][0]["unit"] == "Ah"
    assert result["predictions"][0]["kind"] == "algebraic"


def test_synthetic_algebraic_capacity_is_unsupported(api) -> None:
    result = api.predict_capacity_from_soh(
        "SYNTHETIC_BATTERY_LIFECYCLE", [{"cycle_count": 10.0}]
    )
    assert result["status"] == "unsupported"


def test_rul_is_unavailable_for_snapshots_and_insufficient_sequences(api) -> None:
    snapshot = api.predict_rul("SOLID_STATE_TELEMETRY_2026")
    assert snapshot["status"] == "unavailable"
    assert snapshot["approved_for_use"] is False
    assert snapshot["reason"]

    na_ion = api.predict_rul("Zenodo_IonTech_NaIon")
    assert na_ion["status"] == "unavailable"
    assert na_ion["approved_for_use"] is False
    assert na_ion["predictions"] == []


def test_rul_legacy_nmc_is_experimental_and_never_approved(api) -> None:
    legacy = api.predict_rul("LEGACY_NMC_SEQUENTIAL")
    assert legacy["status"] == "ok"
    assert legacy["approved_for_use"] is False
    assert legacy["readiness"] == "EXPERIMENTAL"
    assert len(legacy["predictions"]) == 3
    for record in legacy["predictions"]:
        assert record["model_id"]
        assert record["validation_status"].startswith("EXPERIMENTAL")
        assert record["current_cycle"] > 0
        assert record["predicted_eol_cycle"] > record["current_cycle"]
        assert 0 <= record["estimated_rul_cycles"]
        assert (
            record["lower_uncertainty_bound"]
            <= record["estimated_rul_cycles"]
            <= record["upper_uncertainty_bound"]
        )


def test_rul_censored_lfp_cell_reports_no_validated_accuracy(api) -> None:
    censored = api.predict_rul("LFP_MultiStat_B5")
    assert censored["status"] == "ok"
    assert censored["approved_for_use"] is False
    assert len(censored["predictions"]) == 1
    record = censored["predictions"][0]
    assert record["prediction_status"] == "censored_projection"
    assert record["validation_status"] == "EXPERIMENTAL_CENSORED_NO_EOL"
    assert any("not a calibrated" in warning for warning in record["warnings"])


def test_sequence_prediction_uses_causal_windows_and_fallback(api) -> None:
    features = api.hybrid_requirements()["feature_columns"]
    rows = []
    for index in range(5):
        row = {column: 1.0 for column in features}
        row["current_soh_pct"] = 90.0 - index
        row["v_mean"] = 3.6 - 0.01 * index
        rows.append(row)
    result = api.predict_sequence(rows, chemistry="NMC", cell_id="CUSTOM")
    assert result["status"] == "ok"
    assert len(result["predictions"]) == 5
    availability = [row["sequence_available"] for row in result["predictions"]]
    assert availability == [False, False, False, False, True]
    assert all(math.isfinite(row["prediction"]) for row in result["predictions"])
    assert result["unit"] == "%"
    assert result["readiness"] == "EXPERIMENTAL"
    assert any("persistence" in warning.lower() for warning in result["warnings"])


def test_short_sequence_falls_back_to_tabular_predictions(api) -> None:
    features = api.hybrid_requirements()["feature_columns"]
    rows = [{column: 1.0 for column in features} for _ in range(3)]
    result = api.predict_sequence(rows, chemistry="NMC")
    assert result["status"] == "ok"
    assert all(not row["sequence_available"] for row in result["predictions"])
    assert any("fallback" in warning.lower() for warning in result["warnings"])


def test_sequence_route_rejects_unsupported_chemistry(api) -> None:
    result = api.predict_sequence([{"v_mean": 3.6}], chemistry="LFP")
    assert result["status"] == "unsupported"
    assert "NMC" in result["reason"]


def test_empty_input_is_rejected(api) -> None:
    with pytest.raises(CustomInputError):
        api.predict("EV_FAILURE_PREDICTION", "state_of_health", [])
    with pytest.raises(CustomInputError):
        api.predict_sequence([], chemistry="NMC")
    with pytest.raises(KeyError):
        api.predict("EV_FAILURE_PREDICTION", "no_such_target", [{"cycle_count": 1.0}])
