"""Tests for the grouped multi-model Member 2 pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from member1_pipeline import loader
from member2_modeling.data_integration.profiles import (
    feature_columns_for_target,
    target_spec,
)
from member2_modeling.features import TabularFeaturePipeline
from member2_modeling.snapshot_ml import (
    TASKS,
    InferenceInputError,
    SnapshotInferenceAPI,
    estimate_range,
    estimate_range_from_record,
    load_registry,
    train_task,
    verify_registry,
)
from member2_modeling.snapshot_ml.range_estimate import RangeEstimateError
from member2_modeling.snapshot_ml.splits import (
    SplitError,
    make_grouped_split,
)

ARTIFACT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "snapshot_ml"


@pytest.fixture(scope="module")
def synthetic_frame() -> pd.DataFrame:
    return loader.load_dataset("SYNTHETIC_BATTERY_LIFECYCLE")


def test_every_task_has_no_target_or_leakage_feature() -> None:
    for spec in TASKS.values():
        features = feature_columns_for_target(spec.dataset_id, spec.target)
        forbidden = {spec.target, *target_spec(spec.dataset_id, spec.target)["leakage_columns"]}
        assert features, spec.task_id
        assert not (set(features) & forbidden), spec.task_id


def test_grouped_split_isolates_entities(synthetic_frame: pd.DataFrame) -> None:
    target = synthetic_frame["discharge_capacity_ah"].to_numpy(dtype="float64")
    split = make_grouped_split(
        synthetic_frame.reset_index(drop=True),
        target,
        entity_column="battery_id",
        task_type="regression",
        random_state=7,
    )
    entities = synthetic_frame["battery_id"].astype(str).to_numpy()
    train = set(entities[split.train])
    val = set(entities[split.val])
    test = set(entities[split.test])
    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert split.counts()["train"] + split.counts()["val"] + split.counts()["test"] == len(
        synthetic_frame
    )
    # Repeated synthetic battery ids must never straddle a boundary.
    repeated = set(synthetic_frame["battery_id"][synthetic_frame["battery_id"].duplicated()])
    for entity in repeated:
        present = sum(entity in group for group in (train, val, test))
        assert present == 1


def test_grouped_split_rejects_single_entity() -> None:
    frame = pd.DataFrame({"entity": ["a"] * 40, "target": np.arange(40.0)})
    with pytest.raises(SplitError):
        make_grouped_split(
            frame,
            frame["target"].to_numpy(),
            entity_column="entity",
            task_type="regression",
        )


def test_preprocessing_is_fitted_on_train_only() -> None:
    train = pd.DataFrame(
        {"x": [1.0, 2.0, np.nan, 4.0], "c": ["a", "a", "b", "b"]}
    )
    holdout = pd.DataFrame({"x": [100.0, 200.0], "c": ["z", "a"]})
    pipeline = TabularFeaturePipeline(
        numeric_features=["x"], categorical_features=["c"], target="y"
    ).fit(train)
    schema = pipeline.to_schema()
    assert schema["numeric_medians"]["x"] == pytest.approx(2.0)
    assert schema["categorical_vocabularies"]["c"] == ["a", "b"]
    transformed = pipeline.transform(holdout)
    # The holdout's 100/200 are retained; they do not become training medians.
    assert transformed[0, 0] == pytest.approx(100.0)
    assert transformed[1, 0] == pytest.approx(200.0)
    # Unseen category is all zeros, not an invented category.
    assert transformed[0, 1:].tolist() == [0.0, 0.0]
    assert train["x"].isna().sum() == 1


def test_regression_task_trains_reproducibly(tmp_path: Path) -> None:
    spec = TASKS["synthetic_capacity"]
    first = train_task(spec, output_root=tmp_path, random_state=13)
    second = train_task(spec, output_root=tmp_path, random_state=13)

    assert first["config"]["split"]["entity_overlap"] == {
        "train_val": 0,
        "train_test": 0,
        "val_test": 0,
    }
    assert first["config"]["split"]["sha256"] == second["config"]["split"]["sha256"]
    assert first["metrics"]["test"]["mae"] == pytest.approx(
        second["metrics"]["test"]["mae"], rel=1e-9
    )
    assert first["config"]["forbidden_columns"]
    config = json.loads((tmp_path / "synthetic_capacity" / "config.json").read_text())
    assert config["random_state"] == 13
    test_rows = pd.read_parquet(tmp_path / "synthetic_capacity" / "test_predictions.parquet")
    assert len(test_rows) == first["config"]["rows"]["test"]
    assert np.isfinite(test_rows["prediction"]).all()
    assert test_rows["prediction"].notna().all()


def test_classification_task_outputs_probabilities(tmp_path: Path) -> None:
    spec = TASKS["synthetic_class"]
    result = train_task(spec, output_root=tmp_path, random_state=5)
    metrics = result["metrics"]
    assert 0.0 <= metrics["test"]["balanced_accuracy"] <= 1.0
    test_rows = pd.read_parquet(tmp_path / "synthetic_class" / "test_predictions.parquet")
    probability_columns = [column for column in test_rows if column.startswith("probability_")]
    assert len(probability_columns) == 3
    sums = test_rows[probability_columns].sum(axis=1).to_numpy()
    assert np.allclose(sums, 1.0, atol=1e-8)


def test_range_estimate_is_labelled_non_measured() -> None:
    result = estimate_range(60.0, 90.0, discharge_efficiency_pct=95.0)
    assert result["is_measured"] is False
    assert result["validated_accuracy"] is False
    assert result["validation_status"] == "not_validated_no_range_target"
    assert result["range_km"] == pytest.approx(60.0 * 0.9 * 0.95 / 0.18)
    scenarios = result["range_km_scenarios"]
    assert scenarios["optimistic"] > scenarios["central"] > scenarios["conservative"]
    with pytest.raises(RangeEstimateError):
        estimate_range(60.0, 101.0)
    with pytest.raises(RangeEstimateError):
        estimate_range_from_record({"battery_capacity_kwh": 60.0})


def test_registry_checksums_when_artifacts_exist() -> None:
    if not (ARTIFACT_ROOT / "model_registry.json").is_file():
        pytest.skip("Full snapshot model registry has not been generated.")
    registry = load_registry(ARTIFACT_ROOT)
    assert set(registry["tasks"]) == set(TASKS)
    verification = verify_registry(ARTIFACT_ROOT)
    assert verification["ok"], verification["problems"]


def test_inference_accepts_valid_rows_and_rejects_unsafe_inputs() -> None:
    if not (ARTIFACT_ROOT / "model_registry.json").is_file():
        pytest.skip("Full snapshot model registry has not been generated.")
    api = SnapshotInferenceAPI(ARTIFACT_ROOT)

    frame = loader.load_dataset("SYNTHETIC_BATTERY_LIFECYCLE")
    features = api.info("synthetic_capacity")["required_features"]
    valid = frame.iloc[0][features].to_dict()

    response = api.predict("synthetic_capacity", [valid])
    assert len(response["predictions"]) == 1
    assert np.isfinite(response["predictions"][0]["prediction"])
    assert response["production_ready"] is False
    assert response["unit"].startswith("Ah")

    incomplete = dict(valid)
    incomplete.pop(features[0])
    with pytest.raises(InferenceInputError):
        api.predict("synthetic_capacity", [incomplete])

    wrong_chemistry = dict(valid)
    wrong_chemistry["electrolyte_type"] = "Unobtainium"
    with pytest.raises(InferenceInputError):
        api.predict("synthetic_capacity", [wrong_chemistry])
    allowed = api.predict(
        "synthetic_capacity", [wrong_chemistry], allow_unseen_chemistry=True
    )
    assert any("Unsupported chemistry" in warning for warning in allowed["input_validation"]["warnings"])

    with pytest.raises(KeyError):
        api.predict("does_not_exist", [valid])

    leakage = dict(valid)
    leakage["discharge_capacity_ah"] = 999.0
    ignored = api.predict("synthetic_capacity", [leakage])
    assert ignored["predictions"][0]["prediction"] == pytest.approx(
        response["predictions"][0]["prediction"]
    )


def test_classification_inference_probabilities_sum_to_one() -> None:
    if not (ARTIFACT_ROOT / "model_registry.json").is_file():
        pytest.skip("Full snapshot model registry has not been generated.")
    api = SnapshotInferenceAPI(ARTIFACT_ROOT)
    frame = loader.load_dataset("SYNTHETIC_BATTERY_LIFECYCLE")
    features = api.info("synthetic_class")["required_features"]
    response = api.predict("synthetic_class", frame.iloc[:3][features].to_dict("records"))
    for row in response["predictions"]:
        assert sum(row["probabilities"].values()) == pytest.approx(1.0, abs=1e-8)
