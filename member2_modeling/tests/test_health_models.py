"""Tests for snapshot-dataset SOH and capacity models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from member2_modeling.health import (
    HealthModelError,
    TabularHealthModel,
    capacity_semantics,
    derive_capacity_from_soh,
)


def _health_frame(rows: int = 360, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, rows)
    category = rng.choice(["A", "B", "C"], size=rows)
    offsets = {"A": -4.0, "B": 0.0, "C": 5.0}
    chemistry = np.where(np.arange(rows) % 2 == 0, "CHEM1", "CHEM2")
    target = (
        60.0
        + 8.0 * x
        + np.array([offsets[value] for value in category])
        + np.where(chemistry == "CHEM1", 2.0, -2.0)
        + rng.normal(0.0, 0.8, rows)
    )
    return pd.DataFrame(
        {
            "x": x,
            "category": category,
            "chemistry_family": chemistry,
            "target": target,
            "derived_leak": target + 0.25,
        }
    )


def _build_model() -> TabularHealthModel:
    return TabularHealthModel(
        target_column="target",
        feature_columns=["x", "category"],
        chemistry_column="chemistry_family",
        readiness="EXPERIMENTAL",
        random_state=11,
        model_params={
            "n_estimators": 40,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 5,
            "verbose": -1,
        },
    )


def test_tabular_health_model_beats_constant_baseline() -> None:
    frame = _health_frame()
    model = _build_model().fit(frame)
    description = model.describe()

    evaluation = description["evaluation"]
    assert evaluation["counts"]["test"] > 0
    assert evaluation["model"]["mae"] < evaluation["baselines"]["train_median"]["mae"] * 0.8
    assert evaluation["outperforms_train_median"] is True
    assert "CHEM1" in evaluation["per_chemistry"]
    assert "CHEM2" in evaluation["per_chemistry"]
    assert description["readiness"] == "EXPERIMENTAL"
    assert description["production_ready"] is False
    assert "snapshot" in " ".join(evaluation["caveats"]).lower()


def test_tabular_health_model_save_load_roundtrip(tmp_path) -> None:
    frame = _health_frame()
    model = _build_model().fit(frame)
    expected = model.predict(frame.iloc[:50])
    model.save(tmp_path / "health_model")
    restored = TabularHealthModel.load(tmp_path / "health_model")
    assert np.allclose(expected, restored.predict(frame.iloc[:50]), atol=1e-6)


def test_tabular_health_model_rejects_target_or_leakage_features() -> None:
    frame = _health_frame()
    with pytest.raises(HealthModelError):
        TabularHealthModel(
            target_column="target", feature_columns=["x", "target"]
        ).fit(frame)
    with pytest.raises(HealthModelError):
        TabularHealthModel(
            target_column="target",
            feature_columns=["x", "derived_leak"],
            forbidden_columns=["derived_leak"],
        ).fit(frame)


def test_tabular_health_model_requires_fitted_state() -> None:
    with pytest.raises(HealthModelError):
        _build_model().predict(_health_frame())


def test_imputation_statistics_do_not_use_test_rows() -> None:
    frame = _health_frame(rows=400)
    frame.loc[frame.index[:20], "x"] = np.nan
    frame.loc[frame.index[-10:], "x"] = 1e6
    model = _build_model().fit(frame)
    medians = model.describe()["numeric_medians"]
    splits = model.split_indices()
    train_values = frame["x"].to_numpy(dtype="float64")[splits["train"]]
    assert medians["x"] == pytest.approx(float(np.nanmedian(train_values)))


def test_capacity_semantics_distinguish_measured_predicted_and_algebraic() -> None:
    ev = capacity_semantics("EV_FAILURE_PREDICTION")
    assert ev["measured_column"] == "remaining_capacity"
    assert ev["measured_unit"] == "kWh"
    assert ev["is_pack_energy"] is True
    assert ev["predicted_capacity_available"] is True
    assert ev["algebraic"]["source_column"] == "battery_capacity_kwh"

    solid = capacity_semantics("SOLID_STATE_TELEMETRY_2026")
    assert solid["measured_column"] is None
    assert solid["predicted_capacity_available"] is False
    assert solid["algebraic"]["source_column"] == "nominal_capacity_ah"

    synthetic = capacity_semantics("SYNTHETIC_BATTERY_LIFECYCLE")
    assert synthetic["measured_column"] == "discharge_capacity_ah"
    assert synthetic["measured_unit"] == "Ah"
    assert synthetic["is_pack_energy"] is False
    assert synthetic["algebraic"] is None
    assert "synthetic" in " ".join(synthetic["notes"]).lower()


def test_derived_capacity_is_labelled_algebraic_not_measured() -> None:
    derived = derive_capacity_from_soh(
        80.0, 100.0, unit="Ah", scope="cell_nominal_capacity"
    )
    assert derived["capacity"] == pytest.approx(80.0)
    assert derived["kind"] == "algebraic"
    assert derived["basis"] == "soh_pct_times_rated_capacity"
    assert derived["is_measured"] is False
    assert derived["is_predicted"] is False
    assert derived["is_pack_energy"] is False


def test_derived_capacity_rejects_invalid_inputs() -> None:
    for soh, rated in ((-1.0, 100.0), (101.0, 100.0), (80.0, 0.0), (float("nan"), 100.0)):
        with pytest.raises(HealthModelError):
            derive_capacity_from_soh(soh, rated, unit="Ah", scope="cell_nominal_capacity")
