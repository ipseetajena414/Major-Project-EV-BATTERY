"""Tests for chemistry-aware tabular and sequence feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from member2_modeling.features import (
    FeatureSchemaError,
    SequenceFeatureBuilder,
    TabularFeaturePipeline,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chemistry": ["NMC", "LFP", "NMC", "LFP", "NMC", "LFP"],
            "voltage": [3.9, 3.2, 3.8, 3.1, 3.7, 3.0],
            "temperature": [25.0, np.nan, 30.0, np.nan, 35.0, np.nan],
            "cycle_count": [1, 2, 3, 4, 5, 6],
        }
    )


def test_pipeline_imputes_with_training_statistics_only() -> None:
    frame = sample_frame()
    train = frame.iloc[:4]
    test = pd.DataFrame(
        {
            "chemistry": ["NMC", "LFP"],
            "voltage": [3.6, 2.9],
            "temperature": [80.0, np.nan],
            "cycle_count": [7, 8],
        }
    )

    pipeline = TabularFeaturePipeline(
        numeric_features=["voltage", "temperature", "cycle_count"],
        categorical_features=["chemistry"],
    ).fit(train)
    out = pipeline.transform(test)
    names = pipeline.feature_names_out()
    temperature = out[:, names.index("temperature")]

    train_median = float(train["temperature"].median())
    assert temperature[1] == pytest.approx(train_median)
    assert temperature[1] != pytest.approx(float(test["temperature"].dropna().median()))
    assert temperature[1] != pytest.approx(float(test["temperature"].median()))


def test_pipeline_handles_unseen_category_without_crashing() -> None:
    frame = sample_frame()
    pipeline = TabularFeaturePipeline(
        numeric_features=["voltage"],
        categorical_features=["chemistry"],
    ).fit(frame.iloc[:4])
    unseen = pd.DataFrame(
        {"chemistry": ["Sodium-ion"], "voltage": [3.1], "cycle_count": [7]}
    )
    out = pipeline.transform(unseen)
    assert out.shape == (1, len(pipeline.feature_names_out()))
    chemistry_columns = [
        name for name in pipeline.feature_names_out() if name.startswith("chemistry=")
    ]
    assert chemistry_columns
    assert out[0, [pipeline.feature_names_out().index(c) for c in chemistry_columns]].sum() == 0


def test_pipeline_rejects_target_and_leakage_columns() -> None:
    frame = sample_frame().assign(state_of_health=[90, 91, 92, 93, 94, 95])
    with pytest.raises(FeatureSchemaError):
        TabularFeaturePipeline(
            numeric_features=["voltage", "state_of_health"],
            categorical_features=[],
            forbidden_columns=["state_of_health"],
        ).fit(frame)
    with pytest.raises(FeatureSchemaError):
        TabularFeaturePipeline(
            numeric_features=["voltage", "state_of_health"],
            categorical_features=[],
            target="state_of_health",
        ).fit(frame)


def test_pipeline_requires_fitted_state() -> None:
    pipeline = TabularFeaturePipeline(
        numeric_features=["voltage"], categorical_features=[]
    )
    with pytest.raises(FeatureSchemaError):
        pipeline.transform(sample_frame())


def test_sequence_windows_are_causal_and_aligned() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["A"] * 6,
            "cycle_id": [1, 2, 3, 4, 5, 6],
            "v": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "soh": [100.0, 99.0, 98.0, 97.0, 96.0, 95.0],
        }
    )
    builder = SequenceFeatureBuilder(
        feature_columns=["v"],
        target_column="soh",
        entity_column="cell_id",
        order_column="cycle_id",
        sequence_length=3,
    )
    X, y, meta = builder.fit_transform(frame)

    assert X.shape == (4, 3, 1)
    assert y.tolist() == [98.0, 97.0, 96.0, 95.0]
    schema = builder.to_schema()
    mean = schema["scaler_mean"][0]
    scale = schema["scaler_scale"][0]
    expected_first = (np.array([10.0, 11.0, 12.0]) - mean) / scale
    expected_last = (np.array([13.0, 14.0, 15.0]) - mean) / scale
    assert np.allclose(X[0, :, 0], expected_first)
    assert np.allclose(X[-1, :, 0], expected_last)
    assert meta["target_cycle_id"].tolist() == [3, 4, 5, 6]
    assert meta["entity"].tolist() == ["A"] * 4


def test_pipeline_handles_nullable_numeric_dtypes() -> None:
    frame = pd.DataFrame(
        {"v": pd.array([1.0, pd.NA, 3.0, 5.0], dtype="Float64")}
    )
    pipeline = TabularFeaturePipeline(numeric_features=["v"], categorical_features=[]).fit(
        frame
    )
    out = pipeline.transform(pd.DataFrame({"v": pd.array([pd.NA], dtype="Float64")}))
    assert out[0, 0] == pytest.approx(3.0)


def test_pipeline_never_uses_holdout_statistics_for_categories() -> None:
    train = pd.DataFrame({"chemistry": ["NMC", "NMC", "LFP"]})
    pipeline = TabularFeaturePipeline(
        numeric_features=[], categorical_features=["chemistry"]
    ).fit(train)
    out = pipeline.transform(pd.DataFrame({"chemistry": ["NCA", "NMC"]}))
    names = pipeline.feature_names_out()
    assert "chemistry=NCA" not in names
    assert out[0].sum() == 0
    assert out[1, names.index("chemistry=NMC")] == 1.0


def test_sequence_builder_fits_scaler_on_training_rows_only() -> None:
    train = pd.DataFrame(
        {
            "cell_id": ["A"] * 4,
            "cycle_id": [1, 2, 3, 4],
            "v": [10.0, 12.0, 14.0, 16.0],
            "soh": [100.0, 99.0, 98.0, 97.0],
        }
    )
    test = pd.DataFrame(
        {
            "cell_id": ["A"] * 4,
            "cycle_id": [5, 6, 7, 8],
            "v": [1000.0, 1000.0, 1000.0, 1000.0],
            "soh": [96.0, 95.0, 94.0, 93.0],
        }
    )
    builder = SequenceFeatureBuilder(
        feature_columns=["v"],
        target_column="soh",
        entity_column="cell_id",
        order_column="cycle_id",
        sequence_length=2,
    ).fit(train)
    X, _, _ = builder.transform(test)
    # Train mean 13, std sqrt(5); (1000-13)/sqrt(5) must show train-only scaling.
    assert X[0, 0, 0] == pytest.approx((1000.0 - 13.0) / np.sqrt(5.0))


def test_sequence_rejects_target_in_features() -> None:
    with pytest.raises(FeatureSchemaError):
        SequenceFeatureBuilder(
            feature_columns=["v", "soh"],
            target_column="soh",
            entity_column="cell_id",
            order_column="cycle_id",
            sequence_length=2,
        )


def test_pipeline_schema_is_versioned_and_json_safe() -> None:
    frame = sample_frame()
    pipeline = TabularFeaturePipeline(
        numeric_features=["voltage"],
        categorical_features=["chemistry"],
    ).fit(frame)
    schema = pipeline.to_schema()
    assert schema["schema_version"] == "2.0.0"
    assert schema["feature_names"] == pipeline.feature_names_out()
    assert schema["numeric_features"] == ["voltage"]
    assert schema["categorical_features"] == ["chemistry"]
