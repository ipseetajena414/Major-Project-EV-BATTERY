"""Tests for the fused LightGBM + BiLSTM hybrid SOH model.

The fusion layer must combine real branch predictions and must be trained on
out-of-fold predictions, never on in-sample branch outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from member2_modeling.hybrid import (
    HybridError,
    HybridSOHModel,
    OutOfFoldFusion,
    hybrid_eligibility,
)


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def test_fusion_learns_to_trust_the_more_informative_branch() -> None:
    rng = np.random.default_rng(0)
    y = np.linspace(95.0, 72.0, 240)
    tabular = y + rng.normal(0.0, 4.0, y.size)
    sequence = y + rng.normal(0.0, 0.2, y.size)
    available = np.ones(y.size, dtype=bool)

    fusion = OutOfFoldFusion(alpha=0.1).fit(tabular, sequence, available, y)
    fused = fusion.predict(tabular, sequence, available)
    description = fusion.describe()

    assert description["fusion_method"] == "out_of_fold_ridge"
    assert description["uses_sequence"] is True
    assert _mae(y, fused) < _mae(y, tabular)
    coefficients = description["coefficients"]
    assert abs(coefficients["sequence_pred"]) > abs(coefficients["tabular_pred"])


def test_fusion_falls_back_to_tabular_when_sequence_is_unavailable() -> None:
    rng = np.random.default_rng(1)
    y = np.linspace(96.0, 80.0, 120)
    tabular = y + rng.normal(0.0, 1.0, y.size)
    sequence = np.full(y.size, np.nan)
    available = np.zeros(y.size, dtype=bool)

    fusion = OutOfFoldFusion(alpha=1.0).fit(tabular, sequence, available, y)
    fused = fusion.predict(tabular, sequence, available)
    description = fusion.describe()

    assert description["uses_sequence"] is False
    assert np.all(np.isfinite(fused))
    assert _mae(y, fused) == pytest.approx(_mae(y, tabular), abs=0.35)


def test_fusion_rejects_mismatched_lengths() -> None:
    with pytest.raises(HybridError):
        OutOfFoldFusion().fit(
            np.array([1.0, 2.0]), np.array([1.0]), np.array([True, True]), np.array([1.0, 2.0])
        )


def _synthetic_sequential_frame(rows_per_cell: int = 45, cells: tuple[str, ...] = ("A", "B")):
    frames = []
    for offset, cell in enumerate(cells):
        positions = np.arange(rows_per_cell)
        soh = 100.0 - 0.10 * positions - 0.9 * offset + 0.2 * np.sin(positions / 3.0)
        voltage = 3.95 - 0.0035 * positions - 0.02 * offset
        current = 1.9 + 0.01 * np.cos(positions / 4.0)
        frame = pd.DataFrame(
            {
                "cell_id": cell,
                "cycle_id": positions + 1,
                "current_soh_pct": soh,
                "voltage": voltage,
                "current": current,
                "target_soh_pct": np.append(soh[1:], np.nan),
            }
        )
        frame = frame.iloc[:-1].reset_index(drop=True)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


FEATURES = ["current_soh_pct", "voltage", "current"]


def _build_model() -> HybridSOHModel:
    return HybridSOHModel(
        feature_columns=FEATURES,
        target_column="target_soh_pct",
        entity_column="cell_id",
        order_column="cycle_id",
        sequence_length=3,
        n_folds=2,
        random_state=7,
        tabular_params={
            "n_estimators": 25,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 3,
            "verbose": -1,
        },
        sequence_params={
            "hidden_dim": 8,
            "num_layers": 1,
            "epochs": 3,
            "batch_size": 16,
            "patience": 2,
            "learning_rate": 0.01,
        },
    )


def test_hybrid_fit_and_predict_with_real_fusion(tmp_path) -> None:
    frame = _synthetic_sequential_frame()
    model = _build_model().fit(frame)
    output = model.predict(frame)
    description = model.describe()

    expected = {
        "cell_id",
        "cycle_id",
        "tabular_pred",
        "sequence_pred",
        "fused_pred",
        "sequence_available",
    }
    assert expected.issubset(output.columns)
    assert len(output) == len(frame)
    fused = output["fused_pred"].to_numpy()
    assert np.all(np.isfinite(fused))
    assert np.all((fused >= 0.0) & (fused <= 100.0))
    assert bool(output["sequence_available"].any())

    assert description["schema_version"] == "2.0.0"
    assert description["readiness_status"] == "EXPERIMENTAL"
    assert description["fusion"]["fusion_method"] == "out_of_fold_ridge"
    assert description["fusion"]["trained_on"] == "out_of_fold_predictions"
    assert description["branches"]["tabular"]["fitted"] is True
    assert description["branches"]["sequence"]["sequence_length"] == 3

    destination = tmp_path / "hybrid"
    model.save(destination)
    restored = HybridSOHModel.load(destination)
    restored_output = restored.predict(frame)
    assert np.allclose(
        output["fused_pred"].to_numpy(), restored_output["fused_pred"].to_numpy(), atol=1e-6
    )


def test_hybrid_rejects_target_as_feature() -> None:
    with pytest.raises(HybridError):
        HybridSOHModel(
            feature_columns=["current_soh_pct", "target_soh_pct"],
            target_column="target_soh_pct",
            entity_column="cell_id",
            order_column="cycle_id",
            sequence_length=2,
        )


def test_hybrid_requires_fitted_state() -> None:
    model = _build_model()
    with pytest.raises(HybridError):
        model.predict(_synthetic_sequential_frame())


def test_hybrid_predict_keeps_rows_without_sequence_history() -> None:
    frame = _synthetic_sequential_frame()
    model = _build_model().fit(frame)

    short = frame.groupby("cell_id", sort=False).head(2).reset_index(drop=True)
    output = model.predict(short)

    assert len(output) == len(short)
    assert not bool(output["sequence_available"].any())
    assert np.all(np.isfinite(output["fused_pred"]))
    assert output["sequence_pred"].isna().all()
    assert np.all((output["fused_pred"] >= 0.0) & (output["fused_pred"] <= 100.0))


def test_hybrid_eligibility_never_enables_snapshot_sequences() -> None:
    for dataset_id in (
        "EV_FAILURE_PREDICTION",
        "SOLID_STATE_TELEMETRY_2026",
        "SYNTHETIC_BATTERY_LIFECYCLE",
    ):
        result = hybrid_eligibility(dataset_id)
        assert result["sequence_available"] is False
        assert result["hybrid_status"] == "UNAVAILABLE"
        assert result["reason"]

    legacy = hybrid_eligibility("LEGACY_NMC_SEQUENTIAL")
    assert legacy["sequence_available"] is True
    assert legacy["hybrid_status"] == "EXPERIMENTAL"
    assert legacy["reason"]
