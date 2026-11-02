"""Tests for hybrid evaluation helpers and baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from member2_modeling.hybrid.evaluation import (
    mean_fade_baseline,
    persistence_baseline,
    regression_metrics,
)


def test_regression_metrics_known_values() -> None:
    metrics = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(2.0 / 3.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(4.0 / 3.0))
    assert metrics["mape_pct"] == pytest.approx(100.0 * (2.0 / 3.0) / 3.0)


def test_regression_metrics_ignores_non_finite_pairs() -> None:
    metrics = regression_metrics([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])
    assert metrics["n"] == 1
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["r2"] is None


def test_persistence_baseline_uses_current_soh() -> None:
    frame = pd.DataFrame({"current_soh_pct": [99.0, 97.5, 96.0]})
    assert persistence_baseline(frame).tolist() == [99.0, 97.5, 96.0]


def test_mean_fade_uses_training_pairs_only() -> None:
    train = pd.DataFrame(
        {
            "cell_id": ["A", "A", "A"],
            "cycle_id": [1, 2, 3],
            "current_soh_pct": [100.0, 99.5, 99.0],
            "target_soh_pct": [99.5, 99.0, 98.5],
        }
    )
    holdout = pd.DataFrame(
        {
            "cell_id": ["A", "A"],
            "cycle_id": [4, 5],
            "current_soh_pct": [80.0, 79.0],
            "target_soh_pct": [60.0, 59.0],
        }
    )
    predictions = mean_fade_baseline(
        train,
        holdout,
        entity_column="cell_id",
        order_column="cycle_id",
        current_column="current_soh_pct",
        target_column="target_soh_pct",
    )
    # Training fade is exactly 0.5 % per observed record; holdout fades are ignored.
    assert predictions.tolist() == pytest.approx([79.5, 78.5])
