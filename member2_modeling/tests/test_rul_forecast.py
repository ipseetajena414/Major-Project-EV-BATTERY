"""Tests for genuine sequential RUL forecasting and evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from member2_modeling.rul.evaluation import RULEvaluationError, evaluate_rul
from member2_modeling.rul.forecast import (
    FittedModel,
    RULForecastError,
    conformal_quantile,
    fit_candidate,
    make_history,
    make_origins,
    predict_candidate,
)


def _linear_trajectory(n=80, start_soh=100.0, fade_per_cycle=0.5):
    cycle_ids = np.arange(1, n + 1, dtype="int64")
    soh = start_soh - fade_per_cycle * np.arange(n, dtype="float64")
    return cycle_ids, soh


def test_history_rejects_non_causal_or_invalid_inputs() -> None:
    with pytest.raises(RULForecastError, match="strictly increasing"):
        make_history([1, 3, 2], [100.0, 99.0, 98.0])
    with pytest.raises(RULForecastError, match="At least 3"):
        make_history([1, 2], [100.0, 99.0])
    with pytest.raises(RULForecastError, match="finite"):
        make_history([1, 2, 3], [100.0, np.nan, 98.0])


def test_origins_use_observed_cycle_ids_and_never_future_inputs() -> None:
    cycle_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype="int64")
    soh = np.array([100.0, 98.0, 96.0, 94.0, 82.0, 80.5, 79.0, 78.0])
    origins = make_origins(
        "CELL", "DS", "NMC", cycle_ids, soh, eol_pct=80.0, step=2, min_history=3
    )
    assert origins
    crossing = 6  # first SOH strictly below 80
    assert origins[0].target_eol_cycle == int(cycle_ids[crossing])
    for origin in origins:
        assert origin.history.n - 1 < crossing
        assert origin.history.current_cycle < cycle_ids[crossing]
        assert origin.target_rul == cycle_ids[crossing] - origin.history.current_cycle


def test_linear_extrapolation_recovers_known_crossing() -> None:
    cycle_ids, soh = _linear_trajectory(n=60, start_soh=100.0, fade_per_cycle=0.5)
    history = make_history(cycle_ids, soh)  # current SOH = 70.5, already below EOL
    model = fit_candidate("linear_extrapolation", [], 80.0)
    assert predict_candidate(model, history, 80.0) == 0.0

    history = make_history(cycle_ids[:30], soh[:30])  # SOH 85.5 at cycle 30
    rul = predict_candidate(model, history, 80.0)
    assert rul == pytest.approx(11.0, abs=1e-6)
    # Extrapolation only sees the prefix: no future observations are required.
    truncated = make_history(cycle_ids[:30], soh[:30])
    assert predict_candidate(model, truncated, 80.0) == pytest.approx(rul)


def test_no_fade_returns_no_prediction_instead_of_fabricating_eol() -> None:
    cycle_ids = np.arange(1, 21, dtype="int64")
    flat = np.full(20, 95.0)
    model = fit_candidate("linear_extrapolation", [], 80.0)
    assert predict_candidate(model, make_history(cycle_ids, flat), 80.0) is None


def test_conformal_quantile_is_finite_and_monotone_in_coverage() -> None:
    residuals = [1.0, 2.0, 3.0, 4.0, 100.0]
    low = conformal_quantile(residuals, 0.6)
    high = conformal_quantile(residuals, 0.95)
    assert low is not None and high is not None
    assert low <= high
    assert high <= 100.0
    assert conformal_quantile([], 0.9) is None


def test_evaluation_keeps_pre_and_post_threshold_separate() -> None:
    truth = np.array([5.0, 3.0, 0.0, 0.0])
    predictions = np.array([7.0, 3.0, 1.0, 0.0])
    result = evaluate_rul(truth, predictions)
    assert result["counts"] == {"pre_threshold": 2, "post_threshold": 2, "near_eol": 2}
    assert result["pre_threshold"]["mae"] == pytest.approx(1.0)
    assert result["post_threshold"]["mae"] == pytest.approx(0.5)
    assert "never pooled" in result["pooling_rule"]


def test_evaluation_rejects_mismatched_lengths() -> None:
    with pytest.raises(RULEvaluationError):
        evaluate_rul([1.0, 2.0], [1.0])


def test_fitted_model_payload_round_trip_preserves_constant_rate() -> None:
    history = make_history(*_linear_trajectory(n=30))
    origins = [
        origin
        for origin in make_origins(
            "CELL",
            "DS",
            "LFP",
            *_linear_trajectory(n=60),
            eol_pct=80.0,
            step=2,
            min_history=10,
        )
    ]
    model = fit_candidate("median_fade_constant", origins, 80.0)
    restored = FittedModel.from_payload(model.to_payload())
    assert predict_candidate(restored, history, 80.0) == pytest.approx(
        predict_candidate(model, history, 80.0)
    )
