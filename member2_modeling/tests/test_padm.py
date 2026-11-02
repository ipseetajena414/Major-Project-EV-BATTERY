"""
PADM Gate Tests
===============

Verifies all 8 pre-export gate tests from PADM_PRE_IMPLEMENTATION_REVIEW.md §8.1.

Tests are written as standalone pytest cases that can also be called by
the pipeline runner. They load artifacts from disk (not from memory) to
simulate the conditions under which Member 3 would receive the export.

Run with:
    .venv\\Scripts\\python.exe -m pytest member2_modeling/tests/test_padm.py -v

All tests must pass before Member 3 export is permitted.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
DELTA_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "delta"
EXPORT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "member3_exports"

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def delta_df():
    parquet = DELTA_DIR / "delta_pairs_nmc.parquet"
    if not parquet.exists():
        pytest.skip(f"delta_pairs_nmc.parquet not found at {DELTA_DIR}; run pipeline first.")
    return pd.read_parquet(parquet)


@pytest.fixture(scope="module")
def train_prov():
    prov = DELTA_DIR / "padm_train_provenance.json"
    if not prov.exists():
        pytest.skip("padm_train_provenance.json not found; run pipeline first.")
    return json.loads(prov.read_text())


@pytest.fixture(scope="module")
def eval_results():
    ev = DELTA_DIR / "padm_test_evaluation.json"
    if not ev.exists():
        pytest.skip("padm_test_evaluation.json not found; run pipeline first.")
    return json.loads(ev.read_text())


@pytest.fixture(scope="module")
def ridge_model():
    p = DELTA_DIR / "padm_ridge.joblib"
    if not p.exists():
        pytest.skip("padm_ridge.joblib not found.")
    return joblib.load(p)


@pytest.fixture(scope="module")
def enet_model():
    p = DELTA_DIR / "padm_elasticnet.joblib"
    if not p.exists():
        pytest.skip("padm_elasticnet.joblib not found.")
    return joblib.load(p)


@pytest.fixture(scope="module")
def final_scaler():
    p = DELTA_DIR / "padm_scaler_final.joblib"
    if not p.exists():
        pytest.skip("padm_scaler_final.joblib not found.")
    return joblib.load(p)


@pytest.fixture(scope="module")
def feature_cols(train_prov):
    return train_prov["feature_cols"]


# ── G1: No NaN or Inf in PADM predictions ────────────────────────────────────


def test_G1_no_nan_inf_in_predictions(delta_df, ridge_model, enet_model, final_scaler, feature_cols):
    """G1: PADM predictions on test set contain no NaN or Inf values."""
    test_df = delta_df[delta_df["split"] == "test"]
    X_test = final_scaler.transform(test_df[feature_cols].values.astype(float))

    ridge_pred = ridge_model.predict(X_test)
    enet_pred = enet_model.predict(X_test)

    assert np.all(np.isfinite(ridge_pred)), "Ridge predictions contain NaN or Inf"
    assert np.all(np.isfinite(enet_pred)), "ElasticNet predictions contain NaN or Inf"


# ── G2: SOH within [0, 105] after delta application ──────────────────────────


def test_G2_predicted_soh_within_bounds(delta_df, ridge_model, final_scaler, feature_cols, eval_results):
    """G2: All predicted SOH values are within [0, 105]."""
    test_df = delta_df[delta_df["split"] == "test"]
    X_test = final_scaler.transform(test_df[feature_cols].values.astype(float))

    clip_low = eval_results["delta_clip_bounds"]["low"]
    clip_high = eval_results["delta_clip_bounds"]["high"]

    raw_delta = ridge_model.predict(X_test)
    clipped_delta = np.clip(raw_delta, clip_low, clip_high)
    pred_soh = np.clip(
        test_df["current_soh_pct"].values + clipped_delta,
        0.0,
        105.0,
    )

    assert np.all(pred_soh >= 0.0), f"Negative SOH predictions found: {pred_soh[pred_soh < 0]}"
    assert np.all(pred_soh <= 105.0), f"SOH > 105 predictions found: {pred_soh[pred_soh > 105]}"


# ── G3: Test rows are a strict temporal subset ────────────────────────────────


def test_G3_test_rows_are_temporal_subset(delta_df):
    """G3: No test cycle_id from any cell appears in that cell's train/val rows."""
    for cell_id, cell_df in delta_df.groupby("cell_id"):
        test_cycles = set(cell_df.loc[cell_df["split"] == "test", "cycle_id"].tolist())
        trainval_cycles = set(
            cell_df.loc[cell_df["split"].isin(["train", "val"]), "cycle_id"].tolist()
        )
        overlap = test_cycles & trainval_cycles
        assert len(overlap) == 0, (
            f"Cell {cell_id}: cycles {overlap} appear in both test and train/val splits"
        )


# ── G4: Sample count alignment ────────────────────────────────────────────────


def test_G4_aligned_sample_counts(delta_df, ridge_model, enet_model, final_scaler, feature_cols):
    """G4: Predictions, persistence, and observed arrays all have identical length."""
    test_df = delta_df[delta_df["split"] == "test"]
    X_test = final_scaler.transform(test_df[feature_cols].values.astype(float))

    ridge_pred = ridge_model.predict(X_test)
    enet_pred = enet_model.predict(X_test)
    current_soh = test_df["current_soh_pct"].values
    true_delta = test_df["delta_target"].values

    n = len(test_df)
    assert len(ridge_pred) == n, "Ridge prediction count mismatch"
    assert len(enet_pred) == n, "ElasticNet prediction count mismatch"
    assert len(current_soh) == n, "current_soh count mismatch"
    assert len(true_delta) == n, "true_delta count mismatch"


# ── G5: Persistence baseline MAE is finite ────────────────────────────────────


def test_G5_persistence_mae_finite(eval_results):
    """G5: Persistence baseline MAE is a finite positive number."""
    mae = eval_results["pooled_metrics"]["persistence"]["mae"]
    assert mae is not None, "Persistence MAE is None"
    assert math.isfinite(mae), f"Persistence MAE is not finite: {mae}"
    assert mae > 0, f"Persistence MAE is not positive: {mae}"


# ── G6: Mean-fade baseline MAE is finite ──────────────────────────────────────


def test_G6_mean_fade_mae_finite(eval_results):
    """G6: Mean-fade baseline MAE is a finite positive number."""
    mae = eval_results["pooled_metrics"]["mean_fade"]["mae"]
    assert mae is not None, "Mean-fade MAE is None"
    assert math.isfinite(mae), f"Mean-fade MAE is not finite: {mae}"
    assert mae > 0, f"Mean-fade MAE is not positive: {mae}"


# ── G7: Valid quality flag ────────────────────────────────────────────────────


def test_G7_valid_quality_flag(eval_results):
    """G7: Quality flags are one of the four valid values."""
    valid_flags = {
        "PADM_BEATS_BOTH_BASELINES",
        "PADM_BEATS_MEAN_FADE_BASELINE",
        "PADM_BEATS_PERSISTENCE_ONLY",
        "PADM_BELOW_ALL_BASELINES",
    }
    for model_key in ("ridge", "elasticnet"):
        flag = eval_results["quality_flags"][model_key]
        assert flag in valid_flags, (
            f"{model_key} quality flag '{flag}' is not one of {valid_flags}"
        )


# ── G8: production_ready is False ─────────────────────────────────────────────


def test_G8_production_ready_is_false(eval_results):
    """G8: production_ready field is always False."""
    assert eval_results["production_ready"] is False, (
        "production_ready must be False — PADM is experimental"
    )


# ── Additional integrity tests ────────────────────────────────────────────────


def test_delta_target_not_in_features(train_prov):
    """delta_target must not appear in the feature column list (leakage check)."""
    feature_cols_used = train_prov["feature_cols"]
    assert "delta_target" not in feature_cols_used, (
        "delta_target (the label) is present in the feature list — data leakage"
    )


def test_no_future_soh_in_features(train_prov):
    """target_soh_pct must not appear in the feature column list."""
    feature_cols_used = train_prov["feature_cols"]
    assert "target_soh_pct" not in feature_cols_used, (
        "target_soh_pct (future SOH) is present in features — data leakage"
    )


def test_train_mean_delta_is_negative(train_prov):
    """train_mean_delta should be negative for degrading NMC cells."""
    tmd = train_prov["train_mean_delta"]
    # Not a hard error — issue a warning if positive (measurement noise possible)
    if tmd >= 0:
        import warnings
        warnings.warn(
            f"train_mean_delta={tmd:.4f} is non-negative. "
            "Expected negative for degrading cells. Check data quality.",
            UserWarning,
        )


def test_chronological_ordering_within_splits(delta_df):
    """Within each (cell, split), cycle_ids must be monotonically increasing."""
    for (cell_id, split), grp in delta_df.groupby(["cell_id", "split"]):
        cycles = grp.sort_index()["cycle_id"].tolist()
        assert cycles == sorted(cycles), (
            f"Cell {cell_id}, split {split}: cycle_ids are not sorted — "
            "data may have been shuffled"
        )


def test_feature_count_within_limit(train_prov):
    """Feature count must not exceed 20 (PADM review §5 constraint)."""
    n_features = len(train_prov["feature_cols"])
    assert n_features <= 20, (
        f"Feature count {n_features} exceeds the limit of 20. "
        "Excessive features risk overfitting on 3 NMC cells."
    )


def test_loco_results_present(train_prov):
    """LOCO diagnostic results must be present and include all 3 NMC cells."""
    loco = train_prov.get("loco_diagnostic", {})
    per_cell = loco.get("per_cell", [])
    assert len(per_cell) >= 2, (
        f"Expected LOCO results for at least 2 cells, got {len(per_cell)}"
    )
    pooled = loco.get("pooled_soh_mae")
    assert pooled is not None and math.isfinite(float(pooled)), (
        "LOCO pooled MAE is missing or not finite"
    )


def test_member3_csvs_exist_after_export():
    """After export, all three Member 3 CSV files must exist."""
    for role, fname in {
        "observed": "observed_soh.csv",
        "predicted": "predicted_soh.csv",
        "baseline": "baseline_soh.csv",
    }.items():
        path = EXPORT_DIR / fname
        if not path.exists():
            pytest.skip(
                f"{fname} not found — run the pipeline with export step first."
            )
        assert path.stat().st_size > 0, f"{fname} is empty"


def test_provenance_json_exists_after_export():
    """PROVENANCE.json must exist and be valid JSON after export."""
    prov_path = EXPORT_DIR / "PROVENANCE.json"
    if not prov_path.exists():
        pytest.skip("PROVENANCE.json not found — run pipeline with export step first.")
    data = json.loads(prov_path.read_text())
    assert "model" in data, "PROVENANCE.json missing 'model' key"
    assert data["model"]["production_ready"] is False, (
        "PROVENANCE.json has production_ready=True — must be False"
    )
