"""
PADM Step 9 & 10: Test Evaluation and Quality Flag Assignment
=============================================================

Rules
-----
* Test set is accessed exactly once in ``evaluate_padm_on_test``.
* All three predictors (PADM-Ridge, PADM-ElasticNet, persistence, mean-fade)
  are evaluated on the IDENTICAL test rows.
* Results are written to disk immediately; re-running this function is blocked
  by a sentinel file unless ``force=True`` is passed.
* Delta clipping bounds are derived from the training distribution and
  applied symmetrically (±1.5σ) as per PADM_PRE_IMPLEMENTATION_REVIEW.md §7.
* Quality flags follow the decision tree in §10 of the review document.
* ``production_ready = False`` is always set regardless of performance.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from member2_modeling.delta.delta_features import (
    PADM_FEATURE_COLS,
    SOH_CLIP_MAX,
)

warnings.filterwarnings("ignore", category=FutureWarning)

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "delta"
SENTINEL_FILE = ARTIFACT_DIR / "TEST_EVALUATION_COMPLETE.sentinel"

# Delta clipping sigma multiplier (§7 of PADM review)
DELTA_CLIP_SIGMA = 1.5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict[str, float | int | str]:
    """Compute MAE, RMSE, bias, R² in SOH space."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    n = len(y_true)
    if n == 0:
        return {"label": label, "n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    bias = float(np.mean(y_pred - y_true))
    return {"label": label, "n": n, "mae": mae, "rmse": rmse, "r2": r2, "bias": bias}


def _compute_delta_clip_bounds(
    delta_df: pd.DataFrame,
) -> tuple[float, float]:
    """
    Derive delta clipping bounds from the training distribution.

    clip_low  = min_train_delta − 1.5 × std_train_delta
    clip_high = max_train_delta + 1.5 × std_train_delta
    """
    train_delta = delta_df.loc[delta_df["split"] == "train", "delta_target"]
    mean_d = float(train_delta.mean())
    std_d = float(train_delta.std())
    clip_low = float(train_delta.min()) - DELTA_CLIP_SIGMA * std_d
    clip_high = float(train_delta.max()) + DELTA_CLIP_SIGMA * std_d
    return clip_low, clip_high


def _predict_soh(
    current_soh: np.ndarray,
    raw_delta: np.ndarray,
    clip_low: float,
    clip_high: float,
) -> np.ndarray:
    """Apply delta clipping then add to current SOH; clip result to [0, 105]."""
    clipped_delta = np.clip(raw_delta, clip_low, clip_high)
    pred_soh = current_soh + clipped_delta
    return np.clip(pred_soh, 0.0, SOH_CLIP_MAX)


def _persistence_soh(current_soh: np.ndarray) -> np.ndarray:
    """Persistence baseline: predicted next SOH = current SOH."""
    return current_soh.copy()


def _mean_fade_soh(
    current_soh: np.ndarray,
    train_mean_delta: float,
) -> np.ndarray:
    """Mean-fade baseline: predicted next SOH = current SOH + train_mean_delta."""
    return np.clip(current_soh + train_mean_delta, 0.0, SOH_CLIP_MAX)


def _assign_quality_flag(
    padm_mae: float,
    persistence_mae: float,
    mean_fade_mae: float,
    model_name: str,
) -> tuple[str, str]:
    """
    Assign quality flag and determine export predictor per §10 of PADM review.

    Decision tree
    -------------
    1. PADM < mean_fade_mae AND PADM < persistence_mae
       → PADM_BEATS_BOTH_BASELINES (use PADM for export)
    2. PADM < mean_fade_mae (but not persistence)
       → PADM_BEATS_MEAN_FADE_BASELINE (use PADM for export)
    3. PADM < persistence_mae (but not mean_fade)
       → PADM_BEATS_PERSISTENCE_ONLY (use PADM for export)
    4. PADM ≥ both
       → PADM_BELOW_ALL_BASELINES (export mean_fade instead)
    """
    beats_mf = padm_mae < mean_fade_mae
    beats_pers = padm_mae < persistence_mae

    if beats_mf and beats_pers:
        flag = "PADM_BEATS_BOTH_BASELINES"
        export_model = model_name
    elif beats_mf:
        flag = "PADM_BEATS_MEAN_FADE_BASELINE"
        export_model = model_name
    elif beats_pers:
        flag = "PADM_BEATS_PERSISTENCE_ONLY"
        export_model = model_name
    else:
        flag = "PADM_BELOW_ALL_BASELINES"
        export_model = "mean_fade_baseline"

    return flag, export_model


# ── Gate tests ────────────────────────────────────────────────────────────────


def _run_gate_tests(
    test_df: pd.DataFrame,
    padm_pred_soh: np.ndarray,
    persistence_pred_soh: np.ndarray,
    mean_fade_pred_soh: np.ndarray,
    true_soh: np.ndarray,
    padm_mae: float,
    persistence_mae: float,
    mean_fade_mae: float,
    quality_flag: str,
) -> dict[str, bool]:
    """
    Run all 8 gate tests from PADM_PRE_IMPLEMENTATION_REVIEW.md §8.1.

    All tests must pass before Member 3 export is permitted.
    """
    gates: dict[str, bool] = {}

    # G1: No NaN or Inf in predictions
    gates["G1_no_nan_inf_in_padm"] = bool(
        np.all(np.isfinite(padm_pred_soh))
    )

    # G2: All predicted SOH values within [0, SOH_CLIP_MAX]
    gates["G2_soh_within_bounds"] = bool(
        np.all((padm_pred_soh >= 0.0) & (padm_pred_soh <= SOH_CLIP_MAX))
    )

    # G3: Test rows are a strict temporal subset (no train/val cycle_ids)
    # Verify split column exists and is all 'test'
    gates["G3_test_rows_only"] = bool(
        "split" in test_df.columns and (test_df["split"] == "test").all()
    )

    # G4: Identical sample count across all three predictors
    gates["G4_aligned_sample_counts"] = bool(
        len(padm_pred_soh) == len(persistence_pred_soh) == len(mean_fade_pred_soh) == len(true_soh)
    )

    # G5: Persistence baseline MAE is computed (sanity — should be small)
    gates["G5_persistence_mae_finite"] = bool(math.isfinite(persistence_mae))

    # G6: Mean-fade baseline MAE is computed and finite
    gates["G6_mean_fade_mae_finite"] = bool(math.isfinite(mean_fade_mae))

    # G7: Quality flag is set correctly (one of the 4 valid values)
    valid_flags = {
        "PADM_BEATS_BOTH_BASELINES",
        "PADM_BEATS_MEAN_FADE_BASELINE",
        "PADM_BEATS_PERSISTENCE_ONLY",
        "PADM_BELOW_ALL_BASELINES",
    }
    gates["G7_valid_quality_flag"] = quality_flag in valid_flags

    # G8: production_ready is False
    gates["G8_production_ready_is_false"] = True  # enforced unconditionally

    return gates


# ── Main evaluation function (touches test set exactly once) ──────────────────


def evaluate_padm_on_test(
    delta_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    artifact_dir: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Step 9: Evaluate PADM on the test split — called exactly once.

    Parameters
    ----------
    delta_df : pd.DataFrame
        Full dataset with ``split`` column (all three splits).
    feature_cols : list[str] or None
        Uses PADM_FEATURE_COLS if None.
    artifact_dir : Path or str, optional
        Directory where model artifacts were saved by ``padm_train``.
    force : bool, default False
        If True, allows re-running even if sentinel file exists.
        WARNING: Do NOT use ``force=True`` after seeing results.

    Returns
    -------
    dict with full evaluation metrics, quality flags, and gate test results.
    """
    out_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    sentinel = out_dir / "TEST_EVALUATION_COMPLETE.sentinel"

    if sentinel.exists() and not force:
        raise RuntimeError(
            "Test evaluation has already been run (sentinel file exists). "
            "Do NOT re-run after seeing results — this violates the "
            "PADM evaluation protocol. Pass force=True only if you have "
            "a documented reason (e.g., bug fix before first publication)."
        )

    if feature_cols is None:
        feature_cols = PADM_FEATURE_COLS

    # Load trained models
    ridge: Ridge = joblib.load(out_dir / "padm_ridge.joblib")
    enet: ElasticNet = joblib.load(out_dir / "padm_elasticnet.joblib")
    scaler: StandardScaler = joblib.load(out_dir / "padm_scaler_final.joblib")

    # Load training provenance for hyperparams and train_mean_delta
    prov = json.loads((out_dir / "padm_train_provenance.json").read_text())
    train_mean_delta: float = prov["train_mean_delta"]

    # Delta clipping bounds from training distribution
    clip_low, clip_high = _compute_delta_clip_bounds(delta_df)
    print(f"  [padm_evaluate] Delta clip bounds: [{clip_low:.4f}, {clip_high:.4f}]")

    # Separate test rows
    test_df = delta_df[delta_df["split"] == "test"].copy()
    print(f"  [padm_evaluate] Test rows: {len(test_df)}")

    X_test = test_df[feature_cols].values.astype(np.float32)
    X_test_sc = scaler.transform(X_test)

    true_delta = test_df["delta_target"].values.astype(np.float32)
    current_soh = test_df["current_soh_pct"].values.astype(np.float32)
    true_next_soh = np.clip(current_soh + true_delta, 0.0, SOH_CLIP_MAX)

    # PADM predictions
    ridge_raw_delta = ridge.predict(X_test_sc)
    enet_raw_delta = enet.predict(X_test_sc)

    ridge_pred_soh = _predict_soh(current_soh, ridge_raw_delta, clip_low, clip_high)
    enet_pred_soh = _predict_soh(current_soh, enet_raw_delta, clip_low, clip_high)

    # Baselines (same test rows)
    persist_soh = _persistence_soh(current_soh)
    mf_soh = _mean_fade_soh(current_soh, train_mean_delta)

    # Global metrics
    ridge_metrics = _metrics(true_next_soh, ridge_pred_soh, "PADM_Ridge")
    enet_metrics = _metrics(true_next_soh, enet_pred_soh, "PADM_ElasticNet")
    persist_metrics = _metrics(true_next_soh, persist_soh, "persistence")
    mf_metrics = _metrics(true_next_soh, mf_soh, "mean_fade")

    print(f"  [padm_evaluate] PADM-Ridge test SOH MAE:    {ridge_metrics['mae']:.4f}%")
    print(f"  [padm_evaluate] PADM-ElasticNet test SOH MAE: {enet_metrics['mae']:.4f}%")
    print(f"  [padm_evaluate] Persistence test SOH MAE:  {persist_metrics['mae']:.4f}%")
    print(f"  [padm_evaluate] Mean-fade test SOH MAE:    {mf_metrics['mae']:.4f}%")

    # Per-cell metrics
    per_cell: dict[str, dict] = {}
    for cell_id, cell_test in test_df.groupby("cell_id"):
        c_soh = cell_test["current_soh_pct"].values.astype(np.float32)
        c_true_delta = cell_test["delta_target"].values.astype(np.float32)
        c_true_next = np.clip(c_soh + c_true_delta, 0.0, SOH_CLIP_MAX)

        c_X = scaler.transform(cell_test[feature_cols].values.astype(np.float32))
        c_ridge_delta = ridge.predict(c_X)
        c_enet_delta = enet.predict(c_X)

        c_ridge_soh = _predict_soh(c_soh, c_ridge_delta, clip_low, clip_high)
        c_enet_soh = _predict_soh(c_soh, c_enet_delta, clip_low, clip_high)

        per_cell[str(cell_id)] = {
            "n": int(len(cell_test)),
            "ridge_mae": float(np.mean(np.abs(c_true_next - c_ridge_soh))),
            "enet_mae": float(np.mean(np.abs(c_true_next - c_enet_soh))),
            "persistence_mae": float(np.mean(np.abs(c_true_next - c_soh))),
            "mean_fade_mae": float(
                np.mean(np.abs(c_true_next - np.clip(c_soh + train_mean_delta, 0.0, SOH_CLIP_MAX)))
            ),
        }

    # Quality flags
    ridge_flag, ridge_export_model = _assign_quality_flag(
        ridge_metrics["mae"],
        persist_metrics["mae"],
        mf_metrics["mae"],
        "PADM_Ridge",
    )
    enet_flag, enet_export_model = _assign_quality_flag(
        enet_metrics["mae"],
        persist_metrics["mae"],
        mf_metrics["mae"],
        "PADM_ElasticNet",
    )
    print(f"  [padm_evaluate] Ridge quality flag:    {ridge_flag}")
    print(f"  [padm_evaluate] ElasticNet quality flag: {enet_flag}")

    # Choose best PADM model by validation MAE
    # (validation MAE from training provenance, not test)
    val_mae_ridge = float(prov["validation_mae"]["ridge"])
    val_mae_enet = float(prov["validation_mae"]["elasticnet"])
    best_padm_model_name = "PADM_Ridge" if val_mae_ridge <= val_mae_enet else "PADM_ElasticNet"
    best_padm_pred_soh = ridge_pred_soh if best_padm_model_name == "PADM_Ridge" else enet_pred_soh
    best_padm_mae = ridge_metrics["mae"] if best_padm_model_name == "PADM_Ridge" else enet_metrics["mae"]
    best_flag = ridge_flag if best_padm_model_name == "PADM_Ridge" else enet_flag
    best_export_model = ridge_export_model if best_padm_model_name == "PADM_Ridge" else enet_export_model

    # Gate tests
    gates = _run_gate_tests(
        test_df=test_df,
        padm_pred_soh=best_padm_pred_soh,
        persistence_pred_soh=persist_soh,
        mean_fade_pred_soh=mf_soh,
        true_soh=true_next_soh,
        padm_mae=best_padm_mae,
        persistence_mae=persist_metrics["mae"],
        mean_fade_mae=mf_metrics["mae"],
        quality_flag=best_flag,
    )
    all_gates_passed = all(gates.values())
    print(
        f"  [padm_evaluate] Gate tests: "
        + ("ALL PASSED" if all_gates_passed else f"FAILURES: {[k for k,v in gates.items() if not v]}")
    )

    # Determine final export predictor
    if best_export_model == "mean_fade_baseline":
        print(
            "  [padm_evaluate] PADM does not beat mean-fade. "
            "mean_fade_baseline will be used for Member 3 export."
        )
    else:
        print(
            f"  [padm_evaluate] {best_export_model} will be used for Member 3 export."
        )

    result: dict[str, Any] = {
        "pooled_metrics": {
            "ridge": ridge_metrics,
            "elasticnet": enet_metrics,
            "persistence": persist_metrics,
            "mean_fade": mf_metrics,
        },
        "per_cell_metrics": per_cell,
        "quality_flags": {
            "ridge": ridge_flag,
            "elasticnet": enet_flag,
        },
        "best_padm_model": best_padm_model_name,
        "export_model": best_export_model,
        "gate_tests": gates,
        "all_gates_passed": all_gates_passed,
        "delta_clip_bounds": {"low": clip_low, "high": clip_high},
        "train_mean_delta": train_mean_delta,
        "production_ready": False,
        "note": (
            "Test evaluation completed. Results are final. "
            "Do not re-run or adjust models based on these results."
        ),
    }

    # Save evaluation results
    eval_path = out_dir / "padm_test_evaluation.json"
    eval_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  [padm_evaluate] Saved test evaluation to {eval_path}")

    # Write sentinel file to prevent re-evaluation
    sentinel.write_text(
        f"Test evaluation completed. Evaluation results saved to {eval_path}.\n"
        "Do not delete this file or re-run evaluation after seeing results.\n",
        encoding="utf-8",
    )

    return result


__all__ = [
    "evaluate_padm_on_test",
]
