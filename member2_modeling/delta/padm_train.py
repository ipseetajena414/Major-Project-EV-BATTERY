"""
PADM Steps 3–8: Model Training
===============================

Implements the full training pipeline for the Persistence-Anchored Delta Model:

  Step 3  — Fit StandardScaler on training rows only.
  Step 4  — Compute train_mean_delta (scalar, negative expected).
  Step 5  — Feature set is locked in delta_features.PADM_FEATURE_COLS; no
             selection loop touches the test set.
  Step 6  — Hyperparameter selection via grid search on validation SOH MAE.
  Step 7  — LOCO (Leave-One-Cell-Out) diagnostic for overfitting assessment.
  Step 8  — Final model fitting on train + val with locked hyperparams.

All decisions follow PADM_PRE_IMPLEMENTATION_REVIEW.md.

Constraints enforced here
--------------------------
* Test rows are NEVER read inside this module.
* Scaler is always re-fitted from scratch within LOCO folds.
* Hyperparameters are chosen once on validation, then frozen.
* Models are saved to member2_modeling/artifacts/delta/.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from member2_modeling.delta.delta_features import PADM_FEATURE_COLS, SOH_CLIP_MAX

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "delta"

# ── Hyperparameter search grids (§6 of PADM review) ──────────────────────────
RIDGE_ALPHA_GRID: list[float] = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

ELASTICNET_GRID: list[tuple[float, float]] = [
    (alpha, l1_ratio)
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]
    for l1_ratio in [0.1, 0.5, 0.9]
]

RANDOM_STATE = 42


# ── Internal helpers ──────────────────────────────────────────────────────────


def _get_xy(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y_delta) arrays for model training/evaluation."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["delta_target"].values.astype(np.float32)
    return X, y


def _soh_mae_from_delta(
    current_soh: np.ndarray,
    true_delta: np.ndarray,
    predicted_delta: np.ndarray,
) -> float:
    """
    Compute MAE in SOH space from predicted deltas.

    predicted_SOH = clip(current_SOH + predicted_delta, 0, SOH_CLIP_MAX)
    true_SOH      = clip(current_SOH + true_delta, 0, SOH_CLIP_MAX)

    This is the single metric used for hyperparameter selection and reporting.
    """
    pred_soh = np.clip(current_soh + predicted_delta, 0.0, SOH_CLIP_MAX)
    true_soh = np.clip(current_soh + true_delta, 0.0, SOH_CLIP_MAX)
    return float(np.mean(np.abs(true_soh - pred_soh)))


def _fit_and_score_ridge(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val_delta: np.ndarray,
    current_soh_val: np.ndarray,
    alpha: float,
    scaler: StandardScaler,
) -> float:
    """Fit Ridge with given alpha, return validation SOH MAE."""
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    pred_delta = model.predict(X_val)
    return _soh_mae_from_delta(current_soh_val, y_val_delta, pred_delta)


def _fit_and_score_elasticnet(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val_delta: np.ndarray,
    current_soh_val: np.ndarray,
    alpha: float,
    l1_ratio: float,
) -> float:
    """Fit ElasticNet with given (alpha, l1_ratio), return validation SOH MAE."""
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=10_000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_tr, y_tr)
    pred_delta = model.predict(X_val)
    return _soh_mae_from_delta(current_soh_val, y_val_delta, pred_delta)


# ── Step 3: Scaler fitting ────────────────────────────────────────────────────


def fit_scaler_on_train(
    delta_df: pd.DataFrame,
    feature_cols: list[str],
) -> StandardScaler:
    """
    Fit StandardScaler exclusively on training rows.

    Parameters
    ----------
    delta_df : pd.DataFrame
        Full dataset (all splits) from ``delta_features.build_and_save_delta_dataset``.
    feature_cols : list[str]
        Locked feature column list.

    Returns
    -------
    Fitted StandardScaler instance.
    """
    train_df = delta_df[delta_df["split"] == "train"]
    X_train, _ = _get_xy(train_df, feature_cols)
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ── Step 4: Training mean delta ───────────────────────────────────────────────


def compute_train_mean_delta(delta_df: pd.DataFrame) -> float:
    """
    Compute mean delta_target from training rows only.

    This scalar is used by the mean-fade baseline predictor.
    It should be negative for degrading cells (SOH decreasing).
    """
    train_delta = delta_df.loc[delta_df["split"] == "train", "delta_target"]
    mean_delta = float(train_delta.mean())
    return mean_delta


# ── Step 6: Hyperparameter selection ──────────────────────────────────────────


def select_hyperparameters(
    delta_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
) -> dict[str, Any]:
    """
    Grid-search hyperparameters for Ridge and ElasticNet on validation SOH MAE.

    Uses the pre-fitted scaler (trained on training rows only) for both
    train and validation transforms.  Test rows are never accessed.

    Returns
    -------
    dict with keys:
        ridge_best_alpha : float
        ridge_val_mae    : float
        enet_best_alpha  : float
        enet_best_l1     : float
        enet_val_mae     : float
        ridge_grid_scores : list[dict]
        enet_grid_scores  : list[dict]
    """
    train_df = delta_df[delta_df["split"] == "train"]
    val_df = delta_df[delta_df["split"] == "val"]

    X_tr, y_tr = _get_xy(train_df, feature_cols)
    X_val, y_val = _get_xy(val_df, feature_cols)

    X_tr_sc = scaler.transform(X_tr)
    X_val_sc = scaler.transform(X_val)

    current_soh_val = val_df["current_soh_pct"].values.astype(np.float32)

    # Ridge grid search
    ridge_scores: list[dict[str, Any]] = []
    best_ridge_mae = float("inf")
    best_ridge_alpha = RIDGE_ALPHA_GRID[0]

    for alpha in RIDGE_ALPHA_GRID:
        mae = _fit_and_score_ridge(
            X_tr_sc, y_tr, X_val_sc, y_val, current_soh_val, alpha, scaler
        )
        ridge_scores.append({"alpha": alpha, "val_soh_mae": mae})
        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_ridge_alpha = alpha

    print(f"  [padm_train] Ridge best alpha={best_ridge_alpha}, val SOH MAE={best_ridge_mae:.4f}%")

    # ElasticNet grid search
    enet_scores: list[dict[str, Any]] = []
    best_enet_mae = float("inf")
    best_enet_alpha = ELASTICNET_GRID[0][0]
    best_enet_l1 = ELASTICNET_GRID[0][1]

    for alpha, l1_ratio in ELASTICNET_GRID:
        mae = _fit_and_score_elasticnet(
            X_tr_sc, y_tr, X_val_sc, y_val, current_soh_val, alpha, l1_ratio
        )
        enet_scores.append({"alpha": alpha, "l1_ratio": l1_ratio, "val_soh_mae": mae})
        if mae < best_enet_mae:
            best_enet_mae = mae
            best_enet_alpha = alpha
            best_enet_l1 = l1_ratio

    print(
        f"  [padm_train] ElasticNet best alpha={best_enet_alpha}, "
        f"l1_ratio={best_enet_l1}, val SOH MAE={best_enet_mae:.4f}%"
    )

    return {
        "ridge_best_alpha": best_ridge_alpha,
        "ridge_val_mae": best_ridge_mae,
        "enet_best_alpha": best_enet_alpha,
        "enet_best_l1": best_enet_l1,
        "enet_val_mae": best_enet_mae,
        "ridge_grid_scores": ridge_scores,
        "enet_grid_scores": enet_scores,
    }


# ── Step 7: LOCO diagnostic ───────────────────────────────────────────────────


def run_loco_diagnostic(
    delta_df: pd.DataFrame,
    feature_cols: list[str],
    ridge_alpha: float,
) -> dict[str, Any]:
    """
    Leave-One-Cell-Out cross-validation diagnostic.

    For each of the 3 NMC cells, train Ridge on the remaining 2 cells
    (train + val rows only) and evaluate on the held-out cell.

    Rules
    -----
    * Test split rows are NEVER used in this function.
    * Scaler is re-fitted from scratch on each fold's training data.
    * This diagnostic does NOT change the selected hyperparameters.

    Returns
    -------
    dict with per-cell MAE and pooled MAE.
    """
    # Exclude test rows entirely
    non_test_df = delta_df[delta_df["split"].isin(["train", "val"])].copy()
    cells = sorted(non_test_df["cell_id"].unique().tolist())

    loco_results: list[dict[str, Any]] = []

    for held_out_cell in cells:
        fold_train = non_test_df[non_test_df["cell_id"] != held_out_cell]
        fold_held = non_test_df[non_test_df["cell_id"] == held_out_cell]

        if len(fold_train) == 0 or len(fold_held) == 0:
            continue

        X_tr, y_tr = _get_xy(fold_train, feature_cols)
        X_hld, y_hld = _get_xy(fold_held, feature_cols)

        # Re-fit scaler on this fold's training data only
        fold_scaler = StandardScaler()
        fold_scaler.fit(X_tr)

        X_tr_sc = fold_scaler.transform(X_tr)
        X_hld_sc = fold_scaler.transform(X_hld)

        model = Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)
        model.fit(X_tr_sc, y_tr)

        pred_delta = model.predict(X_hld_sc)
        current_soh_hld = fold_held["current_soh_pct"].values.astype(np.float32)
        mae = _soh_mae_from_delta(current_soh_hld, y_hld, pred_delta)

        loco_results.append(
            {
                "held_out_cell": held_out_cell,
                "n_held_out": int(len(fold_held)),
                "n_train_cells": len(cells) - 1,
                "soh_mae": mae,
            }
        )
        print(
            f"  [padm_train] LOCO: held-out={held_out_cell}, "
            f"n={len(fold_held)}, SOH MAE={mae:.4f}%"
        )

    pooled_errors: list[float] = []
    for fold_train_cell in cells:
        fold_hld = non_test_df[non_test_df["cell_id"] == fold_train_cell]
        fold_tr = non_test_df[non_test_df["cell_id"] != fold_train_cell]
        if len(fold_tr) == 0 or len(fold_hld) == 0:
            continue
        X_tr, y_tr = _get_xy(fold_tr, feature_cols)
        X_hld, y_hld = _get_xy(fold_hld, feature_cols)
        fold_scaler = StandardScaler()
        fold_scaler.fit(X_tr)
        model = Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)
        model.fit(fold_scaler.transform(X_tr), y_tr)
        pred_delta = model.predict(fold_scaler.transform(X_hld))
        current_soh = fold_hld["current_soh_pct"].values.astype(np.float32)
        errors = np.abs(
            np.clip(current_soh + pred_delta, 0.0, SOH_CLIP_MAX)
            - np.clip(current_soh + y_hld, 0.0, SOH_CLIP_MAX)
        )
        pooled_errors.extend(errors.tolist())

    pooled_mae = float(np.mean(pooled_errors)) if pooled_errors else float("nan")
    print(f"  [padm_train] LOCO pooled SOH MAE={pooled_mae:.4f}%")

    return {
        "per_cell": loco_results,
        "pooled_soh_mae": pooled_mae,
        "note": (
            "LOCO is diagnostic only. "
            "Hyperparameters are NOT changed based on LOCO results."
        ),
    }


# ── Step 8: Final model fitting ───────────────────────────────────────────────


def fit_final_models(
    delta_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    ridge_alpha: float,
    enet_alpha: float,
    enet_l1_ratio: float,
) -> dict[str, Any]:
    """
    Fit Ridge and ElasticNet on the combined train + val split.

    Scaler is re-fitted on train + val combined.
    Test rows are NOT used.

    Returns
    -------
    dict with:
        ridge           : fitted Ridge model
        elasticnet      : fitted ElasticNet model
        scaler          : fitted StandardScaler (on train + val)
        train_mean_delta: scalar float
    """
    trainval_df = delta_df[delta_df["split"].isin(["train", "val"])].copy()
    X_tv, y_tv = _get_xy(trainval_df, feature_cols)

    # Re-fit scaler on train + val
    final_scaler = StandardScaler()
    final_scaler.fit(X_tv)
    X_tv_sc = final_scaler.transform(X_tv)

    ridge = Ridge(alpha=ridge_alpha, random_state=RANDOM_STATE)
    ridge.fit(X_tv_sc, y_tv)

    enet = ElasticNet(
        alpha=enet_alpha,
        l1_ratio=enet_l1_ratio,
        max_iter=10_000,
        random_state=RANDOM_STATE,
    )
    enet.fit(X_tv_sc, y_tv)

    train_mean_delta = compute_train_mean_delta(
        delta_df[delta_df["split"] == "train"]
    )
    print(f"  [padm_train] train_mean_delta={train_mean_delta:.6f}%")

    return {
        "ridge": ridge,
        "elasticnet": enet,
        "scaler": final_scaler,
        "train_mean_delta": train_mean_delta,
    }


# ── Full pipeline ──────────────────────────────────────────────────────────────


def train_padm(
    delta_df: pd.DataFrame,
    *,
    feature_cols: list[str] | None = None,
    artifact_dir: Path | str | None = None,
) -> dict[str, Any]:
    """
    Run the full PADM training pipeline (Steps 3–8).

    Parameters
    ----------
    delta_df : pd.DataFrame
        Output of ``delta_features.build_and_save_delta_dataset`` — must
        already have lag NaNs imputed and a ``split`` column.
    feature_cols : list[str] or None
        Uses PADM_FEATURE_COLS if None.
    artifact_dir : Path or str, optional
        Directory to save model artifacts.

    Returns
    -------
    dict with all training metadata and artifact paths.
    """
    if feature_cols is None:
        feature_cols = PADM_FEATURE_COLS

    out_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n  [padm_train] Step 3 — Fitting scaler on training rows...")
    scaler = fit_scaler_on_train(delta_df, feature_cols)
    joblib.dump(scaler, out_dir / "padm_scaler.joblib")

    print("\n  [padm_train] Step 4 — Computing train mean delta...")
    train_mean_delta = compute_train_mean_delta(delta_df)
    print(f"  [padm_train] train_mean_delta={train_mean_delta:.6f}%")
    if train_mean_delta > 0:
        print(
            "  [padm_train] WARNING: train_mean_delta is positive. "
            "Expected negative for degrading cells. Check data."
        )

    print("\n  [padm_train] Step 6 — Hyperparameter selection on validation SOH MAE...")
    hp = select_hyperparameters(delta_df, feature_cols, scaler)

    print("\n  [padm_train] Step 7 — LOCO diagnostic (diagnostic only)...")
    loco = run_loco_diagnostic(delta_df, feature_cols, hp["ridge_best_alpha"])

    print("\n  [padm_train] Step 8 — Final model fit on train + val...")
    final = fit_final_models(
        delta_df,
        feature_cols,
        ridge_alpha=hp["ridge_best_alpha"],
        enet_alpha=hp["enet_best_alpha"],
        enet_l1_ratio=hp["enet_best_l1"],
    )

    # Save model artifacts
    joblib.dump(final["ridge"], out_dir / "padm_ridge.joblib")
    joblib.dump(final["elasticnet"], out_dir / "padm_elasticnet.joblib")
    joblib.dump(final["scaler"], out_dir / "padm_scaler_final.joblib")
    print(f"  [padm_train] Models saved to {out_dir}")

    # Save training provenance
    provenance: dict[str, Any] = {
        "module": "member2_modeling/delta/padm_train",
        "feature_cols": feature_cols,
        "hyperparameters": {
            "ridge_alpha": hp["ridge_best_alpha"],
            "enet_alpha": hp["enet_best_alpha"],
            "enet_l1_ratio": hp["enet_best_l1"],
        },
        "validation_mae": {
            "ridge": hp["ridge_val_mae"],
            "elasticnet": hp["enet_val_mae"],
        },
        "train_mean_delta": final["train_mean_delta"],
        "loco_diagnostic": loco,
        "ridge_grid_scores": hp["ridge_grid_scores"],
        "enet_grid_scores": hp["enet_grid_scores"],
        "production_ready": False,
        "notes": [
            "Test set was not accessed during training or hyperparameter selection.",
            "Scaler re-fitted on train+val for final models.",
            "LOCO diagnostic is informational only; hyperparameters not changed.",
        ],
    }
    (out_dir / "padm_train_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    return {
        "ridge": final["ridge"],
        "elasticnet": final["elasticnet"],
        "scaler": final["scaler"],
        "train_mean_delta": final["train_mean_delta"],
        "hyperparameters": provenance["hyperparameters"],
        "validation_mae": provenance["validation_mae"],
        "loco_diagnostic": loco,
        "artifact_dir": str(out_dir),
    }


__all__ = [
    "fit_scaler_on_train",
    "compute_train_mean_delta",
    "select_hyperparameters",
    "run_loco_diagnostic",
    "fit_final_models",
    "train_padm",
]
