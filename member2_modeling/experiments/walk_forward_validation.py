"""Chronological walk-forward validation for Member 2 experiments.

Two schemes:
  - LFP B5: single-cell expanding-window validation of the unit-safe Ridge model.
  - NMC: pooled per-cell expanding-window validation of persistence, mean-drift,
    Ridge, existing LightGBM and delta-target variants.

All preprocessing, alpha selection and early stopping use only the training
window of each fold. No production artifact is read for inference or written.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from member2_modeling import build_and_train as bt  # noqa: E402
import experimental_features as ef  # noqa: E402

warnings.filterwarnings("ignore")

PARQUET = ROOT / "master_table.parquet"
SCHEMA = ROOT / "member2_modeling" / "feature_schema.json"
RESULTS_DIR = EXPERIMENT_DIR / "results"

N_FOLDS = 5
LFP_INITIAL_FRAC = 0.40
NMC_INITIAL_FRAC = 0.50
INNER_VAL_FRAC = 0.20
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LGBM_SEED = 42


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(obj):
    """Convert numpy/pandas scalars and NaN to strict-JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


# ── Data assembly ────────────────────────────────────────────────────────────


def build_cycle_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(PARQUET)
    raw = raw[raw["dataset_id"] != "LFP_MultiStat_B1"].copy()
    cycles = bt.build_cycle_feature_table(raw)
    cycles = cycles.dropna(subset=["soh_pct"]).copy()
    cycles = cycles.sort_values(["cell_id", "cycle_id"], kind="mergesort").reset_index(drop=True)
    cycles["cycle_position"] = cycles.groupby("cell_id").cumcount()

    extras = ef.compute_cycle_extra_features(
        raw[raw["chemistry"].isin(["NMC", "LFP"])].copy()
    )
    cycles = ef.merge_extra_features(cycles, extras)
    cycles = ef.add_cumulative_features(cycles)
    cycles = cycles.sort_values(["cell_id", "cycle_id"], kind="mergesort").reset_index(drop=True)
    return raw, cycles


def make_pairs(cycles: pd.DataFrame) -> pd.DataFrame:
    """Input features at observed record t, target SOH at the next observed record."""
    frames = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_id", kind="mergesort").reset_index(drop=True)
        if len(group) < 2:
            continue
        current = group.iloc[:-1].copy().reset_index(drop=True)
        nxt = group.iloc[1:].copy().reset_index(drop=True)

        current["target_soh_pct"] = nxt["soh_pct"].to_numpy()
        current["target_cycle_id"] = nxt["cycle_id"].to_numpy()
        current["target_cycle_position"] = nxt["cycle_position"].to_numpy()
        current["current_soh_pct"] = current["soh_pct"].to_numpy()
        current["prev_soh_pct"] = current["soh_pct"].shift(1)
        current["pair_index"] = np.arange(len(current), dtype=int)
        current["target_record_position"] = current["pair_index"] + 1
        current["cycle_gap"] = (current["target_cycle_id"] - current["cycle_id"]).astype(int)
        current["is_gap_pair"] = current["cycle_gap"] > 1
        frames.append(current)
    return pd.concat(frames, ignore_index=True)


# ── Fold construction ────────────────────────────────────────────────────────


def expanding_boundaries(n: int, initial_frac: float, n_folds: int) -> list[int]:
    if n < n_folds + 2:
        raise ValueError(f"Not enough records for {n_folds} folds: n={n}")
    initial = max(1, min(int(np.floor(n * initial_frac)), n - 2))
    remaining = n - initial
    raw = [initial]
    for k in range(1, n_folds + 1):
        raw.append(initial + int(round(remaining * k / n_folds)))
    raw[-1] = n

    cuts = [raw[0]]
    for value in raw[1:]:
        if value > cuts[-1]:
            cuts.append(value)
    if cuts[-1] != n:
        cuts.append(n)
    if len(cuts) != n_folds + 1:
        raise ValueError(f"Fold construction produced {len(cuts) - 1} folds, expected {n_folds}")
    return cuts


def make_single_cell_folds(
    pairs: pd.DataFrame,
    initial_frac: float = LFP_INITIAL_FRAC,
    n_folds: int = N_FOLDS,
) -> tuple[list[dict], list[int]]:
    if pairs["cell_id"].nunique() != 1:
        raise ValueError("Single-cell fold construction requires exactly one cell")
    ordered = pairs.sort_values("pair_index").reset_index(drop=True)
    cuts = expanding_boundaries(len(ordered), initial_frac, n_folds)
    folds = []
    for fold_id in range(n_folds):
        folds.append(
            {
                "fold": fold_id + 1,
                "train": ordered.iloc[: cuts[fold_id]].copy(),
                "test": ordered.iloc[cuts[fold_id] : cuts[fold_id + 1]].copy(),
            }
        )
    return folds, cuts


def make_pooled_cell_folds(
    pairs: pd.DataFrame,
    initial_frac: float = NMC_INITIAL_FRAC,
    n_folds: int = N_FOLDS,
) -> tuple[list[dict], dict[str, list[int]]]:
    cuts_by_cell = {}
    for cell_id, group in pairs.groupby("cell_id", sort=True):
        cuts_by_cell[str(cell_id)] = expanding_boundaries(len(group), initial_frac, n_folds)

    folds = []
    for fold_id in range(n_folds):
        train_parts, test_parts, fold_cells = [], [], []
        for cell_id in sorted(pairs["cell_id"].unique()):
            group = (
                pairs[pairs["cell_id"] == cell_id]
                .sort_values("pair_index")
                .reset_index(drop=True)
            )
            cuts = cuts_by_cell[str(cell_id)]
            train = group.iloc[: cuts[fold_id]].copy()
            test = group.iloc[cuts[fold_id] : cuts[fold_id + 1]].copy()
            train_parts.append(train)
            test_parts.append(test)
            fold_cells.append(
                {
                    "cell_id": str(cell_id),
                    "train_n": int(len(train)),
                    "test_n": int(len(test)),
                    "train_cycle_min": int(train["cycle_id"].min()),
                    "train_cycle_max": int(train["cycle_id"].max()),
                    "test_cycle_min": int(test["cycle_id"].min()),
                    "test_cycle_max": int(test["cycle_id"].max()),
                    "train_pair_max": int(train["pair_index"].max()),
                    "test_pair_min": int(test["pair_index"].min()),
                }
            )
        train = pd.concat(train_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        folds.append(
            {
                "fold": fold_id + 1,
                "train": train.sort_values(["cell_id", "pair_index"]).reset_index(drop=True),
                "test": test.sort_values(["cell_id", "pair_index"]).reset_index(drop=True),
                "cells": fold_cells,
            }
        )
    return folds, cuts_by_cell


def inner_split(
    train: pd.DataFrame, val_frac: float = INNER_VAL_FRAC, min_val: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological per-cell inner split: last val_frac of each training window."""
    inner_train, inner_val = [], []
    for cell_id, group in train.groupby("cell_id", sort=True):
        group = group.sort_values("pair_index").reset_index(drop=True)
        n = len(group)
        n_val = max(min_val, int(round(n * val_frac)))
        n_val = min(n_val, max(1, n - 5))
        if n_val >= n:
            n_val = max(1, n // 3)
        inner_train.append(group.iloc[: n - n_val])
        inner_val.append(group.iloc[n - n_val :])
    train_out = pd.concat(inner_train, ignore_index=True)
    val_out = pd.concat(inner_val, ignore_index=True)
    return (
        train_out.sort_values(["cell_id", "pair_index"]).reset_index(drop=True),
        val_out.sort_values(["cell_id", "pair_index"]).reset_index(drop=True),
    )


# ── Models and baselines ─────────────────────────────────────────────────────


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[cols].astype(float).replace([np.inf, -np.inf], np.nan)


def fit_ridge(
    train_full: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    alphas: list[float] = ALPHAS,
) -> tuple[np.ndarray, dict]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_inner = feature_matrix(inner_train, features)
    y_inner = inner_train["target_soh_pct"].to_numpy(dtype=float)
    X_val = feature_matrix(inner_val, features)
    y_val = inner_val["target_soh_pct"].to_numpy(dtype=float)

    best = None
    for alpha in alphas:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        pipe.fit(X_inner, y_inner)
        mae = float(np.mean(np.abs(y_val - pipe.predict(X_val))))
        if best is None or mae < best[0]:
            best = (mae, alpha)
    if best is None:
        raise RuntimeError("Ridge alpha search produced no candidate")

    final = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=best[1])),
        ]
    )
    final.fit(
        feature_matrix(train_full, features),
        train_full["target_soh_pct"].to_numpy(dtype=float),
    )
    pred = final.predict(feature_matrix(test, features))
    info = {
        "ridge_alpha": float(best[1]),
        "inner_val_mae": float(best[0]),
        "n_inner_train": int(len(inner_train)),
        "n_inner_val": int(len(inner_val)),
        "n_train_full": int(len(train_full)),
    }
    return np.asarray(pred, dtype=float), info


def _lgbm_params(n_estimators: int = 500) -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": LGBM_SEED,
        "n_estimators": n_estimators,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
    }


def fit_lgbm(
    train_full: pd.DataFrame,
    inner_train: pd.DataFrame,
    inner_val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target_mode: str = "absolute",
    task_name: str = "",
) -> tuple[np.ndarray, dict]:
    """Train the production LightGBM configuration with train-only early stopping."""
    import lightgbm as lgb

    def target(frame):
        y = frame["target_soh_pct"].to_numpy(dtype=float)
        if target_mode == "delta":
            y = y - frame["current_soh_pct"].to_numpy(dtype=float)
        return y

    X_inner = feature_matrix(inner_train, features)
    X_val = feature_matrix(inner_val, features)
    y_inner = target(inner_train)
    y_val = target(inner_val)

    model = lgb.LGBMRegressor(**_lgbm_params())
    model.fit(
        X_inner,
        y_inner,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    best_iteration = int(model.best_iteration_ or 500)
    final = lgb.LGBMRegressor(**_lgbm_params(n_estimators=max(1, best_iteration)))
    final.fit(feature_matrix(train_full, features), target(train_full))
    pred = final.predict(feature_matrix(test, features))
    if target_mode == "delta":
        pred = pred + test["current_soh_pct"].to_numpy(dtype=float)
    info = {
        "best_iteration": best_iteration,
        "n_estimators_refit": max(1, best_iteration),
        "n_inner_train": int(len(inner_train)),
        "n_inner_val": int(len(inner_val)),
        "n_train_full": int(len(train_full)),
        "target_mode": target_mode,
        "task_name": task_name,
    }
    return np.asarray(pred, dtype=float), info


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    if model == "persistence":
        return test["current_soh_pct"].to_numpy(dtype=float)

    if model == "pooled_mean_drift":
        drift = float(
            (train["target_soh_pct"] - train["current_soh_pct"]).mean()
        )
        return test["current_soh_pct"].to_numpy(dtype=float) + drift

    if model == "cell_mean_drift":
        delta = train.assign(
            _delta=train["target_soh_pct"] - train["current_soh_pct"]
        )
        per_cell = delta.groupby("cell_id")["_delta"].mean()
        fallback = float(delta["_delta"].mean())
        drift = test["cell_id"].map(per_cell).fillna(fallback).to_numpy(dtype=float)
        return test["current_soh_pct"].to_numpy(dtype=float) + drift

    if model == "last_delta":
        delta = (test["soh_pct"] - test["prev_soh_pct"]).to_numpy(dtype=float)
        delta = np.where(np.isfinite(delta), delta, 0.0)
        return test["current_soh_pct"].to_numpy(dtype=float) + delta

    if model == "train_mean":
        return np.full(len(test), float(train["target_soh_pct"].mean()))

    if model == "linear_trend":
        def fit(frame):
            if len(frame) < 3:
                return None
            x = frame["target_record_position"].to_numpy(dtype=float)
            y = frame["target_soh_pct"].to_numpy(dtype=float)
            if np.ptp(x) <= 0:
                return None
            return tuple(np.polyfit(x, y, 1))

        global_fit = fit(train)
        if global_fit is None:
            global_fit = (0.0, float(train["target_soh_pct"].mean()))
        pred = pd.Series(index=test.index, dtype=float)
        for cell_id, group in test.groupby("cell_id"):
            local_fit = fit(train[train["cell_id"] == cell_id]) or global_fit
            pred.loc[group.index] = (
                local_fit[0] * group["target_record_position"] + local_fit[1]
            )
        return pred.to_numpy(dtype=float)

    raise ValueError(f"Unknown baseline: {model}")


# ── Evaluation helpers ───────────────────────────────────────────────────────


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    met = bt.evaluate_predictions(y_true, y_pred, "")
    if met["n"] < 2:
        met["r2"] = None
    return met


def evaluate_rows(
    eval_frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    chemistry: str,
    fold_id: int | str,
    scope: str,
) -> tuple[list[dict], list[dict]]:
    """Return (overall rows, per-cell rows) for one fold/model batch."""
    overall, per_cell = [], []
    for model, pred in predictions.items():
        frame = eval_frame.copy()
        frame["prediction"] = pred
        met = _metrics(frame["target_soh_pct"], frame["prediction"])
        persistence = _metrics(
            frame["target_soh_pct"], frame["current_soh_pct"]
        )
        overall.append(
            {
                "chemistry": chemistry,
                "fold": fold_id,
                "scope": scope,
                "cell_id": "ALL",
                "model": model,
                **{k: met[k] for k in ("n", "mae", "rmse", "r2", "mape_pct")},
                "persistence_mae_same_rows": persistence["mae"],
                "mae_delta_vs_persistence": (
                    met["mae"] - persistence["mae"]
                    if met["mae"] is not None and persistence["mae"] is not None
                    else None
                ),
                "beats_persistence": bool(
                    met["mae"] is not None
                    and persistence["mae"] is not None
                    and met["mae"] < persistence["mae"]
                ),
            }
        )
        for cell_id, group in frame.groupby("cell_id"):
            cell_met = _metrics(group["target_soh_pct"], group["prediction"])
            cell_pers = _metrics(group["target_soh_pct"], group["current_soh_pct"])
            per_cell.append(
                {
                    "chemistry": chemistry,
                    "fold": fold_id,
                    "scope": scope,
                    "cell_id": str(cell_id),
                    "model": model,
                    **{k: cell_met[k] for k in ("n", "mae", "rmse", "r2", "mape_pct")},
                    "persistence_mae_same_rows": cell_pers["mae"],
                    "mae_delta_vs_persistence": (
                        cell_met["mae"] - cell_pers["mae"]
                        if cell_met["mae"] is not None and cell_pers["mae"] is not None
                        else None
                    ),
                    "beats_persistence": bool(
                        cell_met["mae"] is not None
                        and cell_pers["mae"] is not None
                        and cell_met["mae"] < cell_pers["mae"]
                    ),
                }
            )
    return overall, per_cell


def _prediction_frame(eval_frame: pd.DataFrame, model: str, pred, fold_id, chemistry):
    frame = eval_frame[
        [
            "cell_id",
            "pair_index",
            "cycle_id",
            "target_cycle_id",
            "cycle_gap",
            "is_gap_pair",
            "current_soh_pct",
            "target_soh_pct",
        ]
    ].copy()
    frame["chemistry"] = chemistry
    frame["fold"] = fold_id
    frame["model"] = model
    frame["prediction"] = np.asarray(pred, dtype=float)
    frame["abs_error"] = (frame["target_soh_pct"] - frame["prediction"]).abs()
    return frame


# ── LFP walk-forward ─────────────────────────────────────────────────────────


def run_lfp(pairs: pd.DataFrame, features: list[str]) -> dict:
    lfp = pairs[pairs["chemistry"] == "LFP"].copy()
    if lfp["cell_id"].nunique() != 1:
        raise ValueError("LFP analysis expects exactly one cell")
    folds, cuts = make_single_cell_folds(lfp)

    fold_rows, cell_rows, predictions = [], [], []
    fold_meta = []
    baseline_models = ["persistence", "pooled_mean_drift", "last_delta", "linear_trend", "train_mean"]

    for fold in folds:
        fold_id = fold["fold"]
        train, test = fold["train"], fold["test"]
        test = test.reset_index(drop=True)
        inner_train, inner_val = inner_split(train)

        preds = {m: baseline_predict(train, test, m) for m in baseline_models}
        ridge_pred, ridge_info = fit_ridge(
            train, inner_train, inner_val, test, features
        )
        preds["ridge_unit_safe"] = ridge_pred

        overall, per_cell = evaluate_rows(
            test, preds, "LFP", fold_id, scope="fold"
        )
        fold_rows.extend(overall)
        cell_rows.extend(per_cell)
        for model, pred in preds.items():
            predictions.append(_prediction_frame(test, model, pred, fold_id, "LFP"))

        n_gap = int(test["is_gap_pair"].sum())
        fold_meta.append(
            {
                "fold": fold_id,
                "train_start_pair": 0,
                "train_end_pair": int(cuts[fold_id - 1] - 1),
                "test_start_pair": int(cuts[fold_id - 1]),
                "test_end_pair": int(cuts[fold_id] - 1),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_cycle_min": int(train["cycle_id"].min()),
                "train_cycle_max": int(train["cycle_id"].max()),
                "test_cycle_min": int(test["cycle_id"].min()),
                "test_cycle_max": int(test["cycle_id"].max()),
                "n_train_gap_pairs": int(train["is_gap_pair"].sum()),
                "n_test_gap_pairs": n_gap,
                "test_gap_pairs": [
                    [int(r.cycle_id), int(r.target_cycle_id)]
                    for r in test[test["is_gap_pair"]].itertuples()
                ],
                "ridge_alpha": ridge_info["ridge_alpha"],
                "ridge_inner_val_mae": ridge_info["inner_val_mae"],
                "n_inner_train": ridge_info["n_inner_train"],
                "n_inner_val": ridge_info["n_inner_val"],
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    pooled_rows = []
    for model, group in predictions_df.groupby("model"):
        met = _metrics(group["target_soh_pct"], group["prediction"])
        pers = _metrics(group["target_soh_pct"], group["current_soh_pct"])
        adjacent = group[~group["is_gap_pair"]]
        adj_met = _metrics(adjacent["target_soh_pct"], adjacent["prediction"])
        pooled_rows.append(
            {
                "chemistry": "LFP",
                "fold": "pooled",
                "scope": "pooled_all",
                "cell_id": "ALL",
                "model": model,
                **{k: met[k] for k in ("n", "mae", "rmse", "r2", "mape_pct")},
                "n_adjacent": int(len(adjacent)),
                "mae_adjacent": adj_met["mae"],
                "rmse_adjacent": adj_met["rmse"],
                "r2_adjacent": adj_met["r2"],
                "persistence_mae_same_rows": pers["mae"],
                "mae_delta_vs_persistence": (
                    met["mae"] - pers["mae"]
                    if met["mae"] is not None and pers["mae"] is not None
                    else None
                ),
                "beats_persistence": bool(
                    met["mae"] is not None
                    and pers["mae"] is not None
                    and met["mae"] < pers["mae"]
                ),
            }
        )

    gap_pairs = (
        lfp[lfp["is_gap_pair"]]
        .sort_values("pair_index")[
            [
                "cell_id",
                "pair_index",
                "cycle_id",
                "target_cycle_id",
                "cycle_gap",
                "current_soh_pct",
                "target_soh_pct",
            ]
        ]
        .copy()
    )
    gap_pairs["delta_soh"] = gap_pairs["target_soh_pct"] - gap_pairs["current_soh_pct"]
    fold_for_pair = {}
    for meta in fold_meta:
        lo, hi = meta["test_start_pair"], meta["test_end_pair"]
        for p in range(lo, hi + 1):
            fold_for_pair[p] = meta["fold"]
    gap_pairs["evaluated_in_fold"] = gap_pairs["pair_index"].map(fold_for_pair)
    gap_pairs["evaluation_status"] = gap_pairs["evaluated_in_fold"].apply(
        lambda x: "tested" if pd.notna(x) else "train_only"
    )

    # Per-gap-pair model errors for pairs that fall in a test window.
    gap_detail = []
    for row in gap_pairs.itertuples():
        if pd.isna(row.evaluated_in_fold):
            continue
        subset = predictions_df[
            (predictions_df["fold"] == int(row.evaluated_in_fold))
            & (predictions_df["pair_index"] == int(row.pair_index))
        ]
        for r in subset.itertuples():
            gap_detail.append(
                {
                    "pair_index": int(row.pair_index),
                    "current_cycle_id": int(row.cycle_id),
                    "target_cycle_id": int(row.target_cycle_id),
                    "fold": int(row.evaluated_in_fold),
                    "model": r.model,
                    "current_soh": float(row.current_soh_pct),
                    "target_soh": float(row.target_soh_pct),
                    "delta_soh": float(row.target_soh_pct - row.current_soh_pct),
                    "prediction": float(r.prediction),
                    "abs_error": float(abs(row.target_soh_pct - r.prediction)),
                }
            )

    fold_metrics_df = pd.DataFrame(fold_rows)
    cell_metrics_df = pd.DataFrame(cell_rows)
    pooled_df = pd.DataFrame(pooled_rows)
    metrics_df = pd.concat([fold_metrics_df, pooled_df], ignore_index=True)

    # Consistency: how often does Ridge beat persistence by fold?
    ridge_fold = fold_metrics_df[fold_metrics_df["model"] == "ridge_unit_safe"]
    consistency = {
        "folds": int(len(ridge_fold)),
        "ridge_beats_persistence_folds": int(ridge_fold["beats_persistence"].sum()),
        "ridge_worse_than_persistence_folds": int((~ridge_fold["beats_persistence"]).sum()),
        "fold_mae": {
            str(int(r.fold)): {
                "ridge": float(r.mae),
                "persistence": float(r.persistence_mae_same_rows),
            }
            for r in ridge_fold.itertuples()
        },
    }

    fold_meta_df = pd.DataFrame(fold_meta)
    fold_meta_df["test_gap_pairs"] = fold_meta_df["test_gap_pairs"].apply(
        lambda x: json.dumps(x)
    )
    gap_detail_df = pd.DataFrame(gap_detail)
    return {
        "fold_metrics": fold_metrics_df,
        "cell_metrics": cell_metrics_df,
        "pooled_metrics": pooled_df,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "folds": fold_meta_df,
        "gap_pairs": gap_pairs,
        "gap_detail": gap_detail_df,
        "consistency": consistency,
        "cuts": cuts,
        "n_pairs": int(len(lfp)),
    }


# ── NMC walk-forward ─────────────────────────────────────────────────────────


def run_nmc(pairs: pd.DataFrame, prod_features: list[str], efc_features: list[str]) -> dict:
    nmc = pairs[pairs["chemistry"] == "NMC"].copy()
    folds, cuts_by_cell = make_pooled_cell_folds(nmc)

    fold_rows, cell_rows, predictions = [], [], []
    fold_meta = []
    model_meta = []
    baseline_models = ["persistence", "pooled_mean_drift", "cell_mean_drift", "linear_trend"]

    for fold in folds:
        fold_id = fold["fold"]
        train, test = fold["train"], fold["test"]
        test = test.reset_index(drop=True)
        inner_train, inner_val = inner_split(train)

        preds = {m: baseline_predict(train, test, m) for m in baseline_models}

        ridge_pred, ridge_info = fit_ridge(
            train, inner_train, inner_val, test, prod_features
        )
        preds["ridge_prod"] = ridge_pred

        for model_name, target_mode, feature_set, task in (
            ("lightgbm_prod", "absolute", prod_features, f"NMC_f{fold_id}_prod"),
            ("delta_lightgbm_prod", "delta", prod_features, f"NMC_f{fold_id}_delta_prod"),
            ("delta_lightgbm_efc", "delta", efc_features, f"NMC_f{fold_id}_delta_efc"),
        ):
            pred, info = fit_lgbm(
                train,
                inner_train,
                inner_val,
                test,
                feature_set,
                target_mode=target_mode,
                task_name=task,
            )
            preds[model_name] = pred
            model_meta.append({"fold": fold_id, "model": model_name, **info})

        overall, per_cell = evaluate_rows(
            test, preds, "NMC", fold_id, scope="fold"
        )
        fold_rows.extend(overall)
        cell_rows.extend(per_cell)
        for model, pred in preds.items():
            predictions.append(_prediction_frame(test, model, pred, fold_id, "NMC"))

        fold_meta.append(
            {
                "fold": fold_id,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "ridge_alpha": ridge_info["ridge_alpha"],
                "ridge_inner_val_mae": ridge_info["inner_val_mae"],
                "n_inner_train": ridge_info["n_inner_train"],
                "n_inner_val": ridge_info["n_inner_val"],
                "cells": fold["cells"],
            }
        )

    predictions_df = pd.concat(predictions, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_rows)
    cell_metrics_df = pd.DataFrame(cell_rows)
    model_meta_df = pd.DataFrame(model_meta)

    pooled_rows = []
    for model, group in predictions_df.groupby("model"):
        met = _metrics(group["target_soh_pct"], group["prediction"])
        pers = _metrics(group["target_soh_pct"], group["current_soh_pct"])
        pooled_rows.append(
            {
                "chemistry": "NMC",
                "fold": "pooled",
                "scope": "pooled_cells",
                "cell_id": "ALL",
                "model": model,
                **{k: met[k] for k in ("n", "mae", "rmse", "r2", "mape_pct")},
                "persistence_mae_same_rows": pers["mae"],
                "mae_delta_vs_persistence": (
                    met["mae"] - pers["mae"]
                    if met["mae"] is not None and pers["mae"] is not None
                    else None
                ),
                "beats_persistence": bool(
                    met["mae"] is not None
                    and pers["mae"] is not None
                    and met["mae"] < pers["mae"]
                ),
            }
        )
        for cell_id, cell_group in group.groupby("cell_id"):
            cell_met = _metrics(cell_group["target_soh_pct"], cell_group["prediction"])
            cell_pers = _metrics(
                cell_group["target_soh_pct"], cell_group["current_soh_pct"]
            )
            pooled_rows.append(
                {
                    "chemistry": "NMC",
                    "fold": "pooled",
                    "scope": "pooled_cell",
                    "cell_id": str(cell_id),
                    "model": model,
                    **{
                        k: cell_met[k]
                        for k in ("n", "mae", "rmse", "r2", "mape_pct")
                    },
                    "persistence_mae_same_rows": cell_pers["mae"],
                    "mae_delta_vs_persistence": (
                        cell_met["mae"] - cell_pers["mae"]
                        if cell_met["mae"] is not None
                        and cell_pers["mae"] is not None
                        else None
                    ),
                    "beats_persistence": bool(
                        cell_met["mae"] is not None
                        and cell_pers["mae"] is not None
                        and cell_met["mae"] < cell_pers["mae"]
                    ),
                }
            )

    pooled_df = pd.DataFrame(pooled_rows)
    metrics_df = pd.concat([fold_metrics_df, pooled_df], ignore_index=True)

    # Consistency of the mean-drift improvement.
    drift_cells = cell_metrics_df[
        cell_metrics_df["model"].isin(["persistence", "pooled_mean_drift", "cell_mean_drift"])
    ]
    pivot = drift_cells.pivot_table(
        index=["fold", "cell_id"],
        columns="model",
        values="mae",
        aggfunc="first",
    ).reset_index()
    pivot["pooled_drift_gain"] = pivot["persistence"] - pivot["pooled_mean_drift"]
    pivot["cell_drift_gain"] = pivot["persistence"] - pivot["cell_mean_drift"]
    pivot["pooled_drift_beats"] = pivot["pooled_drift_gain"] > 0
    pivot["cell_drift_beats"] = pivot["cell_drift_gain"] > 0

    per_cell_summary = (
        pooled_df[(pooled_df["scope"] == "pooled_cell")
                  & (pooled_df["model"].isin(
                      ["persistence", "pooled_mean_drift", "cell_mean_drift",
                       "ridge_prod", "lightgbm_prod", "delta_lightgbm_prod",
                       "delta_lightgbm_efc"]
                  ))]
        .pivot_table(index="cell_id", columns="model", values="mae", aggfunc="first")
        .reset_index()
    )
    for baseline in ("persistence",):
        for model in ("pooled_mean_drift", "cell_mean_drift", "delta_lightgbm_prod"):
            if baseline in per_cell_summary and model in per_cell_summary:
                per_cell_summary[f"{model}_gain_pct"] = (
                    100.0
                    * (per_cell_summary[baseline] - per_cell_summary[model])
                    / per_cell_summary[baseline]
                )

    consistency = {
        "fold_cell_units": int(len(pivot)),
        "pooled_drift_beats_persistence_units": int(pivot["pooled_drift_beats"].sum()),
        "cell_drift_beats_persistence_units": int(pivot["cell_drift_beats"].sum()),
        "fold_cell_detail": pivot.to_dict(orient="records"),
        "per_cell_pooled_mae": per_cell_summary.to_dict(orient="records"),
    }

    fold_meta_df = pd.DataFrame(fold_meta)
    fold_meta_df["cells"] = fold_meta_df["cells"].apply(lambda x: json.dumps(x))
    return {
        "fold_metrics": fold_metrics_df,
        "cell_metrics": cell_metrics_df,
        "pooled_metrics": pooled_df,
        "metrics": metrics_df,
        "predictions": predictions_df,
        "folds": fold_meta_df,
        "model_meta": model_meta_df,
        "consistency": consistency,
        "cuts_by_cell": cuts_by_cell,
        "n_pairs": int(len(nmc)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    nmc_features = list(schema["models"]["nmc_lgbm"]["feature_names"])
    lfp_features = list(schema["models"]["lfp_lgbm"]["feature_names"])
    lfp_safe = ef.unit_safe_lfp_features(lfp_features)
    if len(lfp_safe) != 22:
        raise RuntimeError(f"Expected 22 unit-safe LFP features, got {len(lfp_safe)}")
    if set(lfp_safe) & set(ef.CURRENT_DERIVED_FEATURES):
        raise RuntimeError("Unit-safe LFP feature list still contains current-derived features")

    print("[1/4] Building cycle table and next-observed-record pairs...")
    raw, cycles = build_cycle_table()
    pairs = make_pairs(cycles)
    nmc = pairs[pairs["chemistry"] == "NMC"]
    lfp = pairs[pairs["chemistry"] == "LFP"]
    print(f"  NMC pairs={len(nmc)} cells={nmc['cell_id'].nunique()}")
    print(f"  LFP pairs={len(lfp)} cells={lfp['cell_id'].nunique()}")
    if len(nmc) != 465:
        raise RuntimeError(f"Expected 465 NMC pairs, got {len(nmc)}")
    if len(lfp) != 248:
        raise RuntimeError(f"Expected 248 LFP pairs, got {len(lfp)}")

    print("[2/4] Running LFP expanding-window walk-forward...")
    lfp_out = run_lfp(pairs, lfp_safe)
    lfp_out["folds"].to_csv(RESULTS_DIR / "lfp_walk_forward_folds.csv", index=False)
    lfp_out["metrics"].to_csv(RESULTS_DIR / "lfp_walk_forward_metrics.csv", index=False)
    lfp_out["predictions"].to_csv(
        RESULTS_DIR / "lfp_walk_forward_predictions.csv", index=False
    )
    lfp_out["gap_pairs"].to_csv(
        RESULTS_DIR / "lfp_walk_forward_gap_pairs.csv", index=False
    )
    if len(lfp_out["gap_detail"]):
        lfp_out["gap_detail"].to_csv(
            RESULTS_DIR / "lfp_walk_forward_gap_pair_errors.csv", index=False
        )

    print("[3/4] Running NMC pooled per-cell expanding-window walk-forward...")
    nmc_out = run_nmc(
        pairs, nmc_features, nmc_features + ["efc_cumulative"]
    )
    nmc_out["folds"].to_csv(RESULTS_DIR / "nmc_walk_forward_folds.csv", index=False)
    nmc_out["fold_metrics"].to_csv(
        RESULTS_DIR / "nmc_walk_forward_fold_metrics.csv", index=False
    )
    nmc_out["cell_metrics"].to_csv(
        RESULTS_DIR / "nmc_walk_forward_cell_metrics.csv", index=False
    )
    nmc_out["pooled_metrics"].to_csv(
        RESULTS_DIR / "nmc_walk_forward_pooled_metrics.csv", index=False
    )
    nmc_out["predictions"].to_csv(
        RESULTS_DIR / "nmc_walk_forward_predictions.csv", index=False
    )
    nmc_out["model_meta"].to_csv(
        RESULTS_DIR / "nmc_walk_forward_model_meta.csv", index=False
    )

    print("[4/4] Writing summary JSON...")
    summary = {
        "created": dt.datetime.now().isoformat(),
        "command": "python member2_modeling/experiments/walk_forward_validation.py",
        "parquet_sha256": sha256(PARQUET),
        "schema_sha256": sha256(SCHEMA),
        "script_sha256": sha256(Path(__file__).resolve()),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "data": {
            "labelled_cycles_total": int(len(cycles)),
            "labelled_cycles_nmc": int((cycles["chemistry"] == "NMC").sum()),
            "labelled_cycles_lfp": int((cycles["chemistry"] == "LFP").sum()),
            "nmc_pairs": int(len(nmc)),
            "nmc_cells": sorted(nmc["cell_id"].unique().tolist()),
            "lfp_pairs": int(len(lfp)),
            "lfp_cells": sorted(lfp["cell_id"].unique().tolist()),
            "lfp_cycle_40_null_soh_removed": True,
            "lfp_missing_cycle_blocks": [[101, 150], [201, 400]],
            "lfp_gap_pairs": [
                {
                    "current_cycle_id": int(r.cycle_id),
                    "target_cycle_id": int(r.target_cycle_id),
                    "cycle_gap": int(r.cycle_gap),
                    "current_soh": float(r.current_soh_pct),
                    "target_soh": float(r.target_soh_pct),
                }
                for r in lfp[lfp["is_gap_pair"]].itertuples()
            ],
            "horizon_definition": "next observed labelled record, not next physical cycle",
        },
        "lfp": {
            "folds": lfp_out["folds"].to_dict(orient="records"),
            "consistency": lfp_out["consistency"],
            "pooled_metrics": lfp_out["pooled_metrics"].to_dict(orient="records"),
            "gap_pairs": lfp_out["gap_pairs"].to_dict(orient="records"),
            "feature_set": lfp_safe,
            "n_features": len(lfp_safe),
        },
        "nmc": {
            "folds": nmc_out["folds"].to_dict(orient="records"),
            "consistency": nmc_out["consistency"],
            "pooled_metrics": nmc_out["pooled_metrics"].to_dict(orient="records"),
            "model_meta": nmc_out["model_meta"].to_dict(orient="records"),
            "feature_set": nmc_features,
            "n_features": len(nmc_features),
        },
        "caveats": [
            "LFP has one cell; temporal validation on B5 does not establish independent-cell or chemistry generalization.",
            "The target is the next observed record; 39->41, 100->151 and 200->401 cross missing IDs.",
            "All preprocessing, alpha selection and early stopping are confined to each fold's training window.",
            "Walk-forward test windows are disjoint, but later training windows include earlier test records; folds are not independent.",
            "R2 on small fold-cell subsets is unstable and should not drive model selection.",
            "No production artifact was read for inference or written.",
        ],
    }
    (RESULTS_DIR / "walk_forward_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2), encoding="utf-8"
    )

    print("LFP pooled metrics:")
    print(lfp_out["pooled_metrics"][["model", "n", "mae", "rmse", "r2", "beats_persistence"]].to_string(index=False))
    print("NMC pooled metrics:")
    print(
        nmc_out["pooled_metrics"][nmc_out["pooled_metrics"]["scope"] == "pooled_cells"][
            ["model", "n", "mae", "rmse", "r2", "beats_persistence"]
        ].to_string(index=False)
    )
    print("Wrote walk-forward results to", RESULTS_DIR)


if __name__ == "__main__":
    main()
