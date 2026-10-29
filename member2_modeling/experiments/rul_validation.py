"""Isolated, reproducible validation of the NMC RUL predictor.

Target convention (project-defined, not a physical EOL definition):
  * One row per observed cycle, NMC cycle ids are contiguous so position
    equals physical cycle step for these cells.
  * crossing_pos = first observed position with SOH < 80.0 %.
  * RUL(pos) = crossing_pos - pos for pos < crossing_pos, and 0 at or after
    the crossing. Units are observed cycles.
  * Primary evaluation uses pre-threshold rows (RUL > 0) only.
  * Post-threshold rows (RUL == 0) are reported separately and never pooled
    into the primary score.

No production artifact is written. The shipped model is only read for an
explicit in-sample diagnostic. All validation models are retrained inside
each leave-one-cell-out fold.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_RESULTS = EXPERIMENT_DIR / "results" / "rul"

PARQUET = REPO_ROOT / "master_table.parquet"
SCHEMA = REPO_ROOT / "member2_modeling" / "feature_schema.json"
SHIPPED_MODEL = REPO_ROOT / "member2_modeling" / "nmc_rul_lgbm_model.joblib"

PROTECTED_ARTIFACTS = {
    "master_table_parquet": PARQUET,
    "feature_schema": SCHEMA,
    "model_manifest": REPO_ROOT / "member2_modeling" / "model_manifest.json",
    "shipped_model": SHIPPED_MODEL,
    "build_and_train": REPO_ROOT / "member2_modeling" / "build_and_train.py",
    "predictor": REPO_ROOT / "member2_modeling" / "predictor.py",
}

EOL_PCT = 80.0
CELLS = ("B0005", "B0006", "B0018")
MIN_HISTORY = 8
NEAR_EOL_CYCLES = 10
BUCKETS = ((1, 5), (6, 10), (11, 30), (31, np.inf))

PRODUCTION_TRAIN_FRAC = 0.60
PRODUCTION_VAL_FRAC = 0.20

BASELINE_MODELS = [
    "last_fade_persistence",
    "train_median_constant",
    "soh_extrapolation_global",
    "soh_extrapolation_causal",
    "soh_extrapolation_causal_median",
]
LGBM_MODELS = ["lgbm_pre_eol_fixed200", "lgbm_production_protocol"]
ALL_MODELS = BASELINE_MODELS + LGBM_MODELS

# Exact prior-diagnostic configuration (run_experiments.run_rul), kept so the
# earlier 6.78-cycle result can be reproduced and audited.
PRIOR_LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_depth": 5,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}

# Production configuration (build_and_train.train_lightgbm).
PRODUCTION_LGBM_PARAMS = {
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
    "random_state": 42,
    "n_estimators": 500,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_hashes() -> dict[str, str]:
    return {name: sha256(path) for name, path in PROTECTED_ARTIFACTS.items()}


def load_nmc_cycles(parquet: Path = PARQUET) -> pd.DataFrame:
    from member2_modeling import build_and_train as bt

    raw = pd.read_parquet(Path(parquet))
    nmc = raw[raw["chemistry"] == "NMC"].copy()
    if nmc.empty:
        raise ValueError("no NMC rows in master table")
    with contextlib.redirect_stdout(io.StringIO()):
        cycles = bt.build_cycle_feature_table(nmc)
    cycles = cycles.dropna(subset=["soh_pct"]).copy()
    cycles = cycles.sort_values(["cell_id", "cycle_id"], kind="mergesort").reset_index(drop=True)
    cycles["cycle_position"] = cycles.groupby("cell_id").cumcount()
    cycles["current_soh_pct"] = cycles["soh_pct"]
    return cycles


def load_feature_names(schema_path: Path = SCHEMA) -> list[str]:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return list(schema["models"]["nmc_rul_lgbm"]["feature_names"])


# ── Target definition ────────────────────────────────────────────────────────


def crossing_positions(soh: np.ndarray, convention: str = "first") -> int | None:
    below = np.asarray(soh, dtype=float) < EOL_PCT
    if not below.any():
        return None
    if convention == "first":
        return int(np.flatnonzero(below)[0])
    if convention == "sustained3":
        for k in range(len(below) - 2):
            if below[k] and below[k + 1] and below[k + 2]:
                return int(k)
        return int(np.flatnonzero(below)[0])
    raise ValueError(f"unknown crossing convention: {convention}")


def add_rul_labels(cycles: pd.DataFrame, convention: str = "first") -> pd.DataFrame:
    frames = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_position", kind="mergesort").reset_index(drop=True)
        cross = crossing_positions(group["soh_pct"].to_numpy(dtype=float), convention)
        if cross is None:
            group["rul_cycles"] = np.nan
            group["crossing_position"] = np.nan
        else:
            pos = group["cycle_position"].to_numpy(dtype=int)
            group["rul_cycles"] = np.where(pos < cross, cross - pos, 0).astype(float)
            group["crossing_position"] = float(cross)
        frames.append(group)
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["cell_id", "cycle_position"], kind="mergesort").reset_index(drop=True)


def audit_targets(cycles: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_position").reset_index(drop=True)
        soh = group["soh_pct"].to_numpy(dtype=float)
        below = soh < EOL_PCT
        first = crossing_positions(soh, "first")
        sustained = crossing_positions(soh, "sustained3")
        last_below = int(np.flatnonzero(below)[-1]) if below.any() else None
        runs = int(np.sum(below[1:] & ~below[:-1]) + int(below[0]))
        pos = group["cycle_position"].to_numpy(dtype=int)
        rows.append(
            {
                "cell_id": cell_id,
                "n_cycles": int(len(group)),
                "cycle_id_min": int(group["cycle_id"].min()),
                "cycle_id_max": int(group["cycle_id"].max()),
                "cycle_ids_contiguous": bool(
                    np.array_equal(np.sort(group["cycle_id"].unique()), np.arange(
                        int(group["cycle_id"].min()), int(group["cycle_id"].max()) + 1
                    ))
                ),
                "null_soh": int(group["soh_pct"].isna().sum()),
                "soh_first": float(soh[0]),
                "soh_min": float(soh.min()),
                "soh_max": float(soh.max()),
                "soh_last": float(soh[-1]),
                "first_crossing_position": first,
                "first_crossing_cycle_id": None if first is None else int(group["cycle_id"].iloc[first]),
                "first_crossing_soh": None if first is None else float(soh[first]),
                "sustained3_crossing_position": sustained,
                "sustained3_crossing_cycle_id": None if sustained is None else int(group["cycle_id"].iloc[sustained]),
                "last_below_position": last_below,
                "last_below_cycle_id": None if last_below is None else int(group["cycle_id"].iloc[last_below]),
                "below_runs": runs,
                "cycles_above_eol_after_first_crossing": int(
                    np.sum(~below[first:]) if first is not None else 0
                ),
                "n_pre_threshold": int(np.sum(pos < first)) if first is not None else int(len(group)),
                "n_post_threshold": int(np.sum(pos >= first)) if first is not None else 0,
                "n_pre_if_sustained3": int(np.sum(pos < sustained)) if sustained is not None else int(len(group)),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "eol_threshold_pct": EOL_PCT,
        "threshold_rule": "strict SOH < 80.0 %",
        "convention": (
            "RUL(pos) = first_crossing_position - pos in observed cycles; "
            "0 at and after the crossing"
        ),
        "cells": table.to_dict(orient="records"),
        "cells_starting_below_threshold": table.loc[
            table["soh_first"] < EOL_PCT, "cell_id"
        ].tolist(),
        "cells_never_crossing": table.loc[
            table["first_crossing_position"].isna(), "cell_id"
        ].tolist(),
        "transient_first_crossings": table.loc[
            table["cycles_above_eol_after_first_crossing"] > 0,
            ["cell_id", "cycles_above_eol_after_first_crossing"],
        ].to_dict(orient="records"),
    }
    return table, summary


def production_split_audit(cycles: pd.DataFrame) -> dict:
    """Reproduce the production 60/20/20 per-cell split and inspect test targets."""
    frames = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_position").reset_index(drop=True)
        n = len(group)
        n_train = int(np.floor(n * PRODUCTION_TRAIN_FRAC))
        n_val = int(np.floor(n * PRODUCTION_VAL_FRAC))
        group["split"] = "train"
        group.loc[group.index[n_train:n_train + n_val], "split"] = "val"
        group.loc[group.index[n_train + n_val:], "split"] = "test"
        frames.append(group)
    split = pd.concat(frames, ignore_index=True)
    labeled = add_rul_labels(split, "first")

    per_cell = []
    for cell_id, group in labeled.groupby("cell_id", sort=True):
        row = {"cell_id": cell_id}
        for name in ("train", "val", "test"):
            sub = group[group["split"] == name]
            row[f"{name}_n"] = int(len(sub))
            row[f"{name}_pre_n"] = int((sub["rul_cycles"] > 0).sum())
            row[f"{name}_post_n"] = int((sub["rul_cycles"] == 0).sum())
            if len(sub):
                row[f"{name}_cycle_min"] = int(sub["cycle_id"].min())
                row[f"{name}_cycle_max"] = int(sub["cycle_id"].max())
                row[f"{name}_rul_min"] = float(sub["rul_cycles"].min())
                row[f"{name}_rul_max"] = float(sub["rul_cycles"].max())
        per_cell.append(row)

    test = labeled[labeled["split"] == "test"]
    return {
        "split": "per-cell chronological 60/20/20 (production)",
        "per_cell": per_cell,
        "test_n": int(len(test)),
        "test_pre_eol_n": int((test["rul_cycles"] > 0).sum()),
        "test_post_eol_n": int((test["rul_cycles"] == 0).sum()),
        "test_rul_min": float(test["rul_cycles"].min()),
        "test_rul_max": float(test["rul_cycles"].max()),
        "conclusion": (
            "the production test split is entirely post-threshold, so its RUL "
            "target has zero variance and cannot measure pre-EOL forecasting"
        ),
    }


# ── Folds ────────────────────────────────────────────────────────────────────


def loco_folds(cycles: pd.DataFrame) -> list[dict]:
    cells = sorted(cycles["cell_id"].unique())
    if len(cells) < 2:
        raise ValueError("LOCO needs at least two cells")
    folds = []
    for held in cells:
        train = cycles[cycles["cell_id"] != held].copy()
        test = cycles[cycles["cell_id"] == held].copy()
        if set(train["cell_id"]) & set(test["cell_id"]):
            raise ValueError("cell leakage between train and test")
        folds.append(
            {
                "held_out_cell": held,
                "train": train.sort_values(["cell_id", "cycle_position"]).reset_index(drop=True),
                "test": test.sort_values("cycle_position").reset_index(drop=True),
            }
        )
    return folds


# ── Baselines ────────────────────────────────────────────────────────────────


def _median_training_slope(train_pre: pd.DataFrame) -> float:
    slopes = []
    for _, group in train_pre.groupby("cell_id"):
        group = group.sort_values("cycle_position")
        if len(group) >= 3:
            slopes.append(
                float(np.polyfit(group["cycle_position"], group["soh_pct"], 1)[0])
            )
    return float(np.median(slopes)) if slopes else -0.01


def _apply_slope(soh: np.ndarray, slope: float, fallback: float) -> np.ndarray:
    use = slope if slope < 0 else fallback
    if use >= 0:
        raise ValueError("no negative fade slope available")
    return np.clip((soh - EOL_PCT) / (-use), 0.0, None)


def causal_extrapolation(
    test_cell: pd.DataFrame,
    mode: str,
    fallback_slope: float,
    min_history: int = MIN_HISTORY,
) -> np.ndarray:
    """Per-row RUL estimate from that cell's observed SOH history only."""
    group = test_cell.sort_values("cycle_position").reset_index(drop=True)
    soh = group["soh_pct"].to_numpy(dtype=float)
    pred = np.empty(len(group), dtype=float)
    for p in range(len(group)):
        history = soh[: p + 1]
        estimate = None
        if len(history) >= min_history:
            if mode == "causal_slope":
                slope = float(np.polyfit(np.arange(len(history)), history, 1)[0])
                if slope < 0:
                    estimate = (soh[p] - EOL_PCT) / (-slope)
            elif mode == "causal_median_fade":
                deltas = np.diff(history)
                fade = float(np.median(deltas)) if len(deltas) else 0.0
                if fade < 0:
                    estimate = (soh[p] - EOL_PCT) / (-fade)
            elif mode == "last_fade":
                if p > 0:
                    fade = soh[p] - soh[p - 1]
                    if fade < 0:
                        estimate = (soh[p] - EOL_PCT) / (-fade)
            else:
                raise ValueError(f"unknown causal mode: {mode}")
        if estimate is not None:
            pred[p] = max(0.0, float(estimate))
        elif fallback_slope < 0:
            pred[p] = max(0.0, float((soh[p] - EOL_PCT) / (-fallback_slope)))
        else:
            pred[p] = 0.0
    return pred


def baseline_predictions(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, np.ndarray]:
    train_pre = train[train["rul_cycles"] > 0]
    if train_pre.empty:
        raise ValueError("no pre-threshold training rows")
    slope = _median_training_slope(train_pre)
    median_rul = float(train_pre["rul_cycles"].median())

    test = test.sort_values("cycle_position").reset_index(drop=True)
    soh = test["soh_pct"].to_numpy(dtype=float)

    global_pred = (
        _apply_slope(soh, slope, slope)
        if slope < 0
        else np.full(len(test), median_rul)
    )

    preds = {
        "train_median_constant": np.full(len(test), median_rul),
        "soh_extrapolation_global": global_pred,
    }
    for cell_id, group in test.groupby("cell_id"):
        idx = group.index
        if slope < 0:
            preds.setdefault("last_fade_persistence", np.full(len(test), np.nan))
            preds.setdefault("soh_extrapolation_causal", np.full(len(test), np.nan))
            preds.setdefault("soh_extrapolation_causal_median", np.full(len(test), np.nan))
            preds["last_fade_persistence"][idx] = causal_extrapolation(
                group, "last_fade", slope
            )
            preds["soh_extrapolation_causal"][idx] = causal_extrapolation(
                group, "causal_slope", slope
            )
            preds["soh_extrapolation_causal_median"][idx] = causal_extrapolation(
                group, "causal_median_fade", slope
            )
        else:
            for name in (
                "last_fade_persistence",
                "soh_extrapolation_causal",
                "soh_extrapolation_causal_median",
            ):
                preds[name] = np.full(len(test), median_rul)
    return preds


# ── LightGBM variants ────────────────────────────────────────────────────────


def _predict(model, frame: pd.DataFrame, features: list[str], imputer=None) -> np.ndarray:
    X = frame[features].astype(float)
    if imputer is not None:
        X = imputer.transform(X)
    return np.asarray(model.predict(X), dtype=float)


def fit_predict_lgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    variant: str,
) -> tuple[np.ndarray, dict]:
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer

    if variant == "lgbm_pre_eol_fixed200":
        train_pre = train[train["rul_cycles"] > 0].copy()
        model = lgb.LGBMRegressor(**PRIOR_LGBM_PARAMS)
        model.fit(train_pre[features].astype(float), train_pre["rul_cycles"].to_numpy())
        pred = _predict(model, test, features)
        info = {
            "variant": variant,
            "train_rows": int(len(train_pre)),
            "train_pre_rows": int(len(train_pre)),
            "val_rows": 0,
            "best_iteration": None,
            "n_estimators": PRIOR_LGBM_PARAMS["n_estimators"],
        }
        return pred, info

    if variant == "lgbm_production_protocol":
        train_parts, val_parts = [], []
        for _, group in train.groupby("cell_id"):
            group = group.sort_values("cycle_position").reset_index(drop=True)
            n = len(group)
            n_train = int(np.floor(n * PRODUCTION_TRAIN_FRAC))
            n_val = int(np.floor(n * PRODUCTION_VAL_FRAC))
            train_parts.append(group.iloc[:n_train])
            val_parts.append(group.iloc[n_train:n_train + n_val])
        tr = pd.concat(train_parts, ignore_index=True)
        va = pd.concat(val_parts, ignore_index=True)
        imputer = SimpleImputer(strategy="median").fit(tr[features].astype(float))
        model = lgb.LGBMRegressor(**PRODUCTION_LGBM_PARAMS)
        model.fit(
            imputer.transform(tr[features].astype(float)),
            tr["rul_cycles"].to_numpy(),
            eval_set=[(imputer.transform(va[features].astype(float)), va["rul_cycles"].to_numpy())],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred = _predict(model, test, features, imputer)
        info = {
            "variant": variant,
            "train_rows": int(len(tr)),
            "train_pre_rows": int((tr["rul_cycles"] > 0).sum()),
            "train_post_rows": int((tr["rul_cycles"] == 0).sum()),
            "val_rows": int(len(va)),
            "best_iteration": int(model.best_iteration_ or PRODUCTION_LGBM_PARAMS["n_estimators"]),
            "n_estimators": PRODUCTION_LGBM_PARAMS["n_estimators"],
        }
        return pred, info

    raise ValueError(f"unknown variant: {variant}")


# ── Metrics ──────────────────────────────────────────────────────────────────


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None,
                "median_ae": None, "max_ae": None, "mape_pct": None}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if len(y_true) >= 2:
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else None
    else:
        r2 = None
    nonzero = np.abs(y_true) > 1e-9
    mape = float(np.mean(np.abs(err[nonzero] / y_true[nonzero])) * 100) if nonzero.any() else None
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias": float(np.mean(err)),
        "median_ae": float(np.median(np.abs(err))),
        "max_ae": float(np.max(np.abs(err))),
        "mape_pct": mape,
    }


def _bucket(rul: float) -> str:
    for low, high in BUCKETS:
        if low <= rul <= high:
            return f"{int(low)}-{int(high)}" if np.isfinite(high) else f"{int(low)}+"
    return "unbucketed"


def evaluate(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict], dict[str, pd.DataFrame]]:
    """Per-held-out-cell pre-threshold metrics and per-model prediction frames."""
    by_cell, near = [], []
    prediction_frames = {}
    for model, pred in predictions.items():
        data = frame.copy()
        data["prediction_raw"] = np.asarray(pred, dtype=float)
        data["prediction"] = np.clip(data["prediction_raw"], 0.0, None)
        prediction_frames[model] = data

        pre = data[data["rul_cycles"] > 0]
        for cell_id, group in pre.groupby("cell_id"):
            by_cell.append(
                {"model": model, "scope": "pre_threshold", "held_out_cell": cell_id,
                 **regression_metrics(group["rul_cycles"], group["prediction"]),
                 "negative_raw_predictions": int((group["prediction_raw"] < 0).sum())}
            )
        near_eol = pre[pre["rul_cycles"] <= NEAR_EOL_CYCLES]
        for cell_id, group in near_eol.groupby("cell_id"):
            near.append(
                {"model": model, "scope": f"rul<={NEAR_EOL_CYCLES}", "held_out_cell": cell_id,
                 **regression_metrics(group["rul_cycles"], group["prediction"])}
            )
    return by_cell, near, prediction_frames


def pooled_tables(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pool all held-out folds (each row predicted exactly once)."""
    pooled, near, buckets = [], [], []
    for model, data in predictions.groupby("model"):
        pre = data[data["rul_cycles"] > 0]
        pooled.append(
            {"model": model, "scope": "pre_threshold", "held_out_cell": "ALL",
             **regression_metrics(pre["rul_cycles"], pre["prediction"]),
             "negative_raw_predictions": int((pre["prediction_raw"] < 0).sum())}
        )
        near_eol = pre[pre["rul_cycles"] <= NEAR_EOL_CYCLES]
        near.append(
            {"model": model, "scope": f"rul<={NEAR_EOL_CYCLES}", "held_out_cell": "ALL",
             **regression_metrics(near_eol["rul_cycles"], near_eol["prediction"])}
        )
        for cell_id, group in near_eol.groupby("cell_id"):
            near.append(
                {"model": model, "scope": f"rul<={NEAR_EOL_CYCLES}", "held_out_cell": cell_id,
                 **regression_metrics(group["rul_cycles"], group["prediction"])}
            )
        bucketed = pre.assign(bucket=pre["rul_cycles"].map(_bucket))
        for bucket, group in bucketed.groupby("bucket"):
            buckets.append(
                {"model": model, "scope": "pre_threshold", "held_out_cell": "ALL",
                 "bucket": bucket,
                 **regression_metrics(group["rul_cycles"], group["prediction"])}
            )
    return pd.DataFrame(pooled), pd.DataFrame(near), pd.DataFrame(buckets)


def post_threshold_metrics(
    frame: pd.DataFrame, predictions: dict[str, np.ndarray]
) -> tuple[list[dict], list[dict]]:
    rows, per_cell = [], []
    for model, pred in predictions.items():
        data = frame.copy()
        data["prediction"] = np.clip(np.asarray(pred, dtype=float), 0.0, None)
        post = data[data["rul_cycles"] == 0]
        met = regression_metrics(post["rul_cycles"], post["prediction"])
        rows.append(
            {
                "model": model,
                "n": met["n"],
                "mae": met["mae"],
                "rmse": met["rmse"],
                "mean_prediction": float(np.mean(post["prediction"])) if len(post) else None,
                "max_prediction": float(np.max(post["prediction"])) if len(post) else None,
            }
        )
        for cell_id, group in post.groupby("cell_id"):
            per_cell.append(
                {
                    "model": model,
                    "held_out_cell": cell_id,
                    "n": int(len(group)),
                    "mae": float(np.mean(np.abs(group["prediction"]))),
                    "rmse": float(np.sqrt(np.mean(group["prediction"] ** 2))),
                    "mean_prediction": float(np.mean(group["prediction"])),
                    "max_prediction": float(np.max(group["prediction"])),
                }
            )
    return rows, per_cell


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
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


# ── Shipped model diagnostic ────────────────────────────────────────────────


def shipped_model_diagnostic(
    cycles: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Predict with the shipped artifact on its own training cells (in-sample)."""
    import joblib

    model = joblib.load(SHIPPED_MODEL)
    rows = []
    abs_errors = []
    for cell_id, group in cycles.groupby("cell_id", sort=True):
        group = group.sort_values("cycle_position").reset_index(drop=True)
        pred = np.clip(_predict(model, group, features), 0.0, None)
        pre = group["rul_cycles"].to_numpy(dtype=float) > 0
        abs_errors.append(np.abs(group["rul_cycles"].to_numpy(dtype=float)[pre] - pred[pre]))
        rows.append(
            {
                "cell_id": cell_id,
                "in_sample": True,
                "pre_n": int(pre.sum()),
                "pre_mae": float(np.mean(abs_errors[-1])),
                "pre_mean_prediction": float(np.mean(pred[pre])),
                "pre_pred_min": float(np.min(pred[pre])),
                "pre_pred_max": float(np.max(pred[pre])),
                "post_n": int((~pre).sum()),
                "post_mean_prediction": float(np.mean(pred[~pre])),
                "post_max_prediction": float(np.max(pred[~pre])),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "artifact": str(SHIPPED_MODEL),
        "in_sample": True,
        "note": (
            "the shipped model was trained on all three cells; these predictions "
            "are in-sample and are NOT validation evidence"
        ),
        "per_cell": table.to_dict(orient="records"),
        "pre_mae_overall": float(np.mean(np.concatenate(abs_errors))),
    }
    return table, summary


# ── Experiment ───────────────────────────────────────────────────────────────


def run_experiment(parquet: Path = PARQUET, out_dir: Path | None = None, write: bool = True) -> dict:
    out_dir = Path(out_dir) if out_dir else DEFAULT_RESULTS
    hashes_before = protected_hashes()
    features = load_feature_names()
    cycles = load_nmc_cycles(parquet)
    target_table, target_summary = audit_targets(cycles)
    production = production_split_audit(cycles)

    results = {}
    all_metrics = {}
    for convention in ("first", "sustained3"):
        labeled = add_rul_labels(cycles, convention)
        folds = loco_folds(labeled)
        by_cell, near = [], []
        post_rows, post_cell = [], []
        prediction_rows = []
        meta = {}

        for fold in folds:
            held = fold["held_out_cell"]
            train, test = fold["train"], fold["test"]
            preds = baseline_predictions(train, test)
            fold_meta = {}
            for variant in LGBM_MODELS:
                pred, info = fit_predict_lgbm(train, test, features, variant)
                preds[variant] = pred
                fold_meta[variant] = info
            meta[held] = fold_meta

            bc, ne, pred_frames = evaluate(test, preds)
            by_cell.extend(bc)
            near.extend(ne)
            pr, pc = post_threshold_metrics(test, preds)
            post_rows.extend([{"held_out_cell": held, **r} for r in pr])
            post_cell.extend(pc)

            for model, data in pred_frames.items():
                frame = data[
                    ["cell_id", "cycle_id", "cycle_position", "soh_pct", "rul_cycles",
                     "prediction_raw", "prediction"]
                ].copy()
                frame["held_out_cell"] = held
                frame["model"] = model
                frame["is_pre_threshold"] = frame["rul_cycles"] > 0
                frame["error"] = frame["prediction"] - frame["rul_cycles"]
                frame["abs_error"] = frame["error"].abs()
                prediction_rows.append(frame)

        predictions = pd.concat(prediction_rows, ignore_index=True)
        pooled, near_pooled, buckets = pooled_tables(predictions)
        results[convention] = {
            "by_cell": pd.DataFrame(by_cell),
            "pooled": pooled,
            "near_eol": near_pooled,
            "near_eol_by_cell": pd.DataFrame(near),
            "buckets": buckets,
            "post": pd.DataFrame(post_rows),
            "post_cell": pd.DataFrame(post_cell),
            "predictions": predictions,
            "meta": meta,
            "labeled": labeled,
        }

    shipped_table, shipped_summary = shipped_model_diagnostic(
        add_rul_labels(cycles, "first"), features
    )

    hashes_after = protected_hashes()
    if hashes_after != hashes_before:
        raise RuntimeError("a protected production artifact changed during validation")

    summary = {
        "experiment": "member2_nmc_rul_loco",
        "parquet": str(Path(parquet).resolve()),
        "parquet_sha256": sha256(parquet),
        "schema_sha256": sha256(SCHEMA),
        "shipped_model_sha256": sha256(SHIPPED_MODEL),
        "protected_artifacts_unchanged": True,
        "protected_artifact_sha256": hashes_after,
        "cells": list(CELLS),
        "target": target_summary,
        "production_split": production,
        "feature_count": len(features),
        "features": features,
        "validation": "leave-one-cell-out; pre-threshold rows primary",
        "primary_pooled": results["first"]["pooled"].to_dict(orient="records"),
        "primary_by_cell": results["first"]["by_cell"].to_dict(orient="records"),
        "primary_near_eol": results["first"]["near_eol"].to_dict(orient="records"),
        "post_threshold": results["first"]["post"].to_dict(orient="records"),
        "sensitivity_sustained3_pooled": results["sustained3"]["pooled"].to_dict(orient="records"),
        "sensitivity_sustained3_by_cell": results["sustained3"]["by_cell"].to_dict(orient="records"),
        "shipped_model_diagnostic": shipped_summary,
        "model_metadata": results["first"]["meta"],
        "caveats": [
            "80% is a project-defined threshold, not a manufacturer EOL definition.",
            "SOH is non-monotonic: B0018 first crosses at cycle 45 but returns above 80% for 13 later cycles.",
            "The production test split is 100% post-threshold, so its RUL target has zero variance.",
            "The shipped model was trained on all three cells; its pre-EOL accuracy is in-sample only.",
            "Only three cells are available, so LOCO trains on two cells and tests on one.",
        ],
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        target_table.to_csv(out_dir / "nmc_rul_target_audit.csv", index=False)
        (out_dir / "nmc_rul_production_split_audit.json").write_text(
            json.dumps(_jsonable(production), indent=2), encoding="utf-8"
        )
        for convention, tag in (("first", "primary"), ("sustained3", "sensitivity_sustained3")):
            r = results[convention]
            r["by_cell"].to_csv(out_dir / f"nmc_rul_{tag}_by_cell.csv", index=False)
            r["pooled"].to_csv(out_dir / f"nmc_rul_{tag}_pooled.csv", index=False)
            r["near_eol"].to_csv(out_dir / f"nmc_rul_{tag}_near_eol.csv", index=False)
            r["buckets"].to_csv(out_dir / f"nmc_rul_{tag}_error_buckets.csv", index=False)
            r["predictions"].to_csv(out_dir / f"nmc_rul_{tag}_predictions.csv", index=False)
            if convention == "first":
                r["post"].to_csv(out_dir / "nmc_rul_post_threshold_metrics.csv", index=False)
                r["post_cell"].to_csv(out_dir / "nmc_rul_post_threshold_by_cell.csv", index=False)
        (out_dir / "nmc_rul_model_metadata.json").write_text(
            json.dumps(_jsonable(results["first"]["meta"]), indent=2), encoding="utf-8"
        )
        shipped_table.to_csv(out_dir / "nmc_rul_shipped_model_diagnostic.csv", index=False)
        (out_dir / "nmc_rul_summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
        )

    return {
        "summary": summary,
        "results": results,
        "target_table": target_table,
        "production": production,
        "shipped": shipped_table,
        "out_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NMC RUL LOCO validation")
    parser.add_argument("--parquet", type=Path, default=PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    result = run_experiment(args.parquet, args.out_dir)

    print("\nNMC RUL LOCO validation")
    print("target crossings:")
    print(result["target_table"][
        ["cell_id", "first_crossing_cycle_id", "n_pre_threshold", "n_post_threshold",
         "cycles_above_eol_after_first_crossing"]
    ].to_string(index=False))
    print("\nproduction test composition:")
    prod = result["production"]
    print(f"  test rows={prod['test_n']} pre={prod['test_pre_eol_n']} post={prod['test_post_eol_n']}")
    print("\npre-threshold pooled metrics (first-crossing target):")
    pooled = result["results"]["first"]["pooled"]
    print(pooled[["model", "n", "mae", "rmse", "r2", "bias", "negative_raw_predictions"]]
          .round(3).to_string(index=False))
    print("\nper-cell MAE:")
    by_cell = result["results"]["first"]["by_cell"]
    print(by_cell.pivot(index="held_out_cell", columns="model", values="mae").round(2).to_string())
    print("\nresults written to", result["out_dir"])


if __name__ == "__main__":
    main()
