"""Leave-one-cell-out evaluation for the Na-ion experiment.

Protocol:
  * One sample is a (cell, next-observed-record) pair built from cycle-level
    rows. There are 96 pairs across 4 cells; repeated telemetry rows never
    become samples.
  * Validation is leave-one-cell-out (LOCO): the held-out cell is never used
    for imputation, scaling, alpha selection or fitting.
  * Ridge alpha is selected by an inner LOCO over the three training cells.
  * Every baseline that needs a drift estimate uses training cells only, or
    history from the held-out cell that precedes the prediction time.
  * Two predeclared Ridge targets: absolute SOH and the one-step delta
    (target - current SOH). Both use the same 12 non-redundant features.

No production artifact is imported, written or promoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import naion_features as nf

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = EXPERIMENT_DIR / "results"

ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

PRIMARY_MODELS = [
    "persistence",
    "pooled_mean_drift",
    "causal_cell_mean_drift",
    "causal_cell_trend",
    "last_delta",
    "train_mean",
    "ridge_absolute",
    "ridge_delta",
]

TREND_MIN_HISTORY = 6


# ── Metrics ──────────────────────────────────────────────────────────────────


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) != len(y_pred):
        raise ValueError("prediction length mismatch")
    if len(y_true) == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    if len(y_true) >= 2:
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else None
    else:
        r2 = None
    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mae_pp": mae,
    }


# ── Folds ────────────────────────────────────────────────────────────────────


def loco_splits(pairs: pd.DataFrame) -> list[dict]:
    cells = sorted(pairs["cell_id"].unique())
    if len(cells) < 2:
        raise ValueError("LOCO needs at least two cells")
    splits = []
    for cell_id in cells:
        test = pairs[pairs["cell_id"] == cell_id].copy()
        train = pairs[pairs["cell_id"] != cell_id].copy()
        if test.empty or train.empty:
            raise ValueError(f"empty LOCO fold for {cell_id}")
        if set(train["cell_id"]) & set(test["cell_id"]):
            raise ValueError("cell leakage between train and test")
        splits.append(
            {
                "held_out_cell": str(cell_id),
                "train": train.reset_index(drop=True),
                "test": test.sort_values("pair_index").reset_index(drop=True),
            }
        )
    return splits


# ── Baselines ────────────────────────────────────────────────────────────────


def _pooled_drift(train: pd.DataFrame) -> float:
    return float((train["target_soh_pct"] - train["current_soh_pct"]).mean())


def causal_cell_mean_drift(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Expanding per-cell mean of already-observed one-step deltas.

    At pair p the observed history contains deltas of pairs 0..p-1 only.
    Pair 0 falls back to the training-cell drift.
    """
    fallback = _pooled_drift(train)
    out = pd.Series(index=test.index, dtype=float)
    for _, group in test.groupby("cell_id"):
        group = group.sort_values("pair_index")
        deltas = (group["target_soh_pct"] - group["current_soh_pct"]).to_numpy(dtype=float)
        for k, idx in enumerate(group.index):
            drift = float(deltas[:k].mean()) if k > 0 else fallback
            out.loc[idx] = group["current_soh_pct"].iloc[k] + drift
    return out.to_numpy(dtype=float)


def causal_cell_trend(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Anchored local linear trend from SOH history observed before each pair."""
    fallback = _pooled_drift(train)
    preds = {}
    for cell_id, group in test.groupby("cell_id"):
        group = group.sort_values("pair_index")
        current = group["current_soh_pct"].to_numpy(dtype=float)
        target = group["target_soh_pct"].to_numpy(dtype=float)
        for k, idx in enumerate(group.index):
            history = np.concatenate([[current[0]], target[:k]])
            x = np.arange(len(history), dtype=float)
            if len(history) >= TREND_MIN_HISTORY:
                slope = float(np.polyfit(x, history, 1)[0])
                preds[idx] = current[k] + slope
            else:
                preds[idx] = current[k] + fallback
    return np.array([preds[i] for i in test.index], dtype=float)


def baseline_predict(train: pd.DataFrame, test: pd.DataFrame, model: str) -> np.ndarray:
    if model == "persistence":
        return test["current_soh_pct"].to_numpy(dtype=float)
    if model == "pooled_mean_drift":
        return test["current_soh_pct"].to_numpy(dtype=float) + _pooled_drift(train)
    if model == "causal_cell_mean_drift":
        return causal_cell_mean_drift(train, test)
    if model == "causal_cell_trend":
        return causal_cell_trend(train, test)
    if model == "last_delta":
        delta = (test["soh_pct"] - test["prev_soh_pct"]).to_numpy(dtype=float)
        delta = np.where(np.isfinite(delta), delta, 0.0)
        return test["current_soh_pct"].to_numpy(dtype=float) + delta
    if model == "train_mean":
        return np.full(len(test), float(train["target_soh_pct"].mean()))
    raise ValueError(f"unknown baseline: {model}")


# ── Ridge ────────────────────────────────────────────────────────────────────


def _ridge_pipeline(alpha: float):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def _fit_xy(frame: pd.DataFrame, features: list[str], delta: bool):
    X = nf.feature_matrix(frame, features)
    y = frame["target_soh_pct"].to_numpy(dtype=float)
    if delta:
        y = y - frame["current_soh_pct"].to_numpy(dtype=float)
    return X, y


def fit_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str] | None = None,
    delta: bool = False,
    alphas: list[float] | None = None,
) -> tuple[np.ndarray, dict]:
    """Train-only alpha search via inner LOCO, then refit on all training cells."""
    features = list(features or nf.FEATURE_COLUMNS)
    nf.assert_no_forbidden_features(features)
    alphas = list(alphas or ALPHAS)

    if set(train["cell_id"]) & set(test["cell_id"]):
        raise ValueError("cell leakage between train and test")
    train_cells = sorted(train["cell_id"].unique())
    if len(train_cells) < 2:
        raise ValueError("inner LOCO needs at least two training cells")

    alpha_scores = {}
    for alpha in alphas:
        fold_maes = []
        for inner_cell in train_cells:
            inner_train = train[train["cell_id"] != inner_cell]
            inner_val = train[train["cell_id"] == inner_cell]
            X_fit, y_fit = _fit_xy(inner_train, features, delta)
            X_val, y_val = _fit_xy(inner_val, features, delta)
            pipe = _ridge_pipeline(alpha)
            pipe.fit(X_fit, y_fit)
            fold_maes.append(float(np.mean(np.abs(y_val - pipe.predict(X_val)))))
        alpha_scores[alpha] = {
            "mean_inner_mae": float(np.mean(fold_maes)),
            "fold_maes": fold_maes,
        }

    best_alpha = min(alphas, key=lambda a: (alpha_scores[a]["mean_inner_mae"], a))
    X_train, y_train = _fit_xy(train, features, delta)
    model = _ridge_pipeline(best_alpha)
    model.fit(X_train, y_train)

    X_test, _ = _fit_xy(test, features, delta)
    pred = np.asarray(model.predict(X_test), dtype=float)
    if delta:
        pred = pred + test["current_soh_pct"].to_numpy(dtype=float)

    info = {
        "target_mode": "delta" if delta else "absolute",
        "selected_alpha": float(best_alpha),
        "inner_loco_mae": float(alpha_scores[best_alpha]["mean_inner_mae"]),
        "alpha_scores": {str(k): v for k, v in alpha_scores.items()},
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": len(features),
        "features": features,
    }
    return pred, info


# ── Experiment ───────────────────────────────────────────────────────────────


def _metric_row(frame: pd.DataFrame, pred: np.ndarray, model: str, **extra) -> dict:
    met = regression_metrics(frame["target_soh_pct"], pred)
    pers = regression_metrics(frame["target_soh_pct"], frame["current_soh_pct"])
    row = {
        "model": model,
        **met,
        "persistence_mae_same_rows": pers["mae"],
        "mae_delta_vs_persistence": (
            met["mae"] - pers["mae"] if met["mae"] is not None else None
        ),
        "beats_persistence": bool(
            met["mae"] is not None and pers["mae"] is not None and met["mae"] < pers["mae"]
        ),
    }
    row.update(extra)
    return row


def _predictions_frame(
    frame: pd.DataFrame, pred: np.ndarray, model: str, held_out_cell: str
) -> pd.DataFrame:
    out = frame[
        [
            "cell_id",
            "pair_index",
            "cycle_id",
            "target_cycle_id",
            "cycle_gap",
            "current_soh_pct",
            "target_soh_pct",
        ]
    ].copy()
    out["model"] = model
    out["held_out_cell"] = held_out_cell
    out["prediction"] = np.asarray(pred, dtype=float)
    out["error"] = out["target_soh_pct"] - out["prediction"]
    out["abs_error"] = out["error"].abs()
    return out


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


def run_experiment(
    parquet: Path = nf.PARQUET, out_dir: Path | None = None, write: bool = True
) -> dict:
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
    raw = nf.load_naion_raw(parquet)
    audit = nf.audit_naion(raw, parquet=Path(parquet))
    cycles = nf.extract_cycle_features(raw)
    pairs = nf.build_pairs(cycles)

    if audit["duplicates"]["within_cycle_duplicate_rows"] != 0:
        raise RuntimeError("telemetry duplicates inside a cycle; audit must be revisited")
    if audit["phase_structure"]["failures"]:
        raise RuntimeError("phase structure audit failed")
    if pairs["is_gap_pair"].any():
        raise RuntimeError("Na-ion cycle ids are expected to be contiguous")

    splits = loco_splits(pairs)
    metric_rows, cell_rows, prediction_frames = [], [], []
    ridge_meta = {}

    for split in splits:
        held_out = split["held_out_cell"]
        train, test = split["train"], split["test"]

        for model in PRIMARY_MODELS:
            if model.startswith("ridge_"):
                continue
            pred = baseline_predict(train, test, model)
            metric_rows.append(
                _metric_row(test, pred, model, held_out_cell=held_out, scope="fold")
            )
            prediction_frames.append(_predictions_frame(test, pred, model, held_out))

        for model, delta in (("ridge_absolute", False), ("ridge_delta", True)):
            pred, info = fit_ridge(train, test, nf.FEATURE_COLUMNS, delta=delta)
            metric_rows.append(
                _metric_row(test, pred, model, held_out_cell=held_out, scope="fold")
            )
            prediction_frames.append(_predictions_frame(test, pred, model, held_out))
            ridge_meta.setdefault(model, {})[held_out] = info

    cell_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    pooled_rows = []
    for model in PRIMARY_MODELS:
        frame = predictions[predictions["model"] == model]
        pooled = _metric_row(frame, frame["prediction"].to_numpy(), model, scope="pooled")
        drift_frame = predictions[predictions["model"] == "pooled_mean_drift"]
        drift_mae = regression_metrics(
            drift_frame["target_soh_pct"], drift_frame["prediction"]
        )["mae"]
        pooled["pooled_mean_drift_mae_same_rows"] = drift_mae
        pooled["mae_delta_vs_pooled_mean_drift"] = (
            pooled["mae"] - drift_mae if pooled["mae"] is not None else None
        )
        pooled["beats_pooled_mean_drift"] = bool(
            pooled["mae"] is not None and drift_mae is not None and pooled["mae"] < drift_mae
        )
        per_cell_mae = (
            cell_metrics[cell_metrics["model"] == model]
            .set_index("held_out_cell")["mae"]
            .to_dict()
        )
        pooled["cells_below_persistence_mae"] = int(
            cell_metrics[cell_metrics["model"] == model]["beats_persistence"].sum()
        )
        pooled["cells_total"] = int(len(splits))
        pooled["per_cell_mae"] = per_cell_mae
        pooled_rows.append(pooled)

    pooled_metrics = pd.DataFrame(pooled_rows)

    # Sensitivity: the prototype's 15-feature set contains three exact linear
    # combinations of retained columns. Rerun both delta models to show the
    # conclusion does not depend on that redundancy.
    sensitivity_preds = []
    for split in splits:
        for model, delta in (
            ("ridge_absolute_15feat", False),
            ("ridge_delta_15feat", True),
        ):
            pred, info = fit_ridge(
                split["train"], split["test"], nf.REDUNDANT_FEATURE_COLUMNS, delta=delta
            )
            sensitivity_preds.append(
                _predictions_frame(split["test"], pred, model, split["held_out_cell"])
            )
            ridge_meta.setdefault(model, {})[split["held_out_cell"]] = info
    sensitivity_predictions = pd.concat(sensitivity_preds, ignore_index=True)
    sensitivity_cell = pd.DataFrame(
        [
            _metric_row(
                sensitivity_predictions[
                    (sensitivity_predictions["model"] == model)
                    & (sensitivity_predictions["held_out_cell"] == cell)
                ],
                sensitivity_predictions[
                    (sensitivity_predictions["model"] == model)
                    & (sensitivity_predictions["held_out_cell"] == cell)
                ]["prediction"].to_numpy(),
                model,
                held_out_cell=cell,
                scope="fold",
            )
            for model in ("ridge_absolute_15feat", "ridge_delta_15feat")
            for cell in sorted(pairs["cell_id"].unique())
        ]
    )
    sensitivity_pooled = pd.DataFrame(
        [
            _metric_row(
                sensitivity_predictions[sensitivity_predictions["model"] == model],
                sensitivity_predictions[sensitivity_predictions["model"] == model][
                    "prediction"
                ].to_numpy(),
                model,
                scope="pooled",
            )
            for model in ("ridge_absolute_15feat", "ridge_delta_15feat")
        ]
    )

    delta_stats = {
        "mean": float((pairs["target_soh_pct"] - pairs["current_soh_pct"]).mean()),
        "median": float((pairs["target_soh_pct"] - pairs["current_soh_pct"]).median()),
        "min": float((pairs["target_soh_pct"] - pairs["current_soh_pct"]).min()),
        "max": float((pairs["target_soh_pct"] - pairs["current_soh_pct"]).max()),
        "std": float((pairs["target_soh_pct"] - pairs["current_soh_pct"]).std(ddof=1)),
    }

    summary = {
        "experiment": "member2_naion_loco",
        "chemistry": nf.CHEMISTRY,
        "production_artifact_used": False,
        "target": "SOH (%) at the next observed record",
        "horizon_note": (
            "next observed record, not a fixed cycle step; all Na-ion pairs have "
            "cycle_gap == 1 so the two coincide here"
        ),
        "validation": "leave-one-cell-out, 4 folds",
        "n_cells": int(pairs["cell_id"].nunique()),
        "n_pairs": int(len(pairs)),
        "pairs_per_cell": {
            str(k): int(v) for k, v in pairs.groupby("cell_id").size().items()
        },
        "primary_features": nf.FEATURE_COLUMNS,
        "redundant_sensitivity_features": nf.REDUNDANT_FEATURE_COLUMNS,
        "pair_delta_stats_pp": delta_stats,
        "metric_note": (
            "R2 is dominated by between-cell SOH level differences and must not "
            "be used to select a model; MAE in percentage points is primary"
        ),
        "rul_supported": False,
        "rul_reason": (
            "shallow fade (<1.1 pp over 25 cycles), no end-of-life crossing, "
            "only four cells and 24 one-step pairs per cell"
        ),
        "generalizable_soh_supported": False,
        "generalizable_soh_reason": (
            "one protocol, four cells, 25 cycles; LOCO only tests transfer across "
            "these four cells, not across packs or conditions"
        ),
        "pooled_metrics": pooled_metrics.to_dict(orient="records"),
        "sensitivity_note": (
            "the 15-feature sensitivity set adds three exact linear combinations "
            "of retained columns; its delta-Ridge MAE differs from the primary "
            "12-feature run, which shows alpha selection is unstable with four cells"
        ),
        "ridge_metadata": ridge_meta,
        "audit": audit,
        "audit_flags": {
            "temperature_available": audit["temperature_available"],
            "cells_at_or_below_eol": audit["cells_at_or_below_eol"],
            "phase_failures": audit["phase_structure"]["failures"],
        },
    }

    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "naion_audit.json").write_text(
            json.dumps(_jsonable(audit), indent=2), encoding="utf-8"
        )
        nf.feature_provenance().to_csv(out_dir / "naion_feature_provenance.csv", index=False)
        cycles.to_csv(out_dir / "naion_cycle_features.csv", index=False)
        pairs.drop(columns=["quality_flag"]).to_csv(out_dir / "naion_pairs.csv", index=False)
        cell_metrics.to_csv(out_dir / "naion_metrics_by_cell.csv", index=False)
        pooled_metrics.to_csv(out_dir / "naion_metrics_pooled.csv", index=False)
        predictions.to_csv(out_dir / "naion_predictions.csv", index=False)
        sensitivity_pooled.to_csv(
            out_dir / "naion_sensitivity_redundant_features.csv", index=False
        )
        sensitivity_cell.to_csv(
            out_dir / "naion_sensitivity_by_cell.csv", index=False
        )
        (out_dir / "naion_model_metadata.json").write_text(
            json.dumps(_jsonable(ridge_meta), indent=2), encoding="utf-8"
        )
        (out_dir / "naion_summary.json").write_text(
            json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
        )

    return {
        "summary": summary,
        "cell_metrics": cell_metrics,
        "pooled_metrics": pooled_metrics,
        "sensitivity_pooled": sensitivity_pooled,
        "pairs": pairs,
        "cycles": cycles,
        "out_dir": out_dir,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Na-ion LOCO experiment")
    parser.add_argument("--parquet", type=Path, default=nf.PARQUET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    result = run_experiment(parquet=args.parquet, out_dir=args.out_dir)
    pooled = result["pooled_metrics"].copy()

    print("Na-ion LOCO experiment")
    print(f"pairs={result['summary']['n_pairs']} cells={result['summary']['n_cells']}")
    print()
    print(
        pooled[["model", "n", "mae", "rmse", "beats_persistence"]]
        .to_string(index=False, float_format=lambda v: f"{v:.6f}")
    )
    print()
    by_cell = result["cell_metrics"].pivot(
        index="held_out_cell", columns="model", values="mae"
    )
    print((by_cell * 1e4).round(2).to_string())
    print("\nMAE by held-out cell x 1e4 (percentage points x 1e4)")
    print(f"\nresults written to {result['out_dir']}")


if __name__ == "__main__":
    main()
