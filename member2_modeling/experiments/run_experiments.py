"""Isolated Member 2 experiments: verified features, baselines, ablations, RUL.

Writes only under member2_modeling/experiments/results/. It never writes to
member2_modeling/*.joblib, *.pt, scaler_*, manifest, schema, or the parquet.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
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
EOL = bt.EOL_THRESHOLD_PCT
ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regression_metrics(y_true, y_pred, label: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"label": label, "n": 0, "mae": None, "rmse": None, "r2": None, "mape_pct": None}
    return bt.evaluate_predictions(y_true[mask], y_pred[mask], label)


def feature_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[cols].astype(float)


def ridge_predict(X_train, y_train, X_val, y_val, X_test):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    for alpha in ALPHAS:
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        mae = float(np.mean(np.abs(y_val - val_pred)))
        if best is None or mae < best[0]:
            best = (mae, alpha, pipe)
    return best[2].predict(X_test), {"alpha": best[1], "val_mae": best[0]}


def trend_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """Per-cell linear target-SOH trend; pooled fallback for unseen/short cells."""
    def fit(sub):
        if len(sub) >= 3:
            slope, intercept = np.polyfit(
                sub["target_position"].to_numpy(dtype=float),
                sub["target_soh_pct"].to_numpy(dtype=float),
                1,
            )
            return float(slope), float(intercept)
        return None

    global_fit = fit(train_df)
    if global_fit is None:
        global_fit = (0.0, float(train_df["target_soh_pct"].mean()))

    preds = []
    for _, row in test_df.iterrows():
        sub = train_df[train_df["cell_id"] == row["cell_id"]]
        local = fit(sub)
        slope, intercept = local if local is not None else global_fit
        preds.append(slope * float(row["target_position"]) + intercept)
    return np.asarray(preds)


def lgb_predict(X_train, y_train, X_val, y_val, X_test, task_name: str):
    model = bt.train_lightgbm(
        X_train.to_numpy(), y_train, X_val.to_numpy(), y_val, task_name=task_name
    )
    return model.predict(X_test.to_numpy()), model


def evaluate_model_predictions(name, y, pred, current_soh, split, chemistry, feature_set, extra=None):
    met = regression_metrics(y, pred, f"{chemistry} {name} {feature_set} {split}")
    p_met = regression_metrics(y, current_soh, f"{chemistry} persistence {split}")
    row = {
        "chemistry": chemistry,
        "feature_set": feature_set,
        "model": name,
        "split": split,
        **met,
        "persistence_mae_same_rows": p_met["mae"],
        "mae_delta_vs_persistence": (met["mae"] - p_met["mae"])
        if met["mae"] is not None and p_met["mae"] is not None
        else None,
        "beats_persistence_mae": bool(
            met["mae"] is not None and p_met["mae"] is not None and met["mae"] < p_met["mae"]
        ),
    }
    if extra:
        row.update(extra)
    return row


def run_feature_set(
    chemistry: str,
    feature_set: str,
    cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> list[dict]:
    rows = []
    X_train = feature_matrix(train, cols)
    X_val = feature_matrix(val, cols)
    X_test = feature_matrix(test, cols)
    y_train = train["target_soh_pct"].to_numpy(dtype=float)
    y_val = val["target_soh_pct"].to_numpy(dtype=float)
    y_test = test["target_soh_pct"].to_numpy(dtype=float)

    # Persistence
    for split_name, frame in (("val", val), ("test", test)):
        y = frame["target_soh_pct"].to_numpy(dtype=float)
        cur = frame["current_soh_pct"].to_numpy(dtype=float)
        rows.append(
            evaluate_model_predictions(
                "persistence", y, cur, cur, split_name, chemistry, feature_set
            )
        )

    # Persistence + training mean one-step delta (no feature use)
    mean_delta = float(np.mean(y_train - train["current_soh_pct"].to_numpy(dtype=float)))
    for split_name, frame in (("val", val), ("test", test)):
        cur = frame["current_soh_pct"].to_numpy(dtype=float)
        rows.append(
            evaluate_model_predictions(
                "mean_delta",
                frame["target_soh_pct"].to_numpy(dtype=float),
                cur + mean_delta,
                cur,
                split_name,
                chemistry,
                feature_set,
                {"train_mean_delta": mean_delta},
            )
        )

    # Linear trend
    for split_name, frame in (("val", val), ("test", test)):
        pred = trend_predict(train, frame)
        rows.append(
            evaluate_model_predictions(
                "linear_trend",
                frame["target_soh_pct"].to_numpy(dtype=float),
                pred,
                frame["current_soh_pct"].to_numpy(dtype=float),
                split_name,
                chemistry,
                feature_set,
            )
        )

    # Ridge (alpha selected on validation only)
    pred_test, ridge_info = ridge_predict(X_train, y_train, X_val, y_val, X_test)
    pred_val = None
    for alpha in [ridge_info["alpha"]]:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        pipe.fit(X_train, y_train)
        pred_val = pipe.predict(X_val)
    rows.append(
        evaluate_model_predictions(
            "ridge",
            y_val,
            pred_val,
            val["current_soh_pct"].to_numpy(dtype=float),
            "val",
            chemistry,
            feature_set,
            {"ridge_alpha": ridge_info["alpha"]},
        )
    )
    rows.append(
        evaluate_model_predictions(
            "ridge",
            y_test,
            pred_test,
            test["current_soh_pct"].to_numpy(dtype=float),
            "test",
            chemistry,
            feature_set,
            {"ridge_alpha": ridge_info["alpha"]},
        )
    )

    # LightGBM on absolute target
    pred_lgb_test, _ = lgb_predict(X_train, y_train, X_val, y_val, X_test, f"{chemistry}_{feature_set}")
    pred_lgb_val, _ = lgb_predict(X_train, y_train, X_val, y_val, X_val, f"{chemistry}_{feature_set}_val")
    rows.append(
        evaluate_model_predictions(
            "lightgbm",
            y_val,
            pred_lgb_val,
            val["current_soh_pct"].to_numpy(dtype=float),
            "val",
            chemistry,
            feature_set,
        )
    )
    rows.append(
        evaluate_model_predictions(
            "lightgbm",
            y_test,
            pred_lgb_test,
            test["current_soh_pct"].to_numpy(dtype=float),
            "test",
            chemistry,
            feature_set,
        )
    )

    # Delta LightGBM: target = target_soh - current_soh
    delta_train = y_train - train["current_soh_pct"].to_numpy(dtype=float)
    delta_val = y_val - val["current_soh_pct"].to_numpy(dtype=float)
    pred_delta_test, _ = lgb_predict(
        X_train, delta_train, X_val, delta_val, X_test, f"{chemistry}_{feature_set}_delta"
    )
    pred_delta_val, _ = lgb_predict(
        X_train, delta_train, X_val, delta_val, X_val, f"{chemistry}_{feature_set}_delta_val"
    )
    rows.append(
        evaluate_model_predictions(
            "delta_lightgbm",
            y_val,
            val["current_soh_pct"].to_numpy(dtype=float) + pred_delta_val,
            val["current_soh_pct"].to_numpy(dtype=float),
            "val",
            chemistry,
            feature_set,
        )
    )
    rows.append(
        evaluate_model_predictions(
            "delta_lightgbm",
            y_test,
            test["current_soh_pct"].to_numpy(dtype=float) + pred_delta_test,
            test["current_soh_pct"].to_numpy(dtype=float),
            "test",
            chemistry,
            feature_set,
        )
    )
    return rows


def build_future_split(cycle: pd.DataFrame, dataset_filter) -> pd.DataFrame:
    frames = []
    for _, cell_data in cycle[dataset_filter].groupby("cell_id"):
        fut = bt.create_future_soh_dataset(cell_data, horizon=1)
        if len(fut) > 0:
            frames.append(fut)
    future = pd.concat(frames, ignore_index=True)
    future["target_position"] = future["cycle_position"] + 1
    splits = []
    for _, cell_data in future.groupby("cell_id"):
        splits.append(bt.chronological_split(cell_data))
    return pd.concat(splits, ignore_index=True)


def split_frames(df: pd.DataFrame):
    return (
        df[df["split"] == "train"].copy(),
        df[df["split"] == "val"].copy(),
        df[df["split"] == "test"].copy(),
    )


def run_rul(cycle: pd.DataFrame, nmc_features: list[str]) -> tuple[list[dict], dict]:
    rows = []
    nmc = cycle[cycle["chemistry"] == "NMC"].copy()
    rul_frames = []
    for _, cell_data in nmc.groupby("cell_id"):
        rul_frames.append(bt.compute_rul_labels(cell_data.copy()))
    rul = pd.concat(rul_frames, ignore_index=True)
    # Production RUL training sets current_soh_pct = soh_pct (observed SOH at prediction time)
    rul["current_soh_pct"] = rul["soh_pct"]

    # Production chronological split composition (what the existing RUL test uses)
    prod_split = []
    for _, cell_data in rul.groupby("cell_id"):
        prod_split.append(bt.chronological_split(cell_data))
    prod_split = pd.concat(prod_split, ignore_index=True)
    prod_test = prod_split[prod_split["split"] == "test"]
    composition = {
        "production_rul_test_n": int(len(prod_test)),
        "production_rul_test_post_eol_n": int((prod_test["rul_cycles"] == 0).sum()),
        "production_rul_test_pre_eol_n": int((prod_test["rul_cycles"] > 0).sum()),
        "production_rul_test_rul_min": float(prod_test["rul_cycles"].min()),
        "production_rul_test_rul_max": float(prod_test["rul_cycles"].max()),
    }

    pre = rul[rul["rul_cycles"] > 0].copy()
    cells = sorted(pre["cell_id"].unique())
    pooled_y, pooled_med, pooled_slope, pooled_lgb = [], [], [], []

    for held in cells:
        train = pre[pre["cell_id"] != held]
        test = pre[pre["cell_id"] == held]
        if len(train) < 10 or len(test) < 5:
            continue

        y_tr = train["rul_cycles"].to_numpy(dtype=float)
        y_te = test["rul_cycles"].to_numpy(dtype=float)

        # Naive baseline 1: constant median training RUL
        pred_med = np.full(len(test), float(np.median(y_tr)))

        # Naive baseline 2: SOH-threshold extrapolation using train median slope
        slopes = []
        for _, cdf in train.groupby("cell_id"):
            if len(cdf) >= 3:
                slopes.append(np.polyfit(cdf["cycle_position"], cdf["soh_pct"], 1)[0])
        slope = float(np.median(slopes)) if slopes else 0.0
        if slope < 0:
            pred_slope = np.maximum((test["soh_pct"].to_numpy(dtype=float) - EOL) / (-slope), 0.0)
        else:
            pred_slope = pred_med.copy()

        # Small LightGBM, fixed parameters (no test tuning)
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=5,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(feature_matrix(train, nmc_features), y_tr)
        pred_lgb = model.predict(feature_matrix(test, nmc_features))

        for name, pred in (
            ("constant_median", pred_med),
            ("soh_threshold_extrapolation", pred_slope),
            ("lightgbm_pre_eol", pred_lgb),
        ):
            met = regression_metrics(y_te, pred, f"RUL {name} heldout={held}")
            rows.append(
                {
                    "heldout_cell": held,
                    "model": name,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_rul_min": float(y_tr.min()),
                    "train_rul_max": float(y_tr.max()),
                    "test_rul_min": float(y_te.min()),
                    "test_rul_max": float(y_te.max()),
                    **met,
                }
            )
        pooled_y.append(y_te)
        pooled_med.append(pred_med)
        pooled_slope.append(pred_slope)
        pooled_lgb.append(pred_lgb)

    pooled = {}
    if pooled_y:
        y_all = np.concatenate(pooled_y)
        for name, preds in (
            ("constant_median", pooled_med),
            ("soh_threshold_extrapolation", pooled_slope),
            ("lightgbm_pre_eol", pooled_lgb),
        ):
            pooled[name] = regression_metrics(y_all, np.concatenate(preds), f"RUL {name} pooled")
    return rows, {"production_composition": composition, "pooled_pre_eol": pooled}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    nmc_features = list(schema["models"]["nmc_lgbm"]["feature_names"])
    lfp_features = list(schema["models"]["lfp_lgbm"]["feature_names"])
    lfp_safe = ef.unit_safe_lfp_features(lfp_features)

    print("[1/5] Loading and building production features...")
    raw = pd.read_parquet(PARQUET)
    raw = raw[raw["dataset_id"] != "LFP_MultiStat_B1"].copy()
    prod_cycle = bt.build_cycle_feature_table(raw)
    prod_cycle = prod_cycle.dropna(subset=["soh_pct"]).copy()
    prod_cycle = prod_cycle.sort_values(bt.CYCLE_KEY).reset_index(drop=True)
    prod_cycle["cycle_position"] = prod_cycle.groupby("cell_id").cumcount()

    print("[2/5] Computing cleared experimental features...")
    extras = ef.compute_cycle_extra_features(
        raw[raw["chemistry"].isin(["NMC", "LFP"])].copy()
    )
    cycle = ef.merge_extra_features(prod_cycle, extras)
    cycle = ef.add_cumulative_features(cycle)

    # Persist the isolated cycle table for audit (experimental only).
    cycle.to_parquet(RESULTS_DIR / "cycle_features_experimental.parquet", index=False)

    print("[3/5] Building future-SOH splits...")
    nmc_split = build_future_split(cycle, cycle["chemistry"] == "NMC")
    lfp_split = build_future_split(cycle, cycle["dataset_id"] == "LFP_MultiStat_B5")
    nmc_tr, nmc_va, nmc_te = split_frames(nmc_split)
    lfp_tr, lfp_va, lfp_te = split_frames(lfp_split)

    nmc_sets = {
        "prod": nmc_features,
        "prod_replace_ah": [c for c in nmc_features if c != "ah_throughput"] + ["ah_discharge_trapz"],
        "prod_plus_ah": nmc_features + ["ah_discharge_trapz"],
        "prod_plus_efc": nmc_features + ["efc_cumulative"],
        "prod_plus_thermal": nmc_features
        + [
            "thermal_hours_outside_20_30",
            "thermal_degree_hours_above_30",
            "thermal_degree_hours_below_20",
        ],
        "prod_plus_v_at_q": nmc_features + ["v_at_q20", "v_at_q50", "v_at_q80"],
        "prod_plus_all_cleared": nmc_features
        + [
            "ah_discharge_trapz",
            "efc_cumulative",
            "thermal_hours_outside_20_30",
            "thermal_degree_hours_above_30",
            "thermal_degree_hours_below_20",
            "v_at_q20",
            "v_at_q50",
            "v_at_q80",
        ],
    }
    lfp_sets = {"prod": lfp_features, "unit_safe": lfp_safe}

    print("[4/5] Running NMC/LFP models and ablations...")
    metrics_rows = []
    for feature_set, cols in nmc_sets.items():
        print(f"  NMC {feature_set} ({len(cols)} features)")
        metrics_rows.extend(run_feature_set("NMC", feature_set, cols, nmc_tr, nmc_va, nmc_te))
    for feature_set, cols in lfp_sets.items():
        print(f"  LFP {feature_set} ({len(cols)} features)")
        metrics_rows.extend(run_feature_set("LFP", feature_set, cols, lfp_tr, lfp_va, lfp_te))
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)

    # Selection uses validation MAE only across all model families and feature sets.
    # The test row for the selected candidate is reported once, without re-selection.
    selection = {}
    for chem in ("NMC", "LFP"):
        val_subset = metrics_df[
            (metrics_df["chemistry"] == chem)
            & (metrics_df["split"] == "val")
            & (metrics_df["model"] != "persistence")
        ]
        if not len(val_subset):
            continue
        best = val_subset.sort_values("mae").iloc[0]
        test_row = metrics_df[
            (metrics_df["chemistry"] == chem)
            & (metrics_df["split"] == "test")
            & (metrics_df["model"] == best["model"])
            & (metrics_df["feature_set"] == best["feature_set"])
        ].iloc[0]
        persistence_val = metrics_df[
            (metrics_df["chemistry"] == chem)
            & (metrics_df["split"] == "val")
            & (metrics_df["model"] == "persistence")
        ].iloc[0]
        persistence_test = metrics_df[
            (metrics_df["chemistry"] == chem)
            & (metrics_df["split"] == "test")
            & (metrics_df["model"] == "persistence")
        ].iloc[0]
        selection[chem] = {
            "selected_model": str(best["model"]),
            "selected_feature_set": str(best["feature_set"]),
            "selected_val_mae": float(best["mae"]),
            "persistence_val_mae": float(persistence_val["mae"]),
            "val_mae_improvement_pct": float(
                100.0 * (persistence_val["mae"] - best["mae"]) / persistence_val["mae"]
            ),
            "selected_test_mae": float(test_row["mae"]),
            "persistence_test_mae": float(persistence_test["mae"]),
            "test_mae_improvement_pct": float(
                100.0 * (persistence_test["mae"] - test_row["mae"]) / persistence_test["mae"]
            ),
            "selection_improves_val": bool(best["mae"] < persistence_val["mae"]),
            "selection_improves_test": bool(test_row["mae"] < persistence_test["mae"]),
        }

    # LFP gap-crossing diagnostic: train has 39->41 and 100->151, validation
    # has 200->401, and no test transition crosses a missing-ID gap.
    lfp_train_gap = lfp_tr[(lfp_tr["target_cycle_id"] - lfp_tr["cycle_id"]) > 1].copy()
    lfp_gap = lfp_va[(lfp_va["target_cycle_id"] - lfp_va["cycle_id"]) > 1].copy()
    lfp_adj_val = lfp_va[(lfp_va["target_cycle_id"] - lfp_va["cycle_id"]) == 1].copy()
    lfp_adj_test = lfp_te[(lfp_te["target_cycle_id"] - lfp_te["cycle_id"]) == 1].copy()
    gap_summary = {
        "lfp_train_gap_crossing_n": int(len(lfp_train_gap)),
        "lfp_train_gap_crossing_pairs": [
            {
                "cycle_id": int(r.cycle_id),
                "target_cycle_id": int(r.target_cycle_id),
                "current_soh": float(r.current_soh_pct),
                "target_soh": float(r.target_soh_pct),
                "delta_soh": float(r.target_soh_pct - r.current_soh_pct),
            }
            for r in lfp_train_gap.itertuples()
        ],
        "lfp_val_gap_crossing_n": int(len(lfp_gap)),
        "lfp_val_gap_crossing_pairs": [
            [int(r.cycle_id), int(r.target_cycle_id)]
            for r in lfp_gap.itertuples()
        ],
        "lfp_val_adjacent_n": int(len(lfp_adj_val)),
        "lfp_test_adjacent_n": int(len(lfp_adj_test)),
        "lfp_test_gap_crossing_n": int(len(lfp_te) - len(lfp_adj_test)),
    }
    if len(lfp_gap):
        gap_row = lfp_gap.iloc[0]
        gap_summary.update(
            {
                "lfp_gap_pair_cycle_id": int(gap_row["cycle_id"]),
                "lfp_gap_pair_target_cycle_id": int(gap_row["target_cycle_id"]),
                "lfp_gap_pair_current_soh": float(gap_row["current_soh_pct"]),
                "lfp_gap_pair_target_soh": float(gap_row["target_soh_pct"]),
                "lfp_gap_pair_persistence_abs_error": float(
                    abs(gap_row["target_soh_pct"] - gap_row["current_soh_pct"])
                ),
            }
        )

    print("[5/5] Running pre-EOL RUL leave-one-cell-out diagnostic...")
    rul_rows, rul_summary = run_rul(cycle, nmc_features)
    rul_df = pd.DataFrame(rul_rows)
    if len(rul_df):
        rul_df.to_csv(RESULTS_DIR / "rul_pre_eol_results.csv", index=False)

    results = {
        "created": dt.datetime.now().isoformat(),
        "parquet_sha256": sha256(PARQUET),
        "schema_sha256": sha256(SCHEMA),
        "command": "python member2_modeling/experiments/run_experiments.py",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": __import__("lightgbm").__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "data": {
            "nmc_pairs": int(len(nmc_split)),
            "nmc_train": int(len(nmc_tr)),
            "nmc_val": int(len(nmc_va)),
            "nmc_test": int(len(nmc_te)),
            "lfp_pairs": int(len(lfp_split)),
            "lfp_train": int(len(lfp_tr)),
            "lfp_val": int(len(lfp_va)),
            "lfp_test": int(len(lfp_te)),
        },
        "feature_sets": {
            "nmc": {k: len(v) for k, v in nmc_sets.items()},
            "lfp": {k: len(v) for k, v in lfp_sets.items()},
        },
        "lfp_adjacent_vs_gap": gap_summary,
        "selection_validation_only": selection,
        "rul": rul_summary,
        "metrics": metrics_rows,
        "feature_provenance": ef.FEATURE_PROVENANCE,
        "caveats": [
            "Persistence remains the reference; any improvement is only on the reported test rows.",
            "LFP current units are underdetermined; unit_safe removes all current-derived features.",
            "LFP test contains no gap-crossing pair; the 200->401 pair is in validation.",
            "RUL pre-EOL results are a three-cell leave-one-cell-out diagnostic, not a validated production model.",
            "No formula-generated SOH label was created; measured soh_pct remains the only target.",
        ],
    }
    (RESULTS_DIR / "experiment_results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"Wrote {RESULTS_DIR / 'experiment_results.json'}")
    print(json.dumps(selection, indent=2))
    print(json.dumps(rul_summary, indent=2))


if __name__ == "__main__":
    main()
