"""Train and evaluate the fused hybrid model on eligible sequential data.

Training input is the legacy NASA NMC master table, used explicitly as
labelled legacy research input. The three Member 1 processed datasets are
snapshot-only and are not trained on here; their eligibility is recorded.
Existing Member 2 model artifacts are never overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member2_modeling import build_and_train as bt  # noqa: E402
from member2_modeling.data_integration.profiles import DATASET_IDS  # noqa: E402
from member2_modeling.hybrid import (  # noqa: E402
    LEGACY_NMC_DATASET_ID,
    HybridSOHModel,
    hybrid_eligibility,
)
from member2_modeling.hybrid.evaluation import (  # noqa: E402
    mean_fade_baseline,
    persistence_baseline,
    regression_metrics,
)

PARQUET = REPO_ROOT / "master_table.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "member2_modeling" / "artifacts" / "hybrid_nmc_legacy"

FEATURE_COLUMNS = [
    "current_soh_pct",
    "v_mean",
    "v_min",
    "v_end",
    "v_slope",
    "i_mean",
    "ah_throughput",
    "energy_wh",
    "discharge_time_s",
    "temp_mean",
    "temp_range",
]
TARGET_COLUMN = "target_soh_pct"
ENTITY_COLUMN = "cell_id"
ORDER_COLUMN = "cycle_id"
SEQUENCE_LENGTH = 5
N_FOLDS = 3

MODEL_PARAMS = {
    "tabular": {
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 10,
        "random_state": 42,
    },
    "sequence": {
        "hidden_dim": 32,
        "num_layers": 1,
        "dropout": 0.1,
        "epochs": 60,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "patience": 10,
        "random_state": 42,
    },
    "fusion_alpha": 1.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def load_legacy_nmc_pairs() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_parquet(PARQUET)
    nmc_raw = raw[raw["chemistry"] == "NMC"].copy()
    if nmc_raw.empty:
        raise RuntimeError("legacy master table contains no NMC rows")
    cycles = bt.build_cycle_feature_table(nmc_raw)
    cycles = cycles.dropna(subset=["soh_pct"]).copy()
    cycles = cycles.sort_values([ENTITY_COLUMN, ORDER_COLUMN], kind="mergesort")
    cycles["cycle_position"] = cycles.groupby(ENTITY_COLUMN).cumcount()

    frames = []
    for cell_id, group in cycles.groupby(ENTITY_COLUMN, sort=True):
        pairs = bt.create_future_soh_dataset(group, horizon=1)
        if not pairs.empty:
            frames.append(pairs)
    pairs = pd.concat(frames, ignore_index=True)
    pairs["row_id"] = np.arange(len(pairs), dtype=int)

    metadata = {
        "source_file": str(PARQUET.relative_to(REPO_ROOT)).replace("\\", "/"),
        "source_sha256": sha256(PARQUET),
        "cells": sorted(str(cell) for cell in pairs[ENTITY_COLUMN].unique()),
        "cycles_per_cell": {
            str(cell): int(count)
            for cell, count in cycles.groupby(ENTITY_COLUMN).size().items()
        },
        "model_ready_pairs": int(len(pairs)),
        "target": (
            "SOH of the next observed cycle record (positional shift by 1); "
            "not necessarily the next physical cycle."
        ),
    }
    return pairs.reset_index(drop=True), metadata


def split_chronologically(pairs: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in pairs.groupby(ENTITY_COLUMN, sort=True):
        frames.append(bt.chronological_split(group))
    return pd.concat(frames, ignore_index=True)


def evaluate_split(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    train_frame: pd.DataFrame,
) -> dict[str, Any]:
    truth = frame[TARGET_COLUMN].to_numpy(dtype="float64")
    fused = predictions["fused_pred"].to_numpy()
    tabular = predictions["tabular_pred"].to_numpy()
    sequence = predictions["sequence_pred"].to_numpy()
    available = predictions["sequence_available"].to_numpy(dtype=bool)

    persistence = persistence_baseline(frame)
    mean_fade = mean_fade_baseline(
        train_frame,
        frame,
        entity_column=ENTITY_COLUMN,
        order_column=ORDER_COLUMN,
        current_column="current_soh_pct",
        target_column=TARGET_COLUMN,
    )
    result: dict[str, Any] = {
        "counts": {
            "rows": int(len(frame)),
            "sequence_available": int(available.sum()),
            "sequence_missing": int((~available).sum()),
        },
        "fused_all_rows": regression_metrics(truth, fused),
        "tabular_all_rows": regression_metrics(truth, tabular),
        "persistence_all_rows": regression_metrics(truth, persistence),
        "mean_fade_all_rows": regression_metrics(truth, mean_fade),
    }
    if available.any():
        result["fused_sequence_rows"] = regression_metrics(truth[available], fused[available])
        result["sequence_sequence_rows"] = regression_metrics(
            truth[available], sequence[available]
        )
        result["tabular_sequence_rows"] = regression_metrics(
            truth[available], tabular[available]
        )
        result["persistence_sequence_rows"] = regression_metrics(
            truth[available], persistence[available]
        )
    return result


def comparison(result: dict[str, Any], model_key: str, baseline_key: str) -> bool | None:
    model_mae = result.get(model_key, {}).get("mae")
    baseline_mae = result.get(baseline_key, {}).get("mae")
    if model_mae is None or baseline_mae is None:
        return None
    return bool(model_mae < baseline_mae)


def build_report(
    merged: pd.DataFrame,
    train_frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    evaluation: dict[str, Any] = {}
    for split in ("val", "test"):
        mask = merged["split"] == split
        frame = merged.loc[mask].reset_index(drop=True)
        evaluation[split] = evaluate_split(frame, frame, train_frame)
        evaluation[split]["fusion_beats_persistence_all_rows"] = comparison(
            evaluation[split], "fused_all_rows", "persistence_all_rows"
        )
        evaluation[split]["fusion_beats_tabular_all_rows"] = comparison(
            evaluation[split], "fused_all_rows", "tabular_all_rows"
        )
        if "fused_sequence_rows" in evaluation[split]:
            evaluation[split]["fusion_beats_persistence_sequence_rows"] = comparison(
                evaluation[split], "fused_sequence_rows", "persistence_sequence_rows"
            )
            evaluation[split]["fusion_beats_sequence_sequence_rows"] = comparison(
                evaluation[split], "fused_sequence_rows", "sequence_sequence_rows"
            )

    train_mask = merged["split"] == "train"
    train_frame_eval = merged.loc[train_mask].reset_index(drop=True)
    evaluation["train_in_sample_diagnostic"] = {
        "note": (
            "In-sample branch outputs; the fusion layer itself was trained on "
            "out-of-fold predictions. Not a generalisation estimate."
        ),
        "counts": {"rows": int(train_mask.sum())},
        "fused_all_rows": regression_metrics(
            train_frame_eval[TARGET_COLUMN], train_frame_eval["fused_pred"]
        ),
    }
    return {
        "dataset_id": LEGACY_NMC_DATASET_ID,
        "target": metadata["target"],
        "evaluation": evaluation,
        "readiness": {
            "status": "EXPERIMENTAL",
            "production_ready": False,
            "fusion_trained_on_out_of_fold_predictions": True,
            "survives_sequence_window_rows": True,
            "persistence_outperforms_fusion": not bool(
                evaluation["test"].get("fusion_beats_persistence_all_rows")
            ),
            "mean_fade_outperforms_fusion": not bool(
                comparison(
                    evaluation["test"], "fused_all_rows", "mean_fade_all_rows"
                )
            ),
            "benchmark_warning": (
                "The persistence and training-mean-fade baselines outperform the "
                "fused hybrid on the chronological test split. The fusion improves "
                "on both learned branches but is not production-validated."
            ),
            "notes": [
                "Only the NMC legacy sequential table is eligible for sequence training.",
                "The fusion model is not a production-validated predictor.",
                "Reported test rows are chronological holdout rows per cell.",
            ],
        },
    }


def markdown_report(report: dict[str, Any], metadata: dict[str, Any]) -> str:
    lines = [
        "# Member 2 Hybrid Training Report — Legacy NMC",
        "",
        f"Source: `{metadata['source_file']}`  ",
        f"SHA-256: `{metadata['source_sha256']}`  ",
        f"Cells: {', '.join(metadata['cells'])}  ",
        f"Model-ready pairs: {metadata['model_ready_pairs']}  ",
        f"Target: {metadata['target']}",
        "",
        "## Evaluation",
        "",
        "| Split | Rows | Seq avail. | Fusion MAE | Tabular MAE | Persistence MAE | Mean-fade MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    for split in ("val", "test"):
        section = report["evaluation"][split]
        counts = section["counts"]
        lines.append(
            f"| {split} | {counts['rows']} | {counts['sequence_available']} | "
            f"{fmt(section['fused_all_rows']['mae'])} | "
            f"{fmt(section['tabular_all_rows']['mae'])} | "
            f"{fmt(section['persistence_all_rows']['mae'])} | "
            f"{fmt(section['mean_fade_all_rows']['mae'])} |"
        )
    lines.extend(
        [
            "",
            "## Fusion usage",
            "",
            "- Fusion inputs are chronological out-of-fold branch predictions.",
            "- Rows without enough history use the learned tabular-only fallback.",
            "- The model is EXPERIMENTAL and not production-validated.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output: Path = args.output

    eligibility = {
        dataset_id: hybrid_eligibility(dataset_id) for dataset_id in DATASET_IDS
    }
    eligibility[LEGACY_NMC_DATASET_ID] = hybrid_eligibility(LEGACY_NMC_DATASET_ID)
    if not eligibility[LEGACY_NMC_DATASET_ID]["sequence_available"]:
        raise RuntimeError("legacy NMC data is unexpectedly ineligible for hybrid training")

    print("[1/5] Loading legacy NMC sequential pairs...")
    pairs, metadata = load_legacy_nmc_pairs()
    split_data = split_chronologically(pairs)
    split_counts = {
        str(split): int((split_data["split"] == split).sum())
        for split in ("train", "val", "test")
    }
    metadata["split_counts"] = split_counts
    train_frame = split_data[split_data["split"] == "train"].reset_index(drop=True)
    if train_frame.empty:
        raise RuntimeError("chronological split produced an empty training set")

    print("[2/5] Fitting fused LightGBM + BiLSTM model...")
    model = HybridSOHModel(
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        entity_column=ENTITY_COLUMN,
        order_column=ORDER_COLUMN,
        sequence_length=SEQUENCE_LENGTH,
        n_folds=N_FOLDS,
        random_state=42,
        tabular_params=MODEL_PARAMS["tabular"],
        sequence_params=MODEL_PARAMS["sequence"],
        fusion_alpha=MODEL_PARAMS["fusion_alpha"],
    ).fit(train_frame)

    print("[3/5] Predicting chronological holdout rows...")
    predictions = model.predict(pairs)
    prediction_columns = predictions[
        ["tabular_pred", "sequence_pred", "fused_pred", "sequence_available"]
    ].copy()
    prediction_columns["row_id"] = pairs["row_id"].to_numpy()
    merged = split_data[
        ["row_id", "split", ENTITY_COLUMN, ORDER_COLUMN, "current_soh_pct", TARGET_COLUMN]
    ].merge(prediction_columns, on="row_id", how="inner")
    if len(merged) != len(split_data):
        raise RuntimeError("prediction rows do not align with the chronological split")

    print("[4/5] Evaluating against persistence and mean-fade baselines...")
    report = build_report(merged, train_frame, metadata)
    report["model"] = model.describe()
    report["eligibility"] = eligibility
    report["provenance"] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "master_table_sha256": metadata["source_sha256"],
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQUENCE_LENGTH,
        "n_folds": N_FOLDS,
        "parameters": MODEL_PARAMS,
    }

    print("[5/5] Writing hybrid artifacts...")
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "hybrid_model"
    model.save(model_dir)
    (output / "hybrid_evaluation.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "hybrid_evaluation.md").write_text(
        markdown_report(jsonable(report), metadata), encoding="utf-8"
    )
    manifest = {
        "module": "member2_modeling/hybrid",
        "model_type": "fused_lightgbm_bilstm",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "training_dataset": metadata,
        "dataset_eligibility": eligibility,
        "feature_columns": FEATURE_COLUMNS,
        "sequence_length": SEQUENCE_LENGTH,
        "n_folds": N_FOLDS,
        "readiness_status": "EXPERIMENTAL",
        "production_ready": False,
        "benchmark_warning": report["readiness"]["benchmark_warning"],
        "artifacts": {
            "model_directory": "hybrid_model",
            "evaluation_json": "hybrid_evaluation.json",
            "evaluation_markdown": "hybrid_evaluation.md",
        },
        "protected_artifacts_untouched": [
            "member2_modeling/nmc_lgbm_model.joblib",
            "member2_modeling/lfp_lgbm_model.joblib",
            "member2_modeling/nmc_bilstm_model.pt",
            "member2_modeling/nmc_rul_lgbm_model.joblib",
            "member2_modeling/feature_schema.json",
            "member2_modeling/model_manifest.json",
            "member3/",
            "DATA/processed/",
        ],
    }
    (output / "hybrid_manifest.json").write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    test = report["evaluation"]["test"]
    print(
        "  test rows={rows} seq_available={seq} fused_MAE={fused} "
        "tabular_MAE={tabular} persistence_MAE={persistence}".format(
            rows=test["counts"]["rows"],
            seq=test["counts"]["sequence_available"],
            fused=test["fused_all_rows"]["mae"],
            tabular=test["tabular_all_rows"]["mae"],
            persistence=test["persistence_all_rows"]["mae"],
        )
    )
    print(f"  artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
