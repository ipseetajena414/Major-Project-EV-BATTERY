"""Cross-chemistry stress test: train without one chemistry, score it.

This is a diagnostic for chemistry generalization, not a model-selection or
production claim. The production models are those in ``model_registry.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from member1_pipeline import loader
from member2_modeling.data_integration.profiles import feature_columns_for_target
from member2_modeling.features import TabularFeaturePipeline, infer_feature_types

from .candidates import candidate_by_name
from .metrics import classification_report, regression_report
from .specs import TASKS, task_ids
from .training import _fit_candidate, _jsonable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "member2_modeling" / "artifacts" / "snapshot_ml"


def _holdout_split(frame: pd.DataFrame, entity_column: str, seed: int):
    groups = frame[entity_column].astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train, inner = next(splitter.split(frame, groups=groups))
    return train, inner


def cross_chemistry(task_id: str, *, seed: int = 42) -> dict[str, Any]:
    spec = TASKS[task_id]
    registry = json.loads((DEFAULT_ROOT / "model_registry.json").read_text(encoding="utf-8"))
    selected = registry["tasks"][task_id]["selected_model"]["name"]

    frame = loader.load_dataset(spec.dataset_id)
    frame = frame[frame[spec.target].notna()].reset_index(drop=True)
    features = feature_columns_for_target(spec.dataset_id, spec.target)
    numeric, categorical = infer_feature_types(frame, features)
    chemistry_values = sorted(
        frame[spec.chemistry_column].fillna("__MISSING__").astype(str).unique()
    )

    results: dict[str, Any] = {}
    for held in chemistry_values:
        mask = (
            frame[spec.chemistry_column].fillna("__MISSING__").astype(str) == held
        ).to_numpy()
        pool = frame.loc[~mask].reset_index(drop=True)
        test = frame.loc[mask].reset_index(drop=True)
        if len(test) < 20 or len(pool) < 200:
            results[held] = {"status": "insufficient_rows", "n": int(len(test))}
            continue
        train_idx, val_idx = _holdout_split(pool, spec.entity_column, seed)
        pipeline = TabularFeaturePipeline(
            numeric_features=numeric,
            categorical_features=categorical,
            target=spec.target,
            forbidden_columns=[spec.target],
        ).fit(pool.iloc[train_idx])

        if spec.task_type == "regression":
            y_pool = pd.to_numeric(pool[spec.target], errors="coerce").to_numpy(
                dtype="float64"
            )
            y_test = pd.to_numeric(test[spec.target], errors="coerce").to_numpy(
                dtype="float64"
            )
            classes: list[str] = []
        else:
            classes = sorted(frame[spec.target].astype(str).unique().tolist())
            mapping = {label: index for index, label in enumerate(classes)}
            y_pool = (
                pool[spec.target].astype(str).map(mapping).to_numpy(dtype=int)
            )
            y_test = (
                test[spec.target].astype(str).map(mapping).to_numpy(dtype=int)
            )
            if any(label not in mapping for label in test[spec.target].astype(str).unique()):
                results[held] = {"status": "unseen_class"}
                continue

        candidate = candidate_by_name(selected, spec.task_type, seed)
        fitted = _fit_candidate(
            candidate,
            pipeline.transform(pool.iloc[train_idx]),
            y_pool[train_idx],
            pipeline.transform(pool.iloc[val_idx]),
            y_pool[val_idx],
            spec.task_type,
        )
        estimator = fitted["estimator"]
        prediction = estimator.predict(pipeline.transform(test))

        if spec.task_type == "regression":
            metrics = regression_report(y_test, np.asarray(prediction, dtype="float64"))
            baseline = float(np.median(y_pool[train_idx]))
            metrics["train_median_baseline"] = regression_report(
                y_test, np.full(len(y_test), baseline)
            )
        else:
            predicted_codes = np.asarray(prediction, dtype=int)
            metrics = classification_report(
                np.asarray([classes[code] for code in y_test]),
                np.asarray([classes[code] for code in predicted_codes]),
                labels=classes,
            )
            majority = pd.Series(
                [classes[code] for code in y_pool[train_idx]]
            ).value_counts().idxmax()
            metrics["majority_baseline"] = classification_report(
                np.asarray([classes[code] for code in y_test]),
                np.full(len(y_test), majority),
                labels=classes,
            )
        metrics.update(
            {
                "status": "diagnostic",
                "n_train": int(len(train_idx)),
                "n_validation": int(len(val_idx)),
                "n_heldout": int(len(test)),
                "selected_model": selected,
                "held_out_chemistry": held,
            }
        )
        results[held] = metrics

    return {
        "task_id": task_id,
        "dataset_id": spec.dataset_id,
        "target": spec.target,
        "task_type": spec.task_type,
        "selected_model": selected,
        "purpose": (
            "Train on all other chemistries and score the excluded chemistry. "
            "Diagnostic only; not used to select the production model."
        ),
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=task_ids(), choices=task_ids())
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT / "cross_chemistry.json")
    args = parser.parse_args(argv)
    payload = {}
    for task_id in args.tasks:
        print(f"Cross-chemistry {task_id}...")
        payload[task_id] = cross_chemistry(task_id)
    args.output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
