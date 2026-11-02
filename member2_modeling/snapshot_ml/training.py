"""Train, select and evaluate one Member 2 task on a grouped split.

Selection uses validation data only. The frozen selected model is then scored
once on the untouched test split. No test value influences feature selection,
preprocessing, hyperparameters or model choice.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from member1_pipeline import loader
from member2_modeling.data_integration.profiles import (
    feature_columns_for_target,
    target_spec,
)
from member2_modeling.features import TabularFeaturePipeline, infer_feature_types

from .candidates import Candidate, classification_candidates, regression_candidates
from .metrics import classification_report, regression_report
from .specs import TaskSpec
from .splits import (
    GroupedSplit,
    class_distribution,
    distribution,
    entity_counts,
    make_grouped_split,
)

MODEL_VERSION = "2.0.0"
BOOTSTRAP_SAMPLES = 400


class TrainingError(RuntimeError):
    """Raised when a task cannot be trained defensibly."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is pd.NA:
        return None
    return value


def _bootstrap_regression(
    truth: np.ndarray, prediction: np.ndarray, *, seed: int
) -> dict[str, float] | None:
    if truth.size < 20:
        return None
    rng = np.random.default_rng(seed)
    errors = np.abs(truth - prediction)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype="float64")
    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, errors.size, errors.size)
        draws[index] = float(np.mean(errors[sample]))
    return {
        "mae_ci95_low": float(np.quantile(draws, 0.025)),
        "mae_ci95_high": float(np.quantile(draws, 0.975)),
    }


def _bootstrap_classification(
    truth: np.ndarray, prediction: np.ndarray, *, seed: int
) -> dict[str, float] | None:
    from sklearn.metrics import balanced_accuracy_score

    if truth.size < 20:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_SAMPLES, dtype="float64")
    for index in range(BOOTSTRAP_SAMPLES):
        sample = rng.integers(0, truth.size, truth.size)
        try:
            draws[index] = balanced_accuracy_score(truth[sample], prediction[sample])
        except ValueError:
            draws[index] = np.nan
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    return {
        "balanced_accuracy_ci95_low": float(np.quantile(draws, 0.025)),
        "balanced_accuracy_ci95_high": float(np.quantile(draws, 0.975)),
    }


def _fit_candidate(
    candidate: Candidate,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    task_type: str,
) -> dict[str, Any]:
    estimator = candidate.build()
    started = time.perf_counter()
    is_lightgbm = estimator.__class__.__module__.startswith("lightgbm")
    extra: dict[str, Any] = {}
    if is_lightgbm:
        import lightgbm as lgb

        if task_type == "regression":
            estimator.set_params(eval_metric="l2")
        elif task_type == "binary":
            estimator.set_params(eval_metric="binary_logloss")
        else:
            estimator.set_params(eval_metric="multi_logloss")
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        estimator.fit(X_train, y_train, eval_X=X_val, eval_y=y_val, callbacks=callbacks)
        extra["best_iteration"] = int(getattr(estimator, "best_iteration_", 0) or 0)
    else:
        estimator.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    if task_type == "regression":
        prediction = np.asarray(estimator.predict(X_val), dtype="float64").reshape(-1)
        score = float(np.mean(np.abs(np.asarray(y_val, dtype="float64") - prediction)))
        selection_metric = "validation_mae"
    else:
        encoded = np.asarray(estimator.predict(X_val), dtype=int).reshape(-1)
        from sklearn.metrics import balanced_accuracy_score

        score = float(balanced_accuracy_score(y_val, encoded))
        selection_metric = "validation_balanced_accuracy"
        extra["classes"] = [int(value) for value in estimator.classes_]
    return {
        "estimator": estimator,
        "candidate": candidate.describe(),
        "selection_metric": selection_metric,
        "selection_score": score,
        "fit_seconds": round(elapsed, 3),
        **extra,
    }


def _select(results: list[dict[str, Any]], task_type: str) -> dict[str, Any]:
    if task_type == "regression":
        return min(results, key=lambda row: row["selection_score"])
    return max(results, key=lambda row: row["selection_score"])


def _feature_importance(estimator: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    model = estimator
    if hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model", estimator)
    values: np.ndarray | None = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype="float64").reshape(-1)
    elif hasattr(model, "coef_"):
        coefficient = np.asarray(model.coef_, dtype="float64")
        values = np.mean(np.abs(coefficient), axis=0).reshape(-1)
    if values is None or values.size != len(feature_names):
        return []
    order = np.argsort(values)[::-1][:20]
    return [
        {"feature": feature_names[index], "importance": float(values[index])}
        for index in order
    ]


def _regression_baselines(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    chemistry_column: str,
    target_column: str,
) -> dict[str, Any]:
    median = float(np.median(y_train))
    mean = float(np.mean(y_train))
    chemistry_values = train_frame.groupby(chemistry_column)[target_column].median()
    chemistry_prediction = (
        test_frame[chemistry_column]
        .astype(str)
        .map(chemistry_values.astype(float))
        .fillna(median)
        .to_numpy(dtype="float64")
    )
    return {
        "train_median": regression_report(y_test, np.full(y_test.size, median)),
        "train_mean": regression_report(y_test, np.full(y_test.size, mean)),
        "train_chemistry_median": regression_report(y_test, chemistry_prediction),
    }


def _classification_baselines(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classes: list[str],
    chemistry_column: str,
) -> dict[str, Any]:
    majority = pd.Series(y_train).astype(str).value_counts().idxmax()
    majority_prediction = np.full(y_test.size, majority)
    chemistry_majority = (
        train_frame.assign(_y=y_train)
        .groupby(chemistry_column)["_y"]
        .agg(lambda values: values.astype(str).value_counts().idxmax())
    )
    chemistry_prediction = (
        test_frame[chemistry_column]
        .astype(str)
        .map(chemistry_majority)
        .fillna(majority)
        .to_numpy()
    )
    return {
        "majority_class": {
            "class": majority,
            **classification_report(y_test, majority_prediction, labels=classes),
        },
        "train_chemistry_majority": classification_report(
            y_test, chemistry_prediction, labels=classes
        ),
    }


def _per_chemistry(
    test_frame: pd.DataFrame,
    chemistry_column: str,
    y_test: np.ndarray,
    prediction: np.ndarray,
    *,
    task_type: str,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    values = test_frame[chemistry_column].fillna("__MISSING__").astype(str).to_numpy()
    for chemistry in sorted(set(values)):
        mask = values == chemistry
        if mask.sum() == 0:
            continue
        if task_type == "regression":
            result[chemistry] = regression_report(y_test[mask], prediction[mask])
        else:
            result[chemistry] = classification_report(
                y_test[mask], prediction[mask], labels=classes
            )
        result[chemistry]["n"] = int(mask.sum())
    return result


def _numeric_statistics(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype="float64")
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            stats[column] = None
            continue
        stats[column] = {
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=0)) if finite.size > 1 else 0.0,
            "min": float(finite.min()),
            "max": float(finite.max()),
        }
    return stats


def train_task(
    spec: TaskSpec,
    *,
    output_root: Path | str,
    random_state: int = 42,
) -> dict[str, Any]:
    frame = loader.load_dataset(spec.dataset_id)
    target_specification = target_spec(spec.dataset_id, spec.target)
    features = feature_columns_for_target(spec.dataset_id, spec.target)
    if not features:
        raise TrainingError(f"No leakage-safe features for {spec.dataset_id}/{spec.target}")

    data = frame.reset_index(drop=True).copy()
    missing_target = int(data[spec.target].isna().sum())
    data = data[data[spec.target].notna()].reset_index(drop=True)
    if len(data) < 50:
        raise TrainingError(
            f"{spec.task_id}: only {len(data)} labelled rows after dropping missing targets."
        )

    if spec.task_type == "regression":
        target_values = pd.to_numeric(data[spec.target], errors="coerce").to_numpy(
            dtype="float64"
        )
        if not np.isfinite(target_values).all():
            raise TrainingError(f"{spec.task_id}: target contains non-finite values.")
        classes: list[str] = []
        y = target_values
    else:
        raw = data[spec.target].astype(str)
        classes = sorted(raw.unique().tolist())
        if len(classes) < 2:
            raise TrainingError(f"{spec.task_id}: fewer than two classes.")
        mapping = {label: index for index, label in enumerate(classes)}
        y = raw.map(mapping).to_numpy(dtype=int)
        target_values = y.astype("float64")

    split = make_grouped_split(
        data,
        target_values,
        entity_column=spec.entity_column,
        task_type=spec.task_type,
        random_state=random_state,
    )

    numeric, categorical = infer_feature_types(data, features)
    pipeline = TabularFeaturePipeline(
        numeric_features=numeric,
        categorical_features=categorical,
        target=spec.target,
        forbidden_columns=[spec.target, *target_specification["leakage_columns"]],
    ).fit(data.iloc[split.train])

    X_train = pipeline.transform(data.iloc[split.train])
    X_val = pipeline.transform(data.iloc[split.val])
    X_test = pipeline.transform(data.iloc[split.test])
    y_train = y[split.train]
    y_val = y[split.val]
    y_test = y[split.test]
    feature_names = pipeline.feature_names_out()

    candidates = (
        regression_candidates(random_state)
        if spec.task_type == "regression"
        else classification_candidates(random_state)
    )
    results = []
    for candidate in candidates:
        row = _fit_candidate(candidate, X_train, y_train, X_val, y_val, spec.task_type)
        results.append(row)
        print(
            f"  candidate {row['candidate']['name']}: "
            f"{row['selection_metric']}={row['selection_score']:.4f} "
            f"({row['fit_seconds']}s)"
        )
    selected = _select(results, spec.task_type)
    estimator = selected["estimator"]

    if spec.task_type == "regression":
        test_prediction = np.asarray(estimator.predict(X_test), dtype="float64").reshape(-1)
        val_prediction = np.asarray(estimator.predict(X_val), dtype="float64").reshape(-1)
        residual = np.abs(np.asarray(y_val, dtype="float64") - val_prediction)
        residual_q90 = float(np.quantile(residual, 0.9)) if residual.size else None
        test_metrics = regression_report(y_test, test_prediction, residual_q90)
        test_metrics["bootstrap_ci"] = _bootstrap_regression(
            np.asarray(y_test, dtype="float64"), test_prediction, seed=random_state
        )
        baselines = _regression_baselines(
            data.iloc[split.train],
            data.iloc[split.test],
            np.asarray(y_train, dtype="float64"),
            np.asarray(y_test, dtype="float64"),
            spec.chemistry_column,
            spec.target,
        )
        prediction_label = test_prediction
        uncertainty = {
            "kind": "validation_residual_interval",
            "residual_q90": residual_q90,
            "residual_std": float(residual.std(ddof=0)) if residual.size else None,
            "calibration_status": "empirical_validation_residuals",
            "note": "Prediction interval is +/- residual_q90 around the point estimate.",
        }
    else:
        encoded_prediction = np.asarray(estimator.predict(X_test), dtype=int).reshape(-1)
        test_prediction = np.asarray([classes[index] for index in encoded_prediction])
        probabilities = np.asarray(estimator.predict_proba(X_test), dtype="float64")
        test_metrics = classification_report(
            np.asarray([classes[index] for index in y_test]),
            test_prediction,
            probabilities=probabilities,
            labels=classes,
        )
        test_metrics["bootstrap_ci"] = _bootstrap_classification(
            np.asarray([classes[index] for index in y_test]), test_prediction, seed=random_state
        )
        baselines = _classification_baselines(
            data.iloc[split.train],
            data.iloc[split.test],
            np.asarray([classes[index] for index in y_train]),
            np.asarray([classes[index] for index in y_test]),
            classes,
            spec.chemistry_column,
        )
        prediction_label = test_prediction
        uncertainty = {
            "kind": "class_probabilities",
            "classes": classes,
            "calibration_status": "uncalibrated_class_probabilities",
            "validation_log_loss": None,
        }
        val_encoded = np.asarray(estimator.predict(X_val), dtype=int).reshape(-1)
        from sklearn.metrics import log_loss

        try:
            uncertainty["validation_log_loss"] = float(
                log_loss(y_val, estimator.predict_proba(X_val), labels=list(range(len(classes))))
            )
        except ValueError:
            uncertainty["validation_log_loss"] = None
        uncertainty["validation_balanced_accuracy"] = float(
            np.mean(
                [
                    np.mean(val_encoded[y_val == index] == index)
                    for index in range(len(classes))
                    if np.any(y_val == index)
                ]
            )
        )

    observed_label = (
        np.asarray(y_test, dtype="float64")
        if spec.task_type == "regression"
        else np.asarray([classes[index] for index in y_test])
    )

    artifact_dir = Path(output_root) / spec.task_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / "feature_pipeline.joblib")
    joblib.dump(estimator, artifact_dir / "model.joblib")

    prediction_columns = [spec.entity_column, spec.chemistry_column, _quality_column(data)]
    if "source_row_id" in data.columns:
        prediction_columns.insert(0, "source_row_id")
    test_predictions = data.iloc[split.test][prediction_columns].copy()
    if "source_row_id" in test_predictions.columns:
        test_predictions = test_predictions.rename(
            columns={"source_row_id": "snapshot_row_id"}
        )
    test_predictions["observed"] = observed_label
    test_predictions["prediction"] = prediction_label
    test_predictions["split"] = "test"
    if spec.task_type == "regression":
        if residual_q90 is not None:
            test_predictions["interval_90_low"] = test_prediction - residual_q90
            test_predictions["interval_90_high"] = test_prediction + residual_q90
    else:
        for index, label in enumerate(classes):
            test_predictions[f"probability_{label}"] = probabilities[:, index]
    test_predictions.to_parquet(artifact_dir / "test_predictions.parquet", index=False)

    split_summary = {
        "protocol": split.protocol,
        "strata": split.strata,
        "warning": split.warning,
        "counts": split.counts(),
        "entity_counts": entity_counts(data, split, spec.entity_column),
        "entity_overlap": _overlap(data, split, spec.entity_column),
        "sha256": split.digest(data, spec.entity_column),
        "target_distribution": (
            distribution(data, split, spec.target)
            if spec.task_type == "regression"
            else class_distribution(data, split, spec.target)
        ),
    }

    candidate_report = []
    for row in results:
        entry = {
            "name": row["candidate"]["name"],
            "params": row["candidate"]["params"],
            "scale_features": row["candidate"]["scale_features"],
            "selection_metric": row["selection_metric"],
            "selection_score": row["selection_score"],
            "fit_seconds": row["fit_seconds"],
        }
        if "best_iteration" in row:
            entry["best_iteration"] = row["best_iteration"]
        candidate_report.append(entry)

    dataset_entry = loader.load_registry()["active_datasets"][spec.dataset_id]
    config = {
        "model_version": MODEL_VERSION,
        "task": spec.to_dict(),
        "feature_columns": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "forbidden_columns": sorted(
            {spec.target, *target_specification["leakage_columns"]}
        ),
        "random_state": random_state,
        "split": split_summary,
        "candidate_selection": candidate_report,
        "selected_model": selected["candidate"],
        "selection_rule": (
            "minimum validation MAE"
            if spec.task_type == "regression"
            else "maximum validation balanced accuracy"
        ),
        "final_fit": "trained on the training split only; validation used for early stopping/selection",
        "dataset_provenance": {
            "source_file": dataset_entry["source_file"],
            "source_sha256": dataset_entry["source_sha256"],
            "processed_path": dataset_entry["processed_path"],
            "processed_sha256": dataset_entry["processed_sha256"],
            "data_origin": dataset_entry["data_origin"],
            "record_grain": dataset_entry["record_grain"],
            "sequence_available": False,
        },
        "rows": {
            "total": int(len(data)),
            "dropped_null_target": missing_target,
            "train": int(len(split.train)),
            "val": int(len(split.val)),
            "test": int(len(split.test)),
        },
        "chemistry_column": spec.chemistry_column,
        "entity_column": spec.entity_column,
        "classes": classes,
        "numeric_statistics": _numeric_statistics(data.iloc[split.train], numeric),
        "categorical_vocabularies": pipeline.to_schema()["categorical_vocabularies"],
        "target": {
            "column": spec.target,
            "unit": spec.unit,
            "label_origin": spec.label_origin,
            "definition": target_specification["definition"],
        },
        "status": spec.status,
        "production_ready": False,
        "limitations": list(spec.notes)
        + [
            "Snapshot records do not support temporal, vehicle-level or RUL claims.",
            "Grouped holdout estimates generalization to unseen entities, not to a future time.",
        ],
    }
    (artifact_dir / "config.json").write_text(
        json.dumps(_jsonable(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    metrics = {
        "task_id": spec.task_id,
        "dataset_id": spec.dataset_id,
        "target": spec.target,
        "task_type": spec.task_type,
        "status": spec.status,
        "selected_model": selected["candidate"],
        "test": test_metrics,
        "baselines": baselines,
        "per_chemistry": _per_chemistry(
            data.iloc[split.test],
            spec.chemistry_column,
            np.asarray(
                y_test if spec.task_type == "regression" else [classes[i] for i in y_test]
            ),
            prediction_label,
            task_type=spec.task_type,
            classes=classes if spec.task_type != "regression" else None,
        ),
        "uncertainty": uncertainty,
        "feature_importance": _feature_importance(estimator, feature_names),
        "caveats": [
            "Metrics are computed on a grouped holdout of unseen entities.",
            "No test row was used for selection or preprocessing.",
        ]
        + (
            ["The synthetic dataset is generated, not measured."]
            if spec.dataset_id == "SYNTHETIC_BATTERY_LIFECYCLE"
            else []
        ),
    }
    (artifact_dir / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    schema = pipeline.to_schema()
    schema.update(
        {
            "task_id": spec.task_id,
            "model_version": MODEL_VERSION,
            "feature_columns": features,
            "target": spec.target,
            "unit": spec.unit,
            "entity_column": spec.entity_column,
            "chemistry_column": spec.chemistry_column,
            "classes": classes,
            "numeric_statistics": config["numeric_statistics"],
        }
    )
    (artifact_dir / "schema.json").write_text(
        json.dumps(_jsonable(schema), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "task_id": spec.task_id,
        "artifact_dir": str(artifact_dir),
        "config": config,
        "metrics": metrics,
    }


def _quality_column(frame: pd.DataFrame) -> str:
    return "quality_flag" if "quality_flag" in frame.columns else "source_row_id"


def _overlap(
    frame: pd.DataFrame, split: GroupedSplit, entity_column: str
) -> dict[str, int]:
    train = set(frame.iloc[split.train][entity_column].astype(str))
    val = set(frame.iloc[split.val][entity_column].astype(str))
    test = set(frame.iloc[split.test][entity_column].astype(str))
    return {
        "train_val": len(train & val),
        "train_test": len(train & test),
        "val_test": len(val & test),
    }


__all__ = ["MODEL_VERSION", "TrainingError", "train_task"]
