"""Tabular health (SOH / capacity) models for the Member 1 snapshot datasets.

These datasets contain independent snapshot records, not chronological
histories. The split is therefore a deterministic stratified random split,
documented as an i.i.d. snapshot estimate rather than a temporal forecast.
All imputation and category vocabularies are fitted on the training slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from member2_modeling.features import TabularFeaturePipeline, infer_feature_types
from member2_modeling.hybrid.evaluation import regression_metrics

from .capacity import HealthModelError

SCHEMA_VERSION = "2.0.0"
MIN_ROWS = 30

DEFAULT_MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "verbose": -1,
}


def _strata(values: np.ndarray) -> np.ndarray | None:
    if values.size < 20:
        return None
    quantiles = min(10, max(2, values.size // 40))
    try:
        bins = pd.qcut(pd.Series(values), q=quantiles, duplicates="drop")
    except (ValueError, TypeError):
        return None
    if bins.nunique() < 2:
        return None
    if bins.value_counts().min() < 2:
        return None
    return bins.astype(str).to_numpy()


class TabularHealthModel:
    def __init__(
        self,
        target_column: str,
        feature_columns: Iterable[str],
        *,
        chemistry_column: str | None = None,
        forbidden_columns: Iterable[str] = (),
        model_params: Mapping[str, Any] | None = None,
        random_state: int = 42,
        test_fraction: float = 0.2,
        val_fraction: float = 0.2,
        readiness: str = "EXPERIMENTAL",
        target_kind: str = "measured",
    ) -> None:
        self.target_column = str(target_column)
        self.feature_columns = list(feature_columns)
        if not self.feature_columns:
            raise HealthModelError("feature_columns must not be empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise HealthModelError("feature_columns must be unique")
        self.chemistry_column = chemistry_column
        self.forbidden_columns = set(forbidden_columns)
        if self.target_column in self.feature_columns:
            raise HealthModelError("target column cannot be a feature")
        unsafe = sorted(self.forbidden_columns & set(self.feature_columns))
        if unsafe:
            raise HealthModelError(f"Leakage columns cannot be features: {unsafe}")
        merged = dict(DEFAULT_MODEL_PARAMS)
        if model_params:
            merged.update({str(key): value for key, value in model_params.items()})
        merged["random_state"] = int(random_state)
        self.model_params = merged
        self.random_state = int(random_state)
        self.test_fraction = float(test_fraction)
        self.val_fraction = float(val_fraction)
        self.readiness = str(readiness)
        self.target_kind = str(target_kind)

        self._pipeline: TabularFeaturePipeline | None = None
        self._model: Any = None
        self._fitted = False
        self._split_indices: dict[str, np.ndarray] = {}
        self._evaluation: dict[str, Any] = {}
        self._dropped_null_target = 0
        self._n_rows = 0

    def _validate_frame(self, frame: pd.DataFrame, require_target: bool) -> None:
        required = [*self.feature_columns]
        if require_target:
            required.append(self.target_column)
        if self.chemistry_column:
            required.append(self.chemistry_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise HealthModelError(f"Missing health-model columns: {missing}")

    def fit(self, frame: pd.DataFrame) -> "TabularHealthModel":
        self._validate_frame(frame, require_target=True)
        data = frame.reset_index(drop=True).copy()
        self._dropped_null_target = int(data[self.target_column].isna().sum())
        data = data[data[self.target_column].notna()].reset_index(drop=True)
        if len(data) < MIN_ROWS:
            raise HealthModelError(
                f"At least {MIN_ROWS} labelled rows are required, got {len(data)}"
            )
        self._n_rows = int(len(data))

        target_values = pd.to_numeric(data[self.target_column], errors="coerce").to_numpy(
            dtype="float64"
        )
        if not np.isfinite(target_values).all():
            raise HealthModelError("target must be finite numeric values")
        pool, test = train_test_split(
            np.arange(len(data)),
            test_size=self.test_fraction,
            random_state=self.random_state,
            stratify=_strata(target_values),
        )
        pool_target = target_values[pool]
        train, val = train_test_split(
            pool,
            test_size=self.val_fraction / (1.0 - self.test_fraction),
            random_state=self.random_state,
            stratify=_strata(pool_target),
        )
        self._split_indices = {
            "train": np.sort(train),
            "val": np.sort(val),
            "test": np.sort(test),
        }

        train_frame = data.iloc[train]
        numeric, categorical = infer_feature_types(data, self.feature_columns)
        self._pipeline = TabularFeaturePipeline(
            numeric_features=numeric,
            categorical_features=categorical,
            target=self.target_column,
            forbidden_columns=self.forbidden_columns,
        ).fit(train_frame)
        X_train = self._pipeline.transform(train_frame)
        y_train = target_values[train]
        X_val = self._pipeline.transform(data.iloc[val])
        y_val = target_values[val]
        X_test = self._pipeline.transform(data.iloc[test])
        y_test = target_values[test]

        import lightgbm as lgb

        model = lgb.LGBMRegressor(**self.model_params)
        stopping = min(30, max(5, int(self.model_params["n_estimators"]) // 10))
        model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            callbacks=[
                lgb.early_stopping(stopping_rounds=stopping, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self._model = model

        predictions = model.predict(X_test)
        train_median = float(np.median(y_train))
        median_predictions = np.full(y_test.size, train_median)
        chemistry_predictions = None
        if self.chemistry_column is not None:
            chemistry_predictions = (
                data.iloc[test][self.chemistry_column]
                .map(
                    data.iloc[train]
                    .groupby(self.chemistry_column)[self.target_column]
                    .median()
                )
                .fillna(train_median)
                .to_numpy(dtype="float64")
            )

        evaluation: dict[str, Any] = {
            "split_protocol": "deterministic_stratified_snapshot_split",
            "split_counts": {key: int(value.size) for key, value in self._split_indices.items()},
            "counts": {
                "rows": int(len(data)),
                "dropped_null_target": self._dropped_null_target,
                "test": int(y_test.size),
            },
            "model": regression_metrics(y_test, predictions),
            "baselines": {
                "train_median": regression_metrics(y_test, median_predictions),
                "train_chemistry_median": (
                    None
                    if chemistry_predictions is None
                    else regression_metrics(y_test, chemistry_predictions)
                ),
            },
            "caveats": [
                "Snapshot records: this random split estimates i.i.d. generalization, "
                "not a temporal or vehicle-level forecast.",
                "Early stopping uses a validation slice of the training split only.",
            ],
        }
        evaluation["outperforms_train_median"] = bool(
            evaluation["model"]["mae"] is not None
            and evaluation["model"]["mae"]
            < evaluation["baselines"]["train_median"]["mae"]
        )
        chemistry_baseline = evaluation["baselines"]["train_chemistry_median"]
        evaluation["outperforms_chemistry_median"] = (
            None
            if chemistry_baseline is None or chemistry_baseline["mae"] is None
            else bool(evaluation["model"]["mae"] < chemistry_baseline["mae"])
        )
        if self.chemistry_column is not None:
            per_chemistry: dict[str, Any] = {}
            test_chemistry = data.iloc[test][self.chemistry_column]
            for value in sorted(
                test_chemistry.dropna().astype(str).unique().tolist()
            ):
                mask = test_chemistry.astype(str).to_numpy() == value
                if mask.sum() == 0:
                    continue
                per_chemistry[value] = regression_metrics(
                    y_test[mask], predictions[mask]
                )
            evaluation["per_chemistry"] = per_chemistry
        self._evaluation = evaluation
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise HealthModelError("TabularHealthModel must be fitted before predict().")
        self._validate_frame(frame, require_target=False)
        features = self._pipeline.transform(frame)
        return np.asarray(self._model.predict(features), dtype="float64").reshape(-1)

    def split_indices(self) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise HealthModelError("TabularHealthModel must be fitted before split_indices().")
        return {key: value.copy() for key, value in self._split_indices.items()}

    def describe(self) -> dict[str, Any]:
        if not self._fitted:
            raise HealthModelError("TabularHealthModel must be fitted before describe().")
        schema = self._pipeline.to_schema()
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": "tabular_lightgbm_regressor",
            "target_column": self.target_column,
            "target_kind": self.target_kind,
            "feature_columns": list(self.feature_columns),
            "chemistry_column": self.chemistry_column,
            "readiness": self.readiness,
            "production_ready": False,
            "n_rows": self._n_rows,
            "dropped_null_target": self._dropped_null_target,
            "numeric_medians": schema["numeric_medians"],
            "numeric_features": schema["numeric_features"],
            "categorical_features": schema["categorical_features"],
            "feature_names": schema["feature_names"],
            "evaluation": self._evaluation,
        }

    def save(self, directory: Path | str) -> Path:
        if not self._fitted:
            raise HealthModelError("TabularHealthModel must be fitted before save().")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, destination / "feature_pipeline.joblib")
        joblib.dump(self._model, destination / "model.joblib")
        config = {
            "schema_version": SCHEMA_VERSION,
            "target_column": self.target_column,
            "feature_columns": list(self.feature_columns),
            "chemistry_column": self.chemistry_column,
            "forbidden_columns": sorted(self.forbidden_columns),
            "model_params": self.model_params,
            "random_state": self.random_state,
            "test_fraction": self.test_fraction,
            "val_fraction": self.val_fraction,
            "readiness": self.readiness,
            "target_kind": self.target_kind,
            "n_rows": self._n_rows,
            "dropped_null_target": self._dropped_null_target,
            "split_indices": {key: value.tolist() for key, value in self._split_indices.items()},
        }
        (destination / "model_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "model_schema.json").write_text(
            json.dumps(self.describe(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, directory: Path | str) -> "TabularHealthModel":
        source = Path(directory)
        config_path = source / "model_config.json"
        if not config_path.is_file():
            raise HealthModelError(f"Health-model config not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = cls(
            target_column=config["target_column"],
            feature_columns=config["feature_columns"],
            chemistry_column=config.get("chemistry_column"),
            forbidden_columns=config.get("forbidden_columns", ()),
            model_params=config.get("model_params"),
            random_state=int(config.get("random_state", 42)),
            test_fraction=float(config.get("test_fraction", 0.2)),
            val_fraction=float(config.get("val_fraction", 0.2)),
            readiness=config.get("readiness", "EXPERIMENTAL"),
            target_kind=config.get("target_kind", "measured"),
        )
        model._pipeline = joblib.load(source / "feature_pipeline.joblib")
        model._model = joblib.load(source / "model.joblib")
        model._n_rows = int(config.get("n_rows", 0))
        model._dropped_null_target = int(config.get("dropped_null_target", 0))
        model._split_indices = {
            key: np.asarray(values, dtype=int)
            for key, values in config.get("split_indices", {}).items()
        }
        schema_path = source / "model_schema.json"
        if schema_path.is_file():
            model._evaluation = json.loads(schema_path.read_text(encoding="utf-8")).get(
                "evaluation", {}
            )
        model._fitted = True
        return model


__all__ = ["HealthModelError", "SCHEMA_VERSION", "TabularHealthModel"]
