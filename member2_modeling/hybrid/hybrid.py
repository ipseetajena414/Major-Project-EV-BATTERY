"""Fused LightGBM + BiLSTM SOH model.

Both branches predict the same next-record SOH target. The fusion regressor is
trained on chronological out-of-fold predictions, so its inputs match what it
sees at inference. Rows without enough history still receive a tabular-only
prediction through the fusion fallback; they are never silently dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from member2_modeling.features import SequenceFeatureBuilder

from .branches import SequenceBranch, TabularBranch
from .fusion import HybridError, OutOfFoldFusion

SCHEMA_VERSION = "2.0.0"
READINESS_STATUS = "EXPERIMENTAL"


class HybridSOHModel:
    def __init__(
        self,
        feature_columns: Iterable[str],
        target_column: str,
        entity_column: str,
        order_column: str,
        sequence_length: int = 5,
        n_folds: int = 3,
        random_state: int = 42,
        tabular_params: Mapping[str, Any] | None = None,
        sequence_params: Mapping[str, Any] | None = None,
        fusion_alpha: float = 1.0,
    ) -> None:
        self.feature_columns = list(feature_columns)
        if not self.feature_columns:
            raise HybridError("feature_columns must not be empty")
        if len(set(self.feature_columns)) != len(self.feature_columns):
            raise HybridError("feature_columns must be unique")
        self.target_column = str(target_column)
        self.entity_column = str(entity_column)
        self.order_column = str(order_column)
        forbidden = {self.target_column, self.entity_column, self.order_column}
        overlap = sorted(forbidden & set(self.feature_columns))
        if overlap:
            raise HybridError(
                f"Target or identifier columns cannot be sequence features: {overlap}"
            )
        if int(sequence_length) < 1:
            raise HybridError("sequence_length must be >= 1")
        if int(n_folds) < 1:
            raise HybridError("n_folds must be >= 1")
        self.sequence_length = int(sequence_length)
        self.n_folds = int(n_folds)
        self.random_state = int(random_state)
        self.tabular_params = dict(tabular_params or {})
        self.sequence_params = dict(sequence_params or {})
        self.tabular_params.setdefault("random_state", self.random_state)
        self.sequence_params.setdefault("random_state", self.random_state)
        self.fusion_alpha = float(fusion_alpha)

        self._tabular_branch: TabularBranch | None = None
        self._sequence_branch: SequenceBranch | None = None
        self._sequence_builder: SequenceFeatureBuilder | None = None
        self._fusion: OutOfFoldFusion | None = None
        self._fitted = False
        self._n_training_rows = 0
        self._dropped_null_target = 0
        self._effective_folds = 0
        self._sequence_oof_coverage = 0.0

    # ── data helpers ─────────────────────────────────────────────────────

    def _validate_frame(self, frame: pd.DataFrame, require_target: bool) -> None:
        required = [*self.feature_columns, self.entity_column, self.order_column]
        if require_target:
            required.append(self.target_column)
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise HybridError(f"Missing hybrid-model columns: {missing}")

    def _assign_folds(self, frame: pd.DataFrame) -> np.ndarray:
        folds = np.full(len(frame), -1, dtype=int)
        ordered = frame.sort_values(
            [self.entity_column, self.order_column], kind="mergesort"
        )
        effective = 0
        for _, group in ordered.groupby(self.entity_column, sort=True):
            entity_folds = min(self.n_folds, len(group))
            effective = max(effective, entity_folds)
            for position, index in enumerate(group.index):
                folds[index] = min(entity_folds - 1, int(position * entity_folds / len(group)))
        if (folds < 0).any():
            raise HybridError("fold assignment left rows unassigned")
        self._effective_folds = int(effective or 1)
        return folds

    def _chronological_tail(self, frame: pd.DataFrame, fraction: float = 0.15) -> pd.Index:
        parts = []
        for _, group in frame.sort_values(
            [self.entity_column, self.order_column], kind="mergesort"
        ).groupby(self.entity_column, sort=True):
            count = max(1, int(round(len(group) * fraction)))
            parts.append(group.tail(count).index)
        if not parts:
            return pd.Index([])
        return parts[0].append(parts[1:]) if len(parts) > 1 else parts[0]

    def _split_train_eval(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        if len(frame) < 6:
            return frame, None
        tail_index = self._chronological_tail(frame)
        eval_frame = frame.loc[tail_index]
        train_frame = frame.drop(index=tail_index)
        if train_frame.empty or eval_frame.empty:
            return frame, None
        return train_frame, eval_frame

    def _new_sequence_builder(self) -> SequenceFeatureBuilder:
        return SequenceFeatureBuilder(
            feature_columns=self.feature_columns,
            target_column=self.target_column,
            entity_column=self.entity_column,
            order_column=self.order_column,
            sequence_length=self.sequence_length,
        )

    @staticmethod
    def _key(entity: Any, order: Any) -> tuple[str, int]:
        return str(entity), int(order)

    def _sequence_row_positions(self, frame: pd.DataFrame, meta: pd.DataFrame) -> list[int]:
        lookup: dict[tuple[str, int], int] = {}
        for position, (entity, order) in enumerate(
            zip(frame[self.entity_column], frame[self.order_column])
        ):
            key = self._key(entity, order)
            if key in lookup:
                raise HybridError(f"Duplicate entity/order pair in prediction frame: {key}")
            lookup[key] = position
        positions = []
        for entity, order in zip(meta["entity"], meta["target_cycle_id"]):
            key = self._key(entity, order)
            if key not in lookup:
                raise HybridError(f"Sequence window does not align with a frame row: {key}")
            positions.append(lookup[key])
        return positions

    # ── training ─────────────────────────────────────────────────────────

    def fit(self, frame: pd.DataFrame) -> "HybridSOHModel":
        self._validate_frame(frame, require_target=True)
        data = frame.reset_index(drop=True).copy()
        self._dropped_null_target = int(data[self.target_column].isna().sum())
        data = data[data[self.target_column].notna()].reset_index(drop=True)
        if len(data) < self.sequence_length + 2:
            raise HybridError(
                "Not enough labelled rows to fit the hybrid model with "
                f"sequence_length={self.sequence_length}"
            )
        self._n_training_rows = int(len(data))

        folds = self._assign_folds(data)
        tabular_oof = np.full(len(data), np.nan, dtype="float64")
        sequence_oof = np.full(len(data), np.nan, dtype="float64")

        for fold in range(self._effective_folds):
            train_frame = data.loc[folds != fold].reset_index(drop=True)
            holdout_frame = data.loc[folds == fold].reset_index(drop=True)
            if train_frame.empty or holdout_frame.empty:
                continue

            tab_train, tab_eval = self._split_train_eval(train_frame)
            tabular = TabularBranch(self.tabular_params).fit(
                tab_train[self.feature_columns].to_numpy(dtype="float64"),
                tab_train[self.target_column].to_numpy(dtype="float64"),
                None if tab_eval is None else tab_eval[self.feature_columns].to_numpy(dtype="float64"),
                None if tab_eval is None else tab_eval[self.target_column].to_numpy(dtype="float64"),
            )
            tabular_oof[folds == fold] = tabular.predict(
                holdout_frame[self.feature_columns].to_numpy(dtype="float64")
            )

            builder = self._new_sequence_builder().fit(train_frame)
            X_train, y_train, _ = builder.transform(train_frame)
            if X_train.shape[0] < 2:
                continue
            branch = SequenceBranch(
                input_dim=len(self.feature_columns),
                sequence_length=self.sequence_length,
                params=self.sequence_params,
            ).fit(X_train, y_train)

            X_holdout, _, meta_holdout = builder.transform(holdout_frame)
            if X_holdout.shape[0] == 0:
                continue
            predictions = branch.predict(X_holdout)
            positions = self._sequence_row_positions(holdout_frame, meta_holdout)
            hold_index = np.flatnonzero(folds == fold)
            sequence_oof[hold_index[np.asarray(positions, dtype=int)]] = predictions

        if not np.isfinite(tabular_oof).all():
            raise HybridError("out-of-fold tabular predictions do not cover every row")
        available = np.isfinite(sequence_oof)
        self._sequence_oof_coverage = float(available.mean())
        self._fusion = OutOfFoldFusion(alpha=self.fusion_alpha).fit(
            tabular_oof, sequence_oof, available, data[self.target_column].to_numpy()
        )

        tab_train, tab_eval = self._split_train_eval(data)
        self._tabular_branch = TabularBranch(self.tabular_params).fit(
            tab_train[self.feature_columns].to_numpy(dtype="float64"),
            tab_train[self.target_column].to_numpy(dtype="float64"),
            None if tab_eval is None else tab_eval[self.feature_columns].to_numpy(dtype="float64"),
            None if tab_eval is None else tab_eval[self.target_column].to_numpy(dtype="float64"),
        )

        self._sequence_builder = self._new_sequence_builder().fit(data)
        X_sequences, y_sequences, _ = self._sequence_builder.transform(data)
        if X_sequences.shape[0] < 2:
            raise HybridError("not enough sequence windows to fit the BiLSTM branch")
        if len(y_sequences) >= 6:
            split = len(y_sequences) - max(1, int(round(len(y_sequences) * 0.15)))
            X_seq_train, y_seq_train = X_sequences[:split], y_sequences[:split]
            X_seq_eval, y_seq_eval = X_sequences[split:], y_sequences[split:]
        else:
            X_seq_train, y_seq_train = X_sequences, y_sequences
            X_seq_eval, y_seq_eval = None, None
        self._sequence_branch = SequenceBranch(
            input_dim=len(self.feature_columns),
            sequence_length=self.sequence_length,
            params=self.sequence_params,
        ).fit(X_seq_train, y_seq_train, X_seq_eval, y_seq_eval)
        self._fitted = True
        return self

    # ── inference ────────────────────────────────────────────────────────

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise HybridError("HybridSOHModel must be fitted before predict().")
        self._validate_frame(frame, require_target=False)
        data = frame.reset_index(drop=True)
        if data.empty:
            return pd.DataFrame(
                columns=[
                    self.entity_column,
                    self.order_column,
                    "tabular_pred",
                    "sequence_pred",
                    "fused_pred",
                    "sequence_available",
                ]
            )

        tabular = self._tabular_branch.predict(
            data[self.feature_columns].to_numpy(dtype="float64")
        )
        sequence = np.full(len(data), np.nan, dtype="float64")
        X_sequences, _, meta = self._sequence_builder.transform(
            data, include_missing_target=True
        )
        if X_sequences.shape[0]:
            positions = self._sequence_row_positions(data, meta)
            sequence[np.asarray(positions, dtype=int)] = self._sequence_branch.predict(
                X_sequences
            )
        available = np.isfinite(sequence)
        fused = self._fusion.predict(tabular, sequence, available)

        output = pd.DataFrame(
            {
                self.entity_column: data[self.entity_column].to_numpy(),
                self.order_column: data[self.order_column].to_numpy(),
                "tabular_pred": tabular,
                "sequence_pred": sequence,
                "fused_pred": fused,
                "sequence_available": available,
            }
        )
        if self.target_column in data.columns:
            output[self.target_column] = data[self.target_column].to_numpy()
        return output

    # ── persistence ──────────────────────────────────────────────────────

    def describe(self) -> dict[str, Any]:
        if not self._fitted:
            raise HybridError("HybridSOHModel must be fitted before describe().")
        return {
            "schema_version": SCHEMA_VERSION,
            "model_type": "fused_lightgbm_bilstm",
            "readiness_status": READINESS_STATUS,
            "production_ready": False,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "entity_column": self.entity_column,
            "order_column": self.order_column,
            "sequence_length": self.sequence_length,
            "configured_folds": self.n_folds,
            "effective_folds": self._effective_folds,
            "n_training_rows": self._n_training_rows,
            "dropped_null_target": self._dropped_null_target,
            "sequence_oof_coverage": self._sequence_oof_coverage,
            "fusion": self._fusion.describe(),
            "branches": {
                "tabular": self._tabular_branch.describe(),
                "sequence": self._sequence_branch.describe(),
            },
            "notes": [
                "Fusion inputs are chronological out-of-fold branch predictions.",
                "Rows without sequence history use the learned tabular-only fallback.",
                "No production validation exists; do not claim production readiness.",
            ],
        }

    def save(self, directory: Path | str) -> Path:
        if not self._fitted:
            raise HybridError("HybridSOHModel must be fitted before save().")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        config = {
            "schema_version": SCHEMA_VERSION,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "entity_column": self.entity_column,
            "order_column": self.order_column,
            "sequence_length": self.sequence_length,
            "n_folds": self.n_folds,
            "random_state": self.random_state,
            "tabular_params": self.tabular_params,
            "sequence_params": self.sequence_params,
            "fusion_alpha": self.fusion_alpha,
            "readiness_status": READINESS_STATUS,
            "n_training_rows": self._n_training_rows,
            "dropped_null_target": self._dropped_null_target,
            "effective_folds": self._effective_folds,
            "sequence_oof_coverage": self._sequence_oof_coverage,
        }
        (destination / "hybrid_config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        joblib.dump(self._tabular_branch, destination / "tabular_branch.joblib")
        joblib.dump(self._sequence_builder, destination / "sequence_builder.joblib")
        joblib.dump(self._fusion, destination / "fusion.joblib")
        self._sequence_branch.save(destination / "sequence_branch.pt")
        (destination / "hybrid_schema.json").write_text(
            json.dumps(self.describe(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, directory: Path | str) -> "HybridSOHModel":
        source = Path(directory)
        config_path = source / "hybrid_config.json"
        if not config_path.is_file():
            raise HybridError(f"Hybrid config not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = cls(
            feature_columns=config["feature_columns"],
            target_column=config["target_column"],
            entity_column=config["entity_column"],
            order_column=config["order_column"],
            sequence_length=int(config["sequence_length"]),
            n_folds=int(config["n_folds"]),
            random_state=int(config.get("random_state", 42)),
            tabular_params=config.get("tabular_params"),
            sequence_params=config.get("sequence_params"),
            fusion_alpha=float(config.get("fusion_alpha", 1.0)),
        )
        model._tabular_branch = joblib.load(source / "tabular_branch.joblib")
        model._sequence_builder = joblib.load(source / "sequence_builder.joblib")
        model._fusion = joblib.load(source / "fusion.joblib")
        model._sequence_branch = SequenceBranch.load(source / "sequence_branch.pt")
        model._n_training_rows = int(config.get("n_training_rows", 0))
        model._dropped_null_target = int(config.get("dropped_null_target", 0))
        model._effective_folds = int(config.get("effective_folds", 0))
        model._sequence_oof_coverage = float(config.get("sequence_oof_coverage", 0.0))
        model._fitted = True
        return model


__all__ = ["HybridError", "HybridSOHModel", "READINESS_STATUS", "SCHEMA_VERSION"]
