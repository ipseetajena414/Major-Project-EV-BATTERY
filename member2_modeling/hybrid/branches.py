"""Branch implementations for the fused hybrid SOH model.

The tabular branch is a LightGBM regressor. The sequence branch is a small
bidirectional LSTM that consumes causal windows of the same feature columns.
Both branches predict the same target so a fusion layer can combine them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fusion import HybridError

DEFAULT_TABULAR_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
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

DEFAULT_SEQUENCE_PARAMS = {
    "hidden_dim": 32,
    "num_layers": 1,
    "dropout": 0.1,
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "patience": 8,
}


class TabularBranch:
    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        merged = dict(DEFAULT_TABULAR_PARAMS)
        if params:
            merged.update({str(key): value for key, value in params.items()})
        self.params = merged
        self.model: Any = None
        self.best_iteration: int | None = None
        self.n_training_rows = 0

    def fit(
        self,
        features: Any,
        target: Any,
        eval_features: Any | None = None,
        eval_target: Any | None = None,
    ) -> "TabularBranch":
        import lightgbm as lgb

        X = np.asarray(features, dtype="float64")
        y = np.asarray(target, dtype="float64").reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.size:
            raise HybridError("tabular features and target shapes do not align")
        if y.size == 0:
            raise HybridError("tabular branch requires at least one training row")
        model = lgb.LGBMRegressor(**self.params)
        fit_kwargs: dict[str, Any] = {}
        callbacks = []
        if eval_features is not None and eval_target is not None:
            X_eval = np.asarray(eval_features, dtype="float64")
            y_eval = np.asarray(eval_target, dtype="float64").reshape(-1)
            if X_eval.ndim == 2 and X_eval.shape[0] == y_eval.size and y_eval.size >= 2:
                fit_kwargs["eval_X"] = X_eval
                fit_kwargs["eval_y"] = y_eval
                rounds = min(20, max(1, int(self.params.get("n_estimators", 200)) // 10))
                callbacks = [
                    lgb.early_stopping(stopping_rounds=rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
        model.fit(X, y, callbacks=callbacks or None, **fit_kwargs)
        self.model = model
        self.best_iteration = getattr(model, "best_iteration_", None)
        self.n_training_rows = int(y.size)
        return self

    def predict(self, features: Any) -> np.ndarray:
        if self.model is None:
            raise HybridError("TabularBranch must be fitted before predict().")
        X = np.asarray(features, dtype="float64")
        if X.size == 0:
            return np.zeros(0, dtype="float64")
        return np.asarray(self.model.predict(X), dtype="float64").reshape(-1)

    def describe(self) -> dict[str, Any]:
        return {
            "model": "lightgbm_regressor",
            "fitted": self.model is not None,
            "n_training_rows": self.n_training_rows,
            "best_iteration": self.best_iteration,
            "n_estimators": int(self.params.get("n_estimators", 0)),
        }


def _build_torch_regressor(input_dim: int, params: Mapping[str, Any]):
    import torch.nn as nn

    hidden_dim = int(params["hidden_dim"])
    num_layers = int(params["num_layers"])
    dropout = float(params["dropout"])

    class _BiLSTMRegressor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1),
            )

        def forward(self, values):
            output, _ = self.lstm(values)
            return self.head(output[:, -1, :]).squeeze(-1)

    return _BiLSTMRegressor()


class SequenceBranch:
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        if input_dim < 1:
            raise HybridError("input_dim must be >= 1")
        if sequence_length < 1:
            raise HybridError("sequence_length must be >= 1")
        merged = dict(DEFAULT_SEQUENCE_PARAMS)
        if params:
            merged.update({str(key): value for key, value in params.items()})
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.params = merged
        self._model: Any = None
        self._device = "cpu"
        self.n_training_sequences = 0
        self.epochs_run = 0

    def fit(
        self,
        train_sequences: Any,
        train_targets: Any,
        val_sequences: Any | None = None,
        val_targets: Any | None = None,
    ) -> "SequenceBranch":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(train_sequences, dtype="float32")
        y = np.asarray(train_targets, dtype="float32").reshape(-1)
        if X.ndim != 3 or X.shape[0] != y.size or X.shape[2] != self.input_dim:
            raise HybridError("sequence branch input shapes do not align with input_dim")
        if y.size < 2:
            raise HybridError("sequence branch requires at least two training sequences")

        if val_sequences is None or val_targets is None:
            split = max(1, int(round(y.size * 0.85)))
            split = min(split, y.size - 1)
            X_val = torch.from_numpy(X[split:])
            y_val = torch.from_numpy(y[split:])
            X_train = torch.from_numpy(X[:split])
            y_train = torch.from_numpy(y[:split])
        else:
            X_val = torch.from_numpy(np.asarray(val_sequences, dtype="float32"))
            y_val = torch.from_numpy(np.asarray(val_targets, dtype="float32").reshape(-1))
            X_train = torch.from_numpy(X)
            y_train = torch.from_numpy(y)

        seed = int(self.params.get("random_state", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = _build_torch_regressor(self.input_dim, self.params).to(self._device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(self.params["learning_rate"]),
            weight_decay=float(self.params["weight_decay"]),
        )
        criterion = nn.MSELoss()
        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=int(self.params["batch_size"]),
            shuffle=True,
        )
        best_loss = float("inf")
        best_state = None
        patience_counter = 0
        epochs = int(self.params["epochs"])
        patience = int(self.params["patience"])

        for epoch in range(epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(criterion(model(X_val), y_val).item())
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.epochs_run = epoch + 1
                    break
            self.epochs_run = epoch + 1

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self._model = model
        self.n_training_sequences = int(y.size)
        return self

    def predict(self, sequences: Any) -> np.ndarray:
        import torch

        if self._model is None:
            raise HybridError("SequenceBranch must be fitted before predict().")
        X = np.asarray(sequences, dtype="float32")
        if X.size == 0:
            return np.zeros(0, dtype="float64")
        if X.ndim != 3 or X.shape[2] != self.input_dim:
            raise HybridError("sequence input does not match the fitted input_dim")
        self._model.eval()
        with torch.no_grad():
            output = self._model(torch.from_numpy(X).to(self._device))
        return output.detach().cpu().numpy().astype("float64").reshape(-1)

    def save(self, path: Path) -> None:
        import torch

        if self._model is None:
            raise HybridError("cannot save an unfitted SequenceBranch")
        payload = {
            "state_dict": self._model.state_dict(),
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "params": self.params,
            "n_training_sequences": self.n_training_sequences,
            "epochs_run": self.epochs_run,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path) -> "SequenceBranch":
        import torch

        payload = torch.load(path, map_location="cpu")
        branch = cls(
            input_dim=int(payload["input_dim"]),
            sequence_length=int(payload["sequence_length"]),
            params=payload.get("params"),
        )
        model = _build_torch_regressor(branch.input_dim, branch.params)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        branch._model = model
        branch.n_training_sequences = int(payload.get("n_training_sequences", 0))
        branch.epochs_run = int(payload.get("epochs_run", 0))
        return branch

    def describe(self) -> dict[str, Any]:
        return {
            "model": "bilstm_regressor",
            "fitted": self._model is not None,
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "n_training_sequences": self.n_training_sequences,
            "epochs_run": self.epochs_run,
            "hidden_dim": int(self.params["hidden_dim"]),
        }
