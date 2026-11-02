"""Out-of-fold fusion of tabular and sequence branch predictions.

The fusion regressor is fitted on predictions that each branch made on rows it
was not trained on (out-of-fold stacking). This prevents the fusion layer from
learning to trust an in-sample prediction that cannot be reproduced at
inference time.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

FUSION_METHOD = "out_of_fold_ridge"
MIN_FUSION_ROWS = 5
SOH_CLIP = (0.0, 100.0)


class HybridError(ValueError):
    """Raised for invalid hybrid-model inputs or unfitted usage."""


def _as_float_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype="float64").reshape(-1)
    return array


class OutOfFoldFusion:
    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise HybridError("alpha must be non-negative")
        self.alpha = float(alpha)
        self._fallback: Ridge | None = None
        self._fusion: Ridge | None = None
        self._scaler: StandardScaler | None = None
        self._n_rows = 0
        self._n_sequence_rows = 0
        self._description: dict[str, Any] = {}

    def fit(
        self,
        tabular_pred: Any,
        sequence_pred: Any,
        sequence_available: Any,
        y: Any,
    ) -> "OutOfFoldFusion":
        tabular = _as_float_vector(tabular_pred, "tabular_pred")
        sequence = _as_float_vector(sequence_pred, "sequence_pred")
        available = np.asarray(sequence_available, dtype=bool).reshape(-1)
        target = _as_float_vector(y, "y")
        if not (tabular.size == sequence.size == available.size == target.size):
            raise HybridError(
                "tabular_pred, sequence_pred, sequence_available and y must have equal length"
            )
        finite_rows = np.isfinite(tabular) & np.isfinite(target)
        if not finite_rows.any():
            raise HybridError("fusion requires at least one finite tabular prediction and target")
        if not finite_rows.all():
            raise HybridError("fusion inputs must not contain non-finite tabular predictions or targets")

        tabular = tabular[finite_rows]
        sequence = sequence[finite_rows]
        available = available[finite_rows]
        target = target[finite_rows]

        self._n_rows = int(target.size)
        self._fallback = Ridge(alpha=self.alpha).fit(tabular.reshape(-1, 1), target)

        usable = available & np.isfinite(sequence)
        self._n_sequence_rows = int(usable.sum())
        if self._n_sequence_rows >= MIN_FUSION_ROWS and np.ptp(target[usable]) > 0:
            features = np.column_stack([tabular[usable], sequence[usable]])
            self._scaler = StandardScaler().fit(features)
            self._fusion = Ridge(alpha=self.alpha).fit(
                self._scaler.transform(features), target[usable]
            )
        else:
            self._scaler = None
            self._fusion = None

        self._description = self._build_description()
        return self

    def predict(
        self, tabular_pred: Any, sequence_pred: Any, sequence_available: Any
    ) -> np.ndarray:
        if self._fallback is None:
            raise HybridError("OutOfFoldFusion must be fitted before predict().")
        tabular = _as_float_vector(tabular_pred, "tabular_pred")
        sequence = _as_float_vector(sequence_pred, "sequence_pred")
        available = np.asarray(sequence_available, dtype=bool).reshape(-1)
        if not (tabular.size == sequence.size == available.size):
            raise HybridError(
                "tabular_pred, sequence_pred and sequence_available must have equal length"
            )
        finite = np.isfinite(tabular)
        if not finite.all():
            raise HybridError("tabular predictions must be finite")

        fused = self._fallback.predict(tabular.reshape(-1, 1))
        if self._fusion is not None and self._scaler is not None:
            mask = available & np.isfinite(sequence)
            if mask.any():
                features = np.column_stack([tabular[mask], sequence[mask]])
                fused[mask] = self._fusion.predict(self._scaler.transform(features))
        return np.clip(fused, SOH_CLIP[0], SOH_CLIP[1])

    def _build_description(self) -> dict[str, Any]:
        fallback_coefficients = {
            "intercept": float(self._fallback.intercept_) if self._fallback else None,
            "tabular_pred": float(self._fallback.coef_[0]) if self._fallback else None,
        }
        description: dict[str, Any] = {
            "fusion_method": FUSION_METHOD,
            "trained_on": "out_of_fold_predictions",
            "in_sample_fusion": False,
            "alpha": self.alpha,
            "uses_sequence": self._fusion is not None,
            "n_rows": self._n_rows,
            "n_sequence_rows": self._n_sequence_rows,
            "fallback_coefficients": fallback_coefficients,
        }
        if self._fusion is not None:
            description["coefficients"] = {
                "intercept": float(self._fusion.intercept_),
                "tabular_pred": float(self._fusion.coef_[0]),
                "sequence_pred": float(self._fusion.coef_[1]),
            }
        else:
            description["coefficients"] = {
                "intercept": fallback_coefficients["intercept"],
                "tabular_pred": fallback_coefficients["tabular_pred"],
                "sequence_pred": 0.0,
            }
        return description

    def describe(self) -> dict[str, Any]:
        if not self._description:
            raise HybridError("OutOfFoldFusion must be fitted before describe().")
        return dict(self._description)
