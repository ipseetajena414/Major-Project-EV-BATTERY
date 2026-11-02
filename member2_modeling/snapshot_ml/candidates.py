"""Candidate estimators compared on identical grouped splits.

Only estimators already available in the pinned Member 2 environment are used:
scikit-learn 1.9.1 and LightGBM 4.7.0. No new dependency is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class Candidate:
    name: str
    estimator: Any
    scale: bool = False
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        if self.scale:
            return Pipeline(
                [("scale", StandardScaler(with_mean=True, with_std=True)), ("model", self.estimator)]
            )
        return self.estimator

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "scale_features": self.scale, "params": self.params}


def regression_candidates(random_state: int = 42) -> list[Candidate]:
    return [
        Candidate(
            "ridge",
            Ridge(alpha=1.0),
            scale=True,
            params={"alpha": 1.0},
        ),
        Candidate(
            "elasticnet",
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, tol=1e-2),
            scale=True,
            params={"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 2000},
        ),
        Candidate(
            "random_forest",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                max_features=0.6,
                n_jobs=-1,
                random_state=random_state,
            ),
            params={
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": 0.6,
            },
        ),
        Candidate(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=random_state,
            ),
            params={"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31},
        ),
        Candidate(
            "lightgbm",
            _lightgbm_regressor(random_state),
            params={
                "n_estimators": 600,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
            },
        ),
    ]


def classification_candidates(random_state: int = 42) -> list[Candidate]:
    return [
        Candidate(
            "logistic_regression",
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                class_weight="balanced",
            ),
            scale=True,
            params={"C": 1.0, "class_weight": "balanced"},
        ),
        Candidate(
            "random_forest",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=2,
                max_features=0.6,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=random_state,
            ),
            params={
                "n_estimators": 400,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
            },
        ),
        Candidate(
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.1,
                early_stopping=True,
                validation_fraction=0.1,
                class_weight="balanced",
                random_state=random_state,
            ),
            params={"max_iter": 300, "learning_rate": 0.08, "max_leaf_nodes": 31},
        ),
        Candidate(
            "lightgbm",
            _lightgbm_classifier(random_state),
            params={
                "n_estimators": 600,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "class_weight": "balanced",
            },
        ),
    ]


def _lightgbm_regressor(random_state: int):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )


def _lightgbm_classifier(random_state: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_alpha=0.05,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )


def candidate_by_name(name: str, task_type: str, random_state: int = 42) -> Candidate:
    pool = (
        regression_candidates(random_state)
        if task_type == "regression"
        else classification_candidates(random_state)
    )
    for candidate in pool:
        if candidate.name == name:
            return candidate
    raise KeyError(
        f"Unknown candidate {name!r} for {task_type}; available: "
        f"{[candidate.name for candidate in pool]}"
    )


__all__ = [
    "Candidate",
    "candidate_by_name",
    "classification_candidates",
    "regression_candidates",
]
