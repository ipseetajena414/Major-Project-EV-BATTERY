"""RUL estimation from genuine cycle histories.

Every candidate is trained and evaluated on cycle-level SOH only.  Future
telemetry is never invented: the autoregressive PADM variant and the direct
models use only information available at the prediction origin.  The
``master_table`` within-cycle telemetry is not reused here because future
voltage, temperature and current are unknown.

Units are *observed cycles*.  ``predicted_eol_cycle`` is computed in source
cycle-id space and therefore inherits the contiguity assumption documented in
the provenance.  Nothing in this module approves RUL for operational use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from member2_modeling.rul.labels import EOL_PCT

MODEL_VERSION = "1.0.0"
MAX_EXTRAPOLATION_CYCLES = 5000.0
MIN_HISTORY = 8
DEFAULT_ALPHA = 10.0


class RULForecastError(ValueError):
    """Raised when a history or model request cannot produce an honest RUL."""


@dataclass(frozen=True)
class RULHistory:
    cycle_ids: np.ndarray
    soh_pct: np.ndarray

    def __post_init__(self) -> None:
        ids = np.asarray(self.cycle_ids)
        soh = np.asarray(self.soh_pct, dtype="float64")
        if ids.ndim != 1 or soh.ndim != 1 or ids.size != soh.size:
            raise RULForecastError("cycle_ids and soh_pct must be equal-length 1-D arrays")
        if ids.size < 3:
            raise RULForecastError("At least 3 observed cycles are required")
        if not np.issubdtype(ids.dtype, np.integer):
            try:
                ids = ids.astype("int64")
            except (TypeError, ValueError) as exc:
                raise RULForecastError("cycle_ids must be integers") from exc
        if not np.isfinite(soh).all():
            raise RULForecastError("soh_pct must be finite")
        if (soh <= 0).any() or (soh > 110).any():
            raise RULForecastError("soh_pct must lie within (0, 110]")
        if (np.diff(ids) <= 0).any():
            raise RULForecastError("cycle_ids must be strictly increasing")
        object.__setattr__(self, "cycle_ids", ids.astype("int64"))
        object.__setattr__(self, "soh_pct", soh)

    @property
    def n(self) -> int:
        return int(self.soh_pct.size)

    @property
    def positions(self) -> np.ndarray:
        return np.arange(self.n, dtype="int64")

    @property
    def current_cycle(self) -> int:
        return int(self.cycle_ids[-1])

    @property
    def current_soh(self) -> float:
        return float(self.soh_pct[-1])

    def truncated(self, end_index: int) -> "RULHistory":
        return RULHistory(self.cycle_ids[: end_index + 1], self.soh_pct[: end_index + 1])


def make_history(cycle_ids: Iterable[Any], soh_pct: Iterable[Any]) -> RULHistory:
    try:
        ids = np.asarray(list(cycle_ids))
        soh = np.asarray(list(soh_pct), dtype="float64")
    except (TypeError, ValueError) as exc:
        raise RULForecastError("cycle_ids and soh_pct must be numeric sequences") from exc
    return RULHistory(ids, soh)


@dataclass(frozen=True)
class Origin:
    cell_id: str
    dataset_id: str
    chemistry: str
    history: RULHistory
    target_rul: float | None
    target_eol_cycle: int | None
    crossing_position: int | None
    in_training_cell: bool = True

    @property
    def origin_position(self) -> int:
        return self.history.n - 1


def make_origins(
    cell_id: str,
    dataset_id: str,
    chemistry: str,
    cycle_ids: Iterable[Any],
    soh_pct: Iterable[Any],
    *,
    eol_pct: float = EOL_PCT,
    step: int = 2,
    min_history: int = MIN_HISTORY,
) -> list[Origin]:
    """Build causal origins with RUL targets from a full observed trajectory."""
    history = make_history(cycle_ids, soh_pct)
    soh = history.soh_pct
    below = soh < float(eol_pct)
    crossing = int(np.argmax(below)) if bool(below.any()) else None
    if crossing is None:
        return []
    if crossing < min_history:
        return []
    eol_cycle = int(history.cycle_ids[crossing])
    origins: list[Origin] = []
    for index in range(min_history - 1, crossing, step):
        prefix = history.truncated(index)
        origins.append(
            Origin(
                cell_id=cell_id,
                dataset_id=dataset_id,
                chemistry=chemistry,
                history=prefix,
                target_rul=float(eol_cycle - int(prefix.cycle_ids[-1])),
                target_eol_cycle=eol_cycle,
                crossing_position=crossing,
            )
        )
    return origins


# ── feature engineering ─────────────────────────────────────────────────

SUMMARY_FEATURE_NAMES = (
    "current_soh_pct",
    "origin_position",
    "current_cycle_id",
    "total_fade_pct",
    "fade_rate_all",
    "fade_rate_20",
    "fade_rate_50",
    "curvature_20_50",
    "soh_std_20",
    "soh_last_delta",
    "log_history_length",
)

PADM_FEATURE_NAMES = (
    "current_soh_pct",
    "delta_lag_1",
    "delta_lag_2",
    "delta_mean_3",
    "delta_mean_10",
    "cycle_position_over_100",
)


def _slope(positions: np.ndarray, values: np.ndarray) -> float | None:
    if positions.size < 2:
        return None
    centered = positions.astype("float64") - float(positions.mean())
    denominator = float(np.sum(centered**2))
    if denominator <= 0:
        return None
    return float(np.sum(centered * (values - float(values.mean()))) / denominator)


def _window_slope(soh: np.ndarray, window: int) -> float | None:
    window = min(window, soh.size)
    if window < 2:
        return None
    positions = np.arange(soh.size - window, soh.size, dtype="float64")
    return _slope(positions, soh[-window:])


def _fade_rate(soh: np.ndarray, window: int | None = None) -> float:
    slope = _window_slope(soh, window or soh.size)
    if slope is None or not np.isfinite(slope):
        return 0.0
    return float(max(0.0, -slope))


def summary_features(history: RULHistory) -> np.ndarray:
    soh = history.soh_pct
    deltas = np.diff(soh)
    fade_all = _fade_rate(soh)
    fade_20 = _fade_rate(soh, 20)
    fade_50 = _fade_rate(soh, 50)
    recent = soh[-min(20, soh.size) :]
    return np.asarray(
        [
            history.current_soh,
            float(history.n - 1),
            float(history.current_cycle),
            float(soh[0] - soh[-1]),
            fade_all,
            fade_20,
            fade_50,
            fade_20 - fade_50,
            float(np.std(recent)),
            float(deltas[-1]) if deltas.size else 0.0,
            float(np.log1p(history.n)),
        ],
        dtype="float64",
    )


def padm_features(history: RULHistory) -> np.ndarray:
    soh = history.soh_pct
    deltas = np.diff(soh)
    if deltas.size == 0:
        raise RULForecastError("PADM features require at least one observed delta")
    lag_1 = float(deltas[-1])
    lag_2 = float(deltas[-2]) if deltas.size >= 2 else lag_1
    mean_3 = float(np.mean(deltas[-min(3, deltas.size) :]))
    mean_10 = float(np.mean(deltas[-min(10, deltas.size) :]))
    return np.asarray(
        [
            history.current_soh,
            lag_1,
            lag_2,
            mean_3,
            mean_10,
            float(history.n - 1) / 100.0,
        ],
        dtype="float64",
    )


# ── candidate models ────────────────────────────────────────────────────


@dataclass
class FittedModel:
    name: str
    kind: str
    estimator: Any | None
    rate: float | None = None
    feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "estimator": self.estimator,
            "rate": self.rate,
            "feature_names": self.feature_names,
            "metadata": self.metadata,
            "model_version": MODEL_VERSION,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "FittedModel":
        return cls(
            name=payload["name"],
            kind=payload["kind"],
            estimator=payload.get("estimator"),
            rate=payload.get("rate"),
            feature_names=tuple(payload.get("feature_names", ())),
            metadata=dict(payload.get("metadata", {})),
        )


def _extrapolate_ols(
    history: RULHistory,
    eol_pct: float,
    *,
    window: int | None = None,
    log_space: bool = False,
) -> float | None:
    soh = history.soh_pct
    positions = np.arange(soh.size, dtype="float64")
    if window is not None and soh.size > window:
        positions = positions[-window:]
        values = soh[-window:]
    else:
        values = soh
    if history.current_soh <= eol_pct:
        return 0.0
    if log_space:
        if (values <= 0).any():
            return None
        values = np.log(values)
        target = float(np.log(eol_pct))
    else:
        target = float(eol_pct)
    slope = _slope(positions, values)
    if slope is None or not np.isfinite(slope) or slope >= -1e-9:
        return None
    intercept = float(values.mean()) - slope * float(positions.mean())
    x_eol = (target - intercept) / slope
    rul = float(x_eol - (history.n - 1))
    if not np.isfinite(rul) or rul > MAX_EXTRAPOLATION_CYCLES:
        return None
    return float(max(0.0, rul))


def _extrapolate_last_delta(history: RULHistory, eol_pct: float, window: int = 5) -> float | None:
    if history.current_soh <= eol_pct:
        return 0.0
    deltas = np.diff(history.soh_pct)
    if deltas.size == 0:
        return None
    rate = float(np.mean(deltas[-min(window, deltas.size) :]))
    if not np.isfinite(rate) or rate >= -1e-6:
        return None
    rul = (history.current_soh - float(eol_pct)) / (-rate)
    if not np.isfinite(rul) or rul > MAX_EXTRAPOLATION_CYCLES:
        return None
    return float(max(0.0, rul))


def _padm_rollout(
    history: RULHistory,
    eol_pct: float,
    estimator: Any,
    *,
    max_steps: int = int(MAX_EXTRAPOLATION_CYCLES),
) -> tuple[float | None, dict[str, Any]]:
    soh = history.soh_pct.astype("float64").copy()
    position = history.n - 1
    if float(soh[-1]) <= float(eol_pct):
        return 0.0, {"steps": 0, "capped": False}
    previous_soh = float(soh[-1])
    for step in range(1, max_steps + 1):
        features = padm_features(
            RULHistory(
                np.arange(soh.size, dtype="int64") + 1,
                soh,
            )
        )
        delta = float(np.asarray(estimator.predict(features.reshape(1, -1)))[0])
        delta = min(delta, 0.0)
        if delta > -1e-9:
            return None, {"steps": step, "capped": True, "reason": "no_fade"}
        previous_soh = float(soh[-1])
        soh = np.append(soh, previous_soh + delta)
        position += 1
        if soh[-1] <= float(eol_pct):
            decrement = previous_soh - float(soh[-1])
            fraction = (previous_soh - float(eol_pct)) / decrement if decrement > 0 else 0.0
            rul = (step - 1) + float(np.clip(fraction, 0.0, 1.0))
            return float(max(0.0, rul)), {"steps": step, "capped": False}
    return None, {"steps": max_steps, "capped": True, "reason": "horizon"}


def fit_candidate(name: str, origins: list[Origin], eol_pct: float) -> FittedModel:
    if name == "linear_extrapolation":
        return FittedModel(name=name, kind="within_history", estimator=None)
    if name == "exponential_extrapolation":
        return FittedModel(name=name, kind="within_history", estimator=None)
    if name == "degradation_rate_20":
        return FittedModel(name=name, kind="within_history", estimator=None)
    if name == "last_delta_persistence":
        return FittedModel(name=name, kind="within_history", estimator=None)
    if name == "median_fade_constant":
        rates = []
        for origin in origins:
            if origin.history.n < 3:
                continue
            rate = _fade_rate(origin.history.soh_pct, 20)
            if rate > 0:
                rates.append(rate)
        if not rates:
            raise RULForecastError("No positive fade rate in training origins")
        return FittedModel(
            name=name,
            kind="constant_rate",
            estimator=None,
            rate=float(np.median(rates)),
        )
    if name == "padm_ar":
        raise RULForecastError(
            "PADM-AR requires next-observed SOH deltas; use fit_padm_with_future()"
        )
    if name in {"direct_ridge", "direct_hgb"}:
        if len(origins) < 10:
            raise RULForecastError(f"{name} needs at least 10 training origins")
        x = np.vstack([summary_features(origin.history) for origin in origins])
        y = np.asarray([float(origin.target_rul) for origin in origins], dtype="float64")
        if name == "direct_ridge":
            estimator = make_pipeline(StandardScaler(), Ridge(alpha=DEFAULT_ALPHA))
        else:
            estimator = HistGradientBoostingRegressor(
                max_iter=200,
                max_leaf_nodes=15,
                min_samples_leaf=10,
                learning_rate=0.05,
                random_state=0,
            )
        estimator.fit(x, y)
        return FittedModel(
            name=name,
            kind="direct",
            estimator=estimator,
            feature_names=SUMMARY_FEATURE_NAMES,
        )
    raise RULForecastError(f"Unknown RUL candidate {name!r}")


def fit_padm_with_future(
    samples: list[tuple[RULHistory, float]],
) -> FittedModel:
    """Fit the autoregressive PADM on (history, next observed SOH delta)."""
    if len(samples) < 5:
        raise RULForecastError("PADM-AR needs at least 5 training origins")
    x = np.vstack([padm_features(history) for history, _ in samples])
    y = np.asarray([float(delta) for _, delta in samples], dtype="float64")
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(x, y)
    return FittedModel(
        name="padm_ar",
        kind="padm",
        estimator=estimator,
        feature_names=PADM_FEATURE_NAMES,
    )


def predict_candidate(model: FittedModel, history: RULHistory, eol_pct: float) -> float | None:
    if history.current_soh <= float(eol_pct):
        return 0.0
    if model.name == "linear_extrapolation":
        return _extrapolate_ols(history, eol_pct)
    if model.name == "exponential_extrapolation":
        return _extrapolate_ols(history, eol_pct, log_space=True)
    if model.name == "degradation_rate_20":
        return _extrapolate_ols(history, eol_pct, window=20)
    if model.name == "last_delta_persistence":
        return _extrapolate_last_delta(history, eol_pct)
    if model.name == "median_fade_constant":
        rate = model.rate
        if rate is None or rate <= 0:
            return None
        rul = (history.current_soh - float(eol_pct)) / rate
        if rul > MAX_EXTRAPOLATION_CYCLES:
            return None
        return float(max(0.0, rul))
    if model.name == "padm_ar":
        rul, _ = _padm_rollout(history, eol_pct, model.estimator)
        return rul
    if model.name in {"direct_ridge", "direct_hgb"}:
        features = summary_features(history).reshape(1, -1)
        value = float(np.asarray(model.estimator.predict(features))[0])
        if not np.isfinite(value):
            return None
        return float(np.clip(value, 0.0, MAX_EXTRAPOLATION_CYCLES))
    raise RULForecastError(f"Unknown fitted model {model.name!r}")


CANDIDATE_NAMES = (
    "last_delta_persistence",
    "degradation_rate_20",
    "median_fade_constant",
    "linear_extrapolation",
    "exponential_extrapolation",
    "padm_ar",
    "direct_ridge",
    "direct_hgb",
)

WITHIN_HISTORY_MODELS = frozenset(
    {
        "linear_extrapolation",
        "exponential_extrapolation",
        "degradation_rate_20",
        "last_delta_persistence",
    }
)

MODEL_DESCRIPTIONS = {
    "last_delta_persistence": "Mean SOH change over the last 5 cycles held constant.",
    "degradation_rate_20": "Linear SOH fade over the last 20 cycles extrapolated.",
    "median_fade_constant": "Training median fade per cycle applied as a constant.",
    "linear_extrapolation": "OLS SOH trend over the full observed history extrapolated.",
    "exponential_extrapolation": "Log-linear exponential decay extrapolated.",
    "padm_ar": "Autoregressive SOH-delta Ridge model rolled forward causally.",
    "direct_ridge": "Ridge regression on causal trajectory summary features.",
    "direct_hgb": "Histogram gradient boosting on causal trajectory summary features.",
}


def conformal_quantile(residuals: Iterable[float], coverage: float = 0.90) -> float | None:
    values = np.asarray(
        [abs(float(value)) for value in residuals if np.isfinite(float(value))],
        dtype="float64",
    )
    if values.size == 0:
        return None
    level = min(1.0, max(0.0, float(coverage)))
    return float(np.quantile(values, level, method="higher"))


def projected_curve_interval(
    history: RULHistory,
    eol_pct: float,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Model-based interval for a censored projection.

    The interval spans the linear and exponential curve families.  It is a
    model-spread range, not a calibrated confidence interval, and its coverage
    has never been validated because the cell never reached EOL.
    """
    points: dict[str, float] = {}
    linear = _extrapolate_ols(history, eol_pct)
    exponential = _extrapolate_ols(history, eol_pct, log_space=True)
    if linear is not None:
        points["linear"] = linear
    if exponential is not None:
        points["exponential"] = exponential
    if not points:
        return None, None, {"method": "model_spread", "points": points}
    values = list(points.values())
    return float(min(values)), float(max(values)), {
        "method": "curve_family_spread",
        "points": points,
        "validated_coverage": False,
        "note": (
            "Range spans linear and exponential extrapolations. It excludes model "
            "misspecification and has no validated coverage."
        ),
    }


def build_rul_record(
    *,
    battery_id: str,
    history: RULHistory,
    eol_pct: float,
    point: float | None,
    lower: float | None,
    upper: float | None,
    model_id: str,
    chemistry: str,
    dataset_id: str,
    prediction_status: str,
    validation_status: str,
    provenance: dict[str, Any],
    interval_method: str | None,
    interval_coverage_target: float | None = None,
    warnings: Iterable[str] = (),
) -> dict[str, Any]:
    current_cycle = history.current_cycle
    point_value = None if point is None else float(max(0.0, point))
    predicted_eol = (
        int(round(current_cycle + point_value)) if point_value is not None else None
    )
    lower_value = None if lower is None else float(max(0.0, lower))
    upper_value = None if upper is None else float(max(0.0, upper))
    return {
        "battery_id": battery_id,
        "dataset_id": dataset_id,
        "current_cycle": current_cycle,
        "current_soh_pct": history.current_soh,
        "eol_threshold_pct": float(eol_pct),
        "predicted_eol_cycle": predicted_eol,
        "estimated_rul_cycles": point_value,
        "lower_uncertainty_bound": lower_value,
        "upper_uncertainty_bound": upper_value,
        "model_id": model_id,
        "model_version": MODEL_VERSION,
        "chemistry": chemistry,
        "prediction_status": prediction_status,
        "validation_status": validation_status,
        "provenance": {
            **provenance,
            "dataset_id": dataset_id,
            "history_length": history.n,
            "origin_cycle_id": current_cycle,
            "origin_soh_pct": history.current_soh,
            "model_description": MODEL_DESCRIPTIONS.get(model_id),
            "interval_method": interval_method,
            "interval_coverage_target": interval_coverage_target,
            "eol_threshold_assumption": (
                f"{eol_pct:.1f}% SOH is a project-defined end-of-life threshold; "
                "it is configurable and is not a manufacturer warranty limit."
            ),
            "unit": "observed_cycles",
        },
        "warnings": [
            "RUL is an experimental estimate, not an approved operational value.",
            *list(warnings),
        ],
    }


__all__ = [
    "CANDIDATE_NAMES",
    "DEFAULT_ALPHA",
    "FittedModel",
    "MAX_EXTRAPOLATION_CYCLES",
    "MIN_HISTORY",
    "MODEL_DESCRIPTIONS",
    "MODEL_VERSION",
    "Origin",
    "PADM_FEATURE_NAMES",
    "RULForecastError",
    "RULHistory",
    "SUMMARY_FEATURE_NAMES",
    "WITHIN_HISTORY_MODELS",
    "build_rul_record",
    "conformal_quantile",
    "fit_candidate",
    "fit_padm_with_future",
    "make_history",
    "make_origins",
    "padm_features",
    "predict_candidate",
    "projected_curve_interval",
    "summary_features",
]
