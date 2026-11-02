"""
PADM Step 11: Member 3 Export Adapter
=======================================

Generates the same three aligned CSV files that member3_export.py produces,
but sources predictions from the PADM instead of the deprecated hybrid model.

If the PADM evaluation determined ``export_model == 'mean_fade_baseline'``,
the mean-fade predictor is used in place of the PADM — this is a valid
scientific result and is fully documented in the provenance JSON.

Contract
--------
* Observed, predicted, and baseline CSVs cover identical test rows.
* All eight gate tests from padm_evaluate must pass before export.
* SOH values must satisfy 0 < SOH ≤ 100 for Member 3 compatibility.
  Rows outside this range are excluded and counted.
* ``production_ready = False`` always.

Usage
-----
    from member2_modeling.delta.padm_export import build_padm_member3_exports
    result = build_padm_member3_exports()
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler

from member2_modeling.delta.delta_features import PADM_FEATURE_COLS, SOH_CLIP_MAX
from member2_modeling.delta.padm_evaluate import (
    _compute_delta_clip_bounds,
    _predict_soh,
    _mean_fade_soh,
    _persistence_soh,
)

WORKSPACE = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "delta"
EXPORT_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "member3_exports"

# Member 3 contract SOH bounds (same as legacy member3_export.py)
MEMBER3_SOH_MIN_EXCLUSIVE = 1.0
MEMBER3_SOH_MAX_INCLUSIVE = 100.0

EXPORT_COLUMNS = [
    "cycle_index",
    "cell_id",
    "chemistry",
    "soh_pct",
    "unit",
    "source",
    "model_name",
    "model_status",
    "split",
    "quality_flag",
    "current_observed_soh_pct",
    "observed_next_soh_pct",
    "window_end_cycle_id",
]

ROLE_FILES = {
    "observed": "observed_soh.csv",
    "predicted": "predicted_soh.csv",
    "baseline": "baseline_soh.csv",
}
PROVENANCE_FILE = "PROVENANCE.json"


# ── Utilities ─────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_soh_member3(value: float) -> bool:
    return (
        math.isfinite(value)
        and MEMBER3_SOH_MIN_EXCLUSIVE < value <= MEMBER3_SOH_MAX_INCLUSIVE
    )


def _format(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: _format(row.get(col)) for col in EXPORT_COLUMNS})


def _role_row(
    shared: dict[str, Any],
    *,
    soh_pct: float,
    source: str,
    model_name: str,
    model_status: str,
    quality_flag: str,
) -> dict[str, Any]:
    return {
        **shared,
        "soh_pct": round(float(soh_pct), 6),
        "source": source,
        "model_name": model_name,
        "model_status": model_status,
        "quality_flag": quality_flag,
    }


def _compute_metrics(
    observed: list[float], predicted: list[float]
) -> dict[str, float]:
    errors = [p - o for o, p in zip(observed, predicted)]
    absolute = [abs(e) for e in errors]
    return {
        "mae": float(sum(absolute) / len(absolute)) if absolute else float("nan"),
        "rmse": float(math.sqrt(sum(e**2 for e in errors) / len(errors))) if errors else float("nan"),
        "bias": float(sum(errors) / len(errors)) if errors else float("nan"),
    }


# ── Main export function ──────────────────────────────────────────────────────


def build_padm_member3_exports(
    out_dir: Path | str | None = None,
    *,
    split_name: str = "test",
    artifact_dir: Path | str | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """
    Generate aligned observed, predicted, and baseline Member 3 CSV files
    using PADM predictions (or mean-fade if PADM failed to beat both baselines).

    Parameters
    ----------
    out_dir : Path or str, optional
        Destination directory for CSVs.  Defaults to
        ``member2_modeling/artifacts/member3_exports/``.
    split_name : str
        Split to export.  Must be 'test'.
    artifact_dir : Path or str, optional
        Directory containing PADM model artifacts.
    feature_cols : list[str] or None
        Uses PADM_FEATURE_COLS if None.

    Returns
    -------
    dict with row counts, metrics, excluded rows, and provenance.
    """
    if split_name != "test":
        raise ValueError("Only 'test' split should be exported to Member 3.")

    destination = Path(out_dir) if out_dir else EXPORT_DIR
    destination.mkdir(parents=True, exist_ok=True)
    art_dir = Path(artifact_dir) if artifact_dir else ARTIFACT_DIR

    if feature_cols is None:
        feature_cols = PADM_FEATURE_COLS

    # Load test evaluation results (gate tests must have passed)
    eval_path = art_dir / "padm_test_evaluation.json"
    if not eval_path.exists():
        raise RuntimeError(
            f"Test evaluation results not found at {eval_path}. "
            "Run padm_evaluate.evaluate_padm_on_test() first."
        )
    eval_results: dict[str, Any] = json.loads(eval_path.read_text())

    if not eval_results.get("all_gates_passed", False):
        failed_gates = [k for k, v in eval_results["gate_tests"].items() if not v]
        raise RuntimeError(
            f"Not all gate tests passed. Failed: {failed_gates}. "
            "Fix issues before exporting to Member 3."
        )

    export_model_name: str = eval_results["export_model"]
    quality_flag: str = (
        eval_results["quality_flags"].get("ridge", "PADM_BELOW_ALL_BASELINES")
        if "Ridge" in export_model_name
        else eval_results["quality_flags"].get("elasticnet", "PADM_BELOW_ALL_BASELINES")
    )
    if export_model_name == "mean_fade_baseline":
        quality_flag = "PADM_BELOW_ALL_BASELINES_USING_MEAN_FADE"

    train_mean_delta: float = float(eval_results["train_mean_delta"])
    clip_low: float = eval_results["delta_clip_bounds"]["low"]
    clip_high: float = eval_results["delta_clip_bounds"]["high"]

    # Load delta pairs (full dataset, for test split)
    delta_parquet = art_dir / "delta_pairs_nmc.parquet"
    if not delta_parquet.exists():
        raise RuntimeError(
            f"Delta pairs parquet not found at {delta_parquet}. "
            "Run delta_features.build_and_save_delta_dataset() first."
        )
    delta_df = pd.read_parquet(delta_parquet)
    test_df = delta_df[delta_df["split"] == split_name].copy()
    test_df = test_df.sort_values(["cell_id", "cycle_id"], kind="mergesort").reset_index(drop=True)

    # Load models and scaler
    scaler: StandardScaler = joblib.load(art_dir / "padm_scaler_final.joblib")

    # Produce PADM predictions
    X_test = scaler.transform(test_df[feature_cols].values.astype(np.float32))

    if "Ridge" in export_model_name:
        model: Ridge = joblib.load(art_dir / "padm_ridge.joblib")
        raw_delta = model.predict(X_test)
        padm_soh_pred = _predict_soh(
            test_df["current_soh_pct"].values.astype(np.float32),
            raw_delta,
            clip_low,
            clip_high,
        )
    elif "ElasticNet" in export_model_name:
        model_en: ElasticNet = joblib.load(art_dir / "padm_elasticnet.joblib")
        raw_delta = model_en.predict(X_test)
        padm_soh_pred = _predict_soh(
            test_df["current_soh_pct"].values.astype(np.float32),
            raw_delta,
            clip_low,
            clip_high,
        )
    else:
        # Mean-fade fallback
        padm_soh_pred = _mean_fade_soh(
            test_df["current_soh_pct"].values.astype(np.float32),
            train_mean_delta,
        )

    # Baseline predictions (persistence and mean-fade use same rows)
    persist_soh_pred = _persistence_soh(test_df["current_soh_pct"].values.astype(np.float32))

    # Build CSV rows
    observed_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    observed_values: list[float] = []
    predicted_values: list[float] = []
    baseline_values: list[float] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for idx, record in enumerate(test_df.to_dict(orient="records")):
        cell_id = str(record["cell_id"])
        target_cycle = int(record.get("target_cycle_id", record["cycle_id"]))
        current_cycle = int(record["cycle_id"])
        observed_soh = float(
            np.clip(
                record["current_soh_pct"] + record["delta_target"],
                0.0,
                SOH_CLIP_MAX,
            )
        )
        predicted_soh = float(padm_soh_pred[idx])
        baseline_soh = float(persist_soh_pred[idx])

        reasons: list[str] = []

        # Member 3 contract: 1 < SOH ≤ 100
        for role, value in (
            ("observed", observed_soh),
            ("predicted", predicted_soh),
            ("baseline", baseline_soh),
        ):
            if not _valid_soh_member3(value):
                reasons.append(
                    f"{role} SOH {value:.4f} is outside Member 3 contract "
                    f"({MEMBER3_SOH_MIN_EXCLUSIVE} < SOH ≤ {MEMBER3_SOH_MAX_INCLUSIVE})"
                )

        key = (cell_id, target_cycle)
        if key in seen:
            reasons.append("duplicate target cycle for this cell")

        if reasons:
            excluded.append(
                {"cell_id": cell_id, "cycle_index": target_cycle, "reasons": reasons}
            )
            continue

        seen.add(key)

        shared = {
            "cycle_index": target_cycle,
            "cell_id": cell_id,
            "chemistry": "NMC",
            "unit": "%",
            "split": split_name,
            "current_observed_soh_pct": round(float(record["current_soh_pct"]), 6),
            "observed_next_soh_pct": round(observed_soh, 6),
            "window_end_cycle_id": current_cycle,
        }

        observed_rows.append(
            _role_row(
                shared,
                soh_pct=observed_soh,
                source="observed",
                model_name="observed_measurement",
                model_status="measured",
                quality_flag="OBSERVED_SOH",
            )
        )
        predicted_rows.append(
            _role_row(
                shared,
                soh_pct=predicted_soh,
                source="predicted",
                model_name=export_model_name,
                model_status="EXPERIMENTAL",
                quality_flag=quality_flag,
            )
        )
        baseline_rows.append(
            _role_row(
                shared,
                soh_pct=baseline_soh,
                source="predicted",
                model_name="persistence_baseline",
                model_status="baseline",
                quality_flag="PERSISTENCE_BASELINE_FROM_OBSERVED_SOH",
            )
        )
        observed_values.append(observed_soh)
        predicted_values.append(predicted_soh)
        baseline_values.append(baseline_soh)

    if not observed_rows:
        raise RuntimeError(
            "No eligible test rows satisfy the Member 3 adapter contract. "
            "All rows were excluded — check SOH bounds and data."
        )

    # Write CSVs
    _write_csv(destination / ROLE_FILES["observed"], observed_rows)
    _write_csv(destination / ROLE_FILES["predicted"], predicted_rows)
    _write_csv(destination / ROLE_FILES["baseline"], baseline_rows)
    print(f"  [padm_export] Wrote {len(observed_rows)} rows to {destination}")
    if excluded:
        print(f"  [padm_export] Excluded {len(excluded)} rows (outside Member 3 contract)")

    # Provenance
    role_records: dict[str, dict] = {
        role: {
            "path": ROLE_FILES[role],
            "sha256": _sha256(destination / ROLE_FILES[role]),
            "rows": len(rows),
        }
        for role, rows in (
            ("observed", observed_rows),
            ("predicted", predicted_rows),
            ("baseline", baseline_rows),
        )
    }

    provenance: dict[str, Any] = {
        "module": "member2_modeling/delta/padm_export",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": "LEGACY_NMC_SEQUENTIAL",
        "split": {
            "name": split_name,
            "protocol": "per-cell chronological 60/20/20 holdout",
        },
        "target": "next_cycle_soh_pct (one-step-ahead)",
        "cycle_index_semantics": (
            "cycle_index is the next observed record's source cycle_id; "
            "horizon is one observed record, not necessarily one physical cycle."
        ),
        "rows": len(observed_rows),
        "excluded_rows": excluded,
        "roles": role_records,
        "model": {
            "name": export_model_name,
            "architecture": "Persistence-Anchored Delta Model (PADM)",
            "artifact_dir": str(art_dir.relative_to(WORKSPACE)),
            "readiness": "EXPERIMENTAL",
            "production_ready": False,
            "quality_flag": quality_flag,
            "gate_tests": eval_results["gate_tests"],
            "all_gates_passed": eval_results["all_gates_passed"],
        },
        "metrics": {
            "predicted": _compute_metrics(observed_values, predicted_values),
            "baseline": _compute_metrics(observed_values, baseline_values),
        },
        "test_evaluation_summary": {
            "padm_model_used": export_model_name,
            "padm_mae": eval_results["pooled_metrics"].get(
                "ridge" if "Ridge" in export_model_name else "elasticnet", {}
            ).get("mae"),
            "persistence_mae": eval_results["pooled_metrics"].get("persistence", {}).get("mae"),
            "mean_fade_mae": eval_results["pooled_metrics"].get("mean_fade", {}).get("mae"),
        },
        "warnings": [
            "Observed rows are measured labels; predicted and baseline rows are NOT observations.",
            (
                f"Export model: {export_model_name}. "
                "If PADM_BELOW_ALL_BASELINES, mean-fade is used — this is the best available predictor."
            ),
            "production_ready=False; do not use for real-world range decisions.",
            "Member 3 range estimates depend on unvalidated vehicle profiles.",
        ],
        "adapter_compatibility": {
            "required_columns": ["cycle_index", "cell_id", "chemistry", "soh_pct"],
            "supported_chemistries": ["NMC"],
            "soh_bounds": f"{MEMBER3_SOH_MIN_EXCLUSIVE} < SOH <= {MEMBER3_SOH_MAX_INCLUSIVE}",
            "validated_by": "member3.integration.member2_range_adapter",
            "member3_files_modified": False,
        },
    }
    (destination / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  [padm_export] PROVENANCE.json written to {destination}")

    return {
        "out_dir": str(destination),
        "rows": len(observed_rows),
        "excluded_rows": len(excluded),
        "export_model": export_model_name,
        "quality_flag": quality_flag,
        "metrics": provenance["metrics"],
        "provenance": provenance,
    }


__all__ = ["build_padm_member3_exports"]
