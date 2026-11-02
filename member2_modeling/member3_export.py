"""Export observed, predicted and baseline SOH artifacts for Member 3.

All three files cover the same chronological holdout rows so Member 3 can
compare roles without joining mismatched cycles. The existing Member 3 adapter
is used read-only for validation; no ``member3/`` file is modified.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from member2_modeling.hybrid import HybridSOHModel
from member2_modeling.hybrid.train_hybrid import (
    ENTITY_COLUMN,
    ORDER_COLUMN,
    TARGET_COLUMN,
    load_legacy_nmc_pairs,
    split_chronologically,
)

WORKSPACE = Path(__file__).resolve().parents[1]
HYBRID_DIR = WORKSPACE / "member2_modeling" / "artifacts" / "hybrid_nmc_legacy"
MODEL_DIR = HYBRID_DIR / "hybrid_model"
MANIFEST_PATH = HYBRID_DIR / "hybrid_manifest.json"

ROLE_FILES = {
    "observed": "observed_soh.csv",
    "predicted": "predicted_soh.csv",
    "baseline": "baseline_soh.csv",
}
PROVENANCE_FILE = "PROVENANCE.json"

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

SOH_MIN_EXCLUSIVE = 1.0
SOH_MAX_INCLUSIVE = 100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_soh(value: float) -> bool:
    return (
        math.isfinite(value)
        and SOH_MIN_EXCLUSIVE < value <= SOH_MAX_INCLUSIVE
    )


def _format(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=EXPORT_COLUMNS, extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _format(row.get(column)) for column in EXPORT_COLUMNS})


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


def _metrics(observed: list[float], predicted: list[float]) -> dict[str, float]:
    errors = [prediction - truth for truth, prediction in zip(observed, predicted)]
    absolute = [abs(error) for error in errors]
    return {
        "mae": float(sum(absolute) / len(absolute)),
        "rmse": float(math.sqrt(sum(error**2 for error in errors) / len(errors))),
        "bias": float(sum(errors) / len(errors)),
    }


def build_member3_exports(
    out_dir: Path | str | None = None,
    *,
    split_name: str = "test",
) -> dict[str, Any]:
    """Generate aligned observed, predicted and baseline Member 3 CSV files."""
    destination = (
        Path(out_dir)
        if out_dir is not None
        else WORKSPACE / "member2_modeling" / "artifacts" / "member3_exports"
    )
    destination.mkdir(parents=True, exist_ok=True)

    pairs, metadata = load_legacy_nmc_pairs()
    model = HybridSOHModel.load(MODEL_DIR)
    predictions = model.predict(pairs.reset_index(drop=True))
    if len(predictions) != len(pairs):
        raise RuntimeError("hybrid predictions do not align with model-ready pairs")

    enriched = pairs.copy()
    for column in ("tabular_pred", "sequence_pred", "fused_pred", "sequence_available"):
        enriched[column] = predictions[column].to_numpy()

    split_frame = split_chronologically(pairs)
    enriched = enriched.merge(
        split_frame[["row_id", "split"]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    holdout = enriched[enriched["split"] == split_name].copy()
    holdout = holdout.sort_values(
        [ENTITY_COLUMN, "target_cycle_id"], kind="mergesort"
    )

    observed_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    observed_values: list[float] = []
    predicted_values: list[float] = []
    baseline_values: list[float] = []
    excluded: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()

    for record in holdout.to_dict(orient="records"):
        cell_id = str(record[ENTITY_COLUMN])
        target_cycle = int(record["target_cycle_id"])
        current_cycle = int(record[ORDER_COLUMN])
        observed_soh = float(record[TARGET_COLUMN])
        predicted_soh = float(record["fused_pred"])
        baseline_soh = float(record["current_soh_pct"])

        reasons = []
        for role, value in (
            ("observed", observed_soh),
            ("predicted", predicted_soh),
            ("baseline", baseline_soh),
        ):
            if not _valid_soh(value):
                reasons.append(
                    f"{role} SOH {value!r} is outside the Member 3 contract "
                    f"({SOH_MIN_EXCLUSIVE} < SOH <= {SOH_MAX_INCLUSIVE})"
                )
        key = (cell_id, target_cycle)
        if key in seen:
            reasons.append("duplicate target cycle for this cell")
        if reasons:
            excluded.append(
                {
                    "cell_id": cell_id,
                    "cycle_index": target_cycle,
                    "reasons": reasons,
                }
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
                model_name="hybrid_nmc_legacy_fused",
                model_status="EXPERIMENTAL",
                quality_flag="MEMBER2_HYBRID_EXPERIMENTAL_UNVERIFIED",
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
            "No eligible holdout rows satisfy the Member 3 adapter contract."
        )

    _write_csv(destination / ROLE_FILES["observed"], observed_rows)
    _write_csv(destination / ROLE_FILES["predicted"], predicted_rows)
    _write_csv(destination / ROLE_FILES["baseline"], baseline_rows)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    role_records = {
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
    provenance = {
        "module": "member2_modeling/member3_export",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": "LEGACY_NMC_SEQUENTIAL",
        "split": {
            "name": split_name,
            "protocol": "per-cell chronological 60/20/20 holdout",
        },
        "target": metadata["target"],
        "cycle_index_semantics": (
            "cycle_index is the next observed record's source cycle_id; the "
            "horizon is one observed record, not necessarily one physical cycle."
        ),
        "rows": len(observed_rows),
        "roles": role_records,
        "source": {
            "file": metadata["source_file"],
            "sha256": metadata["source_sha256"],
            "cells": metadata["cells"],
        },
        "model": {
            "name": "hybrid_nmc_legacy_fused",
            "artifact": MODEL_DIR.relative_to(WORKSPACE).as_posix(),
            "manifest": MANIFEST_PATH.relative_to(WORKSPACE).as_posix(),
            "manifest_sha256": _sha256(MANIFEST_PATH),
            "readiness": "EXPERIMENTAL",
            "production_ready": False,
            "benchmark_warning": manifest.get("benchmark_warning"),
        },
        "metrics": {
            "predicted": _metrics(observed_values, predicted_values),
            "baseline": _metrics(observed_values, baseline_values),
            "predicted_mae": _metrics(observed_values, predicted_values)["mae"],
            "baseline_mae": _metrics(observed_values, baseline_values)["mae"],
        },
        "excluded_rows": excluded,
        "warnings": [
            "Observed rows are measured labels; predicted and baseline rows are "
            "not observations and must not be labelled as measured.",
            "Persistence and training-mean-fade baselines outperform the fused "
            "hybrid on this holdout; range outputs remain experimental.",
            "Member 3 combines SOH with a partially sourced vehicle profile, so "
            "range estimates are not validated real-world predictions.",
        ],
        "adapter_compatibility": {
            "required_columns": ["cycle_index", "cell_id", "chemistry", "soh_pct"],
            "supported_chemistries": ["NMC", "LFP"],
            "soh_bounds": "1 < SOH <= 100",
            "validated_by": "member3.integration.member2_range_adapter",
            "member3_files_modified": False,
        },
    }
    (destination / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "out_dir": str(destination),
        "rows": len(observed_rows),
        "excluded_rows": len(excluded),
        "metrics": provenance["metrics"],
        "provenance": provenance,
    }


__all__ = ["EXPORT_COLUMNS", "PROVENANCE_FILE", "ROLE_FILES", "build_member3_exports"]
