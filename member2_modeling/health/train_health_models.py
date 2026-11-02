"""Train and evaluate SOH / capacity models on the Member 1 snapshot datasets.

Targets and leakage exclusions come from the audited compatibility layer.
This script never writes to DATA/, master_table.parquet or existing Member 2
model artifacts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member1_pipeline import loader  # noqa: E402
from member2_modeling.data_integration.profiles import (  # noqa: E402
    feature_columns_for_target,
    target_spec,
)
from member2_modeling.health import (  # noqa: E402
    TabularHealthModel,
    capacity_semantics,
)

DEFAULT_OUTPUT = REPO_ROOT / "member2_modeling" / "artifacts" / "health"

TRAINING_TARGETS = (
    ("EV_FAILURE_PREDICTION", "state_of_health", "source_label"),
    ("SOLID_STATE_TELEMETRY_2026", "state_of_health_pct", "source_label"),
    ("EV_FAILURE_PREDICTION", "remaining_capacity", "derived_from_health_percent"),
    ("SYNTHETIC_BATTERY_LIFECYCLE", "discharge_capacity_ah", "synthetic_label"),
)

NOT_TRAINED = {
    "EV_FAILURE_PREDICTION": {
        "battery_health_percent": (
            "Duplicate SOH label; state_of_health is the primary audited target."
        ),
        "predicted_remaining_life_cycles": "Supplied projection, not an observed RUL label.",
    },
    "SOLID_STATE_TELEMETRY_2026": {
        "degradation_risk_tier": "Classification target; outside the Phase 4 SOH/capacity regression scope.",
        "projected_cycle_life_80pct_soh": "Projected cycle life is not observed RUL.",
    },
    "SYNTHETIC_BATTERY_LIFECYCLE": {
        "state_of_health": "Categorical synthetic label; regression modules do not consume it.",
    },
}


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def chemistry_column(dataset_id: str) -> str | None:
    entry = loader.load_registry()["active_datasets"][dataset_id]
    for key in ("chemistry_family_column", "chemistry_column"):
        column = entry.get(key)
        if column:
            return str(column)
    return None


def train_target(
    dataset_id: str,
    target: str,
    target_kind: str,
    output: Path,
) -> dict[str, Any]:
    frame = loader.load_dataset(dataset_id)
    spec = target_spec(dataset_id, target)
    features = feature_columns_for_target(dataset_id, target)
    chemistry = chemistry_column(dataset_id)
    model = TabularHealthModel(
        target_column=target,
        feature_columns=features,
        chemistry_column=chemistry,
        forbidden_columns=[target, *spec["leakage_columns"]],
        readiness=spec["readiness"],
        target_kind=target_kind,
        random_state=42,
    ).fit(frame)
    artifact_dir = output / dataset_id / target
    model.save(artifact_dir)
    description = model.describe()
    return {
        "dataset_id": dataset_id,
        "target": target,
        "target_kind": target_kind,
        "readiness": spec["readiness"],
        "label_origin": spec["label_origin"],
        "unit": spec["unit"],
        "feature_count": len(features),
        "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "evaluation": description["evaluation"],
        "model": {
            "model_type": description["model_type"],
            "schema_version": description["schema_version"],
            "feature_names": description["feature_names"],
            "production_ready": description["production_ready"],
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Member 2 Health Model Evaluation — Snapshot Datasets",
        "",
        "Split protocol: deterministic stratified 60/20/20 snapshot split. These",
        "datasets contain independent snapshots, so this estimates i.i.d.",
        "generalization, not a temporal or vehicle-level forecast.",
        "",
        "| Dataset | Target | Kind | Readiness | Test n | Model MAE | Median MAE | Chem-median MAE |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{value:.4f}"

    for dataset_id, entry in report["datasets"].items():
        for target, result in entry["targets"].items():
            evaluation = result["evaluation"]
            baseline = evaluation["baselines"]["train_median"]["mae"]
            chemistry_baseline = evaluation["baselines"]["train_chemistry_median"]
            lines.append(
                f"| `{dataset_id}` | `{target}` | {result['target_kind']} | "
                f"{result['readiness']} | {evaluation['counts']['test']} | "
                f"{fmt(evaluation['model']['mae'])} | {fmt(baseline)} | "
                f"{fmt(None if chemistry_baseline is None else chemistry_baseline['mae'])} |"
            )

    lines.extend(["", "## Per-chemistry test MAE", ""])
    lines.append("| Dataset | Target | Chemistry | n | MAE |")
    lines.append("|---|---|---|---:|---:|")
    for dataset_id, entry in report["chemistry_comparison"].items():
        for target, groups in entry.items():
            for chemistry, metrics in groups.items():
                lines.append(
                    f"| `{dataset_id}` | `{target}` | {chemistry} | "
                    f"{metrics['n']} | {fmt(metrics['mae'])} |"
                )

    lines.extend(["", "## Capacity semantics", ""])
    for dataset_id, semantics in report["capacity"].items():
        lines.append(f"### `{dataset_id}`")
        lines.append("")
        lines.append(
            f"- Capacity column: {semantics['measured_column'] or 'none'} "
            f"({semantics['measured_unit'] or 'n/a'})"
        )
        lines.append(
            f"- Predicted capacity available: {semantics['predicted_capacity_available']}"
        )
        algebraic = semantics["algebraic"]
        lines.append(
            "- Algebraic derivation: "
            + (
                "none"
                if algebraic is None
                else f"SOH x `{algebraic['source_column']}` ({algebraic['unit']})"
            )
        )
        for note in semantics["notes"]:
            lines.append(f"- Note: {note}")
        lines.append("")

    lines.extend(["", "## Not trained in this phase", ""])
    for dataset_id, targets in report["not_trained"].items():
        for target, reason in targets.items():
            lines.append(f"- `{dataset_id}` / `{target}`: {reason}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    registry = loader.load_registry()
    datasets: dict[str, Any] = {
        dataset_id: {"targets": {}} for dataset_id in {row[0] for row in TRAINING_TARGETS}
    }
    chemistry_comparison: dict[str, Any] = {}

    for dataset_id, target, target_kind in TRAINING_TARGETS:
        print(f"Training {dataset_id} / {target}...")
        result = train_target(dataset_id, target, target_kind, output)
        datasets[dataset_id]["targets"][target] = result
        evaluation = result["evaluation"]
        chemistry_comparison.setdefault(dataset_id, {})[target] = evaluation.get(
            "per_chemistry", {}
        )
        print(
            "  test n={n} model_MAE={mae} median_MAE={baseline} "
            "beats_median={beats}".format(
                n=evaluation["counts"]["test"],
                mae=evaluation["model"]["mae"],
                baseline=evaluation["baselines"]["train_median"]["mae"],
                beats=evaluation["outperforms_train_median"],
            )
        )

    for dataset_id, entry in datasets.items():
        processed = registry["active_datasets"][dataset_id]
        entry["provenance"] = {
            "source_file": processed["source_file"],
            "source_sha256": processed["source_sha256"],
            "processed_sha256": processed["processed_sha256"],
            "data_origin": processed["data_origin"],
            "record_grain": processed["record_grain"],
        }

    report = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "registry_version": registry.get("registry_version"),
        "split_protocol": "deterministic_stratified_snapshot_split_60_20_20",
        "datasets": datasets,
        "chemistry_comparison": chemistry_comparison,
        "capacity": {
            dataset_id: capacity_semantics(dataset_id)
            for dataset_id in (
                "EV_FAILURE_PREDICTION",
                "SOLID_STATE_TELEMETRY_2026",
                "SYNTHETIC_BATTERY_LIFECYCLE",
            )
        },
        "not_trained": NOT_TRAINED,
        "warnings": [
            "Snapshot splits are not temporal validation.",
            "Synthetic labels are not measured experimental truth.",
            "No model in this report is production-validated.",
        ],
        "protected_artifacts_untouched": [
            "DATA/processed/",
            "master_table.parquet",
            "member3/",
            "member2_modeling/*_model.joblib",
            "member2_modeling/model_manifest.json",
            "member2_modeling/feature_schema.json",
        ],
    }
    (output / "health_model_evaluation.json").write_text(
        json.dumps(jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "health_model_evaluation.md").write_text(
        markdown_report(jsonable(report)), encoding="utf-8"
    )
    print(f"Artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
