"""Tests for Member 3 observed/predicted/baseline exports."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from member2_modeling.member3_export import ROLE_FILES, build_member3_exports

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    destination = tmp_path_factory.mktemp("member3_exports")
    summary = build_member3_exports(destination)
    return destination, summary


def test_exports_are_separate_and_adapter_compatible(exported) -> None:
    from member3.integration.member2_range_adapter import load_prediction_artifact

    destination, _ = exported
    loaded = {}
    for role, filename in ROLE_FILES.items():
        path = destination / filename
        assert path.is_file()
        loaded[role] = load_prediction_artifact(path)

    keys = {
        role: {(row["cell_id"], row["cycle_index"]) for row in rows}
        for role, rows in loaded.items()
    }
    assert keys["observed"] == keys["predicted"] == keys["baseline"]
    assert len(loaded["observed"]) > 0

    assert all(row["source"] == "observed" for row in loaded["observed"])
    assert all(row["source"] == "predicted" for row in loaded["predicted"])
    assert all(row["source"] == "predicted" for row in loaded["baseline"])

    import csv

    with (destination / ROLE_FILES["predicted"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        raw = list(csv.DictReader(handle))
    assert {row["unit"] for row in raw} == {"%"}


def test_predicted_and_baseline_are_not_observed_values(exported) -> None:
    from member3.integration.member2_range_adapter import load_prediction_artifact

    destination, _ = exported
    observed = {
        (row["cell_id"], row["cycle_index"]): row
        for row in load_prediction_artifact(destination / ROLE_FILES["observed"])
    }
    predicted = load_prediction_artifact(destination / ROLE_FILES["predicted"])
    baseline = load_prediction_artifact(destination / ROLE_FILES["baseline"])

    for row in predicted:
        truth = observed[(row["cell_id"], row["cycle_index"])]
        assert row["observed_next_soh_pct"] == pytest.approx(truth["soh_pct"])
        assert row["current_observed_soh_pct"] is not None
        assert row["model_name"] != truth["model_name"]
        assert row["quality_flag"] != truth["quality_flag"]

    for row in baseline:
        truth = observed[(row["cell_id"], row["cycle_index"])]
        assert row["observed_next_soh_pct"] == pytest.approx(truth["soh_pct"])
        assert row["model_name"] == "persistence_baseline"


def test_provenance_records_roles_metrics_and_readiness(exported) -> None:
    from member3.integration.member2_range_adapter import load_prediction_artifact

    destination, _ = exported
    provenance = json.loads((destination / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert set(provenance["roles"]) == set(ROLE_FILES)
    assert provenance["rows"] > 0
    assert provenance["model"]["readiness"] == "EXPERIMENTAL"
    assert provenance["model"]["production_ready"] is False
    assert provenance["source"]["sha256"]
    assert provenance["model"]["benchmark_warning"]
    assert isinstance(provenance["excluded_rows"], list)

    observed = {
        (row["cell_id"], row["cycle_index"]): row
        for row in load_prediction_artifact(destination / ROLE_FILES["observed"])
    }
    predicted = load_prediction_artifact(destination / ROLE_FILES["predicted"])
    errors = [
        abs(row["soh_pct"] - observed[(row["cell_id"], row["cycle_index"])]["soh_pct"])
        for row in predicted
    ]
    assert provenance["metrics"]["predicted_mae"] == pytest.approx(
        sum(errors) / len(errors), rel=1e-6
    )
    assert math.isfinite(provenance["metrics"]["baseline_mae"])


def test_member3_range_pipeline_consumes_predicted_export_read_only(exported, tmp_path) -> None:
    from member3.integration.member2_range_adapter import run_pipeline

    destination, _ = exported
    member3 = REPO_ROOT / "member3"
    summary = run_pipeline(
        destination / ROLE_FILES["predicted"],
        member3 / "vehicle_configs" / "tata_curvv_ev_55kwh_partial_reference.json",
        member3 / "drive_cycles" / "udds_city.csv",
        tmp_path / "range.csv",
    )
    assert summary["rows"] > 0
    assert summary["soh_source"] == "predicted"
    assert summary["range_min_km"] > 0
