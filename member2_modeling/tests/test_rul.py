"""Tests for RUL target conventions, availability and evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from member2_modeling.rul import (
    EOL_PCT,
    RULTargetError,
    RULNotApprovedError,
    add_rul_labels,
    crossing_position,
    evaluate_rul,
    require_approved_rul,
    rul_availability,
)


def test_first_and_sustained_crossings_differ_on_transient_dips() -> None:
    soh = np.array([85.0, 79.0, 78.0, 81.0, 80.5, 79.0, 78.0, 77.0])
    assert crossing_position(soh, "first") == 1
    assert crossing_position(soh, "sustained3") == 5


def test_threshold_is_strict_and_exact_eol_is_not_crossed() -> None:
    assert crossing_position(np.array([85.0, EOL_PCT, 79.9]), "first") == 2
    assert crossing_position(np.array([85.0, 80.0, 80.1]), "first") is None


def test_add_rul_labels_marks_pre_post_and_unavailable_cells() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["A"] * 5 + ["B"] * 3,
            "cycle_position": [0, 1, 2, 3, 4, 0, 1, 2],
            "soh_pct": [85.0, 82.0, 79.0, 78.0, 77.0, 90.0, 89.0, 88.0],
        }
    )
    labelled = add_rul_labels(frame, convention="first")
    a = labelled[labelled["cell_id"] == "A"].reset_index(drop=True)
    b = labelled[labelled["cell_id"] == "B"].reset_index(drop=True)

    assert a["rul_cycles"].tolist() == [2.0, 1.0, 0.0, 0.0, 0.0]
    assert a["is_post_threshold"].tolist() == [False, False, True, True, True]
    assert b["rul_cycles"].isna().all()
    assert b["is_post_threshold"].isna().all()
    assert (a["crossing_position"] == 2).all()
    assert frame.columns.tolist() == ["cell_id", "cycle_position", "soh_pct"]


def test_sustained_convention_changes_rul_labels() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["A"] * 8,
            "cycle_position": np.arange(8),
            "soh_pct": [85.0, 79.0, 78.0, 81.0, 80.5, 79.0, 78.0, 77.0],
        }
    )
    first = add_rul_labels(frame, convention="first")
    sustained = add_rul_labels(frame, convention="sustained3")
    assert first["rul_cycles"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert sustained["rul_cycles"].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0]


def test_unknown_crossing_convention_is_rejected() -> None:
    with pytest.raises(RULTargetError):
        crossing_position([90.0, 79.0], "sustained2")


def test_evaluation_keeps_pre_and_post_threshold_separate() -> None:
    truth = np.array([10.0, 5.0, 3.0, 0.0, 0.0])
    predictions = np.array([12.0, 6.0, 2.0, 1.0, 0.0])
    result = evaluate_rul(truth, predictions)

    assert result["counts"] == {
        "pre_threshold": 3,
        "post_threshold": 2,
        "near_eol": 2,
    }
    assert result["pre_threshold"]["mae"] == pytest.approx(4.0 / 3.0)
    assert result["post_threshold"]["mae"] == pytest.approx(0.5)
    assert set(result["buckets"]) == {"1-5", "6-10", "11-30", "31+"}
    assert result["buckets"]["6-10"]["n"] == 1
    assert result["buckets"]["1-5"]["n"] == 1
    assert "never pooled" in result["pooling_rule"].lower()


def test_rul_is_unavailable_for_every_snapshot_dataset() -> None:
    for dataset_id in (
        "EV_FAILURE_PREDICTION",
        "SOLID_STATE_TELEMETRY_2026",
        "SYNTHETIC_BATTERY_LIFECYCLE",
    ):
        availability = rul_availability(dataset_id)
        assert availability["status"] == "UNAVAILABLE"
        assert availability["approved_for_use"] is False
        assert availability["rul_cycles"] is None
        assert availability["reason"]


def test_legacy_nmc_rul_is_experimental_but_never_approved() -> None:
    availability = rul_availability("LEGACY_NMC_SEQUENTIAL")
    assert availability["status"] == "EXPERIMENTAL_PRE_EOL_ONLY"
    assert availability["approved_for_use"] is False
    assert availability["unit"] == "observed_cycles"
    assert availability["eol_threshold_pct"] == EOL_PCT
    with pytest.raises(RULNotApprovedError):
        require_approved_rul("LEGACY_NMC_SEQUENTIAL")


def test_rul_report_audits_conventions_and_references_existing_evidence(tmp_path) -> None:
    from member2_modeling.data_integration import DATASET_IDS
    from member2_modeling.rul import build_rul_report

    report = build_rul_report(out_dir=tmp_path)
    assert set(report["availability"]) == set(DATASET_IDS) | {"LEGACY_NMC_SEQUENTIAL"}
    for dataset_id in DATASET_IDS:
        assert report["availability"][dataset_id]["status"] == "UNAVAILABLE"
    legacy = report["availability"]["LEGACY_NMC_SEQUENTIAL"]
    assert legacy["status"] == "EXPERIMENTAL_PRE_EOL_ONLY"
    assert legacy["approved_for_use"] is False

    conventions = report["legacy_target_audit"]["conventions"]
    first = {row["cell_id"]: row for row in conventions["first"]["cells"]}
    sustained = {row["cell_id"]: row for row in conventions["sustained3"]["cells"]}
    assert first["B0005"]["crossing_position"] == 74
    assert first["B0018"]["crossing_position"] == 44
    assert sustained["B0018"]["crossing_position"] == 58
    assert first["B0005"]["pre_threshold_cycles"] == 74

    evidence = report["existing_validation_evidence"]
    assert evidence["read_only"] is True
    assert evidence["primary_pre_eol_mae"] == pytest.approx(6.78218097639423)

    assert (tmp_path / "rul_availability.json").is_file()
    markdown = (tmp_path / "MEMBER2_RUL_TARGET_AUDIT.md").read_text(encoding="utf-8")
    assert "never approved" in markdown.lower()
