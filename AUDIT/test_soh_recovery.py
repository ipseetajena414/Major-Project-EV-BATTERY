"""
Member 1 - tests for the missing / questionable SOH label recovery audit.

Asserts the evidence-backed recovery decisions against the canonical master
table and the raw/audit sources. Read-only: never modifies data.

Run:
    python -m unittest AUDIT/test_soh_recovery.py -v
  or:
    python AUDIT/test_soh_recovery.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import member1_audit as base  # noqa: E402
import soh_recovery as recovery  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "member1_audit_outputs"


class SohRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hashes_before = {
            "parquet": base.sha256_file(base.MASTER_PARQUET),
            "csv": base.sha256_file(base.MASTER_CSV),
        }
        cls.master = base.load_master()
        cls.summary = recovery.load_b5_summary()
        cls.cycle_level, cls.sample, cls.feasibility = recovery.load_b1_tables()
        cls.b5 = recovery.b5_cycle40_evidence(cls.summary)
        cls.b5["source_artifact"] = "B5 raw zip + B5 audit summary"
        cls.b1 = recovery.b1_recovery_evidence(
            cls.master, cls.cycle_level, cls.sample, cls.feasibility
        )
        cls.baseline = base.candidate_cycle_labels(cls.master, base.run_audit()["checks"])
        cls.log = recovery.classification_frame(cls.master, cls.baseline, cls.b5, cls.b1)
        cls.report = recovery.build_report(cls.master, cls.log, cls.b5, cls.b1, cls.hashes_before)

    # --- guardrails ----------------------------------------------------
    def test_canonical_dataset_unchanged(self):
        self.assertEqual(self.hashes_before["parquet"], base.EXPECTED_SHA256_PARQUET)
        self.assertEqual(self.hashes_before["csv"], base.EXPECTED_SHA256_CSV)
        self.assertEqual(
            base.sha256_file(base.MASTER_PARQUET), base.EXPECTED_SHA256_PARQUET
        )
        self.assertFalse(self.report["metadata"]["canonical_dataset_modified"])
        guard = self.report["guardrails"]
        self.assertFalse(guard["interpolation_used"])
        self.assertFalse(guard["forward_fill_used"])
        self.assertEqual(guard["fabricated_values"], 0)

    def test_outputs_are_written_only_inside_audit_outputs(self):
        for name in ("soh_recovery_log.csv", "soh_recovery_report.json"):
            path = (OUT_DIR / name).resolve()
            self.assertTrue(path.is_file(), name)
            self.assertEqual(path.parent, OUT_DIR.resolve())

    # --- inventory -----------------------------------------------------
    def test_every_cycle_is_classified(self):
        self.assertEqual(len(self.log), 819)
        self.assertEqual(len(self.log.drop_duplicates(["dataset_id", "cell_id", "cycle_id"])), 819)
        self.assertEqual(self.log["label_state"].notna().all(), True)

    def test_label_state_inventory(self):
        counts = self.log["label_state"].value_counts().to_dict()
        self.assertEqual(counts["MISSING_NOT_RECOVERABLE"], 1)
        self.assertEqual(counts["QUESTIONABLE_RECOVERED_WITH_CONDITIONS"], 1)
        self.assertEqual(counts["CONDITIONAL_SOURCE_CAVEAT"], 249)
        self.assertEqual(counts["VALID_DEFINITIONAL"], 107)

    def test_only_one_missing_soh_label(self):
        missing = self.log[self.log["master_soh_pct"].isna()]
        self.assertEqual(len(missing), 1)
        row = missing.iloc[0]
        self.assertEqual(row["dataset_id"], "LFP_MultiStat_B5")
        self.assertEqual(int(row["cycle_id"]), 40)

    # --- B5 cycle 40: investigated, NOT recoverable ---------------------
    def test_b5_cycle40_is_not_recoverable(self):
        self.assertEqual(self.b5["recovery_status"], "NOT_RECOVERABLE")
        self.assertIsNone(self.b5["recovered_capacity_ah"])
        self.assertIsNone(self.b5["recovered_soh_pct"])
        self.assertFalse(self.b5["discharge_complete"])
        self.assertEqual(self.b5["cycle_status"], "DISCHARGE_INTERRUPTED")
        self.assertEqual(self.b5["capacity_ground_truth"], "NO")
        self.assertEqual(self.b5["modelling_use_capacity_soh"], "EXCLUDE")
        self.assertTrue(pd.isna(self.b5["conditional_capacity_ah"]))
        self.assertTrue(pd.isna(self.b5["conditional_soh_pct"]))

    def test_b5_cycle40_partial_throughput_matches_raw_workbook(self):
        raw = self.b5["raw_workbook"]
        self.assertTrue(raw["available"])
        self.assertAlmostEqual(self.b5["partial_capacity_trapz_ah"], 0.5428382, places=6)
        self.assertAlmostEqual(raw["partial_capacity_trapz_ah"], 0.5428382, places=6)
        self.assertEqual(raw["discharge_rows"], 3129)
        self.assertAlmostEqual(raw["voltage_end_v"], 3.212842, places=6)
        self.assertAlmostEqual(raw["voltage_min_v"], 3.212842, places=6)
        self.assertEqual(raw["samples_at_or_below_cutoff"], 0)
        self.assertTrue(raw["record_ends_at_last_discharge_row"])

    def test_b5_cycle40_partial_is_a_fraction_of_a_full_cycle(self):
        # A complete discharge would deliver roughly a full cell capacity; the
        # recorded segment is far short, proving it is not usable as capacity.
        self.assertLess(self.b5["partial_fraction_of_neighbour_median"], 0.6)
        self.assertGreater(self.b5["neighbouring_full_cycle_median_ah"], 1.0)
        self.assertEqual(raw_rows_for_cycle(self.summary), 1)

    def test_b5_cycle40_adopted_values_are_null(self):
        row = self.log[
            (self.log["dataset_id"] == "LFP_MultiStat_B5") & (self.log["cycle_id"] == 40)
        ].iloc[0]
        self.assertEqual(row["recovery_status"], "NOT_RECOVERABLE")
        self.assertTrue(pd.isna(row["adopted_capacity_ah"]))
        self.assertTrue(pd.isna(row["adopted_soh_pct"]))
        self.assertEqual(row["target_confidence_after"], "INVALID")
        self.assertFalse(bool(row["usable_as_target_after"]))
        # partial throughput is preserved for provenance, not as a target
        self.assertAlmostEqual(float(row["master_capacity_ah"]), 0.5428382, places=6)

    # --- B1: label-cycle mismatch, recovered with conditions -----------
    def test_b1_value_is_recovered_from_matching_cycle(self):
        self.assertEqual(self.b1["telemetry_global_cycle_est"], 2)
        self.assertEqual(self.b1["collapsed_checkpoint_last_cycle_est"], 50)
        self.assertAlmostEqual(self.b1["master_capacity_ah"], 1.202223, places=6)
        self.assertAlmostEqual(self.b1["recovered_capacity_ah"], 1.220155, places=6)
        self.assertAlmostEqual(self.b1["recovered_soh_pct"], 97.612, places=3)
        self.assertEqual(self.b1["cycle_offset_confidence"], "VERIFIED")
        self.assertTrue(self.b1["discharge_segment_complete"])
        self.assertEqual(self.b1["feasibility_overall"], "FEASIBLE")
        self.assertEqual(self.b1["recovery_status"], "RECOVERED_WITH_CONDITIONS")
        self.assertEqual(self.b1["tier_after_recovery"], "CONDITIONAL")

    def test_b1_recovered_row_is_conditional_not_high(self):
        row = self.log[self.log["dataset_id"] == "LFP_MultiStat_B1"].iloc[0]
        self.assertEqual(row["target_confidence_before"], "INVALID")
        self.assertEqual(row["target_confidence_after"], "CONDITIONAL")
        self.assertEqual(row["recovery_status"], "RECOVERED_WITH_CONDITIONS")
        self.assertAlmostEqual(float(row["adopted_capacity_ah"]), 1.220155, places=6)
        self.assertAlmostEqual(float(row["adopted_soh_pct"]), 97.612, places=3)
        self.assertTrue(bool(row["usable_as_target_after"]))
        for field in ("source_artifact", "source_locator", "method", "evidence"):
            self.assertTrue(str(row[field]).strip(), field)

    def test_b1_recovery_keeps_unit_caveat(self):
        self.assertIn("UNDERDETERMINED", self.b1["current_unit_status"])
        self.assertEqual(self.b1["capacity_status"], "COMPUTED_VERIFIED_WITH_CONDITIONS")

    # --- no silent fabrication -----------------------------------------
    def test_only_the_b1_cycle_changes_value(self):
        changed = self.log[
            self.log["adopted_soh_pct"].notna()
            & self.log["master_soh_pct"].notna()
            & ~np.isclose(
                self.log["adopted_soh_pct"].astype(float),
                self.log["master_soh_pct"].astype(float),
                equal_nan=True,
            )
        ]
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed.iloc[0]["dataset_id"], "LFP_MultiStat_B1")

    def test_conditional_b5_values_are_not_touched(self):
        b5 = self.log[
            (self.log["dataset_id"] == "LFP_MultiStat_B5") & (self.log["cycle_id"] != 40)
        ]
        self.assertEqual(len(b5), 249)
        self.assertTrue((b5["recovery_status"] == "NOT_REQUIRED").all())
        self.assertTrue(
            np.allclose(b5["adopted_soh_pct"].astype(float), b5["master_soh_pct"].astype(float))
        )
        self.assertTrue((b5["target_confidence_after"] == "CONDITIONAL").all())

    def test_definitional_cases_are_retained_not_modified(self):
        nasa_over = self.log[
            self.log["issue_type"] == "SOH_ABOVE_100PCT_DUE_TO_RATED_BASELINE"
        ]
        self.assertEqual(len(nasa_over), 7)
        self.assertTrue((nasa_over["adopted_soh_pct"].astype(float) > 100).all())
        naion = self.log[self.log["dataset_id"] == "Zenodo_IonTech_NaIon"]
        self.assertEqual(len(naion), 100)
        self.assertTrue(
            np.allclose(naion["adopted_soh_pct"].astype(float), naion["master_soh_pct"].astype(float))
        )

    # --- impact --------------------------------------------------------
    def test_usable_cycle_counts(self):
        before = self.report["usable_cycle_counts"]["before"]
        after = self.report["usable_cycle_counts"]["after"]
        self.assertEqual(before["usable_as_target"], 817)
        self.assertEqual(after["usable_as_target"], 818)
        self.assertEqual(self.report["usable_cycle_counts"]["delta_usable"], 1)
        self.assertEqual(before["by_tier"], {"CONDITIONAL": 249, "HIGH": 568, "INVALID": 2})
        self.assertEqual(after["by_tier"], {"CONDITIONAL": 250, "HIGH": 568, "INVALID": 1})
        # no value is upgraded to HIGH
        self.assertEqual(
            self.report["usable_cycle_counts"]["high_confidence_before"],
            self.report["usable_cycle_counts"]["high_confidence_after"],
        )

    def test_no_promotion_of_any_other_invalid_cycle(self):
        promoted = self.log[
            (self.log["target_confidence_before"] == "INVALID")
            & (self.log["target_confidence_after"] != "INVALID")
        ]
        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted.iloc[0]["dataset_id"], "LFP_MultiStat_B1")

    # --- persisted artifacts -------------------------------------------
    def test_persisted_log_matches_in_memory_result(self):
        path = OUT_DIR / "soh_recovery_log.csv"
        if not path.exists():
            self.skipTest("run AUDIT/soh_recovery.py to generate the log")
        saved = pd.read_csv(path)
        self.assertEqual(len(saved), len(self.log))
        self.assertEqual(list(saved.columns), list(self.log.columns))
        self.assertEqual(int(saved["usable_as_target_after"].sum()), 818)

    def test_persisted_report_counts(self):
        path = OUT_DIR / "soh_recovery_report.json"
        if not path.exists():
            self.skipTest("run AUDIT/soh_recovery.py to generate the report")
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["usable_cycle_counts"]["before"]["usable_as_target"], 817)
        self.assertEqual(saved["usable_cycle_counts"]["after"]["usable_as_target"], 818)
        self.assertTrue(saved["guardrails"]["hashes_match_expected"])
        self.assertEqual(saved["scope"]["recovered_cycles"], 1)
        self.assertEqual(saved["scope"]["not_recoverable_cycles"], 1)


def raw_rows_for_cycle(summary: pd.DataFrame) -> int:
    return int((summary["global_cycle_id_nominal_checkpoint"] == 40).sum())


if __name__ == "__main__":
    unittest.main(verbosity=2)
