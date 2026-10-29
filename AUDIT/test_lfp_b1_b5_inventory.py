"""
Member 1 - tests for the read-only LFP B1-B5 inventory / validation-readiness audit.

These tests re-verify explicit expectations against the raw archives, the audit
bundles and the canonical master table. They never modify data.

Run:
    python -m unittest AUDIT/test_lfp_b1_b5_inventory.py -v
  or:
    python AUDIT/test_lfp_b1_b5_inventory.py
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lfp_b1_b5_inventory as inv  # noqa: E402

EXPECTED_COVERAGE_STATUS = {
    "BATTERY_1": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
    "BATTERY_2": "SUITABLE_FOR_INDEPENDENT_VALIDATION",
    "BATTERY_3": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
    "BATTERY_4": "POTENTIALLY_SUITABLE_AFTER_VERIFICATION",
    "BATTERY_5": "BLOCKED_BY_LABEL_QUALITY",
}


class ArchiveIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {
            "B1_AUDIT": inv.B1_AUDIT,
            "B3_RAW": inv.B3_RAW,
            "B4_RAW": inv.B4_RAW,
            "B5_RAW": inv.B5_RAW,
            "B6_AUDIT": inv.B6_AUDIT,
        }

    def test_archive_hashes_match_expected(self):
        for key, path in self.paths.items():
            self.assertTrue(path.exists(), key)
            self.assertEqual(
                inv.sha256_file(path), inv.EXPECTED_ARCHIVE_SHA256[key], key
            )

    def test_master_table_hashes(self):
        self.assertEqual(inv.sha256_file(inv.MASTER_PARQUET), inv.MASTER_SHA256)
        self.assertEqual(inv.sha256_file(inv.MASTER_CSV), inv.MASTER_CSV_SHA256)


class RawInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inv_b1 = inv.archive_cycle_inventory(inv.B1_RAW)
        cls.inv_b3 = inv.archive_cycle_inventory(inv.B3_RAW)
        cls.inv_b4 = inv.archive_cycle_inventory(inv.B4_RAW)
        cls.inv_b5 = inv.archive_cycle_inventory(inv.B5_RAW)

    def test_workbook_and_cycle_sheet_counts(self):
        for label, result in (
            ("B1", self.inv_b1),
            ("B3", self.inv_b3),
            ("B4", self.inv_b4),
            ("B5", self.inv_b5),
        ):
            workbooks, sheets = inv.EXPECTED_CYCLE_SHEETS[label]
            self.assertEqual(result["workbook_count"], workbooks, label)
            self.assertEqual(result["cycle_sheet_count"], sheets, label)

    def test_counts_are_not_empty_per_workbook(self):
        for result in (self.inv_b1, self.inv_b3, self.inv_b4, self.inv_b5):
            self.assertGreater(len(result["workbooks"]), 0)
            for name, counts in result["workbooks"].items():
                self.assertGreaterEqual(counts["cycle_sheets"], 1, name)
                self.assertGreaterEqual(counts["sheets"], counts["cycle_sheets"], name)


class SampleReDerivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verified = inv.verify_sample_cycles()

    def test_b1_reproduces_audit_capacity(self):
        b1 = self.verified["B1"]
        self.assertEqual(b1["audit_checkpoint"], 1500)
        self.assertEqual(b1["audit_global_cycle_est"], 1401)
        self.assertAlmostEqual(b1["derived"]["discharge"]["capacity_ah"], 1.102908, places=5)
        self.assertAlmostEqual(b1["audit_capacity_ah"], 1.102908, places=6)
        self.assertLess(abs(b1["delta_capacity_ah"]), 1e-6)
        self.assertAlmostEqual(b1["derived"]["discharge"]["v_end"], 2.494421, places=4)

    def test_b3_reproduces_audit_capacity_and_soh(self):
        b3 = self.verified["B3"]
        self.assertAlmostEqual(b3["derived"]["discharge"]["capacity_ah"], 1.165491, places=6)
        self.assertAlmostEqual(b3["audit_capacity_ah"], 1.165491, places=6)
        self.assertLess(abs(b3["delta_capacity_ah"]), 1e-9)
        self.assertAlmostEqual(b3["derived"]["discharge"]["v_end"], 2.496826, places=4)
        self.assertAlmostEqual(b3["audit_soh_pct"], 93.239, places=3)
        self.assertEqual(b3["audit_quality_tier"], "A")

    def test_b4_raw_discharge_is_complete_despite_audit_block(self):
        b4 = self.verified["B4"]
        self.assertIsNone(b4["audit_capacity_ah"])
        discharge = b4["derived"]["discharge"]
        self.assertLess(discharge["v_end"], 2.5)
        self.assertGreater(discharge["capacity_ah"], 1.0)
        self.assertLess(discharge["capacity_ah"], 1.3)
        self.assertGreater(discharge["mean_abs_current_a"], 3.5)
        self.assertLess(discharge["mean_abs_current_a"], 4.0)

    def test_current_sign_convention_discharge_negative(self):
        for key in ("B1", "B3", "B4"):
            discharge = self.verified[key]["derived"]["discharge"]
            charge = self.verified[key]["derived"]["charge"]
            self.assertIsNotNone(discharge, key)
            self.assertIsNotNone(charge, key)
            self.assertGreater(charge["v_end"], discharge["v_end"], key)


class MasterCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = inv.master_facts()

    def test_master_shape(self):
        self.assertEqual(self.facts["rows"], 515_270)
        self.assertEqual(len(self.facts["columns"]), 13)

    def test_b1_single_cycle_in_master(self):
        b1 = self.facts["datasets"]["LFP_MultiStat_B1"]
        self.assertEqual(b1["rows"], 4_789)
        self.assertEqual(b1["observed_cycle_ids"], 1)
        self.assertEqual(b1["cell_cycle_pairs"], 1)
        self.assertEqual(b1["soh_null_rows"], 0)

    def test_b5_master_coverage(self):
        b5 = self.facts["datasets"]["LFP_MultiStat_B5"]
        self.assertEqual(b5["rows"], 333_344)
        self.assertEqual(b5["observed_cycle_ids"], 250)
        self.assertEqual(b5["cell_cycle_pairs"], 250)
        self.assertEqual(b5["labelled_cell_cycle_pairs"], 249)
        self.assertEqual(self.facts["b5_cycle_blocks"], [[1, 100], [151, 200], [401, 500]])
        self.assertEqual(self.facts["b5_labelled_cycles"], 249)
        self.assertEqual(self.facts["b5_next_observed_record_pairs"], 248)
        self.assertEqual(self.facts["b5_null_soh_cycles"], [40])
        self.assertEqual(self.facts["b5_null_soh_rows"], 625)

    def test_b5_missing_cycle_ids_are_absent_not_labeled(self):
        missing = set(self.facts["b5_missing_cycle_ids"])
        self.assertEqual(missing, set(range(101, 151)) | set(range(201, 401)))
        self.assertEqual(len(missing), 250)


class B5AuditSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = inv.b5_facts()

    def test_audit_cycle_rows_and_labels(self):
        self.assertEqual(self.facts["audit_cycle_rows"], 1_499)
        self.assertEqual(self.facts["audit_labelled_soh"], 1_492)
        self.assertEqual(
            self.facts["audit_excluded_cycles"],
            [40, 583, 749, 890, 1099, 1108, 1290],
        )

    def test_capacity_is_never_ground_truth(self):
        self.assertEqual(self.facts["capacity_ground_truth_values"], ["NO"])

    def test_charge_balance_discrepancy(self):
        self.assertEqual(self.facts["paired_cycles"], 1_498)
        self.assertEqual(self.facts["cycles_with_qdis_gt_qchg"], 1_488)
        self.assertAlmostEqual(self.facts["ce_median"], 1.027787, places=6)
        self.assertGreater(self.facts["cycles_with_qdis_gt_qchg"] / self.facts["paired_cycles"], 0.99)


class B2DerivedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.facts = inv.b2_facts()

    def test_cycle_counts(self):
        self.assertEqual(self.facts["derived_cycles"], 1_506)
        self.assertEqual(self.facts["cycles_in_analysis"], 1_496)
        self.assertEqual(self.facts["raw_workbooks_referenced"], 20)
        self.assertEqual(self.facts["raw_integrity_rows"], 20)

    def test_soh_scale_is_fraction(self):
        self.assertGreaterEqual(self.facts["soh_rated_min"], 0.80)
        self.assertLessEqual(self.facts["soh_rated_max"], 1.0)

    def test_channel_change_and_splits(self):
        self.assertEqual(self.facts["channel_change_cycle"], 999)
        self.assertEqual(self.facts["channels"], {"8": 998, "2": 508})
        self.assertEqual(list(self.facts["qv_curves_shape"]), [1_506, 3])
        self.assertEqual(self.facts["splits"]["train"]["n"], 1_047)
        self.assertEqual(self.facts["splits"]["val"]["n"], 224)
        self.assertEqual(self.facts["splits"]["test"]["n"], 225)

    def test_verification_all_passed(self):
        self.assertTrue(self.facts["verification_all_passed"])
        self.assertGreaterEqual(self.facts["verification_checks"], 14)


class B4GapTests(unittest.TestCase):
    def test_unaudited_workbooks_and_complete_discharges(self):
        gap = inv.b4_coverage_gap()
        self.assertEqual(gap["raw_workbooks"], 21)
        self.assertEqual(gap["audited_workbooks"], 16)
        self.assertEqual(
            gap["unaudited_workbooks"],
            ["Cycle_0150.xlsx", "Cycle_0200.xlsx", "Cycle_0250.xlsx",
             "Cycle_0300.xlsx", "Cycle_0350.xlsx"],
        )
        self.assertEqual(gap["audited_cycle_sheets"], 1_350)
        self.assertEqual(gap["audited_complete_discharges"], 1_347)
        self.assertEqual(gap["audited_no_discharge"], 3)
        self.assertLess(gap["raw_median_abs_discharge_current_a"], -3.5)
        self.assertGreater(gap["raw_median_charge_current_a"], 0.5)


class B6NoteTests(unittest.TestCase):
    def test_b6_raw_present_and_expected_counts(self):
        note = inv.b6_note()
        self.assertTrue(note["raw_present_in_workspace"])
        self.assertEqual(note["referenced_workbooks"], 19)
        self.assertEqual(note["cycle_like_sheets"], 1_702)
        self.assertEqual(note["complete_discharge_labels"], 1_697)


class DeliverableMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coverage = pd.read_csv(inv.AUDIT_DIR / "MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv")
        cls.features = pd.read_csv(inv.AUDIT_DIR / "MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv")

    def test_coverage_schema_and_rows(self):
        self.assertEqual(list(self.coverage.columns), inv.COVERAGE_COLUMNS)
        self.assertEqual(len(self.coverage), 5)
        self.assertEqual(
            sorted(self.coverage["cell_id"]), ["BATTERY_1", "BATTERY_2", "BATTERY_3", "BATTERY_4", "BATTERY_5"]
        )

    def test_coverage_suitability_statuses(self):
        actual = dict(zip(self.coverage["cell_id"], self.coverage["independent_validation_suitability"]))
        self.assertEqual(actual, EXPECTED_COVERAGE_STATUS)

    def test_coverage_master_row_counts(self):
        rows = dict(zip(self.coverage["cell_id"], self.coverage["row_count"]))
        self.assertEqual(rows["BATTERY_1"], 4_789)
        self.assertEqual(rows["BATTERY_5"], 333_344)
        for cell in ("BATTERY_2", "BATTERY_3", "BATTERY_4"):
            self.assertEqual(rows[cell], 0, cell)

    def test_no_guessed_values_use_explicit_unknown(self):
        for column in ("blocker", "recommended_action"):
            self.assertFalse(self.coverage[column].isna().any(), column)
        joined = " ".join(self.coverage["current_unit_status"].tolist())
        self.assertIn("AMPERES", joined)
        self.assertIn("label error", joined)

    def test_feature_matrix_schema_and_rows(self):
        self.assertEqual(list(self.features.columns), inv.FEATURE_COLUMNS)
        self.assertEqual(len(self.features), 70)
        counts = self.features.groupby("cell_id").size().to_dict()
        self.assertEqual(counts, {f"BATTERY_{i}": 14 for i in range(1, 6)})

    def test_feature_matrix_uses_controlled_status_vocabulary(self):
        allowed = {"READY", "READY_WITH_CAVEAT", "CONDITIONAL", "BLOCKED", "NOT_AVAILABLE"}
        self.assertTrue(set(self.features["suitability"]).issubset(allowed))

    def test_temperature_features_blocked_without_measurement(self):
        temp = self.features[self.features["feature_name"].str.contains("temperature", case=False)]
        self.assertGreater(len(temp), 0)
        self.assertTrue((temp["suitability"] == "NOT_AVAILABLE").all())


class EvidenceFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = inv.OUT_DIR / "lfp_b1_b5_evidence.json"
        cls.evidence = json.loads(path.read_text(encoding="utf-8"))

    def test_guardrails_recorded(self):
        guard = self.evidence["guardrails"]
        self.assertTrue(guard["hashes_match_expected"])
        self.assertEqual(guard["master_table_sha256"]["parquet"], inv.MASTER_SHA256)
        self.assertFalse(guard["member2_artifacts_modified"])

    def test_workspace_archive_hashes(self):
        records = self.evidence["workspace_archive_hashes"]
        self.assertEqual(set(records), {"B1_RAW", "B2_DERIVED"})
        for key, record in records.items():
            self.assertTrue(record["matches_expected"], key)
            self.assertEqual(record["sha256"], inv.EXPECTED_WORKSPACE_SHA256[key], key)

    def test_attachments_are_byte_identical(self):
        for key, record in self.evidence["attachments_reconciled"]["archive_hashes"].items():
            self.assertTrue(record["matches_expected"], key)

    def test_evidence_sample_deltas(self):
        samples = self.evidence["independent_sample_verification"]
        self.assertLess(abs(samples["B1"]["delta_capacity_ah"]), 1e-6)
        self.assertLess(abs(samples["B3"]["delta_capacity_ah"]), 1e-9)
        self.assertTrue(math.isclose(samples["B4"]["derived"]["discharge"]["capacity_ah"],
                                     1.1864182094444444, rel_tol=0, abs_tol=1e-12))


if __name__ == "__main__":
    unittest.main(verbosity=2)
