"""
Member 1 - tests for the B2/B6 raw-archive follow-up.

Verifies archive identity, candidate-table structure, chronology invariants,
matrix extension and canonical-data immutability. Read-only.

Run:
    python -m unittest AUDIT/test_lfp_b2_b6_followup.py -v
  or:
    python AUDIT/test_lfp_b2_b6_followup.py
"""

from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lfp_b2_candidate as b2  # noqa: E402
import lfp_b6_chronology as b6  # noqa: E402
import lfp_b1_b5_inventory as inv  # noqa: E402

AUDIT_DIR = Path(__file__).resolve().parent
CAND_DIR = AUDIT_DIR / "candidates"
B2_CANDIDATE = CAND_DIR / "B2_candidate_cycle_labels.csv"
B6_CANDIDATE = CAND_DIR / "B6_candidate_cycle_chronology.csv"
B2_SUMMARY = CAND_DIR / "B2_candidate_summary.json"
B6_SUMMARY = CAND_DIR / "B6_candidate_chronology_summary.json"
COVERAGE = AUDIT_DIR / "MEMBER1_LFP_B1_B6_COVERAGE_MATRIX.csv"
FEATURES = AUDIT_DIR / "MEMBER1_LFP_B1_B6_FEATURE_READINESS_MATRIX.csv"
CHANGELOG = AUDIT_DIR / "MEMBER1_LFP_B1_B6_MATRIX_CHANGELOG.json"


class ArchiveIdentityTests(unittest.TestCase):
    def test_b2_archive_hash(self):
        self.assertTrue(b2.B2_RAW.exists())
        self.assertEqual(b2.sha256_file(b2.B2_RAW), b2.B2_RAW_SHA256)

    def test_b2_derived_archive_hash(self):
        self.assertEqual(b2.sha256_file(b2.B2_DERIVED), b2.B2_DERIVED_SHA256)

    def test_b6_archive_hash(self):
        self.assertTrue(b6.B6_RAW.exists())
        self.assertEqual(b6.sha256_file(b6.B6_RAW), b6.B6_RAW_SHA256)

    def test_b6_audit_package_hash(self):
        self.assertEqual(b6.sha256_file(b6.B6_AUDIT), b6.B6_AUDIT_SHA256)

    def test_canonical_master_unchanged(self):
        self.assertEqual(inv.sha256_file(inv.MASTER_PARQUET), inv.MASTER_SHA256)
        self.assertEqual(inv.sha256_file(inv.MASTER_CSV), inv.MASTER_CSV_SHA256)


class B2CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(B2_CANDIDATE)
        cls.summary = json.loads(B2_SUMMARY.read_text(encoding="utf-8"))

    def test_columns_and_rows(self):
        self.assertEqual(len(self.df), 1506)
        self.assertEqual(list(self.df.columns), b2.CANDIDATE_COLUMNS)

    def test_cycle_index_contiguous(self):
        self.assertEqual(self.df["cycle_index"].tolist(), list(range(1, 1507)))

    def test_label_counts(self):
        counts = self.df["label_status"].value_counts().to_dict()
        self.assertEqual(counts.get("conditional_nominal_soh"), 1496)
        self.assertEqual(counts.get("excluded_flagged"), 10)

    def test_raw_manifest_verified(self):
        self.assertTrue(self.summary["raw_manifest"]["all_hashes_match"])
        self.assertEqual(self.summary["raw_manifest"]["workbooks"], 20)
        self.assertEqual(self.summary["raw_manifest"]["missing_from_raw"], [])
        self.assertEqual(self.summary["raw_manifest"]["extra_in_raw"], [])

    def test_spot_checks_match(self):
        checks = self.summary["spot_checks"]
        self.assertEqual([c["cycle_index"] for c in checks], [50, 1506])
        for check in checks:
            self.assertTrue(check["matches"])
            self.assertLessEqual(check["abs_diff_ah"], 1e-6)

    def test_excluded_outlier_is_flagged_not_deleted(self):
        row = self.df.loc[self.df["cycle_index"] == 1411].iloc[0]
        self.assertFalse(bool(row["in_analysis"]))
        self.assertAlmostEqual(row["soh_rated_pct"], 7.021551, places=5)

    def test_in_analysis_soh_range(self):
        low, high = self.summary["in_analysis_soh_rated_pct_range"]
        self.assertAlmostEqual(low, 85.543809, places=5)
        self.assertAlmostEqual(high, 96.681094, places=5)


class B6ChronologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(B6_CANDIDATE)
        cls.summary = json.loads(B6_SUMMARY.read_text(encoding="utf-8"))

    def test_columns_and_rows(self):
        self.assertEqual(len(self.df), 1702)
        self.assertEqual(list(self.df.columns), b6.CANDIDATE_COLUMNS)

    def test_candidate_index_unique_and_dense(self):
        self.assertEqual(self.df["candidate_global_index"].tolist(), list(range(1, 1703)))

    def test_no_duplicate_or_backwards_timestamps(self):
        checks = self.summary["candidate_order_validation"]
        self.assertEqual(checks["duplicate_timestamps"], 0)
        self.assertEqual(checks["backwards_timestamps_in_candidate_order"], 0)
        self.assertEqual(checks["within_workbook_tab_order_backwards"], 0)

    def test_checkpoint_blocks_contiguous(self):
        checks = self.summary["candidate_order_validation"]
        self.assertEqual(checks["checkpoint_regressions_in_candidate_order"], 0)
        self.assertEqual(checks["non_contiguous_checkpoint_blocks"], {})
        grouped = self.df.groupby("workbook_checkpoint")["candidate_global_index"].agg(["min", "max", "count"])
        for _, row in grouped.iterrows():
            self.assertEqual(row["max"] - row["min"] + 1, row["count"])

    def test_header_file_list_agreement(self):
        recon = self.summary["timestamp_reconciliation"]
        self.assertEqual(recon["header_vs_file_list_mismatches"], 0)
        self.assertEqual(recon["sheets_without_any_timestamp"], 0)
        self.assertEqual(recon["sheets_with_both_sources"], 1620)

    def test_generic_sheets_present_and_ordered(self):
        generic = self.df[self.df["sheet_type"] == "generic_sheet_tab"]
        self.assertEqual(len(generic), 16)
        self.assertTrue(generic["worksheet"].str.match(r"^Sheet\d+$").all())
        self.assertTrue((generic["candidate_global_index"].between(602, 617)).all())
        ordered = generic.sort_values("candidate_global_index")
        self.assertEqual(ordered["worksheet"].tolist(), [f"Sheet{i}" for i in range(16, 0, -1)])

    def test_label_counts_and_range(self):
        counts = self.df["label_status"].value_counts().to_dict()
        self.assertEqual(counts.get("conditional_nominal_soh"), 1697)
        self.assertEqual(counts.get("unlabelled_screen_failed"), 5)
        labels = self.df["soh_nominal_pct_conditional"].dropna()
        self.assertAlmostEqual(labels.min(), 81.698938, places=5)
        self.assertAlmostEqual(labels.median(), 87.988006, places=5)
        self.assertAlmostEqual(labels.max(), 99.652912, places=5)

    def test_no_global_cycle_id_manufactured(self):
        self.assertFalse(any(c.lower() == "global_cycle_id" for c in self.df.columns))
        self.assertIn("candidate_global_index", self.df.columns)

    def test_workbook_record_counts(self):
        counts = self.df.groupby("source_file").size().to_dict()
        self.assertEqual(counts["Cycle_0050.xlsx"], 51)
        self.assertEqual(counts["Cycle_0700.xlsx"], 101)
        self.assertEqual(sum(1 for v in counts.values() if v == 100), 14)


class MatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coverage = pd.read_csv(COVERAGE)
        cls.features = pd.read_csv(FEATURES)
        cls.changelog = json.loads(CHANGELOG.read_text(encoding="utf-8"))

    def test_coverage_has_six_cells(self):
        self.assertEqual(
            set(self.coverage["cell_id"]),
            {f"BATTERY_{i}" for i in range(1, 7)},
        )
        self.assertEqual(len(self.coverage), 6)

    def test_coverage_columns_and_change_markers(self):
        self.assertIn("change_vs_b1_b5_report", self.coverage.columns)
        self.assertIn("source_cycle_records", self.coverage.columns)
        self.assertTrue((self.coverage["change_vs_b1_b5_report"].notna()).all())
        b2_row = self.coverage.loc[self.coverage["cell_id"] == "BATTERY_2"].iloc[0]
        self.assertEqual(b2_row["source_status"], "SOURCE_VERIFIED_RAW_AND_DERIVED")
        self.assertEqual(b2_row["source_cycle_records"], 1506)
        b6_row = self.coverage.loc[self.coverage["cell_id"] == "BATTERY_6"].iloc[0]
        self.assertEqual(b6_row["source_cycle_records"], 1702)
        self.assertEqual(b6_row["source_labelled_records"], 1697)

    def test_feature_matrix_includes_b6(self):
        self.assertEqual(len(self.features), 85)
        self.assertEqual((self.features["cell_id"] == "BATTERY_6").sum(), 15)
        self.assertIn("change_vs_b1_b5_report", self.features.columns)
        b6 = self.features[self.features["cell_id"] == "BATTERY_6"]
        self.assertIn("efc_cycle_index", set(b6["feature_name"]))
        self.assertEqual(
            b6.loc[b6["feature_name"] == "temperature", "suitability"].iloc[0],
            "NOT_AVAILABLE",
        )

    def test_b2_feature_updates_are_marked(self):
        b2 = self.features[self.features["cell_id"] == "BATTERY_2"]
        voltage = b2.loc[b2["feature_name"] == "voltage_series"].iloc[0]
        self.assertEqual(voltage["suitability"], "READY_WITH_CAVEAT")
        self.assertEqual(voltage["input_availability"], "AVAILABLE_IN_SOURCE_ONLY")
        self.assertIn("CHANGED", voltage["change_vs_b1_b5_report"])

    def test_changelog_supersedes_without_overwriting(self):
        self.assertIn("MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv", self.changelog["supersedes"])
        self.assertIn("MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv", self.changelog["supersedes"])
        self.assertTrue((AUDIT_DIR / "MEMBER1_LFP_B1_B5_COVERAGE_MATRIX.csv").exists())
        self.assertTrue((AUDIT_DIR / "MEMBER1_LFP_FEATURE_READINESS_MATRIX.csv").exists())


class CsvParseTests(unittest.TestCase):
    def test_all_new_csvs_parse_with_expected_rows(self):
        for path, expected in (
            (B2_CANDIDATE, 1506),
            (B6_CANDIDATE, 1702),
            (COVERAGE, 6),
            (FEATURES, 85),
        ):
            with open(path, encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(len(rows) - 1, expected, path.name)
            self.assertGreater(len(rows[0]), 5, path.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
