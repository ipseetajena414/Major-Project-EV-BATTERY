"""
Member 1 - end-to-end pipeline tests.

Executed against produced artefacts (canonical master, candidate master,
per-dataset candidate exports, coverage/inventory tables, handoff documents).

Run:
    python -m unittest AUDIT.test_member1_pipeline -v
or:
    python AUDIT/test_member1_pipeline.py
"""

from __future__ import annotations

import hashlib
import io
import json
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT = REPO_ROOT / "AUDIT"
CAND = AUDIT / "candidates"

CANONICAL_PARQUET = REPO_ROOT / "master_table.parquet"
CANONICAL_CSV = REPO_ROOT / "master_table.csv"
CAND_PARQUET = REPO_ROOT / "master_table_candidate.parquet"
CAND_VALIDATION = REPO_ROOT / "master_table_candidate_validation.json"
CAND_MANIFEST = REPO_ROOT / "master_table_candidate_manifest.csv"
COVERAGE = AUDIT / "MEMBER1_DATASET_COVERAGE.csv"
BASELINE = AUDIT / "MEMBER1_BASELINE.json"
FEATURE_MANIFEST = AUDIT / "MEMBER1_FEATURE_TARGET_MANIFEST.csv"
TARGET_DEFS = AUDIT / "MEMBER1_TARGET_DEFINITIONS.csv"

BASELINE_PARQUET_SHA = "6fa7475d1810007803a0f55431730fdedbb2eec7127cecddf3219a4d9073b2bb"
BASELINE_CSV_SHA = "9e23792615b2f08692944680d041957f776428b7563a90b214b77ee74629895b"

MASTER_SCHEMA = [
    "chemistry", "dataset_id", "cell_id", "protocol_id", "cycle_id", "time_sec",
    "voltage_v", "current_a", "temp_c", "capacity_ah", "soh_pct", "quality_flag",
    "source_file",
]

EXPECTED_COVERAGE = {
    "LFP_MultiStat_B1": ("CONDITIONAL", "CONDITIONAL"),
    "LFP_MultiStat_B2": ("CONDITIONAL", "CONDITIONAL"),
    "LFP_MultiStat_B3": ("CONDITIONAL", "CONDITIONAL"),
    "LFP_MultiStat_B4": ("CONDITIONAL", "BLOCKED"),
    "LFP_MultiStat_B5": ("CONDITIONAL", "CONDITIONAL"),
    "LFP_MultiStat_B6": ("CONDITIONAL", "CONDITIONAL"),
    "NASA_Ames_PCoE": ("VERIFIED", "VERIFIED"),
    "Zenodo_IonTech_NaIon": ("VERIFIED", "VERIFIED"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_zip_parquet(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        raw = archive.read(member)
    return pd.read_parquet(io.BytesIO(raw))


class TestCanonicalIntegrity(unittest.TestCase):
    def test_01_canonical_parquet_hash_unchanged(self):
        self.assertEqual(sha256_file(CANONICAL_PARQUET), BASELINE_PARQUET_SHA)

    def test_02_canonical_csv_hash_unchanged(self):
        self.assertEqual(sha256_file(CANONICAL_CSV), BASELINE_CSV_SHA)

    def test_03_canonical_row_count(self):
        df = pd.read_parquet(CANONICAL_PARQUET, columns=["dataset_id"])
        self.assertEqual(len(df), 515270)

    def test_04_baseline_file_agrees(self):
        if not BASELINE.exists():
            self.skipTest("MEMBER1_BASELINE.json not present")
        baseline = read_json(BASELINE)
        self.assertEqual(baseline.get("rows"), 515270)
        self.assertTrue(baseline.get("master_parquet_sha256_matches_baseline"))
        self.assertTrue(baseline.get("master_csv_sha256_matches_baseline"))


class TestCandidateStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cand = pd.read_parquet(CAND_PARQUET)
        cls.validation = read_json(CAND_VALIDATION)
        cls.manifest = pd.read_csv(CAND_MANIFEST)

    def test_05_schema_exact_13_columns(self):
        self.assertEqual(list(self.cand.columns), MASTER_SCHEMA)
        self.assertTrue(self.validation["schema_conformity"])

    def test_06_row_count_matches_manifest(self):
        self.assertEqual(len(self.cand), 1148003)
        self.assertEqual(int(self.manifest["rows"].sum()), len(self.cand))
        self.assertEqual(self.validation["candidate_rows"], len(self.cand))

    def test_07_candidate_parquet_sha_matches_validation(self):
        self.assertEqual(
            self.validation["candidate_parquet_sha256"], sha256_file(CAND_PARQUET)
        )

    def test_08_no_duplicate_keys(self):
        dup = self.cand.duplicated(
            subset=["dataset_id", "cell_id", "cycle_id", "time_sec"]
        ).sum()
        self.assertEqual(int(dup), 0)
        self.assertEqual(self.validation["duplicate_keys"], 0)

    def test_09_all_eight_datasets_present(self):
        self.assertEqual(
            set(self.cand["dataset_id"].unique()), set(EXPECTED_COVERAGE)
        )

    def test_10_value_bounds(self):
        self.assertTrue((self.cand["voltage_v"] > 0).all())
        self.assertTrue((self.cand["voltage_v"] < 6).all())
        self.assertTrue((self.cand["current_a"].abs() < 10).all())
        temp = self.cand["temp_c"].dropna()
        self.assertTrue(((temp > -50) & (temp < 100)).all())
        self.assertTrue((self.cand["capacity_ah"].dropna() >= 0).all())
        soh = self.cand["soh_pct"].dropna()
        self.assertTrue(((soh >= 0) & (soh <= 150)).all())

    def test_11_time_non_negative_and_finite(self):
        self.assertTrue(np.isfinite(self.cand["time_sec"]).all())
        self.assertTrue((self.cand["time_sec"] >= 0).all())

    def test_12_quality_flag_never_empty(self):
        self.assertFalse(self.cand["quality_flag"].isna().any())
        self.assertFalse((self.cand["quality_flag"].astype(str).str.strip() == "").any())

    def test_13_lfp_temperature_all_null(self):
        lfp = self.cand[self.cand["dataset_id"].str.startswith("LFP_")]
        self.assertEqual(int(lfp["temp_c"].notna().sum()), 0)

    def test_14_non_lfp_rows_preserved_exactly(self):
        canonical = pd.read_parquet(CANONICAL_PARQUET)
        keys = ["dataset_id", "cell_id", "cycle_id", "time_sec", "voltage_v", "current_a"]
        for dataset in ["NASA_Ames_PCoE", "Zenodo_IonTech_NaIon"]:
            a = (
                canonical[canonical["dataset_id"] == dataset]
                .sort_values(keys, kind="mergesort")
                .reset_index(drop=True)
            )
            b = (
                self.cand[self.cand["dataset_id"] == dataset]
                .sort_values(keys, kind="mergesort")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
            self.assertEqual(a["capacity_ah"].notna().sum(), b["capacity_ah"].notna().sum())


class TestB2HighestPriority(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.labels = pd.read_csv(CAND / "B2_candidate_cycle_labels.csv")
        cls.mask = pd.read_csv(CAND / "B2_channel_transition_mask.csv")
        cls.summary = read_json(CAND / "B2_channel_transition_summary.json")
        cls.features = read_zip_parquet(
            REPO_ROOT / "DATA" / "B2_LFP_Member1_audit.zip",
            "LFP_Member1_audit/exports/cycle_features.parquet",
        )
        cls.qv = read_zip_parquet(
            REPO_ROOT / "DATA" / "B2_LFP_Member1_audit.zip",
            "LFP_Member1_audit/exports/qv_curves.parquet",
        )

    def test_15_feature_and_curve_shapes(self):
        self.assertEqual(self.features.shape, (1506, 255))
        self.assertEqual(self.qv.shape, (1506, 3))
        self.assertEqual(list(self.qv.columns), ["cycle_index", "discharge_qv", "charge_qv"])

    def test_16_invalid_and_analysis_counts(self):
        invalid = self.features.loc[~self.features["valid_cycle"], "cycle_index"].tolist()
        self.assertEqual(len(invalid), 10)
        self.assertEqual(
            invalid, [41, 101, 151, 972, 998, 999, 1185, 1186, 1411, 1412]
        )
        self.assertEqual(int(self.features["in_analysis"].sum()), 1496)

    def test_17_channel_transition_first_ch2_at_999(self):
        ch2 = self.features[self.features["channel"] == 2]["cycle_index"]
        self.assertEqual(int(ch2.min()), 999)
        counts = self.features["channel"].value_counts().to_dict()
        self.assertEqual(counts[8.0], 998)
        self.assertEqual(counts[2.0], 508)

    def test_18_treatment_counts_and_mask_consistency(self):
        self.assertEqual(self.summary["treatment_exclusion"]["cycles"], 1480)
        self.assertEqual(self.summary["treatment_channel_covariate"]["cycles"], 1496)
        self.assertEqual(self.summary["transition_window_cycles"], 17)
        self.assertEqual(len(self.mask), 1506)
        self.assertIn("EXCLUDE", self.summary["recommendation"])

    def test_19_soh_rated_is_fraction(self):
        soh = self.features["soh_rated"].dropna()
        self.assertGreaterEqual(soh.min(), 0.0)
        self.assertLessEqual(soh.max(), 1.0)
        self.assertLess(soh.max(), 1.01)

    def test_20_leakage_manifest(self):
        manifest = pd.read_csv(FEATURE_MANIFEST)
        roles = dict(zip(manifest["column"], manifest["role"]))
        for target in ["q_discharge_ah", "soh_rated"]:
            self.assertEqual(roles[target], "TARGET")
        for leak in [
            "soh_initial_ref",
            "soh_slope_per_100cyc",
            "q_fade_rate_ah_per_100cyc",
        ]:
            self.assertIn("LEAKAGE", roles[leak])
        self.assertTrue((manifest["role"] != "FEATURE_SAFE").any())
        self.assertGreaterEqual(len(manifest), 250)

    def test_21_b2_labels_match_feature_rows(self):
        self.assertEqual(len(self.labels), 1506)
        self.assertEqual(int(self.labels["in_analysis"].sum()), 1496)
        self.assertEqual(int((~self.labels["valid_cycle"]).sum()), 10)


class TestB1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = read_json(CAND / "member1_cycle_candidates_summary.json")["B1"]
        cls.labels = pd.read_csv(CAND / "B1_candidate_cycle_labels.csv")
        cls.telemetry = pd.read_parquet(
            CAND / "telemetry_B1.parquet",
            columns=["cycle_id", "capacity_ah", "soh_pct", "quality_flag"],
        )

    def test_22_b1_coverage(self):
        self.assertEqual(self.summary["cycles"], 1460)
        self.assertEqual(self.summary["capacity_non_null"], 1456)
        self.assertEqual(self.telemetry["cycle_id"].nunique(), 1460)
        self.assertEqual(len(self.telemetry), 145901)

    def test_23_b1_mapping_bug_fixed(self):
        cycle2 = self.telemetry[self.telemetry["cycle_id"] == 2]
        self.assertFalse(cycle2.empty)
        value = cycle2["capacity_ah"].dropna().unique()
        self.assertEqual(len(value), 1)
        self.assertAlmostEqual(float(value[0]), 1.220155, places=5)
        self.assertNotAlmostEqual(float(value[0]), 1.202223, places=5)
        self.assertGreater(self.telemetry["capacity_ah"].nunique(), 100)

    def test_24_b1_flags(self):
        self.assertEqual(self.summary["blocked_partial"], 4)
        self.assertEqual(self.summary["offset_underdetermined"], 110)
        self.assertEqual(
            int(self.labels["label_status"].eq("BLOCKED_PARTIAL_DISCHARGE").sum()), 4
        )
        self.assertEqual(
            int(self.labels["label_status"].eq("CONDITIONAL_OFFSET_UNDERDETERMINED").sum()),
            110,
        )

    def test_36_b1_cycle_1201_1220_gap_documented(self):
        evidence = read_json(AUDIT / "MEMBER1_B1_CYCLE1300_GAP_EVIDENCE.json")
        self.assertEqual(evidence["decision"], "RETAIN_GAP_NOT_INTEGRATED")
        self.assertEqual(evidence["inventory"]["generic_cycle_export_tabs"], 20)
        self.assertEqual(evidence["inventory"]["step2_cycle_tabs"], 80)
        self.assertEqual(evidence["export_index"]["cycles_listed"], 21)
        exports = evidence["generic_cycle_exports"]
        self.assertEqual(len(exports), 20)
        self.assertEqual(exports[0]["date"], "23/09/2025")
        self.assertEqual(exports[-1]["date"], "26/09/2025")
        gap = self.telemetry[self.telemetry["cycle_id"].between(1201, 1220)]
        self.assertTrue(gap.empty)
        self.assertFalse(evidence["impact_on_candidate"]["cycle_ids_1201_to_1220_present"])


class TestB3(unittest.TestCase):
    def test_25_b3_missing_ids_and_four_channels(self):
        summary = read_json(CAND / "member1_cycle_candidates_summary.json")["B3"]
        self.assertEqual(summary["missing_cycle_ids"], list(range(201, 222)))
        self.assertEqual(summary["cycles"], 1979)
        self.assertEqual(summary["capacity_non_null"], 1973)
        self.assertEqual(len(summary["channels"]), 4)
        self.assertAlmostEqual(summary["capacity_range"][1], 1.165723, places=5)


class TestB4(unittest.TestCase):
    def test_26_b4_targets_blocked(self):
        telemetry = pd.read_parquet(
            CAND / "telemetry_B4.parquet",
            columns=["cycle_id", "capacity_ah", "soh_pct"],
        )
        self.assertEqual(telemetry["cycle_id"].nunique(), 1577)
        self.assertEqual(int(telemetry["capacity_ah"].notna().sum()), 0)
        self.assertEqual(int(telemetry["soh_pct"].notna().sum()), 0)
        coverage = pd.read_csv(COVERAGE).set_index("dataset_id")
        self.assertEqual(coverage.loc["LFP_MultiStat_B4", "target_status"], "BLOCKED")


class TestB5(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = read_json(CAND / "member1_cycle_candidates_summary.json")["B5"]
        cls.labels = pd.read_csv(CAND / "B5_candidate_cycle_labels.csv")

    def test_27_b5_cycle_count_and_file_list_dropped(self):
        self.assertEqual(self.summary["cycles_exported"], 1498)
        self.assertEqual(self.summary["audit_rows_dropped_as_non_cycle"], 1)
        self.assertEqual(
            self.summary["dropped_rows"][0]["worksheet"], "File List"
        )
        self.assertEqual(len(self.labels), 1498)
        self.assertEqual(len(self.labels["cycle_id"].unique()), 1498)

    def test_28_b5_cycle40_interruption(self):
        row = self.labels[
            (self.labels["source_file"] == "Cycle_0050.xlsx")
            & (self.labels["worksheet"] == "Run01_Un1Ch1_Step2_Cycle40")
        ]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(
            float(row["discharge_capacity_ah_conditional"].iloc[0]), 0.5428382, places=7
        )
        self.assertFalse(bool(row["dis_complete"].iloc[0]))
        self.assertIn("DISCHARGE_INTERRUPTED", row["quality_flags"].iloc[0])

    def test_29_b5_qdis_exceeds_qchg_systematically(self):
        d = self.labels["discharge_capacity_ah_conditional"]
        c = self.labels["charge_capacity_ah_unsupported_diagnostic"]
        self.assertEqual(int(d.notna().sum()), 1497)
        valid = d.notna() & c.notna()
        self.assertEqual(int(valid.sum()), 1497)
        self.assertEqual(int((d[valid] > c[valid]).sum()), 1488)
        self.assertEqual(int((self.labels["capacity_ground_truth"] == "NO").sum()), 1498)


class TestB6(unittest.TestCase):
    def test_30_b6_chronology_valid(self):
        summary = read_json(CAND / "B6_candidate_chronology_summary.json")
        self.assertEqual(summary["cycle_rows"], 1702)
        self.assertEqual(summary["candidate_order_validation"]["unorderable_sheets"], 0)
        self.assertEqual(summary["candidate_order_validation"]["backwards_timestamps_in_candidate_order"], 0)
        self.assertEqual(summary["timestamp_reconciliation"]["header_vs_file_list_mismatches"], 0)
        self.assertEqual(summary["labels"]["conditional_nominal_soh"], 1697)


class TestCrossDatasetAndMeta(unittest.TestCase):
    def test_31_coverage_table_statuses(self):
        coverage = pd.read_csv(COVERAGE).set_index("dataset_id")
        self.assertEqual(set(coverage.index), set(EXPECTED_COVERAGE))
        for dataset, (status, target) in EXPECTED_COVERAGE.items():
            self.assertEqual(coverage.loc[dataset, "status"], status, dataset)
            self.assertEqual(coverage.loc[dataset, "target_status"], target, dataset)
            self.assertTrue(bool(coverage.loc[dataset, "included_in_candidate_master"]))
        lfp = coverage[coverage.index.str.startswith("LFP_")]
        self.assertFalse(lfp["temperature_measured"].astype(str).str.upper().eq("YES").any())

    def test_32_telemetry_summaries_complete(self):
        for tag in ["B1", "B2", "B3", "B4", "B5", "B6"]:
            summary = read_json(CAND / f"telemetry_{tag}_summary.json")
            self.assertGreater(summary["rows_written"], 100000, tag)
            self.assertGreater(summary["cycles_in_output"], 1000, tag)
            self.assertEqual(summary["labels_without_raw_sheet_count"], 0, tag)

    def test_33_target_definitions_complete(self):
        defs = pd.read_csv(TARGET_DEFS)
        self.assertEqual(set(defs["dataset_id"]), set(EXPECTED_COVERAGE))
        b2 = defs[defs["dataset_id"] == "LFP_MultiStat_B2"].iloc[0]
        self.assertIn("1.25", b2["formula"])
        self.assertEqual(b2["status"], "CONDITIONAL")
        b4 = defs[defs["dataset_id"] == "LFP_MultiStat_B4"].iloc[0]
        self.assertEqual(b4["status"], "BLOCKED")
        self.assertIn("not derived", b4["formula"].lower())

    def test_34_handoff_contracts_present(self):
        m12 = REPO_ROOT / "MEMBER1_MEMBER2_HANDOFF_CONTRACT.md"
        m23 = REPO_ROOT / "MEMBER2_MEMBER3_INTERFACE_DATA_CONTRACT.md"
        for path in [m12, m23]:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 4000, path)
        text12 = m12.read_text(encoding="utf-8")
        text23 = m23.read_text(encoding="utf-8")
        self.assertIn("soh_pct", text12)
        self.assertIn("B4", text12)
        self.assertIn("0-100", text23.replace("–", "-").replace("—", "-"))
        self.assertIn("pack", text23.lower())

    def test_35_inventory_covers_all_sources(self):
        inventory = pd.read_csv(AUDIT / "MEMBER1_SOURCE_INVENTORY.csv")
        ids = set(inventory["dataset_id"])
        self.assertTrue(set(EXPECTED_COVERAGE).issubset(ids))
        self.assertGreaterEqual(inventory["sha256"].notna().sum(), 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
