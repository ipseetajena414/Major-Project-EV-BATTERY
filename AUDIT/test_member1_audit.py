"""
Member 1 - tests for the read-only master-table audit.

These tests assert explicit, verified expectations against the canonical
master_table.parquet and the audit bundles. They never modify data.

Run:
    python -m unittest AUDIT/test_member1_audit.py -v
  or:
    python AUDIT/test_member1_audit.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import member1_audit as audit  # noqa: E402


class MasterTableAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = audit.load_master()
        cls.report = audit.run_audit()
        cls.checks = cls.report["checks"]

    # --- integrity -----------------------------------------------------
    def test_parquet_sha256_unchanged(self):
        self.assertEqual(
            self.report["checksums_sha256"]["parquet"],
            audit.EXPECTED_SHA256_PARQUET,
        )

    def test_csv_sha256_unchanged(self):
        self.assertEqual(
            self.report["checksums_sha256"]["csv"],
            audit.EXPECTED_SHA256_CSV,
        )

    def test_shape_and_schema(self):
        self.assertEqual(self.df.shape, (515_270, 13))
        self.assertEqual(list(self.df.columns), audit.MASTER_SCHEMA)

    def test_all_contract_assertions_pass(self):
        failed = [a for a in self.report["assertions"] if not a["passed"]]
        self.assertEqual(failed, [], msg=f"failed assertions: {failed}")

    # --- coverage ------------------------------------------------------
    def test_dataset_coverage(self):
        for dataset_id, expected in audit.EXPECTED_DATASETS.items():
            group = self.df[self.df["dataset_id"] == dataset_id]
            self.assertEqual(len(group), expected["rows"], dataset_id)
            self.assertEqual(group["cell_id"].nunique(), expected["cells"], dataset_id)
            self.assertEqual(
                group.groupby(["cell_id", "cycle_id"]).ngroups,
                expected["cell_cycle_pairs"],
                dataset_id,
            )
            self.assertEqual(
                group["chemistry"].unique().tolist(), [expected["chemistry"]], dataset_id
            )

    def test_b5_cycle_id_blocks_and_gaps(self):
        b5 = self.df[self.df["dataset_id"] == "LFP_MultiStat_B5"]
        ids = sorted(int(v) for v in b5["cycle_id"].unique())
        self.assertEqual(audit._contiguous_blocks(ids), audit.B5_EXPECTED_BLOCKS)
        missing = set(range(1, 501)) - set(ids)
        self.assertEqual(missing, set(range(101, 151)) | set(range(201, 401)))

    # --- labels --------------------------------------------------------
    def test_label_is_constant_within_cycle(self):
        for dataset_id, group in self.df.groupby("dataset_id"):
            agg = group.groupby(["cell_id", "cycle_id"]).agg(
                cap=("capacity_ah", "nunique"), soh=("soh_pct", "nunique")
            )
            self.assertEqual(agg["cap"].max(), 1, dataset_id)
            self.assertEqual(agg["soh"].max(), 1, dataset_id)

    def test_b5_null_soh_only_cycle_40(self):
        b5 = self.df[self.df["dataset_id"] == "LFP_MultiStat_B5"]
        cycles = sorted(int(c) for c in b5[b5["soh_pct"].isna()]["cycle_id"].unique())
        self.assertEqual(cycles, [40])

    def test_b5_cycle_40_is_partial_label_not_missing_cycle(self):
        # capacity present but soh null -> partially missing label, not a missing cycle
        c40 = self.df[
            (self.df["dataset_id"] == "LFP_MultiStat_B5") & (self.df["cycle_id"] == 40)
        ]
        self.assertEqual(len(c40), 625)
        self.assertAlmostEqual(float(c40["capacity_ah"].iloc[0]), 0.5428382, places=6)
        self.assertTrue(c40["soh_pct"].isna().all())

    def test_soh_equals_capacity_over_manufacturer_nominal(self):
        for dataset_id, rated in audit.NOMINAL_CAPACITY_AH.items():
            paired = self.df[
                (self.df["dataset_id"] == dataset_id)
                & self.df["capacity_ah"].notna()
                & self.df["soh_pct"].notna()
            ]
            if paired.empty:
                continue
            residual = (
                paired["soh_pct"] - paired["capacity_ah"] / rated * 100.0
            ).abs()
            self.assertLess(residual.max(), 1e-3, dataset_id)

    # --- source-level cross-checks ------------------------------------
    def test_b5_master_capacity_matches_raw_trapz(self):
        cross = self.checks["b5_source_crosscheck"]
        self.assertIsNotNone(cross)
        self.assertEqual(cross["master_capacity_equals_raw_trapz_cycles"], 250)
        self.assertEqual(cross["master_capacity_differs_from_usable_conditional_cycles"], [40])
        self.assertEqual(cross["source_excluded_cycles"], [40])
        self.assertEqual(cross["source_excluded_status"], ["DISCHARGE_INTERRUPTED"])

    def test_b1_label_cycle_mismatch(self):
        cross = self.checks["b1_source_crosscheck"]
        self.assertIsNotNone(cross)
        self.assertEqual(cross["telemetry_global_cycle_est"], [2])
        self.assertAlmostEqual(cross["master_capacity_ah"], 1.202223, places=6)
        self.assertAlmostEqual(cross["audit_capacity_at_telemetry_cycle_ah"], 1.220155, places=6)
        self.assertAlmostEqual(
            cross["audit_capacity_last_cycle_of_checkpoint_ah"], 1.202223, places=6
        )
        self.assertFalse(
            np.isclose(
                cross["master_capacity_ah"],
                cross["audit_capacity_at_telemetry_cycle_ah"],
            )
        )

    # --- temperature / phase / timing ---------------------------------
    def test_temperature_coverage(self):
        nasa = self.checks["temperature"]["NASA_Ames_PCoE"]
        self.assertEqual(nasa["temp_null_rows"], 0)
        self.assertAlmostEqual(nasa["temp_min_c"], 22.35, places=2)
        self.assertAlmostEqual(nasa["temp_max_c"], 42.01, places=2)
        for dataset_id in ("LFP_MultiStat_B1", "LFP_MultiStat_B5", "Zenodo_IonTech_NaIon"):
            cov = self.checks["temperature"][dataset_id]
            self.assertEqual(cov["temp_nonnull_rows"], 0, dataset_id)

    def test_current_phase_conventions(self):
        phase = self.checks["current_phase"]
        self.assertEqual(phase["LFP_MultiStat_B5"]["fraction_negative_discharge"], 1.0)
        self.assertGreater(phase["LFP_MultiStat_B1"]["fraction_positive_charge"], 0.4)
        self.assertGreater(phase["Zenodo_IonTech_NaIon"]["fraction_positive_charge"], 0.4)
        self.assertGreater(phase["NASA_Ames_PCoE"]["fraction_negative_discharge"], 0.95)

    def test_time_sec_semantics_differ_by_source(self):
        semantics = self.checks["time_semantics"]
        self.assertTrue(semantics["NASA_Ames_PCoE"]["resets_each_cycle"])
        for dataset_id in ("LFP_MultiStat_B1", "LFP_MultiStat_B5", "Zenodo_IonTech_NaIon"):
            self.assertFalse(semantics[dataset_id]["resets_each_cycle"], dataset_id)

    def test_timestamps_monotonic_without_duplicates(self):
        for dataset_id, detail in self.checks["duplicates_timing"]["per_dataset"].items():
            self.assertEqual(detail["negative_dt"], 0, dataset_id)
            self.assertEqual(detail["zero_dt"], 0, dataset_id)
            self.assertEqual(detail["duplicate_timestamps_within_cycles"], 0, dataset_id)
        self.assertEqual(self.checks["duplicates_timing"]["duplicate_composite_keys"], 0)

    def test_cycle_labels_are_broadcast_over_rows(self):
        self.assertGreater(self.checks["leakage"]["broadcast_factor_median"], 1)
        self.assertEqual(self.checks["leakage"]["broadcast_factor_max"], 4789)

    # --- candidate labels ---------------------------------------------
    def test_candidate_cycle_labels(self):
        labels = audit.candidate_cycle_labels(self.df, self.checks)
        self.assertEqual(len(labels), 819)
        b5_40 = labels[
            (labels["dataset_id"] == "LFP_MultiStat_B5") & (labels["cycle_id"] == 40)
        ].iloc[0]
        self.assertEqual(b5_40["capacity_label_status"], "INVALID_PARTIAL_DISCHARGE")
        self.assertFalse(bool(b5_40["usable_as_ml_target"]))
        b1 = labels[labels["dataset_id"] == "LFP_MultiStat_B1"].iloc[0]
        self.assertTrue(bool(b1["label_cycle_mismatch"]))
        self.assertFalse(bool(b1["usable_as_ml_target"]))
        nasa = labels[labels["dataset_id"] == "NASA_Ames_PCoE"]
        self.assertTrue(bool(nasa["usable_as_ml_target"].all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
