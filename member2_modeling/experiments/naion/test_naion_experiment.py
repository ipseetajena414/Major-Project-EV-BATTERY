"""Focused tests for the Na-ion experiments.

Covers label broadcasting, next-observed-record pairing, phase ordering,
boundary timestamp duplicates, LOCO separation, train-only preprocessing,
causal baselines and feature exclusions.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR))

import naion_evaluation as ev  # noqa: E402
import naion_features as nf  # noqa: E402


def synthetic_raw(
    cell: str = "C1",
    n_cycles: int = 25,
    n_charge: int = 8,
    n_discharge: int = 6,
    soh0: float = 100.0,
    fade: float = 0.05,
    seed: int = 0,
    boundary_overlap: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    t = 0.0
    for k in range(n_cycles):
        cycle_soh = soh0 - fade * k
        jitter = 0.002 * k + 0.0005 * rng.normal()
        if k > 0 and boundary_overlap:
            t -= 4.0
        for j in range(n_charge):
            rows.append(
                {
                    "chemistry": "Na-ion",
                    "dataset_id": "synthetic",
                    "cell_id": cell,
                    "protocol_id": "P",
                    "cycle_id": k + 1,
                    "time_sec": t,
                    "voltage_v": 3.0 + 0.1 * j + jitter,
                    "current_a": 2.0,
                    "temp_c": np.nan,
                    "capacity_ah": cycle_soh / 100.0,
                    "soh_pct": cycle_soh,
                    "quality_flag": "TEMP_UNMEASURED",
                    "source_file": "synthetic",
                }
            )
            t += 4.0
        for j in range(n_discharge):
            rows.append(
                {
                    "chemistry": "Na-ion",
                    "dataset_id": "synthetic",
                    "cell_id": cell,
                    "protocol_id": "P",
                    "cycle_id": k + 1,
                    "time_sec": t,
                    "voltage_v": 3.6 - 0.1 * j + jitter,
                    "current_a": -2.0,
                    "temp_c": np.nan,
                    "capacity_ah": cycle_soh / 100.0,
                    "soh_pct": cycle_soh,
                    "quality_flag": "TEMP_UNMEASURED",
                    "source_file": "synthetic",
                }
            )
            t += 4.0
    raw = pd.DataFrame(rows)
    raw["_row_order"] = np.arange(len(raw))
    return raw


def synthetic_pairs(cell: str, n_cycles: int = 25, seed: int = 0) -> pd.DataFrame:
    raw = synthetic_raw(cell=cell, n_cycles=n_cycles, seed=seed)
    raw = raw.sort_values(["cell_id", "cycle_id", "time_sec", "_row_order"], kind="mergesort")
    cycles = nf.extract_cycle_features(raw.reset_index(drop=True))
    return nf.build_pairs(cycles)


def _discharge_first_raw() -> pd.DataFrame:
    raw = synthetic_raw(n_cycles=2)
    frames = []
    for _, group in raw.groupby("cycle_id"):
        group = group.sort_values("time_sec")
        chg = group[group["current_a"] > 0].copy()
        dis = group[group["current_a"] < 0].copy()
        start = float(group["time_sec"].min())
        dis["time_sec"] = start + 4.0 * np.arange(len(dis))
        chg["time_sec"] = start + 4.0 * len(dis) + 4.0 * np.arange(len(chg))
        frames.append(pd.concat([dis, chg]))
    out = pd.concat(frames, ignore_index=True)
    out["_row_order"] = np.arange(len(out))
    return out


class LabelTests(unittest.TestCase):
    def test_labels_are_cycle_level_not_row_level(self):
        raw = synthetic_raw(n_cycles=3, n_charge=8, n_discharge=6)
        cycles = nf.extract_cycle_features(raw)
        self.assertEqual(len(cycles), 3)
        self.assertEqual(cycles["n_rows"].tolist(), [14, 14, 14])
        self.assertEqual(cycles["soh_pct"].tolist(), [100.0, 99.95, 99.9])
        audit = nf.label_broadcast_audit(raw)
        self.assertEqual(audit["cycles_with_nonconstant_soh"], 0)
        self.assertEqual(audit["cycles_with_nonconstant_capacity"], 0)
        self.assertAlmostEqual(audit["max_abs_capacity_minus_soh_over_100"], 0.0)

    def test_cycle_table_has_one_row_per_cell_cycle(self):
        raw = synthetic_raw(n_cycles=4)
        cycles = nf.extract_cycle_features(raw)
        self.assertFalse(cycles.duplicated(subset=["cell_id", "cycle_id"]).any())
        self.assertEqual(cycles["cycle_position"].tolist(), [0, 1, 2, 3])


class PairTimingTests(unittest.TestCase):
    def test_target_is_next_observed_record_and_gap_is_recorded(self):
        cycles = pd.DataFrame(
            {
                "cell_id": ["C1"] * 3,
                "cycle_id": [1, 2, 4],
                "cycle_position": [0, 1, 2],
                "soh_pct": [100.0, 99.5, 98.5],
                "current_soh_pct": [100.0, 99.5, 98.5],
            }
        )
        pairs = nf.build_pairs(cycles)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs["target_cycle_id"].tolist(), [2, 4])
        self.assertEqual(pairs["target_soh_pct"].tolist(), [99.5, 98.5])
        self.assertEqual(pairs["cycle_gap"].tolist(), [1, 2])
        self.assertEqual(pairs["is_gap_pair"].tolist(), [False, True])
        self.assertNotIn(3, pairs["target_cycle_id"].tolist())

    def test_pairs_never_cross_cells(self):
        pairs = pd.concat(
            [synthetic_pairs("A", 10, seed=1), synthetic_pairs("B", 8, seed=2)],
            ignore_index=True,
        )
        for cell, group in pairs.groupby("cell_id"):
            self.assertEqual(group["target_cycle_id"].max(), group["cycle_id"].max() + 1)
            self.assertTrue(group["pair_index"].is_monotonic_increasing)


class PhaseTests(unittest.TestCase):
    def test_features_use_the_correct_phase(self):
        raw = synthetic_raw(n_cycles=1, n_charge=8, n_discharge=6)
        cycles = nf.extract_cycle_features(raw)
        row = cycles.iloc[0]
        chg = raw[raw["current_a"] > 0].sort_values("time_sec")
        dis = raw[raw["current_a"] < 0].sort_values("time_sec")
        self.assertAlmostEqual(row["chg_v_mean"], chg["voltage_v"].mean(), places=9)
        self.assertAlmostEqual(row["dis_v_mean"], dis["voltage_v"].mean(), places=9)
        self.assertAlmostEqual(
            row["transition_voltage"], chg["voltage_v"].iloc[-1], places=9
        )
        self.assertAlmostEqual(row["chg_duration_frac"], 28.0 / 52.0, places=9)
        self.assertAlmostEqual(row["chg_v_slope_norm"], 0.7, places=6)
        self.assertAlmostEqual(row["dis_v_slope_norm"], -0.5, places=6)

    def test_phase_structure_rejects_discharge_first(self):
        raw = _discharge_first_raw()
        raw = raw.sort_values(
            ["cell_id", "cycle_id", "time_sec", "_row_order"], kind="mergesort"
        )
        with self.assertRaises(ValueError):
            nf.extract_cycle_features(raw.reset_index(drop=True))

    def test_phase_structure_rejects_missing_phase(self):
        raw = synthetic_raw(n_cycles=1)
        raw = raw[raw["current_a"] > 0].copy()
        raw["_row_order"] = np.arange(len(raw))
        with self.assertRaises(ValueError):
            nf.extract_cycle_features(raw)


class DuplicateTests(unittest.TestCase):
    def test_cycle_boundary_timestamp_is_flagged_but_not_within_cycle(self):
        raw = synthetic_raw(n_cycles=3, boundary_overlap=True)
        audit = nf.duplicate_audit(raw)
        self.assertEqual(audit["within_cycle_duplicate_rows"], 0)
        self.assertEqual(audit["cross_cycle_shared_timestamps"], 2)
        self.assertEqual(audit["cross_cycle_shared_timestamp_rows"], 4)

    def test_clean_synthetic_data_has_no_duplicate_timestamps(self):
        audit = nf.duplicate_audit(synthetic_raw(n_cycles=3))
        self.assertEqual(audit["cross_cycle_shared_timestamps"], 0)
        self.assertEqual(audit["within_cycle_duplicate_rows"], 0)


class FoldTests(unittest.TestCase):
    def test_loco_folds_are_cell_disjoint_and_cover_all_pairs(self):
        pairs = pd.concat(
            [synthetic_pairs(c, 10, seed=i) for i, c in enumerate(["A", "B", "C", "D"])],
            ignore_index=True,
        )
        splits = ev.loco_splits(pairs)
        self.assertEqual(len(splits), 4)
        covered = []
        for split in splits:
            self.assertFalse(
                set(split["train"]["cell_id"]) & set(split["test"]["cell_id"])
            )
            self.assertEqual(split["test"]["cell_id"].nunique(), 1)
            covered.extend(split["test"]["pair_index"].tolist())
        self.assertEqual(len(covered), len(pairs))


class RidgeTests(unittest.TestCase):
    def _toy(self):
        train = pd.concat(
            [synthetic_pairs("A", 20, seed=1), synthetic_pairs("B", 20, seed=2)],
            ignore_index=True,
        )
        test = synthetic_pairs("C", 12, seed=3)
        return train, test

    def test_predictions_do_not_depend_on_test_targets(self):
        train, test = self._toy()
        pred_a, meta_a = ev.fit_ridge(train, test, nf.FEATURE_COLUMNS, delta=True)
        altered = test.copy()
        altered["target_soh_pct"] = altered["target_soh_pct"] + 25.0
        pred_b, meta_b = ev.fit_ridge(train, altered, nf.FEATURE_COLUMNS, delta=True)
        np.testing.assert_allclose(pred_a, pred_b)
        self.assertEqual(meta_a["selected_alpha"], meta_b["selected_alpha"])

    def test_preprocessing_is_fit_on_training_fold_only(self):
        train, test = self._toy()
        full_pred, _ = ev.fit_ridge(train, test, nf.FEATURE_COLUMNS, delta=False)
        subset = test.iloc[:5].copy()
        subset_pred, _ = ev.fit_ridge(train, subset, nf.FEATURE_COLUMNS, delta=False)
        np.testing.assert_allclose(full_pred[:5], subset_pred)

    def test_ridge_is_deterministic(self):
        train, test = self._toy()
        pred_a, meta_a = ev.fit_ridge(train, test, nf.FEATURE_COLUMNS, delta=False)
        pred_b, meta_b = ev.fit_ridge(train, test, nf.FEATURE_COLUMNS, delta=False)
        np.testing.assert_array_equal(pred_a, pred_b)
        self.assertEqual(meta_a["selected_alpha"], meta_b["selected_alpha"])

    def test_ridge_rejects_cell_overlap(self):
        train, test = self._toy()
        leaky = pd.concat([train, test], ignore_index=True)
        with self.assertRaises(ValueError):
            ev.fit_ridge(leaky, test, nf.FEATURE_COLUMNS, delta=False)

    def test_causal_cell_drift_ignores_future_test_rows(self):
        train, test = self._toy()
        full = ev.causal_cell_mean_drift(train, test)
        prefix = ev.causal_cell_mean_drift(train, test.iloc[:6].copy())
        np.testing.assert_allclose(full[:6], prefix)

    def test_causal_cell_trend_ignores_future_test_rows(self):
        train, test = self._toy()
        full = ev.causal_cell_trend(train, test)
        prefix = ev.causal_cell_trend(train, test.iloc[:8].copy())
        np.testing.assert_allclose(full[:8], prefix)

    def test_first_causal_drift_prediction_is_persistence_plus_train_drift(self):
        train, test = self._toy()
        drift = float(
            (train["target_soh_pct"] - train["current_soh_pct"]).mean()
        )
        pred = ev.causal_cell_mean_drift(train, test)
        self.assertAlmostEqual(pred[0], test["current_soh_pct"].iloc[0] + drift)


class FeatureSafetyTests(unittest.TestCase):
    def test_feature_list_excludes_labels_thermal_and_current_derived_values(self):
        features = set(nf.FEATURE_COLUMNS)
        self.assertTrue(features.isdisjoint(nf.FORBIDDEN_FEATURES))
        self.assertIn("current_soh_pct", features)
        self.assertNotIn("soh_pct", features)
        self.assertFalse(any(c.startswith("temp") for c in features))
        self.assertFalse(any(c.startswith("target_") for c in features))

    def test_forbidden_feature_guard_raises(self):
        with self.assertRaises(ValueError):
            nf.assert_no_forbidden_features(["current_soh_pct", "capacity_ah"])
        with self.assertRaises(ValueError):
            nf.assert_no_forbidden_features(["current_soh_pct", "soh_pct"])

    def test_provenance_covers_every_feature_and_keeps_units(self):
        table = nf.feature_provenance()
        self.assertEqual(set(table["feature"]), set(nf.REDUNDANT_FEATURE_COLUMNS))
        self.assertTrue(table["units"].notna().all())
        self.assertTrue(
            set(table["trust_level"]).issubset(
                {"derived_from_cycle_label", "derived_from_measured_index", "derived_from_measured"}
            )
        )

    def test_extract_cycle_features_keeps_temperature_unused_and_absent(self):
        raw = synthetic_raw(n_cycles=2)
        cycles = nf.extract_cycle_features(raw)
        self.assertTrue(raw["temp_c"].isna().all())
        self.assertNotIn("temp_c", cycles.columns)


@unittest.skipUnless(nf.PARQUET.exists(), "master_table.parquet not available")
class RealTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = nf.load_naion_raw()
        cls.audit = nf.audit_naion(cls.raw)
        cls.cycles = nf.extract_cycle_features(cls.raw)
        cls.pairs = nf.build_pairs(cls.cycles)

    def test_four_cells_and_one_hundred_cycles(self):
        self.assertEqual(self.audit["n_cells"], 4)
        self.assertEqual(self.audit["n_cycles"], 100)
        self.assertEqual(
            self.audit["cells"],
            [
                "NA-ion_270040-1-1-64",
                "NA-ion_270040-1-2-63",
                "NA-ion_270040-1-3-62",
                "NA-ion_270040-1-4-61",
            ],
        )
        self.assertFalse(self.audit["temperature_available"])

    def test_labels_are_constant_within_every_real_cycle(self):
        self.assertEqual(self.audit["label_broadcast"]["cycles_with_nonconstant_soh"], 0)
        self.assertEqual(
            self.audit["label_broadcast"]["cycles_with_nonconstant_capacity"], 0
        )

    def test_no_duplicate_rows_within_real_cycles(self):
        self.assertEqual(self.audit["duplicates"]["within_cycle_duplicate_rows"], 0)

    def test_every_real_cycle_has_one_charge_then_discharge_phase(self):
        self.assertEqual(self.audit["phase_structure"]["failures"], [])
        self.assertEqual(self.audit["phase_structure"]["sign_change_counts"], [1])

    def test_current_unit_check_matches_reported_capacity(self):
        units = self.audit["units"]
        self.assertLess(units["discharge_ah_minus_capacity_ah"]["max_abs"], 0.05)
        self.assertLess(units["charge_ah_minus_capacity_ah"]["max_abs"], 0.05)

    def test_real_pairs_are_contiguous_one_step_records(self):
        self.assertEqual(len(self.pairs), 96)
        self.assertEqual(len(self.pairs) // 4, 24)
        self.assertFalse(self.pairs["is_gap_pair"].any())
        self.assertTrue((self.pairs["cycle_gap"] == 1).all())


if __name__ == "__main__":
    unittest.main()
