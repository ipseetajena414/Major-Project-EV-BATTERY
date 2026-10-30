"""Focused tests for the isolated NMC RUL validation.

Covers target construction (first vs sustained crossing, transient rebounds,
cells starting below threshold, gapped cycle ids), leave-one-cell-out
separation, prediction-time feature availability, absence of target leakage,
train-only preprocessing and batch independence, causal prefix invariance,
determinism, and the real-data production-split audit.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
for path in (str(EXPERIMENT_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import rul_validation as rv  # noqa: E402

_CYCLES: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def real_cycles() -> pd.DataFrame:
    global _CYCLES
    if _CYCLES is None:
        _CYCLES = rv.load_nmc_cycles()
    return _CYCLES


def real_features() -> list[str]:
    global _FEATURES
    if _FEATURES is None:
        _FEATURES = rv.load_feature_names()
    return _FEATURES


def synthetic_cycles(soh, cell="X1", cycle_ids=None) -> pd.DataFrame:
    soh = np.asarray(soh, dtype=float)
    n = len(soh)
    ids = np.arange(1, n + 1) if cycle_ids is None else np.asarray(cycle_ids, dtype=int)
    return pd.DataFrame(
        {
            "cell_id": [cell] * n,
            "cycle_id": ids,
            "cycle_position": np.arange(n),
            "soh_pct": soh,
            "current_soh_pct": soh,
        }
    )


class TargetConstructionTests(unittest.TestCase):
    def test_first_crossing_is_strictly_below_80(self):
        cyc = synthetic_cycles([100.0, 90.0, 80.0, 79.9, 78.0])
        out = rv.add_rul_labels(cyc, "first")
        self.assertEqual(out["crossing_position"].iloc[0], 3.0)
        np.testing.assert_allclose(out["rul_cycles"], [3.0, 2.0, 1.0, 0.0, 0.0])

    def test_rul_counts_observed_positions_not_cycle_id_arithmetic(self):
        # Gapped ids: the label is in observed cycles, never id subtraction.
        cyc = synthetic_cycles([100.0, 90.0, 79.0, 78.0, 77.0],
                               cycle_ids=[1, 2, 5, 6, 10])
        out = rv.add_rul_labels(cyc, "first")
        np.testing.assert_allclose(out["rul_cycles"], [2.0, 1.0, 0.0, 0.0, 0.0])

    def test_rebound_after_first_crossing_stays_zero(self):
        cyc = synthetic_cycles([100.0, 79.0, 85.0, 84.0, 83.0, 79.0])
        out = rv.add_rul_labels(cyc, "first")
        np.testing.assert_allclose(out["rul_cycles"], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        table, summary = rv.audit_targets(cyc)
        self.assertEqual(int(table.loc[0, "cycles_above_eol_after_first_crossing"]), 3)
        self.assertEqual(summary["transient_first_crossings"][0]["cell_id"], "X1")

    def test_sustained3_ignores_transient_dips(self):
        cyc = synthetic_cycles([100.0, 79.0, 85.0, 79.5, 85.0, 79.0, 78.0, 77.0])
        first = rv.add_rul_labels(cyc, "first")
        sustained = rv.add_rul_labels(cyc, "sustained3")
        self.assertEqual(first["crossing_position"].iloc[0], 1.0)
        self.assertEqual(sustained["crossing_position"].iloc[0], 5.0)
        self.assertEqual((first["rul_cycles"] > 0).sum(), 1)
        self.assertEqual((sustained["rul_cycles"] > 0).sum(), 5)

    def test_cell_starting_below_threshold_has_no_pre_rows(self):
        cyc = synthetic_cycles([79.0, 78.0, 77.0])
        out = rv.add_rul_labels(cyc, "first")
        np.testing.assert_allclose(out["rul_cycles"], [0.0, 0.0, 0.0])
        table, summary = rv.audit_targets(cyc)
        self.assertEqual(table.loc[0, "first_crossing_position"], 0)
        self.assertEqual(table.loc[0, "n_pre_threshold"], 0)
        self.assertEqual(summary["cells_starting_below_threshold"], ["X1"])

    def test_cell_never_crossing_has_no_labels(self):
        cyc = synthetic_cycles([100.0, 97.0, 92.0])
        out = rv.add_rul_labels(cyc, "first")
        self.assertTrue(out["rul_cycles"].isna().all())
        table, summary = rv.audit_targets(cyc)
        self.assertEqual(summary["cells_never_crossing"], ["X1"])

    def test_labels_are_per_cell(self):
        a = synthetic_cycles([100.0, 90.0, 79.0], cell="A")
        b = synthetic_cycles([100.0, 95.0, 91.0, 86.0, 79.0], cell="B")
        out = rv.add_rul_labels(pd.concat([a, b], ignore_index=True), "first")
        self.assertEqual(out.loc[out["cell_id"] == "A", "crossing_position"].iloc[0], 2.0)
        self.assertEqual(out.loc[out["cell_id"] == "B", "crossing_position"].iloc[0], 4.0)

    def test_labels_do_not_mutate_input(self):
        cyc = synthetic_cycles([100.0, 90.0, 79.0])
        before = cyc.copy()
        rv.add_rul_labels(cyc, "first")
        pd.testing.assert_frame_equal(cyc, before)


class FoldSeparationTests(unittest.TestCase):
    def test_loco_folds_are_disjoint_and_cover_all_rows(self):
        cycles = pd.concat(
            [synthetic_cycles([100.0, 95.0, 90.0], cell="A"),
             synthetic_cycles([100.0, 94.0, 88.0], cell="B"),
             synthetic_cycles([100.0, 93.0, 86.0], cell="C")],
            ignore_index=True,
        )
        folds = rv.loco_folds(cycles)
        self.assertEqual([f["held_out_cell"] for f in folds], ["A", "B", "C"])
        seen = 0
        for fold in folds:
            train_cells = set(fold["train"]["cell_id"])
            test_cells = set(fold["test"]["cell_id"])
            self.assertEqual(test_cells, {fold["held_out_cell"]})
            self.assertFalse(train_cells & test_cells)
            self.assertEqual(len(fold["test"]), 3)
            seen += len(fold["test"])
        self.assertEqual(seen, len(cycles))

    def test_loco_needs_two_cells(self):
        with self.assertRaises(ValueError):
            rv.loco_folds(synthetic_cycles([100.0, 90.0, 80.0]))


class LeakageTests(unittest.TestCase):
    def setUp(self):
        cycles = rv.add_rul_labels(real_cycles(), "first")
        self.fold = rv.loco_folds(cycles)[0]  # B0005 held out
        self.features = real_features()

    def test_features_exclude_identifiers_targets_and_age(self):
        forbidden = {
            "dataset_id", "cell_id", "protocol_id", "cycle_id", "cycle_position",
            "capacity_ah", "soh_pct", "chemistry", "split", "target_soh_pct",
            "target_cycle_id", "rul_cycles", "crossing_position",
        }
        self.assertFalse(forbidden & set(self.features))
        self.assertFalse(any("future" in f or "target" in f for f in self.features))
        # Current-cycle measured SOH is available at prediction time and is
        # the only state-of-health input the model receives.
        self.assertIn("current_soh_pct", self.features)
        available = set(real_cycles().columns)
        self.assertEqual(set(self.features) - available, set())

    def test_schema_matches_loaded_features(self):
        import json

        schema = json.loads(rv.SCHEMA.read_text(encoding="utf-8"))
        self.assertEqual(
            list(schema["models"]["nmc_rul_lgbm"]["feature_names"]), self.features
        )

    def test_predictions_ignore_test_targets(self):
        train, test = self.fold["train"], self.fold["test"]
        base = rv.baseline_predictions(train, test)
        shuffled = test.copy()
        shuffled["rul_cycles"] = np.random.default_rng(0).permutation(
            shuffled["rul_cycles"].to_numpy()
        )
        again = rv.baseline_predictions(train, shuffled)
        for model, pred in base.items():
            np.testing.assert_allclose(pred, again[model])

        train_pred, _ = rv.fit_predict_lgbm(train, test, self.features, "lgbm_pre_eol_fixed200")
        shuffled_pred, _ = rv.fit_predict_lgbm(
            train, shuffled, self.features, "lgbm_pre_eol_fixed200"
        )
        np.testing.assert_allclose(train_pred, shuffled_pred)

    def test_predictions_do_not_require_test_labels(self):
        train, test = self.fold["train"], self.fold["test"]
        stripped = test.drop(columns=["rul_cycles"])
        pred, _ = rv.fit_predict_lgbm(train, stripped, self.features, "lgbm_pre_eol_fixed200")
        self.assertEqual(len(pred), len(test))

    def test_causal_baselines_use_only_observed_history(self):
        train, test = self.fold["train"], self.fold["test"]
        slope = rv._median_training_slope(train[train["rul_cycles"] > 0])
        full = rv.causal_extrapolation(test, "causal_slope", slope)
        for cut in (12, 40):
            prefix = rv.causal_extrapolation(test.iloc[:cut], "causal_slope", slope)
            np.testing.assert_allclose(prefix, full[:cut])

    def test_global_slope_comes_from_training_cells_only(self):
        train, test = self.fold["train"], self.fold["test"]
        pred = rv.baseline_predictions(train, test)["soh_extrapolation_global"]
        # pred = (soh - 80) / -slope, so the implied slope must be identical
        # for every row of the held-out cell and unaffected by test history.
        implied = -(test["soh_pct"].to_numpy() - rv.EOL_PCT) / np.where(pred > 0, pred, np.nan)
        self.assertLess(float(np.nanstd(implied)), 1e-9)
        self.assertAlmostEqual(float(np.nanmedian(implied)),
                               rv._median_training_slope(train[train["rul_cycles"] > 0]))

    def test_production_protocol_is_batch_independent(self):
        train, test = self.fold["train"], self.fold["test"]
        full_pred, info = rv.fit_predict_lgbm(
            train, test, self.features, "lgbm_production_protocol"
        )
        subset = test.iloc[:15]
        subset_pred, _ = rv.fit_predict_lgbm(
            train, subset, self.features, "lgbm_production_protocol"
        )
        np.testing.assert_allclose(subset_pred, full_pred[:15])
        self.assertLess(info["train_rows"], len(train))
        self.assertGreater(info["val_rows"], 0)

    def test_production_protocol_trains_without_held_out_cell(self):
        train, test = self.fold["train"], self.fold["test"]
        _, info = rv.fit_predict_lgbm(train, test, self.features, "lgbm_production_protocol")
        expected_fit = 0
        for _, group in train.groupby("cell_id"):
            n = len(group)
            n_train = int(np.floor(n * rv.PRODUCTION_TRAIN_FRAC))
            n_val = int(np.floor(n * rv.PRODUCTION_VAL_FRAC))
            expected_fit += n_train + n_val
        self.assertEqual(info["train_rows"] + info["val_rows"], expected_fit)
        self.assertLessEqual(info["train_rows"] + info["val_rows"], len(train))
        self.assertGreater(info["train_post_rows"], 0)


class DeterminismTests(unittest.TestCase):
    def test_lightgbm_fits_are_deterministic(self):
        cycles = rv.add_rul_labels(real_cycles(), "first")
        fold = rv.loco_folds(cycles)[1]
        for variant in rv.LGBM_MODELS:
            p1, _ = rv.fit_predict_lgbm(
                fold["train"], fold["test"], real_features(), variant
            )
            p2, _ = rv.fit_predict_lgbm(
                fold["train"], fold["test"], real_features(), variant
            )
            np.testing.assert_array_equal(p1, p2)

    def test_metrics_are_invariant_to_row_order(self):
        y = np.array([5.0, 10.0, 20.0])
        p = np.array([4.0, 12.0, 18.0])
        a = rv.regression_metrics(y, p)
        b = rv.regression_metrics(y[::-1], p[::-1])
        self.assertAlmostEqual(a["mae"], b["mae"])
        self.assertAlmostEqual(a["rmse"], b["rmse"])

    def test_negative_predictions_are_clipped_for_scoring(self):
        cycles = rv.add_rul_labels(real_cycles(), "first")
        fold = rv.loco_folds(cycles)[0]
        preds = rv.baseline_predictions(fold["train"], fold["test"])
        preds["lgbm_pre_eol_fixed200"] = np.full(len(fold["test"]), -3.0)
        by_cell, _, frames = rv.evaluate(fold["test"], preds)
        row = next(r for r in by_cell if r["model"] == "lgbm_pre_eol_fixed200")
        frame = frames["lgbm_pre_eol_fixed200"]
        pre = frame[frame["rul_cycles"] > 0]
        self.assertEqual(row["negative_raw_predictions"], int((pre["prediction_raw"] < 0).sum()))
        self.assertEqual(row["negative_raw_predictions"], len(pre))
        self.assertTrue((frame["prediction"] >= 0).all())


@unittest.skipUnless(rv.PARQUET.exists(), "master_table.parquet not available")
class RealDataAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cycles = rv.add_rul_labels(real_cycles(), "first")
        cls.table, cls.summary = rv.audit_targets(real_cycles())
        cls.production = rv.production_split_audit(real_cycles())

    def test_three_cells_with_contiguous_ids(self):
        self.assertEqual(list(self.table["cell_id"]), ["B0005", "B0006", "B0018"])
        self.assertTrue(self.table["cycle_ids_contiguous"].all())
        self.assertEqual(int(self.table["n_cycles"].sum()), len(self.cycles))

    def test_first_crossings_match_recorded_cycles(self):
        crossings = dict(zip(self.table["cell_id"], self.table["first_crossing_cycle_id"]))
        self.assertEqual(crossings, {"B0005": 75, "B0006": 63, "B0018": 45})

    def test_b0018_has_transient_rebound_after_threshold(self):
        row = self.table.set_index("cell_id").loc["B0018"]
        self.assertEqual(int(row["cycles_above_eol_after_first_crossing"]), 13)
        self.assertGreater(int(row["sustained3_crossing_position"]),
                           int(row["first_crossing_position"]))

    def test_no_cell_starts_below_threshold(self):
        self.assertEqual(self.summary["cells_starting_below_threshold"], [])
        self.assertEqual(self.summary["cells_never_crossing"], [])

    def test_production_test_split_is_entirely_post_threshold(self):
        self.assertEqual(self.production["test_n"], 97)
        self.assertEqual(self.production["test_pre_eol_n"], 0)
        self.assertEqual(self.production["test_post_eol_n"], 97)
        self.assertEqual(self.production["test_rul_max"], 0.0)

    def test_shipped_diagnostic_is_marked_in_sample(self):
        table, summary = rv.shipped_model_diagnostic(self.cycles, real_features())
        self.assertTrue(summary["in_sample"])
        self.assertIn("NOT validation evidence", summary["note"])
        self.assertTrue(table["in_sample"].all())
        self.assertLess(summary["pre_mae_overall"], 0.5)

    def test_write_false_writes_nothing(self):
        before = rv.protected_hashes()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "would_be_results"
            result = rv.run_experiment(write=False, out_dir=out)
            self.assertFalse(out.exists())
            self.assertEqual(result["summary"]["production_split"]["test_n"], 97)
            self.assertEqual(len(result["results"]["first"]["pooled"]), len(rv.ALL_MODELS))
        self.assertEqual(rv.protected_hashes(), before)
        self.assertTrue(result["summary"]["protected_artifacts_unchanged"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
