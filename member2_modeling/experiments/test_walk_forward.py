"""Focused tests for chronological walk-forward validation and leakage prevention."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

import walk_forward_validation as wf  # noqa: E402


def synthetic_pairs(n: int, cell_id: str = "C1", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pair_index = np.arange(n)
    current = 100.0 - 0.05 * pair_index
    target = 100.0 - 0.05 * (pair_index + 1)
    return pd.DataFrame(
        {
            "cell_id": cell_id,
            "pair_index": pair_index,
            "target_record_position": pair_index + 1,
            "cycle_id": pair_index + 1,
            "target_cycle_id": pair_index + 2,
            "cycle_gap": 1,
            "is_gap_pair": False,
            "soh_pct": current,
            "prev_soh_pct": np.r_[np.nan, current[:-1]],
            "current_soh_pct": current,
            "target_soh_pct": target,
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )


class FoldBoundaryTests(unittest.TestCase):
    def test_single_cell_folds_are_chronological_and_expanding(self):
        pairs = synthetic_pairs(248)
        folds, cuts = wf.make_single_cell_folds(pairs, initial_frac=0.40, n_folds=5)
        self.assertEqual(cuts, [99, 129, 159, 188, 218, 248])
        previous_train_n = 0
        covered = []
        for fold in folds:
            train = fold["train"]
            test = fold["test"]
            self.assertLess(train["pair_index"].max(), test["pair_index"].min())
            self.assertEqual(train["pair_index"].min(), 0)
            self.assertGreater(len(train), previous_train_n)
            previous_train_n = len(train)
            covered.extend(test["pair_index"].tolist())
        self.assertEqual(covered, list(range(99, 248)))

    def test_pooled_cell_folds_preserve_each_cell_boundary(self):
        a = synthetic_pairs(30, "A", seed=1)
        b = synthetic_pairs(20, "B", seed=2)
        pairs = pd.concat([a, b], ignore_index=True)
        folds, cuts = wf.make_pooled_cell_folds(pairs, initial_frac=0.5, n_folds=3)
        self.assertEqual(cuts["A"], [15, 20, 25, 30])
        self.assertEqual(cuts["B"], [10, 13, 17, 20])
        for fold in folds:
            for cell_id in ("A", "B"):
                train = fold["train"][fold["train"]["cell_id"] == cell_id]
                test = fold["test"][fold["test"]["cell_id"] == cell_id]
                self.assertLess(
                    train["pair_index"].max(), test["pair_index"].min()
                )
                self.assertEqual(
                    set(fold["train"]["cell_id"]), set(fold["test"]["cell_id"])
                )


class PairTimingTests(unittest.TestCase):
    def test_target_is_next_observed_record_even_across_gap(self):
        cycles = pd.DataFrame(
            {
                "cell_id": ["C1"] * 4,
                "cycle_id": [1, 2, 4, 5],
                "cycle_position": [0, 1, 2, 3],
                "soh_pct": [100.0, 99.0, 97.0, 96.0],
            }
        )
        pairs = wf.make_pairs(cycles)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs["target_cycle_id"].tolist(), [2, 4, 5])
        self.assertEqual(pairs["target_soh_pct"].tolist(), [99.0, 97.0, 96.0])
        self.assertEqual(pairs["cycle_gap"].tolist(), [1, 2, 1])
        self.assertEqual(pairs["is_gap_pair"].tolist(), [False, True, False])
        self.assertNotIn(3, pairs["target_cycle_id"].tolist())

    def test_gap_pair_is_not_treated_as_adjacent_cycle(self):
        cycles = pd.DataFrame(
            {
                "cell_id": ["C1"] * 3,
                "cycle_id": [39, 41, 42],
                "cycle_position": [0, 1, 2],
                "soh_pct": [94.0, 93.9, 93.8],
            }
        )
        pairs = wf.make_pairs(cycles)
        first = pairs.iloc[0]
        self.assertEqual(first["cycle_id"], 39)
        self.assertEqual(first["target_cycle_id"], 41)
        self.assertEqual(first["cycle_gap"], 2)


class LeakageTests(unittest.TestCase):
    def test_unit_safe_features_exclude_current_derived_features(self):
        schema = json.loads(
            (ROOT / "member2_modeling" / "feature_schema.json").read_text(
                encoding="utf-8"
            )
        )
        lfp = list(schema["models"]["lfp_lgbm"]["feature_names"])
        safe = wf.ef.unit_safe_lfp_features(lfp)
        self.assertEqual(len(safe), 22)
        self.assertFalse(set(safe) & set(wf.ef.CURRENT_DERIVED_FEATURES))
        self.assertIn("current_soh_pct", safe)
        self.assertTrue(all(not c.startswith("target_") for c in safe))

    def test_ridge_predictions_do_not_depend_on_test_targets(self):
        train = synthetic_pairs(40)
        test = synthetic_pairs(15, seed=7)
        inner_train, inner_val = wf.inner_split(train)
        features = ["f1", "f2"]
        pred_a, info_a = wf.fit_ridge(
            train, inner_train, inner_val, test, features
        )
        altered = test.copy()
        altered["target_soh_pct"] = altered["target_soh_pct"] + 50.0
        pred_b, info_b = wf.fit_ridge(
            train, inner_train, inner_val, altered, features
        )
        np.testing.assert_allclose(pred_a, pred_b)
        self.assertEqual(info_a["ridge_alpha"], info_b["ridge_alpha"])

    def test_ridge_fit_is_deterministic(self):
        train = synthetic_pairs(40)
        test = synthetic_pairs(15, seed=8)
        inner_train, inner_val = wf.inner_split(train)
        pred_a, info_a = wf.fit_ridge(
            train, inner_train, inner_val, test, ["f1", "f2"]
        )
        pred_b, info_b = wf.fit_ridge(
            train, inner_train, inner_val, test, ["f1", "f2"]
        )
        np.testing.assert_array_equal(pred_a, pred_b)
        self.assertEqual(info_a, info_b)

    def test_lightgbm_fit_is_deterministic(self):
        train = synthetic_pairs(60, seed=11)
        test = synthetic_pairs(20, seed=12)
        inner_train, inner_val = wf.inner_split(train)
        features = ["f1", "f2"]
        pred_a, info_a = wf.fit_lgbm(
            train, inner_train, inner_val, test, features, target_mode="absolute"
        )
        pred_b, info_b = wf.fit_lgbm(
            train, inner_train, inner_val, test, features, target_mode="absolute"
        )
        np.testing.assert_allclose(pred_a, pred_b)
        self.assertEqual(info_a["best_iteration"], info_b["best_iteration"])

    def test_lightgbm_predictions_do_not_depend_on_test_targets(self):
        train = synthetic_pairs(60, seed=13)
        test = synthetic_pairs(20, seed=14)
        inner_train, inner_val = wf.inner_split(train)
        features = ["f1", "f2"]
        pred_a, info_a = wf.fit_lgbm(
            train, inner_train, inner_val, test, features, target_mode="absolute"
        )
        altered = test.copy()
        altered["target_soh_pct"] = altered["target_soh_pct"] + 50.0
        pred_b, info_b = wf.fit_lgbm(
            train, inner_train, inner_val, altered, features, target_mode="absolute"
        )
        np.testing.assert_allclose(pred_a, pred_b)
        self.assertEqual(info_a["best_iteration"], info_b["best_iteration"])


if __name__ == "__main__":
    unittest.main()
