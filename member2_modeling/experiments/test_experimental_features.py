"""Hand-calculated unit tests for the isolated experimental features."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import experimental_features as ef  # noqa: E402


def make_group(times, currents, voltages, temps, chemistry="NMC", cell="B0005"):
    return pd.DataFrame(
        {
            "chemistry": chemistry,
            "dataset_id": "TEST",
            "cell_id": cell,
            "protocol_id": "TEST",
            "cycle_id": 1,
            "time_sec": times,
            "voltage_v": voltages,
            "current_a": currents,
            "temp_c": temps,
        }
    )


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.group = make_group(
            times=[0.0, 1.0, 2.0, 3.0],
            currents=[-1.0, -1.0, -1.0, -1.0],
            voltages=[4.0, 3.9, 3.8, 3.7],
            temps=[25.0, 31.0, 35.0, 29.0],
        )

    def test_trapezoid_ah_hand_calculated(self):
        out = ef.throughput_features(self.group, "NMC")
        # Three intervals x 0.5*(1+1)*1 A*s = 3 A*s = 3/3600 Ah
        self.assertAlmostEqual(out["ah_discharge_trapz"], 3.0 / 3600.0, places=12)
        self.assertEqual(out["discharge_duration_s"], 3.0)
        self.assertTrue(out["throughput_unit_verified"])

    def test_time_weighted_thermal_hand_calculated(self):
        out = ef.thermal_features(self.group, "NMC")
        # Midpoints 28, 33, 32 C; two intervals outside 20-30 C
        self.assertAlmostEqual(out["thermal_hours_outside_20_30"], 2.0 / 3600.0, places=12)
        # (33-30)*1 + (32-30)*1 = 5 C*s
        self.assertAlmostEqual(
            out["thermal_degree_hours_above_30"], 5.0 / 3600.0, places=12
        )
        self.assertEqual(out["thermal_degree_hours_below_20"], 0.0)

    def test_voltage_at_capacity_fractions_hand_calculated(self):
        out = ef.capacity_fraction_voltages(self.group, "NMC")
        self.assertAlmostEqual(out["v_at_q20"], 3.94, places=10)
        self.assertAlmostEqual(out["v_at_q50"], 3.85, places=10)
        self.assertAlmostEqual(out["v_at_q80"], 3.76, places=10)

    def test_unsorted_duplicate_timestamps_do_not_change_integral(self):
        shuffled = self.group.sample(frac=1.0, random_state=0).reset_index(drop=True)
        with_dup = pd.concat([shuffled, shuffled.iloc[[0]]], ignore_index=True)
        out = ef.throughput_features(with_dup, "NMC")
        self.assertAlmostEqual(out["ah_discharge_trapz"], 3.0 / 3600.0, places=12)
        self.assertEqual(out["discharge_n_samples"], 4)


class BlockingTests(unittest.TestCase):
    def test_lfp_current_and_thermal_are_blocked(self):
        group = make_group(
            times=[0.0, 1.0, 2.0],
            currents=[-0.625, -0.625, -0.625],
            voltages=[3.2, 3.1, 3.0],
            temps=[np.nan, np.nan, np.nan],
            chemistry="LFP",
            cell="BATTERY_5",
        )
        thr = ef.throughput_features(group, "LFP")
        self.assertTrue(np.isnan(thr["ah_discharge_trapz"]))
        self.assertFalse(thr["throughput_unit_verified"])
        self.assertEqual(thr["throughput_status"], "BLOCKED_CURRENT_UNIT_UNDERDETERMINED")

        therm = ef.thermal_features(group, "LFP")
        self.assertTrue(np.isnan(therm["thermal_hours_outside_20_30"]))
        self.assertEqual(therm["thermal_status"], "BLOCKED_TEMPERATURE_UNMEASURED")

        vq = ef.capacity_fraction_voltages(group, "LFP")
        self.assertTrue(np.isnan(vq["v_at_q50"]))

    def test_missing_nmc_temperature_is_not_imputed(self):
        group = make_group(
            times=[0.0, 1.0, 2.0],
            currents=[-1.0, -1.0, -1.0],
            voltages=[4.0, 3.9, 3.8],
            temps=[np.nan, np.nan, np.nan],
        )
        out = ef.thermal_features(group, "NMC")
        self.assertTrue(np.isnan(out["thermal_hours_outside_20_30"]))
        self.assertEqual(out["thermal_status"], "BLOCKED_TEMPERATURE_UNMEASURED")


class CumulativeTests(unittest.TestCase):
    def test_cumulative_efc_uses_observed_records_only(self):
        df = pd.DataFrame(
            {
                "chemistry": ["NMC", "NMC", "NMC"],
                "cell_id": ["B0005"] * 3,
                "cycle_id": [1, 2, 5],
                "ah_discharge_trapz": [1.0, 0.5, 2.0],
                "throughput_unit_verified": [True, True, True],
            }
        )
        out = ef.add_cumulative_features(df)
        self.assertEqual(out["throughput_cumulative_ah"].tolist(), [1.0, 1.5, 3.5])
        self.assertEqual(out["efc_cumulative"].tolist(), [0.5, 0.75, 1.75])


class DqDvTests(unittest.TestCase):
    def test_constant_slope_diagnostic(self):
        n = 40
        t = np.arange(n, dtype=float)
        v = 4.0 - 0.01 * t
        group = make_group(
            times=t,
            currents=np.full(n, -1.0),
            voltages=v,
            temps=np.full(n, 25.0),
        )
        out = ef.dqdv_diagnostic(group, "NMC")
        # dV/dQ = (-0.01 V/s) / (1/3600 Ah/s) = -36 V/Ah
        self.assertAlmostEqual(out["dqdv_median_mid"], -36.0, places=3)
        self.assertEqual(out["dqdv_status"], "OK_NMC_DIAGNOSTIC_ONLY")

    def test_too_few_points_blocked(self):
        group = make_group(
            times=[0.0, 1.0, 2.0],
            currents=[-1.0, -1.0, -1.0],
            voltages=[4.0, 3.9, 3.8],
            temps=[25.0, 25.0, 25.0],
        )
        out = ef.dqdv_diagnostic(group, "NMC")
        self.assertTrue(np.isnan(out["dqdv_median_mid"]))
        self.assertEqual(out["dqdv_status"], "INSUFFICIENT_DISCHARGE_SAMPLES")


class FeatureListTests(unittest.TestCase):
    def test_unit_safe_lfp_removes_current_features(self):
        cols = ["v_mean", "ah_throughput", "energy_wh", "avg_power_w", "i_mean", "v_slope"]
        safe = ef.unit_safe_lfp_features(cols)
        self.assertEqual(safe, ["v_mean", "v_slope"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
