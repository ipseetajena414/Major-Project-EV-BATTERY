"""Run from the project root:  python -m unittest discover -s tests -v"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bms, config, placeholders
from run_simulation import run_simulation


class TestBMS(unittest.TestCase):
    def test_unrestricted_approves_everything(self):
        d = bms.evaluate(3.0, 100, 5, 40, config.SCENARIOS["unrestricted"])
        self.assertEqual((d["decision"], d["c_rate"], d["target_soc"]), ("approve", 3.0, 100))

    def test_full_protection_caps_all_limits(self):
        d = bms.evaluate(3.0, 100, 5, 30, config.SCENARIOS["full_protection"])
        self.assertEqual(d["decision"], "modify")
        self.assertLessEqual(d["c_rate"], config.BMS_LIMITS["max_c_rate"])
        self.assertEqual((d["target_soc"], d["min_soc"]), (80.0, 20.0))

    def test_extreme_heat_refuses_fast_charge(self):
        d = bms.evaluate(2.5, 80, 20, 44, config.SCENARIOS["bms_supervised"])
        self.assertEqual((d["decision"], d["c_rate"]), ("reject", config.BMS_LIMITS["fallback_c_rate"]))


class TestModels(unittest.TestCase):
    def test_range_falls_with_soh(self):
        self.assertGreater(placeholders.estimate_range_km(95, 25, "NMC"), placeholders.estimate_range_km(75, 25, "NMC"))

    def test_range_uses_member3_engine(self):
        self.assertEqual(config.RANGE_BACKEND, "member3")
        self.assertIn("RangeCalculator", placeholders.provenance()["range"]["source"])

    def test_gentle_use_degrades_less_than_harsh(self):
        gentle = dict(mean_temp_c=22, max_c_rate=0.5, target_soc=80, min_soc=20, hours_above_80soc_per_cycle=0, n_cycles=30)
        harsh = dict(mean_temp_c=45, max_c_rate=3.0, target_soc=100, min_soc=5, hours_above_80soc_per_cycle=6, n_cycles=30)
        for chem in config.CHEMISTRIES:
            self.assertGreater(placeholders.predict_soh(chem, 100, gentle), placeholders.predict_soh(chem, 100, harsh))


class TestSimulation(unittest.TestCase):
    def test_protection_extends_life_every_chemistry(self):
        for chem in config.CHEMISTRIES:
            a = run_simulation(chem, "unrestricted", "hot")["years_to_eol"]
            b = run_simulation(chem, "full_protection", "hot")["years_to_eol"]
            self.assertGreater(b, a, chem)

    def test_history_and_end_of_life(self):
        r = run_simulation("LFP", "unrestricted", "temperate")
        self.assertTrue(r["eol_reached"])
        self.assertLessEqual(r["final_soh"], config.END_OF_LIFE_SOH)
        self.assertTrue(all(h["range_km"] > 0 for h in r["history"]))


if __name__ == "__main__":
    unittest.main()
