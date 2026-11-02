"""Run from the project root:  python -m unittest discover -s tests -v"""
import os
import sys
import unittest

# Ensure package directory and repo root are accessible
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TESTS_DIR)
_REPO_DIR = os.path.dirname(_PKG_DIR)
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

try:
    from ev_member4 import agents, bms, config, placeholders
    from ev_member4.run_simulation import run_simulation, simulate_bms_negotiation
except (ImportError, ValueError):
    import agents, bms, config, placeholders
    from run_simulation import run_simulation, simulate_bms_negotiation

# Guarantee offline deterministic execution for unit testing regardless of .env credentials
config.USE_LLM = False


class TestBMS(unittest.TestCase):
    def test_unrestricted_approves_everything(self):
        d = bms.evaluate(3.0, 100, 5, 40, config.SCENARIOS["unrestricted"])
        self.assertEqual((d["decision"], d["c_rate"], d["target_soc"]), ("approve", 3.0, 100))

    def test_full_protection_caps_all_limits(self):
        d = bms.evaluate(3.0, 100, 5, 30, config.SCENARIOS["full_protection"])
        self.assertEqual(d["decision"], "modify")
        self.assertLessEqual(d["c_rate"], config.BMS_LIMITS["max_c_rate"])
        self.assertEqual((d["target_soc"], d["min_soc"]), (80.0, 20.0))
        self.assertIn("RULE_SOC_MAX_CEILING", d["rules_triggered"])
        self.assertIn("RULE_CRATE_CEILING", d["rules_triggered"])

    def test_extreme_heat_refuses_fast_charge(self):
        d = bms.evaluate(2.5, 80, 20, 44, config.SCENARIOS["bms_supervised"])
        self.assertEqual((d["decision"], d["c_rate"]), ("reject", config.BMS_LIMITS["fallback_c_rate"]))
        self.assertIn("RULE_THERMAL_LOCKOUT", d["rules_triggered"])

    def test_catastrophic_thermal_shutdown(self):
        """Verify charging is completely aborted (0.0C) if cell temperature exceeds critical limit."""
        custom_limits = {"critical_shutdown_temp_c": 45.0, "max_temp_c": 35.0, "fallback_c_rate": 0.3}
        # Ambient 46C -> even with 0.0C charge, ambient exceeds 45C
        d = bms.evaluate(2.0, 80, 20, 46.0, config.SCENARIOS["bms_supervised"], bms_limits=custom_limits)
        self.assertEqual(d["decision"], "reject")
        self.assertEqual(d["c_rate"], 0.0)
        self.assertIn("RULE_CATASTROPHIC_SHUTDOWN", d["rules_triggered"])


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

    def test_range_returns_none_when_soh_missing(self):
        """Member 4 must never fabricate range when SOH is unavailable."""
        self.assertIsNone(placeholders.estimate_range_km(None, 25.0, "NMC"))


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


class TestArchitectureAndIntegration(unittest.TestCase):
    def test_missing_soh_and_rul_remain_unavailable(self):
        """Verify missing predictions are explicitly None and labeled UNAVAILABLE."""
        res = simulate_bms_negotiation(battery_state=None, member2_predictions=None)
        self.assertIsNone(res["member2_soh_pct"])
        self.assertEqual(res["member2_soh_status"], "UNAVAILABLE")
        self.assertIsNone(res["member2_rul_cycles"])
        self.assertEqual(res["member2_rul_status"], "UNAVAILABLE")
        self.assertIsNone(res["battery_state"]["soh_pct"])
        self.assertIsNone(res["battery_state"]["range_km"])

    def test_member2_predictions_ingested_cleanly(self):
        """Verify authentic Member 2 predictions are ingested and preserved."""
        m2_input = {
            "soh_pct": 84.5,
            "rul_cycles": 320,
            "model_id": "nmc_lgbm_v1",
            "provenance": {"features_count": 29, "mae": 0.38},
        }
        res = simulate_bms_negotiation(member2_predictions=m2_input)
        self.assertEqual(res["member2_soh_pct"], 84.5)
        self.assertEqual(res["member2_soh_status"], "AVAILABLE")
        self.assertEqual(res["member2_rul_cycles"], 320)
        self.assertEqual(res["member2_rul_status"], "AVAILABLE")
        self.assertEqual(res["battery_state"]["soh_pct"], 84.5)
        self.assertIsNotNone(res["battery_state"]["range_km"])

    def test_genuine_multi_turn_negotiation_and_final_verification(self):
        """Verify Round 1 proposal -> BMS evaluation -> Round 2 driver response -> Final BMS check."""
        d_req = {"c_rate": 3.0, "target_soc": 100.0, "min_soc": 5.0, "persona": "aggressive"}
        res = simulate_bms_negotiation(driver_request=d_req, scenario="bms_supervised")
        transcript = res["negotiation_transcript"]
        self.assertEqual(len(transcript), 4)

        # Round 1 Driver
        self.assertEqual(transcript[0]["round"], 1)
        self.assertEqual(transcript[0]["action"]["c_rate"], 3.0)

        # Round 1 BMS
        self.assertEqual(transcript[1]["round"], 1)
        self.assertIn("modify", transcript[1]["decision"])
        self.assertIn("RULE_CRATE_CEILING", transcript[1]["rules_triggered"])

        # Round 2 Driver response
        self.assertEqual(transcript[2]["round"], 2)
        self.assertIn(transcript[2]["action_type"], ("counter", "accept"))

        # Round 2 BMS Final Verification
        self.assertEqual(transcript[3]["round"], 2)
        self.assertLessEqual(transcript[3]["applied_action"]["c_rate"], config.BMS_LIMITS["max_c_rate"])

    def test_dialogue_transcript_preserved_in_history(self):
        """Verify that agent messages and rules_triggered are preserved in history records."""
        res = run_simulation("NMC", "bms_supervised", "hot", is_hypothetical_simulation=True)
        self.assertGreater(len(res["history"]), 0)
        first_record = res["history"][0]
        self.assertIn("driver_message", first_record)
        self.assertIn("bms_message", first_record)
        self.assertIn("rules_triggered", first_record)
        self.assertIn("negotiation_round", first_record)
        self.assertTrue(len(first_record["driver_message"]) > 0)
        self.assertTrue(len(first_record["bms_message"]) > 0)

    def test_package_imports_and_reproducibility(self):
        """Verify package exports and deterministic reproducibility."""
        r1 = simulate_bms_negotiation(scenario="full_protection", climate="temperate")
        r2 = simulate_bms_negotiation(scenario="full_protection", climate="temperate")
        self.assertEqual(r1["applied_action"], r2["applied_action"])
        self.assertEqual(r1["decision"], r2["decision"])
        self.assertEqual(r1["rules_triggered"], r2["rules_triggered"])


if __name__ == "__main__":
    unittest.main()
