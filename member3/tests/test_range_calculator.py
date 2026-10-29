import math
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from range_calculator import RangeCalculator, VehicleConfig


def config(**overrides):
    raw = {
        "vehicle_name": "test_vehicle",
        "nominal_pack_energy_kwh": 60.0,
        "usable_energy_fraction": 0.9,
        "vehicle_mass_kg": 1800.0,
        "drag_coefficient_cd": 0.29,
        "frontal_area_m2": 2.2,
        "rolling_resistance_coefficient_crr": 0.01,
        "air_density_kg_m3": 1.225,
        "drivetrain_efficiency": 0.9,
        "regen_efficiency": 0.5,
        "max_regen_power_kw": 30.0,
        "auxiliary_power_kw": 0.4,
        "hvac_power_kw_default": 0.0,
        "rotational_inertia_factor": 1.05,
        "provenance": {"status": "TEST"}
    }
    raw.update(overrides)
    return VehicleConfig(**raw)


def cycle(speed=10.0, grade=0.0):
    return [
        {"time_s": 0.0, "speed_mps": speed, "grade_pct": grade},
        {"time_s": 60.0, "speed_mps": speed, "grade_pct": grade},
        {"time_s": 120.0, "speed_mps": speed, "grade_pct": grade},
    ]


class RangeCalculatorTests(unittest.TestCase):
    # =========================================================================
    # BASELINE TESTS (Retained for 100% backward compatibility)
    # =========================================================================
    def test_estimate_returns_positive_range(self):
        r = RangeCalculator(config()).estimate(cycle(), soh_pct=100)
        self.assertGreater(r["estimated_range_km"], 0)
        self.assertGreater(r["consumption_kwh_per_100km"], 0)

    def test_lower_soh_reduces_range_linearly_for_same_cycle(self):
        calc = RangeCalculator(config())
        full = calc.estimate(cycle(), soh_pct=100)["estimated_range_km"]
        degraded = calc.estimate(cycle(), soh_pct=80)["estimated_range_km"]
        self.assertAlmostEqual(degraded / full, 0.8, places=6)

    def test_hvac_load_reduces_range(self):
        calc = RangeCalculator(config())
        baseline = calc.estimate(cycle(), hvac_power_kw=0)["estimated_range_km"]
        hvac = calc.estimate(cycle(), hvac_power_kw=3.0)["estimated_range_km"]
        self.assertLess(hvac, baseline)

    def test_uphill_increases_energy_for_same_cycle(self):
        calc = RangeCalculator(config())
        flat = calc.estimate(cycle(grade=0))["consumption_kwh_per_100km"]
        uphill = calc.estimate(cycle(grade=5))["consumption_kwh_per_100km"]
        self.assertGreater(uphill, flat)

    def test_invalid_soh_rejected(self):
        with self.assertRaises(ValueError):
            RangeCalculator(config()).estimate(cycle(), soh_pct=0)

    def test_negative_speed_rejected_by_estimator(self):
        bad = cycle()
        bad[1]["speed_mps"] = -1
        with self.assertRaises(ValueError):
            RangeCalculator(config()).estimate(bad)

    def test_non_increasing_time_rejected(self):
        bad = cycle()
        bad[1]["time_s"] = 0
        with self.assertRaises(ValueError):
            RangeCalculator(config()).estimate(bad)

    def test_zero_distance_cycle_rejected(self):
        stopped = [
            {"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        with self.assertRaises(ValueError):
            RangeCalculator(config()).estimate(stopped)

    # =========================================================================
    # GROUP A: PHYSICS & CONSERVATION LAWS
    # =========================================================================
    def test_kinetic_energy_conservation_closed_cycle(self):
        """On a closed flat cycle with ideal 100% drivetrain & regen, kinetic work sums to zero."""
        cfg = config(
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=500.0,
            rolling_resistance_coefficient_crr=0.0,
            drag_coefficient_cd=0.0001,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        # Triangular speed cycle: 0 -> 20 -> 0 m/s (both speeds above 2.0 m/s for full regen)
        tri_cycle = [
            {"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 20.0, "grade_pct": 0.0},
            {"time_s": 20.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(tri_cycle, ambient_c=20.0, hvac_power_kw=0.0)
        # Analytical kinetic energy: 0.5 * 1800 * 20^2 = 360,000 J = 0.1 kWh
        # Traction energy should almost exactly equal regen recovered energy
        diff = abs(res["traction_energy_kwh"] - res["regen_recovered_kwh"])
        self.assertLess(diff, 0.005)

    def test_analytical_aerodynamic_drag_work(self):
        """Verify constant-speed aero work matches analytical formula 0.5 * rho * Cd * A * v^3 * dt."""
        cfg = config(
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
        )
        calc = RangeCalculator(cfg)
        v = 25.0  # m/s
        dt = 100.0  # s
        const_cycle = [
            {"time_s": 0.0, "speed_mps": v, "grade_pct": 0.0},
            {"time_s": dt, "speed_mps": v, "grade_pct": 0.0},
        ]
        res = calc.estimate(const_cycle, ambient_c=15.0, hvac_power_kw=0.0)
        rho = calc.air_density(15.0)
        expected_joules = 0.5 * rho * cfg.drag_coefficient_cd * cfg.frontal_area_m2 * (v**3) * dt
        expected_kwh = expected_joules / 3.6e6
        self.assertAlmostEqual(res["traction_energy_kwh"], expected_kwh, places=5)

    def test_analytical_rolling_resistance_work(self):
        """Verify flat constant-speed rolling work matches m * g * Crr * distance."""
        cfg = config(
            drag_coefficient_cd=0.00001,
            drivetrain_efficiency=1.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
        )
        calc = RangeCalculator(cfg)
        v = 10.0
        dt = 100.0
        const_cycle = [
            {"time_s": 0.0, "speed_mps": v, "grade_pct": 0.0},
            {"time_s": dt, "speed_mps": v, "grade_pct": 0.0},
        ]
        res = calc.estimate(const_cycle, hvac_power_kw=0.0)
        distance_m = v * dt
        expected_joules = cfg.vehicle_mass_kg * calc.G * cfg.rolling_resistance_coefficient_crr * distance_m
        expected_kwh = expected_joules / 3.6e6
        self.assertAlmostEqual(res["traction_energy_kwh"], expected_kwh, places=4)

    def test_gravitational_potential_energy_recovery(self):
        """Verify downhill elevation drop recovers m * g * delta_h energy."""
        cfg = config(
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=500.0,
            drag_coefficient_cd=0.0001,
            rolling_resistance_coefficient_crr=0.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
        )
        calc = RangeCalculator(cfg)
        # 1000m distance at -5% grade -> delta_h = 1000 * sin(atan(0.05)) ~= 49.937 m
        v = 20.0
        dt = 50.0  # distance = 1000 m
        downhill = [
            {"time_s": 0.0, "speed_mps": v, "grade_pct": -5.0},
            {"time_s": dt, "speed_mps": v, "grade_pct": -5.0},
        ]
        res = calc.estimate(downhill, hvac_power_kw=0.0)
        theta = math.atan(-0.05)
        expected_joules = abs(cfg.vehicle_mass_kg * calc.G * math.sin(theta) * 1000.0)
        expected_kwh = expected_joules / 3.6e6
        self.assertAlmostEqual(res["regen_recovered_kwh"], expected_kwh, places=4)

    def test_rotational_inertia_scaling(self):
        """Verify that rotational_inertia_factor increases kinetic acceleration work proportionally."""
        c1 = config(rotational_inertia_factor=1.00, drivetrain_efficiency=1.0, auxiliary_power_kw=0.0)
        c2 = config(rotational_inertia_factor=1.10, drivetrain_efficiency=1.0, auxiliary_power_kw=0.0)
        accel_cycle = [
            {"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 20.0, "grade_pct": 0.0},
        ]
        e1 = RangeCalculator(c1).estimate(accel_cycle, hvac_power_kw=0.0)["traction_energy_kwh"]
        e2 = RangeCalculator(c2).estimate(accel_cycle, hvac_power_kw=0.0)["traction_energy_kwh"]
        # Inertial work dominates this rapid acceleration, so e2 > e1
        self.assertGreater(e2, e1)

    # =========================================================================
    # GROUP B: REGENERATIVE BRAKING & LIMITS
    # =========================================================================
    def test_max_regen_power_clamping(self):
        """Severe deceleration should clamp regen power to max_regen_power_kw."""
        max_p = 20.0  # kW
        cfg = config(max_regen_power_kw=max_p, regen_efficiency=1.0)
        calc = RangeCalculator(cfg)
        # Decel from 30 m/s to 10 m/s in 2 seconds -> massive braking power (~360 kW mechanical)
        hard_brake = [
            {"time_s": 0.0, "speed_mps": 30.0, "grade_pct": 0.0},
            {"time_s": 2.0, "speed_mps": 10.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(hard_brake, hvac_power_kw=0.0)
        # Clamped regen energy over 2 seconds: max_p * 2 / 3600 kWh
        expected_max_kwh = max_p * (2.0 / 3600.0)
        self.assertAlmostEqual(res["regen_recovered_kwh"], expected_max_kwh, places=5)

    def test_low_speed_regen_cutoff(self):
        """Braking entirely below 0.5 m/s yields zero regenerative recovery."""
        cfg = config(regen_efficiency=0.8, max_regen_power_kw=50.0)
        calc = RangeCalculator(cfg)
        creep_brake = [
            {"time_s": 0.0, "speed_mps": 0.4, "grade_pct": 0.0},
            {"time_s": 5.0, "speed_mps": 0.1, "grade_pct": 0.0},
        ]
        res = calc.estimate(creep_brake, hvac_power_kw=0.0)
        self.assertEqual(res["regen_recovered_kwh"], 0.0)

    def test_zero_regen_efficiency(self):
        """When regen_efficiency is 0.0, no energy is recovered during deceleration."""
        cfg = config(regen_efficiency=0.0)
        calc = RangeCalculator(cfg)
        decel = [
            {"time_s": 0.0, "speed_mps": 20.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 5.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(decel, hvac_power_kw=0.0)
        self.assertEqual(res["regen_recovered_kwh"], 0.0)

    def test_sustained_downhill_cycle_handling(self):
        """A steep downhill descent with net recharge should return NET_GENERATING_DESCENT without crashing."""
        cfg = config(regen_efficiency=0.8, max_regen_power_kw=80.0, auxiliary_power_kw=0.1)
        calc = RangeCalculator(cfg)
        # Steep 15% descent for 2 km
        downhill = [
            {"time_s": 0.0, "speed_mps": 15.0, "grade_pct": -15.0},
            {"time_s": 100.0, "speed_mps": 15.0, "grade_pct": -15.0},
        ]
        res = calc.estimate(downhill, hvac_power_kw=0.0)
        self.assertEqual(res["quality_status"], "NET_GENERATING_DESCENT")
        self.assertEqual(res["estimated_range_km"], float("inf"))
        self.assertLess(res["consumption_kwh_per_km"], 0.0)

    # =========================================================================
    # GROUP C: ENVIRONMENTAL & TEMPERATURE COUPLING
    # =========================================================================
    def test_ideal_gas_air_density_cold_penalty(self):
        """Air density increases in cold weather (-10 C) vs mild (+15 C) following ideal gas law."""
        calc = RangeCalculator(config())
        rho_cold = calc.air_density(-10.0)
        rho_mild = calc.air_density(15.0)
        # Ideal gas law: 101325 / (287.058 * 263.15) = 1.3413 kg/m3 vs 1.2250 kg/m3 (~+9.5%)
        self.assertGreater(rho_cold, rho_mild)
        ratio = rho_cold / rho_mild
        self.assertAlmostEqual(ratio, 288.15 / 263.15, places=4)

    def test_empirical_hvac_temperature_scaling(self):
        """Cold ambient temperature triggers heating demand when hvac_power_kw is None."""
        calc = RangeCalculator(config(hvac_power_kw_default=0.2))
        res_cold = calc.estimate(cycle(), ambient_c=-10.0)
        res_mild = calc.estimate(cycle(), ambient_c=20.0)
        self.assertGreater(res_cold["hvac_power_kw"], res_mild["hvac_power_kw"])
        self.assertLess(res_cold["estimated_range_km"], res_mild["estimated_range_km"])

    def test_extreme_ambient_temperature_rejection(self):
        """Temperatures below -50 C or above +60 C must be rejected."""
        calc = RangeCalculator(config())
        with self.assertRaises(ValueError):
            calc.estimate(cycle(), ambient_c=-60.0)
        with self.assertRaises(ValueError):
            calc.estimate(cycle(), ambient_c=80.0)

    def test_hvac_manual_override_precedence(self):
        """Explicit hvac_power_kw parameter overrides automatic empirical temperature curve."""
        calc = RangeCalculator(config())
        res = calc.estimate(cycle(), ambient_c=-10.0, hvac_power_kw=1.5)
        self.assertEqual(res["hvac_power_kw"], 1.5)
        self.assertEqual(res["hvac_mode"], "MANUAL_OVERRIDE")

    # =========================================================================
    # GROUP D: INPUT VALIDATION & BOUNDARY CHECKS
    # =========================================================================
    def test_fractional_soh_trap_rejected(self):
        """Passing SOH as decimal fraction (e.g. 0.85 for 85%) must be rejected with helpful error."""
        calc = RangeCalculator(config())
        with self.assertRaises(ValueError) as ctx:
            calc.estimate(cycle(), soh_pct=0.85)
        self.assertIn("appears to be a fraction", str(ctx.exception))

    def test_soh_upper_bound_handling(self):
        """SOH <= 120% accepted with note (manufacturing margin); SOH > 120% rejected."""
        calc = RangeCalculator(config())
        res_fresh = calc.estimate(cycle(), soh_pct=103.0)
        self.assertIn("soh_provenance_note", res_fresh)
        with self.assertRaises(ValueError):
            calc.estimate(cycle(), soh_pct=125.0)

    def test_extreme_road_grade_rejected(self):
        """Road grade outside [-30%, 30%] must be rejected."""
        bad_grade = [
            {"time_s": 0.0, "speed_mps": 10.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 10.0, "grade_pct": 45.0},
        ]
        calc = RangeCalculator(config())
        with self.assertRaises(ValueError):
            calc.estimate(bad_grade)

    def test_excessive_hvac_power_rejected(self):
        """HVAC power exceeding 15 kW must be rejected."""
        calc = RangeCalculator(config())
        with self.assertRaises(ValueError):
            calc.estimate(cycle(), hvac_power_kw=25.0)

    def test_unrealistic_cycle_speed_rejected(self):
        """Cycle speed exceeding 120 m/s (432 km/h) must be rejected."""
        bad_speed = [
            {"time_s": 0.0, "speed_mps": 10.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 150.0, "grade_pct": 0.0},
        ]
        calc = RangeCalculator(config())
        with self.assertRaises(ValueError):
            calc.estimate(bad_speed)

    def test_vehicle_config_zero_drag_rejected(self):
        """Non-positive aerodynamic drag coefficient must be rejected."""
        with self.assertRaises(ValueError):
            config(drag_coefficient_cd=0.0)

    def test_vehicle_config_invalid_efficiency_rejected(self):
        """Drivetrain efficiency outside (0, 1] must be rejected."""
        with self.assertRaises(ValueError):
            config(drivetrain_efficiency=1.05)
        with self.assertRaises(ValueError):
            config(drivetrain_efficiency=0.0)

    def test_rotational_inertia_factor_below_one_rejected(self):
        """Rotational inertia factor < 1.0 must be rejected."""
        with self.assertRaises(ValueError):
            config(rotational_inertia_factor=0.95)

    # =========================================================================
    # GROUP E: MEMBER 2 INTEGRATION & TRAJECTORIES
    # =========================================================================
    def test_trajectory_estimation_monotonic_decay(self):
        """Trajectory evaluation over declining SOH series yields strictly decaying range."""
        calc = RangeCalculator(config())
        soh_series = [100.0, 95.0, 90.0, 85.0, 80.0]
        results = calc.estimate_trajectory(cycle(), soh_series, ambient_c=25.0, hvac_power_kw=0.5)
        self.assertEqual(len(results), 5)
        ranges = [r["estimated_range_km"] for r in results]
        for i in range(len(ranges) - 1):
            self.assertGreater(ranges[i], ranges[i+1])

    def test_quality_flag_passthrough(self):
        """Quality flags passed into estimate() or estimate_trajectory() must be preserved in output."""
        calc = RangeCalculator(config())
        res = calc.estimate(cycle(), quality_flag="MEMBER1_VALIDATED", cell_id="LFP_CELL_01")
        self.assertEqual(res["quality_flag"], "MEMBER1_VALIDATED")
        self.assertEqual(res["cell_id"], "LFP_CELL_01")

    def test_trajectory_with_dict_payload(self):
        """Trajectory accepts dicts with cycle_index, soh_pct, and quality_flag."""
        calc = RangeCalculator(config())
        trajectory_dicts = [
            {"cycle_index": 0, "soh_pct": 100.0, "quality_flag": "SYNTHETIC_TEST_ONLY"},
            {"cycle_index": 500, "soh_pct": 90.0, "quality_flag": "SYNTHETIC_TEST_ONLY_PREDICTED"},
            {"cycle_index": 1000, "soh_pct": 80.0, "quality_flag": "SYNTHETIC_TEST_ONLY"},
        ]
        results = calc.estimate_trajectory(cycle(), trajectory_dicts)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[1]["cycle_index"], 500)
        self.assertEqual(results[1]["quality_flag"], "SYNTHETIC_TEST_ONLY_PREDICTED")
        self.assertAlmostEqual(results[2]["estimated_range_km"] / results[0]["estimated_range_km"], 0.8, places=5)

    def test_non_uniform_drive_cycle_time_steps(self):
        """Cycle with non-uniform time steps (dt = 1, 5, 10, 2 s) integrates stably."""
        non_uniform = [
            {"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
            {"time_s": 1.0, "speed_mps": 5.0, "grade_pct": 0.0},
            {"time_s": 6.0, "speed_mps": 15.0, "grade_pct": 0.0},
            {"time_s": 16.0, "speed_mps": 15.0, "grade_pct": 0.0},
            {"time_s": 18.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        calc = RangeCalculator(config())
        res = calc.estimate(non_uniform, hvac_power_kw=0.0)
        self.assertGreater(res["distance_km"], 0.0)
        self.assertGreater(res["estimated_range_km"], 0.0)

    def test_load_cycle_from_csv(self):
        """Test load_cycle loads valid CSV with optional grade_pct defaulting to 0.0."""
        csv_content = "time_s,speed_mps\n0,0.0\n10,5.0\n20,10.0\n"
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        try:
            calc = RangeCalculator(config())
            loaded = calc.load_cycle(temp_path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(loaded[1]["grade_pct"], 0.0)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
