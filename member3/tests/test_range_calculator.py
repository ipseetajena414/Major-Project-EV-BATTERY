import json
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from range_calculator import CONFIG_FIELD_LIMITS, RangeCalculator, VehicleConfig


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
        # Triangular speed cycle: 0 -> 20 -> 0 m/s. Speeds above 2.0 m/s have full
        # regenerative blending; the 0-2.0 m/s tail of the braking ramp is where the
        # blend ramp and the 0.5 m/s cutoff reject recovery, so a small residual
        # (friction braking) remains and traction and regen agree within that band.
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
        # With ideal drivetrain and regen efficiency the closed cycle must balance:
        # traction - regen == friction braking + aerodynamic dissipation in the air
        # (rolling resistance is zero in this configuration). Aero loss is computed
        # independently from the exact integral of v(t)^3 for linear ramps.
        rho = res["air_density_kg_m3"]
        k_aero = 0.5 * rho * cfg.drag_coefficient_cd * cfg.frontal_area_m2
        # int v^3 dt for one linear 0 -> V ramp of duration dt is V^3 * dt / 4.
        aero_dissipated_j = 2.0 * k_aero * (20.0 ** 3 * 10.0 / 4.0)
        self.assertAlmostEqual(
            (res["traction_energy_kwh"] - res["regen_recovered_kwh"] - res["friction_braking_kwh"]) * 3.6e6,
            aero_dissipated_j,
            places=3,
        )
        # The blend ramp below 2.0 m/s is exactly the friction residual.
        self.assertGreater(res["friction_braking_kwh"], 0.0)
        self.assertLess(res["friction_braking_kwh"] * 3.6e6, 0.5 * cfg.vehicle_mass_kg * 2.0 ** 2)

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
        # The unbounded value must be labelled as unusable for range claims.
        limitation_text = " ".join(res["limitations"])
        self.assertIn("not a sustainable repeated-cycle consumption rate", limitation_text)
        self.assertIn("must never be reported as a driving range", limitation_text)

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

    # =========================================================================
    # GROUP F: MANDATORY REGRESSION TESTS (AUDIT VERIFICATION)
    # =========================================================================
    def test_braking_interval_crossing_low_speed_cutoff(self):
        """Braking interval crossing the 0.5 m/s cutoff recovers positive, analytical energy."""
        cfg = config(
            drag_coefficient_cd=1e-9,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=100.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        # Decelerate from 1.0 m/s to 0.0 m/s in 2.0 seconds
        cycle_cross = [
            {"time_s": 0.0, "speed_mps": 1.0, "grade_pct": 0.0},
            {"time_s": 2.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(cycle_cross, hvac_power_kw=0.0)
        # Analytical: beta_eff = (5/72) / 0.5 = 5/36 ~= 0.1388889
        # Mechanical work: 0.5 * 1800 * 1.0^2 = 900 J
        # Recovered: 900 * (5/36) = 125 J -> 125 / 3.6e6 kWh
        expected_kwh = 125.0 / 3.6e6
        self.assertGreater(res["regen_recovered_kwh"], 0.0)
        self.assertAlmostEqual(res["regen_recovered_kwh"], expected_kwh, places=8)

    def test_braking_interval_crossing_complete_blending_region(self):
        """Braking interval crossing full blending range (3.0 -> 0.0 m/s) matches exact integral."""
        cfg = config(
            drag_coefficient_cd=1e-9,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=100.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        cycle_full = [
            {"time_s": 0.0, "speed_mps": 3.0, "grade_pct": 0.0},
            {"time_s": 3.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(cycle_full, hvac_power_kw=0.0)
        # Analytical integral: int_0^3 beta(v)*v dv = int_2^3 v dv + int_0.5^2 ((v-0.5)/1.5)*v dv
        # = 2.5 + 1.125 = 3.625
        # Total work: 0.5 * 1800 * 3.0^2 = 8100 J
        # Recovered: 1800 * 3.625 = 6525 J -> 6525 / 3.6e6 kWh = 0.0018125 kWh
        expected_kwh = 6525.0 / 3.6e6
        self.assertAlmostEqual(res["regen_recovered_kwh"], expected_kwh, places=8)

    def test_equivalent_braking_intervals_split_into_smaller_time_steps(self):
        """Equivalent braking intervals split into sub-steps yield identical recovered energy."""
        cfg = config(
            drag_coefficient_cd=1e-9,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            regen_efficiency=0.85,
            max_regen_power_kw=500.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.05,
        )
        calc = RangeCalculator(cfg)

        # 1. Single 4-second interval from 4.0 m/s to 0.0 m/s (crosses 2.0 and 0.5 m/s)
        single_step = [
            {"time_s": 0.0, "speed_mps": 4.0, "grade_pct": 0.0},
            {"time_s": 4.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]

        # 2. Split exactly at boundary points: 4.0 -> 2.0, 2.0 -> 0.5, 0.5 -> 0.0
        boundary_split = [
            {"time_s": 0.0, "speed_mps": 4.0, "grade_pct": 0.0},
            {"time_s": 2.0, "speed_mps": 2.0, "grade_pct": 0.0},
            {"time_s": 3.5, "speed_mps": 0.5, "grade_pct": 0.0},
            {"time_s": 4.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]

        # 3. Fine-grained uniform discretization (40 steps of dt = 0.1s)
        fine_split = [
            {"time_s": round(0.1 * k, 2), "speed_mps": round(4.0 - 0.1 * k, 4), "grade_pct": 0.0}
            for k in range(41)
        ]

        res_single = calc.estimate(single_step, hvac_power_kw=0.0)
        res_boundary = calc.estimate(boundary_split, hvac_power_kw=0.0)
        res_fine = calc.estimate(fine_split, hvac_power_kw=0.0)

        self.assertAlmostEqual(res_single["regen_recovered_kwh"], res_boundary["regen_recovered_kwh"], places=7)
        self.assertAlmostEqual(res_single["regen_recovered_kwh"], res_fine["regen_recovered_kwh"], places=7)

    def test_max_regen_power_limiting_definition_and_electrical_clamping(self):
        """max_regen_power_kw defines maximum electrical charging power into the battery."""
        max_p_kw = 25.0
        eta_regen = 0.50
        cfg = config(
            max_regen_power_kw=max_p_kw,
            regen_efficiency=eta_regen,
            drivetrain_efficiency=1.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
        )
        calc = RangeCalculator(cfg)
        # Deceleration 30 -> 10 m/s in 2 s generates ~190-580 kW of mechanical braking
        # power, so the unconstrained electrical power (~95-290 kW) exceeds 25 kW for
        # the entire interval. The instantaneous limit and the old interval-average
        # limit therefore coincide here, and the expected recovery is P_max * dt.
        hard_brake = [
            {"time_s": 0.0, "speed_mps": 30.0, "grade_pct": 0.0},
            {"time_s": 2.0, "speed_mps": 10.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(hard_brake, hvac_power_kw=0.0)
        # Clamped electrical energy: 25.0 kW * 2.0 s / 3600.0 s/h
        expected_clamped_kwh = max_p_kw * (2.0 / 3600.0)
        self.assertAlmostEqual(res["regen_recovered_kwh"], expected_clamped_kwh, places=6)
        # Charging is capped electrically, so the mechanical power routed through the
        # motor is the electrical energy divided by regen efficiency, never more than
        # the mechanical braking energy actually available.
        routed_mech = res["regen_mechanical_absorbed_kwh"]
        self.assertAlmostEqual(routed_mech * eta_regen, res["regen_recovered_kwh"], places=9)
        self.assertLessEqual(routed_mech, res["mechanical_braking_kwh"] + 1e-12)
        self.assertAlmostEqual(
            res["friction_braking_kwh"],
            res["mechanical_braking_kwh"] - routed_mech,
            places=9,
        )
        # Model metadata must state that the limit is applied to instantaneous power
        # (updated in the Phase 1 rework: the previous text claimed interval-average
        # limiting, which the event-splitting integrator replaces).
        limitation_text = " ".join(res["limitations"]).lower()
        self.assertIn("instantaneous wheel power", limitation_text)
        self.assertIn("charge-power limit", limitation_text)
        self.assertNotIn("interval-average", limitation_text)

    def test_vehicle_config_rejects_unknown_fields(self):
        """VehicleConfig.from_json() must reject unknown JSON fields rather than silently ignoring them."""
        valid_json = {
            "vehicle_name": "test_ev",
            "nominal_pack_energy_kwh": 60.0,
            "usable_energy_fraction": 0.9,
            "vehicle_mass_kg": 1800.0,
            "drag_coefficient_cd": 0.29,
            "frontal_area_m2": 2.2,
            "rolling_resistance_coefficient_crr": 0.01,
            "unknown_bogus_field": 123.45,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(valid_json, f)
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                VehicleConfig.from_json(temp_path)
            self.assertIn("unknown_bogus_field", str(ctx.exception))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_vehicle_config_rejects_non_dict_json(self):
        """VehicleConfig.from_json() must reject non-dictionary JSON roots."""
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write(json.dumps(["item1", "item2"]))
            temp_path = f.name
        try:
            with self.assertRaises(ValueError) as ctx:
                VehicleConfig.from_json(temp_path)
            self.assertIn("dictionary", str(ctx.exception).lower())
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_hvac_behaviour_around_comfort_deadband(self):
        """HVAC power is continuous and non-decreasing departing the [18, 22] C deadband."""
        cfg = config(hvac_power_kw_default=0.25)
        calc = RangeCalculator(cfg)

        p_inside_18, mode_18 = calc.empirical_hvac_kw(18.0, None)
        p_inside_22, mode_22 = calc.empirical_hvac_kw(22.0, None)
        self.assertEqual(p_inside_18, 0.25)
        self.assertEqual(mode_18, "COMFORT_DEADBAND_VENTILATION")
        self.assertEqual(p_inside_22, 0.25)
        self.assertEqual(mode_22, "COMFORT_DEADBAND_VENTILATION")

        # Just below 18 C: heating demand starts continuously from baseline
        p_below_18, mode_below = calc.empirical_hvac_kw(17.9, None)
        self.assertEqual(mode_below, "EMPIRICAL_HEATING_DEMAND")
        self.assertGreaterEqual(p_below_18, 0.25)
        self.assertAlmostEqual(p_below_18, 0.25 + 0.18 * 0.1, places=5)

        # Just above 22 C: cooling demand starts continuously from baseline
        p_above_22, mode_above = calc.empirical_hvac_kw(22.1, None)
        self.assertEqual(mode_above, "EMPIRICAL_COOLING_DEMAND")
        self.assertGreaterEqual(p_above_22, 0.25)
        self.assertAlmostEqual(p_above_22, 0.25 + 0.12 * 0.1, places=5)

    def test_hvac_elevated_default_no_drop_outside_deadband(self):
        """HVAC demand does not unexpectedly decrease outside deadband when default is elevated."""
        # Baseline default higher than standard cooling cap (3.5 kW)
        cfg = config(hvac_power_kw_default=4.0)
        calc = RangeCalculator(cfg)

        p_deadband, _ = calc.empirical_hvac_kw(20.0, None)
        self.assertEqual(p_deadband, 4.0)

        # In cooling mode (e.g. 23 C), demand must not drop below 4.0 kW
        p_cool, _ = calc.empirical_hvac_kw(23.0, None)
        self.assertGreaterEqual(p_cool, 4.0)

        # In heating mode (e.g. 17 C), demand must not drop below 4.0 kW
        p_heat, _ = calc.empirical_hvac_kw(17.0, None)
        self.assertGreaterEqual(p_heat, 4.0)

        # Even higher baseline default (e.g. 7.0 kW, higher than 6.0 kW heating nominal cap)
        cfg_high = config(hvac_power_kw_default=7.0)
        calc_high = RangeCalculator(cfg_high)
        p_heat_high, _ = calc_high.empirical_hvac_kw(10.0, None)
        self.assertGreaterEqual(p_heat_high, 7.0)

    def test_hvac_across_allowed_temperature_range_and_config_validation(self):
        """HVAC monotonic across allowed range [-50, 60] C and rejects invalid default loads."""
        calc = RangeCalculator(config(hvac_power_kw_default=0.3))

        # Check valid boundary limits
        p_minus_50, _ = calc.empirical_hvac_kw(-50.0, None)
        p_plus_60, _ = calc.empirical_hvac_kw(60.0, None)
        self.assertLessEqual(p_minus_50, 15.0)
        self.assertLessEqual(p_plus_60, 15.0)

        # Monotonic heating response from 18 C down to -50 C
        temps_heating = [18.0, 15.0, 10.0, 0.0, -20.0, -50.0]
        powers_heating = [calc.empirical_hvac_kw(t, None)[0] for t in temps_heating]
        for i in range(len(powers_heating) - 1):
            self.assertLessEqual(powers_heating[i], powers_heating[i + 1])

        # Monotonic cooling response from 22 C up to 60 C
        temps_cooling = [22.0, 25.0, 30.0, 40.0, 50.0, 60.0]
        powers_cooling = [calc.empirical_hvac_kw(t, None)[0] for t in temps_cooling]
        for i in range(len(powers_cooling) - 1):
            self.assertLessEqual(powers_cooling[i], powers_cooling[i + 1])

        # Config validation for hvac_power_kw_default > 15.0 kW
        with self.assertRaises(ValueError):
            config(hvac_power_kw_default=15.1)

    def test_analytical_aerodynamic_drag_work_linear_speed(self):
        """Analytical aerodynamic drag work for linearly varying speed matches int 0.5*rho*Cd*A*v^3 dt."""
        cfg = config(
            drag_coefficient_cd=0.30,
            frontal_area_m2=2.4,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        v0 = 10.0
        v1 = 30.0
        dt = 10.0
        cycle_accel = [
            {"time_s": 0.0, "speed_mps": v0, "grade_pct": 0.0},
            {"time_s": dt, "speed_mps": v1, "grade_pct": 0.0},
        ]
        res = calc.estimate(cycle_accel, ambient_c=15.0, hvac_power_kw=0.0)
        rho = calc.air_density(15.0)

        # Exact analytical integral of v(t)^3 for linear v:
        # int_0^dt v^3 dt = dt * (v0 + v1) * (v0^2 + v1^2) / 4
        # = 10.0 * 40.0 * (100 + 900) / 4 = 100,000 m^4/s^3
        v3_int = dt * (v0 + v1) * (v0**2 + v1**2) / 4.0
        expected_aero_joules = 0.5 * rho * cfg.drag_coefficient_cd * cfg.frontal_area_m2 * v3_int
        expected_aero_kwh = expected_aero_joules / 3.6e6

        # Subtract known exact kinetic acceleration work: 0.5 * m * (30^2 - 10^2)
        kinetic_joules = 0.5 * cfg.vehicle_mass_kg * (v1**2 - v0**2)
        kinetic_kwh = kinetic_joules / 3.6e6

        measured_aero_kwh = res["traction_energy_kwh"] - kinetic_kwh
        self.assertAlmostEqual(measured_aero_kwh, expected_aero_kwh, places=6)

    def test_rotational_inertia_scaling_isolated(self):
        """Rotational inertia factor scales kinetic energy strictly proportionally when isolated."""
        c1 = config(
            rotational_inertia_factor=1.00,
            drag_coefficient_cd=1e-12,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
        )
        c2 = config(
            rotational_inertia_factor=1.20,
            drag_coefficient_cd=1e-12,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
        )
        accel_cycle = [
            {"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
            {"time_s": 10.0, "speed_mps": 25.0, "grade_pct": 0.0},
        ]
        e1 = RangeCalculator(c1).estimate(accel_cycle, hvac_power_kw=0.0)["traction_energy_kwh"]
        e2 = RangeCalculator(c2).estimate(accel_cycle, hvac_power_kw=0.0)["traction_energy_kwh"]

        # Exact ratio must equal 1.20 / 1.00 = 1.20
        self.assertAlmostEqual(e2 / e1, 1.20, places=6)


class RegenerativeBrakingIntegrationTests(unittest.TestCase):
    """Phase 1: regenerative braking driven by the instantaneous wheel-power profile.

    Every expectation is derived either in closed form or from an independent
    midpoint integration of the documented continuous model (`reference_interval`),
    never from the engine under test.
    """

    G = 9.80665

    def ideal_calc(self, **overrides):
        base = dict(
            drag_coefficient_cd=1e-9,
            frontal_area_m2=2.2,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=5000.0,  # non-binding: peak braking power here is <0.5 MW
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        base.update(overrides)
        return RangeCalculator(config(**base))

    def reference_interval(self, cfg, v0, v1, dt, grade_pct=0.0, rho=None, steps=40000):
        """Midpoint integration of P_wheel(v) = (m_eff*a + F_resist)*v + k_aero*v^3."""
        rho = cfg.air_density_kg_m3 if rho is None else rho
        m_eff = cfg.vehicle_mass_kg * cfg.rotational_inertia_factor
        k_aero = 0.5 * rho * cfg.drag_coefficient_cd * cfg.frontal_area_m2
        theta = math.atan(grade_pct / 100.0)
        f_resist = cfg.vehicle_mass_kg * self.G * (
            cfg.rolling_resistance_coefficient_crr * math.cos(theta) + math.sin(theta)
        )
        accel = (v1 - v0) / dt
        eta = cfg.regen_efficiency
        p_max_w = cfg.max_regen_power_kw * 1000.0
        h = dt / steps
        traction = braking = regen_mech = regen_elec = 0.0
        for i in range(steps):
            v = v0 + accel * (i + 0.5) * h
            p_wheel = (m_eff * accel + f_resist) * v + k_aero * v ** 3
            if p_wheel >= 0.0:
                traction += p_wheel * h
            else:
                p_brake = -p_wheel
                braking += p_brake * h
                if v <= 0.5:
                    beta = 0.0
                elif v >= 2.0:
                    beta = 1.0
                else:
                    beta = (v - 0.5) / 1.5
                p_elec = min(eta * beta * p_brake, p_max_w)
                regen_elec += p_elec * h
                regen_mech += (p_elec / eta) * h if eta > 0 else 0.0
        return {
            "traction_j": traction,
            "braking_j": braking,
            "regen_mech_j": regen_mech,
            "regen_elec_j": regen_elec,
        }

    def engine_interval(self, calc, cfg, v0, v1, dt, grade_pct=0.0, ambient_c=None):
        cycle = [
            {"time_s": 0.0, "speed_mps": v0, "grade_pct": grade_pct},
            {"time_s": dt, "speed_mps": v1, "grade_pct": grade_pct},
        ]
        res = calc.estimate(cycle, ambient_c=ambient_c, hvac_power_kw=0.0)
        # traction_energy_kwh is reported battery-side; convert back to wheel-side so
        # it can be compared against the reference wheel-power integral.
        return res, {
            "traction_j": res["traction_energy_kwh"] * 3.6e6 * cfg.drivetrain_efficiency,
            "braking_j": res["mechanical_braking_kwh"] * 3.6e6,
            "regen_mech_j": res["regen_mechanical_absorbed_kwh"] * 3.6e6,
            "regen_elec_j": res["regen_recovered_kwh"] * 3.6e6,
        }

    def assertMatchesReference(self, got, ref, places_rel=6):
        for key in ("traction_j", "braking_j", "regen_mech_j", "regen_elec_j"):
            scale = max(1.0, abs(ref[key]))
            self.assertLess(
                abs(got[key] - ref[key]) / scale,
                10.0 ** -places_rel,
                f"{key}: engine {got[key]!r} vs reference {ref[key]!r}",
            )

    def assertClose(self, actual, expected, rel=1e-9, msg=""):
        """Compare two energies with a relative tolerance."""
        tol = max(1e-9, abs(expected) * rel)
        self.assertLessEqual(abs(actual - expected), tol, msg or f"{actual!r} vs {expected!r}")

    # 1. Braking entirely below 0.5 m/s -> zero regeneration, everything to friction.
    def test_all_blending_below_low_cutoff_recovers_nothing(self):
        calc = self.ideal_calc()
        res, got = self.engine_interval(calc, calc.c, 0.5, 0.0, 5.0)
        self.assertEqual(res["regen_recovered_kwh"], 0.0)
        self.assertEqual(res["regen_mechanical_absorbed_kwh"], 0.0)
        expected_mech_j = 0.5 * calc.c.vehicle_mass_kg * 0.5 ** 2
        self.assertClose(got["braking_j"], expected_mech_j)
        self.assertClose(res["friction_braking_kwh"] * 3.6e6, expected_mech_j)

    # 2. Braking entirely above 2.0 m/s -> full blending availability.
    def test_all_blending_above_full_cutoff_recovers_all_available_energy(self):
        calc = self.ideal_calc()
        res, got = self.engine_interval(calc, calc.c, 25.0, 2.0, 46.0)
        expected_mech_j = 0.5 * calc.c.vehicle_mass_kg * (25.0 ** 2 - 2.0 ** 2)
        self.assertClose(got["braking_j"], expected_mech_j)
        self.assertClose(got["regen_elec_j"], expected_mech_j)
        self.assertEqual(res["friction_braking_kwh"], 0.0)

    # Blend ramp with an analytical closed form (also covers 1.0 -> 0 m/s).
    def test_braking_from_one_mps_to_zero_closed_form(self):
        calc = self.ideal_calc()
        res, got = self.engine_interval(calc, calc.c, 1.0, 0.0, 2.0)
        mechanical_j = 0.5 * calc.c.vehicle_mass_kg * 1.0 ** 2          # 900 J
        recovered_j = mechanical_j * (5.0 / 72.0) / 0.5                 # beta_eff = 5/36
        self.assertClose(got["braking_j"], mechanical_j)
        self.assertClose(got["regen_elec_j"], recovered_j)
        self.assertClose(res["friction_braking_kwh"] * 3.6e6, mechanical_j - recovered_j)

    # 3./4. Braking from 4.0 m/s to zero: exact blend-ramp integral.
    def test_braking_from_four_mps_to_zero_closed_form(self):
        calc = self.ideal_calc()
        res, got = self.engine_interval(calc, calc.c, 4.0, 0.0, 4.0)
        mechanical_j = 0.5 * calc.c.vehicle_mass_kg * 4.0 ** 2
        # int_0^4 beta(v) v dv = int_2^4 v dv + int_0.5^2 ((v-0.5)/1.5) v dv = 6 + 1.125
        recovered_j = calc.c.vehicle_mass_kg * 7.125
        self.assertClose(got["braking_j"], mechanical_j)
        self.assertClose(got["regen_elec_j"], recovered_j)
        self.assertClose(got["regen_elec_j"] + res["friction_braking_kwh"] * 3.6e6, mechanical_j)

    # 5. Braking from 20.0 m/s to zero: exact blend-ramp integral at realistic speed.
    def test_braking_from_twenty_mps_to_zero_closed_form(self):
        calc = self.ideal_calc()
        res, got = self.engine_interval(calc, calc.c, 20.0, 0.0, 10.0)
        mechanical_j = 0.5 * calc.c.vehicle_mass_kg * 20.0 ** 2          # 360,000 J
        recovered_j = calc.c.vehicle_mass_kg * (198.0 + 1.125)           # 358,425 J
        self.assertClose(got["braking_j"], mechanical_j)
        self.assertClose(got["regen_elec_j"], recovered_j)
        self.assertClose(res["friction_braking_kwh"] * 3.6e6, mechanical_j - recovered_j)

    # 6. Constant-speed downhill braking: road load, not inertia, supplies the energy.
    def test_constant_speed_downhill_braking_is_grade_power_limited(self):
        cfg = config(
            drag_coefficient_cd=0.30,
            frontal_area_m2=2.3,
            rolling_resistance_coefficient_crr=0.010,
            drivetrain_efficiency=1.0,
            regen_efficiency=0.70,
            max_regen_power_kw=5000.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        res, got = self.engine_interval(calc, cfg, 12.0, 12.0, 50.0, grade_pct=-8.0)
        ref = self.reference_interval(cfg, 12.0, 12.0, 50.0, grade_pct=-8.0)
        self.assertEqual(got["traction_j"], 0.0)
        self.assertGreater(got["braking_j"], 0.0)
        self.assertMatchesReference(got, ref, places_rel=5)
        # Zero acceleration, so the mechanical braking energy is exactly the negative
        # road-load work over the distance travelled.
        theta = math.atan(-0.08)
        road_force = cfg.vehicle_mass_kg * self.G * (
            cfg.rolling_resistance_coefficient_crr * math.cos(theta) + math.sin(theta)
        )
        aero_force = 0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient_cd * cfg.frontal_area_m2 * 12.0 ** 2
        expected_brake_j = -(road_force + aero_force) * 12.0 * 50.0
        self.assertClose(got["braking_j"], expected_brake_j)
        self.assertClose(res["friction_braking_kwh"], 0.0)

    # 7. Variable-speed downhill braking (grade + inertia + aero + roll together).
    def test_variable_speed_downhill_braking_matches_reference_model(self):
        cfg = config(
            drag_coefficient_cd=0.34,
            frontal_area_m2=2.4,
            rolling_resistance_coefficient_crr=0.012,
            drivetrain_efficiency=0.93,
            regen_efficiency=0.75,
            max_regen_power_kw=45.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.05,
        )
        calc = RangeCalculator(cfg)
        for v0, v1, dt, grade in ((15.0, 5.0, 20.0, -6.0), (8.0, 0.0, 16.0, -4.0), (25.0, 1.0, 30.0, -10.0)):
            res, got = self.engine_interval(calc, cfg, v0, v1, dt, grade)
            ref = self.reference_interval(cfg, v0, v1, dt, grade)
            self.assertMatchesReference(got, ref, places_rel=5)
            self.assertLessEqual(got["regen_mech_j"], got["braking_j"] + 1e-9)

    # 8. Aerodynamic and rolling resistance contributions change the blend weighting.
    def test_blend_weighting_follows_braking_power_not_kinetic_energy(self):
        """Aero-dominated braking must be power-weighted, not kinetic-energy-weighted."""
        cfg = config(
            drag_coefficient_cd=1.5,
            frontal_area_m2=4.0,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=5000.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        v0, v1, dt = 2.0, 0.5, 30.0
        res, got = self.engine_interval(calc, cfg, v0, v1, dt)
        ref = self.reference_interval(cfg, v0, v1, dt)
        self.assertMatchesReference(got, ref, places_rel=5)

        # The superseded model weighted blend beta(v) by kinetic energy (v dv);
        # weighting by the actual braking power must recover less in this case.
        kinetic_weighted_j = ref["braking_j"] * calc.effective_regen_blend_factor(v0, v1)
        power_weighted_j = ref["regen_elec_j"]
        self.assertGreater(kinetic_weighted_j, power_weighted_j)
        ratio = kinetic_weighted_j / power_weighted_j
        self.assertGreater(ratio, 1.01)
        self.assertLess(ratio, 1.10)
        self.assertClose(got["regen_elec_j"], power_weighted_j, rel=1e-5)

    # 9. Regenerative electrical power limiting resolved inside an interval.
    def test_charge_power_limit_crosses_inside_a_single_interval(self):
        cfg = config(
            drag_coefficient_cd=1e-9,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=1.0,
            regen_efficiency=0.8,
            max_regen_power_kw=20.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        v0, v1, dt = 30.0, 2.0, 20.0
        res, got = self.engine_interval(calc, cfg, v0, v1, dt)
        ref = self.reference_interval(cfg, v0, v1, dt)
        self.assertMatchesReference(got, ref, places_rel=5)

        # Analytic crossing speed where unconstrained electrical power equals 20 kW:
        # 0.8 * m_eff * |a| * v = 20 kW  ->  v = 20000 / (0.8 * 1800 * 1.4)
        v_cross = 20000.0 / (0.8 * cfg.vehicle_mass_kg * (28.0 / 20.0))
        self.assertGreater(v_cross, 2.0)
        capped_time_s = (v0 - v_cross) / (28.0 / 20.0)
        expected_elec_j = 20000.0 * capped_time_s + 0.8 * 0.5 * cfg.vehicle_mass_kg * (
            v_cross ** 2 - v1 ** 2
        )
        self.assertAlmostEqual(got["regen_elec_j"], expected_elec_j, delta=2.0)
        self.assertLess(got["regen_elec_j"], ref["braking_j"])

    def test_charge_power_limit_never_exceeded_by_instantaneous_power(self):
        cfg = config(max_regen_power_kw=15.0, regen_efficiency=0.85, drivetrain_efficiency=1.0,
                     drag_coefficient_cd=0.30, rolling_resistance_coefficient_crr=0.012,
                     auxiliary_power_kw=0.0, hvac_power_kw_default=0.0)
        calc = RangeCalculator(cfg)
        cycle = [
            {"time_s": 0.0, "speed_mps": 25.0, "grade_pct": 0.0},
            {"time_s": 60.0, "speed_mps": 5.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(cycle, hvac_power_kw=0.0)
        # A single 60 s interval cannot recover more than the 15 kW charge limit.
        self.assertLessEqual(res["regen_recovered_kwh"], 15.0 * 60.0 / 3600.0 + 1e-12)

    # 10. Regenerative efficiency below 100% is applied once, on the electrical side.
    def test_regen_efficiency_applied_once_and_not_double_counted(self):
        eta = 0.62
        calc = self.ideal_calc(regen_efficiency=eta)
        res, got = self.engine_interval(calc, calc.c, 20.0, 2.0, 18.0)
        self.assertClose(got["regen_elec_j"], eta * got["regen_mech_j"])
        self.assertClose(got["regen_mech_j"], got["braking_j"])
        self.assertEqual(res["friction_braking_kwh"], 0.0)
        self.assertAlmostEqual(
            res["cycle_energy_kwh"],
            res["traction_energy_kwh"] - res["regen_recovered_kwh"] + res["aux_hvac_energy_kwh"],
            places=9,
        )

    # 11. Friction braking absorbs exactly what regeneration cannot.
    def test_friction_braking_accounts_for_remaining_mechanical_energy(self):
        calc = self.ideal_calc(regen_efficiency=0.7, max_regen_power_kw=8.0)
        res, got = self.engine_interval(calc, calc.c, 22.0, 0.0, 22.0)
        self.assertGreater(res["friction_braking_kwh"], 0.0)
        self.assertAlmostEqual(
            res["friction_braking_kwh"] * 3.6e6,
            got["braking_j"] - got["regen_mech_j"],
            places=6,
        )
        self.assertGreaterEqual(got["regen_mech_j"], 0.0)
        self.assertLessEqual(got["regen_mech_j"], got["braking_j"] + 1e-9)

    # 12. Wheel power changes sign inside one interval.
    def test_wheel_power_sign_change_within_interval_is_split(self):
        cfg = config(
            drag_coefficient_cd=1.5,
            frontal_area_m2=4.0,
            rolling_resistance_coefficient_crr=0.0,
            drivetrain_efficiency=0.9,
            regen_efficiency=0.7,
            max_regen_power_kw=60.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
        )
        calc = RangeCalculator(cfg)
        v0, v1, dt = 40.0, 10.0, 15.0
        res, got = self.engine_interval(calc, cfg, v0, v1, dt)
        ref = self.reference_interval(cfg, v0, v1, dt)
        self.assertMatchesReference(got, ref, places_rel=5)
        # The old aggregation of net interval work reported no traction at all here.
        self.assertGreater(res["traction_energy_kwh"], 0.0)
        self.assertGreater(res["regen_recovered_kwh"], 0.0)
        analytic_sign_change_v = math.sqrt(
            -(cfg.vehicle_mass_kg * cfg.rotational_inertia_factor * (v1 - v0) / dt)
            / (0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient_cd * cfg.frontal_area_m2)
        )
        self.assertTrue(v1 < analytic_sign_change_v < v0)

    def legacy_interval_terms(self, calc, cfg, v0, v1, dt, grade_pct=0.0):
        """Reimplementation of the superseded interval-average aggregation.

        The previous engine integrated one net work figure per interval, charged the
        whole interval with a single traction-or-braking decision, and weighted the
        blend factor by kinetic energy. Kept here only as a comparison baseline: it
        silently drops the traction sub-piece of a mixed-sign interval.
        """
        theta = math.atan(grade_pct / 100.0)
        m_eff = cfg.vehicle_mass_kg * cfg.rotational_inertia_factor
        v_avg = 0.5 * (v0 + v1)
        ds = v_avg * dt
        w_inertia = 0.5 * m_eff * (v1**2 - v0**2)
        f_aero = 0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient_cd * cfg.frontal_area_m2 * (
            0.5 * (v0**2 + v1**2))
        w_aero = f_aero * ds
        f_roll = cfg.vehicle_mass_kg * self.G * cfg.rolling_resistance_coefficient_crr * math.cos(theta)
        w_grade = cfg.vehicle_mass_kg * self.G * math.sin(theta) * ds
        w_wheel = w_inertia + w_aero + (f_roll * ds if v_avg > 0 else 0.0) + w_grade
        if w_wheel >= 0:
            return w_wheel / cfg.drivetrain_efficiency, 0.0
        p_brake_kw = -w_wheel / (1000.0 * dt)
        p_elec_kw = min(
            p_brake_kw * cfg.regen_efficiency * calc.effective_regen_blend_factor(v0, v1),
            cfg.max_regen_power_kw,
        )
        return 0.0, p_elec_kw * 1000.0 * dt

    def test_sign_change_makes_net_energy_discretisation_independent(self):
        """Mixed traction/braking motion must integrate identically at any step size.

        The superseded model decided traction-vs-braking from one net work figure per
        interval, so the same motion discretised coarsely vs finely produced different
        battery energies and could flip the cycle between a finite range and a net
        energy gain. The engine must not depend on the sampling of the input cycle.
        """
        cfg = config(
            drag_coefficient_cd=1.5,
            frontal_area_m2=4.0,
            rolling_resistance_coefficient_crr=0.008,
            drivetrain_efficiency=0.90,
            regen_efficiency=0.70,
            max_regen_power_kw=60.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.05,
        )
        calc = RangeCalculator(cfg)
        coarse = [
            {"time_s": 0.0, "speed_mps": 40.0, "grade_pct": 0.0},
            {"time_s": 15.0, "speed_mps": 10.0, "grade_pct": 0.0},
        ]
        fine = [
            {"time_s": 15.0 * i / 40.0, "speed_mps": 40.0 - 30.0 * i / 40.0, "grade_pct": 0.0}
            for i in range(41)
        ]
        res_coarse = calc.estimate(coarse, hvac_power_kw=0.0)
        res_fine = calc.estimate(fine, hvac_power_kw=0.0)
        for key in (
            "traction_energy_kwh",
            "regen_recovered_kwh",
            "mechanical_braking_kwh",
            "friction_braking_kwh",
            "cycle_energy_kwh",
        ):
            self.assertAlmostEqual(res_coarse[key], res_fine[key], places=12)
        self.assertEqual(res_coarse["quality_status"], res_fine["quality_status"])

        # The superseded aggregation reported zero traction for this interval and a
        # different net battery energy, so its answer moved with the step size.
        legacy_traction_j, legacy_regen_j = self.legacy_interval_terms(
            calc, cfg, 40.0, 10.0, 15.0
        )
        self.assertEqual(legacy_traction_j, 0.0)
        self.assertGreater(res_coarse["traction_energy_kwh"], 0.0)
        self.assertNotAlmostEqual(legacy_regen_j / 3.6e6, res_coarse["regen_recovered_kwh"], places=6)
        # This cycle starts fast and ends slow, so a net gain is physically correct:
        # the status must be the honest generating-descent label, not a discretisation
        # artefact, and it must be identical for every subdivision of the same motion.
        self.assertEqual(res_coarse["quality_status"], "NET_GENERATING_DESCENT")
        self.assertEqual(res_coarse["estimated_range_km"], float("inf"))

    # 13. Equivalent drive cycles discretised differently agree (convergence/invariance).
    def test_time_step_discretisation_invariance(self):
        cfg = config(
            drag_coefficient_cd=0.32,
            frontal_area_m2=2.35,
            rolling_resistance_coefficient_crr=0.011,
            drivetrain_efficiency=0.92,
            regen_efficiency=0.78,
            max_regen_power_kw=35.0,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.05,
        )
        calc = RangeCalculator(cfg)
        v0, v1, total_t = 24.0, 0.0, 24.0
        ref = self.reference_interval(cfg, v0, v1, total_t)
        baseline = None
        for steps in (1, 2, 3, 7, 24, 240):
            cycle = [
                {"time_s": total_t * i / steps, "speed_mps": v0 + (v1 - v0) * i / steps, "grade_pct": 0.0}
                for i in range(steps + 1)
            ]
            res = calc.estimate(cycle, hvac_power_kw=0.0)
            got = {
                "traction_j": res["traction_energy_kwh"] * 3.6e6,
                "braking_j": res["mechanical_braking_kwh"] * 3.6e6,
                "regen_mech_j": res["regen_mechanical_absorbed_kwh"] * 3.6e6,
                "regen_elec_j": res["regen_recovered_kwh"] * 3.6e6,
            }
            if baseline is None:
                baseline = got
            for key in got:
                self.assertClose(got[key], baseline[key])
            self.assertMatchesReference(got, ref, places_rel=4)

    def test_udds_cycle_energy_stable_under_resampling(self):
        """Real 1 Hz UDDS trace vs 2 Hz resampling stays within convergence tolerance."""
        calc = RangeCalculator(config())
        cycle = calc.load_cycle(Path(__file__).resolve().parents[1] / "drive_cycles" / "udds_city.csv")
        coarse = cycle[::2]
        res_full = calc.estimate(cycle, hvac_power_kw=0.0)
        res_coarse = calc.estimate(coarse, hvac_power_kw=0.0)
        delta = abs(res_full["cycle_energy_kwh"] - res_coarse["cycle_energy_kwh"])
        self.assertLess(delta / abs(res_full["cycle_energy_kwh"]), 0.02)
        self.assertLessEqual(res_full["regen_recovered_kwh"], res_full["mechanical_braking_kwh"] + 1e-12)

    # 14. Ideal closed-speed cycle conserves energy exactly.
    def test_closed_speed_cycle_ideal_recovery_is_exact(self):
        calc = self.ideal_calc(regen_efficiency=1.0, drivetrain_efficiency=1.0)
        cycle = [
            {"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
            {"time_s": 15.0, "speed_mps": 15.0, "grade_pct": 0.0},
            {"time_s": 30.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        res = calc.estimate(cycle, hvac_power_kw=0.0)
        kinetic_j = 0.5 * calc.c.vehicle_mass_kg * 15.0 ** 2
        self.assertClose(res["traction_energy_kwh"] * 3.6e6, kinetic_j)
        self.assertClose(res["mechanical_braking_kwh"] * 3.6e6, kinetic_j)
        self.assertAlmostEqual(
            res["regen_recovered_kwh"] + res["friction_braking_kwh"],
            res["mechanical_braking_kwh"],
            places=9,
        )
        # Only the sub-2 m/s blend band may be lost; everything else is recovered.
        unrecovered_j = res["friction_braking_kwh"] * 3.6e6
        self.assertGreaterEqual(unrecovered_j, 0.0)
        self.assertLess(unrecovered_j, 0.5 * calc.c.vehicle_mass_kg * 2.0 ** 2)

    # Support: real-root solver used to locate charge-power limit crossings exactly.
    def test_cubic_real_roots_used_for_event_splitting(self):
        roots = RangeCalculator._cubic_real_roots(1.0, 0.0, -1.0, 0.0)   # v^3 - v
        self.assertEqual(len(roots), 3)
        for expected in (-1.0, 0.0, 1.0):
            self.assertTrue(any(abs(r - expected) < 1e-9 for r in roots))
        single = RangeCalculator._cubic_real_roots(1.0, 0.0, 1.0, 1.0)   # one real root
        self.assertEqual(len(single), 1)
        quad = RangeCalculator._cubic_real_roots(0.0, 1.0, -3.0, 2.0)    # degenerate
        self.assertEqual(len(quad), 2)

    # Support: blend availability itself.
    def test_regen_blend_at_speed_thresholds(self):
        calc = self.ideal_calc()
        self.assertEqual(calc.regen_blend_at_speed(0.0), 0.0)
        self.assertEqual(calc.regen_blend_at_speed(0.5), 0.0)
        self.assertClose(calc.regen_blend_at_speed(1.25), 0.5)
        self.assertEqual(calc.regen_blend_at_speed(2.0), 1.0)
        self.assertEqual(calc.regen_blend_at_speed(30.0), 1.0)

    # =========================================================================
    # NUMERICAL EDGE CASES
    # =========================================================================
    def ledger(self, res):
        return {
            "braking_j": res["mechanical_braking_kwh"] * 3.6e6,
            "regen_mech_j": res["regen_mechanical_absorbed_kwh"] * 3.6e6,
            "regen_elec_j": res["regen_recovered_kwh"] * 3.6e6,
            "friction_j": res["friction_braking_kwh"] * 3.6e6,
        }

    def assertLedgerConsistent(self, res, eta, cap_kw, duration_s):
        led = self.ledger(res)
        self.assertGreaterEqual(led["friction_j"], -1e-12)
        self.assertLessEqual(led["regen_mech_j"], led["braking_j"] + 1e-12)
        self.assertClose(led["braking_j"], led["regen_mech_j"] + led["friction_j"], rel=1e-9)
        if eta > 0.0:
            self.assertClose(led["regen_elec_j"], eta * led["regen_mech_j"], rel=1e-9)
        else:
            self.assertEqual(led["regen_elec_j"], 0.0)
        self.assertLessEqual(led["regen_elec_j"], cap_kw * 1000.0 * duration_s + 1e-6)
        return led

    # E1. Zero regenerative efficiency: everything the motor cannot take is friction.
    def test_zero_regen_efficiency_sends_all_braking_to_friction(self):
        cfg = config(regen_efficiency=0.0, max_regen_power_kw=50.0, drivetrain_efficiency=0.9,
                     auxiliary_power_kw=0.0, hvac_power_kw_default=0.0)
        calc = RangeCalculator(cfg)
        res, got = self.engine_interval(calc, cfg, 20.0, 0.0, 20.0)
        led = self.assertLedgerConsistent(res, cfg.regen_efficiency, cfg.max_regen_power_kw, 20.0)
        self.assertEqual(led["regen_elec_j"], 0.0)
        self.assertEqual(led["regen_mech_j"], 0.0)
        self.assertClose(led["friction_j"], led["braking_j"])
        self.assertGreater(led["braking_j"], 0.0)

    # E2. Zero charge-power limit: no electrical recovery even at full blend.
    def test_zero_charge_power_limit_recovers_nothing(self):
        cfg = config(regen_efficiency=0.7, max_regen_power_kw=0.0, drivetrain_efficiency=0.9,
                     auxiliary_power_kw=0.0, hvac_power_kw_default=0.0)
        calc = RangeCalculator(cfg)
        res, got = self.engine_interval(calc, cfg, 20.0, 0.0, 20.0)
        led = self.assertLedgerConsistent(res, cfg.regen_efficiency, cfg.max_regen_power_kw, 20.0)
        self.assertEqual(led["regen_elec_j"], 0.0)
        self.assertEqual(led["regen_mech_j"], 0.0)
        self.assertClose(led["friction_j"], led["braking_j"])

    # E3. Degenerate combination must not divide by zero.
    def test_zero_efficiency_with_zero_limit_does_not_raise(self):
        cfg = config(regen_efficiency=0.0, max_regen_power_kw=0.0, drivetrain_efficiency=0.9,
                     auxiliary_power_kw=0.0, hvac_power_kw_default=0.0)
        calc = RangeCalculator(cfg)
        res, _ = self.engine_interval(calc, cfg, 20.0, 0.0, 20.0)
        led = self.ledger(res)
        self.assertEqual(led["regen_elec_j"], 0.0)
        self.assertEqual(led["regen_mech_j"], 0.0)
        self.assertClose(led["friction_j"], led["braking_j"])

    # E4. Constant speed on the flat: pure traction, zero acceleration term.
    def test_constant_speed_flat_interval_is_pure_traction(self):
        cfg = config(rotational_inertia_factor=1.0, drivetrain_efficiency=1.0, regen_efficiency=0.7,
                     max_regen_power_kw=50.0, auxiliary_power_kw=0.0, hvac_power_kw_default=0.0)
        calc = RangeCalculator(cfg)
        v, dt = 15.0, 30.0
        res, got = self.engine_interval(calc, cfg, v, v, dt)
        theta = math.atan(0.0 / 100.0)
        expected_traction_j = (
            cfg.vehicle_mass_kg * self.G * cfg.rolling_resistance_coefficient_crr * math.cos(theta) * v
            + 0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient_cd * cfg.frontal_area_m2 * v ** 3
        ) * dt
        self.assertClose(got["traction_j"], expected_traction_j, rel=1e-12)
        self.assertEqual(got["braking_j"], 0.0)
        self.assertEqual(got["regen_elec_j"], 0.0)
        self.assertAlmostEqual(res["distance_km"], v * dt / 1000.0, places=12)

    # E5. Constant speed downhill (zero acceleration): braking is pure road-load power.
    def test_zero_acceleration_constant_speed_downhill_is_closed_form(self):
        calc = self.ideal_calc(regen_efficiency=0.7, max_regen_power_kw=5000.0)
        cfg = calc.c
        v, dt, grade = 15.0, 30.0, -6.0
        res, got = self.engine_interval(calc, cfg, v, v, dt, grade_pct=grade)
        theta = math.atan(grade / 100.0)
        k_aero = 0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient_cd * cfg.frontal_area_m2
        brake_j = -(cfg.vehicle_mass_kg * self.G * math.sin(theta) * v + k_aero * v ** 3) * dt
        self.assertClose(got["braking_j"], brake_j, rel=1e-12)
        self.assertEqual(got["traction_j"], 0.0)
        self.assertClose(got["regen_elec_j"], 0.7 * brake_j, rel=1e-12)
        # beta = 1 and the cap is far away, so every joule of braking is routed through
        # the motor: friction braking is zero and the 30 % is the regen efficiency loss
        # on the electrical side, not brake dissipation.
        self.assertClose(got["regen_mech_j"], brake_j, rel=1e-12)
        self.assertClose(self.ledger(res)["friction_j"], 0.0)

    # E6. Speeds that land exactly on a blend threshold must not be double-split.
    def test_blend_threshold_boundary_speeds_exact_values(self):
        calc = self.ideal_calc(regen_efficiency=1.0, max_regen_power_kw=5000.0)
        mass = calc.c.vehicle_mass_kg
        # 0.5 -> 0: entirely below the low cutoff -> no blending at all.
        res, got = self.engine_interval(calc, calc.c, 0.5, 0.0, 5.0)
        self.assertClose(got["braking_j"], 0.5 * mass * 0.5 ** 2, rel=1e-9)
        self.assertEqual(got["regen_elec_j"], 0.0)
        self.assertClose(self.ledger(res)["friction_j"], got["braking_j"], rel=1e-9)
        # 2.0 -> 0.5: starts exactly at full blend, ends exactly at zero blend.
        # recovered = m * integral of beta(v) v dv over [0.5, 2.0] = m * 1.125
        res, got = self.engine_interval(calc, calc.c, 2.0, 0.5, 10.0)
        self.assertClose(got["braking_j"], 0.5 * mass * (2.0 ** 2 - 0.5 ** 2), rel=1e-9)
        self.assertClose(got["regen_elec_j"], mass * 1.125, rel=1e-9)
        # 2.0 -> 0: full blend band loss only.
        res, got = self.engine_interval(calc, calc.c, 2.0, 0.0, 10.0)
        self.assertClose(got["braking_j"], 0.5 * mass * 2.0 ** 2, rel=1e-9)
        self.assertClose(got["regen_elec_j"], mass * 1.125, rel=1e-9)
        self.assertClose(self.ledger(res)["friction_j"], 0.5 * mass * 2.0 ** 2 - mass * 1.125, rel=1e-9)
        # 20 -> 2.0: ends exactly at the full-blend threshold, so nothing is lost.
        res, got = self.engine_interval(calc, calc.c, 20.0, 2.0, 18.0)
        self.assertClose(self.ledger(res)["friction_j"], 0.0)
        self.assertClose(got["regen_elec_j"], got["braking_j"], rel=1e-9)

    # E7. Repeated root of the power-polynomial derivative: cap tangent to the profile.
    def test_charge_limit_tangent_to_braking_power_loses_nothing(self):
        cfg = config(drag_coefficient_cd=1.5, frontal_area_m2=4.0, rolling_resistance_coefficient_crr=0.0,
                     drivetrain_efficiency=1.0, regen_efficiency=0.7, auxiliary_power_kw=0.0,
                     hvac_power_kw_default=0.0, rotational_inertia_factor=1.05)
        v0, v1, dt = 20.0, 0.0, 20.0
        accel = (v1 - v0) / dt
        k_aero = 0.5 * cfg.air_density_kg_m3 * cfg.drag_coefficient_cd * cfg.frontal_area_m2
        # |P| = -(m_eff*a*v + k v^3) peaks interior to the interval at v_p:
        v_peak = math.sqrt(-cfg.vehicle_mass_kg * cfg.rotational_inertia_factor * accel / (3.0 * k_aero))
        self.assertTrue(2.0 < v_peak < v0)
        peak_elec_kw = cfg.regen_efficiency * (
            -(cfg.vehicle_mass_kg * cfg.rotational_inertia_factor * accel * v_peak + k_aero * v_peak ** 3)
        ) / 1000.0
        tangent = config(**{**cfg.__dict__, "max_regen_power_kw": peak_elec_kw})
        uncapped = config(**{**cfg.__dict__, "max_regen_power_kw": 5000.0})
        led_t = self.ledger(RangeCalculator(tangent).estimate(
            [{"time_s": 0.0, "speed_mps": v0, "grade_pct": 0.0},
             {"time_s": dt, "speed_mps": v1, "grade_pct": 0.0}], hvac_power_kw=0.0))
        led_u = self.ledger(RangeCalculator(uncapped).estimate(
            [{"time_s": 0.0, "speed_mps": v0, "grade_pct": 0.0},
             {"time_s": dt, "speed_mps": v1, "grade_pct": 0.0}], hvac_power_kw=0.0))
        # A cap set exactly to the peak must not clip any energy (measure-zero event).
        for key in ("braking_j", "regen_mech_j", "regen_elec_j", "friction_j"):
            self.assertClose(led_t[key], led_u[key], rel=1e-9)
        # A slightly tighter cap must bite, proving the crossing detection is not a no-op.
        tighter = config(**{**cfg.__dict__, "max_regen_power_kw": 0.9 * peak_elec_kw})
        led_g = self.ledger(RangeCalculator(tighter).estimate(
            [{"time_s": 0.0, "speed_mps": v0, "grade_pct": 0.0},
             {"time_s": dt, "speed_mps": v1, "grade_pct": 0.0}], hvac_power_kw=0.0))
        self.assertLess(led_g["regen_elec_j"], led_u["regen_elec_j"])
        self.assertLedgerConsistent(
            RangeCalculator(tighter).estimate(
                [{"time_s": 0.0, "speed_mps": v0, "grade_pct": 0.0},
                 {"time_s": dt, "speed_mps": v1, "grade_pct": 0.0}], hvac_power_kw=0.0),
            cfg.regen_efficiency, 0.9 * peak_elec_kw, dt)

    # E8/E9. Cap crossing exactly at an interval boundary, and no crossing at all.
    def test_charge_limit_crossing_at_interval_boundary_matches_oracle(self):
        cfg_base = config(drag_coefficient_cd=1.5, frontal_area_m2=4.0,
                          rolling_resistance_coefficient_crr=0.0, drivetrain_efficiency=1.0,
                          regen_efficiency=0.7, auxiliary_power_kw=0.0, hvac_power_kw_default=0.0,
                          rotational_inertia_factor=1.0)
        v0, v1, dt = 12.0, 5.0, 14.0
        accel = (v1 - v0) / dt
        k_aero = 0.5 * cfg_base.air_density_kg_m3 * cfg_base.drag_coefficient_cd * cfg_base.frontal_area_m2

        def elec_power_kw(v):
            return cfg_base.regen_efficiency * (
                -(cfg_base.vehicle_mass_kg * accel * v + k_aero * v ** 3)) / 1000.0

        for label, cap_kw in (("at interval end", elec_power_kw(v1)), ("at interval start", elec_power_kw(v0))):
            cfg = config(**{**cfg_base.__dict__, "max_regen_power_kw": cap_kw})
            calc = RangeCalculator(cfg)
            res, got = self.engine_interval(calc, cfg, v0, v1, dt)
            ref = self.reference_interval(cfg, v0, v1, dt)
            self.assertMatchesReference(got, ref, places_rel=4)
            self.assertLedgerConsistent(res, cfg.regen_efficiency, cap_kw, dt)
        # Far above the profile: identical to the uncapped answer.
        cfg_hi = config(**{**cfg_base.__dict__, "max_regen_power_kw": 5000.0})
        calc_hi = RangeCalculator(cfg_hi)
        _, got_hi = self.engine_interval(calc_hi, cfg_hi, v0, v1, dt)
        self.assertMatchesReference(got_hi, self.reference_interval(cfg_hi, v0, v1, dt), places_rel=4)

    # E10. Closed-form cubic solver on repeated, degenerate and trivial polynomials.
    def test_cubic_real_roots_repeated_and_degenerate_cases(self):
        cases = [
            ((1.0, 0.0, -3.0, 2.0), [-2.0, 1.0]),      # (v+2)(v-1)^2 -> repeated root
            ((1.0, -3.0, 3.0, -1.0), [1.0]),           # (v-1)^3
            ((1.0, 3.0, 3.0, 1.0), [-1.0]),            # (v+1)^3
            ((1.0, 0.0, 0.0, 0.0), [0.0]),             # v^3
            ((1.0, 0.0, 0.0, 1.0), [-1.0]),            # one real root
            ((0.0, 1.0, 0.0, -4.0), [-2.0, 2.0]),      # degenerate quadratic
            ((0.0, 0.0, 1.0, -1.0), [1.0]),            # degenerate linear
            ((0.0, 0.0, 0.0, 1.0), []),                # constant -> no root
        ]
        for coeffs, expected in cases:
            got = sorted(RangeCalculator._cubic_real_roots(*coeffs))
            self.assertEqual(len(got), len(expected), f"root count for {coeffs}: {got}")
            for g, e in zip(got, expected):
                self.assertAlmostEqual(g, e, places=8, msg=f"{coeffs}: {got} vs {expected}")

    # E11. Velocity moments must stay finite for constant-speed and zero-duration pieces.
    def test_speed_moment_degenerate_arguments(self):
        self.assertClose(RangeCalculator._speed_moment(3, 5.0, 5.0, 2.0), 2.0 * 5.0 ** 3, rel=1e-15)
        self.assertEqual(RangeCalculator._speed_moment(4, 5.0, 9.0, 0.0), 0.0)
        self.assertClose(RangeCalculator._speed_moment(1, 0.0, 10.0, 10.0), 50.0, rel=1e-15)
        self.assertClose(RangeCalculator._speed_moment(3, 0.0, 10.0, 10.0), 10.0 * 10.0 ** 3 / 4.0, rel=1e-12)

    # =========================================================================
    # CONVERGENCE: SAME TRAJECTORY vs DIFFERENT TRAJECTORY
    # =========================================================================
    @staticmethod
    def refine_cycle(cycle, k):
        """Linearly interpolate k sub-points into every interval, preserving the profile."""
        out = []
        for a, b in zip(cycle[:-1], cycle[1:]):
            for j in range(k):
                f = j / k
                out.append({
                    "time_s": a["time_s"] + (b["time_s"] - a["time_s"]) * f,
                    "speed_mps": a["speed_mps"] + (b["speed_mps"] - a["speed_mps"]) * f,
                    "grade_pct": a.get("grade_pct", 0.0)
                    + (b.get("grade_pct", 0.0) - a.get("grade_pct", 0.0)) * f,
                })
        out.append(dict(cycle[-1]))
        return out

    # E12. Refining the identical trajectory must not change any energy (exactness).
    def test_refinement_of_same_trajectory_is_energy_exact(self):
        calc = RangeCalculator(config())
        cycle = calc.load_cycle(Path(__file__).resolve().parents[1] / "drive_cycles" / "udds_city.csv")
        keys = ("traction_energy_kwh", "regen_recovered_kwh", "mechanical_braking_kwh",
                "regen_mechanical_absorbed_kwh", "friction_braking_kwh", "cycle_energy_kwh",
                "estimated_range_km")
        baseline = calc.estimate(cycle, hvac_power_kw=0.0)
        for k in (2, 4, 8):
            refined = self.refine_cycle(cycle, k)
            self.assertAlmostEqual(refined[1]["speed_mps"], cycle[0]["speed_mps"]
                                   + (cycle[1]["speed_mps"] - cycle[0]["speed_mps"]) / k, places=12)
            res = calc.estimate(refined, hvac_power_kw=0.0)
            for key in keys:
                self.assertClose(res[key], baseline[key], rel=1e-9,
                                 msg=f"k={k} refinement changed {key}")
            self.assertAlmostEqual(res["distance_km"], baseline["distance_km"], places=9)

    # E13. Downsampling changes the input trajectory, not the integrator's accuracy.
    def test_downsampling_is_an_input_representation_change(self):
        """Reported 1 Hz vs 2 Hz / 5 Hz differences are sampling error, bounded by measurement.

        Measured on the flat supplied UDDS trace with the default test vehicle config
        (reference air density, HVAC off): -0.97 % at 2 Hz and -3.48 % at 5 Hz. These
        are NOT convergence failures of the solver -- the integrator is exact for
        whatever trajectory it is given (E12) -- they are the cost of representing a
        1 Hz road with a coarser polyline. The bounds below are the observed values
        rounded up, not an arbitrary tolerance.
        """
        calc = RangeCalculator(config())
        cycle = calc.load_cycle(Path(__file__).resolve().parents[1] / "drive_cycles" / "udds_city.csv")
        full = calc.estimate(cycle, hvac_power_kw=0.0)
        for step, bound in ((2, 0.015), (5, 0.05)):
            decimated = cycle[::step]
            res = calc.estimate(decimated, hvac_power_kw=0.0)
            drift = (res["cycle_energy_kwh"] - full["cycle_energy_kwh"]) / full["cycle_energy_kwh"]
            self.assertNotAlmostEqual(res["cycle_energy_kwh"], full["cycle_energy_kwh"], places=6)
            self.assertLess(abs(drift), bound)
            # The decimated trajectory is itself integrated exactly: refining it further
            # changes nothing, which isolates the drift above as input sampling.
            res_refined = calc.estimate(self.refine_cycle(decimated, 8), hvac_power_kw=0.0)
            self.assertClose(res_refined["cycle_energy_kwh"], res["cycle_energy_kwh"], rel=1e-9)

    # E14. One-shot kinetic harvest must be labelled as not a driving range.
    def test_kinetic_harvest_descent_status_carries_sustainability_caveat(self):
        cfg = config(drag_coefficient_cd=1.5, frontal_area_m2=4.0,
                     rolling_resistance_coefficient_crr=0.008, drivetrain_efficiency=0.90,
                     regen_efficiency=0.70, max_regen_power_kw=60.0, auxiliary_power_kw=0.0,
                     hvac_power_kw_default=0.0, rotational_inertia_factor=1.05)
        calc = RangeCalculator(cfg)
        res = calc.estimate(
            [{"time_s": 0.0, "speed_mps": 40.0, "grade_pct": 0.0},
             {"time_s": 15.0, "speed_mps": 10.0, "grade_pct": 0.0}], hvac_power_kw=0.0)
        self.assertEqual(res["quality_status"], "NET_GENERATING_DESCENT")
        text = " ".join(res["limitations"])
        self.assertIn("initial", text)
        self.assertIn("not a sustainable repeated-cycle consumption rate", text)
        # A normal consuming cycle must not carry the caveat.
        normal = calc.estimate(
            [{"time_s": 0.0, "speed_mps": 0.0, "grade_pct": 0.0},
             {"time_s": 20.0, "speed_mps": 20.0, "grade_pct": 0.0},
             {"time_s": 40.0, "speed_mps": 0.0, "grade_pct": 0.0}], hvac_power_kw=0.0)
        self.assertNotEqual(normal["quality_status"], "NET_GENERATING_DESCENT")
        self.assertNotIn("repeated-cycle", " ".join(normal["limitations"]))


class VehicleConfigValidationTests(unittest.TestCase):
    """Phase 2 §2/§3: field types, physical envelopes, provenance structure, JSON rules."""

    PROFILE_DIR = ROOT / "vehicle_configs"

    REQUIRED = (
        "vehicle_name",
        "nominal_pack_energy_kwh",
        "usable_energy_fraction",
        "vehicle_mass_kg",
        "drag_coefficient_cd",
        "frontal_area_m2",
        "rolling_resistance_coefficient_crr",
    )

    def profiles(self):
        return sorted(self.PROFILE_DIR.glob("*.json")) + [ROOT / "vehicle_config.json"]

    def minimal(self, **overrides):
        raw = {
            "vehicle_name": "minimal_vehicle",
            "nominal_pack_energy_kwh": 40.0,
            "usable_energy_fraction": 0.9,
            "vehicle_mass_kg": 1200.0,
            "drag_coefficient_cd": 0.35,
            "frontal_area_m2": 2.2,
            "rolling_resistance_coefficient_crr": 0.01,
        }
        raw.update(overrides)
        return raw

    def write_json(self, payload):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    # --- backward compatibility of the shipped profiles ---
    def test_every_shipped_configuration_still_loads_and_validates(self):
        paths = self.profiles()
        self.assertEqual(len(paths), 10)  # nine profiles plus the root example config
        for path in paths:
            cfg = VehicleConfig.from_json(path)
            cfg.validate()
            cfg.validate()
            self.assertTrue(cfg.vehicle_name.strip(), path.name)
            self.assertIsInstance(cfg.provenance, dict, path.name)
            self.assertTrue(cfg.provenance["status"], path.name)

    def test_every_shipped_configuration_still_produces_a_finite_estimate(self):
        sample = cycle(15.0)
        for path in self.profiles():
            result = RangeCalculator(VehicleConfig.from_json(path)).estimate(
                sample, soh_pct=90.0, ambient_c=25.0)
            self.assertTrue(math.isfinite(result["consumption_kwh_per_100km"]), path.name)
            self.assertGreater(result["estimated_range_km"], 0.0, path.name)

    def test_shipped_configurations_estimate_without_environment_inputs(self):
        """§7: omitting all Phase 2 environmental arguments keeps legacy air density."""
        for path in self.profiles():
            cfg = VehicleConfig.from_json(path)
            result = RangeCalculator(cfg).estimate(cycle(15.0))
            self.assertEqual(result["air_density_kg_m3"], cfg.air_density_kg_m3, path.name)
            self.assertEqual(result["air_density_source"], "CONFIG_REFERENCE", path.name)
            self.assertEqual(result["headwind_mps"], 0.0, path.name)

    def test_validation_does_not_rewrite_supplied_values(self):
        cfg = config(nominal_pack_energy_kwh=60, vehicle_mass_kg=1800)
        cfg.validate()
        self.assertEqual(cfg.nominal_pack_energy_kwh, 60)
        self.assertIs(cfg.nominal_pack_energy_kwh, 60)  # still the same int object
        self.assertEqual(cfg.vehicle_mass_kg, 1800)

    def test_invalid_value_is_rejected_rather_than_replaced_by_default(self):
        with self.assertRaises(ValueError):
            config(nominal_pack_energy_kwh=0.0)
        with self.assertRaises(ValueError):
            config(vehicle_mass_kg=-1500.0)
        with self.assertRaises(ValueError):
            config(usable_energy_fraction=0.0)

    # --- JSON document rules ---
    def test_unknown_json_key_is_rejected(self):
        path = self.write_json(self.minimal(rolling_resistance_coeffient=0.01))
        with self.assertRaises(ValueError) as ctx:
            VehicleConfig.from_json(path)
        self.assertIn("Unknown field(s)", str(ctx.exception))
        self.assertIn("rolling_resistance_coeffient", str(ctx.exception))

    def test_non_dict_json_is_rejected(self):
        path = self.write_json([self.minimal()])
        with self.assertRaises(ValueError) as ctx:
            VehicleConfig.from_json(path)
        self.assertIn("must be a dictionary object", str(ctx.exception))

    def test_each_missing_required_field_gives_an_informative_error(self):
        for name in self.REQUIRED:
            payload = self.minimal()
            del payload[name]
            path = self.write_json(payload)
            with self.assertRaises(ValueError) as ctx:
                VehicleConfig.from_json(path)
            message = str(ctx.exception)
            self.assertIn("Missing required field(s)", message)
            self.assertIn(name, message)
            self.assertNotIn("unexpected keyword", message)

    def test_optional_fields_use_documented_defaults(self):
        cfg = VehicleConfig(**self.minimal())
        self.assertEqual(cfg.air_density_kg_m3, 1.225)
        self.assertEqual(cfg.drivetrain_efficiency, 0.90)
        self.assertEqual(cfg.regen_efficiency, 0.0)
        self.assertEqual(cfg.max_regen_power_kw, 0.0)
        self.assertEqual(cfg.auxiliary_power_kw, 0.0)
        self.assertEqual(cfg.hvac_power_kw_default, 0.0)
        self.assertEqual(cfg.rotational_inertia_factor, 1.05)
        self.assertIsNone(cfg.rated_range_km_reference)
        self.assertIsNone(cfg.provenance)

    # --- type rules ---
    def test_boolean_values_rejected_for_every_numeric_field(self):
        for name in CONFIG_FIELD_LIMITS:
            with self.assertRaises(ValueError, msg=f"{name} accepted True") as ctx:
                config(**{name: True})
            self.assertIn("not a boolean", str(ctx.exception))

    def test_nan_and_infinity_rejected_for_every_numeric_field(self):
        for name in CONFIG_FIELD_LIMITS:
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.assertRaises(ValueError, msg=f"{name} accepted {bad}") as ctx:
                    config(**{name: bad})
                self.assertIn("must be", str(ctx.exception))

    def test_non_numeric_types_rejected(self):
        for bad in ("1800", None, [1800], {"kg": 1800}, 1 + 2j):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                config(vehicle_mass_kg=bad)

    def test_estimate_rejects_boolean_api_inputs(self):
        calc = RangeCalculator(config())
        for kwargs in (
            {"soh_pct": True},
            {"ambient_c": True},
            {"hvac_power_kw": True},
            {"pressure_pa": True},
            {"altitude_m": True},
            {"headwind_mps": True},
            {"crosswind_mps": True},
        ):
            with self.assertRaises(ValueError, msg=f"accepted {kwargs}") as ctx:
                calc.estimate(cycle(), **kwargs)
            self.assertIn("boolean", str(ctx.exception))

    def test_estimate_rejects_non_finite_api_inputs(self):
        calc = RangeCalculator(config())
        for kwargs in (
            {"soh_pct": float("nan")},
            {"ambient_c": float("inf")},
            {"hvac_power_kw": float("nan")},
            {"pressure_pa": float("inf")},
            {"altitude_m": float("nan")},
            {"headwind_mps": float("inf")},
        ):
            with self.assertRaises(ValueError, msg=f"accepted {kwargs}"):
                calc.estimate(cycle(), **kwargs)

    def test_cycle_points_reject_boolean_and_string_values(self):
        calc = RangeCalculator(config())
        for bad_point in (
            {"time_s": 0.0, "speed_mps": True, "grade_pct": 0.0},
            {"time_s": 0.0, "speed_mps": 10.0, "grade_pct": "5"},
            {"time_s": float("nan"), "speed_mps": 10.0, "grade_pct": 0.0},
        ):
            bad = [bad_point, {"time_s": 60.0, "speed_mps": 10.0, "grade_pct": 0.0}]
            with self.assertRaises(ValueError, msg=f"accepted {bad_point}"):
                calc.estimate(bad)

    # --- physical envelope rules ---
    def test_envelope_rejects_unit_and_transposition_errors(self):
        bad_values = {
            "nominal_pack_energy_kwh": 251.0,
            "usable_energy_fraction": 1.0001,
            "vehicle_mass_kg": 20001.0,
            "drag_coefficient_cd": 2.5,
            "frontal_area_m2": 25.0,
            "rolling_resistance_coefficient_crr": 0.06,
            "air_density_kg_m3": 2.5,
            "drivetrain_efficiency": 0.0,
            "regen_efficiency": 1.1,
            "max_regen_power_kw": 6000.0,
            "auxiliary_power_kw": 15.5,
            "hvac_power_kw_default": 16.0,
            "rotational_inertia_factor": 2.5,
            "rated_range_km_reference": 3001.0,
        }
        for name, value in bad_values.items():
            with self.assertRaises(ValueError, msg=f"{name} accepted {value}") as ctx:
                config(**{name: value})
            self.assertIn(name, str(ctx.exception))

    def test_envelope_extremes_are_accepted(self):
        extremes = {
            "nominal_pack_energy_kwh": 250.0,
            "usable_energy_fraction": 1.0,
            "vehicle_mass_kg": 20000.0,
            "drag_coefficient_cd": 2.0,
            "frontal_area_m2": 20.0,
            "rolling_resistance_coefficient_crr": 0.05,
            "air_density_kg_m3": 2.0,
            "drivetrain_efficiency": 1.0,
            "regen_efficiency": 1.0,
            "max_regen_power_kw": 5000.0,
            "auxiliary_power_kw": 15.0,
            "hvac_power_kw_default": 15.0,
            "rotational_inertia_factor": 2.0,
            "rated_range_km_reference": 3000.0,
        }
        for name, value in extremes.items():
            cfg = config(**{name: value})
            self.assertEqual(getattr(cfg, name), value, name)
        self.assertEqual(config(rolling_resistance_coefficient_crr=0.0).rolling_resistance_coefficient_crr, 0.0)
        self.assertEqual(config(regen_efficiency=0.0).regen_efficiency, 0.0)

    def test_zero_pack_or_mass_is_not_treated_as_absent(self):
        for name in ("nominal_pack_energy_kwh", "vehicle_mass_kg", "frontal_area_m2",
                     "drag_coefficient_cd", "air_density_kg_m3"):
            with self.assertRaises(ValueError, msg=f"{name} accepted 0"):
                config(**{name: 0})

    # --- provenance metadata structure ---
    def test_provenance_must_be_a_mapping(self):
        for bad in ("notes", 7, ["a"], True):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                config(provenance=bad)

    def test_provenance_status_required_to_be_a_non_empty_string(self):
        with self.assertRaises(ValueError):
            config(provenance={"status": 1})
        with self.assertRaises(ValueError):
            config(provenance={"status": "   "})
        self.assertEqual(config(provenance={"status": "PARTIALLY_SOURCED_UNCALIBRATED"}).provenance["status"],
                         "PARTIALLY_SOURCED_UNCALIBRATED")

    def test_provenance_optional_text_keys_must_be_strings(self):
        for key in ("note", "manufacturer_source", "retrieved_date"):
            with self.assertRaises(ValueError, msg=f"{key} accepted 1"):
                config(provenance={"status": "TEST", key: 1})

    def test_provenance_sourced_fields_must_reference_real_config_fields(self):
        with self.assertRaises(ValueError) as ctx:
            config(provenance={"status": "TEST", "sourced_fields": {"bogus_field": "a source"}})
        self.assertIn("unknown config field", str(ctx.exception))
        with self.assertRaises(ValueError):
            config(provenance={"status": "TEST", "sourced_fields": ["nominal_pack_energy_kwh"]})
        with self.assertRaises(ValueError):
            config(provenance={"status": "TEST", "sourced_fields": {"vehicle_mass_kg": 2100}})
        cfg = config(provenance={
            "status": "PARTIALLY_SOURCED_UNCALIBRATED",
            "sourced_fields": {"nominal_pack_energy_kwh": "manufacturer brochure"},
        })
        self.assertIn("nominal_pack_energy_kwh", cfg.provenance["sourced_fields"])

    def test_provenance_assumptions_must_be_a_list_of_strings(self):
        with self.assertRaises(ValueError):
            config(provenance={"status": "TEST", "assumptions": "one long string"})
        with self.assertRaises(ValueError):
            config(provenance={"status": "TEST", "assumptions": [1, 2]})
        cfg = config(provenance={"status": "TEST", "assumptions": ["mass is a placeholder"]})
        self.assertEqual(len(cfg.provenance["assumptions"]), 1)

    def test_provenance_none_is_allowed(self):
        self.assertIsNone(config(provenance=None).provenance)

    # --- identity fields ---
    def test_vehicle_name_rules(self):
        for bad in ("", "   ", 12, None, True):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                config(vehicle_name=bad)
        self.assertEqual(config(vehicle_name="  Real Name ").vehicle_name, "  Real Name ")


class EnvironmentalModelTests(unittest.TestCase):
    """Phase 2 §4: rho = P / (R_air * T) with station pressure and ISA altitude."""

    def setUp(self):
        self.calc = RangeCalculator(config())

    def ideal_gas(self, pressure_pa, temperature_k):
        return pressure_pa / (RangeCalculator.R_SPEC_AIR * temperature_k)

    def test_legacy_behaviour_is_preserved_when_environment_inputs_are_omitted(self):
        cfg = config(air_density_kg_m3=1.1)
        calc = RangeCalculator(cfg)
        self.assertEqual(calc.air_density(None), 1.1)
        self.assertEqual(
            calc.air_density(15.0),
            self.ideal_gas(RangeCalculator.P_ATM_PA, 288.15),
        )
        self.assertEqual(
            calc.resolve_environment()["air_density_source"], "CONFIG_REFERENCE"
        )
        self.assertEqual(
            calc.resolve_environment(ambient_c=5.0)["air_density_source"], "IDEAL_GAS_LAW"
        )

    def test_ideal_gas_identity_across_temperature_pressure_altitude(self):
        for ambient_c in (-30.0, 0.0, 15.0, 40.0):
            for altitude_m in (-400.0, 0.0, 800.0, 2100.0, 7000.0, 10999.0):
                for pressure in (None, 85000.0, 101325.0, 107000.0):
                    env = self.calc.resolve_environment(
                        ambient_c, pressure_pa=pressure, altitude_m=altitude_m)
                    expected_p = (
                        pressure if pressure is not None
                        else self.calc.atmospheric_pressure_pa(altitude_m)
                    )
                    self.assertAlmostEqual(
                        env["air_density_kg_m3"],
                        self.ideal_gas(expected_p, ambient_c + 273.15),
                        places=12,
                    )

    def test_isa_pressure_anchors_and_monotonicity(self):
        calc = RangeCalculator
        self.assertAlmostEqual(calc.atmospheric_pressure_pa(0.0), calc.P_ATM_PA, places=6)
        # Published ISA / standard-atmosphere table values.
        self.assertAlmostEqual(calc.atmospheric_pressure_pa(1000.0), 89874.7, delta=2.0)
        self.assertAlmostEqual(calc.atmospheric_pressure_pa(2000.0), 79495.1, delta=2.0)
        self.assertAlmostEqual(calc.atmospheric_pressure_pa(5000.0), 54019.9, delta=2.0)
        self.assertAlmostEqual(calc.atmospheric_pressure_pa(11000.0), 22632.0, delta=2.0)
        previous = None
        for altitude in range(-500, 11001, 500):
            pressure = calc.atmospheric_pressure_pa(float(altitude))
            if previous is not None:
                self.assertLess(pressure, previous)
            previous = pressure

    def test_isa_temperature_lapse(self):
        self.assertAlmostEqual(self.calc.isa_temperature_k(0.0), 288.15, places=9)
        self.assertAlmostEqual(self.calc.isa_temperature_k(11000.0), 216.65, places=9)
        self.assertAlmostEqual(self.calc.isa_temperature_k(-500.0), 291.4, places=9)

    def test_density_altitude_effect_matches_standard_atmosphere(self):
        # rho(2000 m, ISA) is 1.0065 kg/m3 in standard-atmosphere tables.
        self.assertAlmostEqual(
            self.calc.air_density(None, altitude_m=2000.0), 1.0065, delta=0.001)
        self.assertLess(
            self.calc.air_density(None, altitude_m=3500.0),
            self.calc.air_density(None, altitude_m=350.0),
        )

    def test_explicit_pressure_takes_precedence_over_altitude(self):
        env = self.calc.resolve_environment(
            ambient_c=15.0, pressure_pa=90000.0, altitude_m=3000.0)
        self.assertEqual(env["atmospheric_pressure_pa"], 90000.0)
        self.assertEqual(env["pressure_source"], "PRESSURE_SUPPLIED")
        self.assertAlmostEqual(
            env["air_density_kg_m3"], self.ideal_gas(90000.0, 288.15), places=12)

    def test_altitude_is_applied_once_to_pressure_not_to_a_supplied_ambient(self):
        """A measured ambient is ground temperature: no ISA lapse is added on top."""
        env = self.calc.resolve_environment(ambient_c=20.0, altitude_m=2000.0)
        self.assertEqual(env["temperature_k"], 293.15)
        self.assertEqual(env["temperature_source"], "AMBIENT_SUPPLIED")
        self.assertAlmostEqual(
            env["air_density_kg_m3"],
            self.ideal_gas(self.calc.atmospheric_pressure_pa(2000.0), 293.15),
            places=12,
        )

    def test_temperature_falls_back_to_isa_only_when_ambient_is_omitted(self):
        env = self.calc.resolve_environment(altitude_m=2000.0)
        self.assertEqual(env["temperature_source"], "ISA_AT_ALTITUDE")
        self.assertAlmostEqual(env["temperature_k"], 275.15, places=9)
        env_sea = self.calc.resolve_environment(pressure_pa=100000.0)
        self.assertEqual(env_sea["temperature_source"], "ISA_STANDARD")
        self.assertAlmostEqual(env_sea["temperature_k"], 288.15, places=9)

    def test_altitude_bounds_rejected(self):
        for bad in (11001.0, -501.0, 100000.0, float("nan"), float("inf"), "5000"):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                self.calc.resolve_environment(ambient_c=20.0, altitude_m=bad)
        self.calc.resolve_environment(ambient_c=20.0, altitude_m=-500.0)
        self.calc.resolve_environment(ambient_c=20.0, altitude_m=11000.0)

    def test_pressure_bounds_rejected(self):
        for bad in (19999.0, 120001.0, 0.0, -101325.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}"):
                self.calc.resolve_environment(ambient_c=20.0, pressure_pa=bad)
        self.calc.resolve_environment(ambient_c=20.0, pressure_pa=20000.0)
        self.calc.resolve_environment(ambient_c=20.0, pressure_pa=120000.0)

    def test_absolute_zero_and_extreme_ambient_rejected(self):
        with self.assertRaises(ValueError):
            self.calc.resolve_environment(ambient_c=-300.0)
        with self.assertRaises(ValueError):
            RangeCalculator(config()).estimate(cycle(), ambient_c=-274.0)

    def test_altitude_reduces_consumption_by_lowering_air_density(self):
        high_speed = [
            {"time_s": 0.0, "speed_mps": 30.0, "grade_pct": 0.0},
            {"time_s": 600.0, "speed_mps": 30.0, "grade_pct": 0.0},
        ]
        calc = RangeCalculator(config(
            auxiliary_power_kw=0.0, hvac_power_kw_default=0.0))
        sea = calc.estimate(high_speed, ambient_c=15.0, altitude_m=0.0)
        mountain = calc.estimate(high_speed, ambient_c=15.0, altitude_m=2200.0)
        self.assertLess(mountain["air_density_kg_m3"], sea["air_density_kg_m3"])
        self.assertLess(mountain["consumption_kwh_per_100km"], sea["consumption_kwh_per_100km"])
        self.assertGreater(mountain["estimated_range_km"], sea["estimated_range_km"])

    def test_colder_air_is_denser_and_costs_more_at_speed(self):
        high_speed = [
            {"time_s": 0.0, "speed_mps": 28.0, "grade_pct": 0.0},
            {"time_s": 600.0, "speed_mps": 28.0, "grade_pct": 0.0},
        ]
        calc = RangeCalculator(config(
            auxiliary_power_kw=0.0, hvac_power_kw_default=0.0))
        mild = calc.estimate(high_speed, ambient_c=25.0, hvac_power_kw=0.0)
        cold = calc.estimate(high_speed, ambient_c=-25.0, hvac_power_kw=0.0)
        self.assertGreater(cold["air_density_kg_m3"], mild["air_density_kg_m3"])
        self.assertGreater(cold["consumption_kwh_per_100km"], mild["consumption_kwh_per_100km"])

    def test_environment_is_reported_in_the_result(self):
        result = RangeCalculator(config()).estimate(
            cycle(12.0), ambient_c=10.0, altitude_m=1500.0, pressure_pa=98000.0)
        self.assertEqual(result["atmospheric_pressure_pa"], 98000.0)
        self.assertEqual(result["altitude_m"], 1500.0)
        self.assertEqual(result["air_density_source"], "IDEAL_GAS_LAW")
        self.assertAlmostEqual(
            result["air_density_kg_m3"],
            self.ideal_gas(98000.0, 283.15),
            places=12,
        )
        self.assertTrue(any("rho = P/(R_air*T)" in text for text in result["limitations"]))

    def test_estimate_trajectory_forwards_environmental_inputs(self):
        calc = RangeCalculator(config())
        results = calc.estimate_trajectory(
            cycle(25.0),
            [100.0, 90.0],
            ambient_c=15.0,
            altitude_m=1600.0,
            headwind_mps=5.0,
        )
        self.assertEqual(len(results), 2)
        for point in results:
            self.assertEqual(point["altitude_m"], 1600.0)
            self.assertEqual(point["headwind_mps"], 5.0)
        self.assertGreater(results[0]["estimated_range_km"], results[1]["estimated_range_km"])


class RelativeWindTests(unittest.TestCase):
    """Phase 2 §5: along-track relative air speed, with the sign convention made explicit."""

    def cfg_ideal(self, **overrides):
        base = dict(
            drivetrain_efficiency=1.0,
            regen_efficiency=1.0,
            max_regen_power_kw=5000.0,
            rolling_resistance_coefficient_crr=0.0,
            drag_coefficient_cd=0.6,
            frontal_area_m2=2.5,
            auxiliary_power_kw=0.0,
            hvac_power_kw_default=0.0,
            rotational_inertia_factor=1.0,
            vehicle_mass_kg=2000.0,
        )
        base.update(overrides)
        return config(**base)

    def coast(self, v0, v1, dt, headwind=None, grade=0.0, cfg=None):
        calc = RangeCalculator(cfg or self.cfg_ideal())
        return calc.estimate(
            [
                {"time_s": 0.0, "speed_mps": v0, "grade_pct": grade},
                {"time_s": dt, "speed_mps": v1, "grade_pct": grade},
            ],
            soh_pct=100.0,
            ambient_c=15.0,
            hvac_power_kw=0.0,
            headwind_mps=headwind,
        )

    def aero_power(self, calc, v, headwind):
        """Closed-form aerodynamic wheel power k*u*|u|*v at ground speed v."""
        rho = calc.air_density(15.0)
        k = 0.5 * rho * calc.c.drag_coefficient_cd * calc.c.frontal_area_m2
        u = v + headwind
        return k * u * abs(u) * v

    def reference_interval(self, cfg, v0, v1, dt, headwind, grade_pct=0.0, steps=30000):
        """Midpoint integration of the documented model, independent of the engine."""
        calc = RangeCalculator(cfg)
        rho = calc.air_density(15.0)
        k = 0.5 * rho * cfg.drag_coefficient_cd * cfg.frontal_area_m2
        m = cfg.vehicle_mass_kg
        m_eff = m * cfg.rotational_inertia_factor
        theta = math.atan(grade_pct / 100.0)
        f_resist = m * RangeCalculator.G * (
            cfg.rolling_resistance_coefficient_crr * math.cos(theta) + math.sin(theta))
        accel = (v1 - v0) / dt
        eta = cfg.regen_efficiency
        p_max = max(0.0, cfg.max_regen_power_kw) * 1000.0
        step = dt / steps
        traction = braking = regen_mech = regen_elec = 0.0
        for i in range(steps):
            v = v0 + accel * (i + 0.5) * step
            u = v + headwind
            power = (m_eff * accel + f_resist) * v + k * u * abs(u) * v
            if power >= 0.0:
                traction += power * step
                continue
            braking += -power * step
            if v <= 0.5:
                beta = 0.0
            elif v >= 2.0:
                beta = 1.0
            else:
                beta = (v - 0.5) / 1.5
            electric = min(eta * beta * -power, p_max)
            regen_elec += electric * step
            regen_mech += electric * step / eta
        return {
            "traction_j": traction,
            "braking_j": braking,
            "regen_mech_j": regen_mech,
            "regen_elec_j": regen_elec,
        }

    def engine_interval(self, calc, v0, v1, dt, headwind, grade_pct=0.0):
        cfg = calc.c
        result = calc.estimate(
            [
                {"time_s": 0.0, "speed_mps": v0, "grade_pct": grade_pct},
                {"time_s": dt, "speed_mps": v1, "grade_pct": grade_pct},
            ],
            soh_pct=100.0,
            ambient_c=15.0,
            hvac_power_kw=0.0,
            headwind_mps=headwind,
        )
        return result, {
            "traction_j": result["traction_energy_kwh"] * 3.6e6 * cfg.drivetrain_efficiency,
            "braking_j": result["mechanical_braking_kwh"] * 3.6e6,
            "regen_mech_j": result["regen_mechanical_absorbed_kwh"] * 3.6e6,
            "regen_elec_j": result["regen_recovered_kwh"] * 3.6e6,
        }

    # --- convention and defaults ---
    def test_no_wind_is_identical_to_zero_wind(self):
        calc = RangeCalculator(config())
        sample = cycle(18.0)
        baseline = calc.estimate(sample, ambient_c=15.0)
        for wind in (None, 0.0):
            self.assertEqual(calc.estimate(sample, ambient_c=15.0, headwind_mps=wind), baseline)

    def test_sign_convention_positive_is_headwind(self):
        """headwind_mps > 0 raises relative air speed and cost; < 0 is a tailwind."""
        steady = [
            {"time_s": 0.0, "speed_mps": 25.0, "grade_pct": 0.0},
            {"time_s": 300.0, "speed_mps": 25.0, "grade_pct": 0.0},
        ]
        calc = RangeCalculator(config(
            drivetrain_efficiency=1.0, auxiliary_power_kw=0.0, hvac_power_kw_default=0.0))
        calm = calc.estimate(steady, ambient_c=15.0, hvac_power_kw=0.0)
        head = calc.estimate(steady, ambient_c=15.0, hvac_power_kw=0.0, headwind_mps=10.0)
        tail = calc.estimate(steady, ambient_c=15.0, hvac_power_kw=0.0, headwind_mps=-10.0)
        self.assertGreater(head["consumption_kwh_per_100km"], calm["consumption_kwh_per_100km"])
        self.assertLess(tail["consumption_kwh_per_100km"], calm["consumption_kwh_per_100km"])
        self.assertAlmostEqual(
            head["traction_energy_kwh"],
            (calm["traction_energy_kwh"] * 3.6e6
             + (self.aero_power(calc, 25.0, 10.0) - self.aero_power(calc, 25.0, 0.0)) * 300.0)
            / 3.6e6,
            places=9,
        )

    def test_constant_speed_headwind_matches_closed_form(self):
        calc = RangeCalculator(self.cfg_ideal())
        v, dt, wind = 22.0, 120.0, 8.0
        _, got = self.engine_interval(calc, v, v, dt, wind)
        expected = self.aero_power(calc, v, wind) * dt
        self.assertGreater(expected, 0.0)
        self.assertAlmostEqual(got["traction_j"], expected, places=6)
        self.assertEqual(got["braking_j"], 0.0)

    def test_tailwind_faster_than_vehicle_reverses_the_aero_force(self):
        calc = RangeCalculator(self.cfg_ideal())
        v, dt, wind = 8.0, 60.0, -20.0
        self.assertLess(self.aero_power(calc, v, wind), 0.0)
        _, got = self.engine_interval(calc, v, v, dt, wind)
        self.assertEqual(got["traction_j"], 0.0)
        self.assertAlmostEqual(
            got["braking_j"], -self.aero_power(calc, v, wind) * dt, places=6)
        self.assertAlmostEqual(got["regen_elec_j"], got["braking_j"], places=6)

    def test_coast_deceleration_with_wind_matches_reference_integration(self):
        cfg = self.cfg_ideal(regen_efficiency=0.7, max_regen_power_kw=25.0)
        calc = RangeCalculator(cfg)
        for wind in (0.0, 12.0, -8.0, -20.0, -27.0):
            _, got = self.engine_interval(calc, 30.0, 3.0, 60.0, wind)
            ref = self.reference_interval(cfg, 30.0, 3.0, 60.0, wind)
            for key in ref:
                self.assertAlmostEqual(
                    got[key], ref[key], delta=abs(ref[key]) * 5e-4 + 1.0, msg=f"{key} at wind {wind}")

    def test_grade_and_wind_together_match_reference_integration(self):
        cfg = self.cfg_ideal(
            regen_efficiency=0.7,
            max_regen_power_kw=25.0,
            rolling_resistance_coefficient_crr=0.01,
            rotational_inertia_factor=1.05,
            drivetrain_efficiency=0.9,
        )
        calc = RangeCalculator(cfg)
        for (v0, v1, dt, grade, wind) in (
            (28.0, 6.0, 45.0, -4.0, -12.0),
            (28.0, 6.0, 45.0, 4.0, 15.0),
            (4.0, 26.0, 40.0, 6.0, -18.0),
            (2.0, 0.0, 25.0, -8.0, 3.0),
        ):
            result, got = self.engine_interval(calc, v0, v1, dt, wind, grade_pct=grade)
            ref = self.reference_interval(cfg, v0, v1, dt, wind, grade_pct=grade)
            for key in ref:
                self.assertAlmostEqual(
                    got[key], ref[key], delta=abs(ref[key]) * 5e-4 + 1.0,
                    msg=f"{key} on grade {grade} wind {wind}")
            self.assertLessEqual(result["regen_mechanical_absorbed_kwh"],
                                 result["mechanical_braking_kwh"] + 1e-12)
            self.assertLessEqual(result["friction_braking_kwh"],
                                 result["mechanical_braking_kwh"] + 1e-12)
            self.assertAlmostEqual(
                result["regen_recovered_kwh"],
                cfg.regen_efficiency * result["regen_mechanical_absorbed_kwh"],
                places=12,
            )

    def test_relative_air_speed_sign_change_splits_the_interval(self):
        """Crossing v = |tailwind| flips drag to thrust; the split keeps the integral exact."""
        cfg = self.cfg_ideal(regen_efficiency=0.8, max_regen_power_kw=5000.0)
        calc = RangeCalculator(cfg)
        wind = -15.0
        _, got = self.engine_interval(calc, 25.0, 5.0, 40.0, wind)
        ref = self.reference_interval(cfg, 25.0, 5.0, 40.0, wind)
        self.assertGreater(ref["braking_j"], 0.0)
        for key in ref:
            self.assertAlmostEqual(got[key], ref[key], delta=abs(ref[key]) * 5e-4 + 1.0)

    def test_wind_energy_ledger_never_double_counts(self):
        cfg = self.cfg_ideal(regen_efficiency=0.6, max_regen_power_kw=20.0)
        calc = RangeCalculator(cfg)
        result = self.coast(30.0, 0.0, 90.0, headwind=-10.0, cfg=cfg)
        self.assertLessEqual(result["regen_recovered_kwh"], result["mechanical_braking_kwh"])
        self.assertAlmostEqual(
            result["mechanical_braking_kwh"],
            result["regen_mechanical_absorbed_kwh"] + result["friction_braking_kwh"],
            places=12,
        )

    def test_crosswind_is_rejected_not_silently_flattened(self):
        calc = RangeCalculator(config())
        for bad in (5.0, -5.0, float("nan"), "5"):
            with self.assertRaises(ValueError, msg=f"accepted {bad!r}") as ctx:
                calc.estimate(cycle(15.0), ambient_c=15.0, crosswind_mps=bad)
            self.assertIn("crosswind", str(ctx.exception))

    def test_zero_crosswind_is_accepted_and_matches_no_crosswind(self):
        calc = RangeCalculator(config())
        baseline = calc.estimate(cycle(15.0), ambient_c=15.0)
        explicit = calc.estimate(cycle(15.0), ambient_c=15.0, crosswind_mps=0.0)
        self.assertEqual(baseline, explicit)

    def test_wind_magnitude_bounds_rejected(self):
        calc = RangeCalculator(config())
        for bad in (60.1, -60.1, 500.0):
            with self.assertRaises(ValueError, msg=f"accepted {bad}"):
                calc.estimate(cycle(15.0), ambient_c=15.0, headwind_mps=bad)
        calc.estimate(cycle(15.0), ambient_c=15.0, headwind_mps=60.0)
        calc.estimate(cycle(15.0), ambient_c=15.0, headwind_mps=-60.0)

    def test_wind_does_not_change_distance_or_duration(self):
        result_calm = self.coast(28.0, 4.0, 60.0)
        result_wind = self.coast(28.0, 4.0, 60.0, headwind=-14.0)
        self.assertEqual(result_calm["distance_km"], result_wind["distance_km"])
        self.assertEqual(result_calm["duration_s"], result_wind["duration_s"])
        self.assertEqual(result_wind["headwind_mps"], -14.0)

    def test_wind_limitation_is_reported_only_when_wind_is_used(self):
        calc = RangeCalculator(config())
        calm = calc.estimate(cycle(15.0), ambient_c=15.0)
        windy = calc.estimate(cycle(15.0), ambient_c=15.0, headwind_mps=6.0)
        self.assertFalse(any("along-track wind" in text for text in calm["limitations"]))
        self.assertTrue(any("along-track wind" in text for text in windy["limitations"]))


# ===========================================================================
# Phase 3: HVAC operating modes, cabin thermal model and energy accounting
# ===========================================================================

CABIN_VEHICLE = {
    # A small cabin (C=100 kJ/K) with heavy conductive coupling (UA=1 kW/K) so a
    # 600 s cycle spans several time constants and the thermal evolution is
    # observable. Values are chosen for hand-checkable arithmetic, not realism.
    "nominal_pack_energy_kwh": 40.0,
    "usable_energy_fraction": 0.9,
    "vehicle_mass_kg": 1500.0,
    "drag_coefficient_cd": 0.3,
    "frontal_area_m2": 2.2,
    "rolling_resistance_coefficient_crr": 0.01,
    "air_density_kg_m3": 1.225,
    "drivetrain_efficiency": 0.9,
    "regen_efficiency": 0.5,
    "max_regen_power_kw": 30.0,
    "auxiliary_power_kw": 0.4,
    "hvac_power_kw_default": 0.0,  # no blower base: HVAC energy is pure actuator
    "rotational_inertia_factor": 1.05,
    "provenance": {"status": "TEST"},
    "cabin_thermal_capacitance_j_per_k": 100_000.0,
    "cabin_ua_w_per_k": 1000.0,
    "cabin_target_temp_c": 20.0,
    "heat_pump_cop_reference": 4.0,
    "heat_pump_cop_reference_temp_c": 20.0,
    "heat_pump_cop_slope_per_k": 0.0,
    "heat_pump_cop_min": 4.0,
    "heat_pump_cop_max": 4.0,
    "ac_cooling_cop_reference": 5.0,
    "ac_cooling_cop_reference_temp_c": 35.0,
    "ac_cooling_cop_slope_per_k": 0.0,
    "ac_cooling_cop_min": 5.0,
    "ac_cooling_cop_max": 5.0,
    "ptc_heating_efficiency": 0.8,
    "max_heating_electrical_kw": 15.0,
    "max_cooling_electrical_kw": 15.0,
}

# Legacy result keys that existed before Phase 3; every new key must be additive.
PRE_PHASE3_RESULT_KEYS = {
    "vehicle_name", "distance_km", "duration_s", "average_speed_kmh", "cycle_energy_kwh",
    "traction_energy_kwh", "regen_recovered_kwh", "aux_hvac_energy_kwh",
    "consumption_kwh_per_km", "consumption_kwh_per_100km", "mechanical_braking_kwh",
    "regen_mechanical_absorbed_kwh", "friction_braking_kwh", "usable_energy_kwh",
    "soh_pct", "ambient_c", "air_density_kg_m3", "atmospheric_pressure_pa", "altitude_m",
    "air_density_source", "headwind_mps", "hvac_power_kw", "hvac_mode",
    "estimated_range_km", "quality_status", "quality_flag", "cell_id", "limitations",
}

LEGACY_HVAC_LABELS = {
    "MANUAL_OVERRIDE", "CONFIG_DEFAULT", "COMFORT_DEADBAND_VENTILATION",
    "EMPIRICAL_HEATING_DEMAND", "EMPIRICAL_COOLING_DEMAND",
}


def cabin_config(mode="HEAT_PUMP", **overrides):
    raw = dict(CABIN_VEHICLE, hvac_model_mode=mode)
    raw.update(overrides)
    return config(**raw)


def flat_cycle(total_s=600.0, intervals=10, speed_mps=12.0):
    """Constant-speed trace: traction work is fixed, so only HVAC can differ."""
    dt = total_s / intervals
    return [
        {"time_s": i * dt, "speed_mps": speed_mps, "grade_pct": 0.0}
        for i in range(intervals + 1)
    ]


class HvacModeSelectionTests(unittest.TestCase):
    """Phase 3 Parts 3 and 8: mode selection, defaults, precedence and validation."""

    def profile_paths(self):
        return sorted((ROOT / "vehicle_configs").glob("*.json")) + [ROOT / "vehicle_config.json"]

    def test_shipped_profiles_load_without_any_cabin_parameters(self):
        for path in self.profile_paths():
            cfg = VehicleConfig.from_json(path)
            self.assertEqual(cfg.hvac_model_mode, "EMPIRICAL", path.name)
            self.assertIsNone(cfg.cabin_initial_temp_c, path.name)
            self.assertIsNone(cfg.cabin_recovery_time_constant_s, path.name)

    def test_shipped_profiles_keep_their_legacy_hvac_labels_and_values(self):
        for path in self.profile_paths():
            calc = RangeCalculator(VehicleConfig.from_json(path))
            for ambient in (None, -10.0, 20.0, 40.0):
                result = calc.estimate(flat_cycle(), ambient_c=ambient)
                self.assertIn(result["hvac_mode"], LEGACY_HVAC_LABELS, (path.name, ambient))
                self.assertEqual(result["hvac_model_mode"], "EMPIRICAL")
                self.assertIsNone(result["cabin_temp_end_c"])
                self.assertIsNone(result["hvac_heating_thermal_energy_kwh"])

    def test_json_with_only_legacy_fields_is_valid_and_uses_documented_defaults(self):
        legacy_only = dict(CABIN_VEHICLE, vehicle_name="cabin_test_vehicle")
        for key in (
            "cabin_thermal_capacitance_j_per_k", "cabin_ua_w_per_k", "cabin_target_temp_c",
            "heat_pump_cop_reference", "heat_pump_cop_reference_temp_c",
            "heat_pump_cop_slope_per_k", "heat_pump_cop_min", "heat_pump_cop_max",
            "ac_cooling_cop_reference", "ac_cooling_cop_reference_temp_c",
            "ac_cooling_cop_slope_per_k", "ac_cooling_cop_min", "ac_cooling_cop_max",
            "ptc_heating_efficiency", "max_heating_electrical_kw",
            "max_cooling_electrical_kw", "hvac_model_mode",
        ):
            legacy_only.pop(key, None)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(legacy_only, handle)
            path = handle.name
        try:
            cfg = VehicleConfig.from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(cfg.hvac_model_mode, "EMPIRICAL")
        self.assertEqual(cfg.cabin_thermal_capacitance_j_per_k, 1_000_000.0)
        self.assertEqual(cfg.cabin_ua_w_per_k, 150.0)
        self.assertEqual(cfg.cabin_target_temp_c, 22.0)
        self.assertEqual(cfg.ptc_heating_efficiency, 0.95)
        self.assertEqual(cfg.max_heating_electrical_kw, 8.0)
        self.assertEqual(cfg.max_cooling_electrical_kw, 6.0)

    def test_unknown_new_field_is_rejected_rather_than_ignored(self):
        raw = dict(CABIN_VEHICLE, cabin_thermal_capcitance_j_per_k=1.0)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(raw, handle)
            path = handle.name
        try:
            with self.assertRaises(ValueError) as ctx:
                VehicleConfig.from_json(path)
            self.assertIn("Unknown field", str(ctx.exception))
        finally:
            Path(path).unlink(missing_ok=True)

    def test_omitting_the_new_arguments_reproduces_the_empirical_result_exactly(self):
        calc = RangeCalculator(cabin_config(mode="EMPIRICAL"))
        base = calc.estimate(flat_cycle(), soh_pct=95.0, ambient_c=30.0)
        explicit = calc.estimate(
            flat_cycle(), soh_pct=95.0, ambient_c=30.0,
            hvac_model_mode="EMPIRICAL", cabin_target_temp_c=20.0,
            cabin_initial_temp_c=5.0, cabin_solar_load_w=900.0,
            cabin_occupancy_load_w=300.0, heat_pump_cop=2.0, ac_cooling_cop=2.0,
            max_heating_electrical_kw=1.0, max_cooling_electrical_kw=1.0,
        )
        self.assertEqual(base, explicit)

    def test_legacy_positional_call_is_unchanged(self):
        calc = RangeCalculator(config())
        positional = calc.estimate(cycle(12.0), 100.0, 5.0, 2.0, "FLAG", "CELL")
        keyword = calc.estimate(
            cycle(12.0), soh_pct=100.0, ambient_c=5.0, hvac_power_kw=2.0,
            quality_flag="FLAG", cell_id="CELL",
        )
        self.assertEqual(positional, keyword)
        self.assertEqual(positional["hvac_mode"], "MANUAL_OVERRIDE")

    def test_new_keys_are_additive_and_never_replace_a_legacy_key(self):
        calc = RangeCalculator(cabin_config())
        result = calc.estimate(flat_cycle(), ambient_c=-5.0)
        self.assertTrue(PRE_PHASE3_RESULT_KEYS.issubset(set(result)), PRE_PHASE3_RESULT_KEYS - set(result))

    def test_manual_electrical_override_short_circuits_the_cabin_model(self):
        for mode in ("EMPIRICAL", "HEAT_PUMP", "PTC", "AC_COOLING"):
            calc = RangeCalculator(cabin_config(mode=mode))
            result = calc.estimate(flat_cycle(600.0), ambient_c=-10.0, hvac_power_kw=3.0)
            self.assertEqual(result["hvac_mode"], "MANUAL_OVERRIDE", mode)
            self.assertEqual(result["hvac_model_mode"], mode)
            # The override is electrical kW, never reinterpreted as a thermal load:
            # 3.0 kW for 600 s is exactly 0.5 kWh, with no COP applied.
            self.assertAlmostEqual(result["hvac_energy_kwh"], 0.5, places=15)
            self.assertAlmostEqual(result["aux_hvac_energy_kwh"], 0.5 + 0.4 / 6.0, places=15)
            self.assertIsNone(result["cabin_temp_end_c"], mode)
            self.assertIsNone(result["hvac_heating_thermal_energy_kwh"], mode)
            # The bypass is stated out loud, so a reader cannot mistake the recorded
            # request for the physics that produced the number.
            notes = " ".join(result["limitations"])
            if mode == "EMPIRICAL":
                self.assertNotIn("short-circuited the requested", notes)
            else:
                self.assertIn(f"short-circuited the requested {mode} cabin model", notes)

    def test_cabin_mode_without_ambient_is_rejected_with_an_actionable_message(self):
        calc = RangeCalculator(cabin_config())
        for mode in ("HEAT_PUMP", "PTC", "AC_COOLING"):
            with self.assertRaises(ValueError, msg=mode) as ctx:
                calc.estimate(flat_cycle(), ambient_c=None, hvac_model_mode=mode)
            self.assertIn("ambient_c", str(ctx.exception))

    def test_unsupported_mode_is_rejected_at_config_and_call_sites(self):
        # A configuration must name an implemented mode; there is no HYBRID or AUTO.
        for bad in ("AUTO", "HYBRID", "heat_pump", "", None, 5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                cabin_config(mode=bad)
        # At the call site None means "not specified", so only real values are rejected.
        for bad in ("AUTO", "HYBRID", "heat_pump", "", 5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                RangeCalculator(cabin_config(mode="EMPIRICAL")).estimate(
                    flat_cycle(), ambient_c=5.0, hvac_model_mode=bad
                )
        # Omitting the argument keeps the configured mode.
        omitted = RangeCalculator(cabin_config(mode="EMPIRICAL")).estimate(
            flat_cycle(), ambient_c=5.0, hvac_model_mode=None
        )
        self.assertEqual(omitted["hvac_model_mode"], "EMPIRICAL")
        self.assertEqual(omitted["hvac_mode"], "EMPIRICAL_HEATING_DEMAND")

    def test_derived_cop_curves_match_hand_computed_values_and_clip_bands(self):
        # Dataclass defaults: heating 3.0 at 7 C degrading 0.05/K into [1.2, 5.0],
        # cooling 3.5 at 35 C degrading 0.05/K into [1.5, 6.0].
        calc = RangeCalculator(config())
        self.assertAlmostEqual(calc.heating_cop(-10.0), 3.0 - 0.05 * 17.0, places=12)
        self.assertAlmostEqual(calc.heating_cop(25.0), 3.0 + 0.05 * 18.0, places=12)
        self.assertEqual(calc.heating_cop(-50.0), 1.2)  # clipped at the configured floor
        self.assertEqual(calc.heating_cop(30.0), 4.15)
        self.assertAlmostEqual(calc.cooling_cop(45.0), 3.0, places=12)
        self.assertAlmostEqual(calc.cooling_cop(5.0), 5.0, places=12)
        self.assertEqual(calc.cooling_cop(-50.0), 6.0)  # clipped at the configured ceiling

    def test_explicit_constant_cop_overrides_the_derived_curve(self):
        calc = RangeCalculator(cabin_config())
        derived = calc.resolve_hvac_plan(-10.0, None)
        forced = calc.resolve_hvac_plan(-10.0, None, heat_pump_cop=2.5, ac_cooling_cop=1.5)
        self.assertEqual(derived["heating_cop"], calc.heating_cop(-10.0))
        self.assertEqual(forced["heating_cop"], 2.5)
        self.assertEqual(forced["cooling_cop"], 1.5)

    def test_cop_outside_the_admissible_band_is_rejected(self):
        calc = RangeCalculator(cabin_config())
        for bad in (0.5, 0.0, -1.0, 10.5, 100.0, math.nan, math.inf, True, "3"):
            for kw in ("heat_pump_cop", "ac_cooling_cop"):
                with self.assertRaises(ValueError, msg=f"{kw}={bad!r}"):
                    calc.resolve_hvac_plan(5.0, None, **{kw: bad})

    def test_scenario_overrides_are_validated_against_the_same_envelopes(self):
        calc = RangeCalculator(cabin_config())
        cases = {
            "cabin_target_temp_c": (50.0, -25.0),
            "cabin_solar_load_w": (3001.0, -1.0),
            "cabin_occupancy_load_w": (1501.0, -1.0),
            "max_heating_electrical_kw": (21.0, -1.0),
            "max_cooling_electrical_kw": (21.0, -1.0),
        }
        for name, bad_values in cases.items():
            for bad in bad_values:
                with self.assertRaises(ValueError, msg=f"{name}={bad}"):
                    calc.resolve_hvac_plan(5.0, None, **{name: bad})
        # The envelopes are inclusive, so the boundary values are accepted.
        calc.resolve_hvac_plan(
            5.0, None, cabin_target_temp_c=45.0, cabin_solar_load_w=3000.0,
            cabin_occupancy_load_w=1500.0, max_heating_electrical_kw=20.0,
            max_cooling_electrical_kw=20.0,
        )

    def test_cabin_soak_temperature_is_validated_at_both_entry_points(self):
        for bad in (85.0, -55.0, math.nan, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                cabin_config(**{"cabin_initial_temp_c": bad})
        calc = RangeCalculator(cabin_config())
        for bad in (85.0, -55.0):
            with self.assertRaises(ValueError, msg=repr(bad)):
                calc.estimate(flat_cycle(), ambient_c=5.0, cabin_initial_temp_c=bad)
        # 80 C is admissible: a sun-soaked cabin can exceed ambient temperature.
        self.assertEqual(
            calc.resolve_hvac_plan(40.0, None, cabin_initial_temp_c=80.0)["start_c"], 80.0
        )

    def test_inverted_and_unreachable_cop_bands_are_rejected(self):
        for kwargs in (
            {"heat_pump_cop_min": 5.0, "heat_pump_cop_max": 3.0},
            {"ac_cooling_cop_min": 6.0, "ac_cooling_cop_max": 2.0},
            {"heat_pump_cop_reference": 9.0},   # outside its own [4, 4] band
            {"ac_cooling_cop_reference": 1.0},  # outside its own [5, 5] band
            {"heat_pump_cop_slope_per_k": -0.1},
            {"ac_cooling_cop_slope_per_k": -0.1},
        ):
            with self.assertRaises(ValueError, msg=repr(kwargs)):
                cabin_config(**kwargs)

    def test_ptc_efficiency_must_be_positive_and_at_most_one(self):
        for bad in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError, msg=bad):
                cabin_config(ptc_heating_efficiency=bad)
        cabin_config(ptc_heating_efficiency=1.0)  # ideal resistance heater is admissible


class CabinThermalModelTests(unittest.TestCase):
    """Phase 3 Parts 4-7: the cabin energy balance, actuators and their stability."""

    def test_heat_pump_step_matches_closed_form_arithmetic(self):
        # UA=1000 W/K, C=100 kJ/K, target 20 C, ambient 15 C, cabin 15 C,
        # K_p = C/tau_recovery = UA:  Q_demand = 1000*5 + 1000*5 = 10 kW thermal.
        # COP 4 -> 2.5 kW electrical. Plant asymptote 15 + 10000/1000 = 25 C,
        # so T(60 s) = 25 - 10*exp(-0.6).
        calc = RangeCalculator(cabin_config())
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=15.0)
        step = calc.cabin_thermal_step(plan, 15.0, 60.0)
        self.assertAlmostEqual(step["q_demand_w"], 10_000.0, places=9)
        self.assertAlmostEqual(step["thermal_heating_w"], 10_000.0, places=9)
        self.assertEqual(step["thermal_cooling_w"], 0.0)
        self.assertAlmostEqual(step["electrical_kw"], 2.5, places=15)
        self.assertAlmostEqual(step["cabin_temp_end_c"], 25.0 - 10.0 * math.exp(-0.6), places=12)
        self.assertFalse(step["power_clipped"])
        self.assertFalse(step["setpoint_unmet"])

    def test_ptc_step_matches_closed_form_arithmetic(self):
        # Same 10 kW demand through an 0.8 efficient resistance heater: 12.5 kW
        # electrical, 10 kW thermal, identical plant trajectory to the heat pump.
        calc = RangeCalculator(cabin_config(mode="PTC"))
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=15.0)
        step = calc.cabin_thermal_step(plan, 15.0, 60.0)
        self.assertAlmostEqual(step["electrical_kw"], 12.5, places=15)
        self.assertAlmostEqual(step["thermal_heating_w"], 10_000.0, places=9)
        self.assertAlmostEqual(
            step["thermal_heating_w"], step["electrical_kw"] * 1000.0 * 0.8, places=9
        )
        self.assertAlmostEqual(step["cabin_temp_end_c"], 25.0 - 10.0 * math.exp(-0.6), places=12)

    def test_ac_cooling_step_matches_closed_form_arithmetic(self):
        # Ambient 35, target 25, cabin 35: Q_demand = 1000*(-10) + 1000*(-10) = -20 kW
        # -> COP 5 gives 4 kW electrical, asymptote 35 - 20 = 15 C.
        calc = RangeCalculator(cabin_config(mode="AC_COOLING"))
        plan = calc.resolve_hvac_plan(35.0, None, cabin_target_temp_c=25.0,
                                      cabin_initial_temp_c=35.0)
        step = calc.cabin_thermal_step(plan, 35.0, 60.0)
        self.assertAlmostEqual(step["q_demand_w"], -20_000.0, places=9)
        self.assertAlmostEqual(step["electrical_kw"], 4.0, places=15)
        self.assertEqual(step["thermal_heating_w"], 0.0)
        self.assertAlmostEqual(step["thermal_cooling_w"], 20_000.0, places=9)
        self.assertAlmostEqual(step["cabin_temp_end_c"], 15.0 + 20.0 * math.exp(-0.6), places=12)

    def test_cabin_moves_toward_the_target_and_holds_it_when_uncapped(self):
        calc = RangeCalculator(cabin_config())
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=5.0)
        temp = 5.0
        trajectory = []
        for _ in range(12):
            step = calc.cabin_thermal_step(plan, temp, 60.0)
            trajectory.append(step["cabin_temp_end_c"])
            temp = step["cabin_temp_end_c"]
        # The steady state of the closed loop is exactly the setpoint: the hold load
        # UA(T_target - T_amb) is what the actuator delivers at T = T_target.
        self.assertLess(trajectory[0], trajectory[-1])
        self.assertAlmostEqual(temp, 20.0, places=6)
        self.assertTrue(all(t < 20.0 + 1e-6 for t in trajectory), trajectory)

    def test_cabin_drifts_environmentally_when_no_compatible_actuator_exists(self):
        # AC_COOLING cannot heat, so with a heating demand the cabin relaxes toward
        # ambient with zero HVAC energy: T(t) = 30 + 15*exp(-UA t / C).
        calc = RangeCalculator(cabin_config(mode="AC_COOLING"))
        plan = calc.resolve_hvac_plan(30.0, None, cabin_target_temp_c=45.0,
                                      cabin_initial_temp_c=45.0)
        step = calc.cabin_thermal_step(plan, 45.0, 60.0)
        self.assertEqual(step["electrical_kw"], 0.0)
        self.assertEqual(step["thermal_heating_w"], 0.0)
        self.assertEqual(step["thermal_cooling_w"], 0.0)
        self.assertTrue(step["setpoint_unmet"])
        self.assertAlmostEqual(step["cabin_temp_end_c"], 30.0 + 15.0 * math.exp(-0.6), places=12)

    def test_heating_modes_never_deliver_cooling(self):
        for mode in ("HEAT_PUMP", "PTC"):
            calc = RangeCalculator(cabin_config(mode=mode))
            plan = calc.resolve_hvac_plan(35.0, None, cabin_target_temp_c=25.0,
                                          cabin_initial_temp_c=45.0)
            step = calc.cabin_thermal_step(plan, 45.0, 60.0)
            self.assertEqual(step["electrical_kw"], 0.0, mode)
            self.assertEqual(step["thermal_cooling_w"], 0.0, mode)
            self.assertTrue(step["setpoint_unmet"], mode)
            # A hot soak cabin falls by conduction alone, and a heating actuator
            # cannot reach a setpoint below ambient, so it settles at the ambient
            # asymptote rather than going cold: T(60 s) = 35 + 10*exp(-0.6).
            self.assertAlmostEqual(
                step["cabin_temp_end_c"], 35.0 + 10.0 * math.exp(-0.6), places=12
            )
            self.assertGreater(step["cabin_temp_end_c"], 35.0, mode)

    def test_zero_thermal_demand_costs_nothing(self):
        calc = RangeCalculator(cabin_config())
        plan = calc.resolve_hvac_plan(20.0, None, cabin_initial_temp_c=20.0)
        step = calc.cabin_thermal_step(plan, 20.0, 60.0)
        self.assertEqual(step["q_demand_w"], 0.0)
        self.assertEqual(step["electrical_kw"], 0.0)
        self.assertEqual(step["thermal_heating_w"], 0.0)
        self.assertEqual(step["cabin_temp_end_c"], 20.0)
        result = calc.estimate(flat_cycle(), ambient_c=20.0, cabin_initial_temp_c=20.0)
        self.assertEqual(result["hvac_energy_kwh"], 0.0)
        self.assertEqual(result["hvac_heating_thermal_energy_kwh"], 0.0)
        self.assertEqual(result["cabin_temp_end_c"], 20.0)

    def test_electrical_cap_limits_the_thermal_delivery_consistently(self):
        # Cap the heat pump at 1 kW electrical: delivery is exactly 1 kW * COP, and
        # the cabin then settles at the load-balance equilibrium, not the setpoint.
        calc = RangeCalculator(cabin_config(max_heating_electrical_kw=1.0))
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=5.0)
        step = calc.cabin_thermal_step(plan, 5.0, 60.0)
        self.assertAlmostEqual(step["electrical_kw"], 1.0, places=15)
        self.assertAlmostEqual(step["thermal_heating_w"], 4_000.0, places=9)
        self.assertTrue(step["power_clipped"])
        # Equilibrium: UA(T - T_amb) = Q_delivered -> T = 15 + 4000/1000 = 19 C.
        result = calc.estimate(flat_cycle(3600.0, 60), ambient_c=15.0,
                               cabin_initial_temp_c=5.0)
        self.assertAlmostEqual(result["cabin_temp_end_c"], 19.0, places=2)
        self.assertTrue(result["hvac_power_limited"])
        # 1.0 kW for 3600 s is exactly 1.0 kWh.
        self.assertAlmostEqual(result["hvac_energy_kwh"], 1.0, places=12)
        self.assertAlmostEqual(result["hvac_heating_thermal_energy_kwh"], 4.0, places=12)

    def test_shared_fifteen_kw_ceiling_leaves_headroom_for_the_blower_base(self):
        # A 10 kW blower base plus a 15 kW actuator cap cannot draw 25 kW: the
        # actuator is throttled to the remaining 5 kW and the thermal output follows.
        # Demand here is 1000*(20-0) + 1000*(20-0) = 40 kW thermal = 10 kW electrical
        # at COP 4, so the ceiling binds: 5 kW of actuator headroom, 20 kW of heat.
        calc = RangeCalculator(cabin_config(hvac_power_kw_default=10.0))
        plan = calc.resolve_hvac_plan(0.0, None, cabin_initial_temp_c=0.0)
        step = calc.cabin_thermal_step(plan, 0.0, 60.0)
        self.assertAlmostEqual(step["electrical_kw"], 15.0, places=15)
        self.assertAlmostEqual(step["thermal_heating_w"], 20_000.0, places=9)
        self.assertTrue(step["power_clipped"])
        self.assertAlmostEqual(step["cabin_temp_end_c"], 20.0 - 20.0 * math.exp(-0.6), places=12)
    def test_solar_and_occupancy_loads_enter_the_balance_once(self):
        calc = RangeCalculator(cabin_config())
        plain = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=15.0)
        loaded = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=15.0,
                                        cabin_solar_load_w=600.0,
                                        cabin_occupancy_load_w=400.0)
        self.assertEqual(plain["internal_gains_w"], 0.0)
        self.assertEqual(loaded["internal_gains_w"], 1000.0)
        base_step = calc.cabin_thermal_step(plain, 15.0, 60.0)
        loaded_step = calc.cabin_thermal_step(loaded, 15.0, 60.0)
        # Gains offset exactly 1 kW of heating demand, and shift the plant asymptote
        # upward by (Q_solar + Q_occ)/UA = 1.0 K.
        self.assertAlmostEqual(loaded_step["q_demand_w"], base_step["q_demand_w"] - 1000.0, places=9)
        self.assertAlmostEqual(loaded_step["thermal_heating_w"], base_step["thermal_heating_w"] - 1000.0, places=9)
        self.assertLess(loaded_step["electrical_kw"], base_step["electrical_kw"])

    def test_gains_warm_an_unmet_cabin_toward_their_own_equilibrium(self):
        # AC_COOLING with a heating demand delivers nothing, so the cabin relaxes to
        # ambient + gains/UA = 30 + 2000/1000 = 32 C.
        calc = RangeCalculator(cabin_config(mode="AC_COOLING"))
        plan = calc.resolve_hvac_plan(30.0, None, cabin_target_temp_c=45.0,
                                      cabin_initial_temp_c=30.0, cabin_solar_load_w=2000.0)
        step = calc.cabin_thermal_step(plan, 30.0, 60.0)
        self.assertEqual(step["electrical_kw"], 0.0)
        self.assertAlmostEqual(step["cabin_temp_end_c"], 32.0 - 2.0 * math.exp(-0.6), places=12)

    def test_cooling_load_rises_with_solar_gain_in_cooling_mode(self):
        calc = RangeCalculator(cabin_config(mode="AC_COOLING"))
        shaded = calc.estimate(flat_cycle(600.0, 20), ambient_c=35.0,
                               cabin_target_temp_c=25.0, cabin_initial_temp_c=35.0)
        sunny = calc.estimate(flat_cycle(600.0, 20), ambient_c=35.0,
                              cabin_target_temp_c=25.0, cabin_initial_temp_c=35.0,
                              cabin_solar_load_w=800.0)
        self.assertGreater(sunny["hvac_energy_kwh"], shaded["hvac_energy_kwh"])
        self.assertGreater(sunny["hvac_cooling_thermal_energy_kwh"],
                           shaded["hvac_cooling_thermal_energy_kwh"])
        self.assertGreater(shaded["hvac_cooling_thermal_energy_kwh"], 0.0)
        self.assertEqual(sunny["hvac_heating_thermal_energy_kwh"], 0.0)

    def test_heating_and_cooling_are_never_delivered_in_the_same_step(self):
        calc = RangeCalculator(cabin_config())
        for mode in ("HEAT_PUMP", "PTC", "AC_COOLING"):
            plan = calc.resolve_hvac_plan(-5.0, None, hvac_model_mode=mode,
                                          cabin_initial_temp_c=25.0,
                                          cabin_target_temp_c=20.0)
            temp = 25.0
            for _ in range(10):
                step = calc.cabin_thermal_step(plan, temp, 30.0)
                self.assertFalse(
                    step["thermal_heating_w"] > 0.0 and step["thermal_cooling_w"] > 0.0,
                    f"{mode} delivered both heats and cools in one interval",
                )
                self.assertGreaterEqual(step["electrical_kw"], 0.0)
                temp = step["cabin_temp_end_c"]


class HvacEnergyAccountingTests(unittest.TestCase):
    """Phase 3 Part 9: auxiliary and HVAC buckets are separate and never double-counted."""

    def test_auxiliary_energy_excludes_hvac_and_hvac_excludes_auxiliary(self):
        calc = RangeCalculator(cabin_config(mode="EMPIRICAL", auxiliary_power_kw=0.4,
                                            hvac_power_kw_default=1.1))
        result = calc.estimate(flat_cycle(600.0), ambient_c=20.0)  # deadband: HVAC = 1.1 kW
        self.assertAlmostEqual(result["auxiliary_energy_kwh"], 0.4 * 600.0 / 3600.0, places=15)
        self.assertAlmostEqual(result["hvac_energy_kwh"], 1.1 * 600.0 / 3600.0, places=15)
        self.assertLess(result["auxiliary_energy_kwh"], result["aux_hvac_energy_kwh"])

    def test_blower_base_lives_in_the_hvac_bucket_and_is_counted_exactly_once(self):
        # The other accounting tests run with hvac_power_kw_default=0.0, which hides
        # where the blower base actually lands. Here it is non-zero, so each bucket
        # can be pinned analytically: `auxiliary_power_kw` alone in aux, and the base
        # inside the HVAC bucket next to the actuator it shares a ceiling with.
        base, dur, aux_kw = 0.6, 600.0, 0.4
        blower_kwh = base * dur / 3600.0          # 0.1 kWh
        active = dict(ambient_c=-10.0, cabin_initial_temp_c=-5.0)
        calc_no = RangeCalculator(cabin_config(hvac_power_kw_default=0.0))
        calc_base = RangeCalculator(cabin_config(hvac_power_kw_default=base))
        no_base = calc_no.estimate(flat_cycle(dur, 20), **active)
        with_base = calc_base.estimate(flat_cycle(dur, 20), **active)

        # Aux holds the auxiliary load only: identical in both runs, and exactly aux x t.
        self.assertEqual(no_base["auxiliary_energy_kwh"], with_base["auxiliary_energy_kwh"])
        self.assertAlmostEqual(with_base["auxiliary_energy_kwh"], aux_kw * dur / 3600.0, places=15)
        # The cabin never sees the blower, so the thermal side is untouched; only the
        # electrical bucket grows, and by exactly one copy of the base load.
        self.assertEqual(with_base["cabin_temp_end_c"], no_base["cabin_temp_end_c"])
        self.assertEqual(with_base["hvac_heating_thermal_energy_kwh"],
                         no_base["hvac_heating_thermal_energy_kwh"])
        self.assertAlmostEqual(with_base["hvac_energy_kwh"] - no_base["hvac_energy_kwh"],
                               blower_kwh, places=15)
        self.assertAlmostEqual(with_base["aux_hvac_energy_kwh"] - no_base["aux_hvac_energy_kwh"],
                               blower_kwh, places=15)
        self.assertEqual(with_base["traction_energy_kwh"], no_base["traction_energy_kwh"])

        # No compatible actuator demand: the HVAC bucket is then exactly the blower.
        idle = calc_base.estimate(flat_cycle(dur, 20), ambient_c=45.0,
                                  cabin_initial_temp_c=45.0)
        self.assertEqual(idle["hvac_energy_kwh"], blower_kwh)
        self.assertEqual(idle["hvac_heating_thermal_energy_kwh"], 0.0)
        self.assertTrue(idle["cabin_setpoint_unmet"])

        # EMPIRICAL deadband: the blower is the whole HVAC load, counted once.
        empirical = RangeCalculator(cabin_config(mode="EMPIRICAL",
                                                 hvac_power_kw_default=base))
        deadband = empirical.estimate(flat_cycle(dur, 20), ambient_c=20.0)
        self.assertEqual(deadband["hvac_mode"], "COMFORT_DEADBAND_VENTILATION")
        self.assertEqual(deadband["hvac_energy_kwh"], blower_kwh)

        # A manual override is the TOTAL HVAC electrical load, so it replaces the base
        # instead of adding to it -- the pre-Phase-3 meaning of `hvac_power_kw`.
        override = calc_base.estimate(flat_cycle(dur, 20), ambient_c=-10.0, hvac_power_kw=3.0)
        self.assertAlmostEqual(override["hvac_energy_kwh"], 3.0 * dur / 3600.0, places=15)
        self.assertNotAlmostEqual(override["hvac_energy_kwh"],
                                  (3.0 + base) * dur / 3600.0, places=15)

    def test_combined_field_is_exactly_the_sum_in_every_mode(self):
        # The spec requires the preserved field to be the sum of the two buckets,
        # so this is asserted with exact equality, not a tolerance.
        for mode in ("EMPIRICAL", "HEAT_PUMP", "PTC", "AC_COOLING"):
            calc = RangeCalculator(cabin_config(mode=mode))
            for ambient in (None, -10.0, 20.0, 45.0):
                for override in (None, 0.0, 4.25):
                    if mode != "EMPIRICAL" and ambient is None and override is None:
                        # The cabin model has no meaning without an ambient temperature;
                        # the electrical override is the only path that short-circuits it.
                        with self.assertRaises(ValueError):
                            calc.estimate(flat_cycle(600.0, 20))
                        continue
                    result = calc.estimate(flat_cycle(600.0, 20), ambient_c=ambient,
                                           hvac_power_kw=override)
                    self.assertEqual(
                        result["auxiliary_energy_kwh"] + result["hvac_energy_kwh"],
                        result["aux_hvac_energy_kwh"],
                        (mode, ambient, override),
                    )

    def test_cycle_energy_adds_hvac_exactly_once(self):
        for mode in ("EMPIRICAL", "HEAT_PUMP", "PTC", "AC_COOLING"):
            calc = RangeCalculator(cabin_config(mode=mode))
            result = calc.estimate(flat_cycle(600.0, 20), ambient_c=-10.0)
            self.assertAlmostEqual(
                result["cycle_energy_kwh"],
                result["traction_energy_kwh"] - result["regen_recovered_kwh"]
                + result["auxiliary_energy_kwh"] + result["hvac_energy_kwh"],
                places=15, msg=mode,
            )

    def test_hvac_energy_matches_the_legacy_combined_formula_to_double_precision(self):
        # Before Phase 3 a single accumulator held (auxiliary + HVAC) x dt. The split
        # accumulators must reproduce that number, differing only by float reassociation.
        calc = RangeCalculator(cabin_config(mode="EMPIRICAL", auxiliary_power_kw=0.4,
                                            hvac_power_kw_default=0.25))
        for ambient in (None, -20.0, 5.0, 25.0, 50.0):
            for override in (None, 0.0, 7.5):
                result = calc.estimate(flat_cycle(600.0, 20), ambient_c=ambient,
                                       hvac_power_kw=override)
                legacy = (result["hvac_power_kw"] + 0.4) * 600.0 / 3600.0
                self.assertAlmostEqual(result["aux_hvac_energy_kwh"], legacy, places=15,
                                       msg=(ambient, override))

    def test_zero_hvac_produces_no_hvac_energy(self):
        calc = RangeCalculator(cabin_config())
        manual = calc.estimate(flat_cycle(600.0, 20), ambient_c=5.0, hvac_power_kw=0.0)
        self.assertEqual(manual["hvac_energy_kwh"], 0.0)
        self.assertEqual(manual["aux_hvac_energy_kwh"], manual["auxiliary_energy_kwh"])
        balanced = calc.estimate(flat_cycle(600.0, 20), ambient_c=20.0,
                                 cabin_initial_temp_c=20.0)
        self.assertEqual(balanced["hvac_energy_kwh"], 0.0)
        self.assertEqual(balanced["hvac_heating_thermal_energy_kwh"], 0.0)
        self.assertEqual(balanced["aux_hvac_energy_kwh"], balanced["auxiliary_energy_kwh"])

    def test_more_hvac_electrical_energy_means_less_range(self):
        calc = RangeCalculator(cabin_config())
        kwargs = dict(ambient_c=15.0, cabin_initial_temp_c=15.0)
        efficient = calc.estimate(flat_cycle(600.0, 20), heat_pump_cop=4.0, **kwargs)
        thriftless = calc.estimate(flat_cycle(600.0, 20), heat_pump_cop=2.0, **kwargs)
        # Same thermal delivery, exactly twice the electrical draw.
        self.assertEqual(thriftless["hvac_heating_thermal_energy_kwh"],
                         efficient["hvac_heating_thermal_energy_kwh"])
        self.assertEqual(thriftless["hvac_energy_kwh"] / efficient["hvac_energy_kwh"], 2.0)
        self.assertEqual(thriftless["traction_energy_kwh"], efficient["traction_energy_kwh"])
        self.assertGreater(thriftless["consumption_kwh_per_km"],
                           efficient["consumption_kwh_per_km"])
        self.assertLess(thriftless["estimated_range_km"], efficient["estimated_range_km"])

    def test_reported_hvac_power_is_the_energy_weighted_mean_in_cabin_modes(self):
        calc = RangeCalculator(cabin_config())
        result = calc.estimate(flat_cycle(600.0, 20), ambient_c=15.0,
                               cabin_initial_temp_c=5.0)
        self.assertAlmostEqual(
            result["hvac_power_kw"],
            result["hvac_energy_kwh"] * 3600.0 / result["duration_s"], places=15
        )
        # It must lie between the first interval (recovery load) and the settled load.
        first = calc.cabin_thermal_step(
            calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=5.0), 5.0, 30.0
        )["electrical_kw"]
        settled = calc.cabin_thermal_step(
            calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=20.0), 20.0, 30.0
        )["electrical_kw"]
        self.assertLessEqual(settled, result["hvac_power_kw"])
        self.assertLessEqual(result["hvac_power_kw"], first)

    def test_thermal_ledger_follows_the_electrical_ledger_by_the_cop(self):
        # With no blower base, HVAC electrical energy x COP is the thermal energy
        # delivered: the thermal figure is a diagnostic, never an extra battery load.
        for mode, factor_key in (
            ("HEAT_PUMP", "hvac_heating_cop"), ("AC_COOLING", "hvac_cooling_cop")
        ):
            calc = RangeCalculator(cabin_config(mode=mode))
            ambient = 15.0 if mode == "HEAT_PUMP" else 35.0
            target = 20.0 if mode == "HEAT_PUMP" else 25.0
            result = calc.estimate(flat_cycle(600.0, 20), ambient_c=ambient,
                                   cabin_target_temp_c=target,
                                   cabin_initial_temp_c=ambient)
            thermal = (result["hvac_heating_thermal_energy_kwh"] if mode == "HEAT_PUMP"
                       else result["hvac_cooling_thermal_energy_kwh"])
            self.assertAlmostEqual(thermal, result["hvac_energy_kwh"] * result[factor_key],
                                   places=12, msg=mode)
            self.assertGreater(thermal, result["hvac_energy_kwh"], mode)

    def test_ptc_thermal_ledger_is_the_electrical_ledger_times_efficiency(self):
        calc = RangeCalculator(cabin_config(mode="PTC"))
        result = calc.estimate(flat_cycle(600.0, 20), ambient_c=15.0,
                               cabin_initial_temp_c=15.0)
        self.assertAlmostEqual(
            result["hvac_heating_thermal_energy_kwh"],
            result["hvac_energy_kwh"] * result["hvac_heating_efficiency"], places=12
        )
        # Resistance heating wastes energy, so its thermal delivery is below its draw.
        self.assertLess(result["hvac_heating_thermal_energy_kwh"], result["hvac_energy_kwh"])

    def test_phase_one_braking_ledger_identities_still_hold_under_a_cabin_mode(self):
        # A decelerating cycle engages the regenerator inside a cabin-thermal run, so
        # the Phase 1 identities are re-checked with the new accounting in place.
        cycle = [
            {"time_s": 0.0, "speed_mps": 25.0, "grade_pct": 0.0},
            {"time_s": 20.0, "speed_mps": 25.0, "grade_pct": 0.0},
            {"time_s": 50.0, "speed_mps": 5.0, "grade_pct": 0.0},
            {"time_s": 60.0, "speed_mps": 0.0, "grade_pct": 0.0},
        ]
        calc = RangeCalculator(cabin_config())
        result = calc.estimate(cycle, ambient_c=5.0)
        self.assertAlmostEqual(
            result["mechanical_braking_kwh"],
            result["regen_mechanical_absorbed_kwh"] + result["friction_braking_kwh"],
            places=12,
        )
        self.assertAlmostEqual(
            result["regen_recovered_kwh"],
            0.5 * result["regen_mechanical_absorbed_kwh"], places=12
        )
        self.assertLessEqual(result["regen_recovered_kwh"], result["mechanical_braking_kwh"])
        self.assertGreater(result["hvac_energy_kwh"], 0.0)

    def test_cabin_diagnostics_are_absent_when_the_mode_does_not_use_them(self):
        calc = RangeCalculator(cabin_config())
        for mode in ("EMPIRICAL",):
            result = calc.estimate(flat_cycle(), ambient_c=5.0, hvac_model_mode=mode)
            for key in (
                "cabin_temp_start_c", "cabin_temp_end_c", "cabin_time_constant_s",
                "cabin_target_temp_c", "hvac_heating_cop", "hvac_heating_efficiency",
                "hvac_cooling_cop", "cabin_setpoint_unmet", "hvac_power_limited",
                "hvac_heating_thermal_energy_kwh", "hvac_cooling_thermal_energy_kwh",
            ):
                self.assertIsNone(result[key], (mode, key))


class CabinThermalNumericalStabilityTests(unittest.TestCase):
    """Phase 3 Part 4: the discretization, its stability bound and its convergence."""

    def test_pure_relaxation_is_invariant_to_time_step_refinement(self):
        # With no compatible actuator the plant is a linear relaxation, which the
        # closed-form integrator solves exactly: any grid gives the same end state.
        calc = RangeCalculator(cabin_config(mode="AC_COOLING"))
        expected = 30.0 + 15.0 * math.exp(-1000.0 * 600.0 / 100_000.0)
        for intervals in (5, 10, 25, 60, 120):
            result = calc.estimate(flat_cycle(600.0, intervals), ambient_c=30.0,
                                   cabin_target_temp_c=45.0, cabin_initial_temp_c=45.0)
            self.assertAlmostEqual(result["cabin_temp_end_c"], expected, places=12,
                                   msg=intervals)
            self.assertEqual(result["hvac_energy_kwh"], 0.0, intervals)

    def test_zero_order_hold_energy_converges_at_first_order(self):
        # The control law is held constant over an interval, so refining the grid
        # changes the answer: the measured successive differences halve, which is
        # first-order (O(dt)) convergence toward the continuous-time solution.
        calc = RangeCalculator(cabin_config())
        energies = [
            calc.estimate(flat_cycle(600.0, intervals), ambient_c=15.0,
                          cabin_initial_temp_c=5.0)["hvac_energy_kwh"]
            for intervals in (10, 20, 40, 80)
        ]
        differences = [
            energies[i] - energies[i + 1] for i in range(len(energies) - 1)
        ]
        self.assertTrue(all(delta > 0.0 for delta in differences), differences)
        ratios = [differences[i + 1] / differences[i] for i in range(len(differences) - 1)]
        for ratio in ratios:
            self.assertGreater(ratio, 0.40)
            self.assertLess(ratio, 0.60)
        # Richardson extrapolation of a first-order sequence: 2*L - S is the limit
        # estimate, so successive refinements must bracket it tightly.
        limit_estimate = 2.0 * energies[-1] - energies[-2]
        self.assertAlmostEqual(energies[-1], limit_estimate, delta=0.005)

    def test_discrete_contraction_matches_the_derived_factor(self):
        # Error map derived for the loop: e_next = e * ((1 + k) rho - k) with
        # k = K_p/UA = 1 and rho = exp(-UA dt / C). A small initial error keeps the
        # demand positive, so the actuator stays in its heating regime each step.
        calc = RangeCalculator(cabin_config())
        dt, ua, cap = 200.0, 1000.0, 100_000.0
        rho = math.exp(-ua * dt / cap)
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=18.0)
        temp = 18.0
        errors = [20.0 - temp]
        for _ in range(4):
            step = calc.cabin_thermal_step(plan, temp, dt)
            temp = step["cabin_temp_end_c"]
            errors.append(20.0 - temp)
            self.assertFalse(step["setpoint_unmet"])
        for i in range(len(errors) - 1):
            self.assertAlmostEqual(
                errors[i + 1] / errors[i], (1.0 + 1.0) * rho - 1.0, places=9,
                msg=f"step {i}"
            )

    def test_approach_is_monotone_when_the_contraction_factor_is_positive(self):
        # For a short interval rho > 0.5, so the factor 2 rho - 1 is positive and the
        # cabin climbs to the setpoint without ringing.
        calc = RangeCalculator(cabin_config())
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=5.0)
        temps = []
        temp = 5.0
        for _ in range(8):
            step = calc.cabin_thermal_step(plan, temp, 60.0)
            temp = step["cabin_temp_end_c"]
            temps.append(temp)
        self.assertTrue(all(later >= earlier for earlier, later in zip(temps, temps[1:])), temps)
        self.assertTrue(all(value <= 20.0 + 1e-9 for value in temps), temps)

    def test_intervals_longer_than_the_cabin_constant_oscillate_but_stay_bounded(self):
        # An interval much longer than C/UA makes the factor negative: the cabin rings
        # about the setpoint, but the derived bound |factor| < 1 keeps it bounded.
        calc = RangeCalculator(cabin_config())
        plan = calc.resolve_hvac_plan(15.0, None, cabin_initial_temp_c=5.0)
        temp = 5.0
        extremes = []
        for _ in range(24):
            step = calc.cabin_thermal_step(plan, temp, 1000.0)
            temp = step["cabin_temp_end_c"]
            extremes.append(temp)
            self.assertTrue(math.isfinite(temp))
            self.assertGreaterEqual(temp, -50.0)
            self.assertLessEqual(temp, 80.0)
        first_half = max(abs(value - 20.0) for value in extremes[:6])
        last_half = max(abs(value - 20.0) for value in extremes[-6:])
        self.assertLess(last_half, first_half)

    def test_recovery_constant_shorter_than_the_plant_constant_is_rejected(self):
        for bad_tau in (50.0, 99.99):
            with self.assertRaises(ValueError, msg=bad_tau) as ctx:
                cabin_config(cabin_recovery_time_constant_s=bad_tau)
            self.assertIn("cabin_recovery_time_constant_s", str(ctx.exception))
        # The bound itself is admissible: k = 1 keeps |factor| strictly below one.
        cfg = cabin_config(cabin_recovery_time_constant_s=100.0)
        self.assertEqual(cfg.cabin_recovery_time_constant_s, 100.0)
        plan = RangeCalculator(cfg).resolve_hvac_plan(15.0, None)
        self.assertAlmostEqual(plan["gain_w_per_k"], 1000.0, places=12)
        self.assertAlmostEqual(plan["tau_effective_s"], 100_000.0 / 2000.0, places=12)
        self.assertAlmostEqual(plan["tau_plant_s"], 100.0, places=12)

    def test_a_slower_recovery_setting_lengthens_the_effective_time_constant(self):
        calc = RangeCalculator(cabin_config(cabin_recovery_time_constant_s=200.0))
        plan = calc.resolve_hvac_plan(15.0, None)
        self.assertAlmostEqual(plan["gain_w_per_k"], 500.0, places=12)
        self.assertAlmostEqual(plan["tau_effective_s"], 100_000.0 / 1500.0, places=12)
        # K_p = C/tau_recovery with tau_recovery = 2 C/UA gives k = 0.5, and the
        # derived factor (1 + k) rho - k must match the measured error contraction.
        dt, ua, cap = 60.0, 1000.0, 100_000.0
        rho = math.exp(-ua * dt / cap)
        temp = 18.0
        errors = [20.0 - temp]
        for _ in range(3):
            step = calc.cabin_thermal_step(plan, temp, dt)
            temp = step["cabin_temp_end_c"]
            errors.append(20.0 - temp)
        for i in range(len(errors) - 1):
            self.assertAlmostEqual(errors[i + 1] / errors[i], 1.5 * rho - 0.5, places=9)

    def test_perfectly_insulated_cabin_is_rejected_for_the_thermal_modes(self):
        # UA=0 stays loadable, because the shipped default mode never reads it...
        RangeCalculator(cabin_config(mode="EMPIRICAL", cabin_ua_w_per_k=0.0))
        calc = RangeCalculator(cabin_config(mode="EMPIRICAL", cabin_ua_w_per_k=0.0))
        calc.estimate(flat_cycle(), ambient_c=5.0)
        # ...but a thermal model with no conductive leak has no steady state.
        for mode in ("HEAT_PUMP", "PTC", "AC_COOLING"):
            with self.assertRaises(ValueError, msg=mode) as ctx:
                calc.resolve_hvac_plan(5.0, None, hvac_model_mode=mode)
            self.assertIn("cabin_ua_w_per_k", str(ctx.exception))

    def test_default_cabin_parameters_are_not_used_by_the_empirical_mode(self):
        # Absurd cabin parameters must not perturb an EMPIRICAL run at all, which is
        # how "existing calls are unchanged" is proven for the default profiles.
        cabin_only = {
            key: value for key, value in CABIN_VEHICLE.items()
            if key.startswith(("cabin_", "heat_pump_", "ac_cooling_", "ptc_", "max_"))
        }
        self.assertTrue(cabin_only)
        plain = RangeCalculator(config()).estimate(cycle(12.0), soh_pct=90.0, ambient_c=8.0)
        loaded = RangeCalculator(
            config(hvac_model_mode="EMPIRICAL", **cabin_only)
        ).estimate(cycle(12.0), soh_pct=90.0, ambient_c=8.0)
        self.assertEqual(plain, loaded)

    def test_cabin_limitations_are_reported_only_for_cabin_modes(self):
        cabin = RangeCalculator(cabin_config()).estimate(flat_cycle(), ambient_c=15.0)
        plain = RangeCalculator(cabin_config(mode="EMPIRICAL")).estimate(
            flat_cycle(), ambient_c=15.0
        )
        self.assertTrue(any("Cabin thermal model HEAT_PUMP" in text for text in cabin["limitations"]))
        self.assertTrue(any("not a calibrated cabin simulation" in text
                            for text in cabin["limitations"]))
        self.assertTrue(any("One actuator per mode" in text for text in cabin["limitations"]))
        self.assertFalse(any("Cabin thermal model" in text for text in plain["limitations"]))

    def test_quality_status_stays_requires_validation_for_a_cabin_run(self):
        result = RangeCalculator(cabin_config()).estimate(flat_cycle(), ambient_c=-10.0)
        self.assertEqual(result["quality_status"], "REQUIRES_VALIDATION")


class HvacTrajectoryAndCliTests(unittest.TestCase):
    """Phase 3 Part 10: cabin modes ride through estimate_trajectory() and the CLI."""

    def cabin_keys(self):
        return {key for key in CABIN_VEHICLE
                if key.startswith(("cabin_", "heat_pump_", "ac_cooling_", "ptc_", "max_"))}

    def write_cabin_profile(self, tmp_path):
        """A shipped profile plus the illustrative cabin fields, as a standalone JSON."""
        profile = json.loads((ROOT / "vehicle_config.json").read_text(encoding="utf-8"))
        profile.update({key: CABIN_VEHICLE[key] for key in self.cabin_keys()})
        profile["hvac_model_mode"] = "HEAT_PUMP"
        profile["hvac_power_kw_default"] = 0.0  # isolate the actuator from the blower base
        path = tmp_path / "cabin_profile.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        return path

    def run_cli(self, config_path, *extra):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "range_calculator.py"),
             "--config", str(config_path),
             "--cycle", str(ROOT / "drive_cycles" / "example_cycle.csv")] + list(extra),
            cwd=str(ROOT), capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return json.loads(proc.stdout)

    # ------------------------------------------------------------------ trajectory

    def test_trajectory_forwards_the_cabin_scenario_to_every_point(self):
        calc = RangeCalculator(cabin_config(mode="HEAT_PUMP"))
        rows = calc.estimate_trajectory(
            flat_cycle(600.0, 20), [100.0, 95.0, 80.0], ambient_c=-10.0,
            cabin_target_temp_c=21.0, cabin_initial_temp_c=5.0,
            cabin_solar_load_w=120.0, cabin_occupancy_load_w=200.0, heat_pump_cop=3.0,
        )
        self.assertEqual([row["soh_pct"] for row in rows], [100.0, 95.0, 80.0])
        for row in rows:
            self.assertEqual(row["hvac_model_mode"], "HEAT_PUMP")
            self.assertEqual(row["cabin_target_temp_c"], 21.0)
            self.assertEqual(row["cabin_temp_start_c"], 5.0)
            self.assertEqual(row["hvac_heating_cop"], 3.0)
            self.assertEqual(row["auxiliary_energy_kwh"] + row["hvac_energy_kwh"],
                             row["aux_hvac_energy_kwh"])

    def test_trajectory_identifiers_and_quality_flags_survive_the_new_kwargs(self):
        calc = RangeCalculator(cabin_config(mode="PTC"))
        rows = calc.estimate_trajectory(
            flat_cycle(600.0, 20),
            [{"soh_pct": 96.0, "cycle_index": 12, "cell_id": "CELL_A",
              "quality_flag": "LSTM_PREDICTED"},
             {"soh_pct": 90.0, "cycle_index": 24, "cell_id": "CELL_A",
              "quality_flag": "LSTM_PREDICTED"}],
            ambient_c=35.0, hvac_model_mode="AC_COOLING", cabin_target_temp_c=22.0,
        )
        self.assertEqual([row["cycle_index"] for row in rows], [12, 24])
        self.assertEqual([row["cell_id"] for row in rows], ["CELL_A", "CELL_A"])
        self.assertEqual([row["quality_flag"] for row in rows],
                         ["LSTM_PREDICTED", "LSTM_PREDICTED"])
        self.assertEqual([row["hvac_model_mode"] for row in rows],
                         ["AC_COOLING", "AC_COOLING"])
        self.assertEqual([row["hvac_cooling_cop"] for row in rows],
                         [rows[0]["hvac_cooling_cop"]] * 2)
        self.assertIsNone(rows[0]["hvac_heating_cop"])

    def test_each_trajectory_point_restarts_the_cabin_from_its_soak_state(self):
        # The cabin model never reads SOH, so every point must report the same thermal
        # answer: per-point evaluation is independent, heat is not carried across the
        # ageing series. SOH only shrinks the usable pack, so range still decays.
        calc = RangeCalculator(cabin_config(mode="HEAT_PUMP"))
        rows = calc.estimate_trajectory(flat_cycle(600.0, 20), [100.0, 92.0, 84.0],
                                        ambient_c=-10.0)
        for key in ("cabin_temp_start_c", "cabin_temp_end_c", "hvac_energy_kwh",
                    "hvac_heating_thermal_energy_kwh", "cycle_energy_kwh"):
            self.assertEqual(len({row[key] for row in rows}), 1, key)
        self.assertLess(rows[2]["estimated_range_km"], rows[0]["estimated_range_km"])
        self.assertLess(rows[2]["usable_energy_kwh"], rows[0]["usable_energy_kwh"])

    def test_trajectory_positional_call_is_unchanged(self):
        calc = RangeCalculator(cabin_config(mode="EMPIRICAL"))
        positional = calc.estimate_trajectory(flat_cycle(600.0, 20), [95.0, 90.0],
                                             20.0, 2.0, ["FLAG_A", None], "CELL_P",
                                             None, None, None, None)
        keyword = calc.estimate_trajectory(flat_cycle(600.0, 20), [95.0, 90.0],
                                           ambient_c=20.0, hvac_power_kw=2.0,
                                           quality_flags=["FLAG_A", None],
                                           cell_id="CELL_P")
        self.assertEqual(positional, keyword)
        self.assertTrue(PRE_PHASE3_RESULT_KEYS <= set(positional[0]))

    def test_trajectory_manual_override_equals_the_direct_estimate_rows(self):
        calc = RangeCalculator(cabin_config(mode="HEAT_PUMP"))
        rows = calc.estimate_trajectory(flat_cycle(600.0, 20), [100.0, 88.0],
                                        ambient_c=-10.0, hvac_power_kw=3.0)
        for row, soh in zip(rows, (100.0, 88.0)):
            direct = calc.estimate(flat_cycle(600.0, 20), soh_pct=soh, ambient_c=-10.0,
                                   hvac_power_kw=3.0)
            self.assertEqual(row, direct)
            self.assertEqual(row["hvac_mode"], "MANUAL_OVERRIDE")
            self.assertIsNone(row["cabin_temp_end_c"])

    def test_trajectory_rejects_a_cabin_point_without_ambient(self):
        calc = RangeCalculator(cabin_config(mode="HEAT_PUMP"))
        with self.assertRaises(ValueError):
            calc.estimate_trajectory(flat_cycle(600.0, 20), [100.0])

    # ------------------------------------------------------------------ thermal math

    def test_uncapped_thermal_delivery_is_cop_independent(self):
        # With no electrical cap binding, the closed-loop cabin trajectory cannot know
        # the COP: heat delivered equals demand, so only the electrical bill scales.
        # This is the arithmetic check that the COP is applied on the right side.
        common = dict(ambient_c=5.0, cabin_initial_temp_c=5.0, cabin_target_temp_c=20.0,
                      max_heating_electrical_kw=15.0)
        cheap = RangeCalculator(cabin_config(mode="HEAT_PUMP")).estimate(
            flat_cycle(600.0, 20), heat_pump_cop=4.0, **common)
        better = RangeCalculator(cabin_config(mode="HEAT_PUMP")).estimate(
            flat_cycle(600.0, 20), heat_pump_cop=8.0, **common)
        self.assertFalse(cheap["hvac_power_limited"])
        self.assertFalse(better["hvac_power_limited"])
        self.assertAlmostEqual(cheap["cabin_temp_end_c"], better["cabin_temp_end_c"],
                               places=12)
        self.assertAlmostEqual(cheap["hvac_heating_thermal_energy_kwh"],
                               better["hvac_heating_thermal_energy_kwh"], places=12)
        self.assertAlmostEqual(cheap["hvac_energy_kwh"] / better["hvac_energy_kwh"], 2.0,
                               places=9)
        self.assertGreater(better["estimated_range_km"], cheap["estimated_range_km"])

    # ------------------------------------------------------------------ CLI

    def test_cli_default_run_is_still_the_empirical_model(self):
        result = self.run_cli(ROOT / "vehicle_config.json")
        self.assertEqual(result["hvac_model_mode"], "EMPIRICAL")
        self.assertIn(result["hvac_mode"], LEGACY_HVAC_LABELS)
        for key in ("cabin_temp_end_c", "hvac_heating_thermal_energy_kwh",
                    "cabin_setpoint_unmet", "hvac_heating_cop"):
            self.assertIsNone(result[key])
        self.assertEqual(result["auxiliary_energy_kwh"] + result["hvac_energy_kwh"],
                         result["aux_hvac_energy_kwh"])

    def test_cli_cabin_mode_flags_reach_the_engine(self):
        result = self.run_cli(
            ROOT / "vehicle_config.json", "--hvac-mode", "HEAT_PUMP",
            "--ambient-c", "-5", "--cabin-target-c", "21", "--cabin-initial-c", "2",
            "--solar-load-w", "100", "--occupancy-load-w", "150",
            "--heat-pump-cop", "4", "--max-heating-kw", "10",
        )
        self.assertEqual(result["hvac_model_mode"], "HEAT_PUMP")
        self.assertEqual(result["cabin_target_temp_c"], 21.0)
        self.assertEqual(result["cabin_temp_start_c"], 2.0)
        self.assertEqual(result["hvac_heating_cop"], 4.0)
        self.assertGreater(result["cabin_temp_end_c"], 2.0)
        self.assertGreater(result["hvac_energy_kwh"], 0.0)
        self.assertEqual(result["auxiliary_energy_kwh"] + result["hvac_energy_kwh"],
                         result["aux_hvac_energy_kwh"])

    def test_cli_manual_hvac_kw_keeps_precedence_over_the_thermal_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = self.write_cabin_profile(Path(tmp))
            result = self.run_cli(profile, "--hvac-mode", "HEAT_PUMP",
                                  "--ambient-c", "5", "--hvac-kw", "2.5")
            self.assertEqual(result["hvac_mode"], "MANUAL_OVERRIDE")
            self.assertIsNone(result["cabin_temp_end_c"])
            self.assertAlmostEqual(result["hvac_energy_kwh"],
                                   2.5 * result["duration_s"] / 3600.0, places=12)

    def test_cli_heating_cop_only_changes_the_electrical_bill(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = self.write_cabin_profile(Path(tmp))
            flags = ["--hvac-mode", "HEAT_PUMP", "--ambient-c", "5",
                     "--cabin-initial-c", "5", "--cabin-target-c", "20",
                     "--max-heating-kw", "15", "--heat-pump-cop"]
            cheap = self.run_cli(profile, *(flags + ["4"]))
            better = self.run_cli(profile, *(flags + ["8"]))
            self.assertFalse(cheap["hvac_power_limited"])
            self.assertAlmostEqual(cheap["hvac_heating_thermal_energy_kwh"],
                                   better["hvac_heating_thermal_energy_kwh"], places=10)
            self.assertAlmostEqual(cheap["hvac_energy_kwh"],
                                   2.0 * better["hvac_energy_kwh"], places=8)

    def test_cli_rejects_an_unknown_hvac_mode_before_running(self):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "range_calculator.py"),
             "--config", str(ROOT / "vehicle_config.json"),
             "--cycle", str(ROOT / "drive_cycles" / "example_cycle.csv"),
             "--hvac-mode", "HYBRID"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("invalid choice", proc.stderr)


if __name__ == "__main__":
    unittest.main()

