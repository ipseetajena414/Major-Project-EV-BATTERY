#!/usr/bin/env python3
"""Transparent baseline EV range estimator; physics-based vehicle dynamics model.

Implements interval-based road load integration, temperature-dependent air density,
empirical HVAC thermal scenarios, regenerative braking with low-speed blend cutoff,
and trajectory-based SOH evaluation for multi-agent battery life integration.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class VehicleConfig:
    vehicle_name: str
    nominal_pack_energy_kwh: float
    usable_energy_fraction: float
    vehicle_mass_kg: float
    drag_coefficient_cd: float
    frontal_area_m2: float
    rolling_resistance_coefficient_crr: float
    air_density_kg_m3: float = 1.225
    drivetrain_efficiency: float = 0.90
    regen_efficiency: float = 0.0
    max_regen_power_kw: float = 0.0
    auxiliary_power_kw: float = 0.0
    hvac_power_kw_default: float = 0.0
    rotational_inertia_factor: float = 1.05
    rated_range_km_reference: float | None = None
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "VehicleConfig":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        positive = {
            "nominal_pack_energy_kwh": self.nominal_pack_energy_kwh,
            "vehicle_mass_kg": self.vehicle_mass_kg,
            "frontal_area_m2": self.frontal_area_m2,
            "air_density_kg_m3": self.air_density_kg_m3,
            "drag_coefficient_cd": self.drag_coefficient_cd,
        }
        for name, value in positive.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")

        for name, value in {
            "usable_energy_fraction": self.usable_energy_fraction,
            "drivetrain_efficiency": self.drivetrain_efficiency,
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not (0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1]")

        for name, value in {
            "rolling_resistance_coefficient_crr": self.rolling_resistance_coefficient_crr,
            "regen_efficiency": self.regen_efficiency,
            "max_regen_power_kw": self.max_regen_power_kw,
            "auxiliary_power_kw": self.auxiliary_power_kw,
            "hvac_power_kw_default": self.hvac_power_kw_default,
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")

        if self.regen_efficiency > 1.0:
            raise ValueError("regen_efficiency must be <= 1.0")

        if not isinstance(self.rotational_inertia_factor, (int, float)) or not math.isfinite(self.rotational_inertia_factor) or self.rotational_inertia_factor < 1.0:
            raise ValueError("rotational_inertia_factor must be finite and >= 1.0")

        if self.rated_range_km_reference is not None:
            if not isinstance(self.rated_range_km_reference, (int, float)) or not math.isfinite(self.rated_range_km_reference) or self.rated_range_km_reference <= 0:
                raise ValueError("rated_range_km_reference must be finite and > 0 if specified")


class RangeCalculator:
    """Estimate EV driving range from speed/time/grade cycle and vehicle dynamics.

    Drive-cycle CSV columns:
      time_s: strictly increasing time in seconds
      speed_mps: non-negative vehicle speed in m/s
      grade_pct: road grade as percent (e.g. 5 means 5%, not 5 degrees); optional, defaults to 0

    SOH is a percentage on the 0-100 scale (e.g. 80.0 for 80%).
    Usable pack energy is calculated as:
      nominal_pack_energy_kwh * usable_energy_fraction * (SOH / 100.0)
    """

    G = 9.80665
    P_ATM_PA = 101325.0
    R_SPEC_AIR = 287.058
    V_REGEN_CUTOFF_LOW_MPS = 0.5    # Below 1.8 km/h, friction brakes take over completely
    V_REGEN_CUTOFF_FULL_MPS = 2.0   # Above 7.2 km/h, full regenerative blending is available

    def __init__(self, config: VehicleConfig):
        self.c = config
        self._validate_config()

    def _validate_config(self) -> None:
        self.c.validate()

    def air_density(self, ambient_c: float | None) -> float:
        """Calculate dry air density using the ideal gas law at 101.325 kPa."""
        if ambient_c is None:
            return self.c.air_density_kg_m3
        t_kelvin = ambient_c + 273.15
        if t_kelvin <= 0:
            raise ValueError("ambient_c cannot be at or below absolute zero (-273.15 C)")
        return self.P_ATM_PA / (self.R_SPEC_AIR * t_kelvin)

    def empirical_hvac_kw(self, ambient_c: float | None, override_kw: float | None) -> tuple[float, str]:
        """Compute HVAC electrical demand from ambient temperature or user override."""
        if override_kw is not None:
            return override_kw, "MANUAL_OVERRIDE"
        if ambient_c is None:
            return self.c.hvac_power_kw_default, "CONFIG_DEFAULT"

        # Empirical piecewise automotive thermal comfort curve
        if 18.0 <= ambient_c <= 22.0:
            return self.c.hvac_power_kw_default, "COMFORT_DEADBAND_VENTILATION"
        elif ambient_c < 18.0:
            delta_t = 18.0 - ambient_c
            # Heating ramp: baseline + 0.18 kW/C up to 6.0 kW cap (heat pump/PTC blend)
            p_heat = min(6.0, self.c.hvac_power_kw_default + 0.18 * delta_t)
            return p_heat, "EMPIRICAL_HEATING_DEMAND"
        else:
            delta_t = ambient_c - 22.0
            # Cooling ramp: baseline + 0.12 kW/C up to 3.5 kW cap (AC compressor)
            p_cool = min(3.5, self.c.hvac_power_kw_default + 0.12 * delta_t)
            return p_cool, "EMPIRICAL_COOLING_DEMAND"

    @staticmethod
    def load_cycle(path: str | Path) -> list[dict[str, float]]:
        points: list[dict[str, float]] = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"time_s", "speed_mps"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"Drive-cycle CSV must contain columns {sorted(required)}")
            for row in reader:
                try:
                    t = float(row["time_s"])
                    v = float(row["speed_mps"])
                    grade = float(row.get("grade_pct") or 0.0)
                except (TypeError, ValueError) as e:
                    raise ValueError(f"Invalid numeric drive-cycle row: {row}") from e
                if not all(math.isfinite(x) for x in (t, v, grade)):
                    raise ValueError(f"Non-finite drive-cycle value: {row}")
                if v < 0:
                    raise ValueError("speed_mps cannot be negative")
                if not (-30.0 <= grade <= 30.0):
                    raise ValueError(f"grade_pct {grade}% outside realistic roadway bounds [-30, 30]")
                points.append({"time_s": t, "speed_mps": v, "grade_pct": grade})
        if len(points) < 2:
            raise ValueError("Drive cycle needs at least two rows")
        if any(points[i+1]["time_s"] <= points[i]["time_s"] for i in range(len(points)-1)):
            raise ValueError("time_s must be strictly increasing")
        return points

    def _validate_cycle(self, cycle: list[dict[str, float]]) -> None:
        if not isinstance(cycle, list) or len(cycle) < 2:
            raise ValueError("Drive cycle must be a list containing at least two points")
        for i, pt in enumerate(cycle):
            t = pt.get("time_s")
            v = pt.get("speed_mps")
            g = pt.get("grade_pct", 0.0)
            if t is None or v is None:
                raise ValueError(f"Cycle point {i} missing time_s or speed_mps: {pt}")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (t, v, g)):
                raise ValueError(f"Non-finite or non-numeric value in cycle point {i}: {pt}")
            if v < 0:
                raise ValueError(f"speed_mps cannot be negative at point {i}: {v}")
            if v > 120.0:
                raise ValueError(f"speed_mps {v} m/s (>432 km/h) exceeds realistic automotive bounds")
            if not (-30.0 <= g <= 30.0):
                raise ValueError(f"grade_pct {g}% at point {i} exceeds realistic roadway bounds [-30, 30]")
            if i > 0:
                dt = t - cycle[i-1]["time_s"]
                if dt <= 0:
                    raise ValueError(f"time_s must be strictly increasing at point {i} (dt={dt})")
                if dt > 600.0:
                    raise ValueError(f"Time step {dt}s exceeds maximum allowable interval of 600s")

    def estimate(
        self,
        cycle: list[dict[str, float]],
        soh_pct: float = 100.0,
        ambient_c: float | None = None,
        hvac_power_kw: float | None = None,
        quality_flag: str | None = None,
        cell_id: str | None = None,
    ) -> dict[str, Any]:
        """Estimate EV driving range using interval-based mechanical work integration.

        Args:
            cycle: list of point dicts with 'time_s', 'speed_mps', optional 'grade_pct'
            soh_pct: battery state of health percentage (scale 1.0 to 120.0%)
            ambient_c: ambient temperature in Celsius (-50 to +60 C)
            hvac_power_kw: explicit HVAC load override (0.0 to 15.0 kW)
            quality_flag: optional data quality pedigree from Member 1 or 2
            cell_id: optional cell identifier
        """
        # Validate SOH
        if not isinstance(soh_pct, (int, float)) or not math.isfinite(soh_pct):
            raise ValueError("soh_pct must be a finite numeric value")
        if 0 < soh_pct <= 1.0:
            raise ValueError(
                f"soh_pct={soh_pct} appears to be a fraction (<= 1.0); "
                "expected percentage on 0-100 scale (e.g. 80.0 for 80% SOH)"
            )
        if soh_pct <= 0 or soh_pct > 120.0:
            raise ValueError(f"soh_pct must be in (1.0, 120.0]; received {soh_pct}")

        # Validate ambient temperature
        if ambient_c is not None:
            if not isinstance(ambient_c, (int, float)) or not math.isfinite(ambient_c):
                raise ValueError("ambient_c must be finite or None")
            if not (-50.0 <= ambient_c <= 60.0):
                raise ValueError(f"ambient_c={ambient_c} C outside realistic terrestrial limits [-50, 60]")

        # Validate HVAC load
        if hvac_power_kw is not None:
            if not isinstance(hvac_power_kw, (int, float)) or not math.isfinite(hvac_power_kw) or hvac_power_kw < 0:
                raise ValueError("hvac_power_kw must be finite and >= 0")
            if hvac_power_kw > 15.0:
                raise ValueError(f"hvac_power_kw={hvac_power_kw} kW exceeds realistic automotive HVAC capacity (15 kW)")

        # Validate drive cycle
        self._validate_cycle(cycle)

        # Compute dynamic environmental parameters
        air_density = self.air_density(ambient_c)
        hvac_kw, hvac_mode = self.empirical_hvac_kw(ambient_c, hvac_power_kw)

        c = self.c
        m_eff = c.vehicle_mass_kg * c.rotational_inertia_factor

        total_traction_energy_j = 0.0
        total_regen_energy_j = 0.0
        total_aux_energy_j = 0.0
        total_distance_m = 0.0
        total_time_s = 0.0

        for i in range(len(cycle) - 1):
            p0 = cycle[i]
            p1 = cycle[i + 1]

            t0, t1 = p0["time_s"], p1["time_s"]
            v0, v1 = p0["speed_mps"], p1["speed_mps"]
            g0 = p0.get("grade_pct", 0.0)
            g1 = p1.get("grade_pct", 0.0)

            dt = t1 - t0
            v_avg = 0.5 * (v0 + v1)
            ds = v_avg * dt
            grade_avg = 0.5 * (g0 + g1)
            theta = math.atan(grade_avg / 100.0)

            # 1. Exact kinetic energy change (analytical integral of m_eff * a * v dt)
            w_inertia = 0.5 * m_eff * (v1**2 - v0**2)

            # 2. Aerodynamic drag work (exact integral of 0.5 * rho * Cd * A * v(t)^3 dt for linear v(t))
            v_cubed_mean = 0.5 * (v0**2 + v1**2)
            f_aero_avg = 0.5 * air_density * c.drag_coefficient_cd * c.frontal_area_m2 * v_cubed_mean
            w_aero = f_aero_avg * ds

            # 3. Rolling resistance work (non-zero only while in motion)
            f_roll = (c.vehicle_mass_kg * self.G * c.rolling_resistance_coefficient_crr * math.cos(theta)) if v_avg > 0 else 0.0
            w_roll = f_roll * ds

            # 4. Road grade potential energy change
            f_grade = c.vehicle_mass_kg * self.G * math.sin(theta)
            w_grade = f_grade * ds

            # Net mechanical work required at wheels during interval
            w_wheel = w_inertia + w_aero + w_roll + w_grade

            if w_wheel >= 0:
                # Traction mode: battery delivers energy through drivetrain efficiency
                e_batt_traction = w_wheel / c.drivetrain_efficiency
                total_traction_energy_j += e_batt_traction
            else:
                # Braking mode: regenerative recovery with low-speed blending cutoff
                p_wheel_brake_kw = (-w_wheel) / (1000.0 * dt)
                if v_avg >= self.V_REGEN_CUTOFF_FULL_MPS:
                    blend_factor = 1.0
                elif v_avg <= self.V_REGEN_CUTOFF_LOW_MPS:
                    blend_factor = 0.0
                else:
                    blend_factor = (v_avg - self.V_REGEN_CUTOFF_LOW_MPS) / (self.V_REGEN_CUTOFF_FULL_MPS - self.V_REGEN_CUTOFF_LOW_MPS)

                p_regen_elec_kw = min(p_wheel_brake_kw * c.regen_efficiency * blend_factor, c.max_regen_power_kw)
                e_batt_regen = p_regen_elec_kw * 1000.0 * dt
                total_regen_energy_j += e_batt_regen

            # Auxiliary & HVAC energy consumed constantly over dt
            total_aux_energy_j += (c.auxiliary_power_kw + hvac_kw) * 1000.0 * dt
            total_distance_m += ds
            total_time_s += dt

        distance_km = total_distance_m / 1000.0
        if distance_km <= 0:
            raise ValueError("Drive cycle distance is zero; range cannot be extrapolated")

        # Convert Joules to kWh (1 kWh = 3.6e6 J)
        traction_kwh = total_traction_energy_j / 3.6e6
        regen_kwh = total_regen_energy_j / 3.6e6
        aux_hvac_kwh = total_aux_energy_j / 3.6e6
        net_cycle_energy_kwh = traction_kwh - regen_kwh + aux_hvac_kwh

        usable_energy_kwh = (
            c.nominal_pack_energy_kwh
            * c.usable_energy_fraction
            * (soh_pct / 100.0)
        )

        provenance_status = (c.provenance or {}).get("status")
        base_status = "ILLUSTRATIVE_UNVALIDATED" if provenance_status == "ILLUSTRATIVE_ASSUMPTIONS" else "REQUIRES_VALIDATION"

        if net_cycle_energy_kwh <= 0:
            # Sustained steep descent where net regeneration exceeds consumption
            consumption_kwh_per_km = net_cycle_energy_kwh / distance_km
            range_km = float("inf")
            quality_status = "NET_GENERATING_DESCENT"
            limitations = [
                "Net drive cycle energy is non-positive (net recharge from elevation drop or regenerative braking).",
                "Range is mathematically unbounded under perpetual descent conditions.",
                "SOH is approximated as a linear multiplier on usable pack energy.",
                "Validate against vehicle-specific consumption/range before making real-world claims."
            ]
        else:
            consumption_kwh_per_km = net_cycle_energy_kwh / distance_km
            range_km = usable_energy_kwh / consumption_kwh_per_km
            quality_status = base_status
            limitations = [
                "Range is extrapolated from this cycle's energy per distance.",
                "SOH is approximated as a linear multiplier on usable pack energy.",
                f"HVAC load mode: {hvac_mode} ({hvac_kw:.2f} kW).",
                "Validate against vehicle-specific consumption/range before making real-world claims."
            ]

        res = {
            "vehicle_name": c.vehicle_name,
            "distance_km": distance_km,
            "duration_s": total_time_s,
            "average_speed_kmh": (distance_km / (total_time_s / 3600.0)) if total_time_s > 0 else 0.0,
            "cycle_energy_kwh": net_cycle_energy_kwh,
            "traction_energy_kwh": traction_kwh,
            "regen_recovered_kwh": regen_kwh,
            "aux_hvac_energy_kwh": aux_hvac_kwh,
            "consumption_kwh_per_km": consumption_kwh_per_km,
            "consumption_kwh_per_100km": consumption_kwh_per_km * 100.0,
            "usable_energy_kwh": usable_energy_kwh,
            "soh_pct": soh_pct,
            "ambient_c": ambient_c,
            "air_density_kg_m3": air_density,
            "hvac_power_kw": hvac_kw,
            "hvac_mode": hvac_mode,
            "estimated_range_km": range_km,
            "quality_status": quality_status,
            "quality_flag": quality_flag,
            "cell_id": cell_id,
            "limitations": limitations
        }
        if soh_pct > 100.0:
            res["soh_provenance_note"] = "SOH exceeds 100%; allowable for fresh cells within manufacturing tolerances."
        return res

    def estimate_trajectory(
        self,
        cycle: list[dict[str, float]],
        soh_trajectory: list[float] | list[dict[str, Any]],
        ambient_c: float | None = None,
        hvac_power_kw: float | None = None,
        quality_flags: list[str] | None = None,
        cell_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate range decay across an SOH lifetime trajectory from Member 2.

        Args:
            cycle: drive cycle points
            soh_trajectory: list of SOH percentages or dicts with 'soh_pct'/'cycle_index'
            ambient_c: scenario ambient temperature
            hvac_power_kw: optional explicit HVAC load
            quality_flags: optional list of flags corresponding to each point
            cell_id: optional cell identifier
        """
        results: list[dict[str, Any]] = []
        for idx, item in enumerate(soh_trajectory):
            flag = quality_flags[idx] if (quality_flags and idx < len(quality_flags)) else None
            cid = cell_id
            cycle_idx = None
            if isinstance(item, dict):
                soh = float(item["soh_pct"])
                flag = item.get("quality_flag", flag)
                cid = item.get("cell_id", cid)
                cycle_idx = item.get("cycle_index", idx)
            else:
                soh = float(item)

            r = self.estimate(
                cycle=cycle,
                soh_pct=soh,
                ambient_c=ambient_c,
                hvac_power_kw=hvac_power_kw,
                quality_flag=flag,
                cell_id=cid,
            )
            if cycle_idx is not None:
                r["cycle_index"] = cycle_idx
            results.append(r)
        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline physics-based EV range estimator")
    parser.add_argument("--config", default="vehicle_config.json")
    parser.add_argument("--cycle", default="drive_cycles/example_cycle.csv")
    parser.add_argument("--soh-pct", type=float, default=100.0)
    parser.add_argument("--ambient-c", type=float, default=25.0)
    parser.add_argument("--hvac-kw", type=float, default=None)
    parser.add_argument("--quality-flag", type=str, default=None)
    args = parser.parse_args()

    cfg = VehicleConfig.from_json(args.config)
    calc = RangeCalculator(cfg)
    cycle = calc.load_cycle(args.cycle)
    result = calc.estimate(
        cycle,
        soh_pct=args.soh_pct,
        ambient_c=args.ambient_c,
        hvac_power_kw=args.hvac_kw,
        quality_flag=args.quality_flag,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
