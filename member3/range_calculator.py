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
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Any


def _as_finite_number(value: Any, name: str) -> float:
    """Return `value` as a float, rejecting booleans, non-numerics and non-finite values.

    `bool` is a subclass of `int` in Python, so an unguarded isinstance check accepts
    True as 1.0. Booleans are not engineering measurements and are rejected explicitly.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean: {value!r}")
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{name} must be a number: got {type(value).__name__} {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite (not NaN or infinity): got {value!r}")
    return number


# name -> (lower, lower_inclusive, upper, upper_inclusive, unit)
#
# These are deliberately wide physical envelopes for road vehicles, not design
# limits. Their job is to catch transcription and unit errors (mass in grams, Cd
# written as a percentage, regen power in watts), so nothing plausible is refused.
CONFIG_FIELD_LIMITS: dict[str, tuple[float, bool, float, bool, str]] = {
    "nominal_pack_energy_kwh": (0.0, False, 250.0, True, "kWh"),
    "usable_energy_fraction": (0.0, False, 1.0, True, "fraction"),
    "vehicle_mass_kg": (0.0, False, 20000.0, True, "kg"),
    "drag_coefficient_cd": (0.0, False, 2.0, True, "dimensionless"),
    "frontal_area_m2": (0.0, False, 20.0, True, "m2"),
    "rolling_resistance_coefficient_crr": (0.0, True, 0.05, True, "dimensionless"),
    "air_density_kg_m3": (0.0, False, 2.0, True, "kg/m3"),
    "drivetrain_efficiency": (0.0, False, 1.0, True, "fraction"),
    "regen_efficiency": (0.0, True, 1.0, True, "fraction"),
    "max_regen_power_kw": (0.0, True, 5000.0, True, "kW"),
    "auxiliary_power_kw": (0.0, True, 15.0, True, "kW"),
    "hvac_power_kw_default": (0.0, True, 15.0, True, "kW"),
    "rotational_inertia_factor": (1.0, True, 2.0, True, "dimensionless"),
    "rated_range_km_reference": (0.0, False, 3000.0, True, "km"),
    # --- Cabin thermal / HVAC block (Phase 3) --------------------------------
    # Every one of these is optional and only consulted when `hvac_model_mode`
    # selects a cabin-thermal mode. The shipped profiles leave them at their
    # defaults, and the defaults are ILLUSTRATIVE ENGINEERING PLACEHOLDERS: they
    # are not measured vehicle data, and they are not COP values prescribed by
    # EN 14511 either -- that standard fixes the A7 (7 degC outdoor air) heating
    # and A35 (35 degC outdoor air) cooling test duties, while the reference COPs,
    # their slopes and their clip bands below are our own assumptions.
    # See README "HVAC and Cabin Thermal Model".
    "cabin_target_temp_c": (-20.0, True, 45.0, True, "degC"),
    "cabin_thermal_capacitance_j_per_k": (0.0, False, 1.0e8, True, "J/K"),
    "cabin_ua_w_per_k": (0.0, True, 2000.0, True, "W/K"),
    "cabin_solar_load_w": (0.0, True, 3000.0, True, "W"),
    "cabin_occupancy_load_w": (0.0, True, 1500.0, True, "W"),
    "cabin_recovery_time_constant_s": (0.0, False, 1.0e7, True, "s"),
    "heat_pump_cop_reference": (1.0, True, 10.0, True, "dimensionless"),
    "heat_pump_cop_reference_temp_c": (-30.0, True, 30.0, True, "degC"),
    "heat_pump_cop_slope_per_k": (0.0, True, 1.0, True, "1/K"),
    "heat_pump_cop_min": (1.0, True, 10.0, True, "dimensionless"),
    "heat_pump_cop_max": (1.0, True, 10.0, True, "dimensionless"),
    "ptc_heating_efficiency": (0.0, False, 1.0, True, "fraction"),
    "ac_cooling_cop_reference": (1.0, True, 10.0, True, "dimensionless"),
    "ac_cooling_cop_reference_temp_c": (0.0, True, 60.0, True, "degC"),
    "ac_cooling_cop_slope_per_k": (0.0, True, 1.0, True, "1/K"),
    "ac_cooling_cop_min": (1.0, True, 10.0, True, "dimensionless"),
    "ac_cooling_cop_max": (1.0, True, 10.0, True, "dimensionless"),
    "max_heating_electrical_kw": (0.0, True, 20.0, True, "kW"),
    "max_cooling_electrical_kw": (0.0, True, 20.0, True, "kW"),
}

# HVAC operating modes. `EMPIRICAL` is the legacy whole-cycle comfort curve and
# remains the default, so an existing configuration or call is never silently
# reinterpreted. The other three select the first-order cabin thermal model with
# a single, explicitly chosen actuator.
HVAC_MODES = ("EMPIRICAL", "HEAT_PUMP", "PTC", "AC_COOLING")
CABIN_HVAC_MODES = ("HEAT_PUMP", "PTC", "AC_COOLING")

# Numeric fields whose documented default is "not supplied" (None). Every other
# numeric field must carry a value, so None is an error rather than an absence.
#   rated_range_km_reference        - optional certified figure
#   cabin_initial_temp_c            - None means "cabin has soaked to ambient"
#   cabin_recovery_time_constant_s  - None means "use the plant time constant C/UA"
NULLABLE_CONFIG_FIELDS = frozenset({
    "rated_range_km_reference",
    "cabin_initial_temp_c",
    "cabin_recovery_time_constant_s",
})


def _range_text(name: str, limit: tuple[float, bool, float, bool, str]) -> str:
    lo, lo_inc, hi, hi_inc, unit = limit
    return (
        f"{name} must be in "
        f"{'[' if lo_inc else '('}{lo}, {hi}{']' if hi_inc else ')'} {unit}"
    )


@dataclass
class VehicleConfig:
    """Vehicle parameters and powertrain specifications for range estimation.

    Attributes:
        vehicle_name: Human-readable vehicle identifier.
        nominal_pack_energy_kwh: Total installed battery pack nameplate capacity (kWh).
        usable_energy_fraction: Usable battery fraction (0, 1] accounting for top/bottom buffers.
        vehicle_mass_kg: Total vehicle mass (kg), including passengers/cargo.
        drag_coefficient_cd: Aerodynamic drag coefficient Cd (> 0).
        frontal_area_m2: Vehicle frontal cross-sectional area (m2) (> 0).
        rolling_resistance_coefficient_crr: Rolling resistance coefficient Crr (>= 0).
        air_density_kg_m3: Reference ambient dry air density (kg/m3) at sea level (> 0).
        drivetrain_efficiency: Combined inverter, motor, and gearbox efficiency (0, 1].
        regen_efficiency: Regenerative electrical recovery efficiency (0, 1].
        max_regen_power_kw: Maximum electrical regenerative charging power (kW) accepted by
            the battery pack and inverter (>= 0). Caps battery recovery rate (P_regen_elec <= max_regen_power_kw).
        auxiliary_power_kw: Constant baseline auxiliary electrical power (kW) (>= 0).
        hvac_power_kw_default: Baseline cabin ventilation/HVAC standby power (kW) during comfort
            deadband [18 C, 22 C] (>= 0, <= 15.0 kW).
        rotational_inertia_factor: Rotational inertia mass factor delta >= 1.0 (default 1.05 = +5%).
        rated_range_km_reference: Optional certified reference range (MIDC/WLTP/EPA) in km.
        hvac_model_mode: One of EMPIRICAL (legacy comfort curve, default), HEAT_PUMP, PTC or
            AC_COOLING. The last three use the first-order cabin thermal model.
        cabin_target_temp_c: Cabin setpoint the HVAC works towards (degC).
        cabin_initial_temp_c: Cabin temperature at t=0 (degC); None means the cabin has
            soaked to the ambient temperature, i.e. no pre-conditioning.
        cabin_thermal_capacitance_j_per_k: Lumped cabin thermal capacitance C (J/K).
        cabin_ua_w_per_k: Cabin-to-ambient overall heat-transfer conductance UA (W/K).
        cabin_solar_load_w: Solar gain into the cabin (W), >= 0; 0 disables the term.
        cabin_occupancy_load_w: Metabolic/internal gain from occupants (W), >= 0; 0 disables it.
        cabin_recovery_time_constant_s: Time constant the controller uses to pull the cabin
            to setpoint (s). None uses the plant time constant C/UA. Values shorter than
            C/UA are rejected: they would make the discrete zero-order-hold loop oscillate.
        heat_pump_cop_reference: Heating COP at `heat_pump_cop_reference_temp_c` (dimensionless).
        heat_pump_cop_reference_temp_c: Rating point of that COP (degC); the default 7 degC is the
            air side of the EN 14511 A7 nominal heating duty. In its air-to-water form that duty is
            written A7/W35, where W35 fixes a 35 degC water supply temperature; a cabin heat pump
            delivers air, not water, so only the A7 air side applies and W35 does not. EN 14511
            names the test condition, not the COP, so the reference value is an assumption.
        heat_pump_cop_slope_per_k: COP lost per degC below the rating point (1/K); 0.0
            declares a constant-COP simplified approximation.
        heat_pump_cop_min: Floor for the heating COP (dimensionless, >= 1.0).
        heat_pump_cop_max: Ceiling for the heating COP (dimensionless).
        ptc_heating_efficiency: Fraction of PTC electrical power converted to cabin heat
            (0, 1]; above 1 would create energy and is rejected.
        ac_cooling_cop_reference: Cooling COP at `ac_cooling_cop_reference_temp_c`.
        ac_cooling_cop_reference_temp_c: Rating point of the cooling COP (degC); the default
            35 degC is the EN 14511 A35 nominal cooling duty (35 degC outdoor dry-bulb air),
            a test condition rather than a measured vehicle value or a standard COP.
        ac_cooling_cop_slope_per_k: COP lost per degC above that rating point (1/K).
        ac_cooling_cop_min: Floor for the cooling COP (dimensionless, >= 1.0).
        ac_cooling_cop_max: Ceiling for the cooling COP (dimensionless).
        max_heating_electrical_kw: Electrical power drawn by the heating source at full
            output (kW); thermal delivery is this times COP (heat pump) or times
            ptc_heating_efficiency (PTC).
        max_cooling_electrical_kw: Electrical power drawn by the compressor at full
            cooling output (kW).
        provenance: Optional metadata documenting data sources, assumptions, and validation state.
    """
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
    # ---------------------------------------------------------------------
    # Cabin thermal model (Phase 3). Optional and unused unless `hvac_model_mode`
    # selects a cabin mode, so every existing profile keeps its current behaviour.
    # The defaults are illustrative engineering placeholders for a compact-EV class
    # cabin; they are NOT measured vehicle data and must be calibrated per vehicle
    # before their results are quoted as a real range.
    # ---------------------------------------------------------------------
    hvac_model_mode: str = "EMPIRICAL"
    cabin_target_temp_c: float = 22.0
    cabin_initial_temp_c: float | None = None
    cabin_thermal_capacitance_j_per_k: float = 1_000_000.0
    cabin_ua_w_per_k: float = 150.0
    cabin_solar_load_w: float = 0.0
    cabin_occupancy_load_w: float = 0.0
    cabin_recovery_time_constant_s: float | None = None
    heat_pump_cop_reference: float = 3.0
    heat_pump_cop_reference_temp_c: float = 7.0
    heat_pump_cop_slope_per_k: float = 0.05
    heat_pump_cop_min: float = 1.2
    heat_pump_cop_max: float = 5.0
    ptc_heating_efficiency: float = 0.95
    ac_cooling_cop_reference: float = 3.5
    ac_cooling_cop_reference_temp_c: float = 35.0
    ac_cooling_cop_slope_per_k: float = 0.05
    ac_cooling_cop_min: float = 1.5
    ac_cooling_cop_max: float = 6.0
    max_heating_electrical_kw: float = 8.0
    max_cooling_electrical_kw: float = 6.0
    provenance: dict[str, Any] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "VehicleConfig":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError(f"Vehicle configuration JSON at {path} must be a dictionary object")
        allowed = set(cls.__dataclass_fields__.keys())
        unknown = set(raw.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Unknown field(s) in vehicle config JSON '{path}': {sorted(unknown)}. "
                f"Allowed fields are: {sorted(allowed)}"
            )
        required = sorted(
            name for name, field in cls.__dataclass_fields__.items()
            if field.default is MISSING and field.default_factory is MISSING
        )
        missing = sorted(name for name in required if name not in raw)
        if missing:
            raise ValueError(
                f"Missing required field(s) in vehicle config JSON '{path}': {missing}. "
                f"Required fields are: {required}. Optional fields (have defaults): "
                f"{sorted(allowed - set(required))}"
            )
        try:
            return cls(**raw)
        except ValueError as e:
            raise ValueError(f"Vehicle config JSON '{path}' is invalid: {e}") from e

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Check every field's type, finiteness and physical range; mutate nothing.

        Booleans are rejected (see `_as_finite_number`), so `True` can never act as
        1.0 kg or 1.0 kW. Invalid values raise instead of being replaced by defaults:
        a configuration is never silently repaired.
        """
        if not isinstance(self.vehicle_name, str) or not self.vehicle_name.strip():
            raise ValueError(
                f"vehicle_name must be a non-empty string, got {self.vehicle_name!r}"
            )
        for name, limit in CONFIG_FIELD_LIMITS.items():
            value = getattr(self, name)
            if value is None and name in NULLABLE_CONFIG_FIELDS:
                continue
            lo, lo_inc, hi, _hi_inc, _unit = limit
            number = _as_finite_number(value, name)
            below = number < lo if lo_inc else number <= lo
            if below or number > hi:
                raise ValueError(f"{_range_text(name, limit)}; got {value!r}")
        self._validate_hvac()
        self._validate_provenance()

    def _validate_hvac(self) -> None:
        """Validate the HVAC mode and the cross-field rules of the cabin model.

        These are relationships no single-field envelope can express: a COP band
        that is inverted, a rating point outside its own clip band, a cabin soak
        temperature outside physical limits, or a recovery time constant shorter
        than the plant time constant it drives.
        """
        mode = self.hvac_model_mode
        if not isinstance(mode, str) or mode not in HVAC_MODES:
            raise ValueError(
                f"hvac_model_mode must be one of {list(HVAC_MODES)}, got {mode!r}"
            )
        if self.cabin_initial_temp_c is not None:
            soak = _as_finite_number(self.cabin_initial_temp_c, "cabin_initial_temp_c")
            if not (-50.0 <= soak <= 80.0):
                raise ValueError(
                    f"cabin_initial_temp_c must be in [-50.0, 80.0] degC; got {soak!r}. "
                    "The upper bound allows a sun-soaked cabin to exceed ambient temperature."
                )
        for prefix in ("heat_pump_cop", "ac_cooling_cop"):
            lo = getattr(self, f"{prefix}_min")
            hi = getattr(self, f"{prefix}_max")
            ref = getattr(self, f"{prefix}_reference")
            if lo > hi:
                raise ValueError(
                    f"{prefix}_min={lo} exceeds {prefix}_max={hi}; the COP clip band "
                    "must be ordered"
                )
            if not (lo <= ref <= hi):
                raise ValueError(
                    f"{prefix}_reference={ref} lies outside its own clip band "
                    f"[{lo}, {hi}]; the rated COP would never be reachable"
                )
            if getattr(self, f"{prefix}_slope_per_k") < 0.0:
                raise ValueError(
                    f"{prefix}_slope_per_k must be >= 0: this model degrades the COP "
                    f"monotonically away from its rating point, and a negative slope "
                    f"claims a COP that improves without limit"
                )
        if self.ptc_heating_efficiency <= 0.0:
            raise ValueError("ptc_heating_efficiency must be > 0 (a dead heater cannot deliver heat)")
        tau_rec = self.cabin_recovery_time_constant_s
        ua = self.cabin_ua_w_per_k
        if tau_rec is not None and ua > 0.0:
            tau_plant = self.cabin_thermal_capacitance_j_per_k / ua
            if tau_rec < tau_plant:
                raise ValueError(
                    f"cabin_recovery_time_constant_s={tau_rec} s is shorter than the cabin "
                    f"plant time constant C/UA={tau_plant:.1f} s. With a zero-order-hold "
                    "actuator the discrete thermal loop would oscillate instead of settling; "
                    "raise the recovery constant (or omit it to use C/UA)."
                )

    def _validate_provenance(self) -> None:
        """Validate the structure of the provenance metadata block, if present."""
        provenance = self.provenance
        if provenance is None:
            return
        if not isinstance(provenance, dict):
            raise ValueError(
                f"provenance must be a dictionary of metadata, got {type(provenance).__name__}"
            )
        for key in provenance:
            if not isinstance(key, str):
                raise ValueError(f"provenance keys must be strings, got {key!r}")
        status = provenance.get("status")
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError(f"provenance.status must be a non-empty string, got {status!r}")
        for text_key in ("note", "manufacturer_source", "retrieved_date", "validated_by"):
            value = provenance.get(text_key)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"provenance.{text_key} must be a string when present, "
                    f"got {type(value).__name__}"
                )
        sourced = provenance.get("sourced_fields")
        if sourced is not None:
            if not isinstance(sourced, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in sourced.items()
            ):
                raise ValueError(
                    "provenance.sourced_fields must map config field names to source strings"
                )
            unknown_fields = sorted(set(sourced) - set(self.__dataclass_fields__))
            if unknown_fields:
                raise ValueError(
                    f"provenance.sourced_fields references unknown config field(s): {unknown_fields}"
                )
        assumptions = provenance.get("assumptions")
        if assumptions is not None and (
            not isinstance(assumptions, list)
            or not all(isinstance(a, str) for a in assumptions)
        ):
            raise ValueError("provenance.assumptions must be a list of strings when present")


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
    # International Standard Atmosphere, troposphere (altitude range where it is defined).
    ISA_TEMPERATURE_SEA_LEVEL_K = 288.15
    ISA_LAPSE_RATE_K_PER_M = 0.0065
    ABSOLUTE_ZERO_C = -273.15
    MIN_ALTITUDE_M = -500.0     # Dead Sea shore; below this the model has no road relevance
    MAX_ALTITUDE_M = 11000.0    # ISA tropopause; the pressure law changes above it
    MIN_PRESSURE_PA = 20000.0   # ~17.6 km equivalent; far below any inhabited road
    MAX_PRESSURE_PA = 120000.0  # above the 108.4 kPa world-record sea-level high
    # Aerodynamic inputs are one-dimensional along the direction of travel.
    MAX_WIND_SPEED_MPS = 60.0   # 216 km/h: above any wind a vehicle can be driven in

    @classmethod
    def _validate_altitude(cls, altitude_m: float) -> float:
        altitude = _as_finite_number(altitude_m, "altitude_m")
        if not (cls.MIN_ALTITUDE_M <= altitude <= cls.MAX_ALTITUDE_M):
            raise ValueError(
                f"altitude_m={altitude_m!r} m is outside the supported range "
                f"[{cls.MIN_ALTITUDE_M}, {cls.MAX_ALTITUDE_M}] m: the ISA troposphere "
                "pressure law is only defined up to the tropopause (11 km) and the model "
                "is not intended for locations below -500 m."
            )
        return altitude

    @classmethod
    def atmospheric_pressure_pa(cls, altitude_m: float) -> float:
        """Standard-atmosphere pressure at `altitude_m` above sea level, in Pa.

        ISA troposphere relation:
            P(h) = P0 * (1 - L*h/T0) ** (g0 / (R_air * L)),  exponent ~= 5.2559
        valid for -500 m to 11 km (the tropopause), where temperature falls linearly.
        """
        altitude = cls._validate_altitude(altitude_m)
        base = 1.0 - cls.ISA_LAPSE_RATE_K_PER_M * altitude / cls.ISA_TEMPERATURE_SEA_LEVEL_K
        exponent = cls.G / (cls.R_SPEC_AIR * cls.ISA_LAPSE_RATE_K_PER_M)
        return cls.P_ATM_PA * (base ** exponent)

    @classmethod
    def isa_temperature_k(cls, altitude_m: float) -> float:
        """Standard-atmosphere temperature at `altitude_m`, in kelvin."""
        altitude = cls._validate_altitude(altitude_m)
        return cls.ISA_TEMPERATURE_SEA_LEVEL_K - cls.ISA_LAPSE_RATE_K_PER_M * altitude

    @classmethod
    def _validate_pressure(cls, pressure_pa: float) -> float:
        pressure = _as_finite_number(pressure_pa, "pressure_pa")
        if not (cls.MIN_PRESSURE_PA <= pressure <= cls.MAX_PRESSURE_PA):
            raise ValueError(
                f"pressure_pa={pressure_pa!r} is outside the supported range "
                f"[{cls.MIN_PRESSURE_PA}, {cls.MAX_PRESSURE_PA}] Pa "
                "(surface pressures range from about 87 kPa in intense low-pressure "
                "systems to 108.4 kPa for the world-record high)."
            )
        return pressure

    def resolve_environment(
        self,
        ambient_c: float | None = None,
        pressure_pa: float | None = None,
        altitude_m: float | None = None,
    ) -> dict[str, Any]:
        """Resolve dry-air density from temperature, pressure and altitude.

        Air density always comes from the ideal gas law, rho = P / (R_air * T).

        Precedence (each quantity has exactly one source, so altitude is never applied
        twice):
          * pressure: explicit `pressure_pa` wins; otherwise, if `altitude_m` is given,
            ISA pressure at that altitude; otherwise standard sea-level pressure
            (101325 Pa), matching the previous behaviour.
          * temperature: `ambient_c` wins and is used exactly as supplied - a measured
            ambient is ground temperature, so the ISA lapse rate is NOT applied on top
            of it. Only when `ambient_c` is omitted does ISA temperature at the
            altitude (sea level if no altitude) supply T.
          * when ambient_c, pressure_pa and altitude_m are all omitted the configured
            `air_density_kg_m3` is returned unchanged, preserving legacy behaviour.
        """
        if ambient_c is not None:
            ambient = _as_finite_number(ambient_c, "ambient_c")
            temperature_k = ambient - self.ABSOLUTE_ZERO_C
            if temperature_k <= 0.0:
                raise ValueError(
                    f"ambient_c={ambient_c} C is at or below absolute zero "
                    f"({self.ABSOLUTE_ZERO_C} C)"
                )
            temperature_source = "AMBIENT_SUPPLIED"
        else:
            altitude_for_isa = (
                self._validate_altitude(altitude_m) if altitude_m is not None else 0.0
            )
            temperature_k = self.isa_temperature_k(altitude_for_isa)
            temperature_source = (
                "ISA_STANDARD" if altitude_m is None else "ISA_AT_ALTITUDE"
            )

        if pressure_pa is not None:
            pressure = self._validate_pressure(pressure_pa)
            pressure_source = "PRESSURE_SUPPLIED"
        elif altitude_m is not None:
            pressure = self.atmospheric_pressure_pa(altitude_m)
            pressure_source = "ISA_AT_ALTITUDE"
        else:
            pressure = self.P_ATM_PA
            pressure_source = "ISA_SEA_LEVEL"

        legacy_reference_only = ambient_c is None and pressure_pa is None and altitude_m is None
        density = (
            self.c.air_density_kg_m3
            if legacy_reference_only
            else pressure / (self.R_SPEC_AIR * temperature_k)
        )
        return {
            "air_density_kg_m3": density,
            "atmospheric_pressure_pa": pressure,
            "temperature_k": temperature_k,
            "altitude_m": None if altitude_m is None else self._validate_altitude(altitude_m),
            "pressure_source": pressure_source,
            "temperature_source": temperature_source,
            "air_density_source": (
                "CONFIG_REFERENCE" if legacy_reference_only else "IDEAL_GAS_LAW"
            ),
        }

    def air_density(
        self,
        ambient_c: float | None,
        pressure_pa: float | None = None,
        altitude_m: float | None = None,
    ) -> float:
        """Dry air density from the ideal gas law, rho = P / (R_air * T).

        With only `ambient_c` this is the historical behaviour: standard sea-level
        pressure (101325 Pa) at the supplied temperature, or the configured reference
        density when `ambient_c` is None. `pressure_pa` and `altitude_m` follow the
        precedence documented in `resolve_environment`.
        """
        return self.resolve_environment(
            ambient_c, pressure_pa=pressure_pa, altitude_m=altitude_m
        )["air_density_kg_m3"]

    def __init__(self, config: VehicleConfig):
        self.c = config
        self._validate_config()

    def _validate_config(self) -> None:
        self.c.validate()

    def regen_blend_at_speed(self, v: float) -> float:
        """Pointwise regenerative blending availability beta(v).

        beta = 0 below V_REGEN_CUTOFF_LOW_MPS, 1 above V_REGEN_CUTOFF_FULL_MPS,
        linear in between.
        """
        if v <= self.V_REGEN_CUTOFF_LOW_MPS:
            return 0.0
        if v >= self.V_REGEN_CUTOFF_FULL_MPS:
            return 1.0
        return (v - self.V_REGEN_CUTOFF_LOW_MPS) / (
            self.V_REGEN_CUTOFF_FULL_MPS - self.V_REGEN_CUTOFF_LOW_MPS
        )

    def _blend_affine_for_speed_range(self, v_lo: float, v_hi: float) -> tuple[float, float]:
        """Return (alpha, gamma) with beta(v) = alpha * v + gamma valid across [v_lo, v_hi].

        The caller must have split the interval at the blend thresholds so the whole
        speed range lies inside a single blend region.
        """
        v_mid = 0.5 * (v_lo + v_hi)
        if v_mid <= self.V_REGEN_CUTOFF_LOW_MPS:
            return 0.0, 0.0
        if v_mid >= self.V_REGEN_CUTOFF_FULL_MPS:
            return 0.0, 1.0
        span = self.V_REGEN_CUTOFF_FULL_MPS - self.V_REGEN_CUTOFF_LOW_MPS
        return 1.0 / span, -self.V_REGEN_CUTOFF_LOW_MPS / span

    def effective_regen_blend_factor(self, v0: float, v1: float) -> float:
        """Kinetic-energy-weighted mean of beta(v) over the speed change v0 -> v1.

        LEGACY DIAGNOSTIC ONLY. Retained because it is exact for braking whose
        power is purely inertial (P proportional to v, no aero/roll/grade terms),
        and it is useful for reporting the mean blend availability of an interval.
        It is NOT used by estimate(): applying it to a whole interval's net negative
        wheel work mis-weights aerodynamic, rolling and grade braking power, and it
        cannot represent instantaneous charge-power limiting. See
        `_integrate_wheel_power_interval` for the integrated model actually used.
        """
        if v0 == v1:
            return self.regen_blend_at_speed(v0)

        v_low = self.V_REGEN_CUTOFF_LOW_MPS
        v_full = self.V_REGEN_CUTOFF_FULL_MPS
        delta_v = v_full - v_low

        def _integral_beta_v(v: float) -> float:
            if v <= v_low:
                return 0.0
            elif v <= v_full:
                return (1.0 / delta_v) * ((v**3) / 3.0 - (v_low * (v**2)) / 2.0 + (v_low**3) / 6.0)
            else:
                i_full = (1.0 / delta_v) * ((v_full**3) / 3.0 - (v_low * (v_full**2)) / 2.0 + (v_low**3) / 6.0)
                return i_full + 0.5 * (v**2 - v_full**2)

        num = _integral_beta_v(v1) - _integral_beta_v(v0)
        den = 0.5 * (v1**2 - v0**2)
        blend = num / den
        return max(0.0, min(1.0, blend))

    @staticmethod
    def _speed_moment(n: int, v_start: float, v_end: float, dt: float) -> float:
        """Exact integral of v(t)**n dt for v(t) linear from v_start to v_end over dt."""
        if dt <= 0.0:
            return 0.0
        if v_start == v_end:
            return dt * (v_start ** n)
        return dt * (v_end ** (n + 1) - v_start ** (n + 1)) / ((n + 1) * (v_end - v_start))

    @staticmethod
    def _cbrt(x: float) -> float:
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

    @classmethod
    def _cubic_real_roots(cls, a: float, b: float, c: float, d: float) -> list[float]:
        """Real roots of a*x^3 + b*x^2 + c*x + d = 0 (closed form, stdlib only)."""
        eps = 1e-15
        if abs(a) < eps:
            if abs(b) < eps:
                if abs(c) < eps:
                    return []
                return [-d / c]
            disc = c * c - 4.0 * b * d
            if disc < 0.0:
                return []
            s = math.sqrt(disc)
            return sorted([(-c + s) / (2.0 * b), (-c - s) / (2.0 * b)])

        b_, c_, d_ = b / a, c / a, d / a
        p = c_ - b_ * b_ / 3.0
        q = 2.0 * b_ ** 3 / 27.0 - b_ * c_ / 3.0 + d_
        shift = -b_ / 3.0
        disc = (q / 2.0) ** 2 + (p / 3.0) ** 3

        if disc > eps:
            half_q = -q / 2.0
            root = math.sqrt(disc)
            return [cls._cbrt(half_q + root) + cls._cbrt(half_q - root) + shift]
        if abs(p) < eps and abs(q) < eps:
            return [shift]
        if abs(disc) <= eps:
            u = cls._cbrt(-q / 2.0)
            return sorted([2.0 * u + shift, -u + shift])
        m_ = 2.0 * math.sqrt(-p / 3.0)
        cos_arg = max(-1.0, min(1.0, (3.0 * q) / (2.0 * p) * math.sqrt(-3.0 / p)))
        theta = math.acos(cos_arg)
        return sorted(m_ * math.cos((theta - 2.0 * math.pi * k) / 3.0) + shift for k in range(3))

    @staticmethod
    def _force_coefficients(
        linear_coeff_n: float,
        cubic_coeff_n: float,
        headwind_mps: float,
        v_mid: float,
    ) -> tuple[float, float, float]:
        """Return (L, Q, C) with P_wheel(v) = L*v + Q*v**2 + C*v**3 around `v_mid`.

        With wind, aero power is k*u*|u|*v where u = v + headwind is the relative air
        speed and v is the ground speed. On one side of u = 0 the absolute value is a
        constant sign s, so k*s*v*(v + w)**2 expands into the quadratic-times-v form.
        """
        if headwind_mps == 0.0:
            return linear_coeff_n, 0.0, cubic_coeff_n
        s = 1.0 if (v_mid + headwind_mps) >= 0.0 else -1.0
        return (
            linear_coeff_n + s * cubic_coeff_n * headwind_mps ** 2,
            2.0 * s * cubic_coeff_n * headwind_mps,
            s * cubic_coeff_n,
        )

    def _integrate_wheel_power_interval(
        self,
        v0: float,
        v1: float,
        dt: float,
        linear_coeff_n: float,
        cubic_coeff_n: float,
        headwind_mps: float = 0.0,
    ) -> dict[str, float]:
        """Integrate wheel power over one interval with physical event splitting.

        Wheel power for the interval is modelled as
            P_wheel(v) = linear_coeff_n * v + cubic_coeff_n * v**3      [W]
        where linear_coeff_n = m_eff * a + m*g*(Crr*cos(theta) + sin(theta)) and
        cubic_coeff_n = 0.5 * rho * Cd * A. Speed is linear in time inside the
        interval, so integrating in v-space is exact.

        With `headwind_mps` != 0 the aerodynamic term uses the relative air speed
        u = v + headwind_mps (positive = headwind), so the drag force is
        k*u*|u| and its power is that force times the GROUND speed v. The interval is
        then split at u = 0 as well, and inside each side of that split
            P_wheel(v) = L*v + Q*v**2 + C*v**3,
        which is why the same velocity moments stay exact. headwind_mps = 0 leaves the
        arithmetic of the no-wind model untouched.

        The interval is split at every event that changes the integrand's form:
          * beta(v) blend thresholds (0.5 and 2.0 m/s),
          * the zero of relative air speed (tailwind faster than the vehicle),
          * wheel-power sign changes (traction <-> braking),
          * crossings of the electrical charge-power limit max_regen_power_kw,
            located from the exact critical points of the quartic regen-power
            polynomial, then bisected to machine precision.

        On each resulting sub-interval the integrals are evaluated analytically.
        Returns wheel-side mechanical and battery-side electrical energies in joules.
        """
        eta_regen = self.c.regen_efficiency
        p_max_w = max(0.0, self.c.max_regen_power_kw) * 1000.0

        traction_j = 0.0
        braking_mech_j = 0.0
        regen_mech_j = 0.0
        regen_elec_j = 0.0

        if dt <= 0.0:
            return {
                "traction_mech_j": 0.0,
                "braking_mech_j": 0.0,
                "regen_mech_j": 0.0,
                "regen_elec_j": 0.0,
            }

        accel = (v1 - v0) / dt
        low = self.V_REGEN_CUTOFF_LOW_MPS
        full = self.V_REGEN_CUTOFF_FULL_MPS

        # 1. Blend-threshold and sign-change events, expressed as instants in time.
        breakpoints: set[float] = {0.0, dt}

        def add_speed_crossing(v_target: float) -> None:
            if accel == 0.0:
                return
            t = (v_target - v0) / accel
            if 0.0 < t < dt:
                breakpoints.add(t)

        for v_thr in (low, full):
            if min(v0, v1) < v_thr < max(v0, v1):
                add_speed_crossing(v_thr)
        v_span = (min(v0, v1), max(v0, v1))
        if cubic_coeff_n > 0.0 and linear_coeff_n < 0.0:
            v_zero = math.sqrt(-linear_coeff_n / cubic_coeff_n)
            if v_span[0] < v_zero < v_span[1]:
                add_speed_crossing(v_zero)
        if headwind_mps != 0.0:
            # Relative air speed changes sign at v = -headwind (tailwind only): on the
            # other side the aerodynamic term turns from drag into thrust.
            v_rel_zero = -headwind_mps
            if v_span[0] < v_rel_zero < v_span[1]:
                add_speed_crossing(v_rel_zero)
            for sign in (1.0, -1.0):
                lin, quad, cub = self._force_coefficients(
                    linear_coeff_n, cubic_coeff_n, headwind_mps,
                    v_rel_zero + sign,
                )
                for root in self._cubic_real_roots(0.0, cub, quad, lin):
                    if not v_span[0] < root < v_span[1]:
                        continue
                    if (root + headwind_mps >= 0.0) == (sign > 0.0):
                        add_speed_crossing(root)

        # 2. Charge-power-limit crossings inside each blend/sign sub-interval.
        times = sorted(breakpoints)
        pieces: list[tuple[float, float, bool]] = []  # (t_start, t_end, is_capped)
        for t_start, t_end in zip(times[:-1], times[1:]):
            va = v0 + accel * t_start
            vb = v0 + accel * t_end
            v_lo, v_hi = min(va, vb), max(va, vb)
            alpha, gamma = self._blend_affine_for_speed_range(v_lo, v_hi)
            lin, quad, cub = self._force_coefficients(
                linear_coeff_n, cubic_coeff_n, headwind_mps, 0.5 * (va + vb)
            )

            # Unconstrained electrical braking power as a quartic in v:
            #   P_regen(v) = -eta * (alpha*v + gamma) * (lin*v + quad*v**2 + cub*v**3)
            c4 = -eta_regen * alpha * cub
            c3 = -eta_regen * gamma * cub
            c2 = -eta_regen * alpha * lin
            c1 = -eta_regen * gamma * lin
            if quad != 0.0:
                c3 += -eta_regen * alpha * quad
                c2 += -eta_regen * gamma * quad

            def regen_power(v: float) -> float:
                return v * (c1 + v * (c2 + v * (c3 + v * c4)))

            cut_speeds: list[float] = []
            if accel != 0.0 and eta_regen > 0.0 and v_hi > v_lo and (alpha != 0.0 or gamma != 0.0):
                critical = [
                    r for r in self._cubic_real_roots(4.0 * c4, 3.0 * c3, 2.0 * c2, c1)
                    if v_lo < r < v_hi
                ]
                brackets = sorted([v_lo] + critical + [v_hi])
                for v_a, v_b in zip(brackets[:-1], brackets[1:]):
                    f_a = regen_power(v_a) - p_max_w
                    if f_a == 0.0:
                        cut_speeds.append(v_a)
                        continue
                    f_b = regen_power(v_b) - p_max_w
                    if f_a * f_b < 0.0:
                        lo, hi = v_a, v_b
                        for _ in range(80):
                            mid = 0.5 * (lo + hi)
                            if (regen_power(lo) - p_max_w) * (regen_power(mid) - p_max_w) <= 0.0:
                                hi = mid
                            else:
                                lo = mid
                        cut_speeds.append(0.5 * (lo + hi))

            cuts = {t_start, t_end}
            for v_cut in cut_speeds:
                t_cut = t_start + (v_cut - va) / accel
                if t_start < t_cut < t_end:
                    cuts.add(t_cut)

            ordered = sorted(cuts)
            for ta, tb in zip(ordered[:-1], ordered[1:]):
                if tb <= ta:
                    continue
                v_mid = v0 + accel * (0.5 * (ta + tb))
                pieces.append((ta, tb, regen_power(v_mid) > p_max_w))

        # 3. Analytic integration over every final sub-interval.
        for ta, tb, capped in pieces:
            seg_dt = tb - ta
            va = v0 + accel * ta
            vb = v0 + accel * tb
            m1 = self._speed_moment(1, va, vb, seg_dt)
            m2 = self._speed_moment(2, va, vb, seg_dt)
            m3 = self._speed_moment(3, va, vb, seg_dt)
            m4 = self._speed_moment(4, va, vb, seg_dt)

            lin, quad, cub = self._force_coefficients(
                linear_coeff_n, cubic_coeff_n, headwind_mps, 0.5 * (va + vb)
            )
            work_j = lin * m1 + cub * m3
            if quad != 0.0:
                work_j += quad * m2
            if work_j >= 0.0:
                traction_j += work_j
                continue

            braking_mech_j += -work_j
            alpha, gamma = self._blend_affine_for_speed_range(min(va, vb), max(va, vb))
            # integral of beta(v) * P_wheel(v) dt, using beta = alpha*v + gamma
            beta_weighted_work = (
                alpha * lin * m2
                + alpha * cub * m4
                + gamma * lin * m1
                + gamma * cub * m3
            )
            if quad != 0.0:
                beta_weighted_work += alpha * quad * m3 + gamma * quad * m2
            recovered_mech_j = -beta_weighted_work
            if eta_regen <= 0.0:
                continue
            if capped:
                elec_j = p_max_w * seg_dt
                regen_mech_j += elec_j / eta_regen
            else:
                elec_j = eta_regen * recovered_mech_j
                regen_mech_j += recovered_mech_j
            regen_elec_j += elec_j

        return {
            "traction_mech_j": traction_j,
            "braking_mech_j": braking_mech_j,
            "regen_mech_j": regen_mech_j,
            "regen_elec_j": regen_elec_j,
        }

    def empirical_hvac_kw(self, ambient_c: float | None, override_kw: float | None) -> tuple[float, str]:
        """Compute HVAC electrical demand from ambient temperature or user override."""
        if override_kw is not None:
            return override_kw, "MANUAL_OVERRIDE"
        if ambient_c is None:
            return self.c.hvac_power_kw_default, "CONFIG_DEFAULT"

        # Empirical piecewise automotive thermal comfort curve
        base_kw = self.c.hvac_power_kw_default
        if 18.0 <= ambient_c <= 22.0:
            return base_kw, "COMFORT_DEADBAND_VENTILATION"
        elif ambient_c < 18.0:
            delta_t = 18.0 - ambient_c
            # Heating ramp: baseline + 0.18 kW/C up to 6.0 kW nominal cap (or base_kw if higher), max 15 kW
            cap_heat = min(15.0, max(6.0, base_kw))
            p_heat = min(cap_heat, base_kw + 0.18 * delta_t)
            return p_heat, "EMPIRICAL_HEATING_DEMAND"
        else:
            delta_t = ambient_c - 22.0
            # Cooling ramp: baseline + 0.12 kW/C up to 3.5 kW nominal cap (or base_kw if higher), max 15 kW
            cap_cool = min(15.0, max(3.5, base_kw))
            p_cool = min(cap_cool, base_kw + 0.12 * delta_t)
            return p_cool, "EMPIRICAL_COOLING_DEMAND"

    # ------------------------------------------------------------------
    # Cabin thermal model (Phase 3)
    # ------------------------------------------------------------------

    MAX_HVAC_ELECTRICAL_KW = 15.0  # shared HVAC electrical ceiling for every mode

    def heating_cop(self, ambient_c: float) -> float:
        """Empirical heating COP at `ambient_c`, clipped into the configured band.

        COP(T_amb) = clip(reference - slope x (reference_temp - T_amb), min, max)

        The slope makes the heat pump degrade as it gets colder, which is the only
        qualitative claim this model makes; `reference`, `reference_temp` and
        `slope` are configurable illustrative parameters, not a measured vehicle
        COP map. With `heat_pump_cop_slope_per_k = 0` this degenerates to the
        explicitly documented constant-COP approximation.
        """
        c = self.c
        cop = c.heat_pump_cop_reference - c.heat_pump_cop_slope_per_k * (
            c.heat_pump_cop_reference_temp_c - ambient_c
        )
        return min(max(cop, c.heat_pump_cop_min), c.heat_pump_cop_max)

    def cooling_cop(self, ambient_c: float) -> float:
        """Empirical cooling COP at `ambient_c`, clipped into the configured band.

        COP(T_amb) = clip(reference - slope x (T_amb - reference_temp), min, max)
        so the vapour-compression cycle degrades in extreme heat; the parameters
        are configurable illustrative assumptions, not a compressor map.
        """
        c = self.c
        cop = c.ac_cooling_cop_reference - c.ac_cooling_cop_slope_per_k * (
            ambient_c - c.ac_cooling_cop_reference_temp_c
        )
        return min(max(cop, c.ac_cooling_cop_min), c.ac_cooling_cop_max)

    def resolve_hvac_plan(
        self,
        ambient_c: float | None,
        hvac_power_kw: float | None,
        hvac_model_mode: str | None = None,
        cabin_target_temp_c: float | None = None,
        cabin_initial_temp_c: float | None = None,
        cabin_solar_load_w: float | None = None,
        cabin_occupancy_load_w: float | None = None,
        heat_pump_cop: float | None = None,
        ac_cooling_cop: float | None = None,
        max_heating_electrical_kw: float | None = None,
        max_cooling_electrical_kw: float | None = None,
    ) -> dict[str, Any]:
        """Resolve the scenario's HVAC configuration into one evaluation plan.

        Precedence is unchanged from Phase 2: an explicit `hvac_power_kw` is an
        **electrical** override and short-circuits every thermal model (it is
        never reinterpreted as a thermal load). Otherwise `EMPIRICAL` keeps the
        legacy whole-cycle comfort curve, and `HEAT_PUMP` / `PTC` / `AC_COOLING`
        select the interval-wise first-order cabin model with one actuator.
        """
        mode = self.c.hvac_model_mode if hvac_model_mode is None else hvac_model_mode
        if not isinstance(mode, str) or mode not in HVAC_MODES:
            raise ValueError(
                f"hvac_model_mode must be one of {list(HVAC_MODES)}, got {mode!r}"
            )
        if hvac_power_kw is not None:
            kw = _as_finite_number(hvac_power_kw, "hvac_power_kw")
            if not 0.0 <= kw <= self.MAX_HVAC_ELECTRICAL_KW:
                raise ValueError(
                    f"hvac_power_kw={hvac_power_kw} kW must be in [0.0, "
                    f"{self.MAX_HVAC_ELECTRICAL_KW}] kW"
                )
            return {"kind": "CONSTANT", "electrical_kw": kw, "mode_label": "MANUAL_OVERRIDE",
                    "model_mode": mode}
        if mode == "EMPIRICAL":
            kw, label = self.empirical_hvac_kw(ambient_c, None)
            return {"kind": "CONSTANT", "electrical_kw": kw, "mode_label": label,
                    "model_mode": "EMPIRICAL"}

        if ambient_c is None:
            raise ValueError(
                f"hvac_model_mode={mode} uses the cabin thermal model, which needs an "
                "ambient temperature: pass ambient_c (for example --ambient-c 30)."
            )
        c = self.c
        target = self._scenario_number("cabin_target_temp_c", cabin_target_temp_c,
                                       c.cabin_target_temp_c)
        soak = c.cabin_initial_temp_c if cabin_initial_temp_c is None else cabin_initial_temp_c
        if soak is None:
            # No pre-conditioning: the cabin has equilibrated with the environment.
            start_c = ambient_c
            soak_source = "AMBIENT_SOAK"
        else:
            start_c = _as_finite_number(soak, "cabin_initial_temp_c")
            if not (-50.0 <= start_c <= 80.0):
                raise ValueError(
                    f"cabin_initial_temp_c={soak} degC must be in [-50.0, 80.0] degC"
                )
            soak_source = "EXPLICIT"
        ua = c.cabin_ua_w_per_k
        cap = c.cabin_thermal_capacitance_j_per_k
        if ua <= 0.0:
            raise ValueError(
                f"hvac_model_mode={mode} needs cabin_ua_w_per_k > 0 W/K. An ideally insulated "
                "cabin has no conductive steady state: the heat delivered during an interval is "
                "held constant over that interval (zero-order hold), so with UA=0 the cabin "
                "temperature would march linearly past the setpoint instead of settling. Give the "
                "cabin a leak, or use EMPIRICAL mode."
            )
        tau_plant = cap / ua
        solar = self._scenario_number("cabin_solar_load_w", cabin_solar_load_w, c.cabin_solar_load_w)
        occupancy = self._scenario_number(
            "cabin_occupancy_load_w", cabin_occupancy_load_w, c.cabin_occupancy_load_w
        )
        tau_rec = c.cabin_recovery_time_constant_s
        if tau_rec is None:
            # Fastest setting that keeps the discrete zero-order-hold loop stable.
            gain_w_per_k = ua
            tau_rec_used = tau_plant
        else:
            gain_w_per_k = cap / tau_rec
            tau_rec_used = tau_rec
        max_heat_w = self._scenario_number(
            "max_heating_electrical_kw", max_heating_electrical_kw, c.max_heating_electrical_kw
        ) * 1000.0
        max_cool_w = self._scenario_number(
            "max_cooling_electrical_kw", max_cooling_electrical_kw, c.max_cooling_electrical_kw
        ) * 1000.0
        cop_heat = self._scenario_cop("heat_pump_cop", heat_pump_cop, self.heating_cop(ambient_c))
        cop_cool = self._scenario_cop("ac_cooling_cop", ac_cooling_cop, self.cooling_cop(ambient_c))
        return {
            "kind": "CABIN",
            "model_mode": mode,
            "ambient_c": ambient_c,
            "target_c": target,
            "start_c": start_c,
            "soak_source": soak_source,
            "ua_w_per_k": ua,
            "capacitance_j_per_k": cap,
            "internal_gains_w": solar + occupancy,
            "gain_w_per_k": gain_w_per_k,
            "tau_plant_s": tau_plant,
            "tau_recovery_s": tau_rec_used,
            "tau_effective_s": cap / (ua + gain_w_per_k) if (ua + gain_w_per_k) > 0 else float("inf"),
            "base_kw": c.hvac_power_kw_default,
            "heating_cop": cop_heat,
            "cooling_cop": cop_cool,
            "max_heating_w": max_heat_w,
            "max_cooling_w": max_cool_w,
            "ptc_efficiency": c.ptc_heating_efficiency,
        }

    def _scenario_number(self, name: str, override: Any, default: float) -> float:
        """Validate one optional scenario override against the field's envelope."""
        if override is None:
            return float(default)
        value = _as_finite_number(override, name)
        lo, lo_inc, hi, _hi_inc, _unit = CONFIG_FIELD_LIMITS[name]
        below = value < lo if lo_inc else value <= lo
        if below or value > hi:
            raise ValueError(f"{_range_text(name, CONFIG_FIELD_LIMITS[name])}; got {override!r}")
        return value

    def _scenario_cop(self, name: str, override: Any, derived: float) -> float:
        """Accept an explicit constant COP, else the derived empirical value.

        An explicit COP is the documented simplified approximation: it declares
        that the coefficient does not vary with ambient temperature.
        """
        if override is None:
            return derived
        value = _as_finite_number(override, name)
        if not 1.0 <= value <= 10.0:
            raise ValueError(
                f"{name}={override} must be a finite COP in [1.0, 10.0]: a COP below 1.0 "
                "means the device wastes more electrical energy than the heat it moves "
                "(use PTC mode), and above 10.0 is not a road-vehicle coefficient of "
                "performance."
            )
        return value

    def cabin_thermal_step(
        self,
        plan: dict[str, Any],
        cabin_temp_c: float,
        dt_s: float,
    ) -> dict[str, Any]:
        """Advance the cabin one interval: actuator demand -> electrical draw -> plant.

        Thermal balance (W, positive heats the cabin):

            C dT_cab/dt = UA (T_amb - T_cab) + Q_solar + Q_occupancy + Q_HVAC

        Control law, decided once at the interval start and held constant over it
        (zero-order hold), combining the load needed to hold setpoint outdoors with
        a recovery term proportional to the current error:

            Q_demand = UA (T_target - T_amb) - (Q_solar + Q_occupancy)
                       + K_p (T_target - T_cab),      K_p = C / tau_recovery

        With no clipping this makes the cabin relax to exactly T_target with
        effective time constant C / (UA + K_p); K_p = C / tau_recovery is bounded by
        the configuration guard tau_recovery >= C / UA, which keeps |dT error|
        contraction below 1 per step, so the loop cannot oscillate or diverge.

        Actuator routing, with the electrical cap applied first and the thermal
        delivery following from it, so the two can never disagree:

            heating: P_elec = min(Q_demand / COP, cap_heat);  Q_HVAC = +P_elec x COP
            PTC:     P_elec = min(Q_demand / eta,  cap_heat); Q_HVAC = +P_elec x eta
            cooling: P_elec = min(-Q_demand / COP_c, cap_cool); Q_HVAC = -P_elec x COP_c

        A mode only drives its own actuator: HEAT_PUMP and PTC deliver no cooling and
        AC_COOLING delivers no heating. When the demand points the other way the
        delivery is zero and the setpoint is reported as unmet rather than being
        served by an actuator the configuration did not select. The blower base load
        shares the same 15 kW electrical ceiling, so the actuator is throttled to the
        remaining headroom instead of the total being truncated behind the model's back.

        The plant is integrated in closed form over the interval, so refining the
        time grid leaves the trajectory unchanged to floating-point precision.
        """
        ua = plan["ua_w_per_k"]
        cap = plan["capacitance_j_per_k"]
        gains = plan["internal_gains_w"]
        ambient = plan["ambient_c"]
        target = plan["target_c"]
        base_kw = plan["base_kw"]

        q_demand = (
            ua * (target - ambient)
            - gains
            + plan["gain_w_per_k"] * (target - cabin_temp_c)
        )

        p_elec_w = 0.0
        delivered_w = 0.0
        # Thermal watts delivered per electrical watt drawn by the selected actuator;
        # zero means no compatible actuator is enabled for this demand direction.
        thermal_per_electrical_w = 0.0
        heating = False
        clipped = False
        unmet = False
        if q_demand >= 0.0:
            if plan["model_mode"] == "HEAT_PUMP":
                thermal_per_electrical_w = plan["heating_cop"]
                heating = True
            elif plan["model_mode"] == "PTC":
                thermal_per_electrical_w = plan["ptc_efficiency"]
                heating = True
            else:  # AC_COOLING cannot heat: report, do not silently serve it.
                unmet = q_demand > 0.0
        else:
            if plan["model_mode"] == "AC_COOLING":
                thermal_per_electrical_w = plan["cooling_cop"]
            else:  # HEAT_PUMP / PTC modes carry no enabled cooling actuator.
                unmet = True

        if thermal_per_electrical_w > 0.0:
            cap_w = plan["max_heating_w"] if heating else plan["max_cooling_w"]
            requested_w = abs(q_demand) / thermal_per_electrical_w
            p_elec_w = min(requested_w, cap_w)
            clipped = requested_w > cap_w
            delivered_w = p_elec_w * thermal_per_electrical_w

        # The shared electrical ceiling covers the whole HVAC load, blower base included,
        # so the actuator is throttled rather than the total silently truncated: the
        # thermal delivery always follows the electrical power actually drawn.
        headroom_w = max(0.0, (self.MAX_HVAC_ELECTRICAL_KW - base_kw) * 1000.0)
        if p_elec_w > headroom_w:
            p_elec_w = headroom_w
            delivered_w = p_elec_w * thermal_per_electrical_w
            clipped = True

        heating_w = delivered_w if heating else 0.0
        cooling_w = 0.0 if heating else delivered_w
        q_hvac_w = heating_w - cooling_w
        electrical_kw = base_kw + p_elec_w / 1000.0

        # `resolve_hvac_plan` guarantees UA > 0, so the plant always has a finite
        # steady state and can be integrated in closed form over the interval.
        t_inf = ambient + (q_hvac_w + gains) / ua
        new_temp = t_inf + (cabin_temp_c - t_inf) * math.exp(-ua * dt_s / cap)
        if not (min(cabin_temp_c, t_inf) - 1e-9 <= new_temp <= max(cabin_temp_c, t_inf) + 1e-9):
            raise AssertionError(
                "cabin integration overshot its asymptote: the first-order plant is "
                "monotone between the interval start and its steady state"
            )
        if not math.isfinite(new_temp):
            raise ValueError(
                f"cabin temperature became non-finite at dt={dt_s}s; check "
                "cabin_thermal_capacitance_j_per_k and cabin_ua_w_per_k"
            )

        return {
            "electrical_kw": electrical_kw,
            "thermal_heating_w": heating_w,
            "thermal_cooling_w": cooling_w,
            "q_demand_w": q_demand,
            "cabin_temp_end_c": new_temp,
            "power_clipped": clipped,
            "setpoint_unmet": unmet,
        }

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
            t = _as_finite_number(t, f"cycle point {i} 'time_s'")
            v = _as_finite_number(v, f"cycle point {i} 'speed_mps'")
            g = _as_finite_number(g, f"cycle point {i} 'grade_pct'")
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
        pressure_pa: float | None = None,
        altitude_m: float | None = None,
        headwind_mps: float | None = None,
        crosswind_mps: float | None = None,
        hvac_model_mode: str | None = None,
        cabin_target_temp_c: float | None = None,
        cabin_initial_temp_c: float | None = None,
        cabin_solar_load_w: float | None = None,
        cabin_occupancy_load_w: float | None = None,
        heat_pump_cop: float | None = None,
        ac_cooling_cop: float | None = None,
        max_heating_electrical_kw: float | None = None,
        max_cooling_electrical_kw: float | None = None,
    ) -> dict[str, Any]:
        """Estimate EV driving range using interval-based mechanical work integration.

        Args:
            cycle: list of point dicts with 'time_s', 'speed_mps', optional 'grade_pct'
            soh_pct: battery state of health percentage (scale 1.0 to 120.0%)
            ambient_c: ambient temperature in Celsius (-50 to +60 C)
            hvac_power_kw: explicit HVAC load override (0.0 to 15.0 kW)
            quality_flag: optional data quality pedigree from Member 1 or 2
            cell_id: optional cell identifier
            pressure_pa: station atmospheric pressure in Pa (20-120 kPa); takes
                precedence over altitude for pressure
            altitude_m: elevation above sea level in m (-500 to 11000), used for ISA
                pressure when `pressure_pa` is not supplied
            headwind_mps: along-track wind, positive = headwind, negative = tailwind
                (-60 to +60 m/s); the aero term then uses relative air speed
            crosswind_mps: must be None or 0.0 - a crosswind is not representable in
                this 1-D longitudinal model and is rejected rather than approximated
            hvac_model_mode: overrides the configured mode; EMPIRICAL (default) keeps
                the legacy whole-cycle comfort curve, while HEAT_PUMP, PTC and
                AC_COOLING use the interval-wise first-order cabin thermal model
            cabin_target_temp_c: cabin setpoint (degC) for a cabin-thermal mode
            cabin_initial_temp_c: cabin temperature at t=0 (degC); None soaks to ambient
            cabin_solar_load_w: solar gain into the cabin (W), >= 0
            cabin_occupancy_load_w: occupant/internal gain into the cabin (W), >= 0
            heat_pump_cop: explicit constant heating COP (1.0-10.0), the documented
                simplified approximation; None derives it from the configured
                temperature-dependent relationship
            ac_cooling_cop: explicit constant cooling COP (1.0-10.0); None derives it
            max_heating_electrical_kw: electrical cap of the heating source (kW)
            max_cooling_electrical_kw: electrical cap of the compressor (kW)

        Interpolation model (what is assumed between consecutive cycle points):
            * speed is piecewise linear in time, so acceleration is constant per interval;
            * road grade is piecewise constant at the average of the two endpoint grades;
            * air density, wind and all vehicle parameters are held at their
              scenario values for the whole cycle;
            * HVAC electrical load is constant over the cycle in EMPIRICAL mode and
              in the manual override, and is re-evaluated at the start of every
              interval in the cabin-thermal modes (zero-order hold).
        Energies are then integrated analytically per interval with the split points
        required by the regen model (see `_integrate_wheel_power_interval`). This is
        exact for the interpolated profile above, not for the real road: a coarsely
        sampled cycle still under-resolves actual braking events.

        Braking ledger (kWh): `mechanical_braking_kwh` is the total wheel braking
        energy; `regen_mechanical_absorbed_kwh` is the mechanical share routed through
        the motor; `friction_braking_kwh` is the remainder dissipated by the friction
        brakes; `regen_recovered_kwh` is the electrical energy delivered to the
        battery, i.e. regen_efficiency x regen_mechanical_absorbed_kwh, never more
        than mechanical_braking_kwh.
        """
        # Validate SOH
        soh = _as_finite_number(soh_pct, "soh_pct")
        if 0 < soh <= 1.0:
            raise ValueError(
                f"soh_pct={soh_pct} appears to be a fraction (<= 1.0); "
                "expected percentage on 0-100 scale (e.g. 80.0 for 80% SOH)"
            )
        if soh <= 0 or soh > 120.0:
            raise ValueError(f"soh_pct must be in (1.0, 120.0]; received {soh_pct}")

        # Validate ambient temperature
        if ambient_c is not None:
            ambient = _as_finite_number(ambient_c, "ambient_c")
            if not (-50.0 <= ambient <= 60.0):
                raise ValueError(f"ambient_c={ambient_c} C outside realistic terrestrial limits [-50, 60]")

        # Validate HVAC load
        if hvac_power_kw is not None:
            hvac = _as_finite_number(hvac_power_kw, "hvac_power_kw")
            if hvac < 0:
                raise ValueError("hvac_power_kw must be finite and >= 0")
            if hvac > 15.0:
                raise ValueError(f"hvac_power_kw={hvac_power_kw} kW exceeds realistic automotive HVAC capacity (15 kW)")

        # Validate relative wind
        wind_mps = 0.0 if headwind_mps is None else _as_finite_number(headwind_mps, "headwind_mps")
        if not (-self.MAX_WIND_SPEED_MPS <= wind_mps <= self.MAX_WIND_SPEED_MPS):
            raise ValueError(
                f"headwind_mps={headwind_mps} m/s exceeds the supported wind range "
                f"[{-self.MAX_WIND_SPEED_MPS}, {self.MAX_WIND_SPEED_MPS}] m/s"
            )
        if crosswind_mps is not None:
            crosswind = _as_finite_number(crosswind_mps, "crosswind_mps")
            if crosswind != 0.0:
                raise ValueError(
                    f"crosswind_mps={crosswind_mps} m/s cannot be modelled: the aerodynamic "
                    "term here is one-dimensional along the direction of travel, so a "
                    "crosswind is not equivalent to a longitudinal wind. Leave it unset; "
                    "crosswind effects on Cd and frontal area are an unmodelled limitation."
                )

        # Validate drive cycle
        self._validate_cycle(cycle)

        # Compute dynamic environmental parameters
        environment = self.resolve_environment(
            ambient_c, pressure_pa=pressure_pa, altitude_m=altitude_m
        )
        air_density = environment["air_density_kg_m3"]
        hvac_plan = self.resolve_hvac_plan(
            ambient_c,
            hvac_power_kw,
            hvac_model_mode=hvac_model_mode,
            cabin_target_temp_c=cabin_target_temp_c,
            cabin_initial_temp_c=cabin_initial_temp_c,
            cabin_solar_load_w=cabin_solar_load_w,
            cabin_occupancy_load_w=cabin_occupancy_load_w,
            heat_pump_cop=heat_pump_cop,
            ac_cooling_cop=ac_cooling_cop,
            max_heating_electrical_kw=max_heating_electrical_kw,
            max_cooling_electrical_kw=max_cooling_electrical_kw,
        )
        cabin_mode = hvac_plan["kind"] == "CABIN"
        hvac_mode = f"CABIN_{hvac_plan['model_mode']}" if cabin_mode else hvac_plan["mode_label"]
        # `hvac_power_kw` keeps its established meaning: the electrical HVAC load. In a
        # cabin-thermal mode it varies per interval, so the reported value becomes the
        # energy-weighted mean over the cycle (computed after the loop).
        hvac_kw = hvac_plan.get("electrical_kw", 0.0)

        c = self.c
        m_eff = c.vehicle_mass_kg * c.rotational_inertia_factor
        aero_coeff = 0.5 * air_density * c.drag_coefficient_cd * c.frontal_area_m2

        total_traction_energy_j = 0.0
        total_regen_energy_j = 0.0
        total_braking_mech_energy_j = 0.0
        total_regen_mech_energy_j = 0.0
        total_aux_energy_j = 0.0
        total_hvac_energy_j = 0.0
        total_hvac_heating_thermal_j = 0.0
        total_hvac_cooling_thermal_j = 0.0
        cabin_temp_c = hvac_plan["start_c"] if cabin_mode else None
        cabin_temp_start_c = cabin_temp_c
        hvac_power_clipped = False
        hvac_setpoint_unmet = False
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

            # Wheel power is P_wheel(v) = linear_coeff * v + aero_coeff * v**3 with
            # linear_coeff = m_eff * a + m * g * (Crr * cos(theta) + sin(theta)).
            # Speed is linear in time inside the interval and grade is held at the
            # interval average, so this representation reproduces the exact interval
            # work of the analytical terms (kinetic, aerodynamic, rolling, gradient).
            accel = (v1 - v0) / dt
            resistance_force = c.vehicle_mass_kg * self.G * (
                c.rolling_resistance_coefficient_crr * math.cos(theta) + math.sin(theta)
            )
            breakdown = self._integrate_wheel_power_interval(
                v0=v0,
                v1=v1,
                dt=dt,
                linear_coeff_n=m_eff * accel + resistance_force,
                cubic_coeff_n=aero_coeff,
                headwind_mps=wind_mps,
            )

            total_traction_energy_j += breakdown["traction_mech_j"] / c.drivetrain_efficiency
            total_regen_energy_j += breakdown["regen_elec_j"]
            total_braking_mech_energy_j += breakdown["braking_mech_j"]
            total_regen_mech_energy_j += breakdown["regen_mech_j"]

            # Auxiliary and HVAC electrical loads are accumulated separately, so a
            # kWh figure can never be counted in both buckets.
            if cabin_mode:
                step = self.cabin_thermal_step(hvac_plan, cabin_temp_c, dt)
                hvac_kw = step["electrical_kw"]
                cabin_temp_c = step["cabin_temp_end_c"]
                total_hvac_heating_thermal_j += step["thermal_heating_w"] * dt
                total_hvac_cooling_thermal_j += step["thermal_cooling_w"] * dt
                hvac_power_clipped = hvac_power_clipped or step["power_clipped"]
                hvac_setpoint_unmet = hvac_setpoint_unmet or step["setpoint_unmet"]
            total_aux_energy_j += c.auxiliary_power_kw * 1000.0 * dt
            total_hvac_energy_j += hvac_kw * 1000.0 * dt
            total_distance_m += ds
            total_time_s += dt

        distance_km = total_distance_m / 1000.0
        if distance_km <= 0:
            raise ValueError("Drive cycle distance is zero; range cannot be extrapolated")

        # Convert Joules to kWh (1 kWh = 3.6e6 J)
        traction_kwh = total_traction_energy_j / 3.6e6
        regen_kwh = total_regen_energy_j / 3.6e6
        braking_mech_kwh = total_braking_mech_energy_j / 3.6e6
        regen_mech_kwh = total_regen_mech_energy_j / 3.6e6
        friction_braking_kwh = max(0.0, braking_mech_kwh - regen_mech_kwh)
        auxiliary_kwh = total_aux_energy_j / 3.6e6
        hvac_kwh = total_hvac_energy_j / 3.6e6
        # Backward-compatible combined field: exactly the sum of the two buckets.
        aux_hvac_kwh = auxiliary_kwh + hvac_kwh
        net_cycle_energy_kwh = traction_kwh - regen_kwh + aux_hvac_kwh
        if cabin_mode:
            # Per-interval load: report the energy-weighted mean electrical power.
            hvac_kw = (hvac_kwh * 3600.0 / total_time_s) if total_time_s > 0.0 else 0.0

        usable_energy_kwh = (
            c.nominal_pack_energy_kwh
            * c.usable_energy_fraction
            * (soh_pct / 100.0)
        )

        provenance_status = (c.provenance or {}).get("status")
        base_status = "ILLUSTRATIVE_UNVALIDATED" if provenance_status == "ILLUSTRATIVE_ASSUMPTIONS" else "REQUIRES_VALIDATION"

        if net_cycle_energy_kwh <= 0:
            # Non-positive net energy: sustained descent, or a cycle whose final kinetic
            # energy is lower than its initial kinetic energy (one-shot regen harvest).
            consumption_kwh_per_km = net_cycle_energy_kwh / distance_km
            range_km = float("inf")
            quality_status = "NET_GENERATING_DESCENT"
            limitations = [
                "Net drive cycle energy is non-positive (net recharge from elevation drop or regenerative braking).",
                "Range is mathematically unbounded under perpetual descent conditions.",
                "This cycle gives back more energy than it consumes; part of that is the vehicle's own initial "
                "kinetic energy being recovered once. It is not a sustainable repeated-cycle consumption rate, "
                "so the unbounded value must never be reported as a driving range.",
                "SOH is approximated as a linear multiplier on usable pack energy.",
                "Regenerative recovery integrates instantaneous wheel power, split at blend thresholds, wheel-power sign changes and charge-power limit crossings; speed is piecewise linear and grade piecewise constant between cycle points, so coarse cycles still under-resolve braking events.",
                "max_regen_power_kw is a constant electrical charge-power limit; SOC- and temperature-dependent charge acceptance is not modelled.",
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
                "Regenerative recovery integrates instantaneous wheel power, split at blend thresholds, wheel-power sign changes and charge-power limit crossings; speed is piecewise linear and grade piecewise constant between cycle points, so coarse cycles still under-resolve braking events.",
                "max_regen_power_kw is a constant electrical charge-power limit; SOC- and temperature-dependent charge acceptance is not modelled.",
                "Validate against vehicle-specific consumption/range before making real-world claims."
            ]

        if not cabin_mode and hvac_plan["model_mode"] in CABIN_HVAC_MODES:
            limitations.append(
                f"The electrical override hvac_power_kw={hvac_kw:.2f} kW short-circuited the "
                f"requested {hvac_plan['model_mode']} cabin model, so no thermal state was "
                "evolved: the HVAC energy is the override times the cycle duration, and "
                "hvac_model_mode only records what was asked for."
            )

        if environment["air_density_source"] == "IDEAL_GAS_LAW":
            limitations.append(
                "Air density is ideal-gas rho = P/(R_air*T) with "
                f"P={environment['atmospheric_pressure_pa']:.1f} Pa from "
                f"{environment['pressure_source']} and T={environment['temperature_k']:.2f} K from "
                f"{environment['temperature_source']}; ISA pressure is defined only from "
                f"{self.MIN_ALTITUDE_M} m to {self.MAX_ALTITUDE_M} m."
            )
        if wind_mps != 0.0:
            limitations.append(
                f"Steady along-track wind of {wind_mps:.2f} m/s (positive = headwind) is applied "
                "to the whole cycle through the relative air speed; gusts, wind shear, and "
                "crosswind-induced changes to Cd and frontal area are not modelled."
            )
        if cabin_mode:
            details = (
                f"Cabin thermal model {hvac_plan['model_mode']}: single-node balance "
                f"C dT/dt = UA(T_amb-T_cab) + Q_solar + Q_occupancy + Q_HVAC with "
                f"C={hvac_plan['capacitance_j_per_k']:.0f} J/K, UA={hvac_plan['ua_w_per_k']:.1f} W/K "
                f"(plant time constant {hvac_plan['tau_plant_s']:.0f} s, effective "
                f"{hvac_plan['tau_effective_s']:.0f} s), setpoint {hvac_plan['target_c']:.1f} C from a "
                f"{hvac_plan['soak_source']} start of {hvac_plan['start_c']:.1f} C. Cabin ended the cycle at "
                f"{cabin_temp_c:.2f} C."
            )
            limitations.append(details)
            limitations.append(
                "Cabin capacitance, UA, solar/occupancy gains and the COP or efficiency "
                "parameters are configurable ILLUSTRATIVE assumptions, not measured vehicle "
                "data; this is not a calibrated cabin simulation."
            )
            limitations.append(
                "One actuator per mode: HEAT_PUMP and PTC provide no cooling and AC_COOLING "
                "provides no heating, so a demand of the opposite sign is reported as unmet "
                "instead of being served by an unselected system. Blower, cabin fan and "
                "air-filter pressure drop are folded into the base HVAC load; humidity, "
                "defrost, seat heaters and battery thermal management are not modelled."
            )
            if hvac_plan["model_mode"] == "HEAT_PUMP":
                limitations.append(
                    f"Heating COP {hvac_plan['heating_cop']:.3f} "
                    + (
                        "is an explicitly configured constant (simplified approximation)."
                        if heat_pump_cop is not None
                        else f"is derived from a linear degradation from the rating point "
                        f"{self.c.heat_pump_cop_reference_temp_c} C, clipped to the "
                        f"configured band; the point is the EN 14511 A7 air-side heating duty "
                        f"(W35 does not apply to an air-delivered cabin), and both the duty COP "
                        f"and the slope are illustrative assumptions, not a measured COP map."
                    )
                )
            elif hvac_plan["model_mode"] == "AC_COOLING":
                limitations.append(
                    f"Cooling COP {hvac_plan['cooling_cop']:.3f} "
                    + (
                        "is an explicitly configured constant (simplified approximation)."
                        if ac_cooling_cop is not None
                        else "is derived from a linear degradation with ambient temperature, "
                        "clipped to the configured band; the anchor is the EN 14511 A35 cooling "
                        "duty and both its COP and its slope are illustrative assumptions, "
                        "not a compressor map."
                    )
                )
            if hvac_setpoint_unmet:
                limitations.append(
                    "The selected HVAC mode cannot meet the cabin demand for part of this "
                    "cycle (no compatible actuator, or the electrical cap was reached), so "
                    "the cabin setpoint was not held."
                )
            elif hvac_power_clipped:
                limitations.append(
                    "HVAC electrical demand reached its configured cap for part of the cycle; "
                    "the cabin therefore approached but did not hold the setpoint."
                )

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
            "mechanical_braking_kwh": braking_mech_kwh,
            "regen_mechanical_absorbed_kwh": regen_mech_kwh,
            "friction_braking_kwh": friction_braking_kwh,
            "usable_energy_kwh": usable_energy_kwh,
            "soh_pct": soh_pct,
            "ambient_c": ambient_c,
            "air_density_kg_m3": air_density,
            "atmospheric_pressure_pa": environment["atmospheric_pressure_pa"],
            "altitude_m": environment["altitude_m"],
            "air_density_source": environment["air_density_source"],
            "headwind_mps": wind_mps,
            "hvac_power_kw": hvac_kw,
            "hvac_mode": hvac_mode,
            "hvac_model_mode": hvac_plan["model_mode"],
            "auxiliary_energy_kwh": auxiliary_kwh,
            "hvac_energy_kwh": hvac_kwh,
            "hvac_heating_thermal_energy_kwh": (
                total_hvac_heating_thermal_j / 3.6e6 if cabin_mode else None
            ),
            "hvac_cooling_thermal_energy_kwh": (
                total_hvac_cooling_thermal_j / 3.6e6 if cabin_mode else None
            ),
            "cabin_target_temp_c": hvac_plan["target_c"] if cabin_mode else None,
            "cabin_temp_start_c": cabin_temp_start_c if cabin_mode else None,
            "cabin_temp_end_c": cabin_temp_c if cabin_mode else None,
            "cabin_time_constant_s": hvac_plan["tau_plant_s"] if cabin_mode else None,
            "hvac_heating_cop": (
                hvac_plan["heating_cop"] if cabin_mode and hvac_plan["model_mode"] == "HEAT_PUMP"
                else None
            ),
            "hvac_heating_efficiency": (
                hvac_plan["ptc_efficiency"] if cabin_mode and hvac_plan["model_mode"] == "PTC"
                else None
            ),
            "hvac_cooling_cop": (
                hvac_plan["cooling_cop"] if cabin_mode and hvac_plan["model_mode"] == "AC_COOLING"
                else None
            ),
            "cabin_setpoint_unmet": hvac_setpoint_unmet if cabin_mode else None,
            "hvac_power_limited": hvac_power_clipped if cabin_mode else None,
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
        pressure_pa: float | None = None,
        altitude_m: float | None = None,
        headwind_mps: float | None = None,
        crosswind_mps: float | None = None,
        hvac_model_mode: str | None = None,
        cabin_target_temp_c: float | None = None,
        cabin_initial_temp_c: float | None = None,
        cabin_solar_load_w: float | None = None,
        cabin_occupancy_load_w: float | None = None,
        heat_pump_cop: float | None = None,
        ac_cooling_cop: float | None = None,
        max_heating_electrical_kw: float | None = None,
        max_cooling_electrical_kw: float | None = None,
    ) -> list[dict[str, Any]]:
        """Evaluate range decay across an SOH lifetime trajectory from Member 2.

        Args:
            cycle: drive cycle points
            soh_trajectory: list of SOH percentages or dicts with 'soh_pct'/'cycle_index'
            ambient_c: scenario ambient temperature
            hvac_power_kw: optional explicit HVAC load
            quality_flags: optional list of flags corresponding to each point
            cell_id: optional cell identifier
            pressure_pa/altitude_m/headwind_mps/crosswind_mps: optional environmental
                scenario inputs, passed unchanged to every `estimate` call
            hvac_model_mode/cabin_*/heat_pump_cop/ac_cooling_cop/max_*_electrical_kw:
                optional HVAC scenario inputs, also passed unchanged. Each trajectory
                point is an independent evaluation, so a cabin-thermal mode restarts
                the cabin from its soak temperature at every point rather than
                carrying heat across the SOH series.
        """
        results: list[dict[str, Any]] = []
        for idx, item in enumerate(soh_trajectory):
            flag = quality_flags[idx] if (quality_flags and idx < len(quality_flags)) else None
            cid = cell_id
            cycle_idx = None
            if isinstance(item, dict):
                soh = _as_finite_number(item["soh_pct"], f"soh_trajectory[{idx}] 'soh_pct'")
                flag = item.get("quality_flag", flag)
                cid = item.get("cell_id", cid)
                cycle_idx = item.get("cycle_index", idx)
            else:
                soh = _as_finite_number(item, f"soh_trajectory[{idx}]")

            r = self.estimate(
                cycle=cycle,
                soh_pct=soh,
                ambient_c=ambient_c,
                hvac_power_kw=hvac_power_kw,
                quality_flag=flag,
                cell_id=cid,
                pressure_pa=pressure_pa,
                altitude_m=altitude_m,
                headwind_mps=headwind_mps,
                crosswind_mps=crosswind_mps,
                hvac_model_mode=hvac_model_mode,
                cabin_target_temp_c=cabin_target_temp_c,
                cabin_initial_temp_c=cabin_initial_temp_c,
                cabin_solar_load_w=cabin_solar_load_w,
                cabin_occupancy_load_w=cabin_occupancy_load_w,
                heat_pump_cop=heat_pump_cop,
                ac_cooling_cop=ac_cooling_cop,
                max_heating_electrical_kw=max_heating_electrical_kw,
                max_cooling_electrical_kw=max_cooling_electrical_kw,
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
    parser.add_argument("--pressure-pa", type=float, default=None,
                        help="station atmospheric pressure in Pa (overrides ISA altitude pressure)")
    parser.add_argument("--altitude-m", type=float, default=None,
                        help="elevation above sea level in metres (-500 to 11000)")
    parser.add_argument("--headwind-mps", type=float, default=None,
                        help="along-track wind in m/s; positive = headwind, negative = tailwind")
    parser.add_argument("--crosswind-mps", type=float, default=None,
                        help="accepted only as 0; crosswind is not modelled")
    parser.add_argument("--hvac-mode", choices=list(HVAC_MODES), default=None,
                        help="overrides the configured HVAC mode (default EMPIRICAL)")
    parser.add_argument("--cabin-target-c", type=float, default=None,
                        help="cabin setpoint in degC for a cabin-thermal mode")
    parser.add_argument("--cabin-initial-c", type=float, default=None,
                        help="cabin temperature at t=0 in degC; default soaks to ambient")
    parser.add_argument("--solar-load-w", type=float, default=None,
                        help="solar gain into the cabin in W (0 disables the term)")
    parser.add_argument("--occupancy-load-w", type=float, default=None,
                        help="occupant/internal heat gain into the cabin in W")
    parser.add_argument("--heat-pump-cop", type=float, default=None,
                        help="explicit constant heating COP (simplified approximation)")
    parser.add_argument("--cooling-cop", type=float, default=None,
                        help="explicit constant cooling COP (simplified approximation)")
    parser.add_argument("--max-heating-kw", type=float, default=None,
                        help="maximum electrical heating power in kW")
    parser.add_argument("--max-cooling-kw", type=float, default=None,
                        help="maximum electrical cooling (compressor) power in kW")
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
        pressure_pa=args.pressure_pa,
        altitude_m=args.altitude_m,
        headwind_mps=args.headwind_mps,
        crosswind_mps=args.crosswind_mps,
        hvac_model_mode=args.hvac_mode,
        cabin_target_temp_c=args.cabin_target_c,
        cabin_initial_temp_c=args.cabin_initial_c,
        cabin_solar_load_w=args.solar_load_w,
        cabin_occupancy_load_w=args.occupancy_load_w,
        heat_pump_cop=args.heat_pump_cop,
        ac_cooling_cop=args.cooling_cop,
        max_heating_electrical_kw=args.max_heating_kw,
        max_cooling_electrical_kw=args.max_cooling_kw,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
