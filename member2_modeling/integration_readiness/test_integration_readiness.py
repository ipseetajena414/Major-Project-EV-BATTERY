"""Focused tests for the isolated Member 2 integration contract.

These tests exercise the real predictor and model artifacts read-only. They do
not retrain anything, write any production file, or implement Member 3 (whose
source is not present in this repository). The last test only checks that the
handoff payload stops at the argument mapping: no pack energy, cell capacity or
range value is produced here.
"""
from __future__ import annotations

import json
import sys
import unittest
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member2_modeling.integration_readiness import adapter as ia  # noqa: E402
from member2_modeling.predictor import Member2Predictor  # noqa: E402

MODEL_DIR = REPO_ROOT / "member2_modeling"
SCHEMA_PATH = MODEL_DIR / "integration_readiness" / "battery_health_contract_v1_1_0_draft.json"
SUPERSEDED_SCHEMA_PATH = (
    MODEL_DIR / "integration_readiness" / "battery_health_contract_v1_0_0_draft.json"
)
SAMPLE_PROVENANCE = {
    "cell_id": "SAMPLE_CELL",
    "dataset_id": "SAMPLE_DATASET",
    "cycle_id": 1,
    "cycle_position": 0,
    "prediction_timestamp": "2026-10-29T00:00:00Z",
}
ILLUSTRATIVE_AMBIENT_C = 25.0
ILLUSTRATIVE_HVAC_KW = 0.5


def load_predictor() -> Member2Predictor:
    return Member2Predictor(str(MODEL_DIR))


def sample_features(predictor: Member2Predictor) -> dict:
    return dict(predictor.manifest["sample_io"]["input_features"])


def lfp_features(predictor: Member2Predictor) -> dict:
    names = predictor.get_feature_schema("LFP")["feature_names"]
    full = sample_features(predictor)
    return {name: full[name] for name in names}


def nmc_contract(predictor: Member2Predictor, **kwargs) -> dict:
    return ia.build_health_result(
        predictor,
        sample_features(predictor),
        "NMC",
        SAMPLE_PROVENANCE,
        **kwargs,
    )


class ContractShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_predictor()
        cls.contract = nmc_contract(cls.predictor)
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    def test_top_level_keys_match_schema(self):
        self.assertEqual(set(self.contract), set(self.schema["properties"]))
        for key in self.schema["required"]:
            self.assertIn(key, self.contract)

    def test_nested_keys_match_schema(self):
        for section in ("subject", "soh", "rul", "quality", "source", "request"):
            with self.subTest(section=section):
                props = self.schema["properties"][section]["properties"]
                self.assertEqual(set(self.contract[section]), set(props))
                for key in self.schema["properties"][section]["required"]:
                    self.assertIn(key, self.contract[section])

    def test_contract_version_matches_schema_and_is_newest(self):
        self.assertEqual(self.contract["contract_version"], ia.CONTRACT_VERSION)
        self.assertEqual(self.schema["properties"]["contract_version"]["const"], ia.CONTRACT_VERSION)
        # The superseded draft stays on disk for traceability and must still parse.
        superseded = json.loads(SUPERSEDED_SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            superseded["properties"]["contract_version"]["const"], "1.0.0-draft"
        )
        self.assertNotEqual(
            superseded["properties"]["contract_version"]["const"], ia.CONTRACT_VERSION
        )

    def test_success_is_experimental_never_production(self):
        self.assertEqual(self.contract["status"], "ok_experimental")
        self.assertIs(self.contract["production_ready"], False)

    def test_subject_identifiers(self):
        subject = self.contract["subject"]
        self.assertEqual(subject["chemistry"], "NMC")
        self.assertEqual(subject["cell_id"], "SAMPLE_CELL")
        self.assertEqual(subject["dataset_id"], "SAMPLE_DATASET")
        self.assertEqual(subject["cycle_id"], 1)
        self.assertEqual(subject["cycle_position"], 0)
        self.assertEqual(subject["prediction_timestamp"], "2026-10-29T00:00:00Z")

    def test_cycle_position_is_null_when_not_supplied(self):
        provenance = dict(SAMPLE_PROVENANCE)
        provenance.pop("cycle_position")
        contract = ia.build_health_result(
            self.predictor, sample_features(self.predictor), "NMC", provenance
        )
        self.assertEqual(contract["status"], "ok_experimental")
        self.assertIsNone(contract["subject"]["cycle_position"])
        self.assertEqual(contract["subject"]["cycle_id"], 1)

    def test_member3_minimum_fields_are_present(self):
        # Sufficiency check: every field the handoff document promises a
        # range consumer is present with the documented type, even when null.
        subject = self.contract["subject"]
        soh = self.contract["soh"]
        self.assertIsInstance(subject["cell_id"], str)
        self.assertIsInstance(subject["dataset_id"], str)
        self.assertIsInstance(subject["cycle_id"], int)
        self.assertIsInstance(subject["prediction_timestamp"], str)
        self.assertIsInstance(soh["predicted_soh_pct"], float)
        self.assertIsInstance(soh["current_observed_soh_pct"], float)
        self.assertIsInstance(soh["model_id"], str)
        self.assertIsInstance(soh["validation_status"], str)
        self.assertIsInstance(self.contract["quality"]["quality_flags"], list)
        self.assertIsInstance(self.contract["quality"]["warnings"], list)
        self.assertIsNone(self.contract["rul"]["rul_cycles"])
        self.assertEqual(self.contract["rul"]["status"], "unavailable")

    def test_soh_fields_and_horizon_semantics(self):
        soh = self.contract["soh"]
        self.assertEqual(soh["status"], "ok")
        self.assertAlmostEqual(soh["predicted_soh_pct"], 74.9645, places=4)
        self.assertAlmostEqual(
            soh["current_observed_soh_pct"],
            sample_features(self.predictor)["current_soh_pct"],
            places=9,
        )
        self.assertEqual(soh["unit"], "%")
        self.assertEqual(soh["model_id"], "nmc_lgbm")
        self.assertEqual(soh["model_status"], "experimental_research_stage")
        self.assertEqual(soh["validation_status"], "research_stage_not_production_ready")
        self.assertEqual(soh["uncertainty"], {"status": "not_provided"})
        self.assertEqual(soh["horizon"]["value"], 1)
        self.assertEqual(soh["horizon"]["unit"], "observed_cycle_records")
        self.assertIn("not necessarily the next physical cycle", soh["horizon"]["semantics"])

    def test_rul_is_never_numeric(self):
        rul = self.contract["rul"]
        self.assertEqual(rul["status"], "unavailable")
        self.assertIsNone(rul["rul_cycles"])
        self.assertIs(rul["approved_for_use"], False)
        self.assertEqual(rul["reason"], "UNVALIDATED_PRE_EOL")
        self.assertEqual(rul["model_id"], "nmc_rul_lgbm")
        self.assertEqual(rul["model_status"], "unvalidated_pre_eol")

    def test_adapter_suppresses_a_numeric_rul_the_base_api_returns(self):
        base = self.predictor.estimate_rul(sample_features(self.predictor), "NMC")
        self.assertEqual(base.status, "ok")
        self.assertEqual(base.prediction, 0.0)
        self.assertIsNone(self.contract["rul"]["rul_cycles"])

    def test_capacity_omitted_by_default(self):
        self.assertIsNone(self.contract["capacity"])
        self.assertNotIn("kwh", json.dumps(self.contract).lower())

    def test_quality_flags_and_checks(self):
        quality = self.contract["quality"]
        flags = set(quality["quality_flags"])
        for flag in (
            "MODEL_EXPERIMENTAL_RESEARCH_STAGE",
            "MODEL_UNDERPERFORMS_PERSISTENCE",
            "SOH_HORIZON_OBSERVED_RECORD_NOT_PHYSICAL_CYCLE",
            "CURRENT_SOH_IS_UPSTREAM_LABEL",
            "NO_UNCERTAINTY_QUANTIFICATION",
            "NO_OUT_OF_DISTRIBUTION_CHECK",
            "CHEMISTRY_DECLARED_BY_CALLER",
            "RUL_NOT_APPROVED",
        ):
            self.assertIn(flag, flags)
        self.assertIn("out_of_distribution", quality["checks_not_performed"])
        self.assertIn("vehicle_level_range_validation", quality["checks_not_performed"])
        self.assertEqual(quality["label_provenance"]["status"], "observed")
        self.assertEqual(self.contract["errors"], [])

    def test_source_matches_actual_api(self):
        source = self.contract["source"]
        self.assertEqual(source["module"], "member2_modeling/predictor.py")
        self.assertEqual(source["function"], "Member2Predictor.predict_future_soh")
        self.assertEqual(source["schema_version"], "1.0.0")
        self.assertEqual(source["model_file"], "nmc_lgbm_model.joblib")

    def test_json_serializable_without_nan(self):
        json.dumps(self.contract, allow_nan=False)


class CapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_predictor()
        cls.contract = nmc_contract(cls.predictor, include_capacity=True)

    def test_capacity_is_cell_ah_and_never_pack_energy(self):
        capacity = self.contract["capacity"]
        self.assertAlmostEqual(capacity["capacity_ah"], 1.4993, places=3)
        self.assertEqual(capacity["unit"], "Ah")
        self.assertEqual(capacity["reference_capacity_ah"], 2.0)
        self.assertEqual(capacity["basis"], "derived_soh_pct_times_rated_capacity")
        self.assertEqual(capacity["validation_status"], "not_independently_validated")
        self.assertIs(capacity["is_pack_energy"], False)
        self.assertIn("CAPACITY_DERIVED_FROM_SOH", self.contract["quality"]["quality_flags"])
        self.assertNotIn("kwh", json.dumps(self.contract).lower())


class UnsupportedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_predictor()

    def test_naion_is_explicitly_unsupported(self):
        contract = ia.build_health_result(
            self.predictor, sample_features(self.predictor), "Na-ion", SAMPLE_PROVENANCE
        )
        self.assertEqual(contract["status"], "unsupported")
        self.assertEqual(contract["errors"][0]["code"], "UNSUPPORTED_CHEMISTRY")
        self.assertIn("Na-ion", contract["errors"][0]["message"])
        self.assertIsNone(contract["soh"]["predicted_soh_pct"])
        self.assertIsNone(contract["soh"]["current_observed_soh_pct"])
        self.assertEqual(contract["soh"]["status"], "unavailable")
        self.assertEqual(contract["rul"]["reason"], "CHEMISTRY_UNSUPPORTED")
        self.assertIn("NAION_INFERENCE_UNSUPPORTED", contract["quality"]["quality_flags"])

    def test_lfp_soh_is_available_but_conditional(self):
        contract = ia.build_health_result(
            self.predictor, lfp_features(self.predictor), "LFP", SAMPLE_PROVENANCE
        )
        self.assertEqual(contract["status"], "ok_experimental")
        self.assertAlmostEqual(contract["soh"]["predicted_soh_pct"], 94.3044, places=4)
        self.assertEqual(contract["soh"]["model_id"], "lfp_lgbm")
        self.assertEqual(
            contract["soh"]["validation_status"],
            "research_stage_conditional_labels_not_production_ready",
        )
        self.assertIn("CONDITIONAL_LABELS", contract["quality"]["quality_flags"])
        self.assertEqual(contract["quality"]["label_provenance"]["status"], "conditional")
        self.assertEqual(contract["rul"]["status"], "unavailable")
        self.assertEqual(contract["rul"]["reason"], "CHEMISTRY_UNSUPPORTED")
        self.assertIsNone(contract["rul"]["rul_cycles"])

    def test_unsupported_horizon(self):
        contract = ia.build_health_result(
            self.predictor,
            sample_features(self.predictor),
            "NMC",
            SAMPLE_PROVENANCE,
            horizon_cycles=5,
        )
        self.assertEqual(contract["status"], "unsupported")
        self.assertEqual(contract["errors"][0]["code"], "UNSUPPORTED_HORIZON")
        self.assertIsNone(contract["soh"]["predicted_soh_pct"])
        self.assertIn("HORIZON_UNSUPPORTED", contract["quality"]["quality_flags"])

    def test_unknown_chemistry(self):
        contract = ia.build_health_result(
            self.predictor, {}, "LTO", SAMPLE_PROVENANCE
        )
        self.assertEqual(contract["status"], "unsupported")
        self.assertEqual(contract["errors"][0]["code"], "UNSUPPORTED_CHEMISTRY")


class GuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_predictor()

    def assert_no_prediction(self, contract):
        self.assertIsNone(contract["soh"]["predicted_soh_pct"])
        self.assertEqual(contract["soh"]["status"], "unavailable")
        self.assertEqual(contract["status"], "error")

    def test_missing_provenance(self):
        contract = ia.build_health_result(
            self.predictor, sample_features(self.predictor), "NMC", {}
        )
        self.assert_no_prediction(contract)
        self.assertEqual(contract["errors"][0]["code"], "INVALID_PROVENANCE")

    def test_non_integer_cycle_id(self):
        contract = ia.build_health_result(
            self.predictor,
            sample_features(self.predictor),
            "NMC",
            {"cell_id": "C", "dataset_id": "D", "cycle_id": "42"},
        )
        self.assert_no_prediction(contract)
        self.assertEqual(contract["errors"][0]["field"], "cycle_id")

    def test_invalid_cycle_position(self):
        for bad in (-1, "3", 2.5, True):
            with self.subTest(value=bad):
                provenance = dict(SAMPLE_PROVENANCE, cycle_position=bad)
                contract = ia.build_health_result(
                    self.predictor, sample_features(self.predictor), "NMC", provenance
                )
                self.assert_no_prediction(contract)
                self.assertEqual(
                    [e["field"] for e in contract["errors"]], ["cycle_position"]
                )
                self.assertIsNone(contract["subject"]["cycle_position"])

    def test_missing_feature(self):
        features = sample_features(self.predictor)
        features.pop("v_mean")
        contract = ia.build_health_result(
            self.predictor, features, "NMC", SAMPLE_PROVENANCE
        )
        self.assert_no_prediction(contract)
        self.assertEqual(contract["errors"][0]["code"], "MISSING_FEATURES")
        self.assertIn("v_mean", contract["errors"][0]["message"])

    def test_nan_feature(self):
        features = sample_features(self.predictor)
        features["v_mean"] = float("nan")
        contract = ia.build_health_result(
            self.predictor, features, "NMC", SAMPLE_PROVENANCE
        )
        self.assert_no_prediction(contract)
        self.assertEqual(contract["errors"][0]["code"], "INVALID_FEATURE_VALUES")

    def test_nmc_feature_keys_are_rejected_for_lfp(self):
        # The base predictor accepts this cross-chemistry input with only a
        # warning; the contract guard must reject it.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            base = self.predictor.predict_future_soh(
                sample_features(self.predictor), "LFP", 1
            )
        self.assertEqual(base.status, "ok")
        self.assertTrue(any("unexpected features" in str(w.message).lower() for w in caught))

        contract = ia.build_health_result(
            self.predictor, sample_features(self.predictor), "LFP", SAMPLE_PROVENANCE
        )
        self.assert_no_prediction(contract)
        self.assertEqual(contract["errors"][0]["code"], "FEATURE_SET_MISMATCH")

    def test_non_strict_mode_flags_extra_keys(self):
        contract = ia.build_health_result(
            self.predictor,
            sample_features(self.predictor),
            "LFP",
            SAMPLE_PROVENANCE,
            strict_feature_set=False,
        )
        self.assertEqual(contract["status"], "ok_experimental")
        self.assertIn("EXTRA_FEATURE_KEYS_IGNORED", contract["quality"]["quality_flags"])
        self.assertTrue(
            any("Unexpected feature keys" in warning for warning in contract["quality"]["warnings"])
        )

    def test_invalid_soh_values_are_rejected(self):
        for value in (0.0, 100.0):
            self.assertTrue(ia.validate_soh_output(value))
        for value in (-0.1, 100.1, float("nan"), float("inf"), "80", None, True):
            self.assertFalse(ia.validate_soh_output(value))

    def test_capacity_is_not_soh_validated_by_accident(self):
        # A capacity in Ah is >100 and must still be accepted as finite/non-negative;
        # SOH range validation must not be applied to Ah values.
        self.assertTrue(ia._finite_nonnegative(150.0))
        self.assertFalse(ia._finite_nonnegative(-1.0))
        self.assertFalse(ia._finite_nonnegative(float("nan")))


class RealDataHandoffTests(unittest.TestCase):
    """Prove the handoff can be produced from real cycle records, not just the
    manifest sample. Read-only: the parquet is never written."""

    @classmethod
    def setUpClass(cls):
        import pandas as pd

        from member2_modeling import build_and_train as bt

        cls.bt = bt
        cls.predictor = load_predictor()
        cls.frames = pd.read_parquet(
            str(REPO_ROOT / "master_table.parquet"),
            filters=[
                ("cell_id", "in", ["B0005", "BATTERY_5"]),
                ("cycle_id", "in", [100, 101, 499, 500]),
            ],
        )

    def features_for(self, cell_id, cycle_id):
        frame = self.frames[
            (self.frames["cell_id"] == cell_id) & (self.frames["cycle_id"] == cycle_id)
        ]
        self.assertGreater(len(frame), 0, f"no rows for {cell_id} cycle {cycle_id}")
        return self.bt.extract_cycle_features(frame, frame["chemistry"].iloc[0]), frame

    def build(self, cell_id, cycle_id, chemistry, *, with_current=True):
        features, frame = self.features_for(cell_id, cycle_id)
        soh_label = float(frame["soh_pct"].iloc[0])
        if with_current:
            features["current_soh_pct"] = soh_label
        contract = ia.build_health_result(
            self.predictor,
            features,
            chemistry,
            {
                "cell_id": cell_id,
                "dataset_id": str(frame["dataset_id"].iloc[0]),
                "cycle_id": cycle_id,
                "cycle_position": 0,
                "prediction_timestamp": "2026-10-29T00:00:00Z",
            },
        )
        return contract, soh_label

    def test_real_nmc_cycle_produces_a_complete_handoff(self):
        contract, soh_label = self.build("B0005", 100, "NMC")
        self.assertEqual(contract["status"], "ok_experimental", contract["errors"])
        self.assertAlmostEqual(contract["soh"]["current_observed_soh_pct"], soh_label, places=9)
        handoff = ia.build_range_arguments(
            contract, ambient_c=25.0, hvac_power_kw=0.5
        )
        self.assertEqual(handoff["status"], "ok_experimental")
        self.assertEqual(handoff["arguments"]["cycle"], 100)
        self.assertAlmostEqual(
            handoff["arguments"]["soh_pct"], contract["soh"]["predicted_soh_pct"]
        )
        # The prediction is shown next to the real next-record label, not asserted
        # as accurate: the model is experimental.
        nxt = self.frames[
            (self.frames["cell_id"] == "B0005") & (self.frames["cycle_id"] == 101)
        ]
        self.assertGreater(len(nxt), 0)

    def test_real_lfp_cycle_produces_a_complete_handoff(self):
        contract, _ = self.build("BATTERY_5", 499, "LFP")
        self.assertEqual(contract["status"], "ok_experimental", contract["errors"])
        self.assertEqual(contract["soh"]["model_id"], "lfp_lgbm")
        self.assertIsNone(contract["rul"]["rul_cycles"])

    def test_current_soh_pct_cannot_be_derived_from_telemetry(self):
        # Sufficiency boundary: current_soh_pct comes from the measured SOH label,
        # not from the discharge curve. Without an upstream SOH estimate the
        # handoff is blocked rather than silently defaulted.
        contract, _ = self.build("B0005", 100, "NMC", with_current=False)
        self.assertEqual(contract["status"], "error")
        self.assertEqual(contract["errors"][0]["code"], "MISSING_FEATURES")
        self.assertIn("current_soh_pct", contract["errors"][0]["message"])
        self.assertIsNone(contract["soh"]["predicted_soh_pct"])


class RangeHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = load_predictor()
        cls.contract = nmc_contract(cls.predictor)

    def test_arguments_are_exactly_the_four_provisional_parameters(self):
        handoff = ia.build_range_arguments(
            self.contract,
            ambient_c=ILLUSTRATIVE_AMBIENT_C,
            hvac_power_kw=ILLUSTRATIVE_HVAC_KW,
        )
        self.assertEqual(handoff["status"], "ok_experimental")
        self.assertEqual(
            set(handoff["arguments"]),
            {"soh_pct", "cycle", "ambient_c", "hvac_power_kw"},
        )
        self.assertAlmostEqual(
            handoff["arguments"]["soh_pct"], self.contract["soh"]["predicted_soh_pct"]
        )
        self.assertEqual(handoff["arguments"]["cycle"], 1)
        self.assertEqual(handoff["arguments"]["ambient_c"], ILLUSTRATIVE_AMBIENT_C)
        self.assertEqual(handoff["arguments"]["hvac_power_kw"], ILLUSTRATIVE_HVAC_KW)
        self.assertIs(handoff["signature_verified"], False)
        self.assertIn("cycle_position", handoff["cycle_note"])
        self.assertEqual(handoff["cycle_semantics"], "observed_cycle_id")
        self.assertIn("current_observed_soh_pct", handoff["soh_semantics"])
        self.assertIs(handoff["production_ready"], False)
        self.assertIn("linearly scales usable pack energy", handoff["assumption"])

    def test_missing_vehicle_scenario_inputs_blocks(self):
        handoff = ia.build_range_arguments(self.contract)
        self.assertEqual(handoff["status"], "blocked")
        self.assertEqual(handoff["reason"], "MISSING_INPUTS")
        self.assertEqual(handoff["missing"], ["ambient_c", "hvac_power_kw"])
        self.assertIsNone(handoff["arguments"])

    def test_invalid_scenario_inputs_block(self):
        handoff = ia.build_range_arguments(
            self.contract, ambient_c="25", hvac_power_kw=0.5
        )
        self.assertEqual(handoff["reason"], "INVALID_SCENARIO_INPUT")
        handoff = ia.build_range_arguments(
            self.contract, ambient_c=25.0, hvac_power_kw=-0.1
        )
        self.assertEqual(handoff["reason"], "INVALID_SCENARIO_INPUT")

    def test_unsupported_contract_cannot_reach_range(self):
        naion = ia.build_health_result(
            self.predictor, sample_features(self.predictor), "Na-ion", SAMPLE_PROVENANCE
        )
        handoff = ia.build_range_arguments(
            naion, ambient_c=25.0, hvac_power_kw=0.5
        )
        self.assertEqual(handoff["status"], "blocked")
        self.assertEqual(handoff["reason"], "SOH_UNAVAILABLE")

    def test_handoff_carries_no_energy_or_range_values(self):
        # Member 2 stops at the argument mapping. No pack energy, cell capacity
        # or range figure may cross the boundary; computing them is Member 3 work.
        contract = nmc_contract(self.predictor, include_capacity=True)
        handoff = ia.build_range_arguments(
            contract, ambient_c=ILLUSTRATIVE_AMBIENT_C, hvac_power_kw=ILLUSTRATIVE_HVAC_KW
        )
        self.assertEqual(handoff["status"], "ok_experimental")
        self.assertEqual(
            handoff["arguments"]["soh_pct"], contract["soh"]["predicted_soh_pct"]
        )
        for forbidden in (
            "pack_kwh",
            "usable_pack_kwh",
            "capacity_ah",
            "range_km",
            "energy_kwh",
        ):
            self.assertNotIn(forbidden, handoff["arguments"])
        self.assertEqual(
            set(handoff),
            {
                "status",
                "arguments",
                "missing",
                "signature_verified",
                "signature_note",
                "cycle_note",
                "cycle_semantics",
                "soh_semantics",
                "assumption",
                "production_ready",
            },
        )
        json.dumps(handoff)


if __name__ == "__main__":
    unittest.main(verbosity=2)
