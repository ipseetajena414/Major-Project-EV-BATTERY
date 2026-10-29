import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from range_calculator import RangeCalculator, VehicleConfig
from integration.member2_range_adapter import (
    BASELINE_QUALITY_FLAG,
    DEFAULT_QUALITY_FLAG,
    PredictionArtifactError,
    TrajectoryValidationError,
    build_soh_trajectory,
    estimate_range_trajectory,
    export_results_csv,
    load_prediction_artifact,
    run_pipeline,
)

BASE_COLUMNS = [
    "cycle_index", "cell_id", "chemistry", "soh_pct", "quality_flag",
    "source", "model_name", "model_status", "split",
    "current_observed_soh_pct", "observed_next_soh_pct", "unit",
]


def sample_rows(n=5, start=10, soh=80.0, chemistry="NMC", source="predicted", **overrides):
    rows = []
    for i in range(n):
        row = {
            "cycle_index": start + i,
            "cell_id": "B0005",
            "chemistry": chemistry,
            "soh_pct": soh - i,
            "quality_flag": "TEST_FLAG",
            "source": source,
            "model_name": "nmc_bilstm",
            "model_status": "experimental_research_stage",
            "split": "test",
            "current_observed_soh_pct": soh - i + 0.5,
            "observed_next_soh_pct": soh - i - 0.1,
            "unit": "%",
        }
        row.update(overrides)
        rows.append(row)
    return rows


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def sample_cycle():
    points = []
    speed = 0.0
    for t in range(0, 61):
        if 5 <= t < 20:
            speed = min(15.0, (t - 5) * 1.0)
        elif 20 <= t < 40:
            speed = 15.0
        elif 40 <= t < 55:
            speed = max(0.0, 15.0 - (t - 40) * 1.0)
        else:
            speed = 0.0
        points.append({"time_s": float(t), "speed_mps": speed, "grade_pct": 0.0})
    return points


class AdapterUnitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.config = VehicleConfig.from_json(ROOT / "vehicle_configs" / "illustrative_placeholder.json")
        self.calculator = RangeCalculator(self.config)
        self.cycle = sample_cycle()

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_artifact_preserves_identity_and_order(self):
        path = write_csv(self.dir / "pred.csv", sample_rows(4))
        rows = load_prediction_artifact(path)
        self.assertEqual([r["cycle_index"] for r in rows], [10, 11, 12, 13])
        self.assertEqual({r["cell_id"] for r in rows}, {"B0005"})
        self.assertEqual({r["chemistry"] for r in rows}, {"NMC"})
        self.assertEqual(rows[0]["source"], "predicted")

    def test_missing_artifact_raises_clear_error(self):
        with self.assertRaises(PredictionArtifactError) as ctx:
            load_prediction_artifact(self.dir / "absent.csv")
        self.assertIn("not found", str(ctx.exception))
        self.assertIn("Member 2 inference", str(ctx.exception))

    def test_missing_required_column_rejected(self):
        rows = sample_rows(3)
        for row in rows:
            row.pop("chemistry")
            row.pop("cell_id")
        path = write_csv(self.dir / "bad.csv", rows, columns=[c for c in BASE_COLUMNS if c not in ("chemistry", "cell_id")])
        with self.assertRaises(PredictionArtifactError) as ctx:
            load_prediction_artifact(path)
        self.assertIn("chemistry", str(ctx.exception))

    def test_fractional_soh_rejected(self):
        path = write_csv(self.dir / "fraction.csv", sample_rows(3, soh=0.85))
        with self.assertRaises(TrajectoryValidationError) as ctx:
            load_prediction_artifact(path)
        self.assertIn("fraction", str(ctx.exception))

    def test_out_of_bounds_and_non_finite_soh_rejected(self):
        for bad in (150.0, -5.0, float("nan"), float("inf")):
            rows = sample_rows(2)
            rows[0]["soh_pct"] = bad
            path = write_csv(self.dir / f"bad_{bad}.csv".replace(".", "_"), rows)
            with self.assertRaises(TrajectoryValidationError):
                load_prediction_artifact(path)

    def test_unsupported_chemistry_rejected(self):
        path = write_csv(self.dir / "na.csv", sample_rows(2, chemistry="Na-ion"))
        with self.assertRaises(PredictionArtifactError) as ctx:
            load_prediction_artifact(path)
        self.assertIn("not supported", str(ctx.exception))

    def test_unsupported_unit_rejected(self):
        path = write_csv(self.dir / "unit.csv", sample_rows(2, unit="fraction"))
        with self.assertRaises(PredictionArtifactError) as ctx:
            load_prediction_artifact(path)
        self.assertIn("unit", str(ctx.exception))

    def test_duplicate_cycle_index_rejected(self):
        rows = sample_rows(2)
        rows[1]["cycle_index"] = rows[0]["cycle_index"]
        path = write_csv(self.dir / "dupe.csv", rows)
        with self.assertRaises(PredictionArtifactError) as ctx:
            load_prediction_artifact(path)
        self.assertIn("duplicate", str(ctx.exception))

    def test_non_monotonic_order_rejected(self):
        rows = sample_rows(3)
        rows[0]["cycle_index"], rows[2]["cycle_index"] = rows[2]["cycle_index"], rows[0]["cycle_index"]
        path = write_csv(self.dir / "order.csv", rows)
        with self.assertRaises(PredictionArtifactError) as ctx:
            load_prediction_artifact(path)
        self.assertIn("strictly increasing", str(ctx.exception))

    def test_trajectory_enters_estimator_with_flags(self):
        rows = load_prediction_artifact(write_csv(self.dir / "ok.csv", sample_rows(4)))
        results = estimate_range_trajectory(self.calculator, self.cycle, rows, ambient_c=20.0)
        self.assertEqual(len(results), 4)
        self.assertEqual([r["cycle_index"] for r in results], [10, 11, 12, 13])
        self.assertEqual({r["quality_flag"] for r in results}, {"TEST_FLAG"})
        self.assertEqual({r["cell_id"] for r in results}, {"B0005"})
        self.assertEqual({r["chemistry"] for r in results}, {"NMC"})
        for r in results:
            self.assertTrue(math.isfinite(r["estimated_range_km"]))
            self.assertGreater(r["estimated_range_km"], 0)
            self.assertEqual(r["estimate_quality_status"], "ILLUSTRATIVE_UNVALIDATED")

    def test_default_quality_flag_when_column_absent(self):
        rows = sample_rows(2)
        for row in rows:
            row.pop("quality_flag")
        path = write_csv(self.dir / "default.csv", rows, columns=[c for c in BASE_COLUMNS if c != "quality_flag"])
        loaded = load_prediction_artifact(path)
        trajectory = build_soh_trajectory(loaded)
        self.assertEqual({t["quality_flag"] for t in trajectory}, {DEFAULT_QUALITY_FLAG})

    def test_observed_baseline_mode_requires_column_and_labels_source(self):
        path = write_csv(self.dir / "obs.csv", sample_rows(3))
        rows = load_prediction_artifact(path)
        results = estimate_range_trajectory(
            self.calculator, self.cycle, rows, soh_field="current_observed_soh_pct"
        )
        self.assertEqual({r["quality_flag"] for r in results}, {BASELINE_QUALITY_FLAG})
        self.assertEqual({r["soh_source"] for r in results}, {"observed_anchor"})
        self.assertEqual(
            [r["soh_pct"] for r in results], [80.5, 79.5, 78.5]
        )

    def test_export_schema_and_row_count(self):
        rows = load_prediction_artifact(write_csv(self.dir / "ok.csv", sample_rows(3)))
        results = estimate_range_trajectory(self.calculator, self.cycle, rows, ambient_c=25.0)
        out = export_results_csv(results, self.dir / "out.csv", "config.json", "cycle.csv")
        with open(out, newline="", encoding="utf-8") as handle:
            exported = list(csv.DictReader(handle))
        self.assertEqual(len(exported), 3)
        required = {
            "cycle_index", "cell_id", "chemistry", "soh_pct", "quality_flag",
            "estimated_range_km", "consumption_kwh_per_100km", "soh_source",
        }
        self.assertTrue(required.issubset(exported[0].keys()))
        self.assertEqual([int(r["cycle_index"]) for r in exported], [10, 11, 12])
        self.assertEqual(exported[0]["vehicle_config"], "config.json")

    def test_pipeline_missing_vehicle_config_error(self):
        path = write_csv(self.dir / "ok.csv", sample_rows(3))
        with self.assertRaises(PredictionArtifactError) as ctx:
            run_pipeline(
                path, self.dir / "no_vehicle.json", ROOT / "drive_cycles" / "example_cycle.csv",
                self.dir / "out.csv",
            )
        self.assertIn("Vehicle configuration", str(ctx.exception))

    def test_pipeline_missing_cycle_error(self):
        path = write_csv(self.dir / "ok.csv", sample_rows(3))
        with self.assertRaises(PredictionArtifactError) as ctx:
            run_pipeline(
                path, ROOT / "vehicle_configs" / "illustrative_placeholder.json",
                self.dir / "no_cycle.csv", self.dir / "out.csv",
            )
        self.assertIn("Drive cycle", str(ctx.exception))

    def test_pipeline_end_to_end_counts(self):
        path = write_csv(self.dir / "ok.csv", sample_rows(6))
        summary = run_pipeline(
            path,
            ROOT / "vehicle_configs" / "illustrative_placeholder.json",
            ROOT / "drive_cycles" / "example_cycle.csv",
            self.dir / "out.csv",
            ambient_c=25.0,
        )
        self.assertEqual(summary["rows"], 6)
        self.assertEqual(summary["soh_source"], "predicted")
        with open(summary["output_csv"], newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)


REAL_BILSTM = ROOT / "integration" / "predictions" / "member2_nmc_B0005_bilstm_predictions.csv"
REAL_LGBM = ROOT / "integration" / "predictions" / "member2_nmc_B0005_lgbm_predictions.csv"


class GenuineMember2ArtifactTests(unittest.TestCase):
    @unittest.skipUnless(REAL_BILSTM.is_file(), "genuine Member 2 BiLSTM artifact not present")
    def test_bilstm_artifact_schema(self):
        rows = load_prediction_artifact(REAL_BILSTM)
        self.assertEqual(len(rows), 30)
        self.assertEqual({r["chemistry"] for r in rows}, {"NMC"})
        self.assertEqual({r["model_name"] for r in rows}, {"nmc_bilstm"})
        self.assertEqual({r["source"] for r in rows}, {"predicted"})
        self.assertEqual([r["cycle_index"] for r in rows], sorted(r["cycle_index"] for r in rows))
        self.assertTrue(all(1.0 < r["soh_pct"] <= 100.0 for r in rows))

    @unittest.skipUnless(REAL_LGBM.is_file(), "genuine Member 2 LightGBM artifact not present")
    def test_lgbm_artifact_schema(self):
        rows = load_prediction_artifact(REAL_LGBM)
        self.assertEqual(len(rows), 30)
        self.assertEqual({r["model_name"] for r in rows}, {"nmc_lgbm"})


if __name__ == "__main__":
    unittest.main()
