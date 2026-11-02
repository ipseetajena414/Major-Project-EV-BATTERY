"""Command-line access to the snapshot inference interface.

Examples:
    python -m member2_modeling.snapshot_ml.cli info --task ev_soh
    python -m member2_modeling.snapshot_ml.cli predict --task ev_failure --input records.json
    python -m member2_modeling.snapshot_ml.cli range --input ev_records.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inference import InferenceInputError, SnapshotInferenceAPI
from .member3_bundle import build_bundle
from .range_estimate import RangeEstimateError, estimate_range_from_record

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "snapshot_ml"


def _records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise SystemExit("Input JSON must be an object or a list of objects.")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info")
    info.add_argument("--task", required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--task", required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--allow-unseen-chemistry", action="store_true")
    predict.add_argument("--reject-missing-values", action="store_true")

    range_parser = sub.add_parser("range")
    range_parser.add_argument("--input", type=Path, required=True)

    sub.add_parser("bundle")

    args = parser.parse_args(argv)
    if args.command == "bundle":
        result = build_bundle()
        print(json.dumps(result["validation"], indent=2))
        return 0 if result["validation"]["ok"] else 1

    api = SnapshotInferenceAPI(args.registry)
    if args.command == "info":
        print(json.dumps(api.info(args.task), indent=2, default=str))
        return 0
    if args.command == "predict":
        try:
            result = api.predict(
                args.task,
                _records(args.input),
                allow_unseen_chemistry=args.allow_unseen_chemistry,
                allow_missing_values=not args.reject_missing_values,
            )
        except (InferenceInputError, KeyError) as exc:
            print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, indent=2))
            return 2
        print(json.dumps(result, indent=2))
        return 0
    estimates = []
    for index, record in enumerate(_records(args.input)):
        try:
            estimates.append(estimate_range_from_record(record))
        except RangeEstimateError as exc:
            print(json.dumps({"row": index, "error": str(exc)}, indent=2))
            return 2
    print(json.dumps(estimates, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
