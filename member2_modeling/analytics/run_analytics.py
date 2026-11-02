"""CLI entry point for the Member 5 analytics package.

Usage:
    python -m member2_modeling.analytics.run_analytics [--rebuild-rul]
        [--out-dir PATH] [--eol-pct 80]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from member2_modeling.analytics.export import ANALYTICS_DIR, build_analytics_package
from member2_modeling.rul.labels import EOL_PCT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ANALYTICS_DIR,
        help="Destination directory for analytics_v1 artifacts.",
    )
    parser.add_argument(
        "--rebuild-rul",
        action="store_true",
        help="Recompute RUL artifacts before assembling the package.",
    )
    parser.add_argument(
        "--eol-pct",
        type=float,
        default=EOL_PCT,
        help="End-of-life SOH threshold percentage (default: 80).",
    )
    args = parser.parse_args(argv)

    summary = build_analytics_package(
        out_dir=args.out_dir,
        rebuild_rul=args.rebuild_rul,
        eol_pct=args.eol_pct,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
