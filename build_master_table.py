"""Deprecated entry point.

The previous implementation overwrote ``master_table.parquet`` with a fused
cell-cycle table in which Na-ion voltage and current were hardcoded constants
and Na-ion SOH was derived from resistance. That table is retired and archived
in ``legacy/member1/``.

This script now delegates to the active pipeline, which writes the three
processed snapshot datasets to ``DATA/processed/`` and never writes a unified
master table.
"""

from __future__ import annotations

import sys

from member1_pipeline.build import main

if __name__ == "__main__":
    print(
        "DEPRECATED: build_master_table.py no longer writes master_table.parquet.\n"
        "Delegating to member1_pipeline.build (three separate processed datasets).\n"
    )
    sys.exit(main())
