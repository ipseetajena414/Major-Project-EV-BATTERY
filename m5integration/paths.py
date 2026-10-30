"""Repository paths and sys.path setup shared by all Member 5 adapters.

The package is called ``m5integration`` (not ``integration``) because Member 3 ships
its own ``member3/integration`` package and a second top-level ``integration`` would collide.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER_TABLE = ROOT / "master_table.parquet"
M2_DIR = ROOT / "member2_modeling"
M3_DIR = ROOT / "member3"
M4_DIR = ROOT / "ev_member4"
VEHICLE_DIR = M3_DIR / "vehicle_configs"
CYCLE_DIR = M3_DIR / "drive_cycles"
RAG_CORPUS_DIR = ROOT / "data" / "rag_corpus"
RAG_INDEX_DIR = ROOT / "data" / "rag_index"


def ensure_repo_on_path() -> None:
    """Repo root first so ``member2_modeling`` and ``member3`` resolve to the canonical copies."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
