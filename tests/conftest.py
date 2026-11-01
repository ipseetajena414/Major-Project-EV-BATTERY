from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from member1_pipeline import build  # noqa: E402
from member1_pipeline.config import PROCESSED_DIR  # noqa: E402


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def processed_dir() -> Path:
    return PROCESSED_DIR


@pytest.fixture(scope="session")
def rebuilt_outputs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the full pipeline once into an isolated directory."""
    out_dir = tmp_path_factory.mktemp("member1_rebuild")
    build.build_all(out_dir)
    return out_dir
