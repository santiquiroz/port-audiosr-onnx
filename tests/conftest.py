import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "toolkit"))


def artifacts_path() -> Path:
    return Path(os.environ.get("AUDIOSR_ARTIFACTS") or ROOT / "artifacts")


def has_graphs(folder: Path) -> bool:
    return any(folder.glob("*.onnx"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_artifacts: needs exported .onnx graphs in AUDIOSR_ARTIFACTS (default artifacts/)",
    )


def pytest_runtest_setup(item):
    folder = artifacts_path()
    if item.get_closest_marker("needs_artifacts") and not has_graphs(folder):
        pytest.skip(f"no exported .onnx graphs in {folder}")


@pytest.fixture
def artifacts_dir() -> Path:
    return artifacts_path()
