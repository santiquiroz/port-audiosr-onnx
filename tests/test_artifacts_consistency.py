import json

import manifest
import pytest

pytestmark = pytest.mark.needs_artifacts


@pytest.fixture
def pack(artifacts_dir):
    return json.loads((artifacts_dir / manifest.MANIFEST).read_text(encoding="utf-8"))


def test_manifest_lists_itself(pack):
    assert manifest.MANIFEST in pack["required_files"]


def test_every_required_file_exists(artifacts_dir, pack):
    assert manifest.missing_files(artifacts_dir, pack) == []


def test_every_artifact_is_required_or_a_validation_fixture(artifacts_dir, pack):
    assert manifest.unlisted_files(artifacts_dir, pack) == []
