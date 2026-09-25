import importlib
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = ["validate_ort", "validate_driver", "bench_dml"]
UPFLOW_MODULES = {
    "app": {},
    "app.services": {},
    "app.services.engines": {},
    "app.services.engines.audiosr": {"dsp": None},
    "app.services.engines.audiosr.dsp": {},
    "app.services.engines.audiosr.assets": {"AudioSrAssets": object},
    "app.services.engines.audiosr.driver": {"AudioSrDriver": object},
}
DEFAULT_ART = Path(__file__).resolve().parent.parent / "artifacts"


def stub_upflow(monkeypatch, tmp_path):
    monkeypatch.setenv("UPFLOW_ROOT", str(tmp_path / "no-upflow"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name, attrs in UPFLOW_MODULES.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**attrs))


def fresh_import(monkeypatch, name):
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


@pytest.fixture
def load_script(monkeypatch, tmp_path):
    stub_upflow(monkeypatch, tmp_path)
    return lambda name: fresh_import(monkeypatch, name)


@pytest.mark.parametrize("script", SCRIPTS)
def test_art_follows_audiosr_artifacts(monkeypatch, tmp_path, load_script, script):
    monkeypatch.setenv("AUDIOSR_ARTIFACTS", str(tmp_path))
    assert load_script(script).ART == tmp_path


@pytest.mark.parametrize("script", SCRIPTS)
def test_art_defaults_to_repo_artifacts(monkeypatch, load_script, script):
    monkeypatch.delenv("AUDIOSR_ARTIFACTS", raising=False)
    assert load_script(script).ART == DEFAULT_ART


@pytest.mark.parametrize("script", SCRIPTS)
def test_empty_audiosr_artifacts_falls_back_to_default(monkeypatch, load_script, script):
    monkeypatch.setenv("AUDIOSR_ARTIFACTS", "")
    assert load_script(script).ART == DEFAULT_ART


def test_validate_driver_keeps_repo_baseline(monkeypatch, tmp_path, load_script):
    monkeypatch.setenv("AUDIOSR_ARTIFACTS", str(tmp_path))
    assert load_script("validate_driver").BASE == DEFAULT_ART.parent / "refs" / "baseline"


def test_validate_ort_discovers_graphs_in_override(monkeypatch, tmp_path, load_script):
    (tmp_path / "vocoder.onnx").write_bytes(b"")
    monkeypatch.setenv("AUDIOSR_ARTIFACTS", str(tmp_path))
    assert load_script("validate_ort").exported_graphs() == ["vocoder"]
