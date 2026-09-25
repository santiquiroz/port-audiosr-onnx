import pytest
from test_artifacts_dir import fresh_import, stub_upflow


@pytest.fixture
def driver(monkeypatch, tmp_path):
    stub_upflow(monkeypatch, tmp_path)
    module = fresh_import(monkeypatch, "validate_driver")
    monkeypatch.setattr(module, "FAILURES", [])
    return module


def test_check_records_nan_rel_err_as_failure(driver, capsys):
    driver.check("stage", [float("nan"), 1.0], [1.0, 1.0], tol=1e-3)
    assert driver.FAILURES == ["stage"]
    assert "FAIL" in capsys.readouterr().out


def test_check_records_rel_err_above_tol(driver):
    driver.check("stage", [1.5, 1.0], [1.0, 1.0], tol=1e-3)
    assert driver.FAILURES == ["stage"]


def test_check_accepts_rel_err_within_tol(driver, capsys):
    driver.check("stage", [1.0, 1.0], [1.0, 1.0], tol=1e-3)
    assert driver.FAILURES == []
    assert "ok" in capsys.readouterr().out
