from types import SimpleNamespace

import pytest
import validate_ort

CPU = "CPUExecutionProvider"
DML = "DmlExecutionProvider"


def fake_ort(*providers):
    return SimpleNamespace(get_available_providers=lambda: list(providers))


def fake_run(cpu_err=0.0, dml_err=0.0, dml_exc=None):
    def run(name, providers, label, **_):
        if providers == [CPU]:
            return 10.0, cpu_err
        if dml_exc is not None:
            raise dml_exc
        return 5.0, dml_err

    return run


def exit_code(monkeypatch, argv, available, run):
    monkeypatch.setattr(validate_ort, "ort", fake_ort(*available))
    monkeypatch.setattr(validate_ort, "run", run)
    with pytest.raises(SystemExit) as exc:
        validate_ort.main(argv)
    return exc.value.code


def test_dml_parity_regression_exits_nonzero(monkeypatch):
    code = exit_code(monkeypatch, ["ddpm"], [DML, CPU], fake_run(dml_err=0.5))
    assert code == 1


def test_dml_exception_with_require_dml_exits_nonzero(monkeypatch):
    run = fake_run(dml_exc=RuntimeError("DML device lost"))
    code = exit_code(monkeypatch, ["ddpm", "--require-dml"], [DML, CPU], run)
    assert code == 1


def test_dml_exception_without_flag_still_exits_nonzero(monkeypatch):
    run = fake_run(dml_exc=RuntimeError("DML device lost"))
    code = exit_code(monkeypatch, ["ddpm"], [DML, CPU], run)
    assert code == 1


def test_missing_dml_without_flag_is_skipped(monkeypatch, capsys):
    code = exit_code(monkeypatch, ["ddpm"], [CPU], fake_run())
    assert code == 0
    assert "SKIPPED" in capsys.readouterr().out


def test_missing_dml_never_runs_dml_session(monkeypatch):
    calls = []

    def run(name, providers, label, **_):
        calls.append(providers)
        return 10.0, 0.0

    exit_code(monkeypatch, ["ddpm"], [CPU], run)
    assert calls == [[CPU]]


def test_missing_dml_with_require_dml_exits_nonzero(monkeypatch):
    code = exit_code(monkeypatch, ["ddpm", "--require-dml"], [CPU], fake_run())
    assert code == 1


def test_all_graphs_ok_exits_zero(monkeypatch):
    code = exit_code(monkeypatch, ["vocoder", "ddpm"], [DML, CPU], fake_run())
    assert code == 0


def test_cpu_regression_keeps_checking_other_graphs(monkeypatch, capsys):
    code = exit_code(monkeypatch, ["vocoder", "ddpm"], [DML, CPU], fake_run(cpu_err=0.5))
    out = capsys.readouterr().out
    assert code == 1
    assert "[vocoder]" in out
    assert "[ddpm]" in out


def test_no_exported_graphs_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(validate_ort, "exported_graphs", list)
    code = exit_code(monkeypatch, [], [DML, CPU], fake_run())
    assert code == 1
    assert "no exported graphs" in capsys.readouterr().out


def test_cpu_nan_rel_err_exits_nonzero(monkeypatch):
    code = exit_code(monkeypatch, ["ddpm"], [DML, CPU], fake_run(cpu_err=float("nan")))
    assert code == 1


def test_dml_nan_rel_err_exits_nonzero(monkeypatch):
    code = exit_code(monkeypatch, ["ddpm"], [DML, CPU], fake_run(dml_err=float("nan")))
    assert code == 1
