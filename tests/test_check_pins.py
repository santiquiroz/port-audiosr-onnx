from pathlib import Path

import check_pins
from packaging.requirements import Requirement

TOOLKIT = Path(__file__).resolve().parent.parent / "toolkit"
RELEASE_PINS = {
    "torch": "2.13.0",
    "onnx": "1.22.0",
    "onnxscript": "0.7.1",
    "onnxruntime-directml": "1.24.4",
}


def read_requirements():
    lines = (TOOLKIT / "requirements.txt").read_text().splitlines()
    return [Requirement(line) for line in lines if line.strip() and not line.startswith("#")]


def read_constraints():
    return check_pins.parse_pins((TOOLKIT / "constraints.txt").read_text())


def test_parse_pins_skips_comments_and_blank_lines():
    text = "# header\n\ntorch==2.13.0  # export\nnumpy==1.23.5\n"
    assert check_pins.parse_pins(text) == {"torch": "2.13.0", "numpy": "1.23.5"}


def test_parse_pins_canonicalizes_names():
    text = "Jinja2==3.1.6\ntyping_extensions==4.16.0\nonnx.ir==0.2.1\n"
    assert check_pins.parse_pins(text) == {
        "jinja2": "3.1.6",
        "typing-extensions": "4.16.0",
        "onnx-ir": "0.2.1",
    }


def test_pin_mismatches_empty_when_versions_match():
    pins = {"torch": "2.13.0", "onnx": "1.22.0"}
    assert check_pins.pin_mismatches(pins, pins.get) == []


def test_pin_mismatches_reports_drifted_version():
    pins = {"torch": "2.13.0", "onnx": "1.22.0"}
    installed = {"torch": "2.14.0", "onnx": "1.22.0"}
    assert check_pins.pin_mismatches(pins, installed.get) == ["torch: pinned 2.13.0, installed 2.14.0"]


def test_pin_mismatches_ignores_pins_that_are_not_installed():
    pins = {"torch": "2.13.0", "gradio": "6.17.3"}
    installed = {"torch": "2.13.0"}
    assert check_pins.pin_mismatches(pins, installed.get) == []


def test_requirements_list_onnxscript_for_the_dynamo_export():
    names = [check_pins.canonical_name(r.name) for r in read_requirements()]
    assert names.count("onnxscript") == 1


def test_constraints_pin_the_versions_that_built_the_releases():
    pins = read_constraints()
    assert {name: pins.get(name) for name in RELEASE_PINS} == RELEASE_PINS


def is_satisfied(requirement, pins):
    pinned = pins.get(check_pins.canonical_name(requirement.name))
    return pinned is not None and requirement.specifier.contains(pinned, prereleases=True)


def test_constraints_satisfy_every_requirement():
    pins = read_constraints()
    assert [str(r) for r in read_requirements() if not is_satisfied(r, pins)] == []


def test_constraints_leave_pip_unpinned():
    assert "pip" not in read_constraints()
