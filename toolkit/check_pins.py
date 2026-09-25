"""Check that the export venv matches toolkit/constraints.txt.

Usage: python toolkit/check_pins.py [constraints.txt]
"""
import re
import sys
from importlib import metadata
from pathlib import Path

DEFAULT_CONSTRAINTS = Path(__file__).resolve().parent / "constraints.txt"


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pin(line):
    spec = line.split("#", 1)[0].strip()
    if "==" not in spec:
        return None
    name, version = spec.split("==", 1)
    return canonical_name(name.strip()), version.strip()


def parse_pins(text):
    pins = (parse_pin(line) for line in text.splitlines())
    return dict(pin for pin in pins if pin)


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def pin_mismatches(pins, get_installed):
    found = ((name, pinned, get_installed(name)) for name, pinned in pins.items())
    return [
        f"{name}: pinned {pinned}, installed {actual}"
        for name, pinned, actual in found
        if actual is not None and actual != pinned
    ]


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_CONSTRAINTS
    pins = parse_pins(path.read_text())
    mismatches = pin_mismatches(pins, installed_version)
    for line in mismatches:
        print("PIN MISMATCH", line)
    if mismatches:
        return 1
    print(f"pins OK ({len(pins)} constraints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
