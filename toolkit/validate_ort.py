"""Validates every exported graph against its PyTorch reference tensors on
CPU-EP and DirectML, printing rel-err + timing. Exits 1 on any failure.

Usage: python toolkit/validate_ort.py [graph ...] [--require-dml]
Env:   AUDIOSR_ARTIFACTS (default: <repo>/artifacts)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
ART = Path(os.environ.get("AUDIOSR_ARTIFACTS") or ROOT / "artifacts")

GRAPHS = ["vocoder", "vae_decoder", "vae_feature_extract", "ddpm"]
CPU_EP = "CPUExecutionProvider"
DML_EP = "DmlExecutionProvider"
CPU_TOL = 1e-3
DML_TOL = 1e-2


def load_case(name):
    inputs = []
    i = 0
    while (ART / f"{name}_in{i}.npy").exists():
        inputs.append(np.load(ART / f"{name}_in{i}.npy"))
        i += 1
    ref = np.load(ART / f"{name}_ref.npy")
    return inputs, ref


def rel_err(a, b):
    denom = np.abs(b).max()
    if denom == 0:
        return float(np.abs(a - b).max())
    return float(np.abs(a - b).max() / denom)


def run(name, providers, label, n_warmup=1, n_timed=3):
    path = ART / f"{name}.onnx"
    inputs, ref = load_case(name)
    sess = ort.InferenceSession(str(path), providers=providers)
    feed = {inp.name: arr for inp, arr in zip(sess.get_inputs(), inputs)}
    for _ in range(n_warmup):
        out = sess.run(None, feed)[0]
    times = []
    for _ in range(n_timed):
        t0 = time.perf_counter()
        out = sess.run(None, feed)[0]
        times.append((time.perf_counter() - t0) * 1000)
    err = rel_err(out.astype(np.float32), ref.astype(np.float32))
    print(f"  {label:>9}: {min(times):8.1f} ms   rel-err {err:.6f}")
    return min(times), err


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Validate exported graphs on CPU-EP and DirectML.")
    parser.add_argument("graphs", nargs="*",
                        help="graphs to validate (default: all exported)")
    parser.add_argument("--require-dml", action="store_true",
                        help="fail instead of skipping when DirectML is unavailable")
    return parser.parse_args(argv)


def exported_graphs():
    return [g for g in GRAPHS if (ART / f"{g}.onnx").exists()]


def dml_available():
    # ORT silently falls back to CPU for a missing EP, so check before running.
    return DML_EP in ort.get_available_providers()


def check_cpu(name):
    cpu_ms, cpu_err = run(name, [CPU_EP], "CPU-EP")
    if cpu_err >= CPU_TOL:
        return cpu_ms, [f"{name}: CPU rel-err too high: {cpu_err}"]
    return cpu_ms, []


def check_dml(name, cpu_ms):
    try:
        dml_ms, dml_err = run(name, [DML_EP], "DirectML")
    except Exception as exc:  # noqa: BLE001
        return [f"{name}: DirectML run failed: {exc}"]
    print(f"  speedup: {cpu_ms / dml_ms:.1f}x")
    if dml_err >= DML_TOL:
        return [f"{name}: DML rel-err too high: {dml_err}"]
    return []


def missing_dml(name, require_dml):
    if require_dml:
        return [f"{name}: DirectML unavailable (--require-dml)"]
    print("  DirectML SKIPPED: DmlExecutionProvider not available")
    return []


def validate_graph(name, require_dml):
    print(f"[{name}]")
    cpu_ms, failures = check_cpu(name)
    if not dml_available():
        return failures + missing_dml(name, require_dml)
    return failures + check_dml(name, cpu_ms)


def validate_all(names, require_dml):
    if not names:
        return [f"no exported graphs found in {ART}"]
    return [f for name in names for f in validate_graph(name, require_dml)]


def report(failures):
    for failure in failures:
        print(f"FAILED {failure}")
    print(f"{len(failures)} failure(s)" if failures else "ALL OK")


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    names = args.graphs or exported_graphs()
    failures = validate_all(names, args.require_dml)
    report(failures)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
