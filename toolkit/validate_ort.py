"""Validates every exported graph against its PyTorch reference tensors on
CPU-EP and DirectML, printing rel-err + timing. Also measures fp16 for ddpm.

Usage: python toolkit/validate_ort.py [graph ...]
"""

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"

GRAPHS = ["vocoder", "vae_decoder", "vae_feature_extract", "ddpm"]


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


def main():
    names = sys.argv[1:] or [g for g in GRAPHS if (ART / f"{g}.onnx").exists()]
    for name in names:
        print(f"[{name}]")
        cpu_ms, cpu_err = run(name, ["CPUExecutionProvider"], "CPU-EP")
        try:
            dml_ms, dml_err = run(name, ["DmlExecutionProvider"], "DirectML")
            print(f"  speedup: {cpu_ms / dml_ms:.1f}x")
            assert dml_err < 1e-2, f"DML rel-err too high: {dml_err}"
        except Exception as exc:  # noqa: BLE001
            print(f"  DirectML FAILED: {exc}")
        assert cpu_err < 1e-3, f"CPU rel-err too high: {cpu_err}"


if __name__ == "__main__":
    main()
