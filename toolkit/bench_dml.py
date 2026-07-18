"""End-to-end driver benchmark on DirectML vs CPU (RX 7800 XT target) +
optional fp16 UNet measurement.

Runs the full torch-free pipeline on a 10.24 s clip (latent T=128, the native
window - also exercises the DirectML large-tensor behavior beyond the T=64
export shapes).

Usage: python toolkit/bench_dml.py [--fp16] [steps]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
UPFLOW = Path(os.environ.get(
    "UPFLOW_ROOT", Path.home() / ".openclaw" / "workspace" / "image-upscaler-amd"
))
sys.path.insert(0, str(UPFLOW))

from app.services.engines.audiosr.assets import AudioSrAssets  # noqa: E402
from app.services.engines.audiosr.driver import AudioSrDriver  # noqa: E402


def make_run_graph(providers, ddpm_path=None, timings=None):
    sessions = {}

    def run_graph(name, feeds):
        if name not in sessions:
            path = ddpm_path if (name == "ddpm" and ddpm_path) else ART / f"{name}.onnx"
            opts = ort.SessionOptions()
            sessions[name] = ort.InferenceSession(str(path), opts, providers=providers)
        sess = sessions[name]
        feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
        if sess.get_inputs()[0].type == "tensor(float16)" and name == "ddpm":
            feeds = {k: v.astype(np.float16) if v.dtype == np.float32 else v
                     for k, v in feeds.items()}
        t0 = time.perf_counter()
        out = sess.run(None, feeds)[0]
        if timings is not None:
            timings.setdefault(name, []).append(time.perf_counter() - t0)
        return np.asarray(out, dtype=np.float32)

    return run_graph


def run_once(label, providers, wav, steps, ddpm_path=None, seed=42):
    assets = AudioSrAssets.load(ART)
    timings = {}
    driver = AudioSrDriver(assets, make_run_graph(providers, ddpm_path, timings), seed=seed)
    t0 = time.perf_counter()
    out = driver.restore(wav, ddim_steps=steps)
    wall = time.perf_counter() - t0
    clip_s = wav.shape[-1] / 48000
    print(f"[{label}] wall {wall:.1f}s for {clip_s:.2f}s clip  (RTF {wall / clip_s:.2f})")
    for name, ts in sorted(timings.items()):
        print(f"    {name}: n={len(ts)} avg {1000 * np.mean(ts):.1f} ms  total {sum(ts):.1f}s")
    return out


def main():
    fp16 = "--fp16" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    steps = int(args[0]) if args else 50

    wav, sr = sf.read(str(ROOT / "refs" / "degraded_up48k.wav"), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    assert sr == 48000

    out_dml = run_once("DirectML fp32", ["DmlExecutionProvider"], wav, steps)
    sf.write(str(ROOT / "refs" / "bench_dml_out.wav"), out_dml, 48000)

    out_cpu = run_once("CPU fp32", ["CPUExecutionProvider"], wav, steps)
    n = min(out_dml.shape[-1], out_cpu.shape[-1])
    diff = np.abs(out_dml[:n] - out_cpu[:n]).max()
    rms = np.sqrt(np.mean((out_dml[:n] - out_cpu[:n]) ** 2))
    print(f"[DML vs CPU] maxdiff {diff:.5f}  rms {rms:.6f}")

    if fp16:
        fp16_path = ART / "ddpm_fp16.onnx"
        if not fp16_path.exists():
            print("[fp16] converting ddpm...")
            import onnx
            from onnxconverter_common import float16

            model = onnx.load(str(ART / "ddpm.onnx"))
            # The dynamo-exported graph carries Cast/_to_copy nodes the
            # converter mistypes unless they are blocked from conversion.
            model16 = float16.convert_float_to_float16(
                model, keep_io_types=True, disable_shape_infer=True,
                op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + ["Cast"],
            )
            onnx.save(model16, str(fp16_path),
                      save_as_external_data=True, location="ddpm_fp16.onnx.data")
        out16 = run_once("DirectML fp16-UNet", ["DmlExecutionProvider"], wav, steps,
                         ddpm_path=fp16_path)
        n = min(out16.shape[-1], out_dml.shape[-1])
        diff = np.abs(out16[:n] - out_dml[:n]).max()
        print(f"[fp16 vs fp32 DML] maxdiff {diff:.5f}")
        sf.write(str(ROOT / "refs" / "bench_dml_fp16_out.wav"), out16, 48000)


if __name__ == "__main__":
    main()
