"""Validates the torch-free numpy driver (lives in Upflow) against the
instrumented PyTorch baseline captured by capture_baseline.py.

Feeds the driver the exact noise tensors the baseline drew, runs the ONNX
graphs on CPU-EP, and compares every stage boundary. This is the mandatory
parity gate before the driver ships inside Upflow.

Usage: .venv/Scripts/python.exe toolkit/validate_driver.py
Env:   UPFLOW_ROOT (default: ~/.openclaw/workspace/image-upscaler-amd)
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
BASE = ROOT / "refs" / "baseline"

UPFLOW = Path(os.environ.get(
    "UPFLOW_ROOT", Path.home() / ".openclaw" / "workspace" / "image-upscaler-amd"
))
sys.path.insert(0, str(UPFLOW))

from app.services.engines.audiosr import dsp  # noqa: E402
from app.services.engines.audiosr.assets import AudioSrAssets  # noqa: E402
from app.services.engines.audiosr.driver import AudioSrDriver  # noqa: E402

FAILURES = []


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.abs(b).max()
    if denom == 0:
        return float(np.abs(a - b).max())
    return float(np.abs(a - b).max() / denom)


def check(name, got, ref, tol, min_length=False):
    got = np.asarray(got, dtype=np.float64).squeeze()
    ref = np.asarray(ref, dtype=np.float64).squeeze()
    if min_length and got.ndim == 1 and ref.ndim == 1:
        n = min(got.shape[-1], ref.shape[-1])
        got, ref = got[:n], ref[:n]
    if got.shape != ref.shape:
        print(f"  {name:>28}: SHAPE MISMATCH {got.shape} vs {ref.shape}")
        FAILURES.append(name)
        return
    err = rel_err(got, ref)
    rms = float(np.sqrt(np.mean((got - ref) ** 2)) / max(np.sqrt(np.mean(ref**2)), 1e-12))
    status = "ok" if err <= tol else "FAIL"
    if err > tol:
        FAILURES.append(name)
    print(f"  {name:>28}: rel-err {err:.6f}  rms {rms:.6f}  (tol {tol})  {status}")


def load(name):
    return np.load(BASE / f"{name}.npy")


def make_run_graph():
    sessions = {}

    def run_graph(name, feeds):
        if name not in sessions:
            sessions[name] = ort.InferenceSession(
                str(ART / f"{name}.onnx"), providers=["CPUExecutionProvider"]
            )
        sess = sessions[name]
        feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
        return sess.run(None, feeds)[0]

    return run_graph


class ReplayNoise:
    """Feeds the driver the baseline's recorded torch.randn draws, in order.
    Baseline order: [0]=first-stage z sample (driver never needs it, skipped),
    [1]=cond sample, [2]=x_T, [3..]=one per DDIM step."""

    def __init__(self):
        self.index = 1
        self.total = len(list(BASE.glob("noise_*.npy")))

    def __call__(self, shape):
        arr = np.load(BASE / f"noise_{self.index:03d}.npy").astype(np.float32)
        assert tuple(arr.shape) == tuple(shape), f"noise#{self.index} {arr.shape} vs {shape}"
        self.index += 1
        return arr


def main():
    meta = json.loads((BASE / "meta.json").read_text())
    assets = AudioSrAssets.load(ART)
    run_graph = make_run_graph()

    wav = load("batch_waveform").squeeze().astype(np.float64)
    print(f"input: {wav.shape[-1]} samples, ddim_steps={meta['ddim_steps']}, "
          f"lowpass={meta['random_choices'][0]}")

    print("[stage: mel front-end]")
    stft = dsp.stft_magnitude(wav)
    target = int(round(wav.shape[-1] / 48000 * 100))
    check("stft (padded)", dsp.pad_spec(stft, target), load("batch_stft"), 5e-4)
    mel = dsp.pad_spec(dsp.log_mel(stft, assets.mel_basis), target)
    check("log_mel", mel, load("batch_log_mel_spec"), 5e-3)

    print("[stage: lowpass sim]")
    cutoff_hz = dsp.detect_cutoff_hz(dsp.pad_spec(stft, target))
    print(f"  cutoff: {cutoff_hz:.1f} Hz")
    wav_lp = dsp.lowpass_simulate(wav, cutoff_hz, meta["random_choices"][0])
    check("waveform_lowpass", wav_lp, load("batch_waveform_lowpass"), 5e-3)
    mel_lp = dsp.pad_spec(dsp.log_mel(dsp.stft_magnitude(wav_lp), assets.mel_basis), target)
    check("lowpass_mel", mel_lp, load("batch_lowpass_mel"), 5e-3)

    # From here on, isolate ONNX/driver error from front-end error by feeding
    # the driver the baseline's own tensors at each boundary.
    mel_lp_ref = load("batch_lowpass_mel").squeeze().astype(np.float32)

    print("[stage: conditioner]")
    noise_cond = np.load(BASE / "noise_001.npy").astype(np.float32)
    # max-norm on a 90M-param conv encoder output is outlier-dominated; the
    # rms column is the meaningful signal here (DDIM traj downstream is exact).
    cond = run_graph("vae_feature_extract",
                     {"mel": mel_lp_ref[None, None, ...], "noise": noise_cond})
    check("cond_latent", cond, load("cond_latent"), 1e-2)

    print("[stage: DDIM trajectory]")
    from app.services.engines.audiosr.ddim import DdimSchedule, combine_cfg, ddim_step

    cond_ref = load("cond_latent").astype(np.float32)
    schedule = DdimSchedule.build(assets.alphas_cumprod, meta["ddim_steps"])
    scale = np.float32(assets.scale_factor)
    cond_in = cond_ref * scale
    uncond_in = np.full_like(cond_ref, assets.unconditional_value) * scale
    x = np.load(BASE / "noise_002.npy").astype(np.float32)
    traj_files = sorted(BASE.glob("traj_*.npy"))
    steps = list(enumerate(reversed(schedule.timesteps)))
    for i, t in steps:
        index = len(schedule.timesteps) - i - 1
        ts = np.array([t], dtype=np.int64)
        v_c = run_graph("ddpm", {"x": np.concatenate([x, cond_in], 1), "timesteps": ts})
        v_u = run_graph("ddpm", {"x": np.concatenate([x, uncond_in], 1), "timesteps": ts})
        v = combine_cfg(v_c, v_u, 3.5).astype(np.float32)
        noise = np.load(BASE / f"noise_{i + 3:03d}.npy").astype(np.float32)
        x = ddim_step(x, v, int(t), index, schedule, noise).astype(np.float32)
        if i in (0, len(steps) // 2, len(steps) - 1):
            check(f"traj step {i} (t={t})", x, np.load(traj_files[i]), 2e-2)

    check("final_latent", x, load("final_latent"), 2e-2)

    print("[stage: decode + vocoder]")
    z_ref = load("final_latent").astype(np.float32)
    mel_dec = run_graph("vae_decoder", {"z": z_ref})
    check("decoded_mel", mel_dec, load("decoded_mel"), 2e-3)

    mel_ref = load("decoded_mel").astype(np.float32).copy()
    cutoff_melbin = dsp.locate_cutoff_bin(np.exp(mel_lp_ref.astype(np.float64)), 0.985)
    mel_ref[0, 0, :, :cutoff_melbin] = mel_lp_ref[:, :cutoff_melbin]
    check("mel_after_replace", mel_ref, load("mel_after_replace"), 1e-6)

    wav_voc = run_graph("vocoder", {"mel": load("mel_after_replace").astype(np.float32)[0].transpose(0, 2, 1)})
    check("vocoder_out", np.asarray(wav_voc)[..., :load("vocoder_out").shape[-1]],
          load("vocoder_out"), 2e-3)

    print("[stage: postproc]")
    voc_ref = load("vocoder_out").squeeze().astype(np.float64)
    lp_ref = load("postproc_lowpass_wav").squeeze().astype(np.float64)
    post = dsp.replace_low_band_stft(voc_ref, lp_ref)
    check("postproc_out", post, load("postproc_out").squeeze(), 5e-3)

    print("[stage: full driver end-to-end (replayed noise)]")
    driver = AudioSrDriver(assets, run_graph, noise_source=ReplayNoise())
    out = driver.restore(wav, ddim_steps=meta["ddim_steps"],
                         lowpass_type=meta["random_choices"][0])
    final_ref = load("final_waveform").squeeze()
    check("final_waveform", out, final_ref, 5e-2, min_length=True)

    if FAILURES:
        print(f"\nPARITY FAILED: {FAILURES}")
        sys.exit(1)
    print("\nPARITY OK: all stages within tolerance")


if __name__ == "__main__":
    main()
