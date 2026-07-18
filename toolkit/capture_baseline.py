"""Runs the REAL audiosr generate_batch on a wav and records every random
tensor + every stage boundary, so a torch-free driver can be validated
numerically step by step against this exact run.

Usage: python toolkit/capture_baseline.py <input.wav> [ddim_steps]
Writes refs/baseline/*.npy + meta.json + baseline_out.wav
"""

import json
import sys
from pathlib import Path

import patches

patches.apply_all()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "refs" / "baseline"
OUT.mkdir(parents=True, exist_ok=True)

RECORD = {}
NOISES = []
TRAJ = []
STAGES = {}


def record_stage(name, tensor):
    arr = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    STAGES[name] = arr.shape
    np.save(OUT / f"{name}.npy", arr)


def install_recorders():
    orig_randn = torch.randn

    def rec_randn(*args, **kwargs):
        t = orig_randn(*args, **kwargs)
        NOISES.append(t.detach().cpu().numpy())
        return t

    torch.randn = rec_randn

    orig_choice = np.random.choice

    def rec_choice(a, *args, **kwargs):
        v = orig_choice(a, *args, **kwargs)
        RECORD.setdefault("random_choices", []).append(str(v))
        return v

    np.random.choice = rec_choice

    from audiosr.latent_diffusion.models.ddim import DDIMSampler

    orig_p = DDIMSampler.p_sample_ddim

    def rec_p(self, x, c, t, index, **kwargs):
        x_prev, pred_x0 = orig_p(self, x, c, t, index, **kwargs)
        TRAJ.append((int(t[0].item()), int(index), x_prev.detach().cpu().numpy()))
        return x_prev, pred_x0

    DDIMSampler.p_sample_ddim = rec_p

    from audiosr.latent_diffusion.modules.encoders.modules import VAEFeatureExtract

    orig_fe = VAEFeatureExtract.forward

    def rec_fe(self, batch):
        out = orig_fe(self, batch)
        record_stage("cond_latent", out)
        record_stage("cond_input_mel", batch)
        return out

    VAEFeatureExtract.forward = rec_fe

    from audiosr.latent_diffusion.models import ddpm as ddpm_mod

    orig_decode = ddpm_mod.LatentDiffusion.decode_first_stage

    def rec_decode(self, z):
        record_stage("final_latent", z)
        mel = orig_decode(self, z)
        record_stage("decoded_mel", mel)
        return mel

    ddpm_mod.LatentDiffusion.decode_first_stage = rec_decode

    orig_replace = ddpm_mod.LatentDiffusion.mel_replace_ops

    def rec_replace(self, samples, input):
        record_stage("lowpass_mel_for_replace", input)
        out = orig_replace(self, samples, input)
        record_stage("mel_after_replace", out)
        return out

    ddpm_mod.LatentDiffusion.mel_replace_ops = rec_replace

    orig_voc = ddpm_mod.LatentDiffusion.mel_spectrogram_to_waveform

    def rec_voc(self, mel, **kwargs):
        wav = orig_voc(self, mel, **kwargs)
        record_stage("vocoder_out", wav)
        return wav

    ddpm_mod.LatentDiffusion.mel_spectrogram_to_waveform = rec_voc

    orig_post = ddpm_mod.LatentDiffusion.postprocessing

    def rec_post(self, out_batch, x_batch):
        record_stage("postproc_in", out_batch)
        record_stage("postproc_lowpass_wav", x_batch)
        out = orig_post(self, out_batch, x_batch)
        record_stage("postproc_out", out)
        return out

    ddpm_mod.LatentDiffusion.postprocessing = rec_post


def main():
    input_wav = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "refs" / "degraded_up48k.wav")
    ddim_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    seed = 42

    install_recorders()

    from audiosr import build_model
    from audiosr.pipeline import super_resolution, make_batch_for_super_resolution, seed_everything

    ld = build_model(model_name="basic", device="cpu")

    # capture the batch separately first (same seed) so the driver can start
    # from identical inputs; random state must be reset before the real run.
    seed_everything(seed)
    batch, duration = make_batch_for_super_resolution(input_wav)
    for key in ("waveform", "stft", "log_mel_spec", "waveform_lowpass", "lowpass_mel"):
        record_stage(f"batch_{key}", batch[key])
    RECORD["duration"] = duration
    NOISES.clear()
    RECORD["random_choices"] = []

    waveform = super_resolution(ld, input_wav, seed=seed, ddim_steps=ddim_steps,
                                guidance_scale=3.5)
    record_stage("final_waveform", waveform)
    sf.write(str(OUT / "baseline_out.wav"), waveform[0, 0], 48000)

    for i, n in enumerate(NOISES):
        np.save(OUT / f"noise_{i:03d}.npy", n)
    for k, (t_val, index, x) in enumerate(TRAJ):
        np.save(OUT / f"traj_{k:03d}_t{t_val}_i{index}.npy", x)

    meta = {
        "input_wav": str(input_wav),
        "seed": seed,
        "ddim_steps": ddim_steps,
        "guidance_scale": 3.5,
        "duration": RECORD["duration"],
        "random_choices": RECORD["random_choices"],
        "noise_shapes": [list(n.shape) for n in NOISES],
        "stages": {k: list(v) for k, v in STAGES.items()},
        "traj_len": len(TRAJ),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2)[:2000])
    print("[baseline] captured", len(NOISES), "noise tensors,", len(TRAJ), "traj steps")


if __name__ == "__main__":
    main()
