"""Exports the 4 neural graphs of AudioSR (basic, 48k) to ONNX opset 17,
saves PyTorch reference tensors for parity validation, and emits
manifest.json + the schedule/mel constants the runtime driver needs.

Verified module map (introspect.py):
  ld.first_stage_model            AutoencoderKL 413.8M (decoder 133.5M, vocoder 190.3M)
  ld.cond_stage_models[0]         VAEFeatureExtract 413.8M (own VAE copy - export from HERE)
  ld.model.diffusion_model        UNetModel 258.2M (in 32ch, out 16ch)
  scale_factor 0.3342, v-parameterization, latent 16ch, VAE f=8 both dims.

Outputs into artifacts/:
  vocoder.onnx, vae_decoder.onnx, vae_feature_extract.onnx, ddpm.onnx
  <name>_in*.npy / <name>_ref.npy
  alphas_cumprod.npy, mel_basis.npy, manifest.json
"""

import json
import sys
from pathlib import Path

import patches

patches.apply_all()

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

OPSET = 17
T_LAT = 64          # latent frames of a 5.12 s window (12.5 latents/s, VAE f=8)
MEL_FRAMES = T_LAT * 8
torch.manual_seed(1234)


def save_pair(name, inputs, output):
    for i, arr in enumerate(inputs):
        np.save(ART / f"{name}_in{i}.npy", arr.detach().cpu().numpy())
    np.save(ART / f"{name}_ref.npy", output.detach().cpu().numpy())


def export(module, name, inputs, input_names, output_names, dynamic_axes, dynamo=False):
    path = ART / f"{name}.onnx"
    module.eval()
    with torch.no_grad():
        out = module(*inputs)
    save_pair(name, inputs, out)
    # Legacy JIT exporter (opset 17) is the DML-validated default; the UNet
    # trips a tracer bug there ("invalid unordered_map key") so it goes
    # through the dynamo exporter (opset 18) instead.
    torch.onnx.export(
        module,
        tuple(inputs),
        str(path),
        opset_version=18 if dynamo else OPSET,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=dynamo,
    )
    size_mb = path.stat().st_size / 1e6
    print(f"[export] {name}: {size_mb:.0f} MB, out {tuple(out.shape)}", flush=True)
    return out


class VaeDecoder(torch.nn.Module):
    """decode_first_stage: z / scale_factor -> mel [B,1,T*8,256]."""

    def __init__(self, vae, scale_factor):
        super().__init__()
        self.vae = vae
        self.scale_factor = float(scale_factor)

    def forward(self, z):
        return self.vae.decode(z / self.scale_factor)


class CondEncoder(torch.nn.Module):
    """VAEFeatureExtract, deterministic: posterior.mean + posterior.std * noise.
    Matches vae.encode(mel).sample() when noise ~ N(0,1)."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, mel, noise):
        posterior = self.vae.encode(mel)
        return posterior.mean + posterior.std * noise


class UNetOnly(torch.nn.Module):
    """UNetModel.forward with AudioSR's fixed kwargs (concat conditioning only,
    context_list empty; the driver concats [z_noisy, cond*scale_factor])."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x, timesteps):
        return self.unet(x, timesteps=timesteps, y=None, context_list=[],
                         context_attn_mask_list=[])


def emit_constants(ld):
    np.save(ART / "alphas_cumprod.npy",
            ld.alphas_cumprod.detach().cpu().numpy().astype(np.float64))
    from librosa.filters import mel as librosa_mel_fn
    basis = librosa_mel_fn(sr=48000, n_fft=2048, n_mels=256, fmin=20, fmax=24000)
    np.save(ART / "mel_basis.npy", basis.astype(np.float32))

    required = sorted(
        p.name for p in ART.iterdir()
        if p.suffix in (".onnx", ".npy", ".data") and "_in" not in p.stem and "_ref" not in p.stem
    )
    manifest = {
        "model": "haoheliu/audiosr_basic",
        "license": "MIT",
        "opset": OPSET,
        "required_files": required + ["manifest.json"],
        "sampling_rate": 48000,
        "stft": {"n_fft": 2048, "hop": 480, "win": 2048, "center": False,
                 "pad_reflect": 784, "window": "hann"},
        "mel": {"n_mels": 256, "fmin": 20, "fmax": 24000,
                "log_clip_val": 1e-5, "basis_file": "mel_basis.npy"},
        "latent": {"channels": 16, "f_size": 32, "vae_downsample": 8,
                   "frames_per_second": 12.5},
        "scale_factor": float(ld.scale_factor),
        "scheduler": {"type": "ddim", "beta_schedule": "cosine",
                      "linear_start": 0.0015, "linear_end": 0.0195,
                      "num_train_timesteps": 1000, "eta": 1.0,
                      "parameterization": "v",
                      "alphas_cumprod_file": "alphas_cumprod.npy",
                      "timestep_spacing": "uniform_plus_one"},
        "cfg": {"guidance_scale": 3.5, "unconditional_value": -11.4981},
        "lowpass": {"order": 8, "cutoff_percentile": 0.985,
                    "types": ["butter", "cheby1", "ellip", "bessel"]},
        "window_seconds": 5.12,
        "graphs": {
            "vocoder": {"input": "mel [B,256,frames]", "output": "wav [B,frames*480]"},
            "vae_decoder": {"input": "z [B,16,T,32] (scale_factor baked)",
                            "output": "mel [B,1,T*8,256]"},
            "vae_feature_extract": {"input": "mel [B,1,frames,256] + noise [B,16,T,32]",
                                    "output": "cond latent [B,16,T,32] (unscaled)"},
            "ddpm": {"input": "x [B,32,T,32] = concat(z_noisy, cond*scale_factor) + timesteps [B] int64",
                     "output": "v prediction [B,16,T,32]"},
        },
    }
    (ART / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[export] manifest + constants written", flush=True)


def main():
    only = set(sys.argv[1:])

    from audiosr import build_model

    ld = build_model(model_name="basic", device="cpu")
    ld.eval()
    scale_factor = float(ld.scale_factor)
    print(f"scale_factor={scale_factor}", flush=True)

    def want(name):
        return not only or name in only

    if want("vocoder"):
        voc = ld.first_stage_model.vocoder
        voc_in = torch.randn(1, 256, MEL_FRAMES)
        export(voc, "vocoder", [voc_in], ["mel"], ["wav"],
               {"mel": {0: "batch", 2: "frames"}, "wav": {0: "batch"}})

    if want("vae_decoder"):
        z = torch.randn(1, 16, T_LAT, 32)
        dec = VaeDecoder(ld.first_stage_model, scale_factor)
        export(dec, "vae_decoder", [z], ["z"], ["mel"],
               {"z": {0: "batch", 2: "t"}, "mel": {0: "batch", 2: "frames"}})

    if want("vae_feature_extract"):
        cond_vae = ld.cond_stage_models[0].vae
        mel_in = torch.randn(1, 1, MEL_FRAMES, 256) * 2.0 - 6.0
        with torch.no_grad():
            probe = cond_vae.encode(mel_in)
        noise = torch.randn_like(probe.mean)
        print(f"cond posterior shape: {tuple(probe.mean.shape)}", flush=True)
        enc = CondEncoder(cond_vae)
        export(enc, "vae_feature_extract", [mel_in, noise], ["mel", "noise"], ["cond"],
               {"mel": {0: "batch", 2: "frames"}, "noise": {0: "batch", 2: "t"},
                "cond": {0: "batch", 2: "t"}})

    if want("ddpm"):
        unet = ld.model.diffusion_model
        x = torch.randn(1, 32, T_LAT, 32)
        t = torch.tensor([501], dtype=torch.int64)
        export(UNetOnly(unet), "ddpm", [x, t], ["x", "timesteps"], ["v_pred"],
               {"x": {0: "batch", 2: "t"}, "timesteps": {0: "batch"},
                "v_pred": {0: "batch", 2: "t"}}, dynamo=True)

    # After the graphs so required_files can list what actually exists
    # (ddpm ships as .onnx + external .onnx.data).
    emit_constants(ld)
    print("[export] done", flush=True)


if __name__ == "__main__":
    main()
