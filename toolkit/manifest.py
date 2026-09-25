"""manifest.json of the fp32 pack, built only from the files export_components.py
writes into artifacts/, so it runs (and is tested) without the export stack.
"""

import hashlib
import re
from pathlib import Path

import onnx

MANIFEST = "manifest.json"
GRAPHS = ("vocoder", "vae_decoder", "vae_feature_extract", "ddpm")
CONSTANTS = ("alphas_cumprod.npy", "mel_basis.npy")
DEFAULT_DOMAINS = ("", "ai.onnx")
VALIDATION_FIXTURE = re.compile(rf"(?:{'|'.join(GRAPHS)})_(?:in\d+|ref)\.npy")

STATIC_CONFIG = {
    "sampling_rate": 48000,
    "stft": {"n_fft": 2048, "hop": 480, "win": 2048, "center": False,
             "pad_reflect": 784, "window": "hann"},
    "mel": {"n_mels": 256, "fmin": 20, "fmax": 24000,
            "log_clip_val": 1e-5, "basis_file": "mel_basis.npy"},
    "latent": {"channels": 16, "f_size": 32, "vae_downsample": 8,
               "frames_per_second": 12.5},
}
SAMPLING_CONFIG = {
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


def graph_file(graph: str) -> str:
    return f"{graph}.onnx"


def data_file(graph: str) -> str:
    return f"{graph}.onnx.data"


def asset_files(art_dir: Path) -> list[str]:
    graphs = [graph_file(g) for g in GRAPHS]
    # Only graphs over protobuf's 2 GB cap (ddpm, dynamo) ship external data.
    data = [data_file(g) for g in GRAPHS if (art_dir / data_file(g)).exists()]
    return sorted([*CONSTANTS, *graphs, *data])


def graph_opset(path: Path) -> int:
    model = onnx.load(str(path), load_external_data=False)
    return next(o.version for o in model.opset_import if o.domain in DEFAULT_DOMAINS)


def graph_opsets(art_dir: Path) -> dict[str, int]:
    return {g: graph_opset(art_dir / graph_file(g))
            for g in GRAPHS if (art_dir / graph_file(g)).exists()}


def sha256_of(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def hashes(folder: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256_of(folder / name) for name in names}


def existing(folder: Path, names: list[str]) -> list[str]:
    return [name for name in names if (folder / name).exists()]


def build_manifest(art_dir: Path, scale_factor: float) -> dict:
    assets = asset_files(art_dir)
    return {
        "model": "haoheliu/audiosr_basic",
        "license": "MIT",
        "precision": "fp32",
        "opsets": graph_opsets(art_dir),
        "required_files": [*assets, MANIFEST],
        **STATIC_CONFIG,
        "scale_factor": float(scale_factor),
        **SAMPLING_CONFIG,
        "sha256": hashes(art_dir, existing(art_dir, assets)),
    }


def missing_files(art_dir: Path, manifest: dict) -> list[str]:
    assets = [n for n in manifest["required_files"] if n != MANIFEST]
    return [name for name in assets if not (art_dir / name).exists()]


def is_validation_fixture(name: str) -> bool:
    return VALIDATION_FIXTURE.fullmatch(name) is not None


def unlisted_files(art_dir: Path, manifest: dict) -> list[str]:
    listed = set(manifest["required_files"])
    names = sorted(p.name for p in art_dir.iterdir() if p.is_file())
    return [n for n in names if n not in listed and not is_validation_fixture(n)]
