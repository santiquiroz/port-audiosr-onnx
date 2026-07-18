# port-audiosr-onnx

**The first known ONNX port of [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) — diffusion-based audio super-resolution that runs on *any* DirectX 12 GPU, not just NVIDIA.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017-blue)](https://onnx.ai/)
[![DirectML](https://img.shields.io/badge/DirectML-AMD%20%7C%20Intel%20%7C%20NVIDIA-green)](https://learn.microsoft.com/windows/ai/directml/dml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

## Why this exists

[AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) (Liu et al., *"AudioSR: Versatile Audio Super-resolution at Scale"*) is one of the best open models for general audio super-resolution: it upsamples **any** audio — music, speech, effects — from any bandwidth to true 48 kHz, reconstructing the high frequencies that compression and cheap microphones destroy. It is MIT-licensed.

But in practice you could only run it with **PyTorch + CUDA**. If you own an AMD or Intel GPU — or you want to ship audio restoration inside a Windows app without dragging a 2.5 GB torch runtime along — you were out of luck. CPU inference works, but a diffusion model with a 258M-parameter UNet at 25 DDIM steps is painfully slow on CPU.

**This project decomposes AudioSR into plain ONNX graphs** so the heavy compute runs through [onnxruntime](https://onnxruntime.ai/) on **any execution provider**: DirectML (any DX12 GPU — AMD Radeon, Intel Arc, NVIDIA), CUDA, OpenVINO, or plain CPU. No torch at inference time. No CUDA lock-in. Quality audio super-resolution, democratized.

Measured on an **AMD Radeon RX 7800 XT** (a GPU the original model could never use):

| Graph | Size | CPU-EP | DirectML | Speedup | Correctness vs PyTorch |
|---|---|---|---|---|---|
| VAE decoder (`AutoencoderKL.decode`, scale_factor baked in) | 509 MB | 2071 ms | **247 ms** | **8.4×** | rel-err **0.000** |
| Vocoder (HiFi-GAN Generator, 48 kHz) | 726 MB | 4426 ms | **1261 ms** | **3.5×** | rel-err **0.000** |

*(Table grows as the remaining graphs land — see status below.)*

## How it works

AudioSR is a latent diffusion model. Not everything belongs in ONNX: the sampler is a Python loop, and the mel/STFT front-end is cheap DSP. The port cuts the model at the natural graph boundaries — the same cut Intel's OpenVINO port validated — into **4 neural graphs + a lightweight numpy driver**:

```mermaid
flowchart LR
    subgraph numpy driver — no torch
        A[input wav] --> B[STFT / log-mel<br/>n_fft 2048, hop 480]
        B --> C[lowpass sim +<br/>mel_replace_ops]
        S[DDIM loop · 25 steps<br/>cosine schedule · v-prediction<br/>CFG scale 3.5]
        P[low-band replacement<br/>postproc] --> Q[output wav 48 kHz]
    end
    subgraph ONNX graphs — any EP
        D["vae_feature_extract.onnx<br/>(conditioner, 16ch latent)"]
        U["ddpm.onnx<br/>(UNet 258M — the core)"]
        V["vae_decoder.onnx<br/>(latent → mel)"]
        W["vocoder.onnx<br/>(HiFi-GAN, mel → wav)"]
    end
    C --> D --> S
    S <--> U
    S --> V --> W --> P
```

- **In ONNX:** UNet (`ddpm`), VAE decoder, HiFi-GAN vocoder, VAE feature extractor (conditioner).
- **In numpy:** DDIM sampler (cosine schedule, **v-parameterization**, CFG with 2 UNet calls/step), mel/STFT front-end, scipy lowpass simulation, low-band replacement postprocessing. Unconditional latent for CFG is the constant `-11.4981`.
- **Long audio:** 5.12 s sliding window + overlap-add (the model's native window — and it keeps tensors under DirectML's large-tensor limit).

## Status

| Component | Export | Parity | DirectML |
|---|---|---|---|
| VAE decoder | ✅ | ✅ rel-err 0.000 | ✅ 8.4× |
| Vocoder (HiFi-GAN) | ✅ | ✅ rel-err 0.000 | ✅ 3.5× |
| vae_feature_extract | 🔜 | — | — |
| UNet 258M (`ddpm`) | 🔜 | — | — |
| numpy DDIM/CFG driver | 🔜 | — | — |

## Usage

### 1. Set up the export environment

The upstream `audiosr` package pins ancient deps (`numpy==1.23.5`), so the toolkit lives in its own venv:

```powershell
git clone https://github.com/santiquiroz/port-audiosr-onnx
cd port-audiosr-onnx
pwsh -File toolkit/setup-env.ps1
```

Windows gotchas handled for you: `setuptools<81` (librosa 0.9.2 needs `pkg_resources`), `matplotlib==3.7.5` (newer forces numpy≥1.25), and a `torchaudio.load` → soundfile monkeypatch (torchaudio 2.x has no TorchCodec wheel on Windows).

### 2. Export the graphs

```powershell
.venv\Scripts\python.exe toolkit\export_components.py   # weights auto-download from HF (~6 GB)
```

Produces `artifacts/{vae_decoder,vocoder,vae_feature_extract,ddpm}.onnx` + parity reference tensors + `manifest.json` (sample rate, STFT params, scheduler config, CFG scale — everything a runtime needs).

### 3. Validate on your GPU

```powershell
.venv\Scripts\python.exe toolkit\validate_dml.py
```

Runs every graph on CPU-EP and DirectML, compares against the PyTorch reference outputs, prints the timing table.

### 4. Run inference without torch

The `manifest.json` + graphs are runtime-agnostic. A reference numpy driver (DDIM + CFG + mel front-end, zero torch) ships with [Upflow](https://github.com/santiquiroz/upflow), where this port powers the audio-restore engine on AMD GPUs.

## Model config (from `manifest.json`)

| Param | Value |
|---|---|
| Sample rate | 48 000 Hz |
| STFT | n_fft/win 2048, hop 480, center=False, reflect-pad 784 |
| Mel | 256 bins, fmin 20, fmax 24 000 |
| Scheduler | cosine, linear_start 0.0015, linear_end 0.0195, 1000 train steps |
| Parameterization | **v** (`predict_eps_from_z_and_v`) |
| CFG scale | 3.5, uncond latent = `-11.4981` |
| Latent | 16 ch, 12.8 latents/s, window 5.12 s |
| Vocoder | HiFi-GAN, upsample [6,5,4,2,2] (∏=480=hop), initial_channel 1536 |

## Credits

- **Model & weights:** [haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution) (MIT) — all the science is theirs. This repo is *only* the porting toolkit.
- **Graph-cut validation:** Intel's OpenVINO port of AudioSR de-risked the same decomposition.
- Sibling port: [Apollo → ONNX](https://github.com/santiquiroz/upflow) (band-restoration, same motivation).

## Contributing

PRs welcome — especially: fp16 benchmarks on other GPUs (Arc, RDNA2, Ampere), CUDA/TensorRT EP timings, opset upgrades, and driver ports to other languages (Rust/C#). Open an issue with your GPU + timing table and let's grow the matrix.

## License

MIT. The exported graphs inherit AudioSR's MIT license.
