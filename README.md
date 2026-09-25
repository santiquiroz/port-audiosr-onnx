# port-audiosr-onnx

**The first known ONNX port of [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) — diffusion-based audio super-resolution that runs on *any* DirectX 12 GPU, not just NVIDIA.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![ONNX](https://img.shields.io/badge/ONNX-opset%2017%20%7C%2018-blue)](https://onnx.ai/)
[![DirectML](https://img.shields.io/badge/DirectML-AMD%20%7C%20Intel%20%7C%20NVIDIA-green)](https://learn.microsoft.com/windows/ai/directml/dml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

## Why this exists

[AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) (Liu et al., *"AudioSR: Versatile Audio Super-resolution at Scale"*) is one of the best open models for general audio super-resolution: it upsamples **any** audio — music, speech, effects — from any bandwidth to true 48 kHz, reconstructing the high frequencies that compression and cheap microphones destroy. It is MIT-licensed.

But in practice you could only run it with **PyTorch + CUDA**. If you own an AMD or Intel GPU — or you want to ship audio restoration inside a Windows app without dragging a 2.5 GB torch runtime along — you were out of luck. CPU inference works, but a diffusion model with a 258M-parameter UNet at 50 DDIM steps is painfully slow on CPU.

**This project decomposes AudioSR into plain ONNX graphs** so the heavy compute runs through [onnxruntime](https://onnxruntime.ai/) on **any execution provider**: DirectML (any DX12 GPU — AMD Radeon, Intel Arc, NVIDIA), CUDA, OpenVINO, or plain CPU. No torch at inference time. No CUDA lock-in. Quality audio super-resolution, democratized.

Measured on an **AMD Radeon RX 7800 XT** (a GPU the original model could never use), per-graph on 5.12 s inputs:

| Graph | Size | CPU-EP | DirectML | Speedup | max rel-err vs PyTorch |
|---|---|---|---|---|---|
| UNet 258M (`ddpm`, one CFG pass) | 1.0 GB | 187 ms | **44 ms** | 4.3× | 1e-6 |
| VAE decoder (scale_factor baked in) | 534 MB | 1891 ms | **79 ms** | **24×** | 1e-6 |
| Vocoder (HiFi-GAN, 48 kHz) | 761 MB | 4090 ms | **617 ms** | 6.6× | 3e-5 |
| vae_feature_extract (conditioner) | 360 MB | 915 ms | **43 ms** | **21×** | 1e-5 |

**Full pipeline** (torch-free numpy driver + ONNX graphs, 50 DDIM steps with CFG): a 6 s clip restores in **11.2 s on DirectML (RTF 1.87)** vs 63.3 s on CPU (RTF 10.5) — **5.7× end-to-end**. Final waveform matches the original PyTorch pipeline at max rel-err **8e-4** with replayed noise.

## How it works

AudioSR is a latent diffusion model. Not everything belongs in ONNX: the sampler is a Python loop, and the mel/STFT front-end is cheap DSP. The port cuts the model at the natural graph boundaries — the same cut Intel's OpenVINO port validated — into **4 neural graphs + a lightweight numpy driver**:

```mermaid
flowchart LR
    subgraph numpy driver — no torch
        A[input wav] --> B[STFT / log-mel<br/>n_fft 2048, hop 480]
        B --> C[lowpass sim +<br/>mel_replace_ops]
        S[DDIM loop · 50 steps<br/>cosine schedule · v-prediction<br/>CFG scale 3.5]
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
- **Long audio:** 10.24 s windows (the model's native training window, latent T=128) with a 1.28 s Hann crossfade between them; the input is padded to a multiple of 5.12 s.
- **Opset:** 17 for the three graphs exported with the legacy JIT exporter; `ddpm` is 18 (the 1 GB UNet needs the dynamo exporter).

## Status

| Component | Export | Parity | DirectML |
|---|---|---|---|
| VAE decoder | ✅ | ✅ | ✅ 24× |
| Vocoder (HiFi-GAN) | ✅ | ✅ | ✅ 6.6× |
| vae_feature_extract | ✅ | ✅ | ✅ 21× |
| UNet 258M (`ddpm`) | ✅ | ✅ | ✅ 4.3× |
| numpy DDIM/CFG driver | ✅ | ✅ every stage (`toolkit/validate_driver.py`) | ✅ RTF 1.87 |
| fp16 pack (all 4 graphs) | ✅ `toolkit/export_fp16.py` | ✅ 59.4 dB SI-SDR vs fp32 | ✅ −9.2 % time |

*fp16 note: converted with onnxruntime's own `float16` converter (`keep_io_types`, with `Pow`/`ReduceMean`/`Sqrt`/`Div` kept in fp32 so RMSNorm does not overflow) instead of `onnxconverter-common`, which chokes on the 1 GB dynamo-exported UNet. The pack shrinks from 2.51 to 1.26 GiB, the full pipeline runs 9.2 % faster on DirectML (RX 7800 XT, 5.1 s clip at 50 steps, median of 6 paired A/B runs) and the output stays at 59.4 dB SI-SDR against fp32. fp16 is for GPUs only: the CPU EP has few fp16 kernels, so on CPU use the fp32 pack.*

Pre-exported graphs are published as release assets — no torch needed to consume them:

- [`models-v1.0`](https://github.com/santiquiroz/port-audiosr-onnx/releases/tag/models-v1.0): fp32, ~2.6 GB, runs on every EP including CPU.
- [`models-fp16-v1.0`](https://github.com/santiquiroz/port-audiosr-onnx/releases/tag/models-fp16-v1.0): fp16 weights with fp32 inputs/outputs, 1.26 GiB, GPU only. Its `manifest.json` declares `precision`, lists every `.onnx.data` in `required_files` and carries the SHA-256 of each asset and of the fp32 graphs it came from.

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
.venv\Scripts\python.exe toolkit\validate_ort.py --require-dml
```

Runs every graph on CPU-EP and DirectML, compares against the PyTorch reference outputs, prints the timing table and exits 1 on any parity regression. Pass graph names (`ddpm vocoder …`) to check only those. Without `--require-dml`, a missing DirectML EP is reported as skipped instead of failing.

### 4. Validate the driver and benchmark the pipeline

The numpy driver lives in Upflow, so these two scripts import it from a local checkout pointed to by `UPFLOW_ROOT` (default: `~/.openclaw/workspace/image-upscaler-amd`):

```powershell
$env:UPFLOW_ROOT = "C:\path\to\upflow"
.venv\Scripts\python.exe toolkit\validate_driver.py              # CPU-EP parity of every stage vs refs/baseline
.venv\Scripts\python.exe toolkit\bench_dml.py [--fp16] [steps]   # 10.24 s clip, DirectML vs CPU, 50 steps by default
```

`validate_driver.py` replays the noise tensors captured by `toolkit/capture_baseline.py` and compares the driver against the PyTorch pipeline at every stage boundary. `bench_dml.py --fp16` also converts `ddpm` with the fp16 recipe and times it on DirectML.

The exported graphs are gitignored, so a fresh clone or worktree has none. Point `AUDIOSR_ARTIFACTS` (default: `artifacts/`) at an existing pack instead of re-exporting; `refs/baseline` is always read from the repo:

```powershell
$env:AUDIOSR_ARTIFACTS = "C:\path\to\port-audiosr-onnx\artifacts"
.venv\Scripts\python.exe toolkit\validate_driver.py
```

`validate_driver.py` and `bench_dml.py` only need the fp32 pack (4 graphs, `.npy` constants, `manifest.json`), so an installed Upflow `vendor/audiosr` works too. `validate_ort.py` also needs the `*_in*.npy` / `*_ref.npy` reference tensors that only an export writes. `bench_dml.py --fp16` writes `ddpm_fp16.onnx` into that folder, so don't aim it at a live install.

### 5. Build the fp16 pack

```powershell
.venv\Scripts\python.exe toolkit\export_fp16.py artifacts ..\audiosr-fp16
```

Converts the fp32 pack (source and destination must differ) into the `models-fp16-v1.0` layout and writes its manifest (`precision`, `required_files`, SHA-256).

### 6. Run inference without torch

The `manifest.json` + graphs are runtime-agnostic. A reference numpy driver (DDIM + CFG + mel front-end, zero torch) ships with [Upflow](https://github.com/santiquiroz/upflow), where this port powers the audio-restore engine on AMD GPUs.

## Tests

The `tests/` suite is hermetic: no torch, no GPU, CPU `onnxruntime` only, synthetic graphs and mocked sessions. Use a separate venv, since `onnxruntime` and `onnxruntime-directml` don't mix:

```powershell
python -m venv .venv-test
.venv-test\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv-test\Scripts\python.exe -m pytest tests -q
```

Tests marked `needs_artifacts` check `manifest.json` against the pack: every file in `required_files` exists, and every other file in the folder is a `<graph>_in<N>.npy` / `<graph>_ref.npy` validation tensor. They are skipped when `AUDIOSR_ARTIFACTS` (default: `artifacts/`) holds no `.onnx`, as in a fresh clone; point the variable at an exported pack to run them.

## Model config (from `manifest.json`)

| Param | Value |
|---|---|
| Sample rate | 48 000 Hz |
| STFT | n_fft/win 2048, hop 480, center=False, reflect-pad 784 |
| Mel | 256 bins, fmin 20, fmax 24 000 |
| Scheduler | cosine, linear_start 0.0015, linear_end 0.0195, 1000 train steps |
| Parameterization | **v** (`predict_eps_from_z_and_v`) |
| CFG scale | 3.5, uncond latent = `-11.4981` |
| Latent | 16 ch, 12.5 latents/s (48 000 / 480 / 8); graphs exported at 5.12 s (T=64), driver windows of 10.24 s |
| Vocoder | HiFi-GAN, upsample [6,5,4,2,2] (∏=480=hop), initial_channel 1536 |

## Credits

- **Model & weights:** [haoheliu/versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution) (MIT) — all the science is theirs. This repo is *only* the porting toolkit.
- **Graph-cut validation:** Intel's OpenVINO port of AudioSR de-risked the same decomposition.

## Contributing

PRs welcome — especially: fp16 benchmarks on other GPUs (Arc, RDNA2, Ampere), CUDA/TensorRT EP timings, opset upgrades, and driver ports to other languages (Rust/C#). Open an issue with your GPU + timing table and let's grow the matrix.

## License

MIT. The exported graphs inherit AudioSR's MIT license.
