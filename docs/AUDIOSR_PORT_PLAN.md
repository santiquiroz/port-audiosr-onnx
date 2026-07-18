# AudioSR (haoheliu/versatile_audio_super_resolution, MIT) — ONNX/DirectML port plan

2nd port after Apollo. General-audio super-resolution, any→48kHz. Latent-diffusion (much larger than Apollo).
Baseline verified on CPU (this machine): build 12s + inference 48s for a 10.24s clip @ 25 DDIM steps; UNet = 258M
params; output well-formed. The whole decomposition below is de-risked by **Intel's OpenVINO port** (same graph cut).

## Export decomposition — 5 ONNX graphs + Python glue

Package objects under `audiosr/` (installed venv). Top object = `LatentDiffusion` (`pipeline.build_model`).

**Export to ONNX (the neural graphs):**
1. **UNet / `ddpm`** — the core + main risk (258M params). `DiffusionWrapper.forward` (ddpm.py:1636) concats
   cond latent → 32ch, calls `UNetModel.forward(x[B,32,T,32], timesteps[B]) → [B,16,T,32]`
   (openaimodel.py:837; `context_list=[]`, `y=None` for AudioSR — concat conditioning only). Config: model_channels
   128, channel_mult [1,2,3,5], attn_resolutions [8,4,2], num_res_blocks 2, spatial-transformer depth 1. Called per
   step via `apply_model` (ddpm.py:1027).
2. **VAE decoder / `audio_sr_decoder`** — `AutoencoderKL.decode(z)` (autoencoder.py:111) → `post_quant_conv` +
   `Decoder` (model.py:546). `decode_first_stage` rescales `z/scale_factor` first. latent 16ch f=32 → mel 256.
3. **Vocoder** — HiFi-GAN `Generator` (hifigan/models.py:112, `forward(mel[B,256,T]) → wav[B,1,T*480]`). 48k config:
   upsample_rates [6,5,4,2,2] (∏=480=hop), initial_channel 1536. "Trivial to export" per recon.
4. **`vae_feature_extract`** — `VAEFeatureExtract.forward` (encoders/modules.py:139): `vae.encode(lowpass_mel).sample()`
   → the 16ch cond latent concatenated into the UNet's 32ch input. Uncond latent for CFG = constant `-11.4981`.
5. (optional) **VAE encoder / `audio_sr_encoder`** — `AutoencoderKL.encode` (autoencoder.py:103) if latent-space init needed.

**Keep in Python/numpy (NOT ONNX):**
- DDIM sampler loop + cosine scheduler (`ddim.py` `DDIMSampler`, `p_sample_ddim`@265; base schedule cosine,
  linear_start 0.0015, linear_end 0.0195, 1000 steps). **v-parameterization** (`predict_eps/start_from_z_and_v`).
- CFG combine (ddim.py:293-301): 2 UNet calls, `uncond + scale*(cond-uncond)`, scale 3.5.
- STFT/mel front-end: sr 48000, n_fft/win **2048 (power-of-two)**, hop 480, n_mel 256, fmin 20, fmax 24000,
  hann, center=False, reflect-pad 784, log-mel C=1, librosa mel basis.
- Lowpass simulation (`lowpass.py`, scipy butter/cheby1/ellip/bessel order 8), `mel_replace_ops` (ddpm.py:1567),
  and librosa-STFT **postprocessing** low-band replacement (ddpm.py:1577) — must reproduce faithfully.

## Segmentation (0.0.7 wheel)
No `super_resolution_long_audio` in 0.0.7 — pad up to a multiple of **5.12s** (`utils.read_wav_file:188`);
`latent_t_size 128`, `latent_t_per_second 12.8`. Implement own 5.12s sliding-window + overlap-add for long audio
(and to stay under the DirectML large-tensor limit found in the Apollo port).

## Reuse from the Apollo port
- The **DirectML length limit** (T too large → wrong output) will very likely apply to AudioSR's UNet too → chunk at
  5.12s segments (its native window anyway).
- fp16 was a dead end for Apollo (dispatch-bound); AudioSR's UNet is compute-heavier (258M) so fp16 MIGHT help there —
  test, don't assume.
- Same multi-provider story: one set of ONNX graphs → CPU / DirectML / CUDA / OpenVINO.

## Env gotchas (Windows, from the baseline setup)
- audiosr 0.0.7 hard-pins **numpy==1.23.5**; matplotlib must be **3.7.5** (newer forces numpy≥1.25).
- `setuptools<81` (librosa 0.9.2 needs `pkg_resources`).
- torchaudio 2.11 → TorchCodec (no Windows wheel): monkeypatch `torchaudio.load` to use **soundfile**.

## Proof-of-concept: 2 of 5 graphs exported + DirectML-validated (DONE)

Vocoder (HiFi-GAN) and VAE decoder exported clean at **opset 17, no patches needed** (plain conv nets), and
**validated on this RX 7800 XT**:

| Graph | size | CPU-EP | DirectML | Correctness vs PyTorch |
|---|---|---|---|---|
| VAE decoder (`AutoencoderKL.decode`, scale_factor baked in) | 509 MB | 2071 ms | **247 ms (8× faster)** | rel-err **0.000** |
| Vocoder (HiFi-GAN Generator, 48k) | 726 MB | 4426 ms | **1261 ms (3.5× faster)** | rel-err **0.000** |

**Key finding — AudioSR is compute-bound, so it GPU-accelerates properly** (3.5–8× on DirectML), unlike Apollo
which was dispatch-bound (~1.4×). The heavy UNet (258M params, the diffusion core) is the same class of big-conv
compute → expected to accelerate well too. So despite being more work, **AudioSR is the better GPU-performance
bet** (and MIT-licensed, general-audio). Shapes chain directly: VAE decoder `[B,1,512,256]` → squeeze/permute →
`[B,256,512]` → vocoder. Files: `audiosr-port/{vocoder,vae_decoder}.onnx` + `*_in/_ref.npy` + `export_components.py`.

Remaining graphs: UNet/`ddpm` (main risk + main compute), `vae_feature_extract` (conditioner). Then the Python
DDIM/v-pred/CFG driver + lowpass/mel/postproc glue.

## Effort
~1-2 weeks (recon estimate) — 5 graph exports + a from-scratch DDIM/v-pred/CFG Python driver + the non-neural
front/back glue + per-component parity + DirectML validation. Baseline + full boundary map are DONE (above); the
implementation is the remaining work. Recommend: export vocoder + VAE decoder first (low risk), then the UNet
(main risk + perf bottleneck), then wire the Python sampler, validate end-to-end vs the CPU baseline.
