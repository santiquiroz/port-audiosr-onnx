"""Builds audiosr basic model on CPU and dumps the module layout + configs
needed to pin down the 5 export boundaries. Run once before exporting."""

import json
import sys

import patches

patches.apply_all()

import torch  # noqa: E402


def summarize(module, name):
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {type(module).__module__}.{type(module).__name__}  params={params/1e6:.1f}M")


def main():
    from audiosr import build_model

    model = build_model(model_name="basic", device="cpu")
    ld = model
    print("=== top ===")
    summarize(ld, "latent_diffusion")
    print("attrs:", [a for a in dir(ld) if not a.startswith("_")][:80])

    for attr in ("model", "first_stage_model", "cond_stage_models", "cond_stage_model"):
        obj = getattr(ld, attr, None)
        if obj is None:
            continue
        if isinstance(obj, (list, torch.nn.ModuleList)):
            for i, m in enumerate(obj):
                summarize(m, f"{attr}[{i}]")
        else:
            summarize(obj, attr)

    dw = ld.model
    summarize(dw.diffusion_model, "model.diffusion_model (UNet)")

    fsm = ld.first_stage_model
    for attr in ("decoder", "encoder", "post_quant_conv", "quant_conv", "vocoder"):
        obj = getattr(fsm, attr, None)
        if obj is not None:
            summarize(obj, f"first_stage_model.{attr}")
    voc = getattr(ld, "vocoder", None)
    if voc is not None:
        summarize(voc, "ld.vocoder")

    print("=== scalars ===")
    for attr in ("scale_factor", "parameterization", "num_timesteps", "linear_start",
                 "linear_end", "beta_schedule", "latent_t_size", "latent_f_size",
                 "channels", "cond_stage_key", "sampling_rate"):
        print(attr, "=", getattr(ld, attr, "<missing>"))

    print("=== cond stage detail ===")
    csm = getattr(ld, "cond_stage_models", None)
    if csm is not None:
        for i, m in enumerate(csm):
            print(i, type(m).__name__, {a: getattr(m, a) for a in ("uncond_value",) if hasattr(m, a)})

    print("=== ddim/schedule buffers ===")
    for attr in ("alphas_cumprod", "betas"):
        t = getattr(ld, attr, None)
        if t is not None:
            print(attr, tuple(t.shape), float(t[0]), float(t[-1]))

    cfg = getattr(ld, "config", None) or getattr(ld, "model_config", None)
    if cfg:
        print("=== config ===")
        print(json.dumps(cfg, indent=1, default=str)[:4000])


if __name__ == "__main__":
    sys.exit(main())
