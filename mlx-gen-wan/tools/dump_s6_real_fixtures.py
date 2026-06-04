#!/usr/bin/env python3
"""Dump an S6 **real-weight** parity fixture: a small dual-expert MoE T2V run of the `mlx_video`
reference on the *actual converted* Wan2.2-T2V-A14B weights, for the Rust `Wan14b::generate`
pipeline (`pipeline::denoise_moe` + the z16 VAE) to gate against end-to-end.

Unlike the tiny seeded S1–S5 fixtures, this loads the **real** converted checkpoint
(`convert_wan.py` output) and runs the genuine 40-layer / dim-5120 experts + the real UMT5-XXL text
encoder + the real Wan2.1 z16 VAE. Only the init **noise** is injected (dumped as an array) because
seeded RNG is not portable across the mlx-python / mlx-rs split — everything else (the prompt, the
T5 encode, both experts' forwards, the scheduler, the VAE decode) is the real chain. The Rust test
re-encodes the same prompt (so real-weight T5 parity is checked too), injects this same noise, and
compares the final latents + decoded frames.

Run with the SceneWorks venv, pointing at the converted model dir:

    WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
    WAN_A14B_FIXTURE=/tmp/wan_a14b_s6.safetensors \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s6_real_fixtures.py

Writes (NOT committed — tied to the 54 GB converted weights, which live outside the repo):
  - $WAN_A14B_FIXTURE                     (noise + context + golden latents + golden video)
  - ${WAN_A14B_FIXTURE%.safetensors}.json (metadata: prompt, geometry, routing, thresholds)

The Rust side (`tests/s6_real_parity.rs`, #[ignore]) reads both env vars + this fixture.
"""
import json
import math
import os
from pathlib import Path

import mlx.core as mx

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.loading import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_wan_model,
)
from mlx_video.models.wan.scheduler import FlowUniPCScheduler

# --- Small-but-real generation knobs (keep both experts firing across the boundary) ---
PROMPT = "a red fox trotting across a snowy meadow at sunrise, cinematic"
FRAMES, HEIGHT, WIDTH = 5, 128, 128
STEPS = 6
SCHEDULER = "unipc"


def main():
    model_dir = Path(os.path.expanduser(os.environ["WAN_A14B_MODEL_DIR"]))
    fixture_path = Path(os.path.expanduser(os.environ["WAN_A14B_FIXTURE"]))

    # Config from the converted dir (wan22_t2v_14b: dual, boundary 0.875, shift 12, guide [3,4]).
    with open(model_dir / "config.json") as f:
        cfg_json = json.load(f)
    fields = WanModelConfig.__dataclass_fields__
    cdict = {k: v for k, v in cfg_json.items() if k in fields}
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in cdict and isinstance(cdict[key], list):
            cdict[key] = tuple(cdict[key])
    config = WanModelConfig(**cdict)
    assert config.dual_model, "expected the dual-expert A14B config"

    shift = config.sample_shift
    guide_low, guide_high = config.sample_guide_scale
    neg_prompt = config.sample_neg_prompt
    boundary = config.boundary * config.num_train_timesteps

    vae_stride, patch, z_dim = config.vae_stride, config.patch_size, config.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat, w_lat = HEIGHT // vae_stride[1], WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    f_grid, h_grid, w_grid = t_lat // patch[0], h_lat // patch[1], w_lat // patch[2]

    # --- Real UMT5 encode of the real prompt + negative prompt ---
    print("Loading T5 + tokenizer, encoding prompt...")
    from transformers import AutoTokenizer

    t5 = load_t5_encoder(model_dir / "t5_encoder.safetensors", config)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    context = encode_text(t5, tokenizer, PROMPT, config.text_len)
    context_null = encode_text(t5, tokenizer, neg_prompt, config.text_len)
    mx.eval(context, context_null)
    del t5

    # --- Injected init noise (dumped; Rust uses these exact values) ---
    mx.random.seed(1234)
    noise = mx.random.normal((z_dim, t_lat, h_lat, w_lat)).astype(mx.float32)
    mx.eval(noise)

    # --- Load both real experts; embed contexts per expert; precompute cross-KV + RoPE ---
    print("Loading low/high experts (27 GB each)...")
    low_model = load_wan_model(model_dir / "low_noise_model.safetensors", config)
    high_model = load_wan_model(model_dir / "high_noise_model.safetensors", config)

    def prep(model):
        emb = model.embed_text([context, context_null])  # [2, text_len, dim]
        ctx = mx.concatenate([emb[0:1], emb[1:2]], axis=0)
        kv = model.prepare_cross_kv(ctx)
        rope = model.prepare_rope([(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)])
        return ctx, kv, rope

    ctx_low, kv_low, rope_low = prep(low_model)
    ctx_high, kv_high, rope_high = prep(high_model)

    sched = FlowUniPCScheduler(num_train_timesteps=config.num_train_timesteps)
    sched.set_timesteps(STEPS, shift=shift)

    latents = noise
    routing = []
    print(f"Denoising {STEPS} steps (boundary={boundary})...")
    for t in sched.timesteps.tolist():
        if t >= boundary:
            model, ctx, kv, rope, gs = high_model, ctx_high, kv_high, rope_high, guide_high
            routing.append(["high", t])
        else:
            model, ctx, kv, rope, gs = low_model, ctx_low, kv_low, rope_low, guide_low
            routing.append(["low", t])
        preds = model(
            [latents, latents],
            t=mx.array([t, t]),
            context=ctx,
            seq_len=seq_len,
            cross_kv_caches=kv,
            rope_cos_sin=rope,
        )
        noise_pred = preds[1] + gs * (preds[0] - preds[1])
        latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
        mx.eval(latents)
    final_latents = latents
    assert any(r[0] == "high" for r in routing) and any(r[0] == "low" for r in routing), (
        f"fixture must exercise BOTH experts across boundary {boundary}; routing={routing}"
    )
    del low_model, high_model

    # --- Real z16 VAE decode → [1, 3, F, H, W] in [-1, 1] ---
    print("Decoding with the z16 VAE...")
    vae = load_vae_decoder(model_dir / "vae.safetensors", config)
    video = vae.decode(final_latents[None])
    mx.eval(video)

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(fixture_path),
        {
            "noise": noise,
            "context": context,
            "context_null": context_null,
            "final_latents": final_latents.astype(mx.float32),
            "video": video.astype(mx.float32),
        },
    )
    meta = {
        "prompt": PROMPT,
        "neg_prompt": neg_prompt,
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "steps": STEPS,
        "scheduler": SCHEDULER,
        "shift": shift,
        "guide_low": guide_low,
        "guide_high": guide_high,
        "boundary": config.boundary,
        "boundary_timestep": boundary,
        "num_train_timesteps": config.num_train_timesteps,
        "seq_len": seq_len,
        "routing": routing,
        "final_latents_shape": list(final_latents.shape),
        "video_shape": list(video.shape),
    }
    json_path = fixture_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"routing={routing}")
    print(f"final_latents {tuple(final_latents.shape)}  video {tuple(video.shape)}")
    print(f"wrote {fixture_path}\n      {json_path}")


if __name__ == "__main__":
    main()
