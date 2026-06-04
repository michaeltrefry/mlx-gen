#!/usr/bin/env python3
"""Dump sc-2680 TI2V parity fixtures: a full **image-conditioned mask-blend denoise** run of the
`mlx_video` reference, on a tiny seeded model, for the Rust `pipeline::denoise_ti2v` (+ the DiT's
per-token-timestep `forward_tokens`) to gate against.

Like S4, this is self-contained (no real weights): a tiny dense `WanModel` with seeded random
weights, an **injected** context (T5-embedding stand-in), **injected** initial noise, and an
**injected** encoded-image latent `z_img` (the vae22 encode is gated separately by vae22_parity).
It runs the reference's exact image-conditioned loop (per-token timesteps `t_tokens = mask_tokens·t`
with the first-frame tokens frozen at 0, mask re-applied each step) and the Rust must reproduce the
final latents. Validates the per-token timestep embedding, the mask-blend init, and the per-step
re-freeze end-to-end. (The DiT runs bf16, so the gap is the known cross-build bf16 kernel delta over
the loop — bounded, like S4.)

Run with the SceneWorks venv:
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_ti2v_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/ti2v.json                  (tiny config + run knobs + io shapes)
  - mlx-gen-wan/tests/fixtures/ti2v_pipeline.safetensors  (DiT weights + injected io + mask + golden)
"""
import dataclasses
import json
import math
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.i2v_utils import build_i2v_mask
from mlx_video.models.wan.model import WanModel
from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler

# Same tiny dense config as S4 (the per-token path is z_dim/stride-agnostic — it's the timestep
# embedding + mask that matter, which are identical across Wan variants).
CFG = dataclasses.replace(
    WanModelConfig.wan21_t2v_1_3b(),
    dim=128,
    num_heads=1,      # head_dim = 128
    num_layers=2,
    ffn_dim=256,
    freq_dim=256,
    text_dim=32,
    text_len=8,
    in_dim=16,
    out_dim=16,
    vae_z_dim=16,
    dual_model=False,
)
STEPS = 4
SHIFT = 5.0
GUIDE = 3.0
FRAMES = 5            # t_lat 2 (stride 4) → first frame frozen, second denoised
HEIGHT = 16
WIDTH = 16
CTX_TOKENS = 4
RANDN = lambda *s: (mx.random.normal(s)).astype(mx.float32)  # noqa: E731


def build_model():
    mx.random.seed(0)
    model = WanModel(CFG)
    flat = tree_flatten(model.parameters())
    model.update(tree_unflatten([(k, (mx.random.normal(v.shape) * 0.1)) for k, v in flat]))
    mx.eval(model.parameters())
    return model


def main():
    model = build_model()

    vae_stride = CFG.vae_stride
    patch = CFG.patch_size
    z_dim = CFG.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat = HEIGHT // vae_stride[1]
    w_lat = WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    grid = (t_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
    target_shape = (z_dim, t_lat, h_lat, w_lat)

    i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, patch)  # [C,T,H,W], [1,L]
    mx.eval(i2v_mask, i2v_mask_tokens)

    mx.random.seed(2)
    ctx_cond = RANDN(CTX_TOKENS, CFG.text_dim)
    ctx_uncond = RANDN(CTX_TOKENS, CFG.text_dim)
    mx.random.seed(3)
    init_noise = RANDN(z_dim, t_lat, h_lat, w_lat)
    mx.random.seed(4)
    z_img = RANDN(z_dim, 1, h_lat, w_lat)  # injected encoded-image latent (vae22 encode gated elsewhere)
    mx.eval(ctx_cond, ctx_uncond, init_noise, z_img)

    # --- reference image-conditioned mask-blend loop (mirrors generate_wan.py is_i2v_mask_blend) ---
    context_emb = model.embed_text([ctx_cond, ctx_uncond])
    context_cfg = mx.concatenate([context_emb[0:1], context_emb[1:2]], axis=0)
    cross_kv = model.prepare_cross_kv(context_cfg)
    rope_cos_sin = model.prepare_rope([grid, grid])

    sched = FlowMatchEulerScheduler(num_train_timesteps=CFG.num_train_timesteps)
    sched.set_timesteps(STEPS, shift=SHIFT)

    latents = (1.0 - i2v_mask) * z_img + i2v_mask * init_noise
    for t in sched.timesteps.tolist():
        t_tokens = i2v_mask_tokens * t
        pad_len = seq_len - t_tokens.shape[1]
        if pad_len > 0:
            t_tokens = mx.concatenate([t_tokens, mx.full((1, pad_len), t)], axis=1)
        t_batch = mx.concatenate([t_tokens, t_tokens], axis=0)  # [2, L]
        preds = model(
            [latents, latents],
            t=t_batch,
            context=context_cfg,
            seq_len=seq_len,
            cross_kv_caches=cross_kv,
            rope_cos_sin=rope_cos_sin,
        )
        noise_pred = preds[1] + GUIDE * (preds[0] - preds[1])
        latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
        latents = (1.0 - i2v_mask) * z_img + i2v_mask * latents
        mx.eval(latents)
    final_latents = latents

    save = {}
    for k, v in tree_flatten(model.parameters()):
        save[k] = v.astype(mx.bfloat16)
    save["ctx_cond"] = ctx_cond
    save["ctx_uncond"] = ctx_uncond
    save["init_noise"] = init_noise
    save["z_img"] = z_img
    save["mask"] = i2v_mask.astype(mx.float32)
    save["mask_tokens"] = i2v_mask_tokens.astype(mx.float32)
    save["final_latents"] = final_latents.astype(mx.float32)

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    st = os.path.join(dst, "ti2v_pipeline.safetensors")
    mx.save_safetensors(st, save)

    meta = {
        "config": {
            f.name: list(getattr(CFG, f.name)) if isinstance(getattr(CFG, f.name), tuple) else getattr(CFG, f.name)
            for f in dataclasses.fields(CFG)
        },
        "steps": STEPS,
        "shift": SHIFT,
        "guidance": GUIDE,
        "scheduler": "euler",
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "seq_len": seq_len,
        "grid": list(grid),
        "ctx_tokens": CTX_TOKENS,
        "z_dim": z_dim,
        "t_lat": t_lat,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "final_latents_shape": list(final_latents.shape),
    }
    with open(os.path.join(dst, "ti2v.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"final_latents {tuple(final_latents.shape)}  seq_len {seq_len}  grid {grid}")
    print(f"mask {tuple(i2v_mask.shape)}  mask_tokens {i2v_mask_tokens.tolist()}")
    print(f"wrote {os.path.abspath(st)} ({os.path.getsize(st) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
