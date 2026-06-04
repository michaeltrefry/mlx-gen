#!/usr/bin/env python3
"""Dump S5 parity fixtures: a full **dual-expert MoE** denoise + decode run of the `mlx_video`
reference, on two tiny seeded models, for the Rust `pipeline::denoise_moe` to gate against.

Same self-contained tiny-fixture approach as S4, but two dense `WanModel`s (high/low noise, different
seeds so the boundary swap is observable) driven through the reference's `is_dual` loop: per-step
expert select by `t ≥ boundary·num_train`, per-expert text embeds / cross-KV / RoPE / guidance.

Run with the SceneWorks venv:
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s5_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/s5.json
  - mlx-gen-wan/tests/fixtures/s5_low.safetensors    (low-noise DiT + VAE + injected io + golden)
  - mlx-gen-wan/tests/fixtures/s5_high.safetensors   (high-noise DiT)
"""
import dataclasses
import json
import math
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.model import WanModel
from mlx_video.models.wan.scheduler import FlowMatchEulerScheduler
from mlx_video.models.wan.vae import CausalConv3d, Decoder3d, WanVAE

CFG = dataclasses.replace(
    WanModelConfig.wan22_t2v_14b(),  # dual-expert base (boundary 0.875)
    dim=128,
    num_heads=1,
    num_layers=2,
    ffn_dim=256,
    freq_dim=256,
    text_dim=32,
    text_len=8,
    in_dim=16,
    out_dim=16,
    vae_z_dim=16,
)
VAE_DIM = 4
STEPS = 4
SHIFT = 5.0
GUIDE_LOW = 3.0
GUIDE_HIGH = 4.0
FRAMES, HEIGHT, WIDTH = 5, 16, 16
CTX_TOKENS = 4
RANDN = lambda *s: (mx.random.normal(s)).astype(mx.float32)  # noqa: E731


def seeded_dit(seed: int) -> WanModel:
    mx.random.seed(seed)
    m = WanModel(CFG)
    flat = tree_flatten(m.parameters())
    m.update(tree_unflatten([(k, (mx.random.normal(v.shape) * 0.1)) for k, v in flat]))
    mx.eval(m.parameters())
    return m


def seeded_vae() -> WanVAE:
    mx.random.seed(7)
    vae = WanVAE(z_dim=16, encoder=False)
    vae.decoder = Decoder3d(dim=VAE_DIM, z_dim=16)
    vae.conv2 = CausalConv3d(16, 16, 1)
    keep = {"mean", "std", "inv_std"}
    vflat = tree_flatten(vae.parameters())
    vae.update(
        tree_unflatten(
            [
                (k, v if k.rsplit(".", 1)[-1] in keep else (mx.random.normal(v.shape) * 0.5).astype(mx.float32))
                for k, v in vflat
            ]
        )
    )
    mx.eval(vae.parameters())
    return vae


def main():
    low_model = seeded_dit(10)
    high_model = seeded_dit(20)
    vae = seeded_vae()

    vae_stride, patch, z_dim = CFG.vae_stride, CFG.patch_size, CFG.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat, w_lat = HEIGHT // vae_stride[1], WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    grid = (t_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
    boundary = CFG.boundary * CFG.num_train_timesteps

    mx.random.seed(2)
    ctx_cond, ctx_uncond = RANDN(CTX_TOKENS, CFG.text_dim), RANDN(CTX_TOKENS, CFG.text_dim)
    mx.random.seed(3)
    init_noise = RANDN(z_dim, t_lat, h_lat, w_lat)
    mx.eval(ctx_cond, ctx_uncond, init_noise)

    # per-expert embeds / cross-kv / rope (each model has its own text_embedding)
    def prep(model):
        emb = model.embed_text([ctx_cond, ctx_uncond])  # [2, text_len, dim]
        ccfg = mx.concatenate([emb[0:1], emb[1:2]], axis=0)
        return ccfg, model.prepare_cross_kv(ccfg), model.prepare_rope([grid, grid])

    ctx_low, kv_low, rope_low = prep(low_model)
    ctx_high, kv_high, rope_high = prep(high_model)

    sched = FlowMatchEulerScheduler(num_train_timesteps=CFG.num_train_timesteps)
    sched.set_timesteps(STEPS, shift=SHIFT)

    latents = init_noise
    routing = []
    for t in sched.timesteps.tolist():
        if t >= boundary:
            model, ctx, kv, rope, gs = high_model, ctx_high, kv_high, rope_high, GUIDE_HIGH
            routing.append(("high", t))
        else:
            model, ctx, kv, rope, gs = low_model, ctx_low, kv_low, rope_low, GUIDE_LOW
            routing.append(("low", t))
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
    video = vae.decode(final_latents[None])
    mx.eval(video)

    print(f"boundary={boundary}  routing={routing}")
    assert any(r[0] == "high" for r in routing) and any(r[0] == "low" for r in routing), (
        "fixture must exercise BOTH experts across the boundary"
    )

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)

    low_save = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(low_model.parameters())}
    for k, v in tree_flatten(vae.parameters()):
        low_save[k] = v.astype(mx.float32)
    low_save["ctx_cond"] = ctx_cond
    low_save["ctx_uncond"] = ctx_uncond
    low_save["init_noise"] = init_noise
    low_save["final_latents"] = final_latents.astype(mx.float32)
    low_save["video"] = video.astype(mx.float32)
    mx.save_safetensors(os.path.join(dst, "s5_low.safetensors"), low_save)

    high_save = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(high_model.parameters())}
    mx.save_safetensors(os.path.join(dst, "s5_high.safetensors"), high_save)

    meta = {
        "steps": STEPS,
        "shift": SHIFT,
        "guide_low": GUIDE_LOW,
        "guide_high": GUIDE_HIGH,
        "boundary": CFG.boundary,
        "boundary_timestep": boundary,
        "num_train_timesteps": CFG.num_train_timesteps,
        "seq_len": seq_len,
        "grid": list(grid),
        "routing": [[r[0], r[1]] for r in routing],
        "final_latents_shape": list(final_latents.shape),
        "video_shape": list(video.shape),
    }
    with open(os.path.join(dst, "s5.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"final_latents {tuple(final_latents.shape)}  video {tuple(video.shape)}")
    print(f"wrote s5_low/high.safetensors + s5.json to {os.path.abspath(dst)}")


if __name__ == "__main__":
    main()
