#!/usr/bin/env python3
"""Dump S3 parity fixtures from the `mlx_video` Wan reference: a full Wan DiT forward pass for the
5B, plus a per-block capture for bisection.

Run with the SceneWorks venv:
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_s3_fixtures.py

Side effects (idempotent): converts the 5B DiT (native Wan safetensors → MLX layout, bf16) into the
snapshot dir's `model.safetensors` — the same file the Rust heavy parity test loads.

Writes mlx-gen-wan/tests/fixtures/s3_dit_golden.safetensors (committed; small):
  latent, context (raw [L_text, text_dim]), t, output, plus per-stage hiddens for one config.
"""
import glob
import os

import mlx.core as mx

from mlx_video.convert_wan import load_safetensors_weights, sanitize_wan_transformer_weights
from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.loading import load_wan_model

HOME = os.path.expanduser("~")
HF = os.path.join(HOME, ".cache/huggingface/hub")
OUT_DIR = os.environ.get(
    "WAN_5B_DIR",
    os.path.join(HOME, "Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b"),
)

# Small grid for a fast but representative forward: t_lat=1, h_lat=16, w_lat=16 → patch (1,2,2) →
# grid (1,8,8) → L=64 tokens through all 30 layers. (The NAX fast-SDPA layout sensitivity is
# layout-driven, not size-driven, so 64 tokens through the real DiT layout is a sufficient gate.)
C, T_LAT, H_LAT, W_LAT = 48, 1, 16, 16
L_TEXT = 12
T_VAL = 500.0  # an integer-valued timestep


def ensure_model(config):
    out = os.path.join(OUT_DIR, "model.safetensors")
    if not os.path.exists(out):
        snap = sorted(glob.glob(os.path.join(HF, "models--Wan-AI--Wan2.2-TI2V-5B/snapshots/*")))[-1]
        print(f"Converting 5B DiT from {snap} ...")
        weights = sanitize_wan_transformer_weights(load_safetensors_weights(snap))
        weights = {k: v.astype(mx.bfloat16) for k, v in weights.items()}
        mx.save_safetensors(out, weights)
        print(f"  wrote {len(weights)} tensors → {out}")
    return out


def main():
    config = WanModelConfig.wan22_ti2v_5b()
    model_path = ensure_model(config)
    # Natural bf16 reference: `load_wan_model` keeps the bf16 weights (no upcast), so the DiT runs
    # bf16 matmuls + bf16 SDPA + bf16 cos/sin with an f32 residual stream — exactly the production
    # regime the Rust port now mirrors. Gated bf16-against-bf16 for true production parity.
    model = load_wan_model(model_path, config)

    mx.random.seed(0)
    latent = mx.random.normal((C, T_LAT, H_LAT, W_LAT)).astype(mx.float32)
    context_raw = mx.random.normal((L_TEXT, config.text_dim)).astype(mx.float32)

    grid = (T_LAT // config.patch_size[0], H_LAT // config.patch_size[1], W_LAT // config.patch_size[2])
    seq_len = grid[0] * grid[1] * grid[2]

    context_emb = model.embed_text([context_raw])  # [1, text_len, dim] bf16
    cross_kv = model.prepare_cross_kv(context_emb)
    rope_cs = model.prepare_rope([grid])

    out = model(
        [latent],
        t=mx.array([T_VAL]),
        context=context_emb,
        seq_len=seq_len,
        cross_kv_caches=cross_kv,
        rope_cos_sin=rope_cs,
    )[0]  # [out_dim, T_LAT, H_LAT, W_LAT] f32
    mx.eval(out)

    # Per-stage capture (replicating WanModel.__call__) for the bisection gate.
    p, gs = model._patchify(latent)  # [1, L, dim]
    sin = mx.array([T_VAL])[..., None].astype(mx.float32) * model._inv_freq
    sin_emb = mx.concatenate([mx.cos(sin), mx.sin(sin)], axis=-1)
    e = model.time_embedding_1(model.time_embedding_act(model.time_embedding_0(sin_emb)))
    e0 = model.time_projection(model.time_projection_act(e)).reshape(1, 1, 6, model.dim)
    x = p
    x_b0 = None
    for i, block in enumerate(model.blocks):
        x = block(x, e=e0, seq_lens=[seq_len], grid_sizes=[gs], freqs=model.freqs,
                  context=context_emb, cross_kv_cache=cross_kv[i], rope_cos_sin=rope_cs, attn_mask=None)
        if i == 0:
            x_b0 = x
    x_head = model.head(x, e)  # [1, L, out_dim*prod(patch)]
    mx.eval(p, x_b0, x, x_head)

    golden = {
        "latent": latent,
        "context_raw": context_raw,
        "t": mx.array([T_VAL]),
        "output": out.astype(mx.float32),
        "x_embed": p.astype(mx.float32),
        "x_block0": x_b0.astype(mx.float32),
        "x_blocks": x.astype(mx.float32),
        "x_head": x_head.astype(mx.float32),
    }
    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    mx.save_safetensors(os.path.join(dst, "s3_dit_golden.safetensors"), golden)
    print(f"wrote {os.path.abspath(os.path.join(dst, 's3_dit_golden.safetensors'))}")
    print(f"  grid={grid} seq_len={seq_len} output_shape={list(out.shape)}")


if __name__ == "__main__":
    main()
