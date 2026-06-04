#!/usr/bin/env python
"""sc-2853 reference per-step micro-benchmark — isolate the small-seq Rust-vs-Python deficit.

Mirrors `mlx-gen-wan/tests/perf.rs` on the SAME real expert weights + geometry (480p×25f, seq
~10920), timing one batched **B=2** forward per step (cond+uncond, the reference's CFG path) with the
RoPE + cross-KV caches precomputed once — measured **compiled** (`mx.compile`, the reference default)
vs **eager** (`--no-compile`, what the eager mlx-rs port is limited to).

If eager-Python ≈ Rust (both ~28 s/step) and compiled-Python ≈ 22 s/step, the small-seq deficit is the
`mx.compile` kernel fusion the reference enjoys — NOT the batching/step-caching sc-2853 hypothesized.

    PY=~/Library/Application\\ Support/SceneWorks/python/venv/bin/python
    "$PY" mlx-gen-wan/tools/bench_wan_a14b.py \\
        --model-dir ~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx


def load_config(model_dir: Path):
    from mlx_video.models.wan.config import WanModelConfig

    with open(model_dir / "config.json") as f:
        d = json.load(f)
    d.pop("quantization", None)
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in d and isinstance(d[key], list):
            d[key] = tuple(d[key])
    return WanModelConfig(**{k: v for k, v in d.items() if k in WanModelConfig.__dataclass_fields__})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    cfg = load_config(model_dir)

    from mlx_video.models.wan.loading import load_wan_model

    model = load_wan_model(model_dir / "high_noise_model.safetensors", cfg)

    # Geometry → latent + grid + seq_len (mirror pipeline.rs::latent_shape/seq_len).
    vs = cfg.vae_stride
    t_lat = (args.frames - 1) // vs[0] + 1
    h_lat = args.height // vs[1]
    w_lat = args.width // vs[2]
    pt, ph, pw = cfg.patch_size
    grid = (t_lat // pt, h_lat // ph, w_lat // pw)
    sl = grid[0] * grid[1] * grid[2]
    print(f"geometry: {args.frames}f {args.height}x{args.width} -> latent "
          f"[{cfg.vae_z_dim},{t_lat},{h_lat},{w_lat}], grid {grid}, seq_len={sl}")

    latents = mx.random.normal((cfg.vae_z_dim, t_lat, h_lat, w_lat))
    raw_ctx = mx.random.normal((cfg.text_len, cfg.text_dim))
    context_cfg = model.embed_text([raw_ctx, raw_ctx])  # [2, text_len, dim]
    mx.eval(latents, context_cfg)

    cross_kv = model.prepare_cross_kv(context_cfg)
    rcs = model.prepare_rope([grid, grid])
    mx.eval(cross_kv, rcs)
    t_batch = mx.array([833.0, 833.0])

    def run(call):
        preds = call(
            [latents, latents],
            t=t_batch,
            context=context_cfg,
            seq_len=sl,
            cross_kv_caches=cross_kv,
            rope_cos_sin=rcs,
        )
        mx.eval(preds)
        return preds

    for label, call in (("compiled", mx.compile(model)), ("eager", model)):
        times = []
        for i in range(args.warmup + args.iters):
            t0 = time.time()
            run(call)
            dt = time.time() - t0
            if i >= args.warmup:
                times.append(dt)
        times.sort()
        print(f"[warm s/step] {label:8s} = {times[len(times)//2]:.4f}  "
              f"(min {min(times):.4f} max {max(times):.4f})")


if __name__ == "__main__":
    main()
