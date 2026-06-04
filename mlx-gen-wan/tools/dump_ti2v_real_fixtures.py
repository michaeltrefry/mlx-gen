#!/usr/bin/env python3
"""Dump a sc-2680 **real-weight** parity fixture: small **T2V** + **TI2V** runs of the `mlx_video`
reference on the *actual converted* Wan2.2-TI2V-5B weights, for the Rust dense 5B pipeline (the z48
`Wan22Vae` + `denoise`/`denoise_ti2v` + the DiT `forward`/`forward_tokens`) to gate against e2e.

Unlike the tiny seeded gates, this loads the **real** 5B snapshot (10 GB DiT + 11 GB UMT5 + 2.8 GB
z48 VAE) and runs the genuine chain. Only the init **noise** and the **preprocessed image tensor**
are injected (seeded RNG isn't portable across mlx-python/mlx-rs; the PIL preprocess is gated
separately) — everything else is the real chain. The Rust test re-encodes the same prompt (real T5
parity), re-encodes the same image through the real z48 VAE (real vae22-encode parity), injects this
noise, and compares the final latents + decoded frames for both modes.

Run with the SceneWorks venv, pointing at the 5B snapshot:

    WAN_5B_MODEL_DIR="$HOME/Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b" \
    WAN_5B_FIXTURE=/tmp/wan_5b_ti2v.safetensors \
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_ti2v_real_fixtures.py

Writes (NOT committed — tied to the heavy converted weights):
  - $WAN_5B_FIXTURE                      (noise + image + context + T2V & TI2V goldens)
  - ${WAN_5B_FIXTURE%.safetensors}.json  (metadata: prompt, geometry, thresholds)
"""
import json
import math
import os
from pathlib import Path

import mlx.core as mx

from mlx_video.models.wan.config import WanModelConfig
from mlx_video.models.wan.i2v_utils import build_i2v_mask
from mlx_video.models.wan.loading import (
    encode_text,
    load_t5_encoder,
    load_vae_decoder,
    load_vae_encoder,
    load_wan_model,
)
from mlx_video.models.wan.scheduler import FlowUniPCScheduler
from mlx_video.models.wan.vae22 import denormalize_latents

PROMPT = "a red fox trotting across a snowy meadow at sunrise, cinematic"
FRAMES, HEIGHT, WIDTH = 5, 256, 256   # t_lat 2, h/w_lat 16 (stride 4×16×16)
STEPS = 4
GUIDE = 5.0


def vae22_decode_to_video(vae, latents):
    """Mirror generate_wan.py's wan22 decode: [C,T,H,W] → channels-last, denorm, decode → [1,T',H',W',3]."""
    z = latents.transpose(1, 2, 3, 0)[None]  # [1,T,H,W,C]
    z = denormalize_latents(z)
    video = vae(z)
    mx.eval(video)
    return video


def main():
    model_dir = Path(os.path.expanduser(os.environ["WAN_5B_MODEL_DIR"]))
    fixture_path = Path(os.path.expanduser(os.environ["WAN_5B_FIXTURE"]))

    with open(model_dir / "config.json") as f:
        cfg_json = json.load(f)
    fields = WanModelConfig.__dataclass_fields__
    cdict = {k: v for k, v in cfg_json.items() if k in fields}
    for key in ("patch_size", "vae_stride", "window_size", "sample_guide_scale"):
        if key in cdict and isinstance(cdict[key], list):
            cdict[key] = tuple(cdict[key])
    config = WanModelConfig(**cdict)
    assert config.model_type == "ti2v" and not config.dual_model, "expected the dense TI2V-5B config"

    shift = config.sample_shift
    neg_prompt = config.sample_neg_prompt
    vae_stride, patch, z_dim = config.vae_stride, config.patch_size, config.vae_z_dim
    t_lat = (FRAMES - 1) // vae_stride[0] + 1
    h_lat, w_lat = HEIGHT // vae_stride[1], WIDTH // vae_stride[2]
    seq_len = math.ceil((h_lat * w_lat) / (patch[1] * patch[2]) * t_lat)
    grid = (t_lat // patch[0], h_lat // patch[1], w_lat // patch[2])
    target_shape = (z_dim, t_lat, h_lat, w_lat)

    # --- Real UMT5 encode (staged: freed before the DiT) ---
    print("Loading T5, encoding prompt + neg...")
    from transformers import AutoTokenizer

    t5 = load_t5_encoder(model_dir / "t5_encoder.safetensors", config)
    tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    context = encode_text(t5, tokenizer, PROMPT, config.text_len)
    context_null = encode_text(t5, tokenizer, neg_prompt, config.text_len)
    mx.eval(context, context_null)
    del t5

    # --- Injected init noise + a deterministic image at exactly (W,H) (preprocess = identity) ---
    mx.random.seed(1234)
    noise = mx.random.normal(target_shape).astype(mx.float32)
    # Deterministic image tensor [1,1,H,W,3] in [-1,1] (avoids PIL resize: source == target size).
    ys = mx.arange(HEIGHT).reshape(HEIGHT, 1, 1).astype(mx.float32) / HEIGHT
    xs = mx.arange(WIDTH).reshape(1, WIDTH, 1).astype(mx.float32) / WIDTH
    ch = mx.arange(3).reshape(1, 1, 3).astype(mx.float32) / 3.0
    img = mx.sin(6.2831 * (ys + xs + ch))  # [H,W,3] in [-1,1]
    img_thwc = img[None, None]  # [1,1,H,W,3]
    mx.eval(noise, img_thwc)

    # --- TI2V: encode the image → z_img via the real z48 VAE encoder ---
    print("Encoding image with the z48 VAE encoder...")
    vae_enc = load_vae_encoder(model_dir / "vae.safetensors", config)
    z_img_cl = vae_enc.encode(img_thwc)  # [1,1,h,w,z]
    mx.eval(z_img_cl)
    z_img = z_img_cl[0].transpose(3, 0, 1, 2)  # [z,1,h,w]
    i2v_mask, i2v_mask_tokens = build_i2v_mask(target_shape, patch)
    mx.eval(z_img, i2v_mask, i2v_mask_tokens)
    del vae_enc

    # --- Load the single dense DiT; embed contexts per CFG branch; precompute B=1 cross-KV + RoPE ---
    # The Rust port runs CFG as two **B=1** forwards (a memory tradeoff), not the reference's batched
    # B=2 — and bf16 B=2 ≠ 2×B=1 (~2%/forward, amplified ×guide). To gate the *port* tightly (the
    # per-forward is bit-exact, S3=0.0), dump the golden with the same B=1 forwards.
    print("Loading the 5B DiT...")
    model = load_wan_model(model_dir / "model.safetensors", config)
    emb = model.embed_text([context, context_null])  # [2, text_len, dim]
    ctx_cond, ctx_uncond = emb[0:1], emb[1:2]
    kv_cond = model.prepare_cross_kv(ctx_cond)
    kv_uncond = model.prepare_cross_kv(ctx_uncond)
    rope1 = model.prepare_rope([grid])
    mx.eval(ctx_cond, ctx_uncond, kv_cond, kv_uncond, rope1)

    def run_denoise(per_token, latents_init, mask=None, z_img=None):
        sched = FlowUniPCScheduler(num_train_timesteps=config.num_train_timesteps)
        sched.set_timesteps(STEPS, shift=shift)
        latents = latents_init
        for t in sched.timesteps.tolist():
            if per_token:
                t1 = i2v_mask_tokens * t
                pad = seq_len - t1.shape[1]
                if pad > 0:
                    t1 = mx.concatenate([t1, mx.full((1, pad), t)], axis=1)
            else:
                t1 = mx.array([t])
            pred_cond = model([latents], t=t1, context=ctx_cond, seq_len=seq_len,
                              cross_kv_caches=kv_cond, rope_cos_sin=rope1)[0]
            pred_uncond = model([latents], t=t1, context=ctx_uncond, seq_len=seq_len,
                                cross_kv_caches=kv_uncond, rope_cos_sin=rope1)[0]
            noise_pred = pred_uncond + GUIDE * (pred_cond - pred_uncond)
            latents = sched.step(noise_pred[None], t, latents[None]).squeeze(0)
            if per_token:
                latents = (1.0 - mask) * z_img + mask * latents
            mx.eval(latents)
        return latents

    print("T2V denoise...")
    t2v_latents = run_denoise(False, noise)
    print("TI2V mask-blend denoise...")
    ti2v_init = (1.0 - i2v_mask) * z_img + i2v_mask * noise
    ti2v_latents = run_denoise(True, ti2v_init, mask=i2v_mask, z_img=z_img)

    # Single-forward parity probes (isolate the DiT forward / forward_tokens from the denoise loop):
    # the first-step cond forward for T2V (scalar t) and TI2V (per-token t_tokens).
    _sp = FlowUniPCScheduler(num_train_timesteps=config.num_train_timesteps)
    _sp.set_timesteps(STEPS, shift=shift)
    t0 = float(_sp.timesteps.tolist()[0])
    t2v_fwd0 = model([noise], t=mx.array([t0]), context=ctx_cond, seq_len=seq_len,
                     cross_kv_caches=kv_cond, rope_cos_sin=rope1)[0]
    tt0 = i2v_mask_tokens * t0
    ti2v_fwd0 = model([ti2v_init], t=tt0, context=ctx_cond, seq_len=seq_len,
                      cross_kv_caches=kv_cond, rope_cos_sin=rope1)[0]
    mx.eval(t2v_fwd0, ti2v_fwd0)
    del model

    # --- Real z48 VAE decode for both ---
    print("Decoding with the z48 VAE...")
    vae = load_vae_decoder(model_dir / "vae.safetensors", config)
    t2v_video = vae22_decode_to_video(vae, t2v_latents)
    ti2v_video = vae22_decode_to_video(vae, ti2v_latents)

    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(fixture_path),
        {
            "noise": noise,
            "img_thwc": img_thwc.astype(mx.float32),
            "context": context,
            "context_null": context_null,
            "z_img": z_img.astype(mx.float32),
            "mask": i2v_mask.astype(mx.float32),
            "mask_tokens": i2v_mask_tokens.astype(mx.float32),
            "t2v_final_latents": t2v_latents.astype(mx.float32),
            "t2v_video": t2v_video.astype(mx.float32),
            "ti2v_final_latents": ti2v_latents.astype(mx.float32),
            "ti2v_video": ti2v_video.astype(mx.float32),
            "ti2v_init": ti2v_init.astype(mx.float32),
            "t2v_fwd0": t2v_fwd0.astype(mx.float32),
            "ti2v_fwd0": ti2v_fwd0.astype(mx.float32),
            "t0": mx.array([t0], dtype=mx.float32),
        },
    )
    meta = {
        "prompt": PROMPT,
        "neg_prompt": neg_prompt,
        "frames": FRAMES,
        "height": HEIGHT,
        "width": WIDTH,
        "steps": STEPS,
        "scheduler": "unipc",
        "shift": shift,
        "guidance": GUIDE,
        "seq_len": seq_len,
        "grid": list(grid),
        "z_dim": z_dim,
        "t2v_final_latents_shape": list(t2v_latents.shape),
        "t2v_video_shape": list(t2v_video.shape),
        "ti2v_final_latents_shape": list(ti2v_latents.shape),
        "ti2v_video_shape": list(ti2v_video.shape),
    }
    with open(fixture_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"T2V  latents {tuple(t2v_latents.shape)}  video {tuple(t2v_video.shape)}")
    print(f"TI2V latents {tuple(ti2v_latents.shape)}  video {tuple(ti2v_video.shape)}")
    print(f"wrote {fixture_path}")


if __name__ == "__main__":
    main()
