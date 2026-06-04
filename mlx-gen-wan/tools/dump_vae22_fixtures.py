#!/usr/bin/env python3
"""Dump sc-2680 vae22 parity fixtures from the `mlx_video` Wan2.2 reference: the z48 `vae22` decode
(causal `first_chunk`) + chunked encode, for the Rust port (`src/vae22.rs`) to gate against.

Like the S2 z16 gate, the 5B's production VAE weights are heavy (and not committable), so we gate
against a **self-contained tiny instance**: `dec_dim=8` / `enc_dim=8` Wan22 VAE with deterministically
seeded random weights, but the **real z_dim=48** (so the hardcoded `VAE22_MEAN`/`VAE22_STD` apply and
are gated). Tiny base widths keep the committed fixture small while exercising **every** vae22 code
path: channels-last causal 3-D conv (`2·pad` left), channel-L2 `RMS_norm` (eps 1e-24), per-frame
spatial attention, `DupUp3D`/`AvgDown3D` shortcuts, the up/down `Resample` `time_conv` (incl. the
`first_chunk` interleave + the `downsample3d` chunk-cache), spatial 2×2 patchify, the chunked-encode
`feat_cache`, and mean/std (de)normalization.

Run with the SceneWorks venv that has `mlx_video` + `mlx` installed:

    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/dump_vae22_fixtures.py

Writes (committed; small):
  - mlx-gen-wan/tests/fixtures/vae22.json             (dims, io shapes, VAE22_MEAN/STD)
  - mlx-gen-wan/tests/fixtures/vae22.safetensors      (random weights + decode/encode in+out, f32)
"""
import json
import os

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from mlx_video.models.wan.vae22 import (
    VAE22_MEAN,
    VAE22_STD,
    Wan22VAEDecoder,
    Wan22VAEEncoder,
    denormalize_latents,
)

DEC_DIM = 8   # tiny (production 256) — keeps the fixture small; architecture is dim-parametric
ENC_DIM = 8   # tiny (production 160)
Z_DIM = 48    # real z_dim (VAE22_MEAN/STD are 48-long → kept, not randomized)


def randomize(model, seed):
    """Seeded random weights for every learnable param (conv weight/bias, norm gamma)."""
    mx.random.seed(seed)
    flat = tree_flatten(model.parameters())
    new = [(k, (mx.random.normal(v.shape) * 0.5).astype(mx.float32)) for k, v in flat]
    model.update(tree_unflatten(new))
    mx.eval(model.parameters())
    return model


def main():
    dec = randomize(Wan22VAEDecoder(z_dim=Z_DIM, dim=ENC_DIM, dec_dim=DEC_DIM), 0)
    enc = randomize(Wan22VAEEncoder(z_dim=Z_DIM, dim=ENC_DIM), 1)

    # Merge weights (no key collision: decoder = conv2.* / decoder.*, encoder = conv1.* / encoder.*).
    save = {}
    for k, v in tree_flatten(dec.parameters()):
        save[k] = v.astype(mx.float32)
    for k, v in tree_flatten(enc.parameters()):
        save[k] = v.astype(mx.float32)
    print(f"=== {len(save)} weight tensors (dec_dim={DEC_DIM}, enc_dim={ENC_DIM}, z={Z_DIM}) ===")

    # Decode: a normalized channels-first latent [z, T, H, W] → video [1, T', 16H, 16W, 3].
    # Mirrors generate_wan.py: transpose to channels-last, denormalize, then vae(z).
    mx.random.seed(2)
    dec_in = (mx.random.normal((Z_DIM, 2, 2, 2)) * 0.5).astype(mx.float32)  # [z, T, H, W]
    z = dec_in.transpose(1, 2, 3, 0)[None]  # [1, T, H, W, z]
    z = denormalize_latents(z)
    dec_out = dec(z)  # [1, T', 32, 32, 3] in [-1, 1]
    mx.eval(dec_out)

    # Encode: a channels-last video [1, T, H, W, 3] (T = 1+4k) in [-1, 1] → normalized latent.
    mx.random.seed(3)
    enc_in = mx.clip(mx.random.normal((1, 5, 32, 32, 3)).astype(mx.float32), -1.0, 1.0)
    enc_out = enc.encode(enc_in)  # [1, T_lat, 2, 2, z]
    mx.eval(enc_out)

    # Encode T=1 (the TI2V single-image conditioning path — distinct chunking from T=5).
    mx.random.seed(7)
    enc_in1 = mx.clip(mx.random.normal((1, 1, 32, 32, 3)).astype(mx.float32), -1.0, 1.0)
    enc_out1 = enc.encode(enc_in1)  # [1, 1, 2, 2, z]
    mx.eval(enc_out1)

    # Guard against a degenerate fixture (a zero weight makes the encode input-independent → the
    # gate would pass garbage-to-garbage). Every conv/gamma must be non-zero.
    for k, v in tree_flatten(enc.parameters()):
        assert float(mx.abs(v).max()) > 0, f"encoder param {k} is all-zero (degenerate fixture)"

    print(f"dec_in {tuple(dec_in.shape)} -> dec_out {tuple(dec_out.shape)}")
    print(f"enc_in {tuple(enc_in.shape)} -> enc_out {tuple(enc_out.shape)}")
    print(f"enc_in1 {tuple(enc_in1.shape)} -> enc_out1 {tuple(enc_out1.shape)}")

    save["dec_in"] = dec_in
    save["dec_out"] = dec_out
    save["enc_in"] = enc_in
    save["enc_out"] = enc_out
    save["enc_in1"] = enc_in1
    save["enc_out1"] = enc_out1

    dst = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")
    os.makedirs(dst, exist_ok=True)
    st_path = os.path.join(dst, "vae22.safetensors")
    mx.save_safetensors(st_path, save)

    meta = {
        "dec_dim": DEC_DIM,
        "enc_dim": ENC_DIM,
        "z_dim": Z_DIM,
        "dec_in_shape": list(dec_in.shape),
        "dec_out_shape": list(dec_out.shape),
        "enc_in_shape": list(enc_in.shape),
        "enc_out_shape": list(enc_out.shape),
        "vae22_mean": list(map(float, VAE22_MEAN)),
        "vae22_std": list(map(float, VAE22_STD)),
        "num_weight_tensors": len(save) - 4,
    }
    with open(os.path.join(dst, "vae22.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {os.path.abspath(st_path)} ({os.path.getsize(st_path) / 1e6:.2f} MB)")
    print(f"wrote {os.path.abspath(os.path.join(dst, 'vae22.json'))}")


if __name__ == "__main__":
    main()
