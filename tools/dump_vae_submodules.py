"""Parity fixtures for the Z-Image VAE decoder sub-modules (sc-2344): ResnetBlock2D,
Attention (mid-block self-attention), UpSampler. Validates the convolutional op family
(Conv2d + pytorch-compatible GroupNorm + nearest-upsample) the rest of the decoder builds on.

Small channels (32/64) keep fixtures tiny; GroupNorm uses num_groups=32 so channels must be
multiples of 32. GroupNorm weights/biases are randomized to exercise the affine path. NCHW
I/O (mirrors the fork's per-module transpose convention). fp32.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_vae_submodules.py
"""

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_vae.common.resnet_block_2d import ResnetBlock2D
from mflux.models.z_image.model.z_image_vae.common.attention import Attention
from mflux.models.z_image.model.z_image_vae.decoder.up_sampler import UpSampler

mx.random.seed(0)
out = {}


def randomize_groupnorms(module):
    # Default GroupNorm weight=ones / bias=zeros would hide affine bugs; randomize them.
    for name, _ in tree_flatten(module.parameters()):
        if name.endswith("weight") and ("norm" in name or "group_norm" in name):
            base = module
            for part in name.split(".")[:-1]:
                base = base[int(part)] if part.isdigit() else getattr(base, part)
            base.weight = 1.0 + 0.1 * mx.random.normal(base.weight.shape)
            base.bias = 0.05 * mx.random.normal(base.bias.shape)


def add(prefix, module):
    for k, v in tree_flatten(module.parameters()):
        out[f"{prefix}.{k}"] = v.astype(mx.float32)


# ResnetBlock2D 32 -> 64 (channel change -> 1x1 conv shortcut).
rb = ResnetBlock2D(in_channels=32, out_channels=64, use_conv_shortcut=False)
randomize_groupnorms(rb)
add("rb.w", rb)
rb_x = mx.random.normal((1, 32, 8, 8))
out["rb.in"] = rb_x.astype(mx.float32)
out["rb.out"] = rb(rb_x).astype(mx.float32)

# Attention (mid-block self-attention), channels=64.
attn = Attention(channels=64)
randomize_groupnorms(attn)
add("attn.w", attn)
attn_x = mx.random.normal((1, 64, 8, 8))
out["attn.in"] = attn_x.astype(mx.float32)
out["attn.out"] = attn(attn_x).astype(mx.float32)

# UpSampler 32 -> 32 (nearest-2x then 3x3 conv).
up = UpSampler(in_channels=32, out_channels=32)
add("up.w", up)
up_x = mx.random.normal((1, 32, 8, 8))
out["up.in"] = up_x.astype(mx.float32)
out["up.out"] = up(up_x).astype(mx.float32)

path = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/vae_submodules.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
for k in ("rb.out", "attn.out", "up.out"):
    print(f"  {k}: {out[k].shape}")
# Show conv weight layout (mlx NHWC: [out, kH, kW, in]).
print("conv1.weight layout:", out["rb.w.conv1.weight"].shape)
