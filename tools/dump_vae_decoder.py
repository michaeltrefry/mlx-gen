"""Full VAE decoder-assembly parity fixture (sc-2344).

The fork's `Decoder` hardcodes channels (16→512→…→128→3), whose trained weights are far too
large to commit. Since the sub-modules are channel-parameterizable, we mirror the exact
`Decoder.__call__` STRUCTURE at tiny channels (≥32 for GroupNorm's 32 groups): conv_in →
mid_block → 2 up-blocks (one upsampling, one not) → conv_norm_out → SiLU → conv_out. That
exercises the full assembly wiring (channel changes, block order, upsample) with a tiny,
committable fixture. The Rust `Decoder` is config-driven and runs the same small config.

Run from the mflux fork venv:
    cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_vae_decoder.py
"""

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from mflux.models.z_image.model.z_image_vae.decoder.conv_in import ConvIn
from mflux.models.z_image.model.z_image_vae.decoder.conv_out import ConvOut
from mflux.models.z_image.model.z_image_vae.decoder.conv_norm_out import ConvNormOut
from mflux.models.z_image.model.z_image_vae.decoder.up_decoder_block import UpDecoderBlock
from mflux.models.z_image.model.z_image_vae.common.unet_mid_block import UNetMidBlock

mx.random.seed(0)
out = {}


def randomize_groupnorms(module, prefix):
    for name, _ in tree_flatten(module.parameters()):
        if name.endswith(".weight") and ("norm" in name):
            base = module
            for part in name.split(".")[:-1]:
                base = base[int(part)] if part.isdigit() else getattr(base, part)
            base.weight = 1.0 + 0.1 * mx.random.normal(base.weight.shape)
            base.bias = 0.05 * mx.random.normal(base.bias.shape)


def add(prefix, module):
    for k, v in tree_flatten(module.parameters()):
        out[f"{prefix}.{k}"] = v.astype(mx.float32)


# Small decoder mirroring Decoder.__init__/.__call__ structure.
conv_in = ConvIn(in_channels=16, out_channels=64)
mid_block = UNetMidBlock(channels=64)
up0 = UpDecoderBlock(in_channels=64, out_channels=64, num_layers=3, add_upsample=True)
up1 = UpDecoderBlock(in_channels=64, out_channels=32, num_layers=3, add_upsample=False)
conv_norm_out = ConvNormOut(channels=32)
conv_out = ConvOut(in_channels=32, out_channels=3)

for m, p in [(mid_block, "mid_block"), (up0, "up_blocks.0"), (up1, "up_blocks.1"), (conv_norm_out, "conv_norm_out")]:
    randomize_groupnorms(m, p)

add("conv_in", conv_in)
add("mid_block", mid_block)
add("up_blocks.0", up0)
add("up_blocks.1", up1)
add("conv_norm_out", conv_norm_out)
add("conv_out", conv_out)

latent = mx.random.normal((1, 16, 8, 8))  # (B, C=16, H, W)
h = conv_in(latent)
h = mid_block(h)
h = up0(h)
h = up1(h)
h = conv_norm_out(h)
h = nn.silu(h)
y = conv_out(h)

out["in.latent"] = latent.astype(mx.float32)
out["out.image"] = y.astype(mx.float32)

path = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/vae_decoder.safetensors"
mx.save_safetensors(path, out)
print(f"wrote {path} ({len(out)} tensors)")
print("latent:", latent.shape, "-> image:", y.shape)
