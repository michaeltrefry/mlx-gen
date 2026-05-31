"""Dump Z-Image latent-lifecycle parity fixtures from the frozen mflux fork.

Run from the fork:  cd ~/repos/mflux && uv run python /Users/michael/repos/mlx-gen/tools/dump_z_latents.py

(1) Seeded noise: mx.random.normal([16,1,H/8,W/8], key=mx.random.key(seed)) — the version-drift
    RNG-parity gate (the crate links mlx-rs 0.25's bundled MLX vs the fork's 0.31).
(2) decoded -> image: the fork's ImageUtil denormalize -> to_numpy -> (x*255).round().uint8.
"""

import mlx.core as mx
import numpy as np
from mflux.utils.image_util import ImageUtil

OUT = "/Users/michael/repos/mlx-gen/mlx-gen-z-image/tests/fixtures/z_latents.safetensors"
tensors = {}
meta = {}

# (1) seeded noise — seed 42, 64x64 image -> latent [16,1,8,8]
seed, w, h = 42, 64, 64
noise = mx.random.normal([16, 1, h // 8, w // 8], key=mx.random.key(seed))
tensors["noise"] = noise.astype(mx.float32)
meta["noise_cfg"] = f"{seed},{w},{h}"  # seed,width,height

# (2) decoded VAE tensor [B,C,H,W] in ~[-1.5,1.5] -> RGB8 image bytes
mx.random.seed(7)
decoded = (mx.random.normal([1, 3, 4, 4]) * 1.3).astype(mx.float32)
normalized = ImageUtil._denormalize(decoded)
npy = ImageUtil._to_numpy(normalized)  # [B,H,W,C] f32 in [0,1]
img_u8 = (npy * 255).round().astype("uint8")[0]  # [H,W,C] uint8, batch 0
tensors["decoded"] = decoded
tensors["image_i32"] = mx.array(img_u8.astype(np.int32))  # store as int32 (portable dtype)
meta["image_hwc"] = f"{img_u8.shape[0]},{img_u8.shape[1]},{img_u8.shape[2]}"

mx.save_safetensors(OUT, tensors, meta)
print(f"wrote {OUT}: {sorted(tensors)} meta={meta} noise_shape={tuple(noise.shape)}")
