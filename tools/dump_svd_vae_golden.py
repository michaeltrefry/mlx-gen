"""Dump a golden for the SVD VAE (epic 3040 / sc-3372) from the real diffusers
`AutoencoderKLTemporalDecoder` (the SVD `vae`). Validates the Rust `SvdVae`:
  - `encode_mode` vs `vae.encode(image).latent_dist.mode()`
  - `decode`     vs `vae.decode(z, num_frames=F).sample`

Both directions run in float32 so the parity gate isolates the math from fp16 rounding. Sizes are
kept small (64px image → 8px latent, 3 frames) so the temporal Conv3d path is exercised cheaply.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_svd_vae_golden.py
Writes `mlx-gen-svd/tests/fixtures/svd_vae_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLTemporalDecoder
from safetensors.numpy import save_file

from _paths import fixture, hf_hub_cache

SNAP = (
    hf_hub_cache()
    / "models--stabilityai--stable-video-diffusion-img2vid-xt"
    / "snapshots"
)
snap_dir = next(SNAP.iterdir())
vae_dir = snap_dir / "vae"

vae = AutoencoderKLTemporalDecoder.from_pretrained(vae_dir, torch_dtype=torch.float32)
vae.eval()

rng = np.random.default_rng(3372)
num_frames = 3
# Encoder input: a single image [1,3,64,64] in roughly [-1,1].
image = rng.standard_normal((1, 3, 64, 64)).astype(np.float32)
# Decoder input: a latent [F,4,8,8] (one batch, F frames) — the post-scaling diffusion latent.
z = rng.standard_normal((num_frames, 4, 8, 8)).astype(np.float32)

with torch.no_grad():
    mode = vae.encode(torch.from_numpy(image)).latent_dist.mode()
    mode = mode.cpu().numpy().astype(np.float32)  # [1,4,8,8]
    frames = vae.decode(torch.from_numpy(z), num_frames=num_frames).sample
    frames = frames.cpu().numpy().astype(np.float32)  # [F,3,64,64]

tensors = {
    "image": image,  # NCHW
    "encode_mode": mode,  # NCHW
    "z": z,  # NCHW
    "decode_frames": frames,  # NCHW
    "num_frames": np.array([num_frames], dtype=np.int32),
}
out_path = fixture("mlx-gen-svd/tests/fixtures/svd_vae_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
print("  image:", image.shape, " encode_mode:", mode.shape)
print("  z:", z.shape, " decode_frames:", frames.shape, " num_frames:", num_frames)
print("  mode[0,:, 0, 0]:", mode[0, :, 0, 0])
print("  frames[0, :, 0, 0]:", frames[0, :, 0, 0])
