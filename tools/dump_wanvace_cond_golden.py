"""Golden for the Wan-VACE conditioning host ops (epic 3040 / sc-3388, S2 / sc-3435).

Replicates diffusers `WanVACEPipeline.prepare_masks` + the `prepare_video_latents` masking exactly
(pure tensor ops — no VAE, no checkpoint), so the Rust host port (`vace::prepare_masks` + the
inactive/reactive masking + ref/concat assembly) can be byte-validated. The VAE-encode + normalize is
already validated separately (`WanVae::encode`, sc-2678), so it is NOT exercised here.

Params mirror a valid Wan2.1-VACE geometry: vae_scale_factor_temporal=4, vae_scale_factor_spatial=8,
transformer patch_size[1]=2; num_frames=13 (→ 4 latent frames), H=W=32 (→ new_h=new_w=4), num_ref=1.

Run: /Users/michael/repos/mflux/.venv-0312/bin/python tools/dump_wanvace_cond_golden.py
Writes `mlx-gen-wan/tests/fixtures/wanvace_cond_golden.safetensors`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from _paths import fixture

torch.manual_seed(3435)

VAE_T = 4  # vae_scale_factor_temporal
VAE_S = 8  # vae_scale_factor_spatial
PATCH = 2  # transformer patch_size[1]
F, H, W = 13, 32, 32
NUM_REF = 1

# Inputs (the preprocessed control video + mask, already in [-1,1] / [0,1] as the worker passes them).
video = torch.randn(3, F, H, W)
mask = torch.rand(3, F, H, W)  # in [0,1]


def prepare_masks(mask_, num_ref):
    # Exact copy of WanVACEPipeline.prepare_masks' inner loop (single batch).
    num_channels, num_frames, height, width = mask_.shape
    new_num_frames = (num_frames + VAE_T - 1) // VAE_T
    new_height = height // (VAE_S * PATCH) * PATCH
    new_width = width // (VAE_S * PATCH) * PATCH
    m = mask_[0, :, :, :]
    m = m.view(num_frames, new_height, VAE_S, new_width, VAE_S)
    m = m.permute(2, 4, 0, 1, 3).flatten(0, 1)  # [64, num_frames, new_h, new_w]
    m = torch.nn.functional.interpolate(
        m.unsqueeze(0), size=(new_num_frames, new_height, new_width), mode="nearest-exact"
    ).squeeze(0)
    if num_ref > 0:
        mask_padding = torch.zeros_like(m[:, :num_ref, :, :])
        m = torch.cat([mask_padding, m], dim=1)
    return m


mask_latent = prepare_masks(mask, NUM_REF)  # [64, new_num_frames + num_ref, new_h, new_w]

# The video-latent masking (pre-VAE): where(mask>0.5,1,0); inactive = video·(1−m); reactive = video·m.
m_bin = torch.where(mask > 0.5, 1.0, 0.0)
inactive = video * (1 - m_bin)
reactive = video * m_bin

tensors = {
    "in.video": video.contiguous(),
    "in.mask": mask.contiguous(),
    "out.mask_latent": mask_latent.contiguous(),
    "out.m_bin": m_bin.contiguous(),
    "out.inactive": inactive.contiguous(),
    "out.reactive": reactive.contiguous(),
}
out_path = fixture("mlx-gen-wan/tests/fixtures/wanvace_cond_golden.safetensors")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
save_file(tensors, out_path)
print(f"wrote {out_path}")
for k, v in tensors.items():
    print(f"  {k:18s} {tuple(v.shape)}")
