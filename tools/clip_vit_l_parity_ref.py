#!/usr/bin/env python
"""sc-3622 parity reference: torch `CLIPVisionModelWithProjection` (openai/clip-vit-large-patch14).

Emits a fixed, deterministic normalized pixel tensor + the torch `.image_embeds` it produces, so the
MLX `FluxIpImageEncoder` can be checked against the exact HF output (bypassing preprocessing — both
sides consume the SAME normalized pixels). Run with the torch venv:

    ~/mlx-flux-venv/bin/python tools/clip_vit_l_parity_ref.py /tmp/clip_l

writes <out>.pixels (NHWC [1,224,224,3] f32 LE) and <out>.embeds ([768] f32 LE).
"""
import sys
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/clip_l"

# Deterministic normalized-image-range pixels [1,3,224,224] (no RNG → portable).
H = W = 224
yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
c0 = np.sin(xx / 20.0) + np.cos(yy / 17.0)
c1 = np.sin((xx + yy) / 13.0)
c2 = np.cos(xx / 23.0) - np.sin(yy / 11.0)
px_nchw = np.stack([c0, c1, c2], axis=0).astype(np.float32)[None]  # [1,3,224,224]

model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=torch.float32
).eval()
with torch.no_grad():
    embeds = model(pixel_values=torch.from_numpy(px_nchw)).image_embeds  # [1,768]
embeds = embeds.detach().cpu().numpy().astype("<f4").reshape(-1)

px_nhwc = np.ascontiguousarray(px_nchw.transpose(0, 2, 3, 1)).astype("<f4")
px_nhwc.tofile(out + ".pixels")
embeds.tofile(out + ".embeds")
print(f"wrote {out}.pixels ({px_nhwc.size} f32) + {out}.embeds ({embeds.size} f32)")
print(f"embeds[:6]={embeds[:6]}  norm={np.linalg.norm(embeds):.4f}")
