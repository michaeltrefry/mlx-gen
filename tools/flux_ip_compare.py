"""sc-3624 A/B metric: CLIP-image resemblance between the reference and the two IP renders.

Quantifies whether the MLX XLabs IP-Adapter delivers resemblance comparable to the diffusers torch
path. Uses `openai/clip-vit-large-patch14` (the same tower the IP-Adapter conditions on). Run:

    HF_HUB_OFFLINE=1 ~/mlx-flux-venv/bin/python tools/flux_ip_compare.py /tmp/flux_ab
"""
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPVisionModelWithProjection

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/flux_ab"
MLX_NAME = sys.argv[2] if len(sys.argv) > 2 else "mlx_ip.png"
model = CLIPVisionModelWithProjection.from_pretrained(
    "openai/clip-vit-large-patch14", torch_dtype=torch.float32
).eval()

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def embed(name):
    # Manual CLIP preprocess (no preprocessor_config.json needed offline): resize shortest side to
    # 224 (bicubic), center-crop 224, normalize by CLIP mean/std → NCHW.
    img = Image.open(f"{OUT}/{name}").convert("RGB")
    w, h = img.size
    s = 224 / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    px = torch.from_numpy(arr.transpose(2, 0, 1)[None])
    with torch.no_grad():
        f = model(pixel_values=px).image_embeds  # [1, 768] projected pooled embeds
    return torch.nn.functional.normalize(f, dim=-1)


ref = embed("reference.png")
torch_ip = embed("torch_ip.png")
mlx_ip = embed(MLX_NAME)


def cos(a, b):
    return float((a * b).sum())


print(f"resemblance to reference  | torch IP = {cos(ref, torch_ip):.4f}   mlx IP = {cos(ref, mlx_ip):.4f}")
print(f"torch_ip <-> mlx_ip direct CLIP similarity = {cos(torch_ip, mlx_ip):.4f}")
delta = abs(cos(ref, torch_ip) - cos(ref, mlx_ip))
print(f"|Δ resemblance| = {delta:.4f}  ->  {'MATCH' if delta < 0.07 else 'DIVERGENT'} (tol 0.07)")
