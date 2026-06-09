"""Kolors IP-Adapter-Plus golden — reference for the image-conditioning parity (sc-3098).

`Kwai-Kolors/Kolors-IP-Adapter-Plus` = CLIP-ViT-L/14-336 image tower → the IP-Adapter "plus"
Resampler (`image_proj.*`, 16×2048 tokens) → decoupled cross-attention. This dumps the two
torch-checkable components so the Rust port can isolate each:

 - `pixels` — the CLIP-preprocessed reference image (NHWC [1,336,336,3]): shortest-side bicubic
   resize to 336, center-crop, `[0,255]→[0,1]`, CLIP mean/std (the `CLIPImageProcessor` recipe,
   done manually to avoid the torchvision dep). The Rust `preprocess_clip_image_sized(.,336)` must
   reproduce it.
 - `penultimate` — transformers `CLIPVisionModelWithProjection` `hidden_states[-2]` ([1,577,1024]),
   the input the Resampler consumes. The Rust `ClipVisionEncoder::penultimate` must match.
 - `tokens` — the Resampler output ([1,16,2048]) from a faithful torch port of the Tencent
   IP-Adapter `Resampler` (dim 2048, depth 4, dim_head 64, heads 12 — pinned by the on-disk shapes),
   loaded from the checkpoint's `image_proj.*`. The Rust `Resampler::forward(penultimate)` must match.

Also bundles `image` (raw u8 reference, f32 [0,1] [H,W,3]) for the Rust preprocess check.

Run: ~/repos/mflux/.venv-0312/bin/python tools/dump_kolors_ip_adapter_golden.py
Output (gitignored): tools/golden/kolors_ip_adapter_golden.safetensors
"""

import glob
import math
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage

from _paths import fixture, hf_hub_cache

from transformers import CLIPVisionModelWithProjection

H = W = 512
CROP = 336
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def snapshot() -> Path:
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-IP-Adapter-Plus" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-IP-Adapter-Plus snapshot not found in HF cache")
    return Path(snaps[-1])


# ---- the Tencent IP-Adapter "plus" Resampler (faithful port; the layout mlx-gen-sdxl reuses) ----
def reshape_tensor(x, heads):
    bs, length, _ = x.shape
    return x.view(bs, length, heads, -1).transpose(1, 2)


class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner = dim_head * heads
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def forward(self, x, latents):
        x = self.norm1(x)
        latents = self.norm2(latents)
        b, l, _ = latents.shape
        q = self.to_q(latents)
        kv = self.to_kv(torch.cat((x, latents), dim=-2))
        k, v = kv.chunk(2, dim=-1)
        q, k, v = (reshape_tensor(t, self.heads) for t in (q, k, v))
        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.to_out(out)


def feed_forward(dim, mult=4):
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, inner, bias=False), nn.GELU(), nn.Linear(inner, dim, bias=False)
    )


class Resampler(nn.Module):
    def __init__(self, dim, depth, dim_head, heads, num_queries, embedding_dim, output_dim, ff_mult=4):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(1, num_queries, dim))
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList(
            [nn.ModuleList([PerceiverAttention(dim, dim_head, heads), feed_forward(dim, ff_mult)]) for _ in range(depth)]
        )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))


def make_ref_pixels() -> np.ndarray:
    """A deterministic 512² RGB reference image (uint8)."""
    yy, xx = np.mgrid[0:H, 0:W]
    r = (xx * 255 // (W - 1)).astype(np.uint8)
    g = (yy * 255 // (H - 1)).astype(np.uint8)
    b = ((xx ^ yy) % 256).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img[160:352, 160:352] = np.array([40, 200, 220], dtype=np.uint8)
    return img


def clip_preprocess(pixels: np.ndarray) -> np.ndarray:
    """CLIPImageProcessor: shortest-side bicubic→336, center-crop 336, /255, CLIP mean/std → NCHW."""
    pil = PILImage.fromarray(pixels, "RGB")
    iw, ih = pil.size
    scale = CROP / min(iw, ih)
    rw, rh = max(round(iw * scale), CROP), max(round(ih * scale), CROP)
    pil = pil.resize((rw, rh), PILImage.BICUBIC)
    left, top = (rw - CROP) // 2, (rh - CROP) // 2
    pil = pil.crop((left, top, left + CROP, top + CROP))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD  # HWC
    return arr.transpose(2, 0, 1)[None]  # NCHW [1,3,336,336]


@torch.no_grad()
def main():
    snap = snapshot()
    pixels = make_ref_pixels()
    nchw = clip_preprocess(pixels)

    enc = CLIPVisionModelWithProjection.from_pretrained(snap / "image_encoder", torch_dtype=torch.float32)
    out = enc(torch.tensor(nchw), output_hidden_states=True)
    penultimate = out.hidden_states[-2]  # [1,577,1024]

    # Load the Resampler weights from image_proj.* and run.
    ip_path = snap / "ip_adapter_plus_general.safetensors"
    from safetensors.torch import load_file
    sd = load_file(str(ip_path))
    image_proj = {k[len("image_proj.") :]: v for k, v in sd.items() if k.startswith("image_proj.")}
    resampler = Resampler(
        dim=2048, depth=4, dim_head=64, heads=12, num_queries=16, embedding_dim=1024, output_dim=2048
    ).to(torch.float32)
    missing, unexpected = resampler.load_state_dict(image_proj, strict=False)
    print("resampler load: missing", missing, "unexpected", unexpected)
    tokens = resampler(penultimate.float())  # [1,16,2048]

    def nhwc(t):
        return mx.array(t.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32))

    def arr(t):
        return mx.array(t.detach().cpu().numpy().astype(np.float32))

    tensors = {
        "image": mx.array(pixels.astype(np.float32) / 255.0),  # [H,W,3] in [0,1]
        "pixels": nhwc(torch.tensor(nchw)),  # [1,336,336,3]
        "penultimate": arr(penultimate),  # [1,577,1024]
        "tokens": arr(tokens),  # [1,16,2048]
    }
    mx.eval(list(tensors.values()))
    meta = {"crop": str(CROP), "h": str(H), "w": str(W)}
    out_path = fixture("tools/golden/kolors_ip_adapter_golden.safetensors")
    mx.save_safetensors(out_path, tensors, metadata=meta)
    print(f"wrote {out_path}")
    print(f"  penultimate {tuple(tensors['penultimate'].shape)} tokens {tuple(tensors['tokens'].shape)}")


if __name__ == "__main__":
    main()
