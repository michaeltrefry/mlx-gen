#!/usr/bin/env python
"""Golden dump for the SDXL IP-Adapter spike (sc-3056).

Runs the reference *image* path in **float32** and saves intermediates so the Rust port can be
validated op-by-op, isolated from CLIP image preprocessing:

  - ``pixel_values``    : a deterministic [1, 3, 224, 224] input (seeded; NOT a real photo — parity
                          is parity regardless of input distribution, and this keeps preprocessing
                          out of the model-port test).
  - ``vit_penultimate`` : ViT-H ``hidden_states[-2]`` [1, 257, 1280] — the IP-"plus" image features.
  - ``ip_tokens``       : Resampler output [1, 16, 2048] — the image tokens fed to the UNet.

The Resampler is reimplemented here faithfully (original Tencent ``Resampler``/``PerceiverAttention``
layout) and loaded directly from ``image_proj.*`` of ``ip-adapter-plus_sdxl_vit-h.safetensors`` — so
a mismatch points at the Rust port, not at a diffusers key remap.

Run from the mflux venv (has torch + transformers + safetensors):
    ~/repos/mflux/.venv/bin/python ~/Repos/mlx-gen/tools/dump_ip_adapter_golden.py
"""
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import CLIPVisionModelWithProjection

HUB = Path.home() / ".cache/huggingface/hub"
IPA = (
    HUB
    / "models--h94--IP-Adapter/snapshots/018e402774aeeddd60609b4ecdb7e298259dc729"
)
ENCODER_DIR = IPA / "models/image_encoder"
IPA_WEIGHTS = IPA / "sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
OUT = Path(__file__).resolve().parent / "golden" / "ip_adapter_spike_golden.safetensors"


# ---- Faithful original-IP-Adapter Resampler (matches the image_proj.* checkpoint layout) ----
class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=20):
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
        kv = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        def rs(t):
            return t.reshape(b, -1, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = rs(q), rs(k), rs(v)
        s = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        w = (q * s) @ (k * s).transpose(-2, -1)
        w = torch.softmax(w.float(), dim=-1).type(w.dtype)
        out = w @ v
        out = out.transpose(1, 2).reshape(b, l, -1)
        return self.to_out(out)


def feed_forward(dim, mult=4):
    inner = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner, bias=False),
        nn.GELU(),
        nn.Linear(inner, dim, bias=False),
    )


class Resampler(nn.Module):
    def __init__(self, dim=1280, depth=4, dim_head=64, heads=20, num_queries=16,
                 embed=1280, out=2048):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim))
        self.proj_in = nn.Linear(embed, dim)
        self.proj_out = nn.Linear(dim, out)
        self.norm_out = nn.LayerNorm(out)
        self.layers = nn.ModuleList(
            [nn.ModuleList([PerceiverAttention(dim, dim_head, heads), feed_forward(dim)])
             for _ in range(depth)]
        )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm_out(self.proj_out(latents))


def main():
    torch.manual_seed(0)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic pixel_values (seeded). Isolates the model port from CLIP preprocessing.
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # ViT-H image encoder (f32), penultimate hidden state.
    enc = CLIPVisionModelWithProjection.from_pretrained(ENCODER_DIR, torch_dtype=torch.float32)
    enc.eval()
    with torch.no_grad():
        out = enc(pixel_values, output_hidden_states=True)
        hs = out.hidden_states  # tuple of 33: (pre_ln_out, L0_out, ..., L31_out)
        vit_penultimate = hs[-2]  # [1, 257, 1280] == hs[31] (output of layer 30)
        # Bisection checkpoints to localize any per-layer drift.
        vit_h0 = hs[0]  # embeddings + pre_layrnorm (input to layer 0)
        vit_h1 = hs[1]  # output of layer 0
        vit_h16 = hs[16]  # output of layer 15

    # Resampler (f32) loaded from image_proj.*.
    sd = load_file(str(IPA_WEIGHTS))
    image_proj = {k[len("image_proj."):]: v.float() for k, v in sd.items()
                  if k.startswith("image_proj.")}
    res = Resampler()
    missing, unexpected = res.load_state_dict(image_proj, strict=False)
    assert not missing, f"resampler missing keys: {missing}"
    assert not unexpected, f"resampler unexpected keys: {unexpected}"
    res.eval()
    with torch.no_grad():
        ip_tokens = res(vit_penultimate)  # [1, 16, 2048]

    # --- Primitive isolations (localize the per-layer drift): exact GELU + LayerNorm ---
    torch.manual_seed(1)
    gelu_in = (torch.randn(2048) * 4.0).float()
    gelu_out = torch.nn.functional.gelu(gelu_in)  # ACT2FN["gelu"] == exact erf GELU
    ln_in = torch.randn(257, 1280).float()
    ln = torch.nn.LayerNorm(1280, eps=1e-5)
    with torch.no_grad():
        ln.weight.copy_(torch.randn(1280))
        ln.bias.copy_(torch.randn(1280))
        ln_out = ln(ln_in)

    save_file(
        {
            "pixel_values": pixel_values.contiguous(),
            "vit_h0": vit_h0.contiguous(),
            "vit_h1": vit_h1.contiguous(),
            "vit_h16": vit_h16.contiguous(),
            "vit_penultimate": vit_penultimate.contiguous(),
            "ip_tokens": ip_tokens.contiguous(),
            "gelu_in": gelu_in.contiguous(),
            "gelu_out": gelu_out.contiguous(),
            "ln_in": ln_in.contiguous(),
            "ln_w": ln.weight.detach().contiguous(),
            "ln_b": ln.bias.detach().contiguous(),
            "ln_out": ln_out.detach().contiguous(),
        },
        str(OUT),
    )
    print(f"wrote {OUT}")
    print(f"  pixel_values    {tuple(pixel_values.shape)}")
    print(f"  vit_penultimate {tuple(vit_penultimate.shape)}  "
          f"mean={vit_penultimate.mean():.5f} std={vit_penultimate.std():.5f}")
    print(f"  ip_tokens       {tuple(ip_tokens.shape)}  "
          f"mean={ip_tokens.mean():.5f} std={ip_tokens.std():.5f}")


if __name__ == "__main__":
    main()
