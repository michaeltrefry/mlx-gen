"""Kolors ControlNet single-forward golden — reference for the ControlNet wiring parity (sc-3097).

The Kolors ControlNet (`Kwai-Kolors/Kolors-ControlNet-Pose`) is a standard diffusers SDXL
`ControlNetModel` whose only deltas are the same two ChatGLM-driven pieces as the Kolors U-Net: its
**own** `encoder_hid_proj` (4096→2048, distinct learned weights) and the 5632 add-embedding.

diffusers' `ControlNetModel.forward` does **not** apply `encoder_hid_proj` (unlike the U-Net) — the
Kolors ControlNet pipeline projects the context externally with the controlnet's own weights, then
feeds 2048-d. The Rust `ControlNet::forward` folds that exact projection inside (same weights →
numerically identical), so this golden projects externally to produce the reference.

Conditioning (`context` 4096 + `pooled`) is **reused from `kolors_t2i_golden.safetensors`** so this
dump doesn't reload the 12.5 GB ChatGLM3 encoder — only the ~2.5 GB ControlNet (f32). Dumps the
single-forward down (9) + mid residuals at `conditioning_scale=1.0` (the tight component gate), plus
the fixed `latents` / `control_image` / `context` / `pooled` / `time_ids` the Rust test replays.

Run: ~/repos/mflux/.venv-0312/bin/python tools/dump_kolors_controlnet_golden.py
Output (gitignored): tools/golden/kolors_controlnet_golden.safetensors
"""

import glob
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from _paths import fixture, hf_hub_cache

from diffusers import ControlNetModel

H = W = 512
TIMESTEP = 999.0


def cn_snapshot() -> Path:
    base = hf_hub_cache() / "models--Kwai-Kolors--Kolors-ControlNet-Pose" / "snapshots"
    snaps = sorted(glob.glob(str(base / "*")))
    if not snaps:
        raise SystemExit("Kolors-ControlNet-Pose snapshot not found in HF cache")
    return Path(snaps[-1])


def nhwc(t):  # [B,C,H,W] → [B,H,W,C]
    return mx.array(t.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.float32))


def arr(t):
    return mx.array(t.detach().cpu().numpy().astype(np.float32))


def make_control_pixels() -> np.ndarray:
    """A deterministic 512² 'pose-like' RGB control pattern (uint8): a few bright bars on black."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[100:120, 80:430] = 255  # shoulders bar
    img[120:360, 240:260] = np.array([255, 80, 80], dtype=np.uint8)  # spine
    img[350:370, 150:360] = np.array([80, 255, 80], dtype=np.uint8)  # hips
    for (cy, cx) in [(110, 90), (110, 420), (360, 160), (360, 350)]:  # joints
        img[cy - 8 : cy + 8, cx - 8 : cx + 8] = np.array([80, 80, 255], dtype=np.uint8)
    return img


@torch.no_grad()
def main():
    t2i = fixture("tools/golden/kolors_t2i_golden.safetensors")
    if not Path(t2i).exists():
        raise SystemExit("run dump_kolors_t2i_golden.py first (this reuses its conditioning)")
    cond = mx.load(t2i)
    ctx = torch.tensor(np.array(cond["pos_context"], copy=False))  # [1,256,4096]
    pooled = torch.tensor(np.array(cond["pos_pooled"], copy=False))  # [1,4096]

    cn = ControlNetModel.from_pretrained(cn_snapshot(), torch_dtype=torch.float32)

    pixels = make_control_pixels()
    control = torch.tensor(pixels.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    g = torch.Generator(device="cpu").manual_seed(0)
    latents = torch.randn(1, 4, H // 8, W // 8, generator=g, dtype=torch.float32)
    time_ids = torch.tensor([[float(H), float(W), 0.0, 0.0, float(H), float(W)]])

    ctx2048 = cn.encoder_hid_proj(ctx)  # external projection (the Kolors CN pipeline convention)
    down, mid = cn(
        latents,
        TIMESTEP,
        encoder_hidden_states=ctx2048,
        controlnet_cond=control,
        added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        conditioning_scale=1.0,
        return_dict=False,
    )
    print("down residuals:", [tuple(d.shape) for d in down])
    print("mid:", tuple(mid.shape))

    tensors = {
        "latents": nhwc(latents),
        "control_image": nhwc(control),  # [1,H,W,3] in [0,1]
        "context": arr(ctx),  # 4096 (Rust projects internally with the CN's own encoder_hid_proj)
        "pooled": arr(pooled),
        "time_ids": arr(time_ids),
        "mid": nhwc(mid),
    }
    for i, d in enumerate(down):
        tensors[f"down{i}"] = nhwc(d)
    mx.eval(list(tensors.values()))
    meta = {"h": str(H), "w": str(W), "timestep": str(TIMESTEP), "num_down": str(len(down))}
    out_path = fixture("tools/golden/kolors_controlnet_golden.safetensors")
    mx.save_safetensors(out_path, tensors, metadata=meta)
    print(f"wrote {out_path} ({len(down)} down + mid residuals)")


if __name__ == "__main__":
    main()
