"""sc-5136: golden for the Bernini planner's VAE preprocessing (`VAEVideoTransform`).

The exactly-matchable pieces of `data_utils.MaxLongEdgeMinShortEdgeResize` (+ `make_divisible` /
`_apply_scale`): the target `(new_w, new_h)` for the VAE branch — long edge ≤ `max_size`, short edge
≥ `min_size`, snapped to `stride` via Python banker's `round` — and the normalize-to-[-1,1] (mean=std=
0.5). The resize *interpolation* (PIL bicubic, antialias) is excluded (the Rust port uses the `image`
crate; dims are exact, pixels differ slightly — same divergence as the ViT processor).

Resize math copied **verbatim** from `_vendor/bernini/bernini/data_utils.py`.

Run:
  ~/Repos/mflux/.venv/bin/python tools/dump_bernini_vae_preprocess_golden.py
Fixture -> mlx-gen-bernini/tests/fixtures/vae_preprocess_golden.safetensors
"""

from __future__ import annotations

import os

import numpy as np
import torch
from safetensors.torch import save_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(REPO_ROOT, "mlx-gen-bernini", "tests", "fixtures", "vae_preprocess_golden.safetensors")

MAX_SIZE = 624
MIN_SIZE = 1
STRIDE = 16


# ===== verbatim reference: make_divisible / _apply_scale / MaxLongEdgeMinShortEdgeResize (dims) =====
def make_divisible(value, stride):
    return max(stride, int(round(value / stride) * stride))


def _apply_scale(width, height, scale, stride):
    new_width = make_divisible(round(width * scale), stride)
    new_height = make_divisible(round(height * scale), stride)
    return new_width, new_height


def resize_dims(width, height, max_size=MAX_SIZE, min_size=MIN_SIZE, stride=STRIDE):
    scale = min(max_size / max(width, height), 1.0)
    scale = max(scale, min_size / min(width, height))
    new_width, new_height = _apply_scale(width, height, scale, stride)
    if max(new_width, new_height) > max_size:
        scale = max_size / max(new_width, new_height)
        new_width, new_height = _apply_scale(new_width, new_height, scale, stride)
    return new_width, new_height


# (width, height) cases: small (no downscale), wide, tall, huge, exact-multiple, banker's-round.
CASES = [
    (320, 240),
    (1920, 1080),
    (480, 1280),
    (4000, 4000),
    (624, 624),
    (24, 24),     # tiny: scale-up clamped by min_size=1 -> stays small, snapped to stride
    (200, 200),
]


def main() -> None:
    out = {}
    inp = torch.tensor(CASES, dtype=torch.int32)  # (w, h)
    res = torch.tensor([list(resize_dims(w, h)) for (w, h) in CASES], dtype=torch.int32)  # (new_w, new_h)
    out["resize.in_wh"] = inp.contiguous()
    out["resize.out_wh"] = res.contiguous()

    # normalize: a fixed uint8 RGB image already at a stride-multiple size -> [-1,1] tensor [C,H,W].
    rng = np.random.RandomState(1)
    H, W = 32, 48
    img = rng.randint(0, 256, size=(H, W, 3), dtype=np.uint8)
    # ToTensor (HWC u8 -> CHW float /255) then Normalize(0.5,0.5).
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    t = (t - 0.5) / 0.5
    out["norm.image_hwc_u8"] = torch.from_numpy(img.astype(np.int32)).contiguous()
    out["norm.chw"] = t.contiguous()

    meta = {
        "max_size": str(MAX_SIZE), "min_size": str(MIN_SIZE), "stride": str(STRIDE),
        "norm_h": str(H), "norm_w": str(W),
    }
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    save_file(out, FIXTURE, metadata=meta)
    print(f"wrote {FIXTURE}  ({len(out)} tensors)")
    print(f"  resize: {CASES} -> {res.tolist()}")
    print(f"  norm: img {(H, W, 3)} -> chw {tuple(t.shape)} range [{t.min():.3f},{t.max():.3f}]")


if __name__ == "__main__":
    main()
