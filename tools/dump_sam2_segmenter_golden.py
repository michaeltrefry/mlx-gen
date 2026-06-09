"""sc-3706 — dump the SAM2 box-prompt segmenter (decoder) parity golden.

Runs the MLX-native reference `Sam2ImageSegmenter` (`avbiswas/sam2-mlx`, the impl this crate ports)
on a fixed input + box and bundles, into one gitignored golden safetensors:

  * the full segmenter weights the Rust port reads (`trunk.` / `neck.` / `sam_prompt_encoder.` /
    `sam_mask_decoder.` / `no_mem_embed`),
  * `enc_in` — the NCHW [1,3,1024,1024] input,
  * `box_1024` — the box corners (x1,y1,x2,y2) in 1024-space,
  * `ref_low_res_all` [3,256,256] / `ref_ious` [3] / `ref_best_idx` / `ref_low_res_best` [256,256].

The Rust `tests/segmenter_parity.rs` (`#[ignore]`, macOS) builds `Sam2Segmenter`, runs the box
prompt, and asserts the best low-res mask logits + selected IoU match the reference.

Run (MLX venv + the converted weights, both already present):
  PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python \
      tools/dump_sam2_segmenter_golden.py --size large
"""

from __future__ import annotations

import argparse
import glob
import os

import mlx.core as mx
import numpy as np

from mlx_sam.config import (
    SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER,
    SAM2_1_HIERA_LARGE_IMAGE_ENCODER,
    SAM2_1_HIERA_SMALL_IMAGE_ENCODER,
    SAM2_1_HIERA_TINY_IMAGE_ENCODER,
)
from mlx_sam.models.segmenter import Sam2ImageSegmenter

SIZE_CFG = {
    "tiny": SAM2_1_HIERA_TINY_IMAGE_ENCODER,
    "small": SAM2_1_HIERA_SMALL_IMAGE_ENCODER,
    "base_plus": SAM2_1_HIERA_BASE_PLUS_IMAGE_ENCODER,
    "large": SAM2_1_HIERA_LARGE_IMAGE_ENCODER,
}
HF_REPO = {
    "tiny": "avbiswas/sam2.1-hiera-tiny-mlx",
    "small": "avbiswas/sam2.1-hiera-small-mlx",
    "base_plus": "avbiswas/sam2.1-hiera-base-plus-mlx",
    "large": "avbiswas/sam2.1-hiera-large-mlx",
}
KEEP_PREFIXES = ("trunk.", "neck.", "sam_prompt_encoder.", "sam_mask_decoder.")


def resolve_checkpoint(size: str) -> str:
    from huggingface_hub import snapshot_download

    snap = snapshot_download(HF_REPO[size], allow_patterns=["*.safetensors"])
    files = glob.glob(os.path.join(snap, "*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors in {snap}")
    return files[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZE_CFG), default="large")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "golden"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = SIZE_CFG[args.size]
    ckpt = resolve_checkpoint(args.size)
    print(f"[load] {ckpt}")
    weights = mx.load(ckpt)

    seg = Sam2ImageSegmenter(config=cfg)
    seg.load_weights(list(weights.items()), strict=True)
    mx.eval(seg.parameters())

    rng = np.random.RandomState(args.seed)
    enc_in = mx.array(rng.standard_normal((1, 3, 1024, 1024)).astype(np.float32))

    # Box = two corners (x1,y1,x2,y2) in 1024-space; the prompt encoder labels them 2/3 + pads.
    box = [200.0, 150.0, 820.0, 870.0]
    points = mx.array([[[box[0], box[1]], [box[2], box[3]]]], dtype=mx.float32)  # [1,2,2]
    labels = mx.array([[2, 3]], dtype=mx.int32)  # [1,2]

    encoded = seg.encode_image(enc_in)
    out = seg.predict_from_encoded(encoded, points, labels, multimask_output=True)
    low = out["low_res_masks"]  # [1,3,256,256]
    ious = out["ious"]  # [1,3]
    best = int(mx.argmax(ious[0]).item())
    print(f"[forward] low_res={low.shape} ious={np.asarray(ious[0]).round(4).tolist()} best={best}")

    golden = {k: v for k, v in weights.items() if k.startswith(KEEP_PREFIXES) or k == "no_mem_embed"}
    golden["enc_in"] = enc_in
    golden["box_1024"] = mx.array(np.asarray(box, dtype=np.float32))
    golden["ref_low_res_all"] = low[0]
    golden["ref_low_res_best"] = low[0, best]
    golden["ref_ious"] = ious[0]
    golden["ref_best_idx"] = mx.array(np.asarray([best], dtype=np.int32))
    golden = {k: mx.array(v).astype(mx.float32) if v.dtype != mx.int32 else v for k, v in golden.items()}
    mx.eval(list(golden.values()))

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"sam2_segmenter_golden_{args.size}.safetensors")
    mx.save_safetensors(out_path, golden, metadata={"format": "mlx", "size": args.size})
    print(f"[written] {out_path} ({len(golden)} tensors)")


if __name__ == "__main__":
    main()
