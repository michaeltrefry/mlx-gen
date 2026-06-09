"""sc-3707 — convert an official Meta SAM2.1 `.pt` checkpoint to the `mlx-gen-sam2` MLX safetensors
layout (zero runtime Python; this is a one-time offline provisioning tool).

Produces exactly the key layout the Rust crate loads (`trunk.` / `neck.` / `sam_prompt_encoder.` /
`sam_mask_decoder.` / memory / obj-ptr) so we own the converted artifact end-to-end from the
canonical Meta weights — no dependency on a third party's converted upload. The mapping is a
self-contained vendor of `avbiswas/sam2-mlx`'s `mlx_sam/convert.py` (Apache-2.0), with the per-size
config inlined.

Run (torch venv; the .pt is downloaded on demand):
  ~/mlx-flux-venv/bin/python tools/convert_sam2_to_mlx.py --hf-id facebook/sam2.1-hiera-large \\
      --output tools/golden/sam2.1_hiera_large.safetensors

Then host the output under `SceneWorks/sam2-mlx` (see the story / README).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

# Per-size Hiera config — only the two fields the conversion needs: the block count
# (`sum(stages)`) and the (square) learned position-embedding grid.
_STAGES = {
    "tiny": (1, 2, 7, 2),
    "small": (1, 2, 11, 2),
    "base_plus": (2, 3, 16, 3),
    "large": (2, 6, 36, 4),
}
HF_FILENAMES = {
    "facebook/sam2.1-hiera-tiny": ("tiny", "sam2.1_hiera_tiny.pt"),
    "facebook/sam2.1-hiera-small": ("small", "sam2.1_hiera_small.pt"),
    "facebook/sam2.1-hiera-base-plus": ("base_plus", "sam2.1_hiera_base_plus.pt"),
    "facebook/sam2.1-hiera-large": ("large", "sam2.1_hiera_large.pt"),
}


def config_for_size(size: str) -> SimpleNamespace:
    return SimpleNamespace(hiera=SimpleNamespace(stages=_STAGES[size], pos_embed_hw=(256, 256)))


def _t(sd: dict, key: str) -> np.ndarray:
    return sd[key].detach().cpu().numpy()


def _conv_w(sd: dict, key: str) -> np.ndarray:
    # Torch Conv2d OIHW → MLX Conv2d OHWI.
    return np.transpose(_t(sd, key), (0, 2, 3, 1))


def _conv_transpose_w(sd: dict, key: str) -> np.ndarray:
    # Torch ConvTranspose2d IOHW → MLX OHWI.
    return np.transpose(_t(sd, key), (1, 2, 3, 0))


def _tiled_window_pos(window: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    repeats = [1, 1, math.ceil(target_hw[0] / window.shape[2]), math.ceil(target_hw[1] / window.shape[3])]
    tiled = np.tile(window, repeats)
    return tiled[:, :, : target_hw[0], : target_hw[1]]


def convert_state_dict(sd: dict, size: str) -> dict[str, np.ndarray]:
    import torch.nn.functional as F

    config = config_for_size(size)
    w: dict[str, np.ndarray] = {}

    pe = sd["image_encoder.trunk.pos_embed"]
    pew = sd["image_encoder.trunk.pos_embed_window"]
    pe = F.interpolate(pe, size=config.hiera.pos_embed_hw, mode="bicubic")
    pe_np = pe.detach().cpu().numpy()
    pew_np = pew.detach().cpu().numpy()
    w["trunk.pos_embed_full"] = np.transpose(
        pe_np + _tiled_window_pos(pew_np, config.hiera.pos_embed_hw), (0, 2, 3, 1)
    )
    w["trunk.patch_embed.proj.weight"] = _conv_w(sd, "image_encoder.trunk.patch_embed.proj.weight")
    w["trunk.patch_embed.proj.bias"] = _t(sd, "image_encoder.trunk.patch_embed.proj.bias")

    for i in range(sum(config.hiera.stages)):
        src = f"image_encoder.trunk.blocks.{i}"
        dst = f"trunk.blocks.{i}"
        for name in (
            "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
            "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
            "mlp.layers.0.weight", "mlp.layers.0.bias", "mlp.layers.1.weight", "mlp.layers.1.bias",
        ):
            w[f"{dst}.{name}"] = _t(sd, f"{src}.{name}")
        if f"{src}.proj.weight" in sd:
            w[f"{dst}.proj.weight"] = _t(sd, f"{src}.proj.weight")
            w[f"{dst}.proj.bias"] = _t(sd, f"{src}.proj.bias")

    for i in range(4):
        w[f"neck.convs.{i}.weight"] = _conv_w(sd, f"image_encoder.neck.convs.{i}.conv.weight")
        w[f"neck.convs.{i}.bias"] = _t(sd, f"image_encoder.neck.convs.{i}.conv.bias")

    pe_src, pe_dst = "sam_prompt_encoder", "sam_prompt_encoder"
    w[f"{pe_dst}.pe_layer.positional_encoding_gaussian_matrix"] = _t(sd, f"{pe_src}.pe_layer.positional_encoding_gaussian_matrix")
    for i in range(4):
        w[f"{pe_dst}.point_embeddings.{i}.weight"] = _t(sd, f"{pe_src}.point_embeddings.{i}.weight")
    w[f"{pe_dst}.not_a_point_embed.weight"] = _t(sd, f"{pe_src}.not_a_point_embed.weight")
    w[f"{pe_dst}.no_mask_embed.weight"] = _t(sd, f"{pe_src}.no_mask_embed.weight")
    for idx, conv in ((0, True), (1, False), (3, True), (4, False), (6, True)):
        key = f"{pe_src}.mask_downscaling.{idx}"
        w[f"{pe_dst}.mask_downscaling_{idx}.weight"] = _conv_w(sd, f"{key}.weight") if conv else _t(sd, f"{key}.weight")
        w[f"{pe_dst}.mask_downscaling_{idx}.bias"] = _t(sd, f"{key}.bias")

    md_src, md_dst = "sam_mask_decoder", "sam_mask_decoder"
    w[f"{md_dst}.iou_token.weight"] = _t(sd, f"{md_src}.iou_token.weight")
    w[f"{md_dst}.mask_tokens.weight"] = _t(sd, f"{md_src}.mask_tokens.weight")
    w[f"{md_dst}.obj_score_token.weight"] = _t(sd, f"{md_src}.obj_score_token.weight")
    w[f"{md_dst}.output_upscaling_0.weight"] = _conv_transpose_w(sd, f"{md_src}.output_upscaling.0.weight")
    w[f"{md_dst}.output_upscaling_0.bias"] = _t(sd, f"{md_src}.output_upscaling.0.bias")
    w[f"{md_dst}.output_upscaling_1.weight"] = _t(sd, f"{md_src}.output_upscaling.1.weight")
    w[f"{md_dst}.output_upscaling_1.bias"] = _t(sd, f"{md_src}.output_upscaling.1.bias")
    w[f"{md_dst}.output_upscaling_3.weight"] = _conv_transpose_w(sd, f"{md_src}.output_upscaling.3.weight")
    w[f"{md_dst}.output_upscaling_3.bias"] = _t(sd, f"{md_src}.output_upscaling.3.bias")
    w[f"{md_dst}.conv_s0.weight"] = _conv_w(sd, f"{md_src}.conv_s0.weight")
    w[f"{md_dst}.conv_s0.bias"] = _t(sd, f"{md_src}.conv_s0.bias")
    w[f"{md_dst}.conv_s1.weight"] = _conv_w(sd, f"{md_src}.conv_s1.weight")
    w[f"{md_dst}.conv_s1.bias"] = _t(sd, f"{md_src}.conv_s1.bias")
    for layer in range(2):
        for attn in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                base = f"{md_src}.transformer.layers.{layer}.{attn}.{proj}"
                dst = f"{md_dst}.transformer.layers.{layer}.{attn}.{proj}"
                w[f"{dst}.weight"] = _t(sd, f"{base}.weight")
                w[f"{dst}.bias"] = _t(sd, f"{base}.bias")
        for norm in ("norm1", "norm2", "norm3", "norm4"):
            w[f"{md_dst}.transformer.layers.{layer}.{norm}.weight"] = _t(sd, f"{md_src}.transformer.layers.{layer}.{norm}.weight")
            w[f"{md_dst}.transformer.layers.{layer}.{norm}.bias"] = _t(sd, f"{md_src}.transformer.layers.{layer}.{norm}.bias")
        for mlp_layer in range(2):
            w[f"{md_dst}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.weight"] = _t(sd, f"{md_src}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.weight")
            w[f"{md_dst}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.bias"] = _t(sd, f"{md_src}.transformer.layers.{layer}.mlp.layers.{mlp_layer}.bias")
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        w[f"{md_dst}.transformer.final_attn_token_to_image.{proj}.weight"] = _t(sd, f"{md_src}.transformer.final_attn_token_to_image.{proj}.weight")
        w[f"{md_dst}.transformer.final_attn_token_to_image.{proj}.bias"] = _t(sd, f"{md_src}.transformer.final_attn_token_to_image.{proj}.bias")
    w[f"{md_dst}.transformer.norm_final_attn.weight"] = _t(sd, f"{md_src}.transformer.norm_final_attn.weight")
    w[f"{md_dst}.transformer.norm_final_attn.bias"] = _t(sd, f"{md_src}.transformer.norm_final_attn.bias")
    for i in range(4):
        for layer in range(3):
            w[f"{md_dst}.output_hypernetworks_mlps.{i}.layers.{layer}.weight"] = _t(sd, f"{md_src}.output_hypernetworks_mlps.{i}.layers.{layer}.weight")
            w[f"{md_dst}.output_hypernetworks_mlps.{i}.layers.{layer}.bias"] = _t(sd, f"{md_src}.output_hypernetworks_mlps.{i}.layers.{layer}.bias")
    for head in ("iou_prediction_head", "pred_obj_score_head"):
        for layer in range(3):
            w[f"{md_dst}.{head}.layers.{layer}.weight"] = _t(sd, f"{md_src}.{head}.layers.{layer}.weight")
            w[f"{md_dst}.{head}.layers.{layer}.bias"] = _t(sd, f"{md_src}.{head}.layers.{layer}.bias")

    # Object-pointer + memory globals (video layer — carried so the artifact is a full checkpoint).
    for k in ("no_obj_ptr", "no_mem_embed", "no_mem_pos_enc", "maskmem_tpos_enc", "no_obj_embed_spatial"):
        w[k] = _t(sd, k)
    for layer in range(3):
        w[f"obj_ptr_proj.layers.{layer}.weight"] = _t(sd, f"obj_ptr_proj.layers.{layer}.weight")
        w[f"obj_ptr_proj.layers.{layer}.bias"] = _t(sd, f"obj_ptr_proj.layers.{layer}.bias")
    w["obj_ptr_tpos_proj.weight"] = _t(sd, "obj_ptr_tpos_proj.weight")
    w["obj_ptr_tpos_proj.bias"] = _t(sd, "obj_ptr_tpos_proj.bias")

    me = "memory_encoder"
    conv_map = [(0, "conv0"), (3, "conv1"), (6, "conv2"), (9, "conv3"), (12, "conv4")]
    norm_map = [(1, "norm0"), (4, "norm1"), (7, "norm2"), (10, "norm3")]
    for idx, name in conv_map:
        w[f"{me}.mask_downsampler.{name}.weight"] = _conv_w(sd, f"{me}.mask_downsampler.encoder.{idx}.weight")
        w[f"{me}.mask_downsampler.{name}.bias"] = _t(sd, f"{me}.mask_downsampler.encoder.{idx}.bias")
    for idx, name in norm_map:
        w[f"{me}.mask_downsampler.{name}.weight"] = _t(sd, f"{me}.mask_downsampler.encoder.{idx}.weight")
        w[f"{me}.mask_downsampler.{name}.bias"] = _t(sd, f"{me}.mask_downsampler.encoder.{idx}.bias")
    w[f"{me}.pix_feat_proj.weight"] = _conv_w(sd, f"{me}.pix_feat_proj.weight")
    w[f"{me}.pix_feat_proj.bias"] = _t(sd, f"{me}.pix_feat_proj.bias")
    w[f"{me}.out_proj.weight"] = _conv_w(sd, f"{me}.out_proj.weight")
    w[f"{me}.out_proj.bias"] = _t(sd, f"{me}.out_proj.bias")
    for i in range(2):
        src = f"{me}.fuser.layers.{i}"
        dst = f"{me}.fuser.{i}"
        w[f"{dst}.gamma"] = _t(sd, f"{src}.gamma")
        w[f"{dst}.dwconv.weight"] = _conv_w(sd, f"{src}.dwconv.weight")
        w[f"{dst}.dwconv.bias"] = _t(sd, f"{src}.dwconv.bias")
        w[f"{dst}.norm.weight"] = _t(sd, f"{src}.norm.weight")
        w[f"{dst}.norm.bias"] = _t(sd, f"{src}.norm.bias")
        w[f"{dst}.pwconv1.weight"] = _t(sd, f"{src}.pwconv1.weight")
        w[f"{dst}.pwconv1.bias"] = _t(sd, f"{src}.pwconv1.bias")
        w[f"{dst}.pwconv2.weight"] = _t(sd, f"{src}.pwconv2.weight")
        w[f"{dst}.pwconv2.bias"] = _t(sd, f"{src}.pwconv2.bias")

    ma = "memory_attention"
    for i in range(4):
        for attn in ("self_attn", "cross_attn_image"):
            for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
                w[f"{ma}.layers.{i}.{attn}.{proj}.weight"] = _t(sd, f"{ma}.layers.{i}.{attn}.{proj}.weight")
                w[f"{ma}.layers.{i}.{attn}.{proj}.bias"] = _t(sd, f"{ma}.layers.{i}.{attn}.{proj}.bias")
        for lin in ("linear1", "linear2"):
            w[f"{ma}.layers.{i}.{lin}.weight"] = _t(sd, f"{ma}.layers.{i}.{lin}.weight")
            w[f"{ma}.layers.{i}.{lin}.bias"] = _t(sd, f"{ma}.layers.{i}.{lin}.bias")
        for norm in ("norm1", "norm2", "norm3"):
            w[f"{ma}.layers.{i}.{norm}.weight"] = _t(sd, f"{ma}.layers.{i}.{norm}.weight")
            w[f"{ma}.layers.{i}.{norm}.bias"] = _t(sd, f"{ma}.layers.{i}.{norm}.bias")
    w[f"{ma}.norm.weight"] = _t(sd, f"{ma}.norm.weight")
    w[f"{ma}.norm.bias"] = _t(sd, f"{ma}.norm.bias")
    return w


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a Meta SAM2.1 .pt to mlx-gen-sam2 safetensors.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf-id", choices=list(HF_FILENAMES), help="official Meta repo id to download + convert")
    src.add_argument("--checkpoint", type=Path, help="local .pt path")
    ap.add_argument("--size", choices=list(_STAGES), help="required with --checkpoint")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import torch

    if args.hf_id:
        from huggingface_hub import hf_hub_download

        size, fname = HF_FILENAMES[args.hf_id]
        ckpt = Path(hf_hub_download(args.hf_id, fname))
    else:
        ckpt = args.checkpoint
        size = args.size or next((s for s in _STAGES if s in ckpt.stem), None)
        if size is None:
            raise SystemExit("--size is required for a local --checkpoint whose name has no size hint")

    print(f"[load] {ckpt} (size={size})")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    weights = convert_state_dict(sd, size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output),
        {k: mx.array(v.astype(np.float32)) for k, v in weights.items()},
        metadata={"format": "mlx", "model_id": args.hf_id or f"sam2.1-hiera-{size}", "source": "meta-official-pt"},
    )
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"[written] {args.output} ({len(weights)} tensors)\n[sha256]  {sha}")


if __name__ == "__main__":
    main()
