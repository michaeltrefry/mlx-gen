#!/usr/bin/env python3
"""Convert facexlib's BiSeNet face-parsing net (PuLID's `init_parsing_model('bisenet')`,
`parsing_bisenet.pth`) -> safetensors for the native MLX port (sc-3084), and dump a golden
parse + the PuLID `face_features_image` for the parity test.

Architecture (facexlib `parsing/bisenet.py` + `resnet.py`, num_class=19):
  ResNet18 backbone (conv1 7x7 s2 + 4 stages of 2 BasicBlocks; stages 2/3/4 block0 stride-2 +
  1x1 downsample) -> feat8/feat16/feat32. ContextPath: ARM(256->128)+ARM(512->128) (each
  ConvBNReLU -> conv_atten 1x1 -> bn_atten -> sigmoid -> mul), global-avg conv_avg, nearest
  upsamples, conv_head16/32. FeatureFusionModule(256->256, channel-concat sp+cp, SE-style atten).
  conv_out head (ConvBNReLU 256->256 -> conv_out 1x1 -> 19 logits @ 64²) -> **bilinear upsample
  to 512² with align_corners=True** -> argmax. (conv_out16/conv_out32 are training-only aux heads;
  PuLID consumes `parse(x)[0]` only, so they are not converted.)

Every conv is bias-less with a following BN; we fold BN into the conv (W' = W·γ/√(var+eps),
b' = β - μ·γ/√(var+eps), eps=1e-5) -> biased conv. conv weights OIHW -> MLX OHWI. fp32.

Consumer contract (PuLID `pipeline_flux.py:167-177`): input = align_face/255 (RGB, [0,1]);
parse(normalize(input, ImageNet mean/std)) -> argmax(19); bg labels [0,16,18,7,8,9,14,15] -> white,
else gray(input) -> face_features_image.

Golden input = `align512.0` from the sc-3083 face-align golden (the real pipeline 512² crop).

Outputs under tools/golden/ (gitignored):
  bisenet_parsing.safetensors   -- folded weights (resnet/arm/ffm/conv_out), OHWI
  bisenet_goldens.safetensors   -- input [512,512,3] RGB int32, torch mask [512,512] int32,
                                   logits [512,512,19] f32 (pre-argmax, post-upsample),
                                   face_features_image [512,512,3] f32 (the consumer output)

Run with the bisenet tool venv (torch + facexlib + safetensors + numpy):
  ~/.bisenet-spike/venv/bin/python tools/convert_bisenet.py
"""
import os
import sys

import numpy as np
import torch

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "golden")
EPS = 1e-5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BG_LABELS = [0, 16, 18, 7, 8, 9, 14, 15]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    from facexlib.parsing import init_parsing_model
    from safetensors.numpy import load_file, save_file

    model = init_parsing_model("bisenet", device="cpu")
    model.eval()
    sd = {k: v.numpy() for k, v in model.state_dict().items()}

    def fold(conv_key, bn_prefix):
        """Fold BN(bn_prefix) into bias-less conv(conv_key.weight) -> (OHWI weight, bias)."""
        w = sd[f"{conv_key}.weight"].astype(np.float64)  # OIHW
        g = sd[f"{bn_prefix}.weight"].astype(np.float64)
        b = sd[f"{bn_prefix}.bias"].astype(np.float64)
        m = sd[f"{bn_prefix}.running_mean"].astype(np.float64)
        v = sd[f"{bn_prefix}.running_var"].astype(np.float64)
        scale = g / np.sqrt(v + EPS)
        wf = w * scale[:, None, None, None]
        bf = b - m * scale
        return (
            np.ascontiguousarray(np.transpose(wf, (0, 2, 3, 1)).astype(np.float32)),  # OHWI
            np.ascontiguousarray(bf.astype(np.float32)),
        )

    def plain(conv_key):
        """A bias-less conv with no BN (ffm.conv1/conv2, conv_out.conv_out) -> OHWI weight."""
        w = sd[f"{conv_key}.weight"]  # OIHW
        return np.ascontiguousarray(np.transpose(w, (0, 2, 3, 1)).astype(np.float32))

    out = {}

    def put_folded(dst, conv_key, bn_prefix):
        w, b = fold(conv_key, bn_prefix)
        out[f"{dst}.weight"] = w
        out[f"{dst}.bias"] = b

    # --- ResNet18 backbone
    put_folded("resnet.conv1", "cp.resnet.conv1", "cp.resnet.bn1")
    stages = {"layer1": (64, 64, 1), "layer2": (64, 128, 2), "layer3": (128, 256, 2), "layer4": (256, 512, 2)}
    for ln, (_ci, _co, _stride) in stages.items():
        for b in range(2):
            p = f"cp.resnet.{ln}.{b}"
            d = f"resnet.{ln}.{b}"
            put_folded(f"{d}.conv1", f"{p}.conv1", f"{p}.bn1")
            put_folded(f"{d}.conv2", f"{p}.conv2", f"{p}.bn2")
            if b == 0 and ln != "layer1":
                put_folded(f"{d}.downsample", f"{p}.downsample.0", f"{p}.downsample.1")

    # --- ContextPath
    for arm in ("arm16", "arm32"):
        put_folded(f"{arm}.conv", f"cp.{arm}.conv.conv", f"cp.{arm}.conv.bn")
        put_folded(f"{arm}.conv_atten", f"cp.{arm}.conv_atten", f"cp.{arm}.bn_atten")
    put_folded("conv_head32", "cp.conv_head32.conv", "cp.conv_head32.bn")
    put_folded("conv_head16", "cp.conv_head16.conv", "cp.conv_head16.bn")
    put_folded("conv_avg", "cp.conv_avg.conv", "cp.conv_avg.bn")

    # --- FeatureFusionModule
    put_folded("ffm.convblk", "ffm.convblk.conv", "ffm.convblk.bn")
    out["ffm.conv1.weight"] = plain("ffm.conv1")
    out["ffm.conv2.weight"] = plain("ffm.conv2")

    # --- main output head (BiSeNetOutput.conv is itself a ConvBNReLU; aux conv_out16/32 omitted)
    put_folded("conv_out.conv", "conv_out.conv.conv", "conv_out.conv.bn")
    out["conv_out.conv_out.weight"] = plain("conv_out.conv_out")

    print(f"converted {len(out)} tensors")
    wpath = os.path.join(OUT_DIR, "bisenet_parsing.safetensors")
    save_file(out, wpath)
    print("wrote", wpath)

    # --- golden: parse the real pipeline 512² crop (align512.0) with torch
    fa_path = os.path.join(OUT_DIR, "face_align_goldens.safetensors")
    fa = load_file(fa_path)
    rgb = fa["align512.0"].astype(np.uint8)  # [512,512,3] RGB
    input01 = rgb.astype(np.float32) / 255.0
    norm = (input01 - MEAN) / STD  # [512,512,3]
    nchw = torch.from_numpy(np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[None]))  # [1,3,512,512]
    with torch.no_grad():
        logits = model(nchw)[0]  # [1,19,512,512]
    mask = logits.argmax(dim=1)[0].numpy().astype(np.int32)  # [512,512]

    # consumer: face_features_image = where(bg, white, gray(input01))
    bg = np.isin(mask, BG_LABELS)  # [512,512]
    gray = (0.299 * input01[..., 0] + 0.587 * input01[..., 1] + 0.114 * input01[..., 2])  # [512,512]
    gray3 = np.repeat(gray[..., None], 3, axis=2)  # [512,512,3]
    ffi = np.where(bg[..., None], np.ones_like(gray3), gray3).astype(np.float32)

    g = {
        "input": np.ascontiguousarray(rgb.astype(np.int32)),  # [512,512,3] RGB
        "mask": np.ascontiguousarray(mask),  # [512,512]
        "logits": np.ascontiguousarray(np.transpose(logits[0].numpy(), (1, 2, 0)).astype(np.float32)),  # [512,512,19]
        "face_features_image": np.ascontiguousarray(ffi),  # [512,512,3]
    }
    gpath = os.path.join(OUT_DIR, "bisenet_goldens.safetensors")
    save_file(g, gpath)
    classes, counts = np.unique(mask, return_counts=True)
    print("wrote", gpath, "| mask classes:", dict(zip(classes.tolist(), counts.tolist())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
