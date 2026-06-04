#!/usr/bin/env python3
"""Convert the original `Wan2.2_VAE.pth` → the sanitized channels-last `vae.safetensors` the 5B
snapshot needs (sc-2680). Mirrors `convert_wan.py`'s VAE branch: load the PyTorch checkpoint, run
`sanitize_wan22_vae_weights(include_encoder=True)` (the encoder is needed for TI2V image encode),
save f32 (the reference runs the VAE in float32).

The 5B snapshot (`…/data/models/mlx/wan_2_2_ti2v_5b/`) ships the DiT + T5 + tokenizer but not the
VAE; this fills it in. Both the Rust [`Wan22Vae`] and the reference `Wan22VAEDecoder`/`Encoder` load
the result identically (it is the layout both modules' attribute names expect).

Download the source once (≈2.8 GB):
    python -c "from huggingface_hub import hf_hub_download as d; print(d('Wan-AI/Wan2.2-TI2V-5B','Wan2.2_VAE.pth'))"

Run with the SceneWorks venv (needs torch + mlx_video):
    "$HOME/Library/Application Support/SceneWorks/python/venv/bin/python" \
        mlx-gen-wan/tools/convert_vae22.py [--vae-pth PATH] [--out DIR]
"""
import argparse
import os

import mlx.core as mx

from mlx_video.convert_wan import load_torch_weights
from mlx_video.models.wan.vae22 import sanitize_wan22_vae_weights

DEFAULT_OUT = os.path.expanduser(
    "~/Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--vae-pth",
        default=None,
        help="Path to Wan2.2_VAE.pth (default: fetch from HF cache via hf_hub_download)",
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="5B snapshot dir to write vae.safetensors into")
    args = ap.parse_args()

    vae_pth = args.vae_pth
    if vae_pth is None:
        from huggingface_hub import hf_hub_download

        vae_pth = hf_hub_download("Wan-AI/Wan2.2-TI2V-5B", "Wan2.2_VAE.pth")
    print(f"Loading {vae_pth} ...")
    weights = load_torch_weights(str(vae_pth))
    print(f"  {len(weights)} raw tensors")

    weights = sanitize_wan22_vae_weights(weights, include_encoder=True)
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    print(f"  {len(weights)} sanitized tensors (f32, channels-last)")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "vae.safetensors")
    mx.save_safetensors(out_path, weights)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
