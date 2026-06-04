"""SDXL U-Net single-forward golden in **fp16** — the production dtype (sc-2721).

The production SceneWorks reference runs `StableDiffusionXL(float16=True)`: the U-Net + both CLIP
text encoders are fp16 (the VAE stays f32). This dumps one U-Net forward at float16=True so the Rust
fp16 U-Net port can be checked for cross-version byte-parity (mlx-gen runs the NAX MLX 0.31.2 build;
this golden is produced on the mflux pip wheel). Inputs are saved at their native compute dtype
(f16 latents/conditioning/pooled, f32 time_ids) so the comparison is fp16-vs-fp16, not f32-vs-fp16.

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_unet_golden_fp16.py
"""

import os
import sys

import mlx.core as mx

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

VENDOR_PARENT = os.environ.get(
    "SDXL_VENDOR_PARENT",
    "/Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor",
)
sys.path.insert(0, VENDOR_PARENT)
from mlx_sd import StableDiffusionXL  # noqa: E402
import mlx_sd.model_io as _mio  # noqa: E402

REPO = "stabilityai/stable-diffusion-xl-base-1.0"

# This machine's HF cache holds only the **fp16** diffusers variant of each component
# (`*.fp16.safetensors`); the f32 masters aren't downloaded. The fp16 file is exactly
# `astype(f16)` of the f32 master, so for float16=True (UNet/TE cast to f16) it's the identical
# weight. Point the vendored `_MODELS` at the cached fp16 files so the reference loads offline.
# (The VAE is unused by this single U-Net forward; pointing it at fp16 just lets __init__ succeed.)
_m = _mio._MODELS[REPO]
_m["unet"] = "unet/diffusion_pytorch_model.fp16.safetensors"
_m["text_encoder"] = "text_encoder/model.fp16.safetensors"
_m["text_encoder_2"] = "text_encoder_2/model.fp16.safetensors"
_m["vae"] = "vae/diffusion_pytorch_model.fp16.safetensors"
PROMPT = os.environ.get("SDXL_PROMPT", "a red fox in a forest")
H = int(os.environ.get("SDXL_LATENT_H", "64"))  # 64 -> 512px image
W = int(os.environ.get("SDXL_LATENT_W", "64"))
TIMESTEP = float(os.environ.get("SDXL_TIMESTEP", "999.0"))

sd = StableDiffusionXL(REPO, float16=True)  # production fp16 path
sd.ensure_models_are_loaded()
print("unet dtype:", sd.unet.conv_in.weight.dtype)

# Dual-CLIP conditioning (CFG off -> batch 1), exactly as the generate path builds it. With
# float16=True these come back fp16.
conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=0.0, negative_text="")
time_ids = mx.array([[512, 512, 0, 0, 512, 512.0]])  # f32, as the vendored generate builds it

mx.random.seed(0)
latents = mx.random.normal((1, H, W, 4)).astype(sd.dtype)  # fp16 prior-shaped latents
t = mx.broadcast_to(mx.array(TIMESTEP), [1])

eps = sd.unet(latents, t, encoder_x=conditioning, text_time=(pooled, time_ids))
mx.eval(eps, conditioning, pooled)
print("eps dtype:", eps.dtype, "conditioning dtype:", conditioning.dtype)

# Save at native dtype so the Rust check is fp16-vs-fp16 (byte parity), not within-tolerance f32.
tensors = {
    "latents": latents,          # f16
    "conditioning": conditioning,  # f16
    "pooled": pooled,            # f16
    "time_ids": time_ids,        # f32
    "eps": eps,                  # f16
}
meta = {"prompt": PROMPT, "timestep": str(TIMESTEP), "h": str(H), "w": str(W), "float16": "1"}
out = os.path.join(_GOLDEN_DIR, "sdxl_unet_golden_fp16.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"wrote {out}")
print(f"  latents {tuple(latents.shape)}, conditioning {tuple(conditioning.shape)}, eps {tuple(eps.shape)}")
print(f"  timestep={TIMESTEP}, eps mean|.|={float(mx.mean(mx.abs(eps.astype(mx.float32)))):.5f}")
