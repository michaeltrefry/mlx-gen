"""SDXL U-Net single-forward golden — reference for mlx-gen-sdxl S3 (sc-2400).

Runs the EXACT vendored Apple `UNetModel` (`_vendor/mlx_sd/unet.py`) in **f32** for one forward
(fixed latents, timestep, dual-CLIP conditioning, pooled + the hardcoded `[512,512,0,0,512,512]`
micro-conditioning time_ids) and dumps every input + the predicted eps, so the Rust U-Net port can
be validated to tight tolerance in isolation.

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_unet_golden.py
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

# If only the fp16 diffusers variant is cached (this machine), point `_MODELS` at the `.fp16.`
# files so the reference loads offline. With float16=False they are upcast to f32 — identical
# upcast weights on the Rust side (its f32 loader falls back to the fp16 file too) — which isolates
# any cross-version kernel difference from a weight difference (sc-2721).
if os.environ.get("SDXL_FP16_FILES", "1") == "1":
    _m = _mio._MODELS[REPO]
    _m["unet"] = "unet/diffusion_pytorch_model.fp16.safetensors"
    _m["text_encoder"] = "text_encoder/model.fp16.safetensors"
    _m["text_encoder_2"] = "text_encoder_2/model.fp16.safetensors"
    _m["vae"] = "vae/diffusion_pytorch_model.fp16.safetensors"
PROMPT = os.environ.get("SDXL_PROMPT", "a red fox in a forest")
H = int(os.environ.get("SDXL_LATENT_H", "64"))  # 64 -> 512px image
W = int(os.environ.get("SDXL_LATENT_W", "64"))
TIMESTEP = float(os.environ.get("SDXL_TIMESTEP", "999.0"))

sd = StableDiffusionXL(REPO, float16=False)  # f32 for a tight stage gate
sd.ensure_models_are_loaded()

# Dual-CLIP conditioning (CFG off -> batch 1), exactly as the generate path builds it.
conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=0.0, negative_text="")
time_ids = mx.array([[512, 512, 0, 0, 512, 512.0]])

mx.random.seed(0)
latents = mx.random.normal((1, H, W, 4)).astype(mx.float32)
t = mx.broadcast_to(mx.array(TIMESTEP), [1])

eps = sd.unet(latents, t, encoder_x=conditioning, text_time=(pooled, time_ids))
mx.eval(eps, conditioning, pooled)

tensors = {
    "latents": latents.astype(mx.float32),
    "conditioning": conditioning.astype(mx.float32),
    "pooled": pooled.astype(mx.float32),
    "time_ids": time_ids.astype(mx.float32),
    "eps": eps.astype(mx.float32),
}
meta = {"prompt": PROMPT, "timestep": str(TIMESTEP), "h": str(H), "w": str(W)}
out = os.path.join(_GOLDEN_DIR, "sdxl_unet_golden.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"wrote {out}")
print(f"  latents {tuple(latents.shape)}, conditioning {tuple(conditioning.shape)}, eps {tuple(eps.shape)}")
print(f"  timestep={TIMESTEP}, eps mean|.|={float(mx.mean(mx.abs(eps))):.5f}")
