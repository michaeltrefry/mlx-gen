"""SDXL end-to-end golden — reference for mlx-gen-sdxl S5 (sc-2400).

Replicates the vendored `StableDiffusionXL.generate_latents` denoise by hand so every stage is
captured: the dual-CLIP conditioning, the seeded prior (first RNG draw), the per-step ancestral
trajectory's final latents, and the rendered RGB8 image. Lets the Rust port be validated both
end-to-end (px>8 vs the image) and stage-isolated (feed the prior, check final latents → isolates
the sampler/CFG/RNG from the prior draw).

`FLOAT16=1` dumps the production fp16 path; default is f32 (tight stage gate). Run from the mflux
venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_golden.py
  FLOAT16=1 /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_golden.py
"""

import os
import sys

import mlx.core as mx
import numpy as np

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

# This machine's HF cache holds only the fp16 diffusers variant (`*.fp16.safetensors`). The fp16
# file is exactly `astype(f16)` of the f32 master, so for FLOAT16=1 (U-Net/TE cast to f16) it's the
# identical weight; for FLOAT16=0 it's upcast to f32 (matching the Rust f32 loader's same fallback).
# Point `_MODELS` at the cached fp16 files so the reference loads offline. SDXL_FP16_FILES=0 opts out
# (a snapshot with the f32 masters present).
if os.environ.get("SDXL_FP16_FILES", "1") == "1":
    _m = _mio._MODELS[REPO]
    _m["unet"] = "unet/diffusion_pytorch_model.fp16.safetensors"
    _m["text_encoder"] = "text_encoder/model.fp16.safetensors"
    _m["text_encoder_2"] = "text_encoder_2/model.fp16.safetensors"
    _m["vae"] = "vae/diffusion_pytorch_model.fp16.safetensors"
PROMPT = os.environ.get("SDXL_PROMPT", "a red fox in a forest, highly detailed")
NEGATIVE = os.environ.get("SDXL_NEGATIVE", "blurry, low quality")
SEED = int(os.environ.get("SDXL_SEED", "42"))
STEPS = int(os.environ.get("SDXL_STEPS", "8"))
CFG = float(os.environ.get("SDXL_CFG", "7.0"))
W = int(os.environ.get("SDXL_W", "512"))
H = int(os.environ.get("SDXL_H", "512"))
FLOAT16 = bool(int(os.environ.get("FLOAT16", "0")))

sd = StableDiffusionXL(REPO, float16=FLOAT16)
sd.ensure_models_are_loaded()

# Replicate generate_latents by hand to capture every stage.
mx.random.seed(SEED)
conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=CFG, negative_text=NEGATIVE)
text_time = (pooled, mx.array([[512, 512, 0, 0, 512, 512.0]] * len(pooled)))
prior = sd.sampler.sample_prior((1, H // 8, W // 8, sd.autoencoder.latent_channels), dtype=sd.dtype)

latents = prior
step_latents = []
for t, t_prev in sd.sampler.timesteps(STEPS, start_time=sd.sampler.max_time, dtype=sd.dtype):
    latents = sd._denoising_step(latents, t, t_prev, conditioning, CFG, text_time)
    mx.eval(latents)
    step_latents.append(latents)
final = latents

decoded = sd.decode(final)  # clip(decode/2 + 0.5)
image_u8 = (decoded * 255).astype(mx.uint8)
mx.eval(decoded, image_u8, prior, conditioning, pooled)

# Save PNG.
suffix = "_fp16" if FLOAT16 else ""
from PIL import Image  # noqa: E402

Image.fromarray(np.array(image_u8[0])).convert("RGB").save(
    os.path.join(_GOLDEN_DIR, f"sdxl{suffix}_golden.png")
)

tensors = {
    "prior": prior.astype(mx.float32),
    "conditioning": conditioning.astype(mx.float32),
    "pooled": pooled.astype(mx.float32),
    "final_latents": final.astype(mx.float32),
    "image_u8": image_u8.astype(mx.uint8),
    **{f"step{i}_latents": sl.astype(mx.float32) for i, sl in enumerate(step_latents)},
}
meta = {
    "prompt": PROMPT, "negative": NEGATIVE, "seed": str(SEED), "steps": str(STEPS),
    "cfg": str(CFG), "w": str(W), "h": str(H), "float16": str(int(FLOAT16)),
}
out = os.path.join(_GOLDEN_DIR, f"sdxl{suffix}_golden.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"wrote {out}")
print(f"  prompt={PROMPT!r} seed={SEED} steps={STEPS} cfg={CFG} {W}x{H} float16={FLOAT16}")
print(f"  prior {tuple(prior.shape)}, final {tuple(final.shape)}, image {tuple(image_u8.shape)}")
