"""SDXL img2img golden — reference for mlx-gen-sdxl S6 (sc-2638).

Replicates the vendored `StableDiffusionXL.generate_latents_from_image` by hand: VAE-encode an init
image (mean), `add_noise` at `max_time·strength`, then `int(steps·strength)` Euler-Ancestral steps.
Uses a target-sized init image (no resize) so the gate isolates the encode + add_noise + denoise.
The init image is normalized `2·(u8/255) − 1` — the SAME formula the Rust `preprocess_init_image`
uses (there is no canonical SDXL-mlx img2img preprocessing; we define it).

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_img2img_golden.py
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
PROMPT = os.environ.get("SDXL_PROMPT", "an oil painting of a fox")
NEGATIVE = os.environ.get("SDXL_NEGATIVE", "blurry")
SEED = int(os.environ.get("SDXL_SEED", "7"))
STEPS = int(os.environ.get("SDXL_STEPS", "8"))
CFG = float(os.environ.get("SDXL_CFG", "7.0"))
STRENGTH = float(os.environ.get("SDXL_STRENGTH", "0.7"))
W = int(os.environ.get("SDXL_W", "512"))
H = int(os.environ.get("SDXL_H", "512"))
FLOAT16 = bool(int(os.environ.get("FLOAT16", "0")))  # production path = fp16 (U-Net/TE; VAE stays f32)

# Point `_MODELS` at the cached `.fp16.` files so the reference loads offline (this machine has only
# the fp16 variant). For FLOAT16=1 they're the exact fp16 weights; for FLOAT16=0 upcast to f32.
if os.environ.get("SDXL_FP16_FILES", "1") == "1":
    _m = _mio._MODELS[REPO]
    _m["unet"] = "unet/diffusion_pytorch_model.fp16.safetensors"
    _m["text_encoder"] = "text_encoder/model.fp16.safetensors"
    _m["text_encoder_2"] = "text_encoder_2/model.fp16.safetensors"
    _m["vae"] = "vae/diffusion_pytorch_model.fp16.safetensors"

sd = StableDiffusionXL(REPO, float16=FLOAT16)
sd.ensure_models_are_loaded()

# A deterministic uint8 init image, target-sized (no resize). NHWC [H, W, 3].
rng = np.random.default_rng(123)
init_u8 = rng.integers(0, 256, size=(H, W, 3), dtype=np.uint8)
# Normalize 2·(v/255) − 1 in pure float32, in the SAME op order as the Rust `preprocess_init_image`
# (no f64 promotion) so the encoded x_0 is bit-identical.
v = init_u8.astype(np.float32)
init_norm = np.float32(2.0) * (v / np.float32(255.0)) - np.float32(1.0)
image = mx.array(init_norm)  # [H, W, 3]

mx.random.seed(SEED)
start_step = sd.sampler.max_time * STRENGTH
eff_steps = int(STEPS * STRENGTH)
conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=CFG, negative_text=NEGATIVE)
text_time = (pooled, mx.array([[512, 512, 0, 0, 512, 512.0]] * len(pooled)))

x_0, _ = sd.autoencoder.encode(image[None])  # [1, H/8, W/8, 4] mean
x_t = sd.sampler.add_noise(x_0, mx.array(start_step))  # draw #0

latents = x_t
step_latents = []
ts_used = []
for t, t_prev in sd.sampler.timesteps(eff_steps, start_time=start_step, dtype=sd.dtype):
    ts_used.append(float(t))
    latents = sd._denoising_step(latents, t, t_prev, conditioning, CFG, text_time)
    mx.eval(latents)
    step_latents.append(latents)
final = latents
ts_used.append(0.0)

# Recompute step1's CFG eps (UNet at t=ts_used[1], input = step0 output) for a direct eps gate.
_xu = mx.concatenate([step_latents[0]] * 2, axis=0)
_t1 = mx.broadcast_to(mx.array(ts_used[1]), [2])
_eps = sd.unet(_xu, _t1, encoder_x=conditioning, text_time=text_time)
_et, _en = _eps.split(2)
eps1_cfg = _en + CFG * (_et - _en)
mx.eval(eps1_cfg)

# Reproduce the per-step ancestral noise stream draw-for-draw (add_noise #0, step0 #1, step1 #2)
# to isolate the RNG from the deterministic step ops.
mx.random.seed(SEED)
_n0 = mx.random.normal(x_0.shape)
_n1 = mx.random.normal(x_0.shape)
noise2 = mx.random.normal(x_0.shape)
mx.eval(noise2)

decoded = sd.decode(final)
image_u8 = (decoded * 255).astype(mx.uint8)
mx.eval(decoded, image_u8, x_0, x_t)

tensors = {
    "init_u8": mx.array(init_u8.astype(np.int32)),  # the uint8 init image the Rust test loads
    "x0_mean": x_0.astype(mx.float32),
    "x_t": x_t.astype(mx.float32),
    "final_latents": final.astype(mx.float32),
    "image_u8": image_u8.astype(mx.uint8),
    "timesteps": mx.array(ts_used, dtype=mx.float32),
    "eps1_cfg": eps1_cfg.astype(mx.float32),
    "noise2": noise2.astype(mx.float32),
    **{f"step{i}_latents": sl.astype(mx.float32) for i, sl in enumerate(step_latents)},
}
meta = {
    "prompt": PROMPT, "negative": NEGATIVE, "seed": str(SEED), "steps": str(STEPS),
    "cfg": str(CFG), "strength": str(STRENGTH), "w": str(W), "h": str(H),
}
suffix = "_fp16" if FLOAT16 else ""
out = os.path.join(_GOLDEN_DIR, f"sdxl_img2img{suffix}_golden.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"wrote {out}")
print(f"  float16={FLOAT16} strength={STRENGTH} start_step={start_step} eff_steps={eff_steps}; image {tuple(image_u8.shape)}")
