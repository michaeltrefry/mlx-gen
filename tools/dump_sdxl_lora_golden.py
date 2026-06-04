"""SDXL LoRA golden — reference for mlx-gen-sdxl LoRA merge (sc-2639).

Merges a real kohya LoRA (default: `latent-consistency/lcm-lora-sdxl`) into the vendored SDXL UNet
via the SceneWorks `lora.py` path, records the touched-module count (the parity target — 515 for
LCM-LoRA), and renders the merged UNet with the same by-hand Euler-Ancestral harness as
`dump_sdxl_golden.py`. The Rust port (`apply_sdxl_adapters`, merge into the f32 weights) must match
this image px>8==0 and the touched count.

Also runs a DELTA BIT-EXACTNESS check: the Rust merge computes `(b@a)` in f32 (the pmetal 16-bit
GEMM bug forbids an f16 matmul) then rounds back through f16; this asserts that equals the vendored
f16 `(b@a)` bit-for-bit, so the merge is bit-identical without ever running the buggy GEMM.

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_lora_golden.py
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
from mlx_sd import lora as vlora  # noqa: E402
import mlx_sd.model_io as _mio  # noqa: E402

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
FLOAT16 = bool(int(os.environ.get("FLOAT16", "0")))  # production = fp16 (U-Net/TE; VAE f32)

# This machine's HF cache has only the `.fp16.` variant — point `_MODELS` at it so the reference
# loads offline (exact fp16 weights for FLOAT16=1; upcast to f32 for FLOAT16=0).
if os.environ.get("SDXL_FP16_FILES", "1") == "1":
    _m = _mio._MODELS[REPO]
    _m["unet"] = "unet/diffusion_pytorch_model.fp16.safetensors"
    _m["text_encoder"] = "text_encoder/model.fp16.safetensors"
    _m["text_encoder_2"] = "text_encoder_2/model.fp16.safetensors"
    _m["vae"] = "vae/diffusion_pytorch_model.fp16.safetensors"
LORA = os.environ.get(
    "SDXL_LORA",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--latent-consistency--lcm-lora-sdxl/snapshots/"
        "a18548dd4956b174ec5b0d78d340c8dae0a129cd/pytorch_lora_weights.safetensors"
    ),
)
PROMPT = os.environ.get("SDXL_PROMPT", "a red fox in a forest, highly detailed")
NEGATIVE = os.environ.get("SDXL_NEGATIVE", "blurry, low quality")
SEED = int(os.environ.get("SDXL_SEED", "42"))
STEPS = int(os.environ.get("SDXL_STEPS", "8"))
CFG = float(os.environ.get("SDXL_CFG", "7.0"))
W = int(os.environ.get("SDXL_W", "512"))
H = int(os.environ.get("SDXL_H", "512"))
SCALE = float(os.environ.get("SDXL_LORA_SCALE", "1.0"))

sd = StableDiffusionXL(REPO, float16=FLOAT16)
sd.ensure_models_are_loaded()

# --- Delta bit-exactness check across ALL merged modules ---
# The Rust merge computes `(b@a)` in f32 (the pmetal 16-bit GEMM bug forbids an f16 matmul) then
# rounds back through f16; this proves that equals the vendored f16 `(b@a)` bit-for-bit for every
# module, so the merge is bit-identical to the reference without ever running the buggy GEMM.
import collections  # noqa: E402

import mlx.nn as nn  # noqa: E402
from safetensors import safe_open  # noqa: E402

_k2d = {}
for _name, _m in sd.unet.named_modules():
    if isinstance(_m, nn.Linear):
        _k2d[_name.replace(".", "_")] = _name
        _p, _, _leaf = _name.rpartition(".")
        _alias = vlora._DIFFUSERS_LEAF_ALIASES.get(_leaf)
        if _p and _alias is not None:
            _k2d[f"{_p}.{_alias}".replace(".", "_")] = _name

_trip = collections.defaultdict(dict)
with safe_open(LORA, framework="mlx") as h:
    for k in h.keys():
        rem = k[len("lora_unet_"):]
        for suf, role in ((".lora_down.weight", "down"), (".lora_up.weight", "up"), (".alpha", "alpha")):
            if rem.endswith(suf):
                name = _k2d.get(rem[: -len(suf)])
                if name is not None:
                    _trip[name][role] = h.get_tensor(k)
                break

delta_checked = delta_mismatches = 0
for name, parts in _trip.items():
    if "down" not in parts or "up" not in parts:
        continue
    a, b = parts["down"], parts["up"]
    if a.ndim != 2 or b.ndim != 2:
        continue
    ref = (b @ a).astype(mx.float32)
    rust = (b.astype(mx.float32) @ a.astype(mx.float32)).astype(b.dtype).astype(mx.float32)
    mx.eval(ref, rust)
    delta_checked += 1
    if not bool(mx.all(ref == rust).item()):
        delta_mismatches += 1
print(f"delta bit-exactness (f32-matmul->f16->f32 == vendored f16 matmul): "
      f"checked={delta_checked} mismatches={delta_mismatches}")
assert delta_mismatches == 0, "Rust delta recipe diverges from the vendored f16 matmul"
delta_match = delta_mismatches == 0

# --- Merge the LoRA and render ---
touched = vlora.apply_loras_to_unet(sd.unet, [{"path": LORA, "weight": SCALE}])
print(f"vendored apply_loras_to_unet touched {touched} modules (scale {SCALE})")

mx.random.seed(SEED)
conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=CFG, negative_text=NEGATIVE)
text_time = (pooled, mx.array([[512, 512, 0, 0, 512, 512.0]] * len(pooled)))
prior = sd.sampler.sample_prior((1, H // 8, W // 8, sd.autoencoder.latent_channels), dtype=sd.dtype)

latents = prior
for t, t_prev in sd.sampler.timesteps(STEPS, start_time=sd.sampler.max_time, dtype=sd.dtype):
    latents = sd._denoising_step(latents, t, t_prev, conditioning, CFG, text_time)
    mx.eval(latents)
final = latents

decoded = sd.decode(final)
image_u8 = (decoded * 255).astype(mx.uint8)
mx.eval(decoded, image_u8)

from PIL import Image  # noqa: E402

Image.fromarray(np.array(image_u8[0])).convert("RGB").save(
    os.path.join(_GOLDEN_DIR, "sdxl_lora_golden.png")
)

tensors = {"image_u8": image_u8.astype(mx.uint8)}
meta = {
    "prompt": PROMPT, "negative": NEGATIVE, "seed": str(SEED), "steps": str(STEPS),
    "cfg": str(CFG), "w": str(W), "h": str(H), "scale": str(SCALE),
    "lora_path": LORA, "touched": str(touched), "delta_match": str(int(delta_match)),
}
suffix = "_fp16" if FLOAT16 else ""
out = os.path.join(_GOLDEN_DIR, f"sdxl_lora{suffix}_golden.safetensors")
mx.save_safetensors(out, tensors, meta)
print(f"wrote {out}")
print(f"  float16={FLOAT16} prompt={PROMPT!r} seed={SEED} steps={STEPS} cfg={CFG} {W}x{H} scale={SCALE} touched={touched}")
