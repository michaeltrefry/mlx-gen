"""SDXL Q4/Q8 golden — reference for mlx-gen-sdxl quantization (sc-2641).

The vendored SDXL path runs fp16 and does NOT quantize, so there's no production quant reference. We
build a **vendored-equivalent** one: cast every U-Net + dual-CLIP `nn.Linear` weight to bf16 (the
sc-2604 discipline — SDXL ships fp16/fp32 on disk; quantizing the as-loaded dtype drifts the group
scales → the sc-1975 "Q8 broken on base-1.0"), then `nn.quantize(Linear-only, group_size=64)` —
exactly the Rust scope (`unet.quantize` + both `te.quantize`; VAE stays f32). Renders the quantized
model with the same by-hand harness → the Rust `load(Q).generate()` must match px>8 (tight: both
quantize identically, so the chaos sampler stays on the same trajectory).

Also dumps a per-module **scales byte-match** reference (a real `down_blocks.1...attn1.to_q` weight,
bf16-cast, `mx.quantize`) — the strongest correctness gate, proving the loaded Q8/Q4 scales are
bit-exact on base-1.0 (the sc-1975 root cause).

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_quant_golden.py
"""

import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_HERE, "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

sys.path.insert(0, os.environ.get(
    "SDXL_VENDOR_PARENT", "/Users/michael/Repos/SceneWorks/apps/worker/scene_worker/_vendor"))
from mlx_sd import StableDiffusionXL  # noqa: E402
import mlx_sd.model_io as _mio  # noqa: E402

REPO = "stabilityai/stable-diffusion-xl-base-1.0"
FLOAT16 = bool(int(os.environ.get("FLOAT16", "0")))  # production = fp16 base, then quantize

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

GROUP = 64
PROBE = ("down_blocks", 1, "attentions", 0, "transformer_blocks", 0, "attn1", "query_proj")


def bf16_cast_linears(model):
    """Cast every nn.Linear weight + bias to bf16 in place (the Rust core quantize pre-cast)."""
    for _name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            m.weight = m.weight.astype(mx.bfloat16)
            if "bias" in m and m.bias is not None:
                m.bias = m.bias.astype(mx.bfloat16)


def quantize_model(sd, bits):
    pred = lambda _p, m: isinstance(m, nn.Linear)  # noqa: E731  (Linear-only; embeddings stay dense)
    for comp in (sd.unet, sd.text_encoder_1, sd.text_encoder_2):
        bf16_cast_linears(comp)
        nn.quantize(comp, group_size=GROUP, bits=bits, class_predicate=pred)


def render(sd):
    mx.random.seed(SEED)
    conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=CFG, negative_text=NEGATIVE)
    text_time = (pooled, mx.array([[512, 512, 0, 0, 512, 512.0]] * len(pooled)))
    prior = sd.sampler.sample_prior((1, H // 8, W // 8, sd.autoencoder.latent_channels), dtype=sd.dtype)
    latents = prior
    for t, t_prev in sd.sampler.timesteps(STEPS, start_time=sd.sampler.max_time, dtype=sd.dtype):
        latents = sd._denoising_step(latents, t, t_prev, conditioning, CFG, text_time)
        mx.eval(latents)
    img = (sd.decode(latents) * 255).astype(mx.uint8)
    mx.eval(img)
    return img


def save(name, img, bits):
    from PIL import Image
    Image.fromarray(np.array(img[0])).convert("RGB").save(os.path.join(_GOLDEN_DIR, f"{name}.png"))
    meta = {"prompt": PROMPT, "negative": NEGATIVE, "seed": str(SEED), "steps": str(STEPS),
            "cfg": str(CFG), "w": str(W), "h": str(H), "bits": str(bits), "group": str(GROUP)}
    mx.save_safetensors(os.path.join(_GOLDEN_DIR, f"{name}.safetensors"),
                        {"image_u8": img.astype(mx.uint8)}, meta)
    print(f"wrote {name}.safetensors (bits={bits})")


# --- Per-module scales byte-match reference (Q8 + Q4), from the dense f32 probe weight ---
sd0 = StableDiffusionXL(REPO, float16=False)
sd0.ensure_models_are_loaded()
mod = sd0.unet
for k in PROBE[:-1]:
    mod = mod[k] if isinstance(k, int) else getattr(mod, k)
probe_w = getattr(mod, PROBE[-1]).weight  # f32 [out, in]
ten = {"probe_w_f32": probe_w.astype(mx.float32)}
meta = {"probe_path": "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q", "group": str(GROUP)}
for bits in (8, 4):
    wq, scales, biases = mx.quantize(probe_w.astype(mx.bfloat16), group_size=GROUP, bits=bits)
    ten[f"wq_q{bits}"] = wq
    ten[f"scales_q{bits}"] = scales.astype(mx.float32)
    ten[f"biases_q{bits}"] = biases.astype(mx.float32)
mx.save_safetensors(os.path.join(_GOLDEN_DIR, "sdxl_quant_scales_ref.safetensors"), ten, meta)
print("wrote sdxl_quant_scales_ref.safetensors (probe query_proj, Q8+Q4)")
del sd0

# --- Reference Q8 + Q4 renders (UNet + both TEs, bf16-cast + nn.quantize Linear-only) ---
# At float16=True the base is fp16 (production); quantize bf16-casts the Linear weights either way, so
# the packed weights are identical, but the render runs f16 activations (matching `load(Q)` at fp16).
_sfx = "_fp16" if FLOAT16 else ""
for bits in (8, 4):
    sd = StableDiffusionXL(REPO, float16=FLOAT16)
    sd.ensure_models_are_loaded()
    quantize_model(sd, bits)
    save(f"sdxl_q{bits}{_sfx}_golden", render(sd), bits)
    del sd
    mx.clear_cache()
