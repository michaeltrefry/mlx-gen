"""SDXL LoKr golden — reference for mlx-gen-sdxl LoKr merge (sc-2640).

The vendored SDXL path REJECTS LoKr, so there is no fork golden. We synthesize a deterministic LoKr
(full `lokr_w1`/`lokr_w2`, `kron([16,16],[40,40]) = [640,640]`) over `down_blocks.1`'s self-attention
(attn1) projections — 16 modules, all 640×640 — and merge it manually with the SAME validated
LyCORIS formula the Rust core uses (`δ = (alpha/rank)·kron(w1,w2)`, here alpha=rank=scale=1 so
`δ = kron(w1,w2)` exactly — all single-multiply ops, so the numpy and mlx deltas are bit-identical).

Produces:
- `sdxl_lokr_adapter.safetensors` — the kohya-format LoKr (`lora_unet_<flat>.lokr_w1/w2` + networkType
  meta), the shared input both engines read.
- `sdxl_lokr_golden.safetensors` — the render with ONLY the LoKr merged.
- `sdxl_lokr_stacked_golden.safetensors` — LCM-LoRA (vendored merge) THEN the LoKr, for the
  stacks-with-LoRA gate.

Run from the mflux venv:
  /Users/michael/Repos/mflux/.venv/bin/python3 tools/dump_sdxl_lokr_golden.py
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
LOKR_STD = float(os.environ.get("SDXL_LOKR_STD", "0.2"))

# down_blocks.1: 2 attentions × 2 transformer_blocks; attn1 (self-attn) is 640×640.
ATTNS, TBS = [0, 1], [0, 1]
# diffusers dotted leaf, kohya-flat leaf, vendored mlx-examples module attr
PROJS = [("to_q", "to_q", "query_proj"), ("to_k", "to_k", "key_proj"),
         ("to_v", "to_v", "value_proj"), ("to_out.0", "to_out_0", "out_proj")]


def build_lokr(path):
    rng = np.random.default_rng(20260640)
    tensors = {}
    for a in ATTNS:
        for tb in TBS:
            for _diff, flat, _vend in PROJS:
                stem = f"lora_unet_down_blocks_1_attentions_{a}_transformer_blocks_{tb}_attn1_{flat}"
                w1 = rng.normal(0.0, LOKR_STD, size=(16, 16)).astype(np.float32)
                w2 = rng.normal(0.0, LOKR_STD, size=(40, 40)).astype(np.float32)
                tensors[f"{stem}.lokr_w1"] = mx.array(w1)
                tensors[f"{stem}.lokr_w2"] = mx.array(w2)
    mx.save_safetensors(path, tensors, {"networkType": "lokr", "alpha": "1.0", "rank": "1"})
    return path


def merge_lokr(unet, adapter_path, scale=1.0):
    """Reconstruct each δ = (alpha/rank)·kron(w1,w2)·scale (alpha=rank=1) and merge `W += δ`."""
    from safetensors import safe_open
    tens = {}
    with safe_open(adapter_path, framework="np") as h:
        for k in h.keys():
            tens[k] = h.get_tensor(k)
    n = 0
    for a in ATTNS:
        for tb in TBS:
            for _diff, flat, vend in PROJS:
                stem = f"lora_unet_down_blocks_1_attentions_{a}_transformer_blocks_{tb}_attn1_{flat}"
                w1 = tens[f"{stem}.lokr_w1"]
                w2 = tens[f"{stem}.lokr_w2"]
                delta = (np.kron(w1, w2).astype(np.float32) * np.float32(scale))
                mod = getattr(unet.down_blocks[1].attentions[a].transformer_blocks[tb].attn1, vend)
                # Match production's merge_dense_delta: cast δ to the weight dtype before adding, so a
                # float16=True U-Net merges `W_f16 + δ.astype(f16)` (an f32 δ would promote W to f32).
                mod.weight = mod.weight + mx.array(delta).astype(mod.weight.dtype)
                n += 1
    return n


def render(sd):
    mx.random.seed(SEED)
    conditioning, pooled = sd._get_text_conditioning(PROMPT, n_images=1, cfg_weight=CFG, negative_text=NEGATIVE)
    text_time = (pooled, mx.array([[512, 512, 0, 0, 512, 512.0]] * len(pooled)))
    prior = sd.sampler.sample_prior((1, H // 8, W // 8, sd.autoencoder.latent_channels), dtype=sd.dtype)
    latents = prior
    for t, t_prev in sd.sampler.timesteps(STEPS, start_time=sd.sampler.max_time, dtype=sd.dtype):
        latents = sd._denoising_step(latents, t, t_prev, conditioning, CFG, text_time)
        mx.eval(latents)
    decoded = sd.decode(latents)
    image_u8 = (decoded * 255).astype(mx.uint8)
    mx.eval(image_u8)
    return image_u8


def save(name, image_u8, extra=None):
    from PIL import Image
    Image.fromarray(np.array(image_u8[0])).convert("RGB").save(os.path.join(_GOLDEN_DIR, f"{name}.png"))
    meta = {"prompt": PROMPT, "negative": NEGATIVE, "seed": str(SEED), "steps": str(STEPS),
            "cfg": str(CFG), "w": str(W), "h": str(H), "lokr_modules": "16"}
    if extra:
        meta.update(extra)
    out = os.path.join(_GOLDEN_DIR, f"{name}.safetensors")
    mx.save_safetensors(out, {"image_u8": image_u8.astype(mx.uint8)}, meta)
    print(f"wrote {out}")


adapter = build_lokr(os.path.join(_GOLDEN_DIR, "sdxl_lokr_adapter.safetensors"))

_sfx = "_fp16" if FLOAT16 else ""

# LoKr only.
sd = StableDiffusionXL(REPO, float16=FLOAT16)
sd.ensure_models_are_loaded()
n = merge_lokr(sd.unet, adapter, scale=1.0)
print(f"merged {n} LoKr modules (float16={FLOAT16})")
save(f"sdxl_lokr{_sfx}_golden", render(sd), {"lokr_path": adapter, "scale": "1.0"})

# LCM-LoRA then LoKr (stacking). Fresh UNet.
sd2 = StableDiffusionXL(REPO, float16=FLOAT16)
sd2.ensure_models_are_loaded()
touched = vlora.apply_loras_to_unet(sd2.unet, [{"path": LORA, "weight": 1.0}])
n2 = merge_lokr(sd2.unet, adapter, scale=1.0)
print(f"stacked: LoRA touched {touched} + LoKr merged {n2}")
save(f"sdxl_lokr_stacked{_sfx}_golden", render(sd2),
     {"lora_path": LORA, "lokr_path": adapter, "scale": "1.0", "touched": str(touched)})
