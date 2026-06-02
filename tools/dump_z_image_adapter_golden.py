"""Real-weights Z-Image LoRA + LoKr golden — the reference for the mlx-gen adapter gate (sc-2602).

Run from the fork:  cd ~/repos/mflux && uv run python <path>/dump_z_image_adapter_golden.py
(or: ~/Repos/mflux/.venv/bin/python tools/dump_z_image_adapter_golden.py)

Generates a deterministic synthetic adapter (LoRA, then LoKr) targeting the attention projections
`to_q/k/v/to_out.0` across a few layers — the trained-character case — saves each adapter in the
on-disk format BOTH engines parse (peft `transformer.`-prefixed `lora_A/B(.weight)` + `.alpha` for
LoRA; bare `lokr_w1/w2` + `networkType=lokr`/`alpha`/`rank` metadata for LoKr), applies it through
the fork's real `ZImageInitializer.init(lora_paths=…, lora_scales=[1.0])`, runs the fixed
(prompt, seed, steps, size) render, and dumps the decoded image. The Rust gate
(`tests/adapter_real_weights.rs`) loads the SAME adapter file via `LoadSpec.adapters` and compares
its render px>8 against this golden — so the adapter file is the shared input, the two engines the
variables.
"""

import math
import os

import mlx.core as mx
import numpy as np
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.schedulers.flow_match_euler_discrete_scheduler import (
    FlowMatchEulerDiscreteScheduler as S,
)
from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator
from mflux.models.z_image.model.z_image_text_encoder.prompt_encoder import PromptEncoder
from mflux.models.z_image.z_image_initializer import ZImageInitializer
from mflux.utils.image_util import ImageUtil

_GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
os.makedirs(_GOLDEN_DIR, exist_ok=True)

PROMPT = os.environ.get("ZIMAGE_PROMPT", "a fox")
SEED = int(os.environ.get("ZIMAGE_SEED", "42"))
STEPS = int(os.environ.get("ZIMAGE_STEPS", "4"))
W = int(os.environ.get("ZIMAGE_W", "256"))
H = int(os.environ.get("ZIMAGE_H", "256"))
# Adapter magnitudes (env-overridable for tuning a visible-but-non-destructive effect).
LORA_STD = float(os.environ.get("ZIMAGE_LORA_STD", "0.03"))
LOKR_STD = float(os.environ.get("ZIMAGE_LOKR_STD", "0.06"))
RANK = 8
LAYERS = [0, 14, 29]
PROJS = ["to_q", "to_k", "to_v", "to_out.0"]
DIM = 3840  # all attention projections are [3840, 3840]


def _rng(seed):
    return np.random.default_rng(seed)


def build_lora(path):
    """peft-format LoRA: `transformer.<module>.lora_A/B.weight` [r,in]/[out,r] + `.alpha` (=rank)."""
    rng = _rng(20260602)
    tensors = {}
    for li in LAYERS:
        for proj in PROJS:
            base = f"transformer.layers.{li}.attention.{proj}"
            a = rng.normal(0.0, LORA_STD, size=(RANK, DIM)).astype(np.float32)
            b = rng.normal(0.0, LORA_STD, size=(DIM, RANK)).astype(np.float32)
            tensors[f"{base}.lora_A.weight"] = mx.array(a)
            tensors[f"{base}.lora_B.weight"] = mx.array(b)
            tensors[f"{base}.alpha"] = mx.array(np.array([float(RANK)], dtype=np.float32))
    mx.save_safetensors(path, tensors)
    return path


def build_lokr(path):
    """LyCORIS-format LoKr: bare `<module>.lokr_w1/w2`, kron([64,64],[60,60])=[3840,3840]."""
    rng = _rng(20260603)
    tensors = {}
    for li in LAYERS:
        for proj in PROJS:
            base = f"layers.{li}.attention.{proj}"
            w1 = rng.normal(0.0, LOKR_STD, size=(64, 64)).astype(np.float32)
            w2 = rng.normal(0.0, LOKR_STD, size=(60, 60)).astype(np.float32)
            tensors[f"{base}.lokr_w1"] = mx.array(w1)
            tensors[f"{base}.lokr_w2"] = mx.array(w2)
    meta = {"networkType": "lokr", "alpha": "1.0", "rank": "1"}
    mx.save_safetensors(path, tensors, meta)
    return path


def render(adapter_path):
    class Holder:
        pass

    model = Holder()
    ZImageInitializer.init(
        model,
        model_config=ModelConfig.z_image_turbo(),
        quantize=None,
        lora_paths=[adapter_path],
        lora_scales=[1.0],
    )
    tok = model.tokenizers["z_image"]
    tout = tok.tokenize(PROMPT)
    num_valid = int(mx.sum(tout.attention_mask[0]).item())
    cap_feats = PromptEncoder.encode_prompt(PROMPT, tok, model.text_encoder)

    mu = math.log(3.0)
    sigmas = mx.linspace(1.0, 1.0 / STEPS, STEPS)
    sigmas = S._time_shift_exponential_array(mu, 1.0, sigmas)
    sigmas = mx.concatenate([sigmas, mx.zeros((1,), dtype=sigmas.dtype)], axis=0)

    latents = ZImageLatentCreator.create_noise(SEED, H, W)
    for t in range(STEPS):
        ts = mx.array(1.0 - float(sigmas[t]), dtype=mx.float32)
        v = model.transformer(x=latents, timestep=ts, sigmas=sigmas, cap_feats=cap_feats)
        latents = latents + (sigmas[t + 1] - sigmas[t]) * v
        mx.eval(latents)
    unpacked = ZImageLatentCreator.unpack_latents(latents, H, W)
    decoded = model.vae.decode(unpacked)
    return decoded, num_valid


for kind, builder in [("lora", build_lora), ("lokr", build_lokr)]:
    adapter_path = os.path.join(_GOLDEN_DIR, f"z_image_{kind}_adapter.safetensors")
    builder(adapter_path)
    decoded, num_valid = render(adapter_path)
    img = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(ImageUtil._denormalize(decoded)))
    png = os.path.join(_GOLDEN_DIR, f"z_image_{kind}_golden.png")
    img.save(png)
    out = os.path.join(_GOLDEN_DIR, f"z_image_{kind}_golden.safetensors")
    mx.save_safetensors(
        out,
        {"decoded": decoded.astype(mx.float32)},
        {
            "prompt": PROMPT, "seed": str(SEED), "steps": str(STEPS), "w": str(W), "h": str(H),
            "num_valid": str(num_valid), "kind": kind, "scale": "1.0",
        },
    )
    print(f"wrote {out} + {png} + {adapter_path}; decoded {tuple(decoded.shape)}")
