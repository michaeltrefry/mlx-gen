"""Real-weights FLUX.2-klein-9b LoRA + LoKr golden — the reference for the mlx-gen adapter gate
(sc-2646). Builds a deterministic synthetic adapter (LoRA, then LoKr) targeting the transformer's
double-block attention projections across a few blocks (LoRA also covers a single-block fused
`to_qkv_mlp_proj` + the global `x_embedder`), saves each in the on-disk format BOTH engines parse
(peft `diffusion_model.`-prefixed `lora_A/B.weight` + a bare `.alpha`; bare `lokr_w1/w2` +
`networkType=lokr` metadata), applies it through the fork's real `Flux2Klein(lora_paths=…,
lora_scales=[1.0])`, runs the fixed (prompt, seed, steps, size) render manually, and dumps the
decoded image. Forces f32 (the Rust crate runs f32 activations), so the gate is the cross-build
f32 floor — like the dense e2e — not an f32-vs-bf16 gap.

The Rust gate (`tests/adapter_real_weights.rs`) loads the SAME adapter files via `LoadSpec.adapters`.

Gitignored output. Run from the mflux fork venv:
    cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_adapter_golden.py
"""

import mlx.core as mx
import numpy as np

from mflux.models.common.config.model_config import ModelConfig

ModelConfig.precision = mx.float32  # the Rust pipeline runs f32 activations

from mflux.models.common.config import ModelConfig as MC  # noqa: E402
from mflux.models.common.config.config import Config  # noqa: E402
from mflux.models.flux2.variants import Flux2Klein  # noqa: E402

from _paths import fixture  # noqa: E402

PROMPT = "a red fox resting in fresh snow under soft winter light"
SEED, STEPS, SIZE, GUIDANCE = 0, 4, 256, 1.0
RANK = 8
DOUBLE_BLOCKS = [0, 4, 7]  # 9b has 8 double blocks (0..7)
ATTN_PROJS = ["to_q", "to_k", "to_v", "to_out", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
INNER = 4096  # all double-block attention projections are [4096, 4096]
LORA_STD, LOKR_STD = 0.02, 0.05


def _lora_pair(t, key_base, out_dim, in_dim, rng):
    """peft `lora_A/B` under `diffusion_model.` + a BARE `.alpha` (= 2·rank, a visible effect; the
    fork applies the same scaling via its bare alpha pattern, exercising the loader's alpha fold)."""
    a = rng.normal(0.0, LORA_STD, size=(RANK, in_dim)).astype(np.float32)
    b = rng.normal(0.0, LORA_STD, size=(out_dim, RANK)).astype(np.float32)
    t[f"diffusion_model.{key_base}.lora_A.weight"] = mx.array(a)
    t[f"diffusion_model.{key_base}.lora_B.weight"] = mx.array(b)
    t[f"{key_base}.alpha"] = mx.array(np.array([float(2 * RANK)], dtype=np.float32))


def build_lora(path):
    rng = np.random.default_rng(20260602)
    t = {}
    for blk in DOUBLE_BLOCKS:
        for proj in ATTN_PROJS:
            _lora_pair(t, f"transformer_blocks.{blk}.attn.{proj}", INNER, INNER, rng)
    # Single-block fused projection (q/k/v/mlp jointly): [inner*3 + mlp_hidden*2, inner].
    mlp_hidden = 3 * INNER
    _lora_pair(t, "single_transformer_blocks.0.attn.to_qkv_mlp_proj", INNER * 3 + mlp_hidden * 2, INNER, rng)
    # Global input embedder: [inner, in_channels=128].
    _lora_pair(t, "x_embedder", INNER, 128, rng)
    mx.save_safetensors(path, t)
    return path


def build_lokr(path):
    rng = np.random.default_rng(20260603)
    t = {}
    for blk in DOUBLE_BLOCKS:
        for proj in ATTN_PROJS:
            base = f"transformer_blocks.{blk}.attn.{proj}"
            # kron(w1[64,64], w2[64,64]) = [4096, 4096] = the attention projection delta shape.
            t[f"{base}.lokr_w1"] = mx.array(rng.normal(0.0, LOKR_STD, size=(64, 64)).astype(np.float32))
            t[f"{base}.lokr_w2"] = mx.array(rng.normal(0.0, LOKR_STD, size=(64, 64)).astype(np.float32))
    mx.save_safetensors(path, t, {"networkType": "lokr", "alpha": "1.0", "rank": "1"})
    return path


def render(adapter_path):
    model = Flux2Klein(quantize=None, lora_paths=[adapter_path], lora_scales=[1.0], model_config=MC.flux2_klein_9b())
    config = Config(
        model_config=model.model_config,
        num_inference_steps=STEPS,
        height=SIZE,
        width=SIZE,
        guidance=GUIDANCE,
        scheduler="flow_match_euler_discrete",
    )
    prompt_embeds, text_ids, neg_embeds, neg_ids = model._encode_prompt_pair(
        prompt=PROMPT, negative_prompt=" ", guidance=GUIDANCE
    )
    latents, latent_ids, lat_h, lat_w = model._prepare_generation_latents(seed=SEED, config=config)
    predict = model._predict(model.transformer)
    for t in range(config.init_time_step, config.num_inference_steps):
        noise = predict(
            latents=latents,
            latent_ids=latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=neg_embeds,
            negative_text_ids=neg_ids,
            guidance=GUIDANCE,
            timestep=config.scheduler.timesteps[t],
        )
        latents = config.scheduler.step(noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas)
        mx.eval(latents)
    packed = latents.reshape(latents.shape[0], lat_h, lat_w, latents.shape[-1]).transpose(0, 3, 1, 2)
    decoded = model.vae.decode_packed_latents(packed)  # NCHW [1,3,256,256]
    mx.eval(decoded)
    return decoded.astype(mx.float32)


for kind, builder in [("lora", build_lora), ("lokr", build_lokr)]:
    adapter_path = fixture(f"tools/golden/flux2_{kind}_adapter.safetensors")
    builder(adapter_path)
    decoded = render(adapter_path)
    out = fixture(f"tools/golden/flux2_{kind}_golden.safetensors")
    mx.save_safetensors(
        out,
        {"decoded": decoded},
        {
            "kind": kind, "prompt": PROMPT, "seed": str(SEED), "steps": str(STEPS),
            "width": str(SIZE), "height": str(SIZE),
        },
    )
    print(f"wrote {out} + {adapter_path}; decoded {tuple(decoded.shape)}")
