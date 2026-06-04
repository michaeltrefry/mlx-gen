"""LTX-2.3 **I2V** golden — the single-image conditioning path, both stages (sc-2685).

Runs the reference `generate.py` / `generate_av.py` *video* I2V conditioning end-to-end: inject a
clean image latent at frame 0 (`apply_conditioning` + a per-frame denoise mask), seed each stage via
the noiser, and run the conditioned `denoise(..., state=...)` (per-token σ·mask + `apply_denoise_mask`)
over the real `ltx_2_3_base_q8` transformer (Q8) + upsampler + VAE → uint8 frames. Dumps the committed
I2V golden (`mlx-gen-ltx/tests/fixtures/ltx_i2v_golden{,_bf16}.safetensors`).

The conditioning **image latent** is synthesized deterministically (a fixed-seed normal) and injected —
the VAE *encoder* is gated separately (`tests/vae_parity.rs::encode_matches_reference`), so this golden
isolates exactly the I2V-new code: `apply_conditioning`, the noiser, the per-token timesteps, and
`apply_denoise_mask` (incl. conditioned-frame preservation at strength 1.0). `video_embeddings` is read
from the committed e2e golden fixture (no need to re-run the Gemma text-encoder PHASE A).

Precision: **f32** by default (every module upcast to f32 activations; Q8 packed weights stay U32);
`LTX_BF16=1` runs the reference's native bf16+Q8 production path. The Rust gate (`tests/i2v_parity.rs`)
reproduces both.

**MUST run with mlx 0.31.2** (`quantized_matmul` changed 0.31.0→0.31.2):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg /tmp/mlx312/bin/python tools/dump_ltx_i2v_golden.py
    LTX_BF16=1 MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg /tmp/mlx312/bin/python tools/dump_ltx_i2v_golden.py
"""

import glob
import os
import sys
import types
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())
for _name in ("mlx_vlm", "mlx_vlm.models", "mlx_vlm.models.gemma3"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
_lang = types.ModuleType("mlx_vlm.models.gemma3.language")
_lang.Gemma3Model = object
sys.modules["mlx_vlm.models.gemma3.language"] = _lang
_cfg = types.ModuleType("mlx_vlm.models.gemma3.config")
_cfg.TextConfig = object
sys.modules["mlx_vlm.models.gemma3.config"] = _cfg

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_map  # noqa: E402

from mlx_video.conditioning.latent import (  # noqa: E402
    LatentState,
    VideoConditionByLatentIndex,
    apply_conditioning,
    apply_denoise_mask,
)
from mlx_video.generate_av import create_video_position_grid  # noqa: E402
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType  # noqa: E402
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.transformer import Modality  # noqa: E402
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents  # noqa: E402
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder  # noqa: E402
from mlx_video.utils import to_denoised  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
STAGE1_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
STAGE2_SIGMAS = [0.909375, 0.725, 0.421875, 0.0]
# 256×256, 9 frames → latent_frames 2, stage1 4×4, stage2 8×8.
LF, H1, W1, H2, W2 = 2, 4, 4, 8, 8
# Single-image I2V: pin the first latent frame, full strength (the reference CLI defaults).
FRAME_IDX = 0
STRENGTH = 1.0

BF16 = os.environ.get("LTX_BF16") == "1"
ACT = mx.bfloat16 if BF16 else mx.float32

# `video_embeddings` from the committed e2e golden (bf16 TE output; the DiT keeps/­upcasts it).
e2e = mx.load(fixture("mlx-gen-ltx/tests/fixtures/ltx_e2e_golden.safetensors"))
context = e2e["video_embeddings"].astype(ACT)  # (1, 128, 4096)


def _f32_acts(model):
    if BF16:
        return
    model.update(
        tree_map(lambda p: p.astype(mx.float32) if p.dtype != mx.uint32 else p, model.parameters())
    )
    mx.eval(model.parameters())


config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=32, attention_head_dim=128, in_channels=128, out_channels=128,
    num_layers=48, cross_attention_dim=4096, caption_channels=4096,
    caption_projection_first_linear=False, caption_projection_second_linear=False,
    adaln_embedding_coefficient=9, apply_gated_attention=True,
    audio_num_attention_heads=32, audio_attention_head_dim=64, audio_in_channels=128,
    audio_out_channels=128, audio_cross_attention_dim=2048, audio_caption_channels=2048,
    rope_type=LTXRopeType.SPLIT, double_precision_rope=True,
    positional_embedding_theta=10000.0, positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20], use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
model = LTXModel(config)
raw = mx.load(str(MODEL / "transformer.safetensors"))
video = {k: v for k, v in raw.items() if "audio" not in k and "av_ca" not in k and "a2v" not in k}
qpaths = {k.rsplit(".", 1)[0] for k in video if k.endswith(".scales")}
nn.quantize(model, group_size=64, bits=8,
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and p in qpaths)
model.load_weights(list(video.items()), strict=False)
_f32_acts(model)


def forward_velocity(video_flat, timesteps, positions):
    modality = Modality(latent=video_flat, timesteps=timesteps, positions=positions,
                        context=context, context_mask=None, enabled=True)
    args = model.video_args_preprocessor.prepare(modality, None)
    emb_ts = args.embedded_timestep
    v = args
    for block in model.transformer_blocks.values():
        v, _ = block(video=v, audio=None)
    return model._process_output(model.scale_shift_table, model.norm_out, model.proj_out, v.x, emb_ts)


def denoise(positions, sigmas, state):
    """Conditioned denoise (reference `generate.py::denoise(..., state=...)`)."""
    dtype = state.latent.dtype
    lat = state.latent
    for i in range(len(sigmas) - 1):
        sigma, sn = sigmas[i], sigmas[i + 1]
        b, c, f, h, w = lat.shape
        num_tokens = f * h * w
        flat = mx.transpose(mx.reshape(lat, (b, c, -1)), (0, 2, 1))
        dmf = mx.reshape(state.denoise_mask, (b, 1, f, 1, 1))
        dmf = mx.broadcast_to(dmf, (b, 1, f, h, w))
        dmf = mx.reshape(dmf, (b, num_tokens))
        ts = mx.array(sigma, dtype=dtype) * dmf
        vel = forward_velocity(flat, ts, positions)
        vel = mx.reshape(mx.transpose(vel, (0, 2, 1)), (b, c, f, h, w))
        den = to_denoised(lat, vel, sigma)
        den = apply_denoise_mask(den, state.clean_latent, state.denoise_mask)
        if sn > 0:
            lat = den + mx.array(sn, dtype=dtype) * (lat - den) / mx.array(sigma, dtype=dtype)
        else:
            lat = den
        mx.eval(lat)
    return lat


def noiser(state, noise, noise_scale):
    ns = mx.array(noise_scale, dtype=state.latent.dtype)
    scaled = state.denoise_mask * ns
    one = mx.array(1.0, dtype=state.latent.dtype)
    return LatentState(
        latent=noise * scaled + state.latent * (one - scaled),
        clean_latent=state.clean_latent,
        denoise_mask=state.denoise_mask,
    )


upsampler = load_upsampler(str(MODEL / "upsampler.safetensors"))
vae = load_vae_decoder(str(MODEL), timestep_conditioning=None, use_unified=True)
if not BF16:
    upsampler.update(tree_map(lambda p: p.astype(mx.float32), upsampler.parameters()))
    vae.update(tree_map(lambda p: p.astype(mx.float32), vae.parameters()))
    vae.latents_mean = vae.latents_mean.astype(mx.float32)
    vae.latents_std = vae.latents_std.astype(mx.float32)
mx.eval(upsampler.parameters(), vae.parameters())

# Deterministic conditioning image latents (stand in for the VAE-encoded image; encoder gated
# separately) + stage noise.
mx.random.seed(11)
stage1_image_latent = (mx.random.normal((1, 128, 1, H1, W1)) * 0.5).astype(ACT)
stage2_image_latent = (mx.random.normal((1, 128, 1, H2, W2)) * 0.5).astype(ACT)
stage1_noise = (mx.random.normal((1, 128, LF, H1, W1)) * 0.5).astype(ACT)
stage2_noise = (mx.random.normal((1, 128, LF, H2, W2)) * 0.5).astype(ACT)
pos1 = create_video_position_grid(1, LF, H1, W1)
pos2 = create_video_position_grid(1, LF, H2, W2)

# --- Stage 1: zeros base → condition → noise (σ₀ = 1.0) → conditioned denoise. ---
shape1 = (1, 128, LF, H1, W1)
state1 = LatentState(
    latent=mx.zeros(shape1, dtype=ACT),
    clean_latent=mx.zeros(shape1, dtype=ACT),
    denoise_mask=mx.ones((1, 1, LF, 1, 1), dtype=ACT),
)
state1 = apply_conditioning(state1, [VideoConditionByLatentIndex(
    latent=stage1_image_latent, frame_idx=FRAME_IDX, strength=STRENGTH)])
stage1_mask = state1.denoise_mask
stage1_clean = state1.clean_latent
state1 = noiser(state1, stage1_noise, STAGE1_SIGMAS[0])
stage1_state_latent = state1.latent
s1 = denoise(pos1, STAGE1_SIGMAS, state1)

# --- Upsample 2×. ---
ups = upsample_latents(s1, upsampler, vae.latents_mean, vae.latents_std)

# --- Stage 2: upscaled base → condition → re-noise (σ₀ = STAGE2_SIGMAS[0]) → conditioned denoise. ---
state2 = LatentState(
    latent=ups,
    clean_latent=mx.zeros_like(ups),
    denoise_mask=mx.ones((1, 1, LF, 1, 1), dtype=ACT),
)
state2 = apply_conditioning(state2, [VideoConditionByLatentIndex(
    latent=stage2_image_latent, frame_idx=FRAME_IDX, strength=STRENGTH)])
stage2_mask = state2.denoise_mask
stage2_clean = state2.clean_latent
state2 = noiser(state2, stage2_noise, STAGE2_SIGMAS[0])
stage2_state_latent = state2.latent
final_latents = denoise(pos2, STAGE2_SIGMAS, state2)

vid = vae(final_latents)
vid = mx.transpose(mx.squeeze(vid, axis=0), (1, 2, 3, 0))
frames = (mx.clip((vid + 1.0) / 2.0, 0.0, 1.0) * 255).astype(mx.uint8)
mx.eval(final_latents, frames)
print(f"i2v: ctx{context.shape} img1{stage1_image_latent.shape} -> final{final_latents.shape} -> frames{frames.shape}")

tensors = {
    "video_embeddings": context.astype(mx.bfloat16),
    "strength": mx.array([STRENGTH], dtype=mx.float32),
    "frame_idx": mx.array([FRAME_IDX], dtype=mx.int32),
    "stage1_image_latent": stage1_image_latent,
    "stage2_image_latent": stage2_image_latent,
    "stage1_noise": stage1_noise,
    "stage2_noise": stage2_noise,
    "stage1_positions": pos1.astype(mx.float32),
    "stage2_positions": pos2.astype(mx.float32),
    "stage1_mask": stage1_mask.astype(ACT),
    "stage1_clean": stage1_clean.astype(ACT),
    "stage1_state_latent": stage1_state_latent.astype(ACT),
    "stage1_out": s1.astype(ACT),
    "upsampled": ups.astype(ACT),
    "stage2_mask": stage2_mask.astype(ACT),
    "stage2_clean": stage2_clean.astype(ACT),
    "stage2_state_latent": stage2_state_latent.astype(ACT),
    "final_latents": final_latents.astype(ACT),
    "frames": frames,
}
name = "ltx_i2v_golden_bf16.safetensors" if BF16 else "ltx_i2v_golden.safetensors"
out = fixture(f"mlx-gen-ltx/tests/fixtures/{name}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"res": "256x256", "frames": "9", "strength": str(STRENGTH), "prec": str(ACT)})
print(f"wrote {out}")
