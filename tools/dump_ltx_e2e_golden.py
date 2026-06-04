"""LTX-2.3 e2e golden — PHASE B: the 2-stage pipeline → frames (sc-2679 S6).

Consumes PHASE A's `tools/golden/ltx_e2e_te.safetensors` (reference `input_ids` + `video_embeddings`)
and runs the real `ltx_2_3_base_q8` transformer (Q8) + upsampler + VAE through the 2-stage distilled
denoise at a real resolution (256×256, 9 frames) → uint8 frames. Dumps the committed e2e golden
(`mlx-gen-ltx/tests/fixtures/ltx_e2e_golden.safetensors`).

Precision: **f32** (every module upcast to f32 activations; the Q8 packed weights stay U32) — gates
the port's correctness isolated from bf16 rounding, mirroring the S3b/S5 gates. The Rust e2e
(`tests/e2e_parity.rs`, `Precision::F32Q8`) reproduces it.

**MUST run with mlx 0.31.2** (`quantized_matmul` changed 0.31.0→0.31.2). Run PHASE A first:
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg ~/Repos/mflux/.venv/bin/python tools/dump_ltx_e2e_te.py
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg /tmp/mlx312/bin/python tools/dump_ltx_e2e_golden.py
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

# `LTX_BF16=1` runs the reference's **native** bf16+Q8 pipeline end-to-end (no f32 upcast on the
# transformer / upsampler / VAE / statistics) — the production-precision parity target. Default = the
# f32 quality target (every module upcast to f32 activations; the Q8 packed weights stay U32).
BF16 = os.environ.get("LTX_BF16") == "1"
ACT = mx.bfloat16 if BF16 else mx.float32

te = mx.load(fixture("tools/golden/ltx_e2e_te.safetensors"))
input_ids = te["input_ids"]
context = te["video_embeddings"].astype(ACT)  # (1, 128, 4096)


def _f32_acts(model):
    # f32 target only — upcast every non-packed param to f32. bf16 keeps the native dtype (the
    # reference's production compute: bf16 activations × Q8, bf16 dense/norm/scale-shift params).
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


def denoise(latents, positions, sigmas):
    dtype = latents.dtype
    lat = latents
    for i in range(len(sigmas) - 1):
        sigma, sn = sigmas[i], sigmas[i + 1]
        b, c, f, h, w = lat.shape
        flat = mx.transpose(mx.reshape(lat, (b, c, -1)), (0, 2, 1))
        ts = mx.full((b, f * h * w), sigma, dtype=dtype)
        vel = forward_velocity(flat, ts, positions)
        vel = mx.reshape(mx.transpose(vel, (0, 2, 1)), (b, c, f, h, w))
        den = to_denoised(lat, vel, sigma)
        if sn > 0:
            lat = den + mx.array(sn, dtype=dtype) * (lat - den) / mx.array(sigma, dtype=dtype)
        else:
            lat = den
        mx.eval(lat)
    return lat


upsampler = load_upsampler(str(MODEL / "upsampler.safetensors"))
vae = load_vae_decoder(str(MODEL), timestep_conditioning=None, use_unified=True)
if not BF16:
    # f32 target: upcast the upsampler + VAE + their latent statistics. bf16 keeps them native (the
    # on-disk dtype is bf16 for both — the reference's production decode path).
    upsampler.update(tree_map(lambda p: p.astype(mx.float32), upsampler.parameters()))
    vae.update(tree_map(lambda p: p.astype(mx.float32), vae.parameters()))
    vae.latents_mean = vae.latents_mean.astype(mx.float32)
    vae.latents_std = vae.latents_std.astype(mx.float32)
mx.eval(upsampler.parameters(), vae.parameters())

mx.random.seed(7)
stage1_noise = (mx.random.normal((1, 128, LF, H1, W1)) * 0.5).astype(ACT)
stage2_noise = (mx.random.normal((1, 128, LF, H2, W2)) * 0.5).astype(ACT)
pos1 = create_video_position_grid(1, LF, H1, W1)
pos2 = create_video_position_grid(1, LF, H2, W2)

s1 = denoise(stage1_noise, pos1, STAGE1_SIGMAS)
ups = upsample_latents(s1, upsampler, vae.latents_mean, vae.latents_std)
ns = mx.array(STAGE2_SIGMAS[0], dtype=ups.dtype)
renoised = stage2_noise * ns + ups * (mx.array(1.0, dtype=ups.dtype) - ns)
final_latents = denoise(renoised, pos2, STAGE2_SIGMAS)

vid = vae(final_latents)
vid = mx.transpose(mx.squeeze(vid, axis=0), (1, 2, 3, 0))
frames = (mx.clip((vid + 1.0) / 2.0, 0.0, 1.0) * 255).astype(mx.uint8)
mx.eval(final_latents, frames)
print(f"e2e: context{context.shape} -> final{final_latents.shape} -> frames{frames.shape}")

# Stage latents are saved in the native compute dtype (`ACT`) so the Rust gate checks exact parity
# (the comparison upcasts both sides to f32, lossless). `video_embeddings` is bf16 either way (the TE
# output dtype; the f32 DiT upcasts it, exactly as the f32-act reference transformer does).
tensors = {
    "input_ids": input_ids.astype(mx.int32),
    "video_embeddings": context.astype(mx.bfloat16),
    "stage1_noise": stage1_noise,
    "stage2_noise": stage2_noise,
    "stage1_positions": pos1.astype(mx.float32),
    "stage2_positions": pos2.astype(mx.float32),
    "stage1_out": s1.astype(ACT),
    "upsampled": ups.astype(ACT),
    "renoised": renoised.astype(ACT),
    "final_latents": final_latents.astype(ACT),
    "frames": frames,
}
name = "ltx_e2e_golden_bf16.safetensors" if BF16 else "ltx_e2e_golden.safetensors"
out = fixture(f"mlx-gen-ltx/tests/fixtures/{name}")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"res": "256x256", "frames": "9", "prec": str(ACT)})
print(f"wrote {out}")
