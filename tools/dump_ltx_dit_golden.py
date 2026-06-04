"""LTX-2.3 full DiT (velocity) golden — reference video-only `LTXModel` I/O (sc-2679 S3b).

Builds the reference **VideoOnly** `LTXModel` with the 2.3 config (48 layers, dim 4096, gated, no
caption-projection), loads the real `ltx_2_3_base_q8` `transformer.safetensors` video weights —
`nn.quantize`-ing the attn/ff Linears (group 64 / 8-bit, from `split_model.json`) — and runs one
**f32-activation × Q8** velocity forward over deterministic synthetic inputs (small token grid). The
Rust `LtxDiT` (mlx-gen-ltx/tests/dit_parity.rs, `Precision::F32Q8`) loads the SAME Q8 weights and
must reproduce the velocity.

f32 activations are the port's quality target; the Q8 weights are kept (no dense bf16 transformer
exists). The audio + cross-modal weights are filtered out so the VideoOnly model loads cleanly.

**The golden MUST be generated with mlx 0.31.2** (matching the Rust build): `quantized_matmul`
changed 0.31.0→0.31.2, so a 0.31.0 golden mismatches the Rust quant path by ~5e-4/op (the dense
S0–S2 ops are bit-identical across the two, which is why only the DiT exposed it). The mflux venv is
0.31.0; use a 0.31.2 env.

Run (mlx 0.31.2 env + mlx_video source):
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg \
      /tmp/mlx312/bin/python tools/dump_ltx_dit_golden.py
Output (committed): mlx-gen-ltx/tests/fixtures/ltx_dit_golden.safetensors
"""

import glob
import os
import sys
from pathlib import Path

from _paths import fixture


def _find_mlx_video_src() -> str:
    if env := os.environ.get("MLX_VIDEO_SRC"):
        return str(Path(env).expanduser())
    for cand in sorted(glob.glob(str(Path.home() / ".cache/uv/archive-v0/*/mlx_video"))):
        return str(Path(cand).parent)
    raise SystemExit("Set MLX_VIDEO_SRC to the dir containing `mlx_video/`.")


sys.path.insert(0, _find_mlx_video_src())

import types  # noqa: E402

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

from mlx_video.generate import create_position_grid  # noqa: E402
from mlx_video.models.ltx.config import (  # noqa: E402
    LTXModelConfig,
    LTXModelType,
    LTXRopeType,
)
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.transformer import Modality  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
DIM, CTX = 4096, 16
LF, LH, LW = 2, 4, 4  # latent frames/height/width → S = 32 tokens

# `LTX_BF16=1` emits the reference's **native** bf16+Q8 forward (no f32 upcast, bf16 activations) —
# the production-precision parity target (Rust `Precision::Bf16Q8`). Default = the f32 quality target
# (`Precision::F32Q8`). bf16 uses a non-bf16-exact sigma so the timestep `×1000` rounding is exercised.
BF16 = os.environ.get("LTX_BF16") == "1"
ACT = mx.bfloat16 if BF16 else mx.float32

# Build the AudioVideo model (the reference __call__ only wires the multimodal preprocessor) and run
# it video-only (audio=None) — the video path is independent of the audio/cross-modal modules, which
# stay random + unused. Mirrors generate_av's config build (gated 2.3, no caption-projection).
config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo,
    num_attention_heads=32,
    attention_head_dim=128,
    in_channels=128,
    out_channels=128,
    num_layers=48,
    cross_attention_dim=4096,
    caption_channels=4096,  # no caption-projection → connector out
    caption_projection_first_linear=False,
    caption_projection_second_linear=False,
    adaln_embedding_coefficient=9,
    apply_gated_attention=True,
    audio_num_attention_heads=32,
    audio_attention_head_dim=64,
    audio_in_channels=128,
    audio_out_channels=128,
    audio_cross_attention_dim=2048,
    audio_caption_channels=2048,
    rope_type=LTXRopeType.SPLIT,
    double_precision_rope=True,
    positional_embedding_theta=10000.0,
    positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20],
    use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
model = LTXModel(config)

# Load the video weights (drop audio / cross-modal), quantize the Q8 Linears, load.
raw = mx.load(str(MODEL / "transformer.safetensors"))
video = {
    k: v
    for k, v in raw.items()
    if "audio" not in k and "av_ca" not in k and "a2v" not in k
}
quantized_paths = {k.rsplit(".", 1)[0] for k in video if k.endswith(".scales")}
print(f"video keys {len(video)}, quantized Linears {len(quantized_paths)}")


def _should_quantize(path, module):
    return isinstance(module, nn.Linear) and path in quantized_paths


nn.quantize(model, group_size=64, bits=8, class_predicate=_should_quantize)
model.load_weights(list(video.items()), strict=False)
if not BF16:
    # Pure f32 activations: upcast every non-packed param (dense weights, q/k-norm, scale-shift
    # tables, AND the Q8 scales/biases) to f32 — a lossless bf16→f32 upcast that makes the gate a
    # clean f32 computation (only the packed U32 Q8 weight stays). Matches the Rust F32Q8 path. The
    # bf16 variant keeps the model as loaded (bf16 params + Q8) — the reference's native compute.
    from mlx.utils import tree_map  # noqa: E402

    model.update(
        tree_map(
            lambda p: p.astype(mx.float32) if p.dtype != mx.uint32 else p,
            model.parameters(),
        )
    )
mx.eval(model.parameters())

# Deterministic synthetic inputs in the compute dtype (`ACT`). f32 (default): feeding f32 inputs
# promotes the dense matmuls to f32 and runs quantized_matmul at f32 (the Rust F32Q8 path). bf16:
# native bf16 activations × Q8 (the Rust Bf16Q8 path); the timestep is bf16 with a non-bf16-exact
# sigma so the preprocessor's `timestep × 1000` rounds in bf16 exactly as `denoise_av` does.
mx.random.seed(7)
latent = (mx.random.normal((1, LF * LH * LW, 128)) * 0.5).astype(ACT)
context = (mx.random.normal((1, CTX, DIM)) * 0.5).astype(ACT)
timestep = mx.array([[0.909375 if BF16 else 0.5]], dtype=ACT)  # (1, 1)
positions = create_position_grid(1, LF, LH, LW)  # (1, 3, 32, 2) f32

modality = Modality(
    latent=latent,
    timesteps=timestep,
    positions=positions,
    context=context,
    context_mask=None,
    enabled=True,
)

# Forward, capturing the post-block hidden (tap_h) + embedded timestep (tap_emb_ts) for the output-
# head sanity check; the deep within-block bisection that found the 0.31.0-vs-0.31.2 quantized_matmul
# mismatch lived here and is no longer needed (single block is bit-exact at matched 0.31.2).
args = model.video_args_preprocessor.prepare(modality, None)
emb_ts = args.embedded_timestep
v = args
for block in model.transformer_blocks.values():
    v, _ = block(video=v, audio=None)
h = v.x
vx = model._process_output(
    model.scale_shift_table, model.norm_out, model.proj_out, h, emb_ts
)
mx.eval(vx, h, emb_ts)
print(f"dit: latent{latent.shape} -> velocity{vx.shape} dtype={vx.dtype}")

# Preserve the native dtype (`ACT`) for velocity/tap_h so the bf16 gate checks bf16 bit-exactness
# (the Rust comparison upcasts both sides to f32, lossless). Inputs stay native too.
tensors = {
    "latent": latent,
    "context": context,
    "timestep": timestep,
    "positions": positions.astype(mx.float32),
    "velocity": vx.astype(ACT),
    "tap_h": h.astype(ACT),        # post-48-block hidden (output-head sanity)
    "tap_emb_ts": emb_ts.astype(ACT),
}
name = "ltx_dit_golden_bf16.safetensors" if BF16 else "ltx_dit_golden.safetensors"
out_path = fixture(f"mlx-gen-ltx/tests/fixtures/{name}")
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out_path, tensors, metadata={"S": str(LF * LH * LW), "ctx": str(CTX)})
print(f"wrote {out_path}")
