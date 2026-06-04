"""LTX-2.3 AudioVideo **LoRA** e2e golden (sc-2687) — the production joint denoise WITH a LoRA.

Belt-and-suspenders for the AvDiT LoRA path: applies a full-surface LTX LoRA (video + audio +
cross-modal) to the reference AV `LTXModel` via the forward-time residual
(`mlx_video/lora/apply.py::LoRALinear`, the same strategy the Rust port uses), then runs the
reference `denoise_av` 2-stage joint orchestration → video frames + audio waveform. The Rust
`apply_ltx_adapters` (over `AvDiT`) + `generate_av_latents` + decode must reproduce it
(`tests/av_lora_e2e_parity.rs`).

Inputs (synthetic video/audio conditioning + per-stage noise) are reused verbatim from the committed
**base** AV e2e golden, so this differs from it only by the LoRA — a clean A/B. Runs **f32** (the Rust
`quant_f32` path), the same regime as the base AV e2e gate.

Default LoRA = `Samantha_ltx2.3` (trains video + audio + cross-modal attn/ff/gate — 1632 targets,
exercising the whole AvDiT surface). Override with `LTX_LORA_MULTI=/path/to/lora.safetensors`.

**MUST run with mlx 0.31.2.** Run the base AV e2e dump first.
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg /tmp/mlx312/bin/python tools/dump_ltx_av_e2e_golden.py
    MLX_VIDEO_SRC=~/.cache/uv/archive-v0/DtG1XO51ABFxUGHg /tmp/mlx312/bin/python tools/dump_ltx_av_lora_e2e_golden.py
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

from mlx_video.generate_av import (  # noqa: E402
    DEFAULT_STAGE_1_SIGMAS,
    DEFAULT_STAGE_2_SIGMAS,
    create_audio_position_grid,
    create_video_position_grid,
    denoise_av,
    load_vocoder,
)
from mlx_video.lora.apply import LoRALinear, _normalize_lora_key  # noqa: E402
from mlx_video.lora.loader import load_lora_weights  # noqa: E402
from mlx_video.models.ltx.config import LTXModelConfig, LTXModelType, LTXRopeType  # noqa: E402
from mlx_video.models.ltx.ltx import LTXModel  # noqa: E402
from mlx_video.models.ltx.upsampler import load_upsampler, upsample_latents  # noqa: E402
from mlx_video.models.ltx.video_vae.decoder import load_vae_decoder  # noqa: E402
from mlx_video.models.ltx.audio_vae import AudioDecoder, CausalityAxis, NormType  # noqa: E402

MODEL = Path.home() / "Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8"
LORA = Path(
    os.environ.get(
        "LTX_LORA_MULTI",
        str(
            Path.home()
            / "Library/Application Support/SceneWorks/data/loras/samantha/Samantha_ltx2.3.safetensors"
        ),
    )
)
LF, AF = 2, 9  # 256×256, 9 frames → 2 latent frames; 9 audio frames


def f32(model):
    model.update(
        tree_map(lambda p: p.astype(mx.float32) if p.dtype != mx.uint32 else p, model.parameters())
    )
    mx.eval(model.parameters())


def wrap_loras(root, lora_weights, strength):
    """Wrap each resolved target Linear/QuantizedLinear in a residual `LoRALinear` (the reference's
    forward-time path), counting applied vs skipped — the Rust `apply_ltx_adapters` over `AvDiT`."""
    module_paths = set()
    for name, _ in root.named_modules():
        module_paths.add(name)
        module_paths.add(f"{name}.weight")
    applied, skipped = 0, []
    for lora_key, weights in lora_weights.items():
        norm = _normalize_lora_key(lora_key, module_paths)
        if norm.endswith(".weight"):
            norm = norm[: -len(".weight")]
        parts = norm.split(".")
        parent = root
        try:
            for part in parts[:-1]:
                parent = getattr(parent, part) if not part.isdigit() else parent[int(part)]
            leaf = parts[-1]
            target = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]
        except (AttributeError, IndexError, TypeError):
            skipped.append(lora_key)
            continue
        if isinstance(target, (nn.Linear, nn.QuantizedLinear)):
            wrapped = LoRALinear(target, [(weights, strength)])
            if leaf.isdigit():
                parent[int(leaf)] = wrapped
            else:
                setattr(parent, leaf, wrapped)
            applied += 1
        else:
            skipped.append(lora_key)
    return applied, skipped


config = LTXModelConfig(
    model_type=LTXModelType.AudioVideo, num_attention_heads=32, attention_head_dim=128,
    in_channels=128, out_channels=128, num_layers=48, cross_attention_dim=4096,
    caption_channels=4096, caption_projection_first_linear=False,
    caption_projection_second_linear=False, adaln_embedding_coefficient=9,
    apply_gated_attention=True, audio_num_attention_heads=32, audio_attention_head_dim=64,
    audio_in_channels=128, audio_out_channels=128, audio_cross_attention_dim=2048,
    audio_caption_channels=2048, rope_type=LTXRopeType.SPLIT, double_precision_rope=True,
    positional_embedding_theta=10000.0, positional_embedding_max_pos=[20, 2048, 2048],
    audio_positional_embedding_max_pos=[20], use_middle_indices_grid=True,
    timestep_scale_multiplier=1000,
)
transformer = LTXModel(config)
raw = mx.load(str(MODEL / "transformer.safetensors"))
quantized_paths = {k.rsplit(".", 1)[0] for k in raw if k.endswith(".scales")}
nn.quantize(transformer, group_size=64, bits=8,
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and p in quantized_paths)
transformer.load_weights(list(raw.items()), strict=False)
f32(transformer)
applied, skipped = wrap_loras(transformer, load_lora_weights(LORA), 1.0)
print(f"LoRA {LORA.name}: applied={applied} skipped={len(skipped)}")
mx.eval(transformer.parameters())

upsampler = load_upsampler(str(MODEL), use_unified=True)
f32(upsampler)
vae_decoder = load_vae_decoder(str(MODEL), timestep_conditioning=None, use_unified=True)
mx.eval(vae_decoder.parameters())
audio_decoder = AudioDecoder(ch=128, out_ch=2, ch_mult=(1, 2, 4), num_res_blocks=2,
                             attn_resolutions={8, 16, 32}, resolution=256, z_channels=8,
                             norm_type=NormType.PIXEL, causality_axis=CausalityAxis.HEIGHT,
                             mel_bins=64, mid_block_add_attention=False)
araw = mx.load(str(MODEL / "audio_vae.safetensors"))
audio_decoder.load_weights([(k, v) for k, v in araw.items()], strict=False)
audio_decoder.per_channel_statistics._mean_of_means = araw["per_channel_statistics._mean_of_means"]
audio_decoder.per_channel_statistics._std_of_means = araw["per_channel_statistics._std_of_means"]
f32(audio_decoder)
vocoder = load_vocoder(MODEL, use_unified=True)
f32(vocoder)
audio_sr = int(getattr(vocoder, "output_sampling_rate", getattr(vocoder, "output_sample_rate", 24000)))

# Reuse the base AV e2e golden's injected conditioning + noise (a clean A/B: only the LoRA differs).
base = mx.load(fixture("mlx-gen-ltx/tests/fixtures/ltx_av_e2e_golden.safetensors"))
video_ctx, audio_ctx = base["video_ctx"], base["audio_ctx"]
video_s1, video_s2 = base["video_s1"], base["video_s2"]
audio_s1, audio_s2 = base["audio_s1"], base["audio_s2"]

vpos1 = create_video_position_grid(1, LF, 4, 4)
vpos2 = create_video_position_grid(1, LF, 8, 8)
apos = create_audio_position_grid(1, AF)

S1, S2 = list(DEFAULT_STAGE_1_SIGMAS), list(DEFAULT_STAGE_2_SIGMAS)
vlat, alat = denoise_av(video_s1, audio_s1, vpos1, apos, video_ctx, audio_ctx, None, None,
                        transformer, S1, verbose=False, stage=1, cfg_scale=1.0, use_legacy_euler=True)
vlat = upsample_latents(vlat, upsampler, vae_decoder.latents_mean, vae_decoder.latents_std)
ns = mx.array(S2[0], dtype=mx.float32)
vlat = video_s2 * ns + vlat * (mx.array(1.0, dtype=mx.float32) - ns)
alat = audio_s2 * ns + alat * (mx.array(1.0, dtype=mx.float32) - ns)
vlat, alat = denoise_av(vlat, alat, vpos2, apos, video_ctx, audio_ctx, None, None,
                        transformer, S2, verbose=False, stage=2, cfg_scale=1.0, use_legacy_euler=True)
mx.eval(vlat, alat)

video = vae_decoder(vlat)
video = mx.transpose(mx.squeeze(video, axis=0), (1, 2, 3, 0))
video = (mx.clip((video + 1.0) / 2.0, 0.0, 1.0) * 255).astype(mx.uint8)
mel = audio_decoder(alat)
wav = vocoder(mel)
mx.eval(video, wav)
print(f"av lora e2e: video_latents {vlat.shape} frames {video.shape} | audio_latents {alat.shape} wav {wav.shape}")

tensors = {
    "video_ctx": video_ctx, "audio_ctx": audio_ctx,
    "video_s1": video_s1, "video_s2": video_s2, "audio_s1": audio_s1, "audio_s2": audio_s2,
    "video_latents": vlat.astype(mx.float32), "audio_latents": alat.astype(mx.float32),
    "frames": video, "waveform": wav.astype(mx.float32),
}
out = fixture("mlx-gen-ltx/tests/fixtures/ltx_av_lora_e2e_golden.safetensors")
Path(out).parent.mkdir(parents=True, exist_ok=True)
mx.save_safetensors(out, tensors, metadata={"sr": str(audio_sr), "lora": LORA.name,
                                            "applied": str(applied), "skipped": str(len(skipped))})
print(f"wrote {out}")
