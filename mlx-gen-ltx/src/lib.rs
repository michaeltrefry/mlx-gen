//! # mlx-gen-ltx
//!
//! LTX-2.3 **video** (text-to-video) provider crate for [`mlx-gen`]. Port of the
//! `mlx-video-with-audio` package's LTX video path (`generate_av.py`, `models/ltx/*`,
//! `models/ltx/video_vae/*`) onto Rust + `mlx-rs`.
//!
//! **Scope:** the full **AudioVideo** path (`generate_av.py`, sc-2684) — synchronized audio+video;
//! `generate()` runs the joint dual-modality denoise and returns video frames + an audio track. Built
//! on the sc-2679 video core + **single-image I2V** (sc-2685) + checkpoint-driven **Q4/Q8** quant
//! (sc-2686) + **LoRA in generate** (sc-2687 — forward-time residuals + per-pass strength over the
//! full video+audio+cross-modal surface; see [`adapters`]). LoKr (sc-2393) is a sibling story.
//!
//! This crate self-registers `ltx_2_3` into the `mlx-gen` model registry; load it with
//! `mlx_gen::load("ltx_2_3", spec)`.
//!
//! ## Status (S0–S6 complete)
//! The full text-to-video path is wired and pixel-parity vs the reference `generate_av.py`: Gemma-3
//! tokenizer (byte-exact) → [`LtxTextEncoder`] (Gemma backbone + connector) → seeded noise → the
//! 2-stage distilled denoise ([`pipeline`]: stage-1 half-res → 2× [`upsampler::LatentUpsampler`] →
//! re-noise → stage-2 full-res over the 48-layer [`transformer::LtxDiT`]) → [`vae::LtxVideoVae`]
//! decode → uint8 frames. Built on SPLIT 3-D RoPE (double-precision), an f32 position grid, the
//! distilled sigma schedules, and the legacy dtype-preserving Euler step.
//!
//! The distilled stage-1 sampler is chaos-sensitive, so e2e pixel-parity requires a **bit-exact
//! per-forward DiT** (sc-2842 — the adaLN timestep table must be built in MLX f32, not host f64). Two
//! shipped precisions, both gated bit-exact vs their reference golden: [`transformer::Precision::quant_f32`]
//! (f32 activations × quantized weights — the quality target) and [`transformer::Precision::quant_bf16`]
//! (the reference's native bf16 activations — the production-speed path). The quant geometry (**Q4**/Q8)
//! rides on the checkpoint's `split_model.json` (sc-2686). **I2V** single-image conditioning (sc-2685)
//! is wired into the same 2-stage path ([`conditioning`] + [`pipeline::generate_i2v_latents`], gated
//! bit-exact by `tests/i2v_parity.rs`). **LoRA in generate** (sc-2687) is wired (forward-time residuals
//! + per-pass strength; see [`adapters`]). LoKr (sc-2393) is a sibling.

pub mod adapters;
pub mod audio_vae;
pub mod conditioning;
pub mod config;
pub mod connector;
pub mod convert;
pub mod enhance;
pub mod gemma;
pub mod model;
pub mod pipeline;
pub mod positions;
pub mod rope;
pub mod schedule;
pub mod text_encoder;
pub mod tokenizer;
pub mod transformer;
pub mod upsampler;
pub mod vae;
pub mod vocoder;

pub use adapters::{apply_ltx_adapters, LtxLoraReport};
pub use audio_vae::AudioDecoder;
pub use conditioning::{apply_conditioning, apply_denoise_mask, I2vConditioning};
pub use config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, RopeType, VaeBlock};
pub use connector::Connector;
pub use convert::{convert_and_assemble, LtxConvertOpts};
pub use enhance::{clean_response, EnhanceConfig, SampleParams};
pub use model::{descriptor, load, Ltx, MODEL_ID};
pub use pipeline::{
    decode_audio_track, decode_to_frames, denoise, denoise_av, generate_av_latents,
    generate_i2v_latents, generate_t2v, generate_t2v_latents, preprocess_conditioning_image,
    renoise, to_uint8_frames, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
pub use text_encoder::LtxTextEncoder;
// Tiling moved to `mlx_gen` core (shared with the Wan VAE — sc-2808). Re-export the module + config
// so `mlx_gen_ltx::tiling::*` / `mlx_gen_ltx::TilingConfig` keep resolving for existing callers.
pub use config::{VocoderConfig, VocoderGenConfig};
pub use mlx_gen::tiling::{self, TilingConfig};
pub use tokenizer::LtxTokenizer;
pub use transformer::{to_denoised, AvDiT, LtxDiT, Precision, VideoBlock};
pub use upsampler::{upsample_latents, LatentUpsampler};
pub use vae::LtxVideoVae;
pub use vocoder::{Generator, LtxVocoder, VocoderWithBwe};

use std::sync::atomic::{AtomicBool, Ordering};

/// sc-2963 (rollout of the Wan sc-2957 template): when on, the AvDiT's fusable elementwise *glue* —
/// adaLN affine (`x·(1+scale)+shift`), the gated residuals (`x+out·gate`), the **tanh-GELU FFN
/// activation**, and the split (rotate-halves) RoPE rotation — runs through `mx.compile` so MLX fuses
/// each chain into a single Metal kernel. The big quantized GEMMs / SDPA / `mx.fast` norms stay eager.
///
/// This is the same machine that gave Wan **+14%/step**: at video sequence the FFN GELU runs on a
/// `[B, S, ffn]` tensor of tens-to-hundreds of millions of elements (S up to ~15k at 1280×720), so
/// fusing its ~8 elementwise ops into one kernel is the dominant win. **Bit-exact** to the eager form.
/// **Enabled by the production denoise loops** ([`pipeline::denoise`] / [`pipeline::denoise_av`]);
/// **off by default** so the reference-parity gates run eager. Dtype-preserving (the closures cast
/// nothing) — the f32 / bf16 / quantized compute paths flow through unchanged.
static COMPILE_GLUE: AtomicBool = AtomicBool::new(false);

/// Enable/disable compiled elementwise glue (sc-2963). Process-global; set before the denoise loop.
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.store(on, Ordering::Relaxed);
}

pub(crate) fn compile_glue() -> bool {
    COMPILE_GLUE.load(Ordering::Relaxed)
}
