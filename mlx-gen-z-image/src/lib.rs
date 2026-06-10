//! # mlx-gen-z-image
//!
//! The **Z-Image** (Tongyi Z-Image-turbo) provider crate for [`mlx-gen`](mlx_gen). Depends only
//! on the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator` contract,
//! the registry) and self-registers via `inventory` — linking this crate makes
//! `mlx_gen::load("z_image_turbo", …)` resolve. See `docs/MODEL_ARCHITECTURE.md`.
//!
//! Ported & parity-proven against the frozen Python mflux fork (tolerance 1e-2 — Metal runs
//! fp32 matmul in reduced precision) and validated end-to-end on real bf16 weights (sc-2352):
//! the Qwen text encoder (prompt → `cap_feats`), the flow-match Euler scheduler, the DiT
//! transformer (block, context block, timestep / RoPE embedders, final layer, full forward),
//! and the VAE encoder + decoder. [`load`](model::load) assembles the model from a snapshot
//! directory and [`ZImageTurbo::generate`](model::ZImageTurbo) runs the full prompt→image
//! pipeline, including img2img (VAE-encode an init image + noise blend, sc-2533) and whole-model
//! Q4/Q8 quantization (sc-2532).

pub mod adapters;
pub mod attention;
pub mod context_block;
pub mod control_transformer;
pub mod control_transformer_block;
pub mod feed_forward;
pub mod final_layer;
pub mod loader;
pub mod model;
pub mod model_control;
pub mod pipeline;
pub mod rope_embedder;
pub mod text_encoder;
pub mod timestep_embedder;
pub mod training;
pub mod transformer;
pub mod transformer_block;
pub mod vae;

pub use adapters::apply_z_image_adapters;
pub use context_block::ZImageContextBlock;
pub use control_transformer::{ZImageControlTransformer, CONTROL_IN_DIM};
pub use control_transformer_block::ZImageControlBlock;
pub use final_layer::FinalLayer;
pub use loader::{
    load_control_transformer, load_text_encoder, load_tokenizer, load_transformer, load_vae,
};
pub use model::{descriptor, load, ZImageTurbo, MODEL_ID};
// The control variant registers itself via `inventory`; its `descriptor`/`load`/`MODEL_ID` clash
// with the base model's, so reach them through the `model_control` module path (consumers use the
// registry id `"z_image_turbo_control"`).
pub use model_control::ZImageTurboControl;
pub use pipeline::{
    add_noise_by_interpolation, create_noise, decoded_to_image, denoise,
    denoise_control_with_progress, denoise_with_progress, encode_control_context,
    encode_init_latents, init_time_step, pack_latents, preprocess_init_image, slice_valid,
    unpack_latents,
};
pub use rope_embedder::RopeEmbedder;
pub use timestep_embedder::TimestepEmbedder;
pub use training::{attention_targets, LoraTarget, ZImageLoraTrainer, ZImageTurboTrainer};
pub use transformer::{ZImageTransformer, ZImageTransformerConfig};
pub use transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};

use std::sync::atomic::{AtomicBool, Ordering};

/// sc-2963 (rollout of the Wan sc-2957 template): when on, the DiT's fusable elementwise *glue* —
/// the SwiGLU FFN activation (`silu(h1)·h3`), the gated residuals (`x+gate·norm(out)`), the complex
/// RoPE rotation, and the control-branch hint injection (`x+hint·scale`) — runs through `mx.compile`
/// so MLX fuses each chain into a single Metal kernel (vs one kernel per primitive op when eager).
/// The big GEMMs / SDPA / `mx.fast` RMSNorms stay eager, and the tiny adaLN scale/gate ops are left
/// eager (no fusion to win). **Bit-exact** to the eager form. **Enabled by the production denoise
/// loops** (turbo + control, [`pipeline`]); left **off by default** so the reference-parity gates run
/// eager. The **mixed-precision dtype flow is preserved** — base bf16, f32 `control_context` (sc-2720):
/// the compiled closures cast nothing the eager form didn't, so dtype flows from inputs unchanged.
static COMPILE_GLUE: AtomicBool = AtomicBool::new(false);

/// Enable/disable compiled elementwise glue (sc-2963). Process-global; set before the denoise loop.
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.store(on, Ordering::Relaxed);
}

pub(crate) fn compile_glue() -> bool {
    COMPILE_GLUE.load(Ordering::Relaxed)
}

/// RAII guard (sc-4036 / F-036) that enables compiled glue for its lifetime and restores the prior
/// `COMPILE_GLUE` value on drop. The production render holds one across the whole count loop so the
/// toggle is set once (not redundantly per image) and — unlike a bare [`set_compile_glue`]`(true)`
/// that leaked the process-global on — code running after `generate` returns (e.g. the
/// reference-parity gates the doc above promises run eager) sees the toggle restored, even on an
/// early `?` return.
#[must_use = "dropping the guard restores the prior compile-glue setting; bind it for the render's lifetime"]
pub(crate) struct CompileGlueGuard {
    prev: bool,
}

impl CompileGlueGuard {
    /// Turn compiled glue on, remembering the prior value to restore on drop.
    pub(crate) fn enable() -> Self {
        Self {
            prev: COMPILE_GLUE.swap(true, Ordering::Relaxed),
        }
    }
}

impl Drop for CompileGlueGuard {
    fn drop(&mut self) {
        COMPILE_GLUE.store(self.prev, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod compile_glue_guard_tests {
    use super::{compile_glue, set_compile_glue, CompileGlueGuard};

    // Single-threaded test runner (`.cargo/config.toml` RUST_TEST_THREADS=1) makes the
    // process-global `COMPILE_GLUE` safe to assert on, matching the existing `set_compile_glue`
    // A/B tests in feed_forward / control_transformer.
    #[test]
    fn guard_enables_then_restores_prior_value() {
        // Prior off → on within scope → restored off on drop (the doc's "eager by default" intent).
        set_compile_glue(false);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue(), "guard enables compiled glue for its scope");
        }
        assert!(!compile_glue(), "guard restores the prior (off) on drop");

        // Restores the *prior* value, not a hardcoded false: prior on stays on after drop.
        set_compile_glue(true);
        {
            let _g = CompileGlueGuard::enable();
            assert!(compile_glue());
        }
        assert!(compile_glue(), "guard restores the prior (on) on drop");

        // Leave the global eager, as the reference-parity gates expect.
        set_compile_glue(false);
    }
}
