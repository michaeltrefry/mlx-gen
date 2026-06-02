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
pub use transformer::{ZImageTransformer, ZImageTransformerConfig};
pub use transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
