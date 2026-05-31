//! # mlx-gen-z-image
//!
//! The **Z-Image** (Tongyi Z-Image-turbo) provider crate for [`mlx-gen`](mlx_gen). Depends only
//! on the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator` contract,
//! the registry) and self-registers via `inventory` — linking this crate makes
//! `mlx_gen::load("z_image_turbo", …)` resolve. See `docs/MODEL_ARCHITECTURE.md`.
//!
//! Ported & parity-proven against the frozen Python mflux fork (tolerance 1e-2 — Metal runs
//! fp32 matmul in reduced precision): the DiT transformer (block, context block, timestep /
//! RoPE embedders, final layer, full forward) and the VAE decoder. The Qwen text encoder
//! (prompt → `cap_feats`) and the flow-match Euler scheduler are the remaining pieces; until
//! they land, [`ZImageTurbo::generate`](model::ZImageTurbo) reports the pipeline as pending.

pub mod attention;
pub mod context_block;
pub mod feed_forward;
pub mod final_layer;
pub mod model;
pub mod pipeline;
pub mod rope_embedder;
pub mod timestep_embedder;
pub mod transformer;
pub mod transformer_block;
pub mod vae;

pub use context_block::ZImageContextBlock;
pub use final_layer::FinalLayer;
pub use model::{descriptor, load, ZImageTurbo, MODEL_ID};
pub use pipeline::{create_noise, decoded_to_image, unpack_latents};
pub use rope_embedder::RopeEmbedder;
pub use timestep_embedder::TimestepEmbedder;
pub use transformer::{ZImageTransformer, ZImageTransformerConfig};
pub use transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
