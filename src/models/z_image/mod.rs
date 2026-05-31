//! Z-Image (Tongyi Z-Image-turbo) — the first model ported to mlx-gen. v1 covers the DiT
//! transformer block; the embeddings, final layer, scheduler, VAE, and text encoder land in
//! later stories (sc-2345+). The block is parity-proven against the Python mflux fork.

pub mod attention;
pub mod context_block;
pub mod feed_forward;
pub mod final_layer;
pub mod rope_embedder;
pub mod timestep_embedder;
pub mod transformer;
pub mod transformer_block;
pub mod vae;

pub use context_block::ZImageContextBlock;
pub use final_layer::FinalLayer;
pub use rope_embedder::RopeEmbedder;
pub use timestep_embedder::TimestepEmbedder;
pub use transformer::{ZImageTransformer, ZImageTransformerConfig};
pub use transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
