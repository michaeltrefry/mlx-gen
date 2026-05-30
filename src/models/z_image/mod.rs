//! Z-Image (Tongyi Z-Image-turbo) — the first model ported to mlx-gen. v1 covers the DiT
//! transformer block; the embeddings, final layer, scheduler, VAE, and text encoder land in
//! later stories (sc-2345+). The block is parity-proven against the Python mflux fork.

pub mod attention;
pub mod feed_forward;
pub mod transformer_block;

pub use transformer_block::{ZImageBlockConfig, ZImageTransformerBlock};
