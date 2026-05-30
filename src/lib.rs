//! # mlx-gen
//!
//! Rust-native inference for generative **image and video** models on Apple
//! [MLX](https://github.com/ml-explore/mlx), built on top of `mlx-rs`.
//!
//! **Status: name reserved / work in progress — not yet usable.**
//!
//! Planned families: FLUX / FLUX.2, Qwen-Image, Z-Image (image); Wan2.2, LTX
//! (video). Adapters: LoRA, LoKr (with stacking), ControlNet.

/// Crate-wide result type. Refined into a typed error enum in a later story.
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub mod adapters;
pub mod models;
pub mod quant;
pub mod weights;
