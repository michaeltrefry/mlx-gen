//! # mlx-gen
//!
//! Rust-native inference for generative **image and video** models on Apple
//! [MLX](https://github.com/ml-explore/mlx), built on top of `mlx-rs`.
//!
//! **Status: name reserved / work in progress — not yet usable.**
//!
//! Planned families: FLUX / FLUX.2, Qwen-Image, Z-Image (image); Wan2.2, LTX
//! (video). Adapters: LoRA, LoKr (with stacking), ControlNet.
//!
//! Architecture: a *disciplined hybrid* of the frozen Python mflux fork — see
//! [`ARCHITECTURE.md`](https://github.com/michaeltrefry/mlx-gen/blob/main/ARCHITECTURE.md).

pub mod adapters;
pub mod error;
pub mod models;
pub mod quant;
pub mod tokenizer;
pub mod weights;

pub use error::{Error, Result};
