//! # mlx-gen-seedvr2
//!
//! The **SeedVR2** provider crate for [`mlx-gen`](mlx_gen) — a native-MLX port of the ByteDance
//! one-step diffusion-transformer super-resolution upscaler (epic 4811), reference = the working
//! MLX implementation in the `mflux-sc2257` frozen fork.
//!
//! SeedVR2 is three pieces, none requiring a runtime text encoder:
//!
//! 1. **DiT** — a dual-stream MMDiT with adaptive **spatiotemporal window attention**
//!    (`window=(T,H,W)=(4,3,3)`, shifted on odd layers), 3D axial RoPE, QK-norm, SwiGLU, and AdaLN
//!    modulation from a sinusoidal timestep embedding. 3B (32 layers, dim 2560) primary; 7B optional.
//! 2. **3D causal video VAE** — `CausalConv3d` (causal temporal padding) encoder/decoder with
//!    `temporal_down/up_blocks=2` (4:1 temporal compression), GroupNorm, per-frame spatial attention.
//! 3. **One-step Euler** schedule + a precomputed negative-prompt embedding (`pos_emb.safetensors`).
//!
//! The model is natively video-capable (5-D `(B,C,T,H,W)` tensors); this crate ships **image-mode
//! parity** first (sc-4813), with video mode (temporal chunking/overlap) as the next slice (sc-4814).
//!
//! ## Status
//! Under construction (sc-4813). See the per-module docs as components land.

pub mod color;
pub mod config;
pub mod convert;
pub mod dit;
pub mod pipeline;
pub mod registry;
pub mod vae;
