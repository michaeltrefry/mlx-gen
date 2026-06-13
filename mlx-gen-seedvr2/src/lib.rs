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
//! The model is natively video-capable (5-D `(B,C,T,H,W)` tensors): image mode is the `T=1`
//! special case. **Image mode** (sc-4813) and **video mode** (sc-4814 — multi-frame 5-D pass +
//! temporal chunking with overlap cross-fade + a memory-budgeted chunk sizer; see [`video`]) both
//! ship through one [`registry::Seedvr2Generator`] (`Modality::Both`), dispatched on the request's
//! conditioning: a `Reference` image → [`GenerationOutput::Images`](mlx_gen::GenerationOutput); a
//! `VideoClip` frame sequence → [`GenerationOutput::Video`](mlx_gen::GenerationOutput).
//!
//! ## Status
//! Image + video engine complete (sc-4813 + sc-4814). HD spatial tiling (composing `VAETiler` with
//! temporal chunking) is a tracked follow-up; the budget sizer + per-frame fallback bound memory
//! across the realistic operating range and refuse over-budget HD catchably.

pub mod color;
pub mod config;
pub mod convert;
pub mod dit;
pub mod pipeline;
pub mod registry;
pub mod vae;
pub mod video;
