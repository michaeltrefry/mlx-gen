//! # mlx-gen-qwen-image
//!
//! The **Qwen-Image** (+ Qwen-Image-Edit) provider crate for [`mlx-gen`](mlx_gen). Depends only on
//! the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator` contract, the
//! registry) and — once the model lands — self-registers via `inventory` so that
//! `mlx_gen::load("qwen_image", …)` resolves. See `docs/MODEL_ARCHITECTURE.md`.
//!
//! Ported from the frozen Python mflux fork (`~/repos/mflux/src/mflux/models/qwen/`). The
//! Qwen-Image port lands slice-by-slice (sc-2348): the causal-Conv3d VAE, the Qwen2.5-VL text
//! encoder, the 60-layer dual-stream MMDiT, then the T2I pipeline; Qwen-Image-Edit (vision
//! transformer + reference conditioning) follows in sc-2465.
//!
//! Currently shipped: the **Qwen2-VL image processor** (sc-2341), relocated here from core as the
//! first slice of the port — Qwen-Image-Edit's reference-image preprocessing.

pub mod image_processor;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use image_processor::{ImageInput, ProcessedImage, QwenImageProcessor};
pub use loader::{load_text_encoder, load_tokenizer, load_transformer, load_vae};
pub use model::{descriptor, load, QwenImage, MODEL_ID};
pub use pipeline::{
    compute_guided_noise, create_noise, decoded_to_image, denoise_with_progress, qwen_scheduler,
    unpack_latents,
};
pub use text_encoder::{QwenTextEncoder, QwenTextEncoderConfig};
pub use transformer::{QwenTransformer, QwenTransformerConfig};
pub use vae::QwenVae;
