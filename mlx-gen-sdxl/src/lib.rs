//! # mlx-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`mlx-gen`](mlx_gen). SDXL is a **U-Net**
//! generator (not a DiT like Z-Image/FLUX/Qwen), brought into Rust from Apple's vendored
//! `mlx-examples/stable_diffusion` path (`_vendor/mlx_sd/`, MIT) plus SceneWorks' LoRA merge — the
//! last Python image-inference path (sc-2400, epic 2337).
//!
//! Depends only on the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator`
//! contract, the registry) and self-registers via `inventory` — linking this crate makes
//! `mlx_gen::load("sdxl", …)` resolve. The port reuses the core conv primitives already built for
//! the Z-Image VAE (`conv2d`, pytorch-compatible `group_norm`, `silu`, `upsample_nearest`) and the
//! shared `image`/`weights`/`quant`/`adapters` layers; it adds the SDXL-specific surfaces: the
//! `UNet2DConditionModel` (down/mid/up cross-attention blocks + time/`text_time` micro-conditioning
//! embeddings), the dual CLIP-L + OpenCLIP-bigG text encoders and their CLIP-BPE tokenizer, the
//! SDXL VAE, and the discrete Euler / Euler-Ancestral sampler with real classifier-free guidance.
//!
//! Parity target = the vendored fp16 reference (`StableDiffusionXL.generate_latents`), validated
//! stage-by-stage against goldens (see `tools/dump_sdxl_golden.py`).

pub mod adapters;
pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod sampler;
pub mod text_encoder;
pub mod tokenizer;
pub mod unet;
pub mod vae;

pub use adapters::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, lora_delta, LoraCoverage, SdxlLoraReport,
};
pub use config::{
    BetaSchedule, ClipActivation, ClipTextConfig, DiffusionConfig, UNetConfig, VaeConfig,
};
pub use loader::{
    load_text_encoder_1, load_text_encoder_1_dtype, load_text_encoder_2, load_text_encoder_2_dtype,
    load_tokenizer, load_unet, load_unet_dtype, load_vae,
};
pub use model::{descriptor, load, Sdxl, MODEL_ID};
pub use pipeline::{
    decode_image, decoded_to_image, denoise, encode_conditioning, encode_init_latents,
    preprocess_init_image, seeded_prior, text_time_ids, Denoiser,
};
pub use sampler::EulerSampler;
pub use text_encoder::{ClipOutput, ClipTextEncoder};
pub use tokenizer::{ClipBpeTokenizer, PAD_ID};
pub use unet::UNet2DConditionModel;
pub use vae::Autoencoder;
