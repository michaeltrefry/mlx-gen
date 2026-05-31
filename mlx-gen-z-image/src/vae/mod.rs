//! Z-Image VAE decoder. The image side of the pipeline — latents → RGB. Built from the core
//! convolutional primitives in [`mlx_gen::nn`] (`conv2d` + pytorch-compatible `group_norm` +
//! `upsample_nearest`), validated against the fork in the sub-module parity tests.
//!
//! Modules take/return NCHW (mirroring the fork's per-module transpose convention) and work
//! in NHWC internally, since mlx convs/norms are channels-last.

pub mod attention;
pub mod conv_layers;
pub mod decoder;
pub mod mid_block;
pub mod resnet_block;
pub mod up_decoder_block;
pub mod up_sampler;

pub use attention::VaeAttention;
pub use decoder::{Decoder, Vae, VaeDecoderConfig};
pub use resnet_block::ResnetBlock2D;
pub use up_sampler::UpSampler;
