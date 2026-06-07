//! # mlx-gen-svd
//!
//! Stable Video Diffusion (img2vid-xt) image-to-video provider for mlx-gen (epic 3040, sc-3054).
//! A from-arch port of `stabilityai/stable-video-diffusion-img2vid-xt`:
//! `UNetSpatioTemporalConditionModel` + `AutoencoderKLTemporalDecoder` + the ViT-H
//! `CLIPVisionModelWithProjection` image encoder + the EDM `EulerDiscreteScheduler`, wired through the
//! epic-3018 video runtime (frames → mp4 by the consuming app).
//!
//! Built as slices (mirroring the SDXL port): **S0** config + EDM scheduler (this commit); S1 VAE
//! (2D encoder reuse + temporal decoder); S2 image encoder; S3 UNet; S4 pipeline + provider + e2e
//! parity vs diffusers `StableVideoDiffusionPipeline`. Reuses `mlx-gen-sdxl`'s 2D VAE encoder +
//! CLIP-vision encoder + conv/attn patterns where the spatial parts match.

pub mod config;
pub mod image_encoder;
pub mod scheduler;
pub mod vae;

pub use config::{ImageEncoderConfig, SchedulerConfig, UnetConfig, VaeConfig};
pub use image_encoder::SvdImageEncoder;
pub use scheduler::{euler_step, scale_model_input, v_pred_denoised, EdmSchedule};
pub use vae::SvdVae;
