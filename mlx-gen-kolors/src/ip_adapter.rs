//! Kolors IP-Adapter-Plus (sc-3098) — reference-image conditioning, reusing epic 3041's SDXL
//! IP-Adapter primitive. `Kwai-Kolors/Kolors-IP-Adapter-Plus` is the standard IP-Adapter "plus"
//! stack with two Kolors-specific deltas, both expressible as config:
//!
//!  - the image tower is **CLIP-ViT-L/14-336** (1024-d, 336px → 577 tokens), not the ViT-H the SDXL
//!    IP-Adapter uses — [`VisionConfig::vit_l_14_336`];
//!  - the "plus" [`Resampler`] works at width **2048** (latents `[1,16,2048]`, inner 768), projecting
//!    the 1024-d penultimate → 16×2048 image tokens — [`ResamplerConfig::kolors_plus`].
//!
//! The decoupled cross-attention is identical to SDXL: 70 `ip_adapter.{n}.to_k_ip/to_v_ip` pairs
//! (the IP tokens are 2048-d = the U-Net cross-attention width), installed into the U-Net and added
//! at `ip_scale` alongside the (encoder_hid_proj-projected) ChatGLM3 text path. So this module is a
//! thin loader over the SDXL primitive; the denoise wiring lives on [`crate::Kolors`].

use std::path::Path;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use mlx_gen_sdxl::{
    load_ip_kv_pairs, ClipVisionEncoder, IpImageEncoder, Resampler, ResamplerConfig, VisionConfig,
};

/// The Kolors ViT-L/14-336 CLIP crop size (the IP-Adapter image tower).
pub const KOLORS_IP_IMAGE_SIZE: usize = 336;

/// Load the Kolors IP-Adapter-Plus from a `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot dir:
/// the `image_encoder/` (CLIP-ViT-L/14-336) + `ip_adapter_plus_general.safetensors` (the
/// `image_proj` Resampler + the 70 `ip_adapter.{n}.to_k_ip/to_v_ip` decoupled-attn pairs). Returns
/// the [`IpImageEncoder`] (reference image → 16×2048 tokens) and the K/V pairs to install into the
/// Kolors U-Net via [`crate::Kolors::install_ip_adapter`]. Cast to `dtype`.
pub fn load_kolors_ip_adapter(
    snapshot: &Path,
    dtype: Dtype,
) -> Result<(IpImageEncoder, Vec<(Array, Array)>)> {
    let mut enc_w = Weights::from_file(snapshot.join("image_encoder/model.safetensors"))?;
    enc_w.cast_all(dtype)?;
    let encoder = ClipVisionEncoder::from_weights(&enc_w, &VisionConfig::vit_l_14_336())?;

    let mut ip_w = Weights::from_file(snapshot.join("ip_adapter_plus_general.safetensors"))?;
    ip_w.cast_all(dtype)?;
    let resampler = Resampler::from_weights(&ip_w, "image_proj", &ResamplerConfig::kolors_plus())?;
    let pairs = load_ip_kv_pairs(&ip_w)?;

    Ok((
        IpImageEncoder::with_image_size(encoder, resampler, KOLORS_IP_IMAGE_SIZE),
        pairs,
    ))
}
