//! SVD image-conditioning encoder — the `CLIPVisionModelWithProjection` (OpenCLIP ViT-H/14) that
//! turns the input frame into the `image_embeds` fed to the UNet cross-attention.
//!
//! The transformer body is **exactly** `mlx-gen-sdxl`'s [`ClipVisionEncoder`] (ViT-H/14: 1280-wide,
//! 32 layers, 16 heads, patch 14, 224px, `pre_layrnorm`, gelu, LN 1e-5 — the IP-Adapter image tower).
//! SVD only adds the projection head on top of the pooled CLS token: `pooled = post_layernorm(
//! last_hidden_state[:, 0])`, `image_embeds = visual_projection(pooled)` (Linear 1280→1024, no bias) —
//! diffusers `CLIPVisionTransformer` pooling + `CLIPVisionModelWithProjection`.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::matmul;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_gen_sdxl::{ClipVisionEncoder, VisionConfig};

use crate::config::ImageEncoderConfig;

/// CLIP LN epsilon (matches the body + diffusers `layer_norm_eps`).
const LN_EPS: f32 = 1e-5;

/// The SVD image encoder: the reused ViT-H body + the projection head.
pub struct SvdImageEncoder {
    body: ClipVisionEncoder,
    post_ln_w: Array,
    post_ln_b: Array,
    /// `visual_projection.weight` `[projection_dim, hidden]` (no bias).
    visual_projection: Array,
}

impl SvdImageEncoder {
    /// Load from the SVD `image_encoder/model.safetensors` (`vision_model.*` body +
    /// `vision_model.post_layernorm.*` + top-level `visual_projection.weight`).
    pub fn from_weights(w: &Weights, _cfg: &ImageEncoderConfig) -> Result<Self> {
        let body = ClipVisionEncoder::from_weights(w, &VisionConfig::vit_h_14())?;
        Ok(Self {
            body,
            post_ln_w: w.require("vision_model.post_layernorm.weight")?.clone(),
            post_ln_b: w.require("vision_model.post_layernorm.bias")?.clone(),
            visual_projection: w.require("visual_projection.weight")?.clone(),
        })
    }

    /// `pixel_values` NHWC `[B, 224, 224, 3]` (CLIP-normalized) → `image_embeds` `[B, projection_dim]`.
    /// Mirrors diffusers `self.image_encoder(image).image_embeds`: run the tower → take the CLS token
    /// of the last hidden state → `post_layernorm` → `visual_projection`.
    pub fn image_embeds(&self, pixel_values: &Array) -> Result<Array> {
        let states = self.body.hidden_states(pixel_values)?;
        let last = states.last().expect("hidden_states non-empty"); // [B, 257, hidden]
        let cls = last.take_axis(Array::from_int(0), 1)?; // [B, hidden] (CLS token, axis dropped)
        let pooled = layer_norm(&cls, Some(&self.post_ln_w), Some(&self.post_ln_b), LN_EPS)?;
        // visual_projection is a bias-free Linear with weight [proj, hidden] → embeds = pooled · Wᵀ.
        Ok(matmul(
            &pooled,
            &self.visual_projection.transpose_axes(&[1, 0])?,
        )?)
    }
}
