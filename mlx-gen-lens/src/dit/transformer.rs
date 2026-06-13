//! The top-level Lens DiT (`LensTransformer2DModel`): multi-layer text front-end → `img_in` +
//! timestep embedding → 48 dual-stream blocks → `AdaLayerNormContinuous` + `proj_out` back to patch
//! space. Image-stream output only (the text stream is discarded after the last block).

use mlx_rs::fast::{layer_norm, rms_norm};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::rope::LensRope3d;
use super::{join, load_weight, LensTransformerBlock, Linear};

/// The Lens-Turbo / Lens `transformer/config.json` values.
#[derive(Clone, Copy, Debug)]
pub struct LensDitConfig {
    pub patch_size: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: usize,
    pub num_heads: i32,
    pub head_dim: i32,
    pub inner_dim: i32,
    pub enc_hidden_dim: i32,
    pub axes_dims_rope: [i32; 3],
    pub num_text_layers: usize,
}

impl LensDitConfig {
    pub fn lens() -> Self {
        Self {
            patch_size: 2,
            in_channels: 128,
            out_channels: 32,
            num_layers: 48,
            num_heads: 24,
            head_dim: 64,
            inner_dim: 1536,
            enc_hidden_dim: 2880,
            axes_dims_rope: [8, 28, 28],
            num_text_layers: 4, // selected_layer_index = (5, 11, 17, 23)
        }
    }
}

/// Sinusoidal timestep projection (`Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0,
/// scale=1000)`): `[B] → [B, 256]` as `[cos | sin]`.
fn timestep_proj(timesteps: &Array) -> Result<Array> {
    let (proj_dim, scale, max_period) = (256usize, 1000f32, 10000f32);
    let half = proj_dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|k| (-(max_period.ln()) * k as f32 / half as f32).exp())
        .collect();
    let freq = Array::from_slice(&freqs, &[1, half as i32]);
    let b = timesteps.shape()[0];
    let emb = multiply(&timesteps.reshape(&[b, 1])?, &freq)?;
    let emb = multiply(&emb, Array::from_slice(&[scale], &[1]))?;
    Ok(concatenate_axis(&[&emb.cos()?, &emb.sin()?], 1)?) // flip_sin_to_cos → [cos, sin]
}

/// `AdaLayerNormContinuous`: affine-less LayerNorm scaled/shifted by `linear(silu(temb))` (the Lens
/// checkpoint's `norm_out.linear` **carries a bias** the reference uses). `[scale | shift]` →
/// `norm(x)·(1+scale) + shift`.
struct AdaLayerNormContinuous {
    linear: Linear,
}

impl AdaLayerNormContinuous {
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            linear: Linear::load(w, &join(prefix, "linear"), true, dtype)?,
        })
    }

    fn forward(&self, x: &Array, temb: &Array) -> Result<Array> {
        let mod_params = self.linear.forward(&silu(temb)?)?; // [B, 2·H]
        let parts = split(&mod_params, 2, 1)?; // scale, shift
        let one = Array::from_slice(&[1.0f32], &[1]);
        let scale = add(&parts[0], &one)?.expand_dims(1)?; // [B, 1, H]
        let shift = parts[1].expand_dims(1)?;
        let normed = layer_norm(x, None, None, 1e-6)?;
        Ok(add(&multiply(&normed, &scale)?, &shift)?)
    }
}

/// The Lens denoising DiT.
pub struct LensTransformer {
    img_in: Linear,
    txt_norm: Vec<Array>, // per-layer RMSNorm weights (eps 1e-5)
    txt_in: Linear,
    time_linear_1: Linear,
    time_linear_2: Linear,
    rope: LensRope3d,
    blocks: Vec<LensTransformerBlock>,
    norm_out: AdaLayerNormContinuous,
    proj_out: Linear,
    cfg: LensDitConfig,
    dtype: Dtype,
}

impl LensTransformer {
    /// Load from a diffusers `transformer/` weight set at `dtype` (bf16 production / f32 gate).
    pub fn from_weights(w: &Weights, cfg: &LensDitConfig, dtype: Dtype) -> Result<Self> {
        let mut txt_norm = Vec::with_capacity(cfg.num_text_layers);
        for i in 0..cfg.num_text_layers {
            txt_norm.push(load_weight(w, &format!("txt_norm.{i}"), dtype)?);
        }
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(LensTransformerBlock::from_weights(
                w,
                &format!("transformer_blocks.{i}"),
                cfg.num_heads,
                cfg.head_dim,
                dtype,
            )?);
        }
        Ok(Self {
            img_in: Linear::load(w, "img_in", true, dtype)?,
            txt_norm,
            txt_in: Linear::load(w, "txt_in", true, dtype)?,
            time_linear_1: Linear::load(
                w,
                "time_text_embed.timestep_embedder.linear_1",
                true,
                dtype,
            )?,
            time_linear_2: Linear::load(
                w,
                "time_text_embed.timestep_embedder.linear_2",
                true,
                dtype,
            )?,
            rope: LensRope3d::new(10000.0, cfg.axes_dims_rope),
            blocks,
            norm_out: AdaLayerNormContinuous::from_weights(w, "norm_out", dtype)?,
            proj_out: Linear::load(w, "proj_out", true, dtype)?,
            cfg: *cfg,
            dtype,
        })
    }

    /// `temb = linear_2(silu(linear_1(proj(t))))`, `[B] → [B, inner]`.
    fn time_embed(&self, timestep: &Array) -> Result<Array> {
        let proj = timestep_proj(timestep)?.as_dtype(self.dtype)?;
        let x = silu(&self.time_linear_1.forward(&proj)?)?;
        self.time_linear_2.forward(&x)
    }

    /// Forward.
    ///
    /// - `hidden_states`: `[B, img_len, in_channels]` patchified image latents (`img_len = frame·h·w`).
    /// - `text_feats`: the `num_text_layers` captured gpt-oss layers, each `[B, txt_len, enc_hidden_dim]`.
    /// - `text_valid`: optional `[B, txt_len]` (1 = valid) → additive joint attention mask; `None` =
    ///   all text valid (no padding), the single-prompt path.
    /// - `timestep`: `[B]` in `[0, 1]`.
    /// - `(frame, h, w)`: the latent grid shape (`img_len = frame·h·w`).
    ///
    /// Returns `[B, img_len, patch²·out_channels]` (= 128) patch-space velocity.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,
        text_feats: &[Array],
        text_valid: Option<&Array>,
        timestep: &Array,
        frame: usize,
        h: usize,
        w: usize,
    ) -> Result<Array> {
        assert_eq!(
            text_feats.len(),
            self.cfg.num_text_layers,
            "expected {} text-feature layers, got {}",
            self.cfg.num_text_layers,
            text_feats.len()
        );
        let (b, img_len) = (hidden_states.shape()[0], hidden_states.shape()[1]);
        let txt_len = text_feats[0].shape()[1];

        let mut hidden = self.img_in.forward(hidden_states)?;

        // Multi-layer text front-end: per-layer RMSNorm (eps 1e-5) → channel-concat → txt_in.
        let mut normed = Vec::with_capacity(self.cfg.num_text_layers);
        for (i, feat) in text_feats.iter().enumerate() {
            normed.push(rms_norm(feat, &self.txt_norm[i], 1e-5)?);
        }
        let normed_refs: Vec<&Array> = normed.iter().collect();
        let mut enc = self.txt_in.forward(&concatenate_axis(&normed_refs, -1)?)?;

        let temb = self.time_embed(&timestep.as_dtype(self.dtype)?)?;
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward(frame, h, w, txt_len as usize)?;

        let mask = match text_valid {
            Some(valid) => Some(build_joint_mask(valid, img_len, b, self.dtype)?),
            None => None,
        };

        for block in &self.blocks {
            let (e, hs) = block.forward(
                &hidden,
                &enc,
                &temb,
                &img_cos,
                &img_sin,
                &txt_cos,
                &txt_sin,
                mask.as_ref(),
            )?;
            enc = e;
            hidden = hs;
        }

        let hidden = self.norm_out.forward(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

/// Additive joint attention mask `[B, 1, 1, img_len + txt_len]`: image tokens always valid; text
/// positions follow `text_valid` (1 = valid). Padded positions get a large-negative additive term so
/// SDPA's softmax masks them out (`(valid − 1)·BIG`, valid → 0).
fn build_joint_mask(text_valid: &Array, img_len: i32, b: i32, dtype: Dtype) -> Result<Array> {
    let txt_len = text_valid.shape()[1];
    let img_ones = mlx_rs::ops::ones::<f32>(&[b, img_len])?;
    let valid = concatenate_axis(&[&img_ones, &text_valid.as_dtype(Dtype::Float32)?], 1)?;
    let one = Array::from_slice(&[1.0f32], &[1]);
    let big = Array::from_slice(&[1e9f32], &[1]);
    let additive = multiply(&subtract(&valid, &one)?, &big)?; // valid→0, invalid→ -1e9
    Ok(additive
        .reshape(&[b, 1, 1, img_len + txt_len])?
        .as_dtype(dtype)?)
}
