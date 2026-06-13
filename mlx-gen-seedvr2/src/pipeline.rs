//! SeedVR2 image-mode pipeline (sc-4813).
//!
//! Ties the VAE + DiT into the one-step super-resolution path of the mflux reference
//! `SeedVR2.generate_image`: encode the (bicubic-upscaled) LR image → build the conditioning latent
//! (encoded latent + all-ones mask) → concat fresh noise → DiT (one step) → 1-step Euler
//! (`latents = noise − DiT_out`, since `t_norm=1, s=0`) → VAE decode → crop. Color correction is a
//! separate post-process ([`crate::color`]).
//!
//! The negative-prompt conditioning is a precomputed embedding (`pos_emb.safetensors`, no runtime
//! text encoder), so `generate` takes it as an argument; the registry loads it from the snapshot.

use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::ops::{concatenate_axis, subtract};
use mlx_rs::Array;

use crate::config::DitConfig;
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
}

impl Seedvr2Pipeline {
    pub fn from_weights(vae_w: &Weights, dit_w: &Weights, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            vae: Seedvr2Vae::from_weights(vae_w)?,
            transformer: Seedvr2Transformer::from_weights(dit_w, cfg)?,
        })
    }

    /// Encode the preprocessed image to the conditioning latent `(B,16,T',h,w)` (scaled mean).
    pub fn encode(&self, processed: &Array) -> Result<Array> {
        self.vae.encode(processed)
    }

    /// Build the static condition `[latent, ones-mask]` → `(B, 17, T', h, w)`.
    pub fn condition(latent: &Array) -> Result<Array> {
        let sh = latent.shape();
        let mask =
            Array::ones::<f32>(&[sh[0], 1, sh[2], sh[3], sh[4]])?.as_dtype(latent.dtype())?;
        Ok(concatenate_axis(&[latent, &mask], 1)?)
    }

    /// One denoise step: `vid = [noise, condition]` → DiT → `noise − DiT_out`.
    pub fn denoise(
        &self,
        noise: &Array,
        condition: &Array,
        neg_embed: &Array,
        timestep: &Array,
    ) -> Result<Array> {
        let model_input = concatenate_axis(&[noise, condition], 1)?; // (B,33,T',h,w)
        let dit_out = self
            .transformer
            .forward(&model_input, neg_embed, timestep)?;
        Ok(subtract(noise, &dit_out)?)
    }

    /// Decode latents and crop to `(true_h, true_w)` → `(B,3,true_h,true_w)`.
    pub fn decode_crop(&self, latents: &Array, true_h: i32, true_w: i32) -> Result<Array> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        let t0 = decoded.take_axis(Array::from_int(0), 2)?; // first frame -> (B,3,H,W)
        let h_idx = Array::from_slice(&(0..true_h).collect::<Vec<i32>>(), &[true_h]);
        let w_idx = Array::from_slice(&(0..true_w).collect::<Vec<i32>>(), &[true_w]);
        Ok(t0.take_axis(h_idx, 2)?.take_axis(w_idx, 3)?)
    }

    /// Full model path (no color correction): preprocessed image + injected noise → decoded crop.
    pub fn run_model(
        &self,
        processed: &Array,
        noise: &Array,
        neg_embed: &Array,
        timestep: &Array,
        true_h: i32,
        true_w: i32,
    ) -> Result<Array> {
        let latent = self.encode(processed)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(noise, &cond, neg_embed, timestep)?;
        self.decode_crop(&latents, true_h, true_w)
    }
}
