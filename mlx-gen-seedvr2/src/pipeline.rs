//! SeedVR2 image-mode pipeline (sc-4813).
//!
//! Ties the VAE + DiT into the one-step super-resolution path of the mflux reference
//! `SeedVR2.generate_image`: preprocess the LR image (PIL-bicubic upscale to target, optional
//! `softness` pre-blur, [-1,1]) → VAE encode → conditioning latent (encoded latent + ones-mask) →
//! concat fresh noise → DiT (one step) → 1-step Euler (`latents = noise − DiT_out`) → VAE decode →
//! crop → LAB+wavelet color correction ([`crate::color`]) → RGB8.
//!
//! The negative-prompt conditioning is a precomputed embedding (`pos_emb.safetensors`, no runtime
//! text encoder), bundled in the crate (`data/neg_embed.safetensors`) and loaded at construction.

use mlx_gen::image::{decoded_to_image, resize_bicubic_u8};
use mlx_gen::weights::Weights;
use mlx_gen::{Image, Result};
use mlx_rs::ops::{concatenate_axis, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use crate::config::DitConfig;
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::{color, convert};

/// The 1-step Euler timestep (= `num_train_steps`, which the SeedVR2 scheduler defaults to 1000).
const TIMESTEP: f32 = 1000.0;

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Option<Array>,
    dtype: Dtype,
}

/// Cast every tensor in `w` to `dt`.
fn cast_weights(w: &Weights, dt: Dtype) -> Result<Weights> {
    let mut out = Weights::empty();
    for k in w.keys().map(String::from).collect::<Vec<_>>() {
        out.insert(k.clone(), w.require(&k)?.as_dtype(dt)?);
    }
    Ok(out)
}

/// The bundled precomputed negative-prompt embedding → `(1, 58, 5120)` at `dt`.
fn load_neg_embed(dt: Dtype) -> Result<Array> {
    const BYTES: &[u8] = include_bytes!("../data/neg_embed.safetensors");
    let path = std::env::temp_dir().join("mlx_gen_seedvr2_neg_embed.safetensors");
    if !path.exists() {
        std::fs::write(&path, BYTES)?;
    }
    let w = Weights::from_file(&path)?;
    Ok(w.require("embedding")?.as_dtype(dt)?.expand_dims(0)?)
}

impl Seedvr2Pipeline {
    /// Build from already-converted (MLX-layout) VAE + DiT weights. Used by the parity tests with an
    /// injected neg-embed; `generate` is unavailable until [`Self::load`] sets the bundled embed.
    pub fn from_weights(vae_w: &Weights, dit_w: &Weights, cfg: &DitConfig) -> Result<Self> {
        Ok(Self {
            vae: Seedvr2Vae::from_weights(vae_w)?,
            transformer: Seedvr2Transformer::from_weights(dit_w, cfg)?,
            neg_embed: None,
            dtype: Dtype::Float32,
        })
    }

    /// Load from a raw `numz/SeedVR2_comfyUI` checkpoint dir: convert in-memory (no Python), cast to
    /// `dt`, and attach the bundled neg-embed. `dit_file` selects 3B/7B.
    pub fn load(
        raw_dir: impl AsRef<std::path::Path>,
        dit_file: &str,
        cfg: &DitConfig,
        dt: Dtype,
    ) -> Result<Self> {
        let dir = raw_dir.as_ref();
        let vae_w = cast_weights(
            &convert::convert_vae(&Weights::from_file(dir.join("ema_vae_fp16.safetensors"))?)?,
            dt,
        )?;
        let dit_w = cast_weights(
            &convert::convert_dit(&Weights::from_file(dir.join(dit_file))?)?,
            dt,
        )?;
        let mut p = Self::from_weights(&vae_w, &dit_w, cfg)?;
        p.neg_embed = Some(load_neg_embed(dt)?);
        p.dtype = dt;
        Ok(p)
    }

    /// The bundled negative-prompt embedding `(1,58,5120)` (set by [`Self::load`]).
    pub fn neg_embed(&self) -> Option<&Array> {
        self.neg_embed.as_ref()
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

    /// End-to-end upscale: LR `image` → `(width, height)` super-resolved RGB8 image.
    ///
    /// `softness` (0..1) pre-blurs the input by round-tripping through a `1 + 7·softness`× smaller
    /// size (the reference `--softness`). Both dims must be multiples of 16 (the registry validates).
    pub fn generate(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
    ) -> Result<Image> {
        let neg = self
            .neg_embed
            .as_ref()
            .expect("neg-embed (use Seedvr2Pipeline::load)");
        let processed = self.preprocess(image, width, height, softness)?; // (1,3,H,W) in dtype

        let latent = self.encode(&processed)?;
        let sh = latent.shape();
        let key = random::key(seed)?;
        let noise = random::normal::<f32>(&[1, 16, sh[2], sh[3], sh[4]], None, None, Some(&key))?
            .as_dtype(self.dtype)?;
        let cond = Self::condition(&latent)?;
        let latents = self.denoise(&noise, &cond, neg, &Array::from_f32(TIMESTEP))?;
        let decoded = self.decode_crop(&latents, height, width)?; // (1,3,H,W)

        // color correction uses the bicubic-upscaled LR (the "style") at the same crop.
        let style = processed
            .take_axis(
                Array::from_slice(&(0..height).collect::<Vec<i32>>(), &[height]),
                2,
            )?
            .take_axis(
                Array::from_slice(&(0..width).collect::<Vec<i32>>(), &[width]),
                3,
            )?;
        let corrected = color::apply_color_correction(
            &decoded.as_dtype(Dtype::Float32)?,
            &style.as_dtype(Dtype::Float32)?,
            0.8,
        )?;
        decoded_to_image(&corrected)
    }

    /// LR `Image` → `(1,3,height,width)` in `[-1,1]` at the model dtype. PIL-exact bicubic resize to
    /// the target; optional `softness` pre-blur via a smaller round-trip.
    fn preprocess(&self, image: &Image, width: i32, height: i32, softness: f32) -> Result<Array> {
        let (ih, iw) = (image.height as usize, image.width as usize);
        let (oh, ow) = (height as usize, width as usize);
        let resized: Vec<f32> = if softness > 0.0 {
            let factor = 1.0 + softness.clamp(0.0, 1.0) * 7.0;
            let dw = ((width as f32 / factor) as usize).max(2);
            let dh = ((height as f32 / factor) as usize).max(2);
            let down = resize_bicubic_u8(&image.pixels, ih, iw, dh, dw); // f32 [0,255]
            let down_u8: Vec<u8> = down
                .iter()
                .map(|&v| v.round().clamp(0.0, 255.0) as u8)
                .collect();
            resize_bicubic_u8(&down_u8, dh, dw, oh, ow)
        } else {
            resize_bicubic_u8(&image.pixels, ih, iw, oh, ow)
        };
        // HWC [0,255] f32 → [-1,1] → (1,3,H,W)
        let arr = Array::from_slice(&resized, &[height, width, 3]);
        let arr = subtract(
            &multiply(&arr, Array::from_f32(2.0 / 255.0))?,
            Array::from_f32(1.0),
        )?;
        Ok(arr
            .transpose_axes(&[2, 0, 1])?
            .expand_dims(0)?
            .as_dtype(self.dtype)?)
    }
}
