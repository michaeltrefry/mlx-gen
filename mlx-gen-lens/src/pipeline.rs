//! The end-to-end Lens-Turbo / Lens T2I pipeline (sc-3173) — wires the four ported components into a
//! single `generate`: [`LensTokenizer`](crate::text::LensTokenizer) → the gpt-oss
//! [`LensTextEncoder`](crate::text_encoder::encoder::LensTextEncoder) (multi-layer capture + the
//! `txt_offset = 97` slice) → the [`LensTransformer`](crate::dit::LensTransformer) denoising DiT (with
//! the [`schedule`](crate::schedule) flow-match sigmas + norm-rescaled CFG) → the Flux.2
//! [`vae::decode`](crate::vae) shim.
//!
//! A faithful port of `_vendor/lens/pipeline.py::LensPipeline.__call__`. The two model variants
//! (`lens`, `lens_turbo`) share this code and arch — they differ only in their sampling defaults
//! (registered in [`crate::registry`]).
//!
//! ## Parity-critical details (from the reference `__call__`)
//! - **Encode → offset slice → align.** Positives and negatives are each encoded to the four captured
//!   gpt-oss layers, sliced at `input_ids[97:]`, then zero-padded to a shared `S_txt` and stacked
//!   `[pos; neg]` along the batch axis for the joint CFG forward. An **empty** negative is the
//!   unconditional branch: zero text features + an all-`false` mask (no text tokens), *not* a second
//!   encode (`encode_prompt`).
//! - **Joint CFG batch.** Each step runs the DiT once over `B·2` (here `B = 1`): `hidden = [x; x]`,
//!   `encoder_features = [pos; neg]`. The output splits `cond, uncond`; the per-step guidance is the
//!   **norm-rescaled** CFG ([`schedule::cfg_rescale`]).
//! - **Timestep.** The transformer is fed the *shifted sigma* directly (the reference `timestep /
//!   1000`, where `scheduler.timesteps = sigma · 1000`) — i.e. [`schedule::timesteps`].
//! - **Latents.** `[B, latent_h · latent_w, 128]`, `latent_{h,w} = {height,width} / 16`; the denoise
//!   is the core flow-match Euler step.

use mlx_rs::ops::{concatenate_axis, split, split_sections};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, Error, Image, Result};
use mlx_gen_flux2::{load_vae, Flux2Vae};

use crate::config::GptOssConfig;
use crate::dit::{LensDitConfig, LensTransformer};
use crate::schedule::{self, cfg_rescale, lens_schedule};
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::encoder::LensTextEncoder;
use crate::vae;

/// The VAE downsample factor (`vae_scale_factor`): a Lens latent cell maps to a 16×16 pixel tile
/// (Flux.2's 8× conv VAE composed with the 2× DiT patchify).
pub const VAE_SCALE_FACTOR: u32 = 16;

/// Default harmony-preamble date (`Current date:`). The preamble is the first [`TXT_OFFSET`] tokens,
/// which are **sliced off** before the DiT conditioning, so its `date` line never reaches the image
/// path — a fixed constant keeps generation deterministic regardless of wall-clock. (The Python
/// worker passes the live date; the value is image-irrelevant.)
pub const DEFAULT_DATE: &str = "2025-01-01";

/// Options for a single [`LensPipeline::generate`] call.
pub struct GenerateOptions<'a> {
    pub prompt: &'a str,
    /// Empty ⇒ the unconditional branch (zero text features), matching the reference default `""`.
    pub negative_prompt: &'a str,
    /// Output pixels — both must be divisible by [`VAE_SCALE_FACTOR`] (use [`crate::resolution`]).
    pub height: u32,
    pub width: u32,
    pub num_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    /// Harmony-preamble `Current date:` (image-irrelevant; see [`DEFAULT_DATE`]).
    pub date: &'a str,
}

/// A loaded Lens pipeline: the four components, shared by both variants.
pub struct LensPipeline {
    tokenizer: LensTokenizer,
    encoder: LensTextEncoder,
    transformer: LensTransformer,
    vae: Flux2Vae,
    num_text_layers: usize,
    dtype: Dtype,
}

impl LensPipeline {
    /// Load all four components from a `microsoft/Lens-Turbo` (or `microsoft/Lens`) snapshot directory
    /// at `dtype` (bf16 production / f32 tight-gate). The snapshot is the diffusers multi-component
    /// tree: `tokenizer/tokenizer.json`, `text_encoder/`, `transformer/`, `vae/`. The VAE always runs
    /// f32 internally (the shared Flux.2 decoder).
    pub fn load(snapshot_dir: impl AsRef<std::path::Path>, dtype: Dtype) -> Result<Self> {
        let root = snapshot_dir.as_ref();
        let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;

        let enc_cfg = GptOssConfig::lens();
        let enc_w = Weights::from_dir(root.join("text_encoder"))?;
        let encoder = LensTextEncoder::from_weights(&enc_w, &enc_cfg, dtype)?;

        let dit_cfg = LensDitConfig::lens();
        let dit_w = Weights::from_dir(root.join("transformer"))?;
        let transformer = LensTransformer::from_weights(&dit_w, &dit_cfg, dtype)?;

        let vae = load_vae(root)?;

        Ok(Self {
            tokenizer,
            encoder,
            transformer,
            vae,
            num_text_layers: dit_cfg.num_text_layers,
            dtype,
        })
    }

    /// Encode one prompt to its per-layer DiT text features (sliced at [`TXT_OFFSET`]) + the valid
    /// mask. Returns `(features, mask)` where `features` is `num_text_layers × [1, S, 2880]` and
    /// `mask` is `[1, S]` (all-`1`; a single prompt is unpadded). When the rendered prompt is ≤ the
    /// offset (never, for real prompts) the features collapse to length 0 (`_get_text_embeddings`).
    fn encode_one(&self, prompt: &str, date: &str) -> Result<(Vec<Array>, Array)> {
        let out = self.tokenizer.encode(prompt, date)?;
        let l = out.ids.len();
        let input_ids = Array::from_slice(&out.ids, &[1, l as i32]);
        let layers = self.encoder.encode(&input_ids)?; // num_text_layers × [1, L, 2880]

        let offset = TXT_OFFSET as i32;
        if l as i32 > offset {
            let s = l as i32 - offset;
            // `[:, offset:, :]` — split at `offset` along the sequence axis, keep the tail.
            let features = layers
                .iter()
                .map(|f| Ok(split_sections(f, &[offset], 1)?[1].clone()))
                .collect::<Result<Vec<_>>>()?;
            // Single unpadded prompt ⇒ every retained token is valid.
            let mask = mlx_rs::ops::ones::<f32>(&[1, s])?;
            Ok((features, mask))
        } else {
            // `input_ids` shorter than the offset (never for a real prompt): length-0 features.
            let dim = layers[0].shape()[2];
            let features = (0..self.num_text_layers)
                .map(|_| Ok(mlx_rs::ops::zeros::<f32>(&[1, 0, dim])?.as_dtype(self.dtype)?))
                .collect::<Result<Vec<_>>>()?;
            let mask = mlx_rs::ops::zeros::<f32>(&[1, 0])?;
            Ok((features, mask))
        }
    }

    /// Encode positives + negatives and assemble the joint CFG batch (`encode_prompt` +
    /// `_align_text_features` + the `[pos; neg]` stack). Returns `(encoder_features, encoder_mask)`
    /// where each feature layer is `[2, S_txt, 2880]` and the mask is `[2, S_txt]` (`1` = valid).
    pub fn encode_prompt(
        &self,
        prompt: &str,
        negative_prompt: &str,
        date: &str,
    ) -> Result<(Vec<Array>, Array)> {
        let (pos_feats, pos_mask) = self.encode_one(prompt, date)?;
        let s_pos = pos_feats[0].shape()[1];

        // Empty negative ⇒ the unconditional branch: zero text features matching the positive shape +
        // an all-`false` (all-zero) mask. A non-empty negative is encoded normally.
        let (neg_feats, neg_mask) = if negative_prompt.trim().is_empty() {
            let zeros = pos_feats
                .iter()
                .map(mlx_rs::ops::zeros_like)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            (zeros, mlx_rs::ops::zeros_like(&pos_mask)?)
        } else {
            self.encode_one(negative_prompt, date)?
        };
        let s_neg = neg_feats[0].shape()[1];

        // Pad both to a shared S_txt = max(s_pos, s_neg).
        let target = s_pos.max(s_neg);
        let pos_feats = pad_features(&pos_feats, s_pos, target)?;
        let neg_feats = pad_features(&neg_feats, s_neg, target)?;
        let pos_mask = pad_mask(&pos_mask, s_pos, target)?;
        let neg_mask = pad_mask(&neg_mask, s_neg, target)?;

        // Stack [pos; neg] along the batch axis → the joint CFG forward.
        let mut encoder_features = Vec::with_capacity(self.num_text_layers);
        for (pf, nf) in pos_feats.iter().zip(neg_feats.iter()) {
            encoder_features.push(concatenate_axis(&[pf, nf], 0)?.as_dtype(self.dtype)?);
        }
        let encoder_mask = concatenate_axis(&[&pos_mask, &neg_mask], 0)?; // [2, S_txt]
        Ok((encoder_features, encoder_mask))
    }

    /// The denoising loop over pre-encoded conditioning + an initial latent. Exposed for the e2e
    /// parity gate (which injects the reference's initial latents to factor out cross-RNG noise).
    ///
    /// - `encoder_features`: `num_text_layers × [2, S_txt, 2880]` (`[pos; neg]`).
    /// - `encoder_mask`: `[2, S_txt]` (`1` = valid).
    /// - `init_latents`: `[1, latent_h · latent_w, 128]`.
    ///
    /// Returns the final latents `[1, latent_h · latent_w, 128]` (patch-space; feed to [`vae::decode`]).
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        encoder_features: &[Array],
        encoder_mask: &Array,
        init_latents: &Array,
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        guidance_scale: f32,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize, usize),
    ) -> Result<Array> {
        let schedule = lens_schedule(num_steps, latent_h, latent_w);
        let timesteps = schedule::timesteps(&schedule);

        let mut latents = init_latents.as_dtype(self.dtype)?;
        for (i, &sigma) in timesteps.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            // Joint CFG batch: duplicate the latent (cond/uncond share the same x_t), one DiT call.
            let hidden = concatenate_axis(&[&latents, &latents], 0)?; // [2, seq, 128]
            let timestep = Array::from_slice(&[sigma, sigma], &[2]).as_dtype(self.dtype)?;

            let noise = self.transformer.forward(
                &hidden,
                encoder_features,
                Some(encoder_mask),
                &timestep,
                1,
                latent_h,
                latent_w,
            )?;

            // chunk(2) → cond (the positive, batch 0), uncond (the negative, batch 1).
            let parts = split(&noise, 2, 0)?;
            let noise_pred = cfg_rescale(&parts[0], &parts[1], guidance_scale)?;

            latents = schedule.step(&latents, &noise_pred, i)?;
            on_step(i + 1, num_steps);
        }
        Ok(latents)
    }

    /// Generate a single image (no cancellation / progress). Draws the initial latents from the
    /// global RNG seeded with `opts.seed`.
    pub fn generate(&self, opts: &GenerateOptions) -> Result<Image> {
        self.generate_with_progress(opts, &CancelFlag::default(), &mut |_| {})
    }

    /// Generate a single image, threading a cancel flag and a per-step progress callback
    /// (`on_step(completed_step)`). The registry loops `count` with per-image seeds over this.
    pub fn generate_with_progress(
        &self,
        opts: &GenerateOptions,
        cancel: &CancelFlag,
        on_step: &mut dyn FnMut(usize),
    ) -> Result<Image> {
        if !opts.width.is_multiple_of(VAE_SCALE_FACTOR)
            || !opts.height.is_multiple_of(VAE_SCALE_FACTOR)
        {
            return Err(Error::Msg(format!(
                "lens: height/width must be divisible by {VAE_SCALE_FACTOR} (got {}x{})",
                opts.height, opts.width
            )));
        }
        if opts.num_steps == 0 {
            return Err(Error::Msg("lens: num_steps must be >= 1".into()));
        }
        let latent_h = (opts.height / VAE_SCALE_FACTOR) as usize;
        let latent_w = (opts.width / VAE_SCALE_FACTOR) as usize;
        let seq_len = (latent_h * latent_w) as i32;

        let (encoder_features, encoder_mask) =
            self.encode_prompt(opts.prompt, opts.negative_prompt, opts.date)?;

        mlx_rs::random::seed(opts.seed)?;
        let init = mlx_rs::random::normal::<f32>(&[1, seq_len, 128], None, None, None)?;

        let latents = self.denoise(
            &encoder_features,
            &encoder_mask,
            &init,
            latent_h,
            latent_w,
            opts.num_steps,
            opts.guidance_scale,
            cancel,
            &mut |cur, _total| on_step(cur),
        )?;

        let decoded = vae::decode(&self.vae, &latents, latent_h, latent_w)?; // [1, H, W, 3] in [-1,1]
        decoded_to_image(&decoded)
    }

    /// The loaded VAE (for the e2e parity test's decode step).
    pub fn vae(&self) -> &Flux2Vae {
        &self.vae
    }
}

/// Zero-pad each `[B, cur, C]` feature layer along the sequence axis to length `target`.
fn pad_features(features: &[Array], cur: i32, target: i32) -> Result<Vec<Array>> {
    if cur == target {
        return Ok(features.to_vec());
    }
    let pad = target - cur;
    features
        .iter()
        .map(|f| {
            let (b, c) = (f.shape()[0], f.shape()[2]);
            let z = mlx_rs::ops::zeros::<f32>(&[b, pad, c])?.as_dtype(f.dtype())?;
            Ok(concatenate_axis(&[f, &z], 1)?)
        })
        .collect()
}

/// Zero-pad a `[B, cur]` mask along the sequence axis to length `target`.
fn pad_mask(mask: &Array, cur: i32, target: i32) -> Result<Array> {
    if cur == target {
        return Ok(mask.clone());
    }
    let pad = target - cur;
    let b = mask.shape()[0];
    let z = mlx_rs::ops::zeros::<f32>(&[b, pad])?;
    Ok(concatenate_axis(&[mask, &z], 1)?)
}

/// Convert a decoded image `[1, H, W, 3]` (NHWC) in `[-1, 1]` to an RGB8 [`Image`]
/// (`((x·0.5+0.5).clamp(0,1)·255).round()`), matching the reference `_to_pil` quantization.
fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let x = decoded.as_dtype(Dtype::Float32)?;
    let half = Array::from_f32(0.5);
    let x = mlx_rs::ops::add(&mlx_rs::ops::multiply(&x, &half)?, &half)?;
    let x = mlx_rs::ops::clip(&x, (0.0, 1.0))?;
    let x = mlx_rs::ops::round(&mlx_rs::ops::multiply(&x, Array::from_f32(255.0))?, 0)?;
    let sh = x.shape();
    let (h, w, c) = (sh[1] as u32, sh[2] as u32, sh[3] as u32);
    let n = (h * w * c) as usize;
    let total: i32 = sh.iter().product();
    let flat = x.reshape(&[total])?;
    let pixels: Vec<u8> = flat.as_slice::<f32>()[..n]
        .iter()
        .map(|&v| v as u8)
        .collect();
    Ok(Image {
        width: w,
        height: h,
        pixels,
    })
}
