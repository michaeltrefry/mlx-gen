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
use mlx_rs::ops::{add, concatenate_axis, divide, multiply, pad, subtract};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

use crate::config::DitConfig;
use crate::dit::Seedvr2Transformer;
use crate::vae::Seedvr2Vae;
use crate::video::{self, Chunk, ChunkPlan};
use crate::{color, convert};

/// The 1-step Euler timestep (= `num_train_steps`, which the SeedVR2 scheduler defaults to 1000).
const TIMESTEP: f32 = 1000.0;
/// Post-decode color-correction luminance weight (the reference `apply_color_correction` default).
const LUMINANCE_WEIGHT: f32 = 0.8;

pub struct Seedvr2Pipeline {
    pub vae: Seedvr2Vae,
    pub transformer: Seedvr2Transformer,
    neg_embed: Option<Array>,
    dtype: Dtype,
    /// Resident weight bytes (VAE + DiT at `dtype`) — drives the video memory-budget chunk sizer.
    weights_bytes: usize,
}

/// Cast every tensor in `w` to `dt`.
fn cast_weights(w: &Weights, dt: Dtype) -> Result<Weights> {
    let mut out = Weights::empty();
    for k in w.keys().map(String::from).collect::<Vec<_>>() {
        out.insert(k.clone(), w.require(&k)?.as_dtype(dt)?);
    }
    Ok(out)
}

/// Estimate resident weight bytes for the video memory budget: the raw `fp16` checkpoint file sizes
/// scaled by the load `dtype` (`Bfloat16` keeps the 2 B/param footprint, `Float32` doubles it). File
/// sizes (vs summing per-tensor) match the wan `dit_resident_bytes` convention; the safetensors header
/// overhead is negligible.
fn resident_weight_bytes(files: &[&std::path::Path], dt: Dtype) -> usize {
    let raw: u64 = files
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let ratio = match dt {
        Dtype::Float32 => 2.0, // fp16-on-disk → f32 resident
        _ => 1.0,              // bf16/fp16 resident
    };
    (raw as f64 * ratio) as usize
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
            weights_bytes: 0,
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
        let vae_path = dir.join("ema_vae_fp16.safetensors");
        let dit_path = dir.join(dit_file);
        let weights_bytes = resident_weight_bytes(&[&vae_path, &dit_path], dt);
        let vae_w = cast_weights(&convert::convert_vae(&Weights::from_file(&vae_path)?)?, dt)?;
        let dit_w = cast_weights(&convert::convert_dit(&Weights::from_file(&dit_path)?)?, dt)?;
        let mut p = Self::from_weights(&vae_w, &dit_w, cfg)?;
        p.neg_embed = Some(load_neg_embed(dt)?);
        p.dtype = dt;
        p.weights_bytes = weights_bytes;
        Ok(p)
    }

    /// The bundled negative-prompt embedding `(1,58,5120)` (set by [`Self::load`]).
    pub fn neg_embed(&self) -> Option<&Array> {
        self.neg_embed.as_ref()
    }

    /// Quantize the DiT Linears to `bits` (4 or 8) — group-wise affine, Linear-only (sc-5198). The
    /// VAE stays dense (conv-dominated; its tiny attention Linears are left at `dtype`). `weights_bytes`
    /// is intentionally **not** reduced — keeping the dense estimate makes the video chunk-size budget
    /// conservative (quant shrinks weights, not activations, so the headroom is safe).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.transformer.quantize(bits)
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

    /// Preprocess one LR `Image` to a single-frame clip `(1,3,1,height,width)` in `[-1,1]` at the model
    /// dtype (the [`Self::preprocess`] image + a temporal axis). Public for the tiling gate.
    pub fn preprocess_frame(
        &self,
        image: &Image,
        width: i32,
        height: i32,
        softness: f32,
    ) -> Result<Array> {
        Ok(self
            .preprocess(image, width, height, softness)?
            .expand_dims(2)?)
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

    // -----------------------------------------------------------------------
    // video mode (sc-4814): multi-frame 5-D pass + temporal chunking/overlap
    // -----------------------------------------------------------------------

    /// Decode latents and crop spatially to `(true_h, true_w)`, **keeping all `T` frames** →
    /// `(B,3,T,true_h,true_w)`. The 5-D analog of [`Self::decode_crop`] (which keeps only frame 0).
    pub fn decode_crop_5d(&self, latents: &Array, true_h: i32, true_w: i32) -> Result<Array> {
        let decoded = self.vae.decode(latents)?; // (B,3,T,H,W)
        let h_idx = Array::from_slice(&(0..true_h).collect::<Vec<i32>>(), &[true_h]);
        let w_idx = Array::from_slice(&(0..true_w).collect::<Vec<i32>>(), &[true_w]);
        Ok(decoded.take_axis(h_idx, 3)?.take_axis(w_idx, 4)?)
    }

    /// Multi-frame model path (no color correction): a preprocessed clip `(1,3,T,H,W)` + injected
    /// noise `(1,16,T',h,w)` → decoded crop `(1,3,T,true_h,true_w)`. The video analog of
    /// [`Self::run_model`]; `encode`/`condition`/`denoise` are already `T`-aware.
    pub fn run_model_5d(
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
        self.decode_crop_5d(&latents, true_h, true_w)
    }

    /// Per-frame color-correct a decoded clip `(1,3,T,Hc,Wc)` against its preprocessed `style`
    /// `(1,3,Ts,Hc,Wc)` → `count` RGB8 frames. Frame `t` matches style frame `min(t, Ts-1)` (the
    /// reference `to_uint8_frames`).
    fn frames_from_decoded(
        &self,
        decoded: &Array,
        style: &Array,
        count: i32,
    ) -> Result<Vec<Image>> {
        let style_t = style.shape()[2];
        let mut out = Vec::with_capacity(count as usize);
        for t in 0..count {
            let d = decoded.take_axis(Array::from_int(t), 2)?; // (1,3,Hc,Wc)
            let s = style.take_axis(Array::from_int(t.min(style_t - 1)), 2)?;
            let corrected = color::apply_color_correction(
                &d.as_dtype(Dtype::Float32)?,
                &s.as_dtype(Dtype::Float32)?,
                LUMINANCE_WEIGHT,
            )?;
            out.push(decoded_to_image(&corrected)?);
        }
        Ok(out)
    }

    /// Preprocess one temporal chunk: pixel-frames `[start, start+len)` of `frames`, clamping past the
    /// end to the last real frame (last-frame padding) → `(1,3,len,H,W)` in `[-1,1]`.
    fn preprocess_chunk(
        &self,
        frames: &[Image],
        start: i32,
        len: i32,
        width: i32,
        height: i32,
        softness: f32,
    ) -> Result<Array> {
        let n = frames.len() as i32;
        let mut per = Vec::with_capacity(len as usize);
        for j in 0..len {
            let idx = (start + j).clamp(0, n - 1) as usize;
            per.push(
                self.preprocess(&frames[idx], width, height, softness)?
                    .expand_dims(2)?,
            );
        }
        let refs: Vec<&Array> = per.iter().collect();
        Ok(concatenate_axis(&refs, 2)?)
    }

    /// End-to-end **video** upscale: a sequence of LR `frames` → upscaled `(width, height)` RGB8
    /// frames (same count). Sizes the temporal chunk against the memory budget (or `chunk_override`),
    /// processes each chunk through the 5-D model path with one-step Euler, per-frame color-corrects,
    /// and cross-fades chunk overlaps to close the causal-VAE seam ([`crate::video`]). Falls back to
    /// the per-frame (`T=1`) path under tight memory, and to per-frame **spatial tiling** when even one
    /// full-resolution frame exceeds the budget (HD — sc-5201), so peak stays bounded at any resolution.
    pub fn generate_video(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
        chunk_override: Option<i32>,
    ) -> Result<Vec<Image>> {
        let n = frames.len() as i32;
        if n == 0 {
            return Ok(Vec::new());
        }
        let chunk = match chunk_override {
            Some(c) => video::pad_to_valid_chunk(c),
            None => match video::plan_chunk_size(self.weights_bytes, height, width) {
                ChunkPlan::Chunked(c) => c,
                ChunkPlan::PerFrame => {
                    return self.generate_video_per_frame(frames, width, height, seed, softness)
                }
                // Even one full-resolution frame exceeds the budget → spatially tile each frame
                // (per-frame T=1 + overlap feather blend). Bounds peak at any resolution (sc-5201).
                ChunkPlan::OverBudget { .. } => {
                    return self.generate_video_tiled(frames, width, height, seed, softness)
                }
            },
        };

        let plan = video::plan_chunks(n, chunk, video::DEFAULT_OVERLAP);
        let neg = self
            .neg_embed
            .as_ref()
            .expect("neg-embed (use Seedvr2Pipeline::load)")
            .clone();
        let ts = Array::from_f32(TIMESTEP);
        let mut chunk_frames: Vec<Vec<Image>> = Vec::with_capacity(plan.len());
        for Chunk { start, len } in &plan {
            let clip = self.preprocess_chunk(frames, *start, *len, width, height, softness)?;
            let latent = self.encode(&clip)?;
            let sh = latent.shape();
            // Same noise key for every chunk (the reference chunk-overlap test) → deterministic blend.
            let key = random::key(seed)?;
            let noise =
                random::normal::<f32>(&[1, 16, sh[2], sh[3], sh[4]], None, None, Some(&key))?
                    .as_dtype(self.dtype)?;
            let cond = Self::condition(&latent)?;
            let latents = self.denoise(&noise, &cond, &neg, &ts)?;
            let decoded = self.decode_crop_5d(&latents, height, width)?;
            chunk_frames.push(self.frames_from_decoded(&decoded, &clip, *len)?);
        }
        Ok(video::assemble_overlap(
            &plan,
            &chunk_frames,
            n,
            video::DEFAULT_OVERLAP,
        ))
    }

    /// Per-frame (`T=1`) video fallback: each frame through the still path with a fixed (anchored)
    /// seed — intrinsically temporally stable (spike sc-4812). Used when even an 8-frame chunk
    /// exceeds the memory budget.
    fn generate_video_per_frame(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
    ) -> Result<Vec<Image>> {
        frames
            .iter()
            .map(|f| self.generate(f, width, height, seed, softness))
            .collect()
    }

    /// HD spatial-tiling video path (sc-5201): each frame is upscaled per-frame (`T=1`) but **spatially
    /// tiled** — the budget sizer picks the largest square tile that fits, and the decoded tiles are
    /// feather-blended. Used when even one full-resolution frame exceeds the memory budget; bounds peak
    /// at any resolution. The numeric tiling fidelity is gated in `tests/tiling_parity.rs`.
    fn generate_video_tiled(
        &self,
        frames: &[Image],
        width: i32,
        height: i32,
        seed: u64,
        softness: f32,
    ) -> Result<Vec<Image>> {
        let tile = video::plan_spatial_tile_px(self.weights_bytes, video::safe_budget_gib());
        let overlap = video::SPATIAL_OVERLAP.min(tile / 2);
        let neg = self
            .neg_embed
            .as_ref()
            .expect("neg-embed (use Seedvr2Pipeline::load)")
            .clone();
        let mut out = Vec::with_capacity(frames.len());
        for f in frames {
            let processed = self
                .preprocess(f, width, height, softness)?
                .expand_dims(2)?; // (1,3,1,H,W)
            let decoded = self.run_frame_tiled(&processed, seed, tile, overlap, &neg)?;
            let imgs = self.frames_from_decoded(&decoded, &processed, 1)?;
            out.push(imgs.into_iter().next().expect("one frame"));
        }
        Ok(out)
    }

    /// Upscale one preprocessed frame `(1,3,1,H,W)` by spatial tiling: run the full encode → DiT →
    /// decode path on each overlapping `tile`-px tile (one-step Euler, same-seed noise) and feather-
    /// blend the decoded tiles into a full `(1,3,1,H,W)` frame. The accumulator is evaluated per tile
    /// so only one tile's activations are resident at a time (the memory bound). Public for the gate.
    pub fn run_frame_tiled(
        &self,
        processed: &Array,
        seed: u64,
        tile: i32,
        overlap: i32,
        neg: &Array,
    ) -> Result<Array> {
        let sh = processed.shape(); // (1,3,1,H,W)
        let (height, width) = (sh[3], sh[4]);
        let plan = video::plan_spatial_tiles(height, width, tile, overlap);
        let ts = Array::from_f32(TIMESTEP);
        let mut acc: Option<Array> = None; // (1,3,1,H,W)
        let mut wsum: Option<Array> = None; // (1,1,1,H,W)
        for t in &plan {
            let (th, tw) = (t.y1 - t.y0, t.x1 - t.x0);
            let y_idx = Array::from_slice(&(t.y0..t.y1).collect::<Vec<i32>>(), &[th]);
            let x_idx = Array::from_slice(&(t.x0..t.x1).collect::<Vec<i32>>(), &[tw]);
            let tile_clip = processed.take_axis(y_idx, 3)?.take_axis(x_idx, 4)?; // (1,3,1,th,tw)

            // full model path on the tile (one-step Euler), same noise key as the other tiles.
            let latent = self.encode(&tile_clip)?;
            let sh = latent.shape();
            let key = random::key(seed)?;
            let noise =
                random::normal::<f32>(&[1, 16, sh[2], sh[3], sh[4]], None, None, Some(&key))?
                    .as_dtype(self.dtype)?;
            let cond = Self::condition(&latent)?;
            let latents = self.denoise(&noise, &cond, neg, &ts)?;
            let decoded = self.decode_crop_5d(&latents, th, tw)?; // (1,3,1,th,tw)

            // feather weight tapering on edges that abut a neighbor; placed at (y0,x0).
            let wvec = video::feather_weight(
                th,
                tw,
                t.y0 > 0,
                t.y1 < height,
                t.x0 > 0,
                t.x1 < width,
                overlap,
            );
            let weight = Array::from_slice(&wvec, &[1, 1, 1, th, tw]).as_dtype(self.dtype)?;
            let pad_spec = [
                (0, 0),
                (0, 0),
                (0, 0),
                (t.y0, height - t.y1),
                (t.x0, width - t.x1),
            ];
            let wdec = pad(&multiply(&decoded, &weight)?, &pad_spec[..], None, None)?;
            let wpad = pad(&weight, &pad_spec[..], None, None)?;
            acc = Some(match acc {
                Some(a) => add(&a, &wdec)?,
                None => wdec,
            });
            wsum = Some(match wsum {
                Some(a) => add(&a, &wpad)?,
                None => wpad,
            });
            // materialize so the just-processed tile's activations (and prior graph) are freed.
            eval([acc.as_ref().unwrap(), wsum.as_ref().unwrap()])?;
        }
        Ok(divide(acc.expect("≥1 tile"), wsum.expect("≥1 tile"))?)
    }
}
