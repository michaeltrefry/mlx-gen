//! S4/S5 — the **T2V generation pipeline**: the denoise loop + CFG + VAE decode + frame assembly
//! that turns latents into video. Port of `generate_wan.py`'s `generate_video` — both the
//! single-model dense path ([`denoise`], S4) and the dual-expert MoE path ([`denoise_moe`], S5).
//!
//! This is **reusable machinery**, not a model: the dense loop is exactly what each Wan2.2-A14B MoE
//! expert runs (the MoE adds only the per-step boundary swap) and what the 5B runs (sc-2680, with
//! its z48 VAE). The concrete `Generator::generate` wiring lands in `model.rs`.
//!
//! Shapes are channels-first **`[C, F, H, W]`** (no batch dim) for the latents + scheduler. CFG runs
//! cond + uncond as a **single batched B=2 forward** ([`WanTransformer::forward_cached`]) — the shared
//! latent is patchified once and broadcast across the batch, so each per-step GPU kernel launches once
//! instead of twice (the small-seq win, sc-2853); it stays bit-identical to two B=1 forwards since
//! attention never mixes batch elements. The per-block cross-attention K/V and the RoPE cos/sin are
//! **precomputed once per expert** before the loop (the reference's `prepare_cross_kv` / `prepare_rope`)
//! and reused across all steps, instead of recomputed every forward.

use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::tiling::TilingConfig;
use mlx_gen::{Error, Image, Result};

use crate::scheduler::{make_scheduler, SolverKind};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Align a pixel dimension **down** to a multiple of `patch · vae_stride` (the reference rounds the
/// requested size to the nearest valid grid; sub-tile requests are rejected by `validate`).
pub fn align_dim(value: u32, patch: usize, stride: usize) -> u32 {
    let align = (patch * stride) as u32;
    (value / align) * align
}

/// Latent shape `[z_dim, t_lat, h_lat, w_lat]` for a `frames × H × W` request.
/// `t_lat = (frames − 1) / vae_stride_t + 1`; spatial divide by the vae stride.
pub fn latent_shape(
    frames: usize,
    height: u32,
    width: u32,
    z_dim: usize,
    vae_stride: (usize, usize, usize),
) -> [i32; 4] {
    let t_lat = (frames - 1) / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    [z_dim as i32, t_lat as i32, h_lat as i32, w_lat as i32]
}

/// Transformer sequence length: `ceil(h_lat · w_lat / (patch_h · patch_w) · t_lat)`.
pub fn seq_len(latent: [i32; 4], patch_size: (usize, usize, usize)) -> usize {
    let (_z, t_lat, h_lat, w_lat) = (latent[0], latent[1], latent[2], latent[3]);
    let per_frame = (h_lat as usize * w_lat as usize) as f64 / (patch_size.1 * patch_size.2) as f64;
    (per_frame * t_lat as f64).ceil() as usize
}

/// The largest `(width, height)` that fits within `max_area` while preserving the input aspect ratio
/// and staying aligned to the `(dw, dh)` grid (= `patch · vae_stride`). Port of `generate_wan.py`'s
/// `_best_output_size`: it derives the ideal `(ow, oh)` from `√(max_area·ratio)`, then tries
/// width-first and height-first alignment and keeps whichever distorts the aspect ratio less. Applied
/// only when `config.max_area > 0` and the requested area exceeds it (I2V-14B / TI2V-5B cap, 704×1280).
pub fn best_output_size(width: u32, height: u32, dw: u32, dh: u32, max_area: usize) -> (u32, u32) {
    let (w, h, dw_f, dh_f) = (width as f64, height as f64, dw as f64, dh as f64);
    let area = max_area as f64;
    let ratio = w / h;
    let ow = (area * ratio).sqrt();
    let oh = area / ow;

    // Option 1: align width first, derive height from the remaining area. (`int(x // d * d)`.)
    let ow1 = (ow / dw_f).floor() * dw_f;
    let oh1 = (area / ow1 / dh_f).floor() * dh_f;
    let ratio1 = ow1 / oh1;

    // Option 2: align height first, derive width.
    let oh2 = (oh / dh_f).floor() * dh_f;
    let ow2 = (area / oh2 / dw_f).floor() * dw_f;
    let ratio2 = ow2 / oh2;

    let dist1 = (ratio / ratio1).max(ratio1 / ratio);
    let dist2 = (ratio / ratio2).max(ratio2 / ratio);
    if dist1 < dist2 {
        (ow1 as u32, oh1 as u32)
    } else {
        (ow2 as u32, oh2 as u32)
    }
}

/// Classifier-free guidance combine: `uncond + gs·(cond − uncond)`.
fn cfg_combine(cond: &Array, uncond: &Array, gs: f32) -> Result<Array> {
    Ok(add(
        uncond,
        &multiply(&subtract(cond, uncond)?, scalar(gs))?,
    )?)
}

/// Per-generate caches for one transformer/expert, constant across every denoise step: the bf16 RoPE
/// `(cos, sin)` for the (fixed) grid + each block's cross-attention K/V for the (CFG-batched) context.
/// Mirrors the reference's `prepare_rope` / `prepare_cross_kv`, computed once before the loop.
struct StepCache {
    /// Per-block cross-attention `(k, v)`, each `[batch, n, text_len, d]` (bf16).
    cross_kv: Vec<(Array, Array)>,
    cos: Array,
    sin: Array,
    /// Forward batch width: 2 when CFG is on (cond+uncond stacked), else 1.
    batch: usize,
}

/// Build the per-expert [`StepCache`] from the embedded contexts + the (constant) RoPE grid. When CFG
/// is on (`ctx_uncond = Some`) the cond/uncond contexts are stacked on the batch axis so the cross-K/V
/// is `B=2`; otherwise `B=1`. The caches are evaluated once here (the reference's `mx.eval(cross_kv,
/// rope_cos_sin)`) so each per-step graph reuses them instead of recomputing.
fn build_cache(
    transformer: &WanTransformer,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    grid: (usize, usize, usize),
) -> Result<StepCache> {
    let (context_batch, batch) = match ctx_uncond {
        Some(uncond) => (concatenate_axis(&[ctx_cond, uncond], 0)?, 2),
        None => (ctx_cond.clone(), 1),
    };
    let cross_kv = transformer.prepare_cross_kv(&context_batch)?;
    let (cos, sin) = transformer.prepare_rope(grid)?;
    let mut to_eval: Vec<&Array> = vec![&cos, &sin];
    for (k, v) in &cross_kv {
        to_eval.push(k);
        to_eval.push(v);
    }
    mlx_rs::transforms::eval(to_eval)?;
    Ok(StepCache {
        cross_kv,
        cos,
        sin,
        batch,
    })
}

/// One denoise prediction reusing the precomputed [`StepCache`]: a single batched forward yielding
/// `[cond, uncond]`, combined as `uncond + gs·(cond − uncond)` when CFG is on, else the B=1 cond-only
/// forward.
///
/// `y` is the optional I2V channel-concat conditioning `[20, F, H, W]` (mirrors `WanModel.__call__`'s
/// `y`): when `Some`, it is concatenated **onto the channel axis after** the `[16, …]` noise latent —
/// `[noise(16), mask(4), z_video(16)]` → `[36, F, H, W]` — before patchify, exactly the channel order
/// the I2V-14B `patch_embedding` (in_dim 36) was trained on. The DiT prediction stays `out_dim = 16`,
/// so the scheduler step still consumes/produces the 16-channel latent.
fn predict(
    transformer: &WanTransformer,
    latents: &Array,
    t: f32,
    cache: &StepCache,
    guidance: f32,
    y: Option<&Array>,
) -> Result<Array> {
    let x = match y {
        Some(y) => concatenate_axis(&[latents, y], 0)?,
        None => latents.clone(),
    };
    let preds =
        transformer.forward_cached(&x, t, &cache.cross_kv, &cache.cos, &cache.sin, cache.batch)?;
    if cache.batch == 2 {
        // preds[0] = cond (context row 0), preds[1] = uncond (row 1).
        cfg_combine(&preds[0], &preds[1], guidance)
    } else {
        Ok(preds
            .into_iter()
            .next()
            .expect("B=1 forward yields one output"))
    }
}

/// The dense denoise loop (single model). `ctx_cond`/`ctx_uncond` are
/// [`WanTransformer::embed_text`] outputs; pass `ctx_uncond = None` for the CFG-disabled B=1 fast
/// path. `init_noise` is `[C, F, H, W]` f32. Returns the denoised latents `[out_dim, F, H, W]`
/// (f32). `on_step(i)` is called after each completed step.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    transformer: &WanTransformer,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    guidance: f32,
    ctx_cond: &Array,
    ctx_uncond: Option<&Array>,
    init_noise: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: run the DiT's fusable elementwise glue (adaLN affine, gated residual, gated-GELU FFN,
    // RoPE rotation) through `mx.compile` — bit-exact (proven `max|Δ|=0` real + tiny, perf.rs /
    // compile_parity.rs) and ~14% faster/step at production geometry. Process-global, idempotent.
    crate::transformer::set_compile_glue(true);

    // Precompute the RoPE + cross-K/V caches once (grid + context are constant across steps).
    let grid = transformer.patch_grid(init_noise);
    let cache = build_cache(transformer, ctx_cond, ctx_uncond, grid)?;

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        let pred = predict(transformer, &latents, t, &cache, guidance, None)?;
        latents = sched.step(&pred, &latents)?;
        // Force evaluation each step to bound the lazy graph's peak memory (the reference's
        // per-step `mx.eval(latents)`).
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// One MoE expert: a full transformer + its own (per-model) embedded contexts + guidance scale.
/// Wan2.2-A14B's "MoE" is two complete checkpoints, not token routing — each carries its own
/// `text_embedding`, so contexts are embedded per expert.
pub struct Expert<'a> {
    pub transformer: &'a WanTransformer,
    /// `embed_text` output for this expert (cond).
    pub ctx_cond: Array,
    /// `embed_text` output for this expert (uncond); `None` ⇒ CFG disabled for this expert.
    pub ctx_uncond: Option<Array>,
    /// This expert's guidance scale (the `low`/`high` of the dual `sample_guide_scale`).
    pub guidance: f32,
}

/// The dual-expert MoE denoise loop (Wan2.2-A14B). Each step picks the **high-noise** expert while
/// the integer timestep is `≥ boundary_timestep` (`config.boundary · num_train_timesteps`, e.g.
/// `0.875 · 1000 = 875`) and the **low-noise** expert below it — switching the transformer, the
/// per-expert contexts, and the per-expert guidance together. Reduces to [`denoise`] when both
/// experts are the same model.
///
/// `y` is the optional I2V-14B channel-concat conditioning `[20, F, H, W]` ([`build_i2v_y`]),
/// concatenated onto each forward's noise latent (see [`predict`]); `None` for T2V. It is constant
/// across steps and shared by both experts (the conditioning doesn't change with the noise level).
#[allow(clippy::too_many_arguments)]
pub fn denoise_moe(
    low: &Expert,
    high: &Expert,
    boundary_timestep: f32,
    kind: SolverKind,
    num_train_timesteps: usize,
    steps: usize,
    shift: f32,
    init_noise: &Array,
    y: Option<&Array>,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let mut sched = make_scheduler(kind, num_train_timesteps);
    sched.set_timesteps(steps, shift);
    let timesteps: Vec<f32> = sched.timesteps().to_vec();

    // sc-2957: compiled elementwise glue (bit-exact, ~14% faster/step) — see `denoise`.
    crate::transformer::set_compile_glue(true);

    // Precompute each expert's RoPE + cross-K/V caches once (the grid is shared — the channel-concat
    // `y` doesn't change F/H/W — and each expert's contexts are constant across steps).
    let grid = low.transformer.patch_grid(init_noise);
    let low_cache = build_cache(
        low.transformer,
        &low.ctx_cond,
        low.ctx_uncond.as_ref(),
        grid,
    )?;
    let high_cache = build_cache(
        high.transformer,
        &high.ctx_cond,
        high.ctx_uncond.as_ref(),
        grid,
    )?;

    let mut latents = init_noise.clone();
    for (i, &t) in timesteps.iter().enumerate() {
        let (e, cache) = if t >= boundary_timestep {
            (high, &high_cache)
        } else {
            (low, &low_cache)
        };
        let pred = predict(e.transformer, &latents, t, cache, e.guidance, y)?;
        latents = sched.step(&pred, &latents)?;
        mlx_rs::transforms::eval([&latents])?;
        on_step(i + 1);
    }
    Ok(latents)
}

/// Decode denoised latents `[C, F, H, W]` → an RGB video tensor `[F_out, H_out, W_out, 3]` of
/// `uint8` (the reference's `(video + 1)/2 · 255`, clamped). Uses the Wan 2.1 z16 VAE (S2). When
/// `tiling` is `Some`, decodes via [`WanVae::decode_tiled`] (memory-bounded for large/long video;
/// it falls back to a single pass when the config doesn't fire); `None` is always single-pass.
pub fn decode_to_frames(
    vae: &WanVae,
    latents: &Array,
    tiling: Option<&TilingConfig>,
) -> Result<Array> {
    // WanVae::decode[_tiled] expect/return a leading batch dim: [1, 3, F, H, W] in [-1, 1].
    let z = latents.reshape(&prepend1(latents.shape()))?;
    let video = match tiling {
        Some(cfg) => vae.decode_tiled(&z, cfg)?,
        None => vae.decode(&z)?,
    };
    // [1,3,F,H,W] → [F,H,W,3]
    let sh = video.shape(); // [1,3,F,H,W]
    let (f, h, w) = (sh[2], sh[3], sh[4]);
    let chw = video
        .reshape(&[3, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // [F,H,W,3]
                                         // [-1,1] → [0,255] uint8
    let scaled = multiply(&add(&chw, scalar(1.0))?, scalar(127.5))?;
    let clamped = minimum(&maximum(&scaled, scalar(0.0))?, scalar(255.0))?;
    Ok(clamped.as_dtype(mlx_rs::Dtype::Uint8)?)
}

/// Split a `[F, H, W, 3]` `uint8` video tensor (the [`decode_to_frames`] output) into one
/// [`Image`] per frame. The tensor is transpose-strided, so a raw `as_slice` would read the
/// physical (pre-transpose) buffer — `reshape` first re-materializes it in logical C-order, then we
/// chunk the contiguous bytes `H·W·3` at a time (see [[mlx_rs_as_slice_physical_buffer]]).
pub fn frames_to_images(frames_u8: &Array) -> Result<Vec<Image>> {
    let sh = frames_u8.shape(); // [F, H, W, 3]
    let (f, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let total: i32 = f * h * w * c;
    let flat = frames_u8.reshape(&[total])?; // materialize logical NHWC order
    let bytes = flat.as_slice::<u8>();
    let per = (h * w * c) as usize;
    let mut out = Vec::with_capacity(f as usize);
    for i in 0..f as usize {
        out.push(Image {
            width: w as u32,
            height: h as u32,
            pixels: bytes[i * per..(i + 1) * per].to_vec(),
        });
    }
    Ok(out)
}

/// `[d0, d1, ...]` → `[1, d0, d1, ...]` (prepend a batch axis).
fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

// ===========================================================================================
// I2V-14B channel-concat conditioning (port of `generate_wan.py`'s `is_i2v_channel_concat` setup)
// ===========================================================================================

/// Python `round()` — round half to **even** (banker's rounding), matching `round(img.width * scale)`
/// in the reference's image preprocessing. (Rust `f64::round` rounds half away from zero, which would
/// differ on exact `.5` derived sizes.)
fn py_round(x: f64) -> usize {
    let floor = x.floor();
    let frac = x - floor;
    // Round up on frac > 0.5, or on an exact tie (frac == 0.5) when `floor` is odd (→ even).
    let round_up = frac > 0.5 || (frac == 0.5 && (floor as i64) % 2 != 0);
    (if round_up { floor + 1.0 } else { floor }) as usize
}

/// Preprocess an I2V conditioning image to `[3, height, width]` f32 in `[-1, 1]` (CHW), matching the
/// reference's inline pipeline: **cover-fit** LANCZOS resize (`scale = max(W/iw, H/ih)`, new dims
/// `round(·)`), **center-crop** to the target, then `px/255·2 − 1`. The resize is the core PIL-exact
/// fixed-point integer LANCZOS ([`resize_lanczos_u8`]), so it's bit-identical to PIL's `Image.LANCZOS`.
pub fn preprocess_i2v_image(image: &Image, width: u32, height: u32) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (width as usize, height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "i2v image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // Cover-fit: scale so the image covers the target, then round to integer dims (PIL `round`).
    let scale = (tw as f64 / iw as f64).max(th as f64 / ih as f64);
    let nw = py_round(iw as f64 * scale).max(tw);
    let nh = py_round(ih as f64 * scale).max(th);
    let resized: Vec<f32> = if (nh, nw) == (ih, iw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, nh, nw)
    };
    // Center-crop the (integer-valued) resized HWC buffer to (th, tw), then normalize → CHW [-1,1].
    let x1 = (nw - tw) / 2;
    let y1 = (nh - th) / 2;
    let mut chw = vec![0f32; 3 * th * tw];
    let plane = th * tw;
    for yy in 0..th {
        for xx in 0..tw {
            let src = ((y1 + yy) * nw + (x1 + xx)) * 3;
            for c in 0..3 {
                chw[c * plane + yy * tw + xx] = 2.0 * (resized[src + c] / 255.0) - 1.0;
            }
        }
    }
    Ok(Array::from_slice(&chw, &[3, th as i32, tw as i32]))
}

/// The I2V-14B 4-channel temporal mask `[4, T_lat, h_lat, w_lat]` (f32): `1.0` for the first latent
/// temporal frame (all 4 channels, all spatial), `0.0` elsewhere. The reference builds this via a
/// `ones`/`zeros` → `repeat(first,4)` → `reshape(·,T_lat,4,·,·)` → `transpose` dance over the
/// `[1, F, h_lat, w_lat]` per-frame mask (first frame 1, rest 0); the result is exactly this pattern
/// (the per-frame mask collapses to "the first 4 of `F+3` temporal slots", which is latent frame 0).
fn build_i2v_mask(t_lat: usize, h_lat: usize, w_lat: usize) -> Array {
    let plane = h_lat * w_lat;
    let mut data = vec![0f32; 4 * t_lat * plane];
    for c in 0..4 {
        let base = c * t_lat * plane; // temporal index 0 of channel c
        for p in 0..plane {
            data[base + p] = 1.0;
        }
    }
    Array::from_slice(&data, &[4, t_lat as i32, h_lat as i32, w_lat as i32])
}

/// Build the I2V-14B channel-concat conditioning `y = [mask(4), z_video(16)]` → `[20, T_lat, h_lat,
/// w_lat]` (f32). Port of `generate_wan.py`'s `is_i2v_channel_concat` branch: a conditioning video
/// (first frame = the preprocessed image, the remaining `frames−1` zero) is encoded by the 2.1 z16
/// `WanVae` → `z_video [16, T_lat, …]`, and concatenated under the temporal mask. `vae` must carry
/// encoder weights. The result is `Some(y)` fed to [`denoise_moe`].
pub fn build_i2v_y(
    vae: &WanVae,
    image: &Image,
    frames: usize,
    height: u32,
    width: u32,
    vae_stride: (usize, usize, usize),
) -> Result<Array> {
    let (h, w) = (height as i32, width as i32);
    // Conditioning video [3, F, H, W]: first frame = image, rest zeros.
    let first = preprocess_i2v_image(image, width, height)?.reshape(&[3, 1, h, w])?;
    let rest = Array::zeros::<f32>(&[3, frames as i32 - 1, h, w])?;
    let video = concatenate_axis(&[&first, &rest], 1)?; // [3, F, H, W]

    // VAE-encode → [1, 16, T_lat, h_lat, w_lat], drop the batch axis → [16, T_lat, h_lat, w_lat].
    let z_video = vae.encode(&video.reshape(&[1, 3, frames as i32, h, w])?)?;
    let z_video = z_video.reshape(&z_video.shape()[1..])?;

    let t_lat = (frames - 1) / vae_stride.0 + 1;
    let h_lat = height as usize / vae_stride.1;
    let w_lat = width as usize / vae_stride.2;
    let mask = build_i2v_mask(t_lat, h_lat, w_lat);

    Ok(concatenate_axis(&[&mask, &z_video], 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_dim_rounds_down_to_tile() {
        // patch 2 × vae_stride 8 = 16-px grid.
        assert_eq!(align_dim(1280, 2, 8), 1280);
        assert_eq!(align_dim(1281, 2, 8), 1280);
        assert_eq!(align_dim(1295, 2, 8), 1280);
        assert_eq!(align_dim(1296, 2, 8), 1296);
    }

    #[test]
    fn latent_shape_and_seq_len_match_reference_formulas() {
        // 49 frames, 512×512, z16, stride (4,8,8), patch (1,2,2).
        let ls = latent_shape(49, 512, 512, 16, (4, 8, 8));
        assert_eq!(ls, [16, 13, 64, 64]); // (49-1)/4+1=13, 512/8=64
        let sl = seq_len(ls, (1, 2, 2));
        // ceil(64*64/(2*2) * 13) = 1024 * 13 = 13312
        assert_eq!(sl, 13312);
    }

    #[test]
    fn cfg_combine_is_uncond_plus_gs_delta() {
        let cond = Array::from_slice(&[2.0f32, 4.0], &[2]);
        let uncond = Array::from_slice(&[1.0f32, 1.0], &[2]);
        let got = cfg_combine(&cond, &uncond, 3.0).unwrap();
        // 1 + 3*(2-1) = 4 ; 1 + 3*(4-1) = 10
        assert_eq!(got.as_slice::<f32>(), &[4.0, 10.0]);
    }

    #[test]
    fn py_round_is_half_to_even() {
        assert_eq!(py_round(19.2), 19);
        assert_eq!(py_round(16.0), 16);
        assert_eq!(py_round(0.5), 0); // half → even (down)
        assert_eq!(py_round(1.5), 2); // half → even (up)
        assert_eq!(py_round(2.5), 2); // half → even (down)
        assert_eq!(py_round(2.500001), 3); // just over half → up
    }

    #[test]
    fn best_output_size_caps_area_and_aligns() {
        // 1280×720 over the I2V/TI2V 704×1280 cap, 16-px grid → width-first wins (less distortion).
        let (w, h) = best_output_size(1280, 720, 16, 16, 704 * 1280);
        assert_eq!((w, h), (1264, 704));
        assert!((w * h) as usize <= 704 * 1280, "must fit within max_area");
        assert_eq!(w % 16, 0);
        assert_eq!(h % 16, 0);
    }

    #[test]
    fn build_i2v_mask_is_one_at_first_latent_frame() {
        // [4, T_lat=2, 1, 1]: channel-major, temporal index 0 → 1.0, index 1 → 0.0.
        let m = build_i2v_mask(2, 1, 1);
        assert_eq!(m.shape(), &[4, 2, 1, 1]);
        assert_eq!(m.as_slice::<f32>(), &[1., 0., 1., 0., 1., 0., 1., 0.]);
    }
}
