//! S5 — the **2-stage distilled T2V pipeline**: the denoise loop + the stage transition (2× spatial
//! upsample + re-noise) + video output. Port of the `mlx_video` reference `generate_av.py` video path
//! (`denoise_av` with audio disabled + the `generate_video_with_audio` stage orchestration).
//!
//! The shipped `base_q8` is a **unified split-weight** checkpoint (`split_model.json` `format:
//! "split"`), so the reference takes its `legacy_unified_sampler` branch: **fixed distilled sigmas**
//! ([`STAGE1_SIGMAS`] 8 steps + [`STAGE2_SIGMAS`] 3 steps) and the **legacy dtype-preserving Euler**
//! update. The distilled 2.3 model bakes in guidance, so `effective_cfg_scale = 1.0` → **no CFG**
//! (single forward per step, no negative prompt). T2V has no I2V conditioning state.
//!
//! Flow ([`generate_t2v`]): random noise → stage-1 denoise at half-res → [`upsample_latents`] 2× →
//! re-noise (`noise·σ₂₀ + latent·(1−σ₂₀)`) → stage-2 denoise at full-res → VAE decode →
//! `(x+1)/2·255` uint8 frames `[F, H, W, 3]` (the consuming app muxes these to MP4 — matching the
//! Wan sibling, MP4 encoding is out of the crate).
//!
//! **Precision (S5 gate).** Run in the **f32** regime (latents f32, transformer [`Precision::F32Q8`],
//! upsampler/VAE f32) to gate the pipeline *math* — the 2-stage orchestration, the legacy Euler, the
//! re-noise, the flatten/unflatten — bit-tight, isolated from bf16 rounding (consistent with the S3b
//! DiT gate). The bf16-**production** end-to-end px>8 verdict is S6, which wires the real text encoder
//! and the public `generate()`. Honors "divergence is not rounding": the parity test localizes any
//! gap (per-stage latents + decoded frames) rather than writing it off.

use mlx_rs::ops::{add, broadcast_to, divide, maximum, minimum, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{Error, Image, Result};

use crate::conditioning::{apply_conditioning, apply_denoise_mask, I2vConditioning};
use crate::transformer::{to_denoised, LtxDiT};
use crate::upsampler::{upsample_latents, LatentUpsampler};
use crate::vae::LtxVideoVae;

/// Distilled stage-1 sigmas (`DEFAULT_STAGE_1_SIGMAS`, 8 denoise steps).
pub const STAGE1_SIGMAS: [f32; 9] = [
    1.0, 0.993_75, 0.987_5, 0.981_25, 0.975, 0.909_375, 0.725, 0.421_875, 0.0,
];
/// Distilled stage-2 sigmas (`DEFAULT_STAGE_2_SIGMAS`, 3 denoise steps). `STAGE2_SIGMAS[0]` is the
/// stage-transition re-noise scale.
pub const STAGE2_SIGMAS: [f32; 4] = [0.909_375, 0.725, 0.421_875, 0.0];

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Force a logically-contiguous copy (see `vae.rs`): host reads (`as_slice`) return the *physical*
/// buffer, so an array left strided by the `(F,H,W,C)` transpose reads scrambled.
fn contiguous(x: &Array) -> Result<Array> {
    let shape = x.shape().to_vec();
    Ok(x.reshape(&[-1])?.reshape(&shape)?)
}

/// The legacy dtype-preserving Euler update (the `use_legacy_euler` branch): for `σ_next > 0`,
/// `x' = denoised + σ_next·(x − denoised)/σ`; at the final step (`σ_next = 0`), `x' = denoised`.
/// Computed in `x`'s dtype (the σ scalars are cast to it) — algebraically `x + (σ_next − σ)·v` but
/// kept in the reference's exact op order/dtype for bit-parity.
pub fn euler_step(x: &Array, denoised: &Array, sigma: f32, sigma_next: f32) -> Result<Array> {
    if sigma_next <= 0.0 {
        return Ok(denoised.clone());
    }
    let dt = x.dtype();
    let sn = scalar(sigma_next).as_dtype(dt)?;
    let sg = scalar(sigma).as_dtype(dt)?;
    let step = divide(&multiply(&sn, &subtract(x, denoised)?)?, &sg)?;
    Ok(add(denoised, &step)?)
}

/// One stage's denoise loop. Distilled: **no CFG**, **legacy Euler**.
///
/// * `latents` — `(B, 128, F, H, W)` NCFHW, the stage's dtype (f32 here, S5 gate). For I2V this is
///   the conditioned + noised [`I2vConditioning::latent`].
/// * `context` — `(B, ctx, inner)` text embeddings (the connector output / S6's text encoder).
/// * `positions` — `(B, 3, S, 2)` position grid for this stage's latent dims.
/// * `sigmas` — the stage schedule; `sigmas.len() − 1` denoise steps.
/// * `state` — `None` for T2V (uniform per-token σ); `Some` for I2V (per-token `σ·mask`, with the
///   denoised output blended toward the clean conditioning each step — the reference `denoise(...,
///   state=...)` path that pins the conditioned frame).
/// * `on_step` — progress callback, fired once per completed step.
pub fn denoise(
    dit: &LtxDiT,
    latents: &Array,
    context: &Array,
    positions: &Array,
    sigmas: &[f32],
    state: Option<&I2vConditioning>,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let dt = latents.dtype();
    let sh = latents.shape();
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
    let num_tokens = f * h * w;
    let mut lat = latents.clone();

    for i in 0..sigmas.len() - 1 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // (B, C, F, H, W) → (B, C, S) → (B, S, C) packed tokens.
        let flat = lat.reshape(&[b, c, -1])?.transpose_axes(&[0, 2, 1])?;
        // Per-token timesteps, shape (B, num_tokens): T2V → uniform σ; I2V → σ·mask (conditioned
        // tokens get 0). Matches the reference `denoise`.
        let ts = match state {
            Some(st) => st.token_timesteps(sigma, h, w)?,
            None => broadcast_to(&scalar(sigma).as_dtype(dt)?, &[b, num_tokens])?,
        };
        let velocity = dit.forward(&flat, &ts, context, None, positions)?;
        // (B, S, C) → (B, C, S) → (B, C, F, H, W).
        let velocity = velocity
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[b, c, f, h, w])?;
        let sig = scalar(sigma).as_dtype(dt)?;
        let mut denoised = to_denoised(&lat, &velocity, &sig)?;
        // I2V: pin the conditioned frame(s) to the clean image latent (reference `apply_denoise_mask`).
        if let Some(st) = state {
            denoised = apply_denoise_mask(&denoised, &st.clean_latent, &st.denoise_mask)?;
        }
        lat = euler_step(&lat, &denoised, sigma, sigma_next)?;
        mlx_rs::transforms::eval([&lat])?;
        on_step(i + 1);
    }
    Ok(lat)
}

/// Stage-transition re-noise: `noise·scale + latent·(1 − scale)`, dtype-preserving. `1 − scale` is
/// computed in `latent`'s dtype (`array(1) − array(scale)`), matching the reference exactly.
pub fn renoise(latents: &Array, noise: &Array, noise_scale: f32) -> Result<Array> {
    let dt = latents.dtype();
    let s = scalar(noise_scale).as_dtype(dt)?;
    let one_minus = subtract(&scalar(1.0).as_dtype(dt)?, &s)?;
    Ok(add(&multiply(noise, &s)?, &multiply(latents, &one_minus)?)?)
}

/// VAE-decode latents `(B, 128, F, H, W)` → `(F, H, W, 3)` uint8 frames. Reference order:
/// squeeze batch → `(F, H, W, 3)` → `clip((x+1)/2, 0, 1)·255` → uint8.
pub fn decode_to_frames(vae: &LtxVideoVae, latents: &Array) -> Result<Array> {
    to_uint8_frames(&vae.decode(latents)?)
}

/// `(B=1, 3, F, H, W)` video in ~[-1, 1] → `(F, H, W, 3)` uint8. The reference clips `(x+1)/2` to
/// `[0, 1]` *before* scaling by 255, so the result saturates at 255 (truncating cast).
pub fn to_uint8_frames(video: &Array) -> Result<Array> {
    let sh = video.shape(); // (1, 3, F, H, W)
    let (c, f, h, w) = (sh[1], sh[2], sh[3], sh[4]);
    let dt = video.dtype();
    let chw = video
        .reshape(&[c, f, h, w])?
        .transpose_axes(&[1, 2, 3, 0])?; // (F, H, W, 3)
    let half = divide(
        &add(&chw, &scalar(1.0).as_dtype(dt)?)?,
        &scalar(2.0).as_dtype(dt)?,
    )?;
    let clipped = minimum(
        &maximum(&half, &scalar(0.0).as_dtype(dt)?)?,
        &scalar(1.0).as_dtype(dt)?,
    )?;
    let scaled = multiply(&clipped, &scalar(255.0).as_dtype(dt)?)?;
    contiguous(&scaled.as_dtype(Dtype::Uint8)?)
}

/// Prepare an I2V conditioning image for VAE encoding (reference `prepare_image_for_encoding` ∘
/// `load_image`): PIL-LANCZOS scale the RGB8 image to the stage pixel resolution `(target_height,
/// target_width)` (a no-op when already sized), normalize `[0,255] → [-1,1]`, and lay out as **NCFHW**
/// `[1, 3, 1, H, W]` f32 — the single-frame video the [`LtxVideoVae::encode`](crate::vae::LtxVideoVae)
/// expects. The reference resizes the *original* image directly to each stage's pixel resolution, so
/// the caller passes `height/2 × width/2` for stage 1 and `height × width` for stage 2.
pub fn preprocess_conditioning_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "I2V conditioning image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // PIL LANCZOS on the uint8 image (no-op when already at target size), matching `load_image`.
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    // /255 then [-1,1], as NHWC.
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let nhwc = Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]);
    // NHWC → NCHW → insert the singleton temporal axis → (1, 3, 1, H, W).
    let nchw = nhwc.transpose_axes(&[0, 3, 1, 2])?; // (1, 3, H, W)
    Ok(nchw.reshape(&[1, 3, 1, th as i32, tw as i32])?)
}

/// The full 2-stage distilled T2V latent pipeline: stage-1 denoise → 2× upsample → re-noise →
/// stage-2 denoise. `stage1_noise`/`stage2_noise` are the (injected) initial + re-noise samples,
/// `context` the shared text embeddings, `*_positions` each stage's grid, `latent_{mean,std}` the VAE
/// `per_channel_statistics`. Returns the final full-res latents `(B, 128, F, H, W)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v_latents(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let lat = denoise(
        dit,
        stage1_noise,
        context,
        stage1_positions,
        &STAGE1_SIGMAS,
        None,
        on_step,
    )?;
    let lat = upsample_latents(&lat, upsampler, latent_mean, latent_std)?;
    let lat = renoise(&lat, stage2_noise, STAGE2_SIGMAS[0])?;
    denoise(
        dit,
        &lat,
        context,
        stage2_positions,
        &STAGE2_SIGMAS,
        None,
        on_step,
    )
}

/// The full 2-stage distilled **I2V** latent pipeline (reference `generate.py` / `generate_av.py`
/// video path with `state`): stage-1 condition + noise + conditioned denoise → 2× upsample → stage-2
/// condition + re-noise + conditioned denoise. Differs from [`generate_t2v_latents`] only in the
/// conditioning state: each stage injects its VAE-encoded image latent at `frame_idx` (clean latent +
/// per-frame `1 − strength` mask), seeds the loop via the [`I2vConditioning::noised`] noiser (so the
/// conditioned frame is pinned and the rest gets the stage's noise), and runs the conditioned denoise.
///
/// * `stage1_image_latent` `(B, 128, 1, h1, w1)` / `stage2_image_latent` `(B, 128, 1, h2, w2)` — the
///   conditioning image VAE-encoded at each stage's latent resolution.
/// * `stage1_noise` / `stage2_noise` — the stage noise (the reference draws fresh `normal`; the
///   parity seam injects the reference samples). The conditioned frame ignores it (mask).
/// * `frame_idx` / `strength` — single-image I2V uses `frame_idx = 0`; `strength = 1.0` fully pins
///   the conditioned frame.
///
/// Returns the final full-res latents `(B, 128, F, h2, w2)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_i2v_latents(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    stage1_image_latent: &Array,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_image_latent: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    frame_idx: i32,
    strength: f32,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    // Stage 1: condition over a zero base, noise (σ₀ = 1.0), conditioned denoise. The image latent is
    // cast to the base/noise dtype (the f32 VAE encoder feeds a bf16 path with a sub-ULP cast — the
    // same post-encode quality island as the VAE decode; a no-op when both are already f32/bf16).
    let zeros1 = Array::zeros::<f32>(stage1_noise.shape())?.as_dtype(stage1_noise.dtype())?;
    let cond1 = stage1_image_latent.as_dtype(zeros1.dtype())?;
    let st1 = apply_conditioning(&zeros1, &cond1, frame_idx, strength)?;
    let st1 = st1.noised(stage1_noise, STAGE1_SIGMAS[0])?;
    let lat = denoise(
        dit,
        &st1.latent,
        context,
        stage1_positions,
        &STAGE1_SIGMAS,
        Some(&st1),
        on_step,
    )?;

    // Upsample 2×.
    let lat = upsample_latents(&lat, upsampler, latent_mean, latent_std)?;

    // Stage 2: condition over the upscaled latent, re-noise (σ₀ = STAGE2_SIGMAS[0]), conditioned denoise.
    let cond2 = stage2_image_latent.as_dtype(lat.dtype())?;
    let st2 = apply_conditioning(&lat, &cond2, frame_idx, strength)?;
    let st2 = st2.noised(stage2_noise, STAGE2_SIGMAS[0])?;
    denoise(
        dit,
        &st2.latent,
        context,
        stage2_positions,
        &STAGE2_SIGMAS,
        Some(&st2),
        on_step,
    )
}

/// [`generate_t2v_latents`] + VAE decode → uint8 frames `(F, H, W, 3)`.
#[allow(clippy::too_many_arguments)]
pub fn generate_t2v(
    dit: &LtxDiT,
    upsampler: &LatentUpsampler,
    vae: &LtxVideoVae,
    stage1_noise: &Array,
    stage1_positions: &Array,
    stage2_noise: &Array,
    stage2_positions: &Array,
    context: &Array,
    latent_mean: &Array,
    latent_std: &Array,
    on_step: &mut dyn FnMut(usize),
) -> Result<Array> {
    let latents = generate_t2v_latents(
        dit,
        upsampler,
        stage1_noise,
        stage1_positions,
        stage2_noise,
        stage2_positions,
        context,
        latent_mean,
        latent_std,
        on_step,
    )?;
    decode_to_frames(vae, &latents)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    #[test]
    fn preprocess_conditioning_image_layout_and_norm() {
        // 1×2 RGB image, white pixel then black pixel (HWC). No-op resize (target == source).
        let image = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };
        let got = preprocess_conditioning_image(&image, 2, 1).unwrap();
        // NCFHW (1, 3, 1, 1, 2): 255 → 1.0, 0 → -1.0; each channel holds [w0=1, w1=-1].
        assert_eq!(got.shape(), &[1, 3, 1, 1, 2]);
        let c = mlx_rs::ops::reshape(&got, &[-1]).unwrap();
        assert_eq!(c.as_slice::<f32>(), &[1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);
    }

    #[test]
    fn preprocess_conditioning_image_resizes_to_target() {
        // 4×4 → 2×2: LANCZOS path (values gated by core image tests); just check the output layout.
        let image = Image {
            width: 4,
            height: 4,
            pixels: vec![128u8; 4 * 4 * 3],
        };
        let got = preprocess_conditioning_image(&image, 2, 2).unwrap();
        assert_eq!(got.shape(), &[1, 3, 1, 2, 2]);
    }

    #[test]
    fn euler_step_matches_reference_formula() {
        // x' = denoised + σ_next·(x − denoised)/σ.
        let x = arr(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let den = arr(&[0.5, 1.0, 1.5, 2.0], &[4]);
        let (sigma, sigma_next) = (0.5_f32, 0.25_f32);
        let got = euler_step(&x, &den, sigma, sigma_next).unwrap();
        let want: Vec<f32> = (0..4)
            .map(|i| {
                let (xv, dv) = (x.as_slice::<f32>()[i], den.as_slice::<f32>()[i]);
                dv + sigma_next * (xv - dv) / sigma
            })
            .collect();
        for (g, w) in got.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "euler {g} vs {w}");
        }
    }

    #[test]
    fn euler_step_final_is_denoised() {
        let x = arr(&[1.0, 2.0], &[2]);
        let den = arr(&[9.0, 8.0], &[2]);
        let got = euler_step(&x, &den, 0.42, 0.0).unwrap();
        assert_eq!(got.as_slice::<f32>(), den.as_slice::<f32>());
    }

    #[test]
    fn renoise_matches_reference_formula() {
        // noise·scale + latent·(1−scale).
        let lat = arr(&[1.0, 2.0, 3.0], &[3]);
        let noise = arr(&[0.0, 1.0, -1.0], &[3]);
        let scale = 0.909_375_f32;
        let got = renoise(&lat, &noise, scale).unwrap();
        let want: Vec<f32> = (0..3)
            .map(|i| {
                let (nv, lv) = (noise.as_slice::<f32>()[i], lat.as_slice::<f32>()[i]);
                nv * scale + lv * (1.0 - scale)
            })
            .collect();
        for (g, w) in got.as_slice::<f32>().iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "renoise {g} vs {w}");
        }
    }

    #[test]
    fn to_uint8_frames_clips_and_scales() {
        // (1,3,1,1,2): values spanning below/within/above [-1,1]. Channel values chosen so
        // (x+1)/2·255 lands on exact integers (no trunc-vs-round ambiguity).
        let video = arr(&[-2.0, -1.0, 0.2, 1.0, 0.6, 2.0], &[1, 3, 1, 1, 2]);
        let frames = to_uint8_frames(&video).unwrap();
        assert_eq!(frames.shape(), &[1, 1, 2, 3]); // (F,H,W,3)
        assert_eq!(frames.dtype(), Dtype::Uint8);
        // Layout (F,H,W,C): channels at w=0 are x=[-2,0.2,0.6]; at w=1 x=[-1,1,2].
        // (x+1)/2 clip[0,1] ·255: -2→0, 0.2→153, 0.6→204, -1→0, 1→255, 2→255.
        let got = frames.as_slice::<u8>();
        assert_eq!(got, &[0, 153, 204, 0, 255, 255]);
    }
}
