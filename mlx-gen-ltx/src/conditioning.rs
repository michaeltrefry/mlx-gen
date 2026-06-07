//! I2V latent conditioning — port of the reference `mlx_video/conditioning/latent.py`. Injects a
//! VAE-encoded conditioning image as a **clean latent** at a chosen frame index and drives the denoise
//! loop with a per-frame **denoise mask** so the conditioned frame is preserved while the rest is
//! generated. Used by both stages of the I2V pipeline ([`crate::pipeline`]).
//!
//! The shape convention matches the rest of the VAE/pipeline: latents are **NCFHW**
//! `(B, 128, F, H, W)`; the mask is `(B, 1, F, 1, 1)` (one value per latent frame, broadcast over
//! channels + space). `1.0` = full denoise (generate), `0.0` = keep the clean conditioning. A
//! conditioning at `frame_idx` with `strength s` sets the mask there to `1 − s` (so `s = 1.0` →
//! mask 0 → the frame is fully pinned to the image latent; `s = 0.0` → mask 1 → no effect).
//!
//! Reference `generate.py` / `generate_av.py` wire exactly **one** image at **one** frame (default 0).
//! [`apply_conditioning`] keeps the general per-frame structure (a clean latent of `cond_f ≥ 1`
//! frames at any index) so the parity-plus multi-keyframe / first-last-frame extension is mechanically
//! reachable, but the [`crate::model`] Generator only wires the single-image case (strict parity).
//!
//! Everything is **dtype-preserving** (the `mx.array(1.0, dtype)` pattern from the reference): the
//! conditioning state, the noiser, and the mask all stay in the latent's dtype so the I2V path is
//! bit-exact to the reference at both `f32` and `bf16`.

use mlx_rs::ops::{add, broadcast_to, concatenate_axis, multiply, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::{Error, Result};

/// A scalar in `dt` (the dtype-preserving `mx.array(v, dtype=…)`).
fn scalar(v: f32, dt: Dtype) -> Result<Array> {
    Ok(Array::from_slice(&[v], &[1]).as_dtype(dt)?)
}

/// Temporal slice `x[:, :, i:i+1]` (a single latent frame, axis 2).
fn frame(x: &Array, i: i32) -> Result<Array> {
    let idx = Array::from_slice(&[i], &[1]);
    Ok(x.take_axis(idx, 2)?)
}

/// The I2V conditioning state (reference `LatentState`): the current (noised) latent, the clean
/// conditioning latent, and the per-frame denoise mask. `clean_latent` + `denoise_mask` are fixed
/// across the denoise loop; only `latent` evolves (it seeds the loop).
#[derive(Clone)]
pub struct I2vConditioning {
    /// Current latent `(B, C, F, H, W)` — seeds the denoise loop (already noised by [`Self::noised`]).
    pub latent: Array,
    /// Clean conditioning latent `(B, C, F, H, W)`: the image latent at the conditioned frame(s),
    /// zeros elsewhere. [`crate::pipeline::denoise`] blends toward this where the mask is `< 1`.
    pub clean_latent: Array,
    /// Per-frame denoise mask `(B, 1, F, 1, 1)`: `1 − strength` at the conditioned frame(s), `1`
    /// elsewhere.
    pub denoise_mask: Array,
}

/// One replace-latent keyframe: a clean `(B, C, cond_f, H, W)` latent pinned at output latent frame
/// `frame_idx` with `strength` (mask `1 − strength`). For single-image I2V `cond_f = 1`.
#[derive(Clone, Copy)]
pub struct Keyframe<'a> {
    pub latent: &'a Array,
    pub frame_idx: i32,
    pub strength: f32,
}

/// Build the conditioning state by injecting `cond_latent` (a clean `(B, C, cond_f, H, W)` latent —
/// for single-image I2V `cond_f = 1`) at `frame_idx` over `base_latent` `(B, C, F, H, W)`. The
/// single-keyframe form of [`apply_keyframes`] (strict-parity I2V; reference `apply_conditioning`).
pub fn apply_conditioning(
    base_latent: &Array,
    cond_latent: &Array,
    frame_idx: i32,
    strength: f32,
) -> Result<I2vConditioning> {
    apply_keyframes(
        base_latent,
        &[Keyframe {
            latent: cond_latent,
            frame_idx,
            strength,
        }],
    )
}

/// Build the conditioning state by injecting **multiple** clean keyframe latents at their frame
/// indices over `base_latent` `(B, C, F, H, W)` — the replace-latent mechanism (reference
/// `VideoConditionByLatentIndex` applied per item; **first_last_frame** = two keyframes at `0` and the
/// last latent frame). Mirrors the reference's per-item `apply_to`: each keyframe **overwrites** the
/// `latent` + `clean_latent` and sets the `denoise_mask` to `1 − strength` over its covered frames;
/// uncovered frames keep `base_latent` (latent), `0` (clean), `1` (mask). When two keyframes overlap,
/// the **later** one in the list wins (sequential application, matching torch).
///
/// Because this only rewrites existing grid frames in place (no appended tokens), the resulting state
/// drives the **existing grid** [`crate::pipeline::denoise`] / [`crate::pipeline::denoise_av`] loops
/// unchanged — FLF needs no token-native loop.
pub fn apply_keyframes(base_latent: &Array, keyframes: &[Keyframe]) -> Result<I2vConditioning> {
    let dt = base_latent.dtype();
    let sh = base_latent.shape(); // (B, C, F, H, W)
    let (b, c, f, h, w) = (sh[0], sh[1], sh[2], sh[3], sh[4]);

    let mask_gen = broadcast_to(
        &scalar(1.0, dt)?.reshape(&[1, 1, 1, 1, 1])?,
        &[b, 1, 1, 1, 1],
    )?;
    let zero_frame = broadcast_to(
        &scalar(0.0, dt)?.reshape(&[1, 1, 1, 1, 1])?,
        &[b, c, 1, h, w],
    )?;

    // Per-output-frame assignment: which keyframe (if any) owns this frame, and its source sub-index.
    // Later keyframes override earlier ones (sequential `apply_to`).
    let mut owner: Vec<Option<(usize, i32)>> = vec![None; f as usize];
    for (ki, kf) in keyframes.iter().enumerate() {
        let cs = kf.latent.shape();
        let (cond_c, cond_f, cond_h, cond_w) = (cs[1], cs[2], cs[3], cs[4]);
        if (cond_c, cond_h, cond_w) != (c, h, w) {
            return Err(Error::Msg(format!(
                "keyframe {ki} latent spatial shape ({cond_c},{cond_h},{cond_w}) != target ({c},{h},{w})"
            )));
        }
        if kf.frame_idx < 0 || kf.frame_idx >= f {
            return Err(Error::Msg(format!(
                "keyframe {ki} frame index {} out of bounds for {f} latent frames",
                kf.frame_idx
            )));
        }
        let end_idx = (kf.frame_idx + cond_f).min(f);
        for i in kf.frame_idx..end_idx {
            owner[i as usize] = Some((ki, i - kf.frame_idx));
        }
    }

    let mut latent_frames = Vec::with_capacity(f as usize);
    let mut clean_frames = Vec::with_capacity(f as usize);
    let mut mask_frames = Vec::with_capacity(f as usize);
    for i in 0..f {
        match owner[i as usize] {
            Some((ki, sub)) => {
                let kf = &keyframes[ki];
                let cond = frame(kf.latent, sub)?;
                latent_frames.push(cond.clone());
                clean_frames.push(cond);
                let mask_keep = broadcast_to(
                    &scalar(1.0 - kf.strength, dt)?.reshape(&[1, 1, 1, 1, 1])?,
                    &[b, 1, 1, 1, 1],
                )?;
                mask_frames.push(mask_keep);
            }
            None => {
                latent_frames.push(frame(base_latent, i)?);
                clean_frames.push(zero_frame.clone());
                mask_frames.push(mask_gen.clone());
            }
        }
    }

    let latent = concatenate_axis(&latent_frames.iter().collect::<Vec<_>>(), 2)?;
    let clean_latent = concatenate_axis(&clean_frames.iter().collect::<Vec<_>>(), 2)?;
    let denoise_mask = concatenate_axis(&mask_frames.iter().collect::<Vec<_>>(), 2)?;
    Ok(I2vConditioning {
        latent,
        clean_latent,
        denoise_mask,
    })
}

impl I2vConditioning {
    /// Apply the stage-entry noiser (reference: `noise·(mask·scale) + latent·(1 − mask·scale)`), in
    /// the latent dtype. At conditioned frames (`mask = 1 − strength`, `0` when `strength = 1`) the
    /// clean image latent is preserved; elsewhere (`mask = 1`) this is the plain `noise·scale +
    /// latent·(1 − scale)` re-noise. Returns a new state with `latent` replaced (`clean_latent` +
    /// `denoise_mask` unchanged).
    pub fn noised(&self, noise: &Array, noise_scale: f32) -> Result<Self> {
        let dt = self.latent.dtype();
        let scale = scalar(noise_scale, dt)?;
        let scaled_mask = multiply(&self.denoise_mask, &scale)?; // (B,1,F,1,1)
        let one_minus = subtract(&scalar(1.0, dt)?, &scaled_mask)?;
        let latent = add(
            &multiply(noise, &scaled_mask)?,
            &multiply(&self.latent, &one_minus)?,
        )?;
        Ok(Self {
            latent,
            clean_latent: self.clean_latent.clone(),
            denoise_mask: self.denoise_mask.clone(),
        })
    }

    /// Per-token timesteps `σ·mask` shaped `(B, num_tokens)` for the DiT (reference: conditioned
    /// tokens get timestep `0`, the rest `σ`). The mask `(B,1,F,1,1)` is broadcast to `(B,1,F,H,W)`
    /// then flattened to token order `F·H·W`.
    pub fn token_timesteps(&self, sigma: f32, h: i32, w: i32) -> Result<Array> {
        let dt = self.latent.dtype();
        let ms = self.denoise_mask.shape(); // (B,1,F,1,1)
        let (b, f) = (ms[0], ms[2]);
        let mask_flat =
            broadcast_to(&self.denoise_mask, &[b, 1, f, h, w])?.reshape(&[b, f * h * w])?;
        Ok(multiply(&scalar(sigma, dt)?, &mask_flat)?)
    }
}

// ===================================================================================================
// Keyframe-append (IC-LoRA in-context) conditioning — extend_clip / video_bridge / replace_person.
// ===================================================================================================
//
// Port of `VideoConditionByKeyframeIndex.apply_to`: instead of overwriting grid frames in place (the
// replace-latent path above), the conditioning clip's VAE latents are **appended** as extra in-context
// tokens at the end of the token sequence, with their own RoPE positions (frame axis offset by
// `frame_idx`) and a `1 − strength` denoise mask. The target tokens attend to them; an IC-LoRA adapter
// is what teaches the DiT to use them. This is token-native: it operates on the flat `(B, S, C)` token
// sequence (the LTX DiT forward is fully token+positions driven), so the appended tokens never need to
// form a grid. Used by the stage-1 [`crate::pipeline::denoise_av_tokens`] loop.

/// A token-native video latent state (reference `LatentState` for the video stream): the latent as a
/// flat token sequence `(B, S, C)`, the matching per-token `clean_latent` `(B, S, C)`, `denoise_mask`
/// `(B, S, 1)` (`1` = generate, `1 − strength` at conditioning tokens), and `positions` `(B, 3, S, 2)`.
#[derive(Clone)]
pub struct VideoTokenState {
    pub latent: Array,
    pub clean_latent: Array,
    pub denoise_mask: Array,
    pub positions: Array,
    /// The target token count (the first `target_tokens` tokens are the generated grid; the rest are
    /// appended conditioning). `unpatchify` reads exactly these back into a grid.
    pub target_tokens: i32,
}

/// Patchify a latent grid `(B, C, F, H, W)` → tokens `(B, F·H·W, C)` (patch size 1, the reference
/// `VideoLatentPatchifier.patchify`: `b c f h w -> b (f h w) c`).
pub fn patchify_grid(grid: &Array) -> Result<Array> {
    let sh = grid.shape(); // (B, C, F, H, W)
    let (b, c) = (sh[0], sh[1]);
    Ok(grid.reshape(&[b, c, -1])?.transpose_axes(&[0, 2, 1])?)
}

/// Inverse of [`patchify_grid`] for the generated grid: tokens `(B, F·H·W, C)` → `(B, C, F, H, W)`.
pub fn unpatchify_grid(tokens: &Array, c: i32, f: i32, h: i32, w: i32) -> Result<Array> {
    let b = tokens.shape()[0];
    Ok(tokens
        .transpose_axes(&[0, 2, 1])?
        .reshape(&[b, c, f, h, w])?)
}

impl VideoTokenState {
    /// The base (T2V) token state over a noise grid `(B, C, F, H, W)` with its main `positions`
    /// `(B, 3, F·H·W, 2)`: latent = flattened noise, clean = 0, denoise_mask = 1 (all-generate).
    pub fn base(noise_grid: &Array, positions: &Array) -> Result<Self> {
        let dt = noise_grid.dtype();
        let latent = patchify_grid(noise_grid)?;
        let s = latent.shape()[1];
        let b = latent.shape()[0];
        let clean_latent = Array::zeros::<f32>(latent.shape())?.as_dtype(dt)?;
        let denoise_mask = broadcast_to(&scalar(1.0, dt)?.reshape(&[1, 1, 1])?, &[b, s, 1])?;
        Ok(Self {
            latent,
            clean_latent,
            denoise_mask,
            positions: positions.clone(),
            target_tokens: s,
        })
    }

    /// Per-token timesteps `σ · denoise_mask` shaped `(B, S)` for the DiT (conditioning tokens get
    /// `σ·(1−strength)`, fully-pinned `strength=1` → `0`).
    pub fn token_timesteps(&self, sigma: f32) -> Result<Array> {
        let dt = self.latent.dtype();
        let sh = self.denoise_mask.shape(); // (B, S, 1)
        let flat = self.denoise_mask.reshape(&[sh[0], sh[1]])?;
        Ok(multiply(&scalar(sigma, dt)?, &flat)?)
    }
}

/// Build the RoPE positions for an appended keyframe clip of latent shape `(cf, h, w)` placed at
/// `frame_idx` — port of `VideoConditionByKeyframeIndex`'s `get_pixel_coords(get_patch_grid_bounds…)`
/// then `+= frame_idx`, `÷ fps` on the frame axis. The causal first-frame fix is applied **only when
/// `frame_idx == 0`** (matching the reference). Output `(1, 3, cf·h·w, 2)`, f32, token order C-major
/// over `(frame, height, width)` with `[start, end]` last. Spatial axes are not divided by fps.
pub fn keyframe_append_positions(
    cf: usize,
    h: usize,
    w: usize,
    frame_idx: i32,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
) -> Array {
    let hw = h * w;
    let num = cf * hw;
    let causal = frame_idx == 0;
    let mut data = vec![0f32; 3 * num * 2];
    for p in 0..num {
        let t = (p / hw) as i64;
        let rem = p % hw;
        let hh = (rem / w) as i64;
        let ww = (rem % w) as i64;
        for e in 0..2i64 {
            // frame axis: latent·scale → causal-fix (frame_idx==0 only) → += frame_idx (int) → /fps.
            let mut frame_pix = (t + e) * temporal_scale;
            if causal {
                frame_pix = (frame_pix + 1 - temporal_scale).max(0);
            }
            frame_pix += frame_idx as i64;
            let frame_f = frame_pix as f32 / fps;
            let height_f = ((hh + e) * spatial_scale) as f32;
            let width_f = ((ww + e) * spatial_scale) as f32;
            let base = p * 2 + e as usize;
            data[base] = frame_f;
            data[base + num * 2] = height_f;
            data[base + 2 * num * 2] = width_f;
        }
    }
    Array::from_slice(&data, &[1, 3, num as i32, 2])
}

/// Append a keyframe clip to a [`VideoTokenState`] — the IC-LoRA in-context conditioning op (reference
/// `VideoConditionByKeyframeIndex.apply_to`). `clip_latent` is the VAE-encoded clip `(B, C, cf, h, w)`
/// at the **target** spatial resolution; it is patchified and concatenated onto `latent`/`clean_latent`
/// (token axis), with `denoise_mask = 1 − strength` and positions from [`keyframe_append_positions`].
#[allow(clippy::too_many_arguments)]
pub fn append_keyframe_clip(
    state: &VideoTokenState,
    clip_latent: &Array,
    frame_idx: i32,
    strength: f32,
    temporal_scale: i64,
    spatial_scale: i64,
    fps: f32,
) -> Result<VideoTokenState> {
    let dt = state.latent.dtype();
    let cs = clip_latent.shape(); // (B, C, cf, h, w)
    let (b, cf, h, w) = (cs[0], cs[2] as usize, cs[3] as usize, cs[4] as usize);
    let tokens = patchify_grid(&clip_latent.as_dtype(dt)?)?; // (B, cf·h·w, C)
    let n = tokens.shape()[1];
    let denoise_mask = broadcast_to(
        &scalar(1.0 - strength, dt)?.reshape(&[1, 1, 1])?,
        &[b, n, 1],
    )?;
    let positions =
        keyframe_append_positions(cf, h, w, frame_idx, temporal_scale, spatial_scale, fps);
    let positions = if b > 1 {
        broadcast_to(&positions, &[b, 3, n, 2])?
    } else {
        positions
    };
    Ok(VideoTokenState {
        latent: concatenate_axis(&[&state.latent, &tokens], 1)?,
        clean_latent: concatenate_axis(&[&state.clean_latent, &tokens], 1)?,
        denoise_mask: concatenate_axis(&[&state.denoise_mask, &denoise_mask], 1)?,
        positions: concatenate_axis(&[&state.positions, &positions], 2)?,
        target_tokens: state.target_tokens,
    })
}

/// Blend a denoised latent toward the clean conditioning by the mask (reference `apply_denoise_mask`):
/// `denoised·mask + clean·(1 − mask)`. Where `mask = 0` (a fully conditioned frame) the output is the
/// clean image latent; where `mask = 1` it is the denoised generation.
pub fn apply_denoise_mask(denoised: &Array, clean: &Array, mask: &Array) -> Result<Array> {
    let dt = denoised.dtype();
    let one_minus = subtract(&scalar(1.0, dt)?, mask)?;
    Ok(add(
        &multiply(denoised, mask)?,
        &multiply(clean, &one_minus)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr(v: &[f32], shape: &[i32]) -> Array {
        Array::from_slice(v, shape)
    }

    #[test]
    fn apply_conditioning_pins_frame_and_builds_mask() {
        // base (1,1,3,1,1) = [10,20,30]; cond (1,1,1,1,1) = [99] at frame_idx=1, strength=0.75.
        let base = arr(&[10.0, 20.0, 30.0], &[1, 1, 3, 1, 1]);
        let cond = arr(&[99.0], &[1, 1, 1, 1, 1]);
        let st = apply_conditioning(&base, &cond, 1, 0.75).unwrap();
        // latent: frame 1 replaced by the cond, others keep base.
        assert_eq!(st.latent.as_slice::<f32>(), &[10.0, 99.0, 30.0]);
        // clean: cond at frame 1, zeros elsewhere.
        assert_eq!(st.clean_latent.as_slice::<f32>(), &[0.0, 99.0, 0.0]);
        // mask: 1 - strength at frame 1, 1 elsewhere.
        assert_eq!(st.denoise_mask.shape(), &[1, 1, 3, 1, 1]);
        let m = st.denoise_mask.as_slice::<f32>();
        assert!((m[0] - 1.0).abs() < 1e-6);
        assert!((m[1] - 0.25).abs() < 1e-6);
        assert!((m[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apply_keyframes_pins_first_and_last_frame() {
        // first_last_frame: base (1,1,4,1,1)=[10,20,30,40]; keyframe A=[99] @0 s=1.0,
        // keyframe B=[88] @3 s=0.5. Frames 1,2 stay base.
        let base = arr(&[10.0, 20.0, 30.0, 40.0], &[1, 1, 4, 1, 1]);
        let a = arr(&[99.0], &[1, 1, 1, 1, 1]);
        let bb = arr(&[88.0], &[1, 1, 1, 1, 1]);
        let st = apply_keyframes(
            &base,
            &[
                Keyframe {
                    latent: &a,
                    frame_idx: 0,
                    strength: 1.0,
                },
                Keyframe {
                    latent: &bb,
                    frame_idx: 3,
                    strength: 0.5,
                },
            ],
        )
        .unwrap();
        assert_eq!(st.latent.as_slice::<f32>(), &[99.0, 20.0, 30.0, 88.0]);
        assert_eq!(st.clean_latent.as_slice::<f32>(), &[99.0, 0.0, 0.0, 88.0]);
        // mask: 1-1.0=0 @0; 1 @1,2; 1-0.5=0.5 @3.
        let m = st.denoise_mask.as_slice::<f32>();
        assert!((m[0] - 0.0).abs() < 1e-6);
        assert!((m[1] - 1.0).abs() < 1e-6);
        assert!((m[2] - 1.0).abs() < 1e-6);
        assert!((m[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn apply_keyframes_later_overrides_on_overlap() {
        // Two keyframes both at frame 0; the later (in list) wins.
        let base = arr(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let a = arr(&[5.0], &[1, 1, 1, 1, 1]);
        let bb = arr(&[7.0], &[1, 1, 1, 1, 1]);
        let st = apply_keyframes(
            &base,
            &[
                Keyframe {
                    latent: &a,
                    frame_idx: 0,
                    strength: 1.0,
                },
                Keyframe {
                    latent: &bb,
                    frame_idx: 0,
                    strength: 1.0,
                },
            ],
        )
        .unwrap();
        assert_eq!(st.latent.as_slice::<f32>(), &[7.0, 2.0]);
    }

    #[test]
    fn noiser_pins_full_strength_frame() {
        // strength=1 → mask 0 at frame 0 → that frame keeps the clean latent regardless of noise.
        let base = arr(&[0.0, 0.0], &[1, 1, 2, 1, 1]);
        let cond = arr(&[7.0], &[1, 1, 1, 1, 1]);
        let st = apply_conditioning(&base, &cond, 0, 1.0).unwrap();
        let noise = arr(&[5.0, 5.0], &[1, 1, 2, 1, 1]);
        // scale 1.0 (stage-1 σ₀). frame 0: scaled_mask 0 → 5·0 + 7·1 = 7 (pinned); frame 1: mask 1 →
        // 5·1 + 0·0 = 5.
        let noised = st.noised(&noise, 1.0).unwrap();
        assert_eq!(noised.latent.as_slice::<f32>(), &[7.0, 5.0]);
    }

    #[test]
    fn token_timesteps_zero_at_conditioned_frame() {
        // 2 frames, 1x1 spatial → 2 tokens; strength=1 → frame0 timestep 0, frame1 = sigma.
        let base = arr(&[0.0, 0.0], &[1, 1, 2, 1, 1]);
        let cond = arr(&[1.0], &[1, 1, 1, 1, 1]);
        let st = apply_conditioning(&base, &cond, 0, 1.0).unwrap();
        let ts = st.token_timesteps(0.9, 1, 1).unwrap();
        assert_eq!(ts.shape(), &[1, 2]);
        assert_eq!(ts.as_slice::<f32>(), &[0.0, 0.9]);
    }

    #[test]
    fn keyframe_append_positions_frame0_matches_main_grid() {
        // frame_idx=0 with causal fix == the main grid's frame-0 positions (causal-fixed). cf=1,h=1,w=2.
        let p = keyframe_append_positions(1, 1, 2, 0, 8, 32, 24.0);
        assert_eq!(p.shape(), &[1, 3, 2, 2]);
        let v = p.as_slice::<f32>();
        // frame axis (d=0): start clip(0+1-8,0)=0 → 0/24; end clip(8+1-8,0)=1 → 1/24. Same for both w.
        let at = |d: usize, tok: usize, e: usize| v[(d * 2 + tok) * 2 + e];
        assert!((at(0, 0, 0) - 0.0).abs() < 1e-7);
        assert!((at(0, 0, 1) - 1.0 / 24.0).abs() < 1e-7);
        // height axis (d=1): start 0, end 32.
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 1), 32.0);
        // width axis (d=2): token0 w=0 → [0,32]; token1 w=1 → [32,64].
        assert_eq!(at(2, 0, 0), 0.0);
        assert_eq!(at(2, 1, 0), 32.0);
        assert_eq!(at(2, 1, 1), 64.0);
    }

    #[test]
    fn keyframe_append_positions_offset_frame_no_causal() {
        // frame_idx=3 (>0): NO causal fix; frame = (t*8) + 3, /fps. cf=1,h=1,w=1.
        let p = keyframe_append_positions(1, 1, 1, 3, 8, 32, 24.0);
        let v = p.as_slice::<f32>();
        // frame start = (0+3)/24; end = (8+3)/24.
        assert!((v[0] - 3.0 / 24.0).abs() < 1e-7);
        assert!((v[1] - 11.0 / 24.0).abs() < 1e-7);
    }

    #[test]
    fn append_keyframe_clip_extends_tokens_and_mask() {
        // base grid (1,2,1,1,1) → 1 target token; append a 1-frame clip (1,2,1,1,1) at frame 0 s=1.0.
        let noise = arr(&[3.0, 4.0], &[1, 2, 1, 1, 1]);
        let pos = crate::positions::create_position_grid(1, 1, 1, 1);
        let st = VideoTokenState::base(&noise, &pos).unwrap();
        assert_eq!(st.latent.shape(), &[1, 1, 2]); // (B, S=1, C=2)
        assert_eq!(st.target_tokens, 1);

        let clip = arr(&[7.0, 9.0], &[1, 2, 1, 1, 1]);
        let st2 = append_keyframe_clip(&st, &clip, 0, 1.0, 8, 32, 24.0).unwrap();
        // S grows by the clip's token count (1).
        assert_eq!(st2.latent.shape(), &[1, 2, 2]);
        assert_eq!(st2.positions.shape(), &[1, 3, 2, 2]);
        assert_eq!(st2.denoise_mask.shape(), &[1, 2, 1]);
        assert_eq!(st2.target_tokens, 1); // unchanged
                                          // appended latent token == clip tokens; appended mask = 1-strength = 0.
        let lat = st2.latent.as_slice::<f32>(); // (1,2,2): [tok0=[3,4], tok1=[7,9]]
        assert_eq!(&lat[2..4], &[7.0, 9.0]);
        let m = st2.denoise_mask.as_slice::<f32>();
        assert!((m[0] - 1.0).abs() < 1e-6); // target token: generate
        assert!((m[1] - 0.0).abs() < 1e-6); // appended cond token: pinned
    }

    #[test]
    fn apply_denoise_mask_blends() {
        // mask 0 → clean; mask 1 → denoised; mask 0.5 → midpoint.
        let denoised = arr(&[10.0, 10.0, 10.0], &[3]);
        let clean = arr(&[2.0, 2.0, 2.0], &[3]);
        let mask = arr(&[0.0, 1.0, 0.5], &[3]);
        let got = apply_denoise_mask(&denoised, &clean, &mask).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[2.0, 10.0, 6.0]);
    }
}
