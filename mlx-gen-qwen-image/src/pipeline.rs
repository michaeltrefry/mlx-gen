//! Qwen-Image T2I sampling pipeline — ports of the fork's `FluxLatentCreator` (Qwen reuses it),
//! `LinearScheduler` sigma schedule, `QwenImage.compute_guided_noise` (true-CFG with norm
//! correction), the denoise loop (`variants/txt2img/qwen_image.py`), and `ImageUtil.to_image`.
//!
//! Latents live as a **packed** token sequence `[1, (h/16)·(w/16), 64]` throughout the loop
//! (the noise is created already packed, Flux-style), and are unpacked to the VAE's `[1, 16, h/8,
//! w/8]` only at decode. Conditioning runs **two** transformer forwards per step (positive +
//! negative) combined by classifier-free guidance.

use mlx_rs::ops::{add, concatenate_axis, divide, multiply, subtract, sum_axes};
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{CancelFlag, DiffusionSampler, Error, FlowMatchEuler, Image, Progress, Result};

use crate::control_transformer::QwenControlNet;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// VAE latent channel count.
pub const LATENT_CHANNELS: i32 = 16;
/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;
/// 2×2 patchify of the latent into the transformer's `in_channels = 16·4 = 64` token features.
pub const PATCH: u32 = 2;

// fork qwen-image scheduler shift params (`ModelConfig.qwen_image`).
const SIGMA_BASE_SHIFT: f32 = 0.5;
const SIGMA_MAX_SHIFT: f32 = 0.9;
const SIGMA_BASE_SEQ_LEN: f32 = 256.0;
const SIGMA_MAX_SEQ_LEN: f32 = 8192.0;
const SIGMA_SHIFT_TERMINAL: f32 = 0.02;

// The decoded-tensor → Image step is identical across families and now lives in core (F-006);
// re-exported so `crate::pipeline::decoded_to_image` and the crate's public surface are unchanged.
pub use mlx_gen::image::decoded_to_image;

/// Seeded txt2img latent noise — shape `[1, (h/16)·(w/16), 64]`, f32. Port of
/// `FluxLatentCreator.create_noise` (`mx.random.normal` with `key(seed)`); the noise is created
/// *already packed*, so packing is a no-op for T2I. The fork casts to the model precision (bf16)
/// when the latents enter the loop; this returns the raw f32 sample for seeded-RNG parity.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    let key = random::key(seed)?;
    let seq = ((height / 16) * (width / 16)) as i32;
    let shape = [1, seq, (LATENT_CHANNELS * (PATCH * PATCH) as i32)];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Port of `FluxLatentCreator.unpack_latents`: packed tokens `[1, seq, 64]` → VAE latent
/// `[1, 16, h/8, w/8]`. `reshape([1, h/16, w/16, 16, 2, 2]) → transpose(0,3,1,4,2,5) → reshape`.
pub fn unpack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    let (lh, lw) = ((height / 16) as i32, (width / 16) as i32);
    let c = LATENT_CHANNELS;
    let p = PATCH as i32;
    let x = latents.reshape(&[1, lh, lw, c, p, p])?;
    let x = x.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    Ok(x.reshape(&[1, c, lh * p, lw * p])?)
}

/// Port of `FluxLatentCreator.pack_latents` (inverse of [`unpack_latents`]): VAE latent
/// `[1, 16, h/8, w/8]` → packed tokens `[1, (h/16)·(w/16), 64]`. Used by the Qwen-Image-Edit
/// dual-latent path to fold the encoded reference into the transformer's token sequence.
pub fn pack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    let (lh, lw) = ((height / 16) as i32, (width / 16) as i32);
    let c = LATENT_CHANNELS;
    let p = PATCH as i32;
    let x = latents.reshape(&[1, c, lh, p, lw, p])?;
    let x = x.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
    Ok(x.reshape(&[1, lh * lw, c * p * p])?)
}

/// Resolve the img2img start step (the fork's `Config.init_time_step`): for a reference image with
/// `strength` in `(0, 1]`, `max(1, floor(num_steps · strength))`; otherwise `0` (pure txt2img).
/// Higher strength → later start → fewer denoise steps → output stays closer to the init image
/// (the fork's convention). Shared shape with the Z-Image port (sc-2533).
pub fn init_time_step(num_steps: usize, strength: Option<f32>) -> usize {
    match strength {
        Some(s) if s > 0.0 => {
            let s = s.clamp(0.0, 1.0);
            // Python `int(num_steps * strength)` truncates toward zero == floor for s >= 0.
            ((num_steps as f32 * s) as usize).max(1)
        }
        _ => 0,
    }
}

/// Scale an RGB8 init image to `target` dims with PIL LANCZOS (the fork's `scale_to_dimensions`,
/// a no-op when already sized), normalize `[0,255] → [-1,1]`, and lay out as NCHW `[1, 3, H, W]`
/// f32 — the input the VAE encoder expects. Port of `ImageUtil.to_array(scale_to_dimensions(...))`.
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "init image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    // PIL LANCZOS on the uint8 image (no-op when already at target size), matching the fork.
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    // /255 then [-1,1], as NHWC, then transpose to NCHW (the fork's `to_array` convention).
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    let nhwc = Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]);
    Ok(nhwc.transpose_axes(&[0, 3, 1, 2])?)
}

/// img2img init image → packed clean latents `[1, (h/16)·(w/16), 64]` (f32). Port of the fork's
/// `LatentCreator.encode_image` ∘ `QwenLatentCreator.pack_latents`: PIL-LANCZOS scale to the target
/// dims, normalize `[0,255] → [-1,1]` NCHW, VAE-encode (causal-Conv3d → scaled 16-ch latent), drop
/// the temporal axis, and pack. Mirrors the Edit `encode_reference_latents` encode, minus the
/// dual-latent `cond_grid` (T2I img2img blends into the noise rather than concatenating).
pub fn encode_init_latents(vae: &QwenVae, image: &Image, width: u32, height: u32) -> Result<Array> {
    let image_nchw = preprocess_init_image(image, width, height)?; // [1, 3, H, W]
    let latent = vae.encode(&image_nchw)?.squeeze_axes(&[2])?; // [1, 16, 1, H/8, W/8] → [1, 16, H/8, W/8]
    pack_latents(&latent, width, height) // [1, (h/16)·(w/16), 64]
}

/// Port of `LatentCreator.add_noise_by_interpolation`: `(1 - sigma) * clean + sigma * noise`. The
/// img2img blend that seeds the denoise loop at `sigma = sigmas[init_time_step]`.
pub fn add_noise_by_interpolation(clean: &Array, noise: &Array, sigma: f32) -> Result<Array> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    Ok(add(&multiply(clean, one_minus)?, &multiply(noise, s)?)?)
}

/// Qwen-Image's flow-match sigma schedule: `linspace(1, 1/n, n)` run through the exponential
/// time-shift (`mu` from image area) **and** the terminal-sigma rescale, with a trailing `0`.
/// Port of `LinearScheduler._get_sigmas` (the `requires_sigma_shift` + `sigma_shift_terminal`
/// path). The core [`FlowMatchEuler`] uses FLUX's empirical `mu` and no terminal shift, so we
/// build the Vec here and wrap it.
pub fn qwen_scheduler(num_steps: usize, width: u32, height: u32) -> FlowMatchEuler {
    FlowMatchEuler {
        sigmas: qwen_sigmas(num_steps, width, height),
    }
}

fn qwen_sigmas(num_steps: usize, width: u32, height: u32) -> Vec<f32> {
    let n = num_steps.max(1);
    // linspace(1.0, 1.0/n, n)
    let (start, end) = (1.0_f32, 1.0_f32 / n as f32);
    let linspace: Vec<f32> = (0..n)
        .map(|i| {
            if n == 1 {
                start
            } else {
                start + (end - start) * (i as f32) / ((n - 1) as f32)
            }
        })
        .collect();

    let m = (SIGMA_MAX_SHIFT - SIGMA_BASE_SHIFT) / (SIGMA_MAX_SEQ_LEN - SIGMA_BASE_SEQ_LEN);
    let b = SIGMA_BASE_SHIFT - m * SIGMA_BASE_SEQ_LEN;
    let mu = m * (width as f32 * height as f32 / 256.0) + b;
    let e = mu.exp();
    // exp(mu) / (exp(mu) + (1/sigma - 1))
    let mut shifted: Vec<f32> = linspace
        .iter()
        .map(|&s| e / (e + (1.0 / s - 1.0)))
        .collect();

    // terminal-sigma rescale so the last shifted sigma hits `1 - terminal`.
    let one_minus: Vec<f32> = shifted.iter().map(|&s| 1.0 - s).collect();
    let scale = one_minus[one_minus.len() - 1] / (1.0 - SIGMA_SHIFT_TERMINAL);
    for (s, om) in shifted.iter_mut().zip(&one_minus) {
        *s = 1.0 - om / scale;
    }

    shifted.push(0.0);
    shifted
}

/// True classifier-free guidance with norm correction. Port of `QwenImage.compute_guided_noise`:
/// `combined = neg + g·(pos − neg)`, then rescale `combined` to the L2 norm of the positive
/// prediction (over the channel axis). Keeps the guided velocity's magnitude matched to the
/// conditional one — prevents the over-saturation plain CFG would introduce at `g = 4`.
pub fn compute_guided_noise(pos: &Array, neg: &Array, guidance: f32) -> Result<Array> {
    let combined = add(neg, &multiply(&subtract(pos, neg)?, scalar(guidance))?)?;
    let cond_norm = l2_over_channels(pos)?;
    let comb_norm = l2_over_channels(&combined)?;
    Ok(multiply(&combined, &divide(&cond_norm, &comb_norm)?)?)
}

/// `sqrt(sum(x², axis=-1, keepdims) + 1e-12)` — per-token L2 norm over the channel axis.
fn l2_over_channels(x: &Array) -> Result<Array> {
    let last = (x.shape().len() - 1) as i32;
    let sq = sum_axes(&multiply(x, x)?, &[last], true)?;
    Ok(add(&sq, scalar(1e-12))?.sqrt()?)
}

/// Flow-match denoise loop driven by a [`DiffusionSampler`] (the production [`FlowMatchSampler`]
/// wrapping `qwen_scheduler`, or the few-step Lightning schedule — sc-2909), with progress and
/// cooperative cancellation. The sampler owns the schedule (`num_steps`/`timestep`/`step`); Qwen
/// feeds the **raw sigma** ([`DiffusionSampler::timestep`]) as the transformer timestep. Returns
/// the final packed latents.
///
/// `neg_embeds` selects the guidance mode:
/// - `Some(neg)` → **true CFG**: two forwards/step (positive + negative) combined via
///   [`compute_guided_noise`] at `guidance` (the production path).
/// - `None` → **CFG-off**: a single forward/step (the velocity *is* the positive prediction). This
///   is the Lightning fast path — the distillation LoRAs are CFG-distilled, so the negative forward
///   and norm-correction are skipped (a 2× saving on top of the few steps); `guidance` is ignored.
///
/// `start_step` is the fork's `Config.init_time_step`: `0` for txt2img (loop over every step), or
/// [`init_time_step`] for img2img (loop `range(init_time_step, steps)` so the blended init latents
/// are denoised from the matching sigma). Progress reports `steps - start_step` total steps.
#[allow(clippy::too_many_arguments)]
pub fn denoise_with_progress(
    transformer: &QwenTransformer,
    sampler: &dyn DiffusionSampler,
    latents: Array,
    pos_embeds: &Array,
    neg_embeds: Option<&Array>,
    guidance: f32,
    width: u32,
    height: u32,
    start_step: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // sc-2963 (rollout of sc-2957): run the MMDiT's fusable elementwise glue (adaLN affine, gated
    // residual, tanh-GELU FFN, RoPE rotation) through `mx.compile` — bit-exact (`max|Δ|=0`,
    // compile_parity.rs) and a per-step win at production geometry. Process-global, idempotent.
    crate::transformer::set_compile_glue(true);
    let mut latents = latents;
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    let total = (sampler.num_steps() - start_step) as u32;
    for t in start_step..sampler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let sigma = sampler.timestep(t);
        // `None` joint mask: the prompt embeds carry no padding into the transformer, so parity is
        // proven maskless (see `build_joint_mask`).
        let pos = transformer.forward(&latents, pos_embeds, None, sigma, lh, lw, &[])?;
        let velocity = match neg_embeds {
            Some(neg) => {
                let neg = transformer.forward(&latents, neg, None, sigma, lh, lw, &[])?;
                compute_guided_noise(&pos, &neg, guidance)?
            }
            None => pos,
        };
        latents = sampler.step(&velocity, &latents, t)?;
        on_progress(Progress::Step {
            current: (t - start_step) as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Qwen-Image **ControlNet** (strict pose) denoise loop (epic 3401 / sc-3572). Like
/// [`denoise_with_progress`], but each step first runs the control branch
/// ([`QwenControlNet::forward`]) over the current latents + the (constant) packed control image to
/// get the per-block residuals, then runs the base transformer with those residuals injected
/// ([`QwenTransformer::forward_control`]) scaled by `control_scale`. Under true CFG the control
/// branch runs once per guidance branch (positive + negative), mirroring diffusers; the Lightning
/// CFG-off path (`neg_embeds = None`) runs it once. `control_scale = 0` reproduces the base T2I
/// forward (the residuals are zeroed at the injection).
#[allow(clippy::too_many_arguments)]
pub fn denoise_control_with_progress(
    transformer: &QwenTransformer,
    controlnet: &QwenControlNet,
    sampler: &dyn DiffusionSampler,
    latents: Array,
    control_cond: &Array,
    pos_embeds: &Array,
    neg_embeds: Option<&Array>,
    guidance: f32,
    control_scale: f32,
    width: u32,
    height: u32,
    start_step: usize,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    crate::transformer::set_compile_glue(true);
    let mut latents = latents;
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    let total = (sampler.num_steps() - start_step) as u32;
    for t in start_step..sampler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let sigma = sampler.timestep(t);
        let pos_res = controlnet.forward(&latents, control_cond, pos_embeds, sigma, lh, lw)?;
        let pos = transformer.forward_control(
            &latents,
            pos_embeds,
            None,
            sigma,
            lh,
            lw,
            &[],
            Some(&pos_res),
            control_scale,
        )?;
        let velocity = match neg_embeds {
            Some(neg) => {
                let neg_res = controlnet.forward(&latents, control_cond, neg, sigma, lh, lw)?;
                let neg = transformer.forward_control(
                    &latents,
                    neg,
                    None,
                    sigma,
                    lh,
                    lw,
                    &[],
                    Some(&neg_res),
                    control_scale,
                )?;
                compute_guided_noise(&pos, &neg, guidance)?
            }
            None => pos,
        };
        latents = sampler.step(&velocity, &latents, t)?;
        on_progress(Progress::Step {
            current: (t - start_step) as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Qwen-Image-**Edit** dual-latent denoise loop, driven by a [`DiffusionSampler`] (sc-2909). Each
/// step concatenates the noise latents with the (static) packed reference latents over the sequence
/// axis, runs the transformer with the reference `cond_grids` so the RoPE spans `[noise] +
/// references`, slices the velocity back to the noise prefix, then takes an Euler step. Port of
/// `QwenImageEdit.generate_image`'s loop.
///
/// `neg_embeds` selects the guidance mode (as in [`denoise_with_progress`]): `Some(neg)` = true CFG
/// (two forwards/step), `None` = CFG-off single forward (the Lightning fast path — the velocity is
/// the positive prediction; `guidance` is ignored).
#[allow(clippy::too_many_arguments)]
pub fn denoise_edit_with_progress(
    transformer: &QwenTransformer,
    sampler: &dyn DiffusionSampler,
    latents: Array,
    static_image_latents: &Array,
    cond_grids: &[(usize, usize)],
    pos_embeds: &Array,
    neg_embeds: Option<&Array>,
    guidance: f32,
    width: u32,
    height: u32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // sc-2963 (rollout of sc-2957): compiled elementwise glue in the Edit denoise loop too — see
    // `denoise_with_progress`. Bit-exact, process-global, idempotent.
    crate::transformer::set_compile_glue(true);
    let mut latents = latents;
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    let total = sampler.num_steps() as u32;
    for t in 0..sampler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let noise_seq = latents.shape()[1];
        let sigma = sampler.timestep(t);
        let hidden = concatenate_axis(&[&latents, static_image_latents], 1)?;
        // `None` joint mask (as in T2I): the spliced prompt embeds are full-valid.
        let pos = slice_seq(
            &transformer.forward(&hidden, pos_embeds, None, sigma, lh, lw, cond_grids)?,
            noise_seq,
        )?;
        let velocity = match neg_embeds {
            Some(neg) => {
                let neg = slice_seq(
                    &transformer.forward(&hidden, neg, None, sigma, lh, lw, cond_grids)?,
                    noise_seq,
                )?;
                compute_guided_noise(&pos, &neg, guidance)?
            }
            None => pos,
        };
        latents = sampler.step(&velocity, &latents, t)?;
        on_progress(Progress::Step {
            current: t as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Slice the transformer velocity `[1, full_seq, 64]` back to the noise prefix `[1, n, 64]`.
fn slice_seq(x: &Array, n: i32) -> Result<Array> {
    let idx = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n]);
    Ok(x.take_axis(&idx, 1)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_shape_is_packed() {
        let n = create_noise(0, 1024, 1024).unwrap();
        assert_eq!(n.shape(), &[1, 4096, 64]);
    }

    #[test]
    fn unpack_inverts_to_vae_latent_shape() {
        let packed = create_noise(0, 512, 768).unwrap(); // h=768,w=512
        let lat = unpack_latents(&packed, 512, 768).unwrap();
        // [1, 16, h/8, w/8]
        assert_eq!(lat.shape(), &[1, 16, 96, 64]);
    }

    #[test]
    fn pack_is_inverse_of_unpack() {
        let packed = create_noise(7, 512, 768).unwrap(); // [1, 48*32, 64]
        let vae_latent = unpack_latents(&packed, 512, 768).unwrap(); // [1,16,96,64]
        let repacked = pack_latents(&vae_latent, 512, 768).unwrap();
        assert_eq!(repacked.shape(), packed.shape());
        let (a, b) = (repacked.as_slice::<f32>(), packed.as_slice::<f32>());
        assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6));
    }

    #[test]
    fn qwen_schedule_shape_and_terminal() {
        let s = qwen_sigmas(4, 1024, 1024);
        assert_eq!(s.len(), 5);
        assert_eq!(*s.last().unwrap(), 0.0);
        // strictly decreasing over the shifted part (1.0 → … → terminal).
        assert!(s[..4].windows(2).all(|w| w[0] > w[1]));
        // terminal rescale forces the last shifted sigma to the terminal `0.02`.
        assert!((s[3] - 0.02).abs() < 1e-4, "got {}", s[3]);
        // first sigma stays at 1.0 (linspace start, shift fixes 1.0 -> 1.0).
        assert!((s[0] - 1.0).abs() < 1e-4, "got {}", s[0]);
    }

    #[test]
    fn init_time_step_matches_fork() {
        // txt2img: no/zero strength → 0.
        assert_eq!(init_time_step(4, None), 0);
        assert_eq!(init_time_step(4, Some(0.0)), 0);
        // floor(steps·strength), clamped to >= 1.
        assert_eq!(init_time_step(4, Some(0.6)), 2); // floor(2.4)
        assert_eq!(init_time_step(8, Some(0.5)), 4);
        assert_eq!(init_time_step(4, Some(0.1)), 1); // floor(0.4)=0 → max(1)
        assert_eq!(init_time_step(4, Some(1.0)), 4);
        assert_eq!(init_time_step(4, Some(2.0)), 4); // strength clamps to 1.0
    }

    #[test]
    fn blend_endpoints() {
        let clean = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let noise = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 2, 2]);
        // sigma=0 → all clean; sigma=1 → all noise.
        let c = add_noise_by_interpolation(&clean, &noise, 0.0).unwrap();
        let n = add_noise_by_interpolation(&clean, &noise, 1.0).unwrap();
        assert_eq!(c.as_slice::<f32>(), clean.as_slice::<f32>());
        assert_eq!(n.as_slice::<f32>(), noise.as_slice::<f32>());
    }

    #[test]
    fn preprocess_init_image_shape_and_range() {
        // 2×2 RGB, no resize (target == source): pixels map [0,255] → [-1,1] NCHW.
        let img = Image {
            width: 2,
            height: 2,
            pixels: vec![0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255],
        };
        let pre = preprocess_init_image(&img, 2, 2).unwrap();
        assert_eq!(pre.shape(), &[1, 3, 2, 2]);
        let v = pre.as_slice::<f32>();
        assert!(v.iter().all(|&x| (-1.0..=1.0).contains(&x)));
        // first pixel (0,0,0) → -1 across channels; channel-planar NCHW so index 0,4,8 are R,G,B@(0,0).
        assert!((v[0] + 1.0).abs() < 1e-6);
    }

    #[test]
    fn guided_noise_matches_positive_norm() {
        // when pos == neg, guided == pos (combined == pos, norm ratio == 1).
        let pos = Array::from_slice(&[3.0f32, 4.0, 0.0, 0.0], &[1, 2, 2]);
        let g = compute_guided_noise(&pos, &pos, 4.0).unwrap();
        let got = g.as_slice::<f32>();
        let want = pos.as_slice::<f32>();
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
