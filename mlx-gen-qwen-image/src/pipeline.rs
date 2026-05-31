//! Qwen-Image T2I sampling pipeline — ports of the fork's `FluxLatentCreator` (Qwen reuses it),
//! `LinearScheduler` sigma schedule, `QwenImage.compute_guided_noise` (true-CFG with norm
//! correction), the denoise loop (`variants/txt2img/qwen_image.py`), and `ImageUtil.to_image`.
//!
//! Latents live as a **packed** token sequence `[1, (h/16)·(w/16), 64]` throughout the loop
//! (the noise is created already packed, Flux-style), and are unpacked to the VAE's `[1, 16, h/8,
//! w/8]` only at decode. Conditioning runs **two** transformer forwards per step (positive +
//! negative) combined by classifier-free guidance.

use mlx_rs::ops::{add, divide, maximum, minimum, multiply, round, subtract, sum_axes};
use mlx_rs::{random, Array};

use mlx_gen::{CancelFlag, Error, FlowMatchEuler, Image, Progress, Result};

use crate::transformer::QwenTransformer;

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

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

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

/// Flow-match Euler denoise loop with classifier-free guidance, progress, and cooperative
/// cancellation. Each step runs the transformer twice (positive + negative conditioning),
/// combines via [`compute_guided_noise`], and takes an Euler step. The fork passes the **raw
/// sigma** (`scheduler.sigmas[t]`) as the transformer timestep. Returns the final packed latents.
#[allow(clippy::too_many_arguments)]
pub fn denoise_with_progress(
    transformer: &QwenTransformer,
    scheduler: &FlowMatchEuler,
    latents: Array,
    pos_embeds: &Array,
    neg_embeds: &Array,
    guidance: f32,
    width: u32,
    height: u32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    let mut latents = latents;
    let (lh, lw) = ((height / 16) as usize, (width / 16) as usize);
    let total = scheduler.num_steps() as u32;
    for t in 0..scheduler.num_steps() {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let sigma = scheduler.sigmas[t];
        let pos = transformer.forward(&latents, pos_embeds, None, sigma, lh, lw)?;
        let neg = transformer.forward(&latents, neg_embeds, None, sigma, lh, lw)?;
        let guided = compute_guided_noise(&pos, &neg, guidance)?;
        latents = scheduler.step(&latents, &guided, t)?;
        on_progress(Progress::Step {
            current: t as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Decoded VAE tensor → RGB8 [`Image`]. Mirrors the fork's `ImageUtil`: denormalize
/// `clip(x/2 + 0.5, 0, 1)`, drop the temporal axis (5-D → 4-D), `NCHW → NHWC`, then
/// `(x*255).round()` as `uint8`, taking the first batch element.
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let half = scalar(0.5);
    let x = add(&multiply(decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    let x = if x.shape().len() == 5 {
        x.squeeze_axes(&[2])?
    } else {
        x
    };
    let x = x.transpose_axes(&[0, 2, 3, 1])?; // NCHW -> NHWC
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;

    let sh = x.shape();
    let (h, w, c) = (sh[1] as u32, sh[2] as u32, sh[3] as u32);
    let n = (h * w * c) as usize;
    // `transpose_axes` yields a strided view; `reshape` re-materializes C-order so the slice is
    // logical NHWC. Take batch 0.
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
