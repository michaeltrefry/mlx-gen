//! SDXL T2I sampling pipeline — composes the dual-CLIP conditioning, the seeded prior, the
//! Euler-Ancestral denoise loop with real classifier-free guidance, and the VAE decode. Port of the
//! vendored `StableDiffusionXL.generate_latents` + `_denoising_loop` + `decode`.
//!
//! The U-Net, text encoders, sampler, and CFG run **fp16**, matching the production reference
//! (`StableDiffusionXL(float16=True)`); the VAE runs f32 (it promotes the f16 latents on decode).
//! The RNG is seeded once per image, then the sampler draws the prior + per-step ancestral noise from
//! the global stream — reproducing the reference's exact noise sequence for a seed.

use mlx_rs::ops::{add, concatenate_axis, maximum, minimum, multiply, round, subtract};
use mlx_rs::{random, Array};

use mlx_gen::array::scalar;
use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::{CancelFlag, Error, Image, Progress, Result};

use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// VAE spatial downscale (latent is image/8 per side).
pub const SPATIAL_SCALE: u32 = 8;
/// Latent channel count.
pub const LATENT_CHANNELS: i32 = 4;

/// The SDXL micro-conditioning `time_ids`, hardcoded `[512, 512, 0, 0, 512, 512]` per row — the
/// vendored `StableDiffusionXL.generate_latents` quirk (it does NOT pass the real
/// original/target sizes). Reproduced verbatim for parity. `batch` rows.
pub fn text_time_ids(batch: i32) -> Array {
    let row = [512.0f32, 512.0, 0.0, 0.0, 512.0, 512.0];
    let mut v = Vec::with_capacity(batch as usize * 6);
    for _ in 0..batch {
        v.extend_from_slice(&row);
    }
    Array::from_slice(&v, &[batch, 6])
}

/// Run both CLIP encoders over the (CFG) token batch and assemble the SDXL conditioning:
/// `concat(te1.hidden[-2], te2.hidden[-2])` and `te2.pooled`. `tokens` is `[B, N]` (B=2 with CFG).
pub fn encode_conditioning(
    te1: &ClipTextEncoder,
    te2: &ClipTextEncoder,
    tokens: &Array,
) -> Result<(Array, Array)> {
    let o1 = te1.forward(tokens)?;
    let o2 = te2.forward(tokens)?;
    let h1 = &o1.hidden_states[o1.hidden_states.len() - 2];
    let h2 = &o2.hidden_states[o2.hidden_states.len() - 2];
    let conditioning = concatenate_axis(&[h1, h2], -1)?;
    Ok((conditioning, o2.pooled))
}

/// Components needed for one denoise run (borrowed from the loaded model).
pub struct Denoiser<'a> {
    pub unet: &'a UNet2DConditionModel,
    pub sampler: &'a EulerSampler,
}

/// Run the Euler-Ancestral denoise loop with CFG over `steps` from `start_time` down to 0 — txt2img
/// passes `start_time = sampler.max_time()`, img2img passes `max_time · strength` (sc-2638).
/// `latents` is the seeded init `[1, h, w, 4]`; `conditioning`/`pooled`/`time_ids` carry the CFG
/// batch (B = 2 when `cfg > 1`). Returns the final latents; progress per step; `cancel` between steps.
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    d: &Denoiser,
    mut latents: Array,
    conditioning: &Array,
    pooled: &Array,
    time_ids: &Array,
    steps: usize,
    start_time: f32,
    cfg: f32,
    cancel: &CancelFlag,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<Array> {
    // A zero-step denoise (img2img at strength ≤ 1/steps) is a no-op: return the init latents
    // unchanged, matching the reference's `int(num_steps · strength)` loop count. Guards the
    // degenerate `timesteps(0, ·)` (0/0) and the σ=0 ancestral step that would otherwise NaN.
    if steps == 0 {
        return Ok(latents);
    }
    let cfg_on = cfg > 1.0;
    let total = steps as u32;
    for (i, (t, t_prev)) in d
        .sampler
        .timesteps(steps, start_time)
        .into_iter()
        .enumerate()
    {
        if cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let x_unet = if cfg_on {
            concatenate_axis(&[&latents, &latents], 0)?
        } else {
            latents.clone()
        };
        let eps = d.unet.forward(&x_unet, t, conditioning, pooled, time_ids)?;
        let eps = if cfg_on {
            let row = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0);
            let eps_text = row(0)?;
            let eps_neg = row(1)?;
            // `eps_neg + cfg·(eps_text − eps_neg)`. The reference's `cfg_weight` is a python float that
            // weak-casts to the eps dtype, so CFG runs in the compute dtype — cast the scalar to the
            // eps dtype here too. An f32 `cfg` would promote an fp16 eps to f32, and the sampler step
            // (which keys off `eps.dtype()`) would then run f32, silently leaving the latents f32.
            let cfg_s = scalar(cfg).as_dtype(eps_text.dtype())?;
            add(
                &eps_neg,
                &multiply(&subtract(&eps_text, &eps_neg)?, &cfg_s)?,
            )?
        } else {
            eps
        };
        latents = d.sampler.step(&eps, &latents, t, t_prev)?;
        // Force evaluation each step (the reference's per-step `mx.eval`). Beyond bounding the lazy
        // graph, this materializes the global-RNG state split between steps so the ancestral noise
        // stream is byte-identical to the reference — leaving it lazy across all steps perturbs the
        // draws and re-introduces the chaotic divergence (sc-2400 S5).
        latents.eval()?;
        on_progress(Progress::Step {
            current: i as u32 + 1,
            total,
        });
    }
    Ok(latents)
}

/// Seed the global RNG and sample the prior latents `[1, height/8, width/8, 4]` (NHWC, f32).
pub fn seeded_prior(sampler: &EulerSampler, seed: u64, width: u32, height: u32) -> Result<Array> {
    random::seed(seed)?;
    sampler.sample_prior(&[
        1,
        (height / SPATIAL_SCALE) as i32,
        (width / SPATIAL_SCALE) as i32,
        LATENT_CHANNELS,
    ])
}

/// Preprocess an init image for img2img: PIL-LANCZOS resize to the target dims (no-op when already
/// sized), normalize `[0,255] → [-1,1]`, lay out NHWC `[1, H, W, 3]` f32 — the input the VAE encoder
/// expects. Uses the core PIL-exact resampler (`resize_lanczos_u8`).
pub fn preprocess_init_image(
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let (iw, ih) = (image.width as usize, image.height as usize);
    let (tw, th) = (target_width as usize, target_height as usize);
    if image.pixels.len() != iw * ih * 3 {
        return Err(Error::Msg(format!(
            "sdxl init image pixel buffer {} != {iw}x{ih}x3",
            image.pixels.len()
        )));
    }
    let resized: Vec<f32> = if (ih, iw) == (th, tw) {
        image.pixels.iter().map(|&p| p as f32).collect()
    } else {
        resize_lanczos_u8(&image.pixels, ih, iw, th, tw)
    };
    let norm: Vec<f32> = resized.iter().map(|&v| 2.0 * (v / 255.0) - 1.0).collect();
    Ok(Array::from_slice(&norm, &[1, th as i32, tw as i32, 3]))
}

/// img2img init latents: preprocess the image → VAE-encode mean `[1, h, w, 4]` (NHWC). The fork's
/// `generate_latents_from_image` uses the encoder mean (not a sample) as `x_0`.
pub fn encode_init_latents(
    vae: &Autoencoder,
    image: &Image,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let nhwc = preprocess_init_image(image, target_width, target_height)?;
    vae.encode_mean(&nhwc)
}

/// Convert a VAE-decoded NHWC tensor `[1, H, W, 3]` (≈`[-1, 1]`) to an RGB8 [`Image`]:
/// `clip(x·0.5 + 0.5, 0, 1) · 255` (the vendored `StableDiffusion.decode` + txt2image recipe).
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let half = scalar(0.5);
    let x = add(&multiply(decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;
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

/// Decode final latents `[1, h, w, 4]` to an RGB8 image.
pub fn decode_image(vae: &Autoencoder, latents: &Array) -> Result<Image> {
    let decoded = vae.decode(latents)?;
    decoded_to_image(&decoded)
}
