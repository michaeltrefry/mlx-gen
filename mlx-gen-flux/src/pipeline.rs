//! FLUX.1 sampling-pipeline primitives whose math is stable before the model blocks land.
//! These mirror `FluxLatentCreator` and the fork's default `LinearScheduler`.

use mlx_gen::{Error, Result};
use mlx_rs::{random, Array};

pub fn image_seq_len(width: u32, height: u32) -> usize {
    ((height / 16) * (width / 16)) as usize
}

/// Seeded FLUX txt2img latent noise: `[1, (height/16) * (width/16), 64]`.
pub fn create_noise(seed: u64, width: u32, height: u32) -> Result<Array> {
    validate_multiple_of_16(width, height)?;
    let key = random::key(seed)?;
    let shape = [1, image_seq_len(width, height) as i32, 64];
    Ok(random::normal::<f32>(&shape[..], None, None, Some(&key))?)
}

/// Pack VAE latents `[1, 16, height/8, width/8]` into FLUX DiT tokens
/// `[1, (height/16) * (width/16), 64]`.
pub fn pack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    validate_multiple_of_16(width, height)?;
    let h = (height / 16) as i32;
    let w = (width / 16) as i32;
    let latents = latents.reshape(&[1, 16, h, 2, w, 2])?;
    let latents = latents.transpose_axes(&[0, 2, 4, 1, 3, 5])?;
    Ok(latents.reshape(&[1, h * w, 64])?)
}

/// Unpack FLUX DiT tokens `[1, (height/16) * (width/16), 64]` back to VAE latents
/// `[1, 16, height/8, width/8]`.
pub fn unpack_latents(latents: &Array, width: u32, height: u32) -> Result<Array> {
    validate_multiple_of_16(width, height)?;
    let h = (height / 16) as i32;
    let w = (width / 16) as i32;
    let latents = latents.reshape(&[1, h, w, 16, 2, 2])?;
    let latents = latents.transpose_axes(&[0, 3, 1, 4, 2, 5])?;
    Ok(latents.reshape(&[1, 16, h * 2, w * 2])?)
}

/// Fork `LinearScheduler` sigmas. `requires_sigma_shift` is true for FLUX.1-dev and false for
/// FLUX.1-schnell. The shift constants are the fork's defaults: base `(256, 0.5)`, max
/// `(4096, 1.15)`, no terminal stretch for FLUX.1.
pub fn build_linear_sigmas(
    num_steps: usize,
    width: u32,
    height: u32,
    requires_sigma_shift: bool,
) -> Vec<f32> {
    let n = num_steps.max(1);
    let mut sigmas: Vec<f32> = (0..n)
        .map(|i| {
            if n == 1 {
                1.0
            } else {
                1.0 + ((1.0 / n as f32) - 1.0) * (i as f32) / ((n - 1) as f32)
            }
        })
        .collect();

    if requires_sigma_shift {
        let seq = image_seq_len(width, height) as f32;
        let base_seq_len = 256.0_f32;
        let max_seq_len = 4096.0_f32;
        let base_shift = 0.5_f32;
        let max_shift = 1.15_f32;
        let m = (max_shift - base_shift) / (max_seq_len - base_seq_len);
        let b = base_shift - m * base_seq_len;
        let mu = m * seq + b;
        let e = mu.exp();
        sigmas = sigmas
            .into_iter()
            .map(|t| e / (e + (1.0 / t - 1.0)))
            .collect();
    }

    sigmas.push(0.0);
    sigmas
}

fn validate_multiple_of_16(width: u32, height: u32) -> Result<()> {
    if width % 16 != 0 || height % 16 != 0 {
        return Err(Error::Msg(format!(
            "flux1: width and height must be multiples of 16, got {width}x{height}"
        )));
    }
    Ok(())
}
