//! SDXL masked **inpaint** (sc-3057) — the diffusers `StableDiffusionXLInpaintPipeline` *legacy
//! mask-blend* on the base 4-channel UNet (the worker's `_as_inpaint_pipe`, which wraps the resident
//! edit checkpoint, not a 9-channel inpaint UNet). It rides the existing pixel-parity img2img path
//! and adds a per-step latent blend: regenerate the **white** mask region, keep the **black** region
//! pinned to the (noised) original.
//!
//! Convention bridge: mlx-gen stores latents in the vendored mlx_sd **renormalized** space
//! (`x/√(σ²+1)`), whereas diffusers' `EulerDiscreteScheduler.add_noise` is raw (`x + noise·σ`). The
//! blend's "kept" term must live in the same space as the running latents, so it uses
//! [`EulerSampler::add_noise_with`] (`(x₀ + noise·σ) · rsqrt(σ²+1)`) with the **fixed** prior noise.
//! Because the blend draws no RNG, the ancestral noise stream is identical to plain img2img — so a
//! mask of all-white (repaint everything) reduces inpaint to img2img *bit-for-bit*.

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::image::resize_bicubic_u8;
use mlx_gen::media::Image;
use mlx_gen::Result;

use crate::sampler::EulerSampler;

/// Build the latent-resolution **binary** inpaint mask `[1, h/8, w/8, 1]` (f32) from a mask image.
/// White (luma ≥ 0.5) → 1.0 (repaint), black → 0.0 (keep). Mirrors diffusers'
/// `VaeImageProcessor(do_binarize=True, do_convert_grayscale=True)` + `F.interpolate(..., nearest)`:
/// grayscale-convert (PIL "L" luma), binarize at image resolution, then nearest 8× downsample
/// (the top-left pixel of each 8×8 block — torch nearest's `floor(dst·scale)`).
pub fn preprocess_mask(mask: &Image, width: u32, height: u32) -> Result<Array> {
    let (w, h) = (width as usize, height as usize);
    // Align to W×H (the worker's `load_mask_image` already does; resize defensively otherwise).
    let luma: Vec<u8> = if (mask.width as usize, mask.height as usize) == (w, h) {
        rgb_to_luma(&mask.pixels)
    } else {
        let resized = resize_bicubic_u8(
            &mask.pixels,
            mask.height as usize,
            mask.width as usize,
            h,
            w,
        );
        // resize returns f32 HWC RGB in [0,255]; collect to u8 then luma.
        let u8s: Vec<u8> = resized
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        rgb_to_luma(&u8s)
    };
    // Binarize at image res, then nearest 8× downsample.
    let (lh, lw) = (h / 8, w / 8);
    let mut latent = Vec::with_capacity(lh * lw);
    for ly in 0..lh {
        for lx in 0..lw {
            let v = luma[(ly * 8) * w + (lx * 8)]; // top-left of the 8×8 block
            latent.push(if v as f32 / 255.0 >= 0.5 { 1.0f32 } else { 0.0 });
        }
    }
    Ok(Array::from_slice(&latent, &[1, lh as i32, lw as i32, 1]))
}

/// PIL "L" grayscale luma: `round(R·299/1000 + G·587/1000 + B·114/1000)` per pixel.
fn rgb_to_luma(rgb: &[u8]) -> Vec<u8> {
    rgb.chunks_exact(3)
        .map(|p| {
            let l = (p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114 + 500) / 1000;
            l.min(255) as u8
        })
        .collect()
}

/// Per-step blend state for the legacy inpaint: keep `(1-mask)` pinned to the init noised to the
/// step's `σ`, regenerate `mask`. Holds the **fixed** prior noise (reused every step) and the clean
/// init latents `x₀`.
pub struct InpaintBlend<'a> {
    sampler: &'a EulerSampler,
    /// `[1, h, w, 1]` latent-res mask, 1 = repaint, 0 = keep.
    mask: Array,
    /// `[1, h, w, 4]` clean VAE-encoded init latents (`image_latents`).
    x0: Array,
    /// `[1, h, w, 4]` fixed unit-normal prior noise (the same draw that seeded the init latent).
    noise: Array,
    /// Per-step "next" time `t_prev` (`schedule[i].1`) — the σ the kept region is noised to after
    /// step `i`. The last entry is 0 ⇒ the final blend is against the clean init.
    t_prev: Vec<f32>,
}

impl<'a> InpaintBlend<'a> {
    pub fn new(
        sampler: &'a EulerSampler,
        mask: Array,
        x0: Array,
        noise: Array,
        t_prev: Vec<f32>,
    ) -> Self {
        Self {
            sampler,
            mask,
            x0,
            noise,
            t_prev,
        }
    }

    /// Blend after denoise step `i`: `latents = (1 - mask)·init_noised + mask·latents`, where
    /// `init_noised = add_noise_with(x₀, noise, σ(t_prev[i]))`. Draws no RNG.
    pub fn blend(&self, latents: &Array, i: usize) -> Result<Array> {
        let t = self.t_prev[i];
        let init_noised = self.sampler.add_noise_with(&self.x0, &self.noise, t)?;
        let one = scalar(1.0);
        let keep = multiply(&subtract(&one, &self.mask)?, &init_noised)?;
        let repaint = multiply(&self.mask, latents)?;
        Ok(add(&keep, &repaint)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, f: impl Fn(u32, u32) -> u8) -> Image {
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = f(x, y);
                pixels.extend_from_slice(&[v, v, v]);
            }
        }
        Image {
            width: w,
            height: h,
            pixels,
        }
    }

    #[test]
    fn preprocess_mask_binarizes_and_downsamples_nearest() {
        // 16×16: left half (cols 0-7) white, right half black. Latent 2×2 samples the top-left of
        // each 8×8 block: col-0 block → pixel(.,0)=white→1, col-1 block → pixel(.,8)=black→0.
        let m = gray(16, 16, |x, _| if x < 8 { 255 } else { 0 });
        let a = preprocess_mask(&m, 16, 16).unwrap();
        assert_eq!(a.shape(), &[1, 2, 2, 1]);
        let v = a.as_slice::<f32>();
        assert_eq!(v, &[1.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn preprocess_mask_threshold_is_half() {
        // 127 (< 0.5·255 = 127.5) → 0; 128 → 1.
        let lo = preprocess_mask(&gray(8, 8, |_, _| 127), 8, 8).unwrap();
        let hi = preprocess_mask(&gray(8, 8, |_, _| 128), 8, 8).unwrap();
        assert_eq!(lo.as_slice::<f32>(), &[0.0]);
        assert_eq!(hi.as_slice::<f32>(), &[1.0]);
    }
}
