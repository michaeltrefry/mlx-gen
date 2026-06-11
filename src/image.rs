//! Image helpers split across the backend boundary (epic 3720, Appendix A):
//!
//! - The PIL-compatible resampling ([`resize_bicubic_u8`] & friends) and the host-side mask /
//!   geometry ops ([`contain_box`], [`outpaint_border_mask`], [`union_masks`]) are pure and now live
//!   in [`gen_core::imageops`]; they are re-exported here so the historical `mlx_gen::image::…`
//!   paths keep resolving.
//! - [`decoded_to_image`] — the VAE-decoded-tensor → [`Image`] denormalize/quantize step — operates
//!   on an `mlx_rs::Array` and stays here.

use mlx_rs::ops::{add, maximum, minimum, multiply, round};
use mlx_rs::Array;

use crate::array::scalar;
use crate::media::Image;
use crate::Result;

pub use gen_core::imageops::*;

/// Denormalize a VAE-decoded tensor to an RGB8 [`Image`]: `clip(x·0.5 + 0.5, 0, 1)` → drop the
/// singleton temporal axis (5-D → 4-D) → NCHW→NHWC → `(x·255).round()` → `u8`, taking batch 0.
/// Identical across the Z-Image and Qwen-Image pipelines (the decoded tensor must already be f32).
pub fn decoded_to_image(decoded: &Array) -> Result<Image> {
    let half = scalar(0.5);
    // denormalize: clip(x*0.5 + 0.5, 0, 1)
    let x = add(&multiply(decoded, &half)?, &half)?;
    let x = minimum(&maximum(&x, scalar(0.0))?, scalar(1.0))?;
    // drop the singleton temporal axis if present (5-D → 4-D)
    let x = if x.shape().len() == 5 {
        x.squeeze_axes(&[2])?
    } else {
        x
    };
    // NCHW → NHWC
    let x = x.transpose_axes(&[0, 2, 3, 1])?;
    // (x*255).round() to integer pixel values.
    let x = round(&multiply(&x, scalar(255.0))?, 0)?;

    let sh = x.shape();
    let (h, w, c) = (sh[1] as u32, sh[2] as u32, sh[3] as u32);
    let n = (h * w * c) as usize;
    // `transpose_axes` yields a strided view; a raw `as_slice` would read physical (pre-transpose)
    // order. `reshape` re-materializes in C-order, so the slice is logical NHWC. Take batch 0.
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

    /// `resize_u8` (now in gen-core) must be **bit-identical** to PIL `Image.BICUBIC` (the
    /// fixed-point integer path), not merely close — this is what gives the conditioning images
    /// pixel-parity with the fork (sc-2465: an f64-coefficient resampler diverged ±1–2 ULP at
    /// gradient cliffs → 24% e2e px>8). The golden tensor is read via `crate::weights::Weights`
    /// (MLX), so this test stays in mlx-gen. Golden via `tools/dump_pil_resize_golden.py`.
    #[test]
    #[ignore = "needs tools/golden/pil_resize_golden.safetensors (run tools/dump_pil_resize_golden.py)"]
    fn resize_bicubic_matches_pil_512_to_384() {
        let g = crate::weights::Weights::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/golden/pil_resize_golden.safetensors"
        ))
        .unwrap();
        // Sawtooth: `(x+y)%256`, ×2, ×3 — sharp 255→0 cliffs where bicubic implementations diverge.
        let mut saw = Vec::with_capacity(512 * 512 * 3);
        let mut smo = Vec::with_capacity(512 * 512 * 3);
        for y in 0..512u32 {
            for x in 0..512u32 {
                let b = (y + x) % 256;
                saw.push(b as u8);
                saw.push(((b * 2) % 256) as u8);
                saw.push(((b * 3) % 256) as u8);
                let v = ((x + y) / 4).min(255) as u8;
                smo.push(v);
                smo.push(v);
                smo.push(v);
            }
        }
        let cmp = |got: &[f32], pil: &[i32]| -> (usize, i32) {
            assert_eq!(got.len(), pil.len(), "len");
            got.iter()
                .zip(pil)
                .fold((0usize, 0i32), |(n, m), (&gv, &pv)| {
                    let d = (gv as i32 - pv).abs();
                    (n + (d != 0) as usize, m.max(d))
                })
        };
        let (saw_diff, saw_max) = cmp(
            &resize_bicubic_u8(&saw, 512, 512, 384, 384),
            g.require("pil384").unwrap().as_slice::<i32>(),
        );
        let (smo_diff, smo_max) = cmp(
            &resize_bicubic_u8(&smo, 512, 512, 384, 384),
            g.require("pil384_smooth").unwrap().as_slice::<i32>(),
        );
        println!("vs PIL 512->384: sawtooth {saw_diff} diff (max {saw_max}), smooth {smo_diff} diff (max {smo_max})");
        assert_eq!(
            saw_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the cliff gradient"
        );
        assert_eq!(
            smo_diff, 0,
            "resize_u8 must bit-match PIL BICUBIC on the smooth ramp"
        );
    }
}
