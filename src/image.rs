//! PIL-compatible image resampling — the shared, model-agnostic resize primitive used by the
//! provider crates' img2img / edit / control preprocessing (e.g. the fork's `scale_to_dimensions`
//! and the Qwen2-VL image processor).
//!
//! [`resize_u8`] bit-matches PIL's `ImagingResample` 8-bit path: float filter coefficients
//! quantized to `PRECISION_BITS` fixed-point, an integer multiply-accumulate seeded with the
//! rounding bias, then `clip8` (`>>PRECISION_BITS` + clamp). Reproducing PIL's *fixed-point*
//! arithmetic (not just "a bicubic") is what gives the edit/img2img conditioning images
//! pixel-parity with the frozen Python fork — an f64-coefficient resampler diverges ±1–2 ULP at
//! gradient cliffs (sc-2465: 24% e2e px>8). Lives in core so every model reuses one copy.
//!
//! Also hosts [`decoded_to_image`] — the VAE-decoded-tensor → [`Image`] denormalize/quantize step,
//! identical across the provider crates' pipelines (F-006).

use mlx_rs::ops::{add, maximum, minimum, multiply, round};
use mlx_rs::Array;

use crate::array::scalar;
use crate::media::Image;
use crate::Result;

/// PIL `bicubic_filter` (Keys cubic, a = -0.5), support 2.0.
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Normalized sinc, `sin(πx)/(πx)`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// PIL `lanczos_filter` (a = 3): `sinc(x)·sinc(x/3)`, support 3.0.
fn lanczos3(x: f64) -> f64 {
    if x.abs() < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Per-output-pixel resampling coefficients for a 1-D axis resize, matching PIL's
/// `precompute_coeffs`: antialias by scaling the filter support when downscaling, clamp the
/// window to the input bounds, and renormalize the (possibly truncated) weights to sum to 1.
/// `support_radius` is the filter's base support (2.0 bicubic, 3.0 lanczos).
fn precompute_coeffs(
    in_size: usize,
    out_size: usize,
    support_radius: f64,
    filter: &dyn Fn(f64) -> f64,
) -> Vec<(usize, Vec<f64>)> {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = support_radius * filterscale;
    let mut out = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let xmin = ((center - support + 0.5).floor() as i64).max(0) as usize;
        let xmax = ((center + support + 0.5).floor() as i64).min(in_size as i64) as usize;
        let mut weights = Vec::with_capacity(xmax - xmin);
        let mut total = 0.0;
        for x in xmin..xmax {
            let w = filter((x as f64 - center + 0.5) / filterscale);
            weights.push(w);
            total += w;
        }
        if total != 0.0 {
            for w in &mut weights {
                *w /= total;
            }
        }
        out.push((xmin, weights));
    }
    out
}

/// PIL's `PRECISION_BITS` for the 8-bit resample path (`32 - 8 - 2`): filter coefficients are
/// quantized to this many fractional bits and the convolution is accumulated in integers.
const PRECISION_BITS: u32 = 32 - 8 - 2;

/// PIL `clip8` for the resample accumulator (which already carries the `1<<(PRECISION_BITS-1)`
/// rounding bias): shift down by `PRECISION_BITS` and clamp to `[0,255]`.
#[inline]
fn clip8(acc: i64) -> f32 {
    if acc <= 0 {
        return 0.0;
    }
    let v = acc >> PRECISION_BITS;
    if v >= 255 {
        255.0
    } else {
        v as f32
    }
}

/// Quantize PIL float coefficients to fixed-point integers — `normalize_coeffs_8bpc`: round half
/// away from zero at `1<<PRECISION_BITS` (matches C's `(int)(±0.5 + w·2^PRECISION_BITS)`).
fn quantize_coeffs(coeffs: &[(usize, Vec<f64>)]) -> Vec<(usize, Vec<i64>)> {
    let scale = (1i64 << PRECISION_BITS) as f64;
    coeffs
        .iter()
        .map(|(xmin, w)| {
            let ik = w
                .iter()
                .map(|&c| {
                    if c < 0.0 {
                        (c * scale - 0.5) as i64
                    } else {
                        (c * scale + 0.5) as i64
                    }
                })
                .collect();
            (*xmin, ik)
        })
        .collect()
}

/// Two-pass (horizontal then vertical) separable resize of a uint8 HWC image, bit-matching PIL's
/// `ImagingResample` 8-bit path: float coefficients quantized to `PRECISION_BITS` fixed-point, an
/// integer multiply-accumulate seeded with the rounding bias, then `clip8` (`>>PRECISION_BITS` +
/// clamp) between/after passes. Returns f32 HWC with integer-valued samples in `[0, 255]`.
/// Assumes 3 channels (RGB).
fn resize_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    support_radius: f64,
    filter: &dyn Fn(f64) -> f64,
) -> Vec<f32> {
    let c = 3usize;
    let bias = 1i64 << (PRECISION_BITS - 1);

    // Horizontal pass: (in_h, in_w) -> (in_h, out_w).
    let hcoeffs = quantize_coeffs(&precompute_coeffs(in_w, out_w, support_radius, filter));
    let mut horiz = vec![0f32; in_h * out_w * c];
    for y in 0..in_h {
        for (xx, (xmin, w)) in hcoeffs.iter().enumerate() {
            for ch in 0..c {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += src[(y * in_w + xmin + k) * c + ch] as i64 * wk;
                }
                horiz[(y * out_w + xx) * c + ch] = clip8(acc);
            }
        }
    }

    // Vertical pass: (in_h, out_w) -> (out_h, out_w). Reads the integer-valued horiz samples.
    let vcoeffs = quantize_coeffs(&precompute_coeffs(in_h, out_h, support_radius, filter));
    let mut out = vec![0f32; out_h * out_w * c];
    for (yy, (ymin, w)) in vcoeffs.iter().enumerate() {
        for x in 0..out_w {
            for ch in 0..c {
                let mut acc = bias;
                for (k, &wk) in w.iter().enumerate() {
                    acc += horiz[((ymin + k) * out_w + x) * c + ch] as i64 * wk;
                }
                out[(yy * out_w + x) * c + ch] = clip8(acc);
            }
        }
    }
    out
}

/// PIL `Image.BICUBIC` resize of a uint8 RGB HWC image. Returns f32 HWC, integer-valued `[0,255]`.
pub fn resize_bicubic_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    resize_u8(src, in_h, in_w, out_h, out_w, 2.0, &cubic)
}

/// PIL `Image.LANCZOS` resize of a uint8 RGB HWC image (the fork's `scale_to_dimensions`). Returns
/// f32 HWC, integer-valued `[0,255]`.
pub fn resize_lanczos_u8(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    resize_u8(src, in_h, in_w, out_h, out_w, 3.0, &lanczos3)
}

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

/// Round-half-to-even (Python `round`), for pixel-geometry parity with the worker's `_contain_box`
/// (Rust's `f64::round` is half-away-from-zero, which differs at exact `.5`). Positive inputs only.
fn round_half_even(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// Where a `src_w`×`src_h` image lands when **contained** (long edge fits) and centered in a
/// `width`×`height` box: `(new_w, new_h, left, top)`. Mirrors the worker's `_contain_box` (Python
/// `round` = half-to-even) so the kept rect and a padded source line up exactly.
pub fn contain_box(src_w: u32, src_h: u32, width: u32, height: u32) -> (u32, u32, i32, i32) {
    let ratio = (width as f64 / src_w as f64).min(height as f64 / src_h as f64);
    let new_w = round_half_even(src_w as f64 * ratio).max(1) as u32;
    let new_h = round_half_even(src_h as f64 * ratio).max(1) as u32;
    let left = (width as i32 - new_w as i32) / 2;
    let top = (height as i32 - new_h as i32) / 2;
    (new_w, new_h, left, top)
}

/// Outpaint inpaint mask (the worker's `outpaint_border_mask`): an RGB8 grayscale mask —
/// **white (255) = the padded border to GENERATE, black (0) = the centered source rect to KEEP**
/// (inpaint convention: white = repaint). Geometry matches a "pad"/contain fit so the mask aligns
/// with the padded source. Pure host-side op; the engine consumes it as a `Conditioning::Mask`.
///
/// The worker's optional gaussian **feather** is intentionally omitted: the inpaint pipeline
/// binarizes the mask (`do_binarize`), and a symmetric blur's 0.5 crossing stays on the original
/// edge, so after the 8× latent downsample the feather is a no-op (it only rounds corners
/// sub-latent-pixel). Callers that want the seam softened should feather post-decode, not here.
pub fn outpaint_border_mask(src_w: u32, src_h: u32, width: u32, height: u32) -> Image {
    let (w, h) = (width.max(1), height.max(1));
    let (new_w, new_h, left, top) = contain_box(src_w, src_h, w, h);
    let mut pixels = vec![255u8; (w * h * 3) as usize]; // white = generate
    for y in 0..new_h as i32 {
        let cy = top + y;
        if cy < 0 || cy >= h as i32 {
            continue;
        }
        for x in 0..new_w as i32 {
            let cx = left + x;
            if cx < 0 || cx >= w as i32 {
                continue;
            }
            let idx = ((cy as u32 * w + cx as u32) * 3) as usize;
            pixels[idx] = 0; // black = keep
            pixels[idx + 1] = 0;
            pixels[idx + 2] = 0;
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Per-pixel max ("white wins" — PIL `ImageChops.lighter`) of two equal-size RGB8 masks. Unions a
/// user edit region with a generated outpaint border.
pub fn union_masks(a: &Image, b: &Image) -> Result<Image> {
    if (a.width, a.height) != (b.width, b.height) || a.pixels.len() != b.pixels.len() {
        return Err(crate::Error::Msg(format!(
            "union_masks: size mismatch {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        )));
    }
    let pixels = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .map(|(&x, &y)| x.max(y))
        .collect();
    Ok(Image {
        width: a.width,
        height: a.height,
        pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outpaint_border_mask_keeps_centered_source() {
        // A 50×100 source contained in a 200×200 canvas: long edge (100) fits → ratio 2.0 →
        // 100×200 kept rect, centered at left=50, top=0. White border L/R, black center column.
        let m = outpaint_border_mask(50, 100, 200, 200);
        assert_eq!((m.width, m.height), (200, 200));
        let px = |x: u32, y: u32| m.pixels[((y * 200 + x) * 3) as usize];
        assert_eq!(px(0, 100), 255, "left border = generate");
        assert_eq!(px(199, 100), 255, "right border = generate");
        assert_eq!(px(100, 100), 0, "center = keep");
        assert_eq!(px(50, 100), 0, "kept rect starts at left=50");
        assert_eq!(px(49, 100), 255, "just outside kept rect = generate");
    }

    #[test]
    fn union_masks_white_wins() {
        let a = Image {
            width: 2,
            height: 1,
            pixels: vec![255, 255, 255, 0, 0, 0],
        };
        let b = Image {
            width: 2,
            height: 1,
            pixels: vec![0, 0, 0, 0, 0, 0],
        };
        let u = union_masks(&a, &b).unwrap();
        assert_eq!(u.pixels, vec![255, 255, 255, 0, 0, 0]);
    }

    /// `resize_u8` must be **bit-identical** to PIL `Image.BICUBIC` (the fixed-point integer path),
    /// not merely close — this is what gives the conditioning images pixel-parity with the fork
    /// (sc-2465: an f64-coefficient resampler diverged ±1–2 ULP at gradient cliffs → 24% e2e px>8).
    /// Golden via `tools/dump_pil_resize_golden.py` (gitignored, like the other goldens).
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
