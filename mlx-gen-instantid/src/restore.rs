//! InstantID face-restoration compositing (sc-3380).
//!
//! The face-restore pass (`instantid_adapter.py::_restore_face`, sc-2063) is an ADetailer-style
//! re-render of the cropped face through the **same InstantID pipe** (IdentityNet only) followed by a
//! feathered paste-back. This module holds the net-new image-compositing pieces — the feathered
//! elliptical alpha mask and the alpha paste-back; the crop/re-render orchestration lives in
//! [`crate::model::InstantId::restore_face`].
//!
//! The gate is **directional** (epic 3109: identity + seamlessness, not bit-exact vs PIL): the mask is
//! a filled ellipse in the inner `[0.1, 0.9]` box of the crop, Gaussian-blurred so the paste-back has
//! no hard edges — mirroring the reference's `ImageDraw.ellipse(...) + GaussianBlur`. A true separable
//! Gaussian (σ = blur radius) stands in for PIL's box-blur approximation; the result is a smooth
//! feather, which is all the composite needs.

use mlx_gen::media::Image;

/// Build the feathered elliptical alpha mask for a `crop_w × crop_h` face crop — a filled ellipse in
/// the inner `[0.1·w, 0.1·h, 0.9·w, 0.9·h]` box, Gaussian-blurred by `max(4, crop_w / 12)` (the
/// reference's feather radius). Returns alpha in `[0, 1]`, length `crop_w · crop_h` (row-major).
pub fn feather_mask(crop_w: usize, crop_h: usize) -> Vec<f32> {
    let (w, h) = (crop_w as f64, crop_h as f64);
    // PIL `ImageDraw.ellipse([x0,y0,x1,y1])` fills the ellipse inscribed in the box.
    let (x0, y0) = (0.1 * w, 0.1 * h);
    let (x1, y1) = (0.9 * w, 0.9 * h);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    let (rx, ry) = (((x1 - x0) / 2.0).max(1.0), ((y1 - y0) / 2.0).max(1.0));

    let mut mask = vec![0f32; crop_w * crop_h];
    for y in 0..crop_h {
        for x in 0..crop_w {
            // Pixel-center test against the ellipse.
            let dx = (x as f64 + 0.5 - cx) / rx;
            let dy = (y as f64 + 0.5 - cy) / ry;
            if dx * dx + dy * dy <= 1.0 {
                mask[y * crop_w + x] = 1.0;
            }
        }
    }
    let radius = (crop_w / 12).max(4);
    feather_blur(&mask, crop_w, crop_h, radius)
}

/// Feather a single-channel mask (clamp-to-edge) with a **3-pass separable box-blur cascade**. Three
/// iterated box blurs approximate a Gaussian (central-limit) of effective σ ≈ `radius`, but each pass
/// costs a constant ~4 ops per pixel via a sliding running sum — vs the `O(radius)` taps of the old
/// direct Gaussian convolution, whose half-width `ceil(3σ)` made the feather scale `O(crop³)` (≈1s on
/// a 1024² crop, F-091). The composite gate is directional (a smooth feather, not bit-parity vs PIL),
/// which a box cascade satisfies.
fn feather_blur(img: &[f32], w: usize, h: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || w == 0 || h == 0 {
        return img.to_vec();
    }
    let mut a = img.to_vec();
    let mut b = vec![0f32; w * h];
    for _ in 0..3 {
        box_blur_h(&a, &mut b, w, h, radius);
        box_blur_v(&b, &mut a, w, h, radius);
    }
    a
}

/// Horizontal box blur (window `2·r+1`, clamp-to-edge) via a sliding running sum: `src → dst`.
// The loop index drives the sliding-window bounds (`add`/`rem`), so it can't be an iterator.
#[allow(clippy::needless_range_loop)]
fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let norm = 1.0 / (2 * r + 1) as f64;
    let last = w as isize - 1;
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        let out = &mut dst[y * w..y * w + w];
        // Initial window [-r, r] for x = 0 (clamped to the edge).
        let mut sum: f64 = (-(r as isize)..=r as isize)
            .map(|j| row[j.clamp(0, last) as usize] as f64)
            .sum();
        out[0] = (sum * norm) as f32;
        for x in 1..w {
            // Slide: drop the pixel leaving on the left, add the one entering on the right.
            let add = (x as isize + r as isize).min(last) as usize;
            let rem = (x as isize - 1 - r as isize).max(0) as usize;
            sum += row[add] as f64 - row[rem] as f64;
            out[x] = (sum * norm) as f32;
        }
    }
}

/// Vertical box blur (window `2·r+1`, clamp-to-edge) via a sliding running sum: `src → dst`.
fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    let norm = 1.0 / (2 * r + 1) as f64;
    let last = h as isize - 1;
    for x in 0..w {
        let mut sum: f64 = (-(r as isize)..=r as isize)
            .map(|j| src[j.clamp(0, last) as usize * w + x] as f64)
            .sum();
        dst[x] = (sum * norm) as f32;
        for y in 1..h {
            let add = (y as isize + r as isize).min(last) as usize;
            let rem = (y as isize - 1 - r as isize).max(0) as usize;
            sum += src[add * w + x] as f64 - src[rem * w + x] as f64;
            dst[y * w + x] = (sum * norm) as f32;
        }
    }
}

/// Alpha-composite a `crop_w × crop_h` RGB8 patch `small` onto `base` at top-left `(ax, ay)`, using
/// per-pixel `alpha` in `[0, 1]` — the feathered paste-back (`Image.paste(small, (a,b), mask)`).
/// `out = base·(1-α) + small·α`, rounded to u8. The crop box is assumed in-bounds (clamped by the
/// caller); any out-of-bounds pixel is skipped.
pub fn paste_alpha(
    base: &mut Image,
    small: &[u8],
    crop_w: usize,
    crop_h: usize,
    ax: usize,
    ay: usize,
    alpha: &[f32],
) {
    let bw = base.width as usize;
    let bh = base.height as usize;
    for y in 0..crop_h {
        let by = ay + y;
        if by >= bh {
            break;
        }
        for x in 0..crop_w {
            let bx = ax + x;
            if bx >= bw {
                break;
            }
            let a = alpha[y * crop_w + x].clamp(0.0, 1.0);
            let si = (y * crop_w + x) * 3;
            let di = (by * bw + bx) * 3;
            for c in 0..3 {
                let b = base.pixels[di + c] as f32;
                let s = small[si + c] as f32;
                base.pixels[di + c] = (b * (1.0 - a) + s * a).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sc-3380: the feather mask is opaque at the crop center and feathers to ~transparent at the
    /// corners (a seamless paste-back has no hard edge).
    #[test]
    fn feather_mask_is_centered_and_soft() {
        let (w, h) = (200usize, 240usize);
        let m = feather_mask(w, h);
        assert_eq!(m.len(), w * h);
        let center = m[(h / 2) * w + w / 2];
        let corner = m[0];
        let edge_mid = m[(h / 2) * w]; // left edge, vertical center (outside the inner ellipse)
        assert!(center > 0.95, "center alpha {center} should be ~opaque");
        assert!(
            corner < 0.05,
            "corner alpha {corner} should be ~transparent"
        );
        assert!(
            edge_mid < center,
            "edge alpha {edge_mid} should feather below the center {center}"
        );
        // No hard edge: every value is a valid, finite alpha in [0, 1].
        assert!(m.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    /// F-091: the sliding-running-sum box blur must equal a direct clamp-to-edge windowed average,
    /// and a constant image must blur to itself (catches a normalization / window-bounds bug).
    #[test]
    fn box_blur_matches_direct_average() {
        let (w, h, r) = (5usize, 4usize, 1usize);
        let img: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();
        let mut got = vec![0f32; w * h];
        box_blur_h(&img, &mut got, w, h, r);

        let last = w as isize - 1;
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0f64;
                for j in -(r as isize)..=r as isize {
                    let sx = (x as isize + j).clamp(0, last) as usize;
                    acc += img[y * w + sx] as f64;
                }
                let want = (acc / (2 * r + 1) as f64) as f32;
                assert!(
                    (got[y * w + x] - want).abs() < 1e-5,
                    "box_blur_h[{x},{y}] {} vs {want}",
                    got[y * w + x]
                );
            }
        }

        // A constant feathered image stays constant (3-pass cascade, any radius).
        let flat = vec![0.7f32; w * h];
        let blurred = feather_blur(&flat, w, h, 2);
        assert!(blurred.iter().all(|&v| (v - 0.7).abs() < 1e-5));
    }

    /// sc-3380: alpha paste-back is a straight per-pixel blend — α=1 replaces, α=0 keeps.
    #[test]
    fn paste_alpha_blends() {
        let mut base = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 3],
        };
        let small = vec![200u8; 2 * 2 * 3]; // a 2×2 patch of 200
        let alpha = vec![1.0, 0.0, 0.5, 1.0]; // one of each blend
        paste_alpha(&mut base, &small, 2, 2, 1, 1, &alpha);
        // (1,1) α=1 → 200; (2,1) α=0 → 0; (1,2) α=0.5 → 100; (2,2) α=1 → 200.
        let px = |x: usize, y: usize| base.pixels[(y * 4 + x) * 3];
        assert_eq!(px(1, 1), 200);
        assert_eq!(px(2, 1), 0);
        assert_eq!(px(1, 2), 100);
        assert_eq!(px(2, 2), 200);
        // Untouched pixel stays 0.
        assert_eq!(px(0, 0), 0);
    }
}
