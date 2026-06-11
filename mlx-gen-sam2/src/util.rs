//! Small host/NN helpers shared across the SAM2 modules — one copy each, so a fix (an eps or
//! border-handling change) lands in a single place (F-171).

use mlx_rs::ops::{add, mean_axes, multiply, rsqrt, square, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

/// Index of the maximum value, NaN-safe via [`f32::total_cmp`] — a NaN in the IoU-head output can't
/// panic the way `partial_cmp(..).unwrap()` would. Matches [`Iterator::max_by`] semantics (ties
/// resolve to the **last** maximum, so for finite inputs this is identical to the previous
/// `max_by(partial_cmp)`), and an empty slice returns `0`. Shared by the three SAM2 best-IoU
/// multimask selections (F-169).
pub(crate) fn argmax_f32(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Join a weight-key `prefix` and `leaf` with a `.` (no leading dot when `prefix` is empty). The
/// `tree_flatten`-style key builder every SAM2 module's `from_weights` uses (F-171).
pub(crate) fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// `x[:, start..end, ...]` — take the contiguous `start..end` index range along axis 1. The single
/// take-range helper behind the SAM2 token / mask / IoU slices (F-171).
pub(crate) fn take_range(x: &Array, start: i32, end: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
    Ok(x.take_axis(&idx, 1)?)
}

/// `LayerNorm2d`: normalize an **NCHW** tensor over the channel axis (per spatial position) with a
/// per-channel affine `weight`/`bias`, stabilized by `eps`. One implementation for both the
/// mask-decoder and memory-encoder uses (F-171).
pub(crate) fn layer_norm_2d(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let mean = mean_axes(x, &[1], true)?;
    let centered = subtract(x, &mean)?;
    let var = mean_axes(&square(&centered)?, &[1], true)?;
    let normed = multiply(&centered, &rsqrt(&add(&var, Array::from_f32(eps))?)?)?;
    let w = weight.reshape(&[1, -1, 1, 1])?;
    let b = bias.reshape(&[1, -1, 1, 1])?;
    Ok(add(&multiply(&normed, &w)?, &b)?)
}

/// Host f32 bilinear resize of a single `in_h × in_w` plane to `out_h × out_w` (half-pixel centers,
/// `align_corners = False`, edge-clamped). The shared image/mask resampler (F-171).
pub(crate) fn bilinear_resize_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; out_h * out_w];
    let sy = in_h as f32 / out_h as f32;
    let sx = in_w as f32 / out_w as f32;
    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).clamp(0.0, (in_h - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).clamp(0.0, (in_w - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;
            let v00 = src[y0 * in_w + x0];
            let v01 = src[y0 * in_w + x1];
            let v10 = src[y1 * in_w + x0];
            let v11 = src[y1 * in_w + x1];
            let top = v00 + (v01 - v00) * wx;
            let bot = v10 + (v11 - v10) * wx;
            out[oy * out_w + ox] = top + (bot - top) * wy;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_f32_max_ties_empty() {
        assert_eq!(argmax_f32(&[0.1, 0.9, 0.3]), 1);
        // Ties resolve to the last maximum — same as the replaced `max_by(partial_cmp)`.
        assert_eq!(argmax_f32(&[0.5, 0.5, 0.5]), 2);
        // Empty → 0, preserving the call sites' `unwrap_or(0)`.
        assert_eq!(argmax_f32(&[]), 0);
    }

    #[test]
    fn argmax_f32_is_nan_safe() {
        // The point of F-169: a NaN must NOT panic (vs `partial_cmp(..).unwrap()`). `total_cmp`
        // gives a total order, so this returns some valid in-bounds index instead.
        let with_nan = [0.2_f32, f32::NAN, 0.8];
        let idx = argmax_f32(&with_nan);
        assert!(idx < with_nan.len());
    }

    #[test]
    fn join_dots_nonempty_prefix() {
        assert_eq!(join("", "weight"), "weight");
        assert_eq!(join("blocks.0", "norm.weight"), "blocks.0.norm.weight");
    }

    #[test]
    fn take_range_slices_axis1() {
        // [1, 4, 2] → take rows 1..3 of axis 1 → [1, 2, 2] holding the original rows 1 and 2.
        let x = Array::from_slice(&[0, 0, 1, 1, 2, 2, 3, 3], &[1, 4, 2]);
        let got = take_range(&x, 1, 3).unwrap();
        assert_eq!(got.shape(), &[1, 2, 2]);
        assert_eq!(got.as_slice::<i32>(), &[1, 1, 2, 2]);
    }

    #[test]
    fn bilinear_resize_doubles_and_interpolates() {
        // 2×2 → 4×4: corners preserved, an interpolated midpoint stays within the source range.
        let src = vec![0.0f32, 1.0, 1.0, 2.0];
        let up = bilinear_resize_f32(&src, 2, 2, 4, 4);
        assert_eq!(up.len(), 16);
        assert_eq!(up[0], 0.0);
        assert!((0.0..=1.0).contains(&up[3]));
    }
}
