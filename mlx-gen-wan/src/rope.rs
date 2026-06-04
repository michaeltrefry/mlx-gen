//! 3-axis factorized 3-D RoPE — port of `models/wan/rope.py` (`rope_params`,
//! `rope_precompute_cos_sin`, `rope_apply`).
//!
//! The head dimension `d` (= 128 for both the 5B and 14B) is split across three axes
//! (temporal / height / width). The reference builds three separate `rope_params` tables with
//! per-axis dimension normalization and concatenates them along the frequency axis:
//!
//! ```text
//! temporal: rope_params(1024, d − 4·(d//6))   → half = (d − 4·(d//6))/2   (= 22 for d=128)
//! height:   rope_params(1024, 2·(d//6))        → half = (d//6)            (= 21)
//! width:    rope_params(1024, 2·(d//6))        → half = (d//6)            (= 21)
//! ```
//!
//! giving a `[1024, d/2, 2]` (cos, sin) table whose `d/2` columns are `[22 | 21 | 21]`. The whole
//! grid is computed in **f64** and cast to **f32** (the reference's `np.float64 → astype(f32)`),
//! and RoPE is applied in **f32** in the attention path. Pairs are **interleaved**:
//! `x.reshape(..., d/2, 2)` rotates `(x[2k], x[2k+1])` by `(cos[k], sin[k])`.

use mlx_rs::error::Exception;
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use mlx_gen::Result;

/// The complex RoPE rotation `(a+bi)·(cos+sin·i)` → `(out_real, out_imag)`. When the sc-2957 compile
/// toggle is on, MLX fuses the 4 multiplies + add/sub into one kernel (vs 6 eager ops on ~200 MB f32
/// tensors, applied to both q and k every block). Bit-exact to the eager form.
fn rope_rotate(x_real: &Array, x_imag: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let f = |inp: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
        let (xr, xi, cos, sin) = (&inp[0], &inp[1], &inp[2], &inp[3]);
        let out_real = subtract(&multiply(xr, cos)?, &multiply(xi, sin)?)?;
        let out_imag = add(&multiply(xr, sin)?, &multiply(xi, cos)?)?;
        Ok(vec![out_real, out_imag])
    };
    let args = [x_real.clone(), x_imag.clone(), cos.clone(), sin.clone()];
    let mut out = if crate::transformer::compile_glue() {
        compile(f, true)(&args)?
    } else {
        f(&args)?
    };
    let imag = out.pop().unwrap();
    let real = out.pop().unwrap();
    Ok((real, imag))
}

const MAX_SEQ_LEN: usize = 1024;
const ROPE_THETA: f64 = 10000.0;

/// The precomputed 3-axis RoPE frequency table: host-side cos/sin laid out `[MAX_SEQ_LEN, half_d]`
/// row-major, with column layout `[temporal_half | axis_half | axis_half]`.
pub struct RopeTable {
    pub half_d: usize,
    pub temporal_half: usize,
    pub axis_half: usize,
    cos: Vec<f32>, // [MAX_SEQ_LEN * half_d]
    sin: Vec<f32>,
}

impl RopeTable {
    /// Build the table for a given `head_dim` (θ = 10000). Mirrors `WanModel.freqs`.
    pub fn new(head_dim: usize) -> Self {
        let d6 = head_dim / 6;
        let temporal_dim = head_dim - 4 * d6; // 44 for d=128
        let axis_dim = 2 * d6; // 42
        let temporal_half = temporal_dim / 2; // 22
        let axis_half = axis_dim / 2; // 21
        let half_d = temporal_half + axis_half + axis_half; // 64 = head_dim/2

        let mut cos = vec![0f32; MAX_SEQ_LEN * half_d];
        let mut sin = vec![0f32; MAX_SEQ_LEN * half_d];

        // Per-axis inverse frequencies: inv[j] = theta^(-(2j)/axis_dim).
        let inv_freqs = |axis_dim: usize, n_half: usize| -> Vec<f64> {
            (0..n_half)
                .map(|j| ROPE_THETA.powf(-((2 * j) as f64) / axis_dim as f64))
                .collect()
        };
        let inv_t = inv_freqs(temporal_dim, temporal_half);
        let inv_a = inv_freqs(axis_dim, axis_half);

        for pos in 0..MAX_SEQ_LEN {
            let row = pos * half_d;
            let p = pos as f64;
            // temporal columns [0, temporal_half)
            for (j, &inv) in inv_t.iter().enumerate() {
                let ang = p * inv;
                cos[row + j] = ang.cos() as f32;
                sin[row + j] = ang.sin() as f32;
            }
            // height columns [temporal_half, temporal_half + axis_half)
            for (j, &inv) in inv_a.iter().enumerate() {
                let ang = p * inv;
                cos[row + temporal_half + j] = ang.cos() as f32;
                sin[row + temporal_half + j] = ang.sin() as f32;
            }
            // width columns [temporal_half + axis_half, half_d)
            for (j, &inv) in inv_a.iter().enumerate() {
                let ang = p * inv;
                cos[row + temporal_half + axis_half + j] = ang.cos() as f32;
                sin[row + temporal_half + axis_half + j] = ang.sin() as f32;
            }
        }

        Self {
            half_d,
            temporal_half,
            axis_half,
            cos,
            sin,
        }
    }

    /// Precompute the per-position `(cos, sin)` tensors for a constant grid `(f, h, w)`. Returns two
    /// f32 arrays of shape `[seq_len, 1, half_d]` (`seq_len = f·h·w`) ready to broadcast against a
    /// `[B, seq_len, n_heads, half_d]` real/imag split. Mirrors `rope_precompute_cos_sin`.
    pub fn precompute_cos_sin(&self, grid: (usize, usize, usize)) -> Result<(Array, Array)> {
        let (f, h, w) = grid;
        let seq_len = f * h * w;
        let half_d = self.half_d;
        let t0 = self.temporal_half;
        let t1 = self.temporal_half + self.axis_half;

        let mut cos_out = vec![0f32; seq_len * half_d];
        let mut sin_out = vec![0f32; seq_len * half_d];

        // Sequence position p (row-major over the grid) = (ti, hi, wi).
        let mut p = 0usize;
        for ti in 0..f {
            for hi in 0..h {
                for wi in 0..w {
                    let dst = p * half_d;
                    // temporal columns [0, t0) indexed by ti; height [t0, t1) by hi; width [t1, half_d) by wi.
                    let src_t = ti * half_d;
                    let src_h = hi * half_d;
                    let src_w = wi * half_d;
                    cos_out[dst..dst + t0].copy_from_slice(&self.cos[src_t..src_t + t0]);
                    sin_out[dst..dst + t0].copy_from_slice(&self.sin[src_t..src_t + t0]);
                    cos_out[dst + t0..dst + t1].copy_from_slice(&self.cos[src_h + t0..src_h + t1]);
                    sin_out[dst + t0..dst + t1].copy_from_slice(&self.sin[src_h + t0..src_h + t1]);
                    cos_out[dst + t1..dst + half_d]
                        .copy_from_slice(&self.cos[src_w + t1..src_w + half_d]);
                    sin_out[dst + t1..dst + half_d]
                        .copy_from_slice(&self.sin[src_w + t1..src_w + half_d]);
                    p += 1;
                }
            }
        }

        let shape = [seq_len as i32, 1, half_d as i32];
        Ok((
            Array::from_slice(&cos_out, &shape),
            Array::from_slice(&sin_out, &shape),
        ))
    }
}

/// Apply 3-axis RoPE to a Q or K tensor using precomputed `(cos, sin)`.
///
/// * `x` — `[B, S, n_heads, head_dim]` (f32; the attention path casts up before calling).
/// * `cos`, `sin` — `[seq_len, 1, half_d]` from [`RopeTable::precompute_cos_sin`], `seq_len ≤ S`.
///
/// Rotates the first `seq_len` positions with **interleaved** pairs and leaves any padding tail
/// (`seq_len..S`) untouched — the "same grid across the batch" vectorized path of `rope_apply`.
pub fn rope_apply(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let shape = x.shape();
    let (b, s, n, d) = (shape[0], shape[1], shape[2], shape[3]);
    let half_d = d / 2;
    let seq_len = cos.shape()[0];

    // Split off the rotated prefix [b, seq_len, n, d] (and any padding tail) along axis 1.
    let (head, tail) = if seq_len < s {
        let head_idx = Array::from_slice(&(0..seq_len).collect::<Vec<i32>>(), &[seq_len]);
        let tail_idx = Array::from_slice(&(seq_len..s).collect::<Vec<i32>>(), &[s - seq_len]);
        (x.take_axis(&head_idx, 1)?, Some(x.take_axis(&tail_idx, 1)?))
    } else {
        (x.clone(), None)
    };

    // Interleaved pairs: [b, seq_len, n, half_d, 2] → real/imag halves on the last axis.
    let x5 = head.reshape(&[b, seq_len, n, half_d, 2])?;
    let parts = split(&x5, 2, 4)?;
    let x_real = parts[0].reshape(&[b, seq_len, n, half_d])?;
    let x_imag = parts[1].reshape(&[b, seq_len, n, half_d])?;

    // (a + bi)·(cos + sin·i) = (a·cos − b·sin) + (a·sin + b·cos)i.
    let (out_real, out_imag) = rope_rotate(&x_real, &x_imag, cos, sin)?;

    // Interleave back: concat on a new trailing axis → [b, seq_len, n, half_d, 2] → [b, seq_len, n, d].
    let real5 = out_real.reshape(&[b, seq_len, n, half_d, 1])?;
    let imag5 = out_imag.reshape(&[b, seq_len, n, half_d, 1])?;
    let stacked = concatenate_axis(&[&real5, &imag5], 4)?;
    let rotated = stacked.reshape(&[b, seq_len, n, d])?;

    match tail {
        Some(t) => Ok(concatenate_axis(&[&rotated, &t], 1)?),
        None => Ok(rotated),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_layout_for_5b_head_dim() {
        let t = RopeTable::new(128);
        assert_eq!(t.temporal_half, 22);
        assert_eq!(t.axis_half, 21);
        assert_eq!(t.half_d, 64);
        // Position 0 → all angles 0 → cos 1, sin 0.
        assert_eq!(t.cos[0], 1.0);
        assert_eq!(t.sin[0], 0.0);
        // First temporal frequency at pos 1, col 0: inv = theta^0 = 1 → angle 1 rad.
        assert!((t.cos[64] - 1.0_f32.cos()).abs() < 1e-5);
        assert!((t.sin[64] - 1.0_f32.sin()).abs() < 1e-5);
    }

    #[test]
    fn precompute_shape_and_norm_preserving() {
        let t = RopeTable::new(128);
        let (cos, sin) = t.precompute_cos_sin((2, 3, 4)).unwrap(); // seq_len 24
        assert_eq!(cos.shape(), &[24, 1, 64]);
        assert_eq!(sin.shape(), &[24, 1, 64]);

        // RoPE is an orthogonal rotation per (a,b) pair → preserves L2 norm.
        let x = Array::ones::<f32>(&[1, 24, 2, 128]).unwrap();
        let y = rope_apply(&x, &cos, &sin).unwrap();
        assert_eq!(y.shape(), x.shape());
        let xn: f32 = mlx_rs::ops::sum(multiply(&x, &x).unwrap(), None)
            .unwrap()
            .item();
        let yn: f32 = mlx_rs::ops::sum(multiply(&y, &y).unwrap(), None)
            .unwrap()
            .item();
        assert!((xn - yn).abs() / xn < 1e-4, "norm changed: {xn} vs {yn}");
    }

    #[test]
    fn rope_apply_leaves_padding_tail() {
        let t = RopeTable::new(128);
        let (cos, sin) = t.precompute_cos_sin((1, 2, 2)).unwrap(); // seq_len 4
                                                                   // S = 6 (2 padding tokens). Tail must be returned unchanged.
        let x = Array::ones::<f32>(&[1, 6, 1, 128]).unwrap();
        let y = rope_apply(&x, &cos, &sin).unwrap();
        assert_eq!(y.shape(), &[1, 6, 1, 128]);
        let ys = y.as_slice::<f32>();
        // Last token (index 5) is in the tail → untouched ones.
        let tail_off = 5 * 128;
        assert!((ys[tail_off] - 1.0).abs() < 1e-6);
    }
}
