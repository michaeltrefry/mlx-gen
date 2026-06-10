//! SPLIT 3-D RoPE, **double-precision** — port of `models/ltx/rope.py`'s
//! `_precompute_freqs_cis_double_precision` (the path `generate_av.py` takes for LTX-2.3:
//! `rope_type="split"`, `double_precision=True`) plus `apply_split_rotary_emb`.
//!
//! The reference computes the frequency grid in numpy **float64** and only down-casts the final
//! cos/sin tables to float32 for the GPU — the comment in `rope.py` warns that bf16 positions
//! degrade video quality. We mirror that exactly: the whole grid is built in Rust `f64` from the
//! f32 position grid, emitting `f32` `Array`s. (Building in f32 throughout would diverge from the
//! reference's f64 accumulation over 682 log-spaced frequencies.)
//!
//! For the video stream `dim = inner_dim = heads · head_dim = 4096` (NOT head_dim): the freqs are
//! padded to `dim/2 = 2048` then reshaped to `(B, heads, T, head_dim/2)` = `(B, 32, T, 64)` — the
//! per-head half-rotation tables consumed by [`apply_split_rotary_emb`].

use std::f64::consts::PI;

use mlx_rs::error::Exception;
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use mlx_gen::{Error, Result};

/// The GPT-NeoX "rotate-halves" rotation `(first·cos − second·sin, second·cos + first·sin)` on the
/// already-split f32 halves. Fused into one kernel when the sc-2963 glue toggle is on (vs 6 eager ops,
/// applied to q and k every block — the dominant elementwise RoPE cost at video sequence). Bit-exact.
fn rope_rotate(first: &Array, second: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let f = |inp: &[Array]| -> std::result::Result<Vec<Array>, Exception> {
        let (a, b, c, s) = (&inp[0], &inp[1], &inp[2], &inp[3]);
        let out_first = subtract(&multiply(a, c)?, &multiply(b, s)?)?;
        let out_second = add(&multiply(b, c)?, &multiply(a, s)?)?;
        Ok(vec![out_first, out_second])
    };
    let args = [first.clone(), second.clone(), cos.clone(), sin.clone()];
    let mut out = if crate::compile_glue() {
        compile(f, true)(&args)?
    } else {
        f(&args)?
    };
    let out_second = out.pop().unwrap();
    let out_first = out.pop().unwrap();
    Ok((out_first, out_second))
}

/// Precompute the SPLIT RoPE `(cos, sin)` tables in double precision.
///
/// * `positions` — the f32 position grid `(B, n_pos_dims, T, 2)` from [`crate::positions`]. The
///   video stream uses `n_pos_dims=3`; the audio stream + the cross-modal q/k tables use a 1-D grid
///   (`n_pos_dims=1`, the time axis).
/// * `dim` — RoPE dimension = the **inner dim** (`heads · head_dim`); 2048 for audio / cross-modal.
/// * `theta` — base frequency (10000).
/// * `max_pos` — per-axis maxima (`[20, 2048, 2048]` video, `[20]` audio/cross); `len == n_pos_dims`.
/// * `num_attention_heads` — heads to fold the freqs into.
///
/// Returns `(cos, sin)`, each f32 `(B, num_attention_heads, T, dim/2/heads)`.
pub fn precompute_split_freqs_cis(
    positions: &Array,
    dim: i32,
    theta: f64,
    max_pos: &[i32],
    num_attention_heads: i32,
) -> Result<(Array, Array)> {
    let shape = positions.shape();
    if shape.len() != 4 {
        return Err(Error::Msg(format!(
            "precompute_split_freqs_cis: positions must be rank-4 (B, n_pos_dims, T, 2), got {shape:?}"
        )));
    }
    let batch = shape[0] as usize;
    let n_pos_dims = shape[1] as usize;
    let seq = shape[2] as usize; // T
    if shape[3] != 2 {
        return Err(Error::Msg(format!(
            "precompute_split_freqs_cis: positions last axis must be 2 (start, end), got {}",
            shape[3]
        )));
    }
    if n_pos_dims != max_pos.len() {
        return Err(Error::Msg(format!(
            "precompute_split_freqs_cis: n_pos_dims ({n_pos_dims}) must equal max_pos.len() ({})",
            max_pos.len()
        )));
    }

    let pos = positions.as_slice::<f32>();
    // C-order index into (B, 3, T, 2): ((b*3 + d)*T + t)*2 + e.
    let idx = |b: usize, d: usize, t: usize, e: usize| ((b * n_pos_dims + d) * seq + t) * 2 + e;

    let n_elem = 2 * n_pos_dims; // 6
    let mut num_indices = (dim as usize) / n_elem; // 4096/6 = 682
    if num_indices == 0 {
        num_indices = 1;
    }

    // indices[i] = theta^linspace(0,1,num_indices)[i] * (pi/2), in f64.
    // linspace(log(1)/log(theta)=0, log(theta)/log(theta)=1, num_indices).
    // linspace(0, 1, num_indices) as numpy computes it: y[i] = i · step (step = 1/(num-1)).
    let step = if num_indices == 1 {
        0.0
    } else {
        1.0 / (num_indices - 1) as f64
    };
    let indices: Vec<f64> = (0..num_indices)
        .map(|i| theta.powf(i as f64 * step) * (PI / 2.0))
        .collect();

    let current = num_indices * n_pos_dims; // 2046
    let expected = (dim as usize) / 2; // 2048
    let pad_size = expected.saturating_sub(current); // 2
    let head_half = expected / (num_attention_heads as usize); // 64

    // Build the padded cos/sin in (B, T, expected) order, then reshape to (B, heads, T, head_half).
    let total = batch * (num_attention_heads as usize) * seq * head_half;
    let mut cos_out = vec![0f32; total];
    let mut sin_out = vec![0f32; total];

    for b in 0..batch {
        for t in 0..seq {
            // scaled position per axis (use middle of [start,end], fractional, then *2-1).
            // Sized to `n_pos_dims` (the grid's axis count), not a fixed 3 — `scaled[d]` is later
            // indexed by `d = k % n_pos_dims`, which would overflow a `[_; 3]` for >3-axis grids.
            let mut scaled = vec![0f64; n_pos_dims];
            for (d, s) in scaled.iter_mut().enumerate() {
                let start = pos[idx(b, d, t, 0)] as f64;
                let end = pos[idx(b, d, t, 1)] as f64;
                let mid = (start + end) / 2.0;
                let frac = mid / max_pos[d] as f64;
                *s = frac * 2.0 - 1.0;
            }

            // padded vector of length `expected`: [pad (cos=1/sin=0)] ++ freqs.
            // freqs[k] for k = i*n_pos_dims + d → scaled[d] * indices[i] (idx-major, dim-minor).
            for h in 0..(num_attention_heads as usize) {
                for p in 0..head_half {
                    let flat = h * head_half + p; // 0..expected
                    let (c, s);
                    if flat < pad_size {
                        c = 1.0f32;
                        s = 0.0f32;
                    } else {
                        let k = flat - pad_size; // 0..current
                        let i = k / n_pos_dims;
                        let d = k % n_pos_dims;
                        let ang = scaled[d] * indices[i];
                        c = ang.cos() as f32;
                        s = ang.sin() as f32;
                    }
                    let o = ((b * (num_attention_heads as usize) + h) * seq + t) * head_half + p;
                    cos_out[o] = c;
                    sin_out[o] = s;
                }
            }
        }
    }

    let out_shape = [
        batch as i32,
        num_attention_heads,
        seq as i32,
        head_half as i32,
    ];
    Ok((
        Array::from_slice(&cos_out, &out_shape),
        Array::from_slice(&sin_out, &out_shape),
    ))
}

/// Apply SPLIT (half-rotation) RoPE to a head-split tensor.
///
/// * `x` — `(B, H, T, D)` (q or k after head reshape).
/// * `cos`, `sin` — `(B, H, T, D/2)` from [`precompute_split_freqs_cis`].
///
/// Splits the last axis into halves `(a, b)` and rotates:
/// `out = [a·cos − b·sin, b·cos + a·sin]` — the GPT-NeoX "rotate-halves" form, matching
/// `rope.py::apply_split_rotary_emb`.
pub fn apply_split_rotary_emb(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    // Compute in f32 for precision, then cast back to the input dtype — matching the reference's
    // `_apply_split_rope` (`...astype(input_dtype)`). This keeps a bf16 caller (the connector /
    // bf16 DiT) bf16 while an f32 caller stays f32.
    let in_dtype = x.dtype();
    let x = x.as_dtype(mlx_rs::Dtype::Float32)?;
    let cos = cos.as_dtype(mlx_rs::Dtype::Float32)?;
    let sin = sin.as_dtype(mlx_rs::Dtype::Float32)?;
    let axis = (x.ndim() - 1) as i32;
    let halves = split(&x, 2, axis)?;
    let (out_first, out_second) = rope_rotate(&halves[0], &halves[1], &cos, &sin)?;
    Ok(concatenate_axis(&[&out_first, &out_second], axis)?.as_dtype(in_dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::positions::create_position_grid;

    #[test]
    fn split_freqs_shape_and_padding() {
        // inner_dim 4096, 32 heads → head_half 64; T from a small grid.
        let pos = create_position_grid(1, 2, 2, 2); // 8 patches
        let (cos, sin) =
            precompute_split_freqs_cis(&pos, 4096, 10000.0, &[20, 2048, 2048], 32).expect("rope");
        assert_eq!(cos.shape(), &[1, 32, 8, 64]);
        assert_eq!(sin.shape(), &[1, 32, 8, 64]);

        // pad_size = 2048 - 682*3 = 2: head 0, freqs p=0,1 are pad → cos=1, sin=0.
        let c = cos.as_slice::<f32>();
        let s = sin.as_slice::<f32>();
        // index (b=0,h=0,t=0,p): ((0*32+0)*8+0)*64 + p = p.
        assert_eq!(c[0], 1.0);
        assert_eq!(c[1], 1.0);
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], 0.0);
        // p=2 is the first real frequency (i=0, d=0): scaled[0]*indices[0]; index[0]=pi/2.
        // The middle/fractional makes this small but generally != the pad sentinel.
        assert!(c[2] <= 1.0 && c[2] >= -1.0);
    }

    #[test]
    fn handles_more_than_three_pos_dims() {
        // F-054: a 4-axis grid used to panic on the fixed `[0f64; 3]` (`scaled[d]`, `d` up to 3).
        let pos = Array::from_slice(&[0f32; 8], &[1, 4, 1, 2]); // (B=1, n_pos_dims=4, T=1, 2)
        let r = precompute_split_freqs_cis(&pos, 4096, 10000.0, &[20, 20, 20, 20], 32);
        assert!(r.is_ok(), "4-axis grid must not panic/error: {:?}", r.err());
        assert_eq!(r.unwrap().0.shape(), &[1, 32, 1, 64]);
    }

    #[test]
    fn rejects_malformed_positions() {
        // F-054: shape mismatches are now typed errors, not release-vanishing `debug_assert`s.
        // last axis must be 2 (start, end):
        let bad_last = Array::from_slice(&[0f32; 9], &[1, 3, 1, 3]);
        assert!(precompute_split_freqs_cis(&bad_last, 4096, 1e4, &[1, 1, 1], 32).is_err());
        // n_pos_dims must equal max_pos.len():
        let mismatch = Array::from_slice(&[0f32; 6], &[1, 3, 1, 2]);
        assert!(precompute_split_freqs_cis(&mismatch, 4096, 1e4, &[1, 1], 32).is_err());
        // positions must be rank-4:
        let rank3 = Array::from_slice(&[0f32; 6], &[1, 3, 2]);
        assert!(precompute_split_freqs_cis(&rank3, 4096, 1e4, &[1, 1, 1], 32).is_err());
    }

    #[test]
    fn apply_rotary_is_norm_preserving() {
        // The half-rotation is orthogonal per (a,b) pair → preserves L2 norm.
        let pos = create_position_grid(1, 1, 2, 2); // 4 patches
        let (cos, sin) =
            precompute_split_freqs_cis(&pos, 4096, 10000.0, &[20, 2048, 2048], 32).unwrap();
        // x: (1, 32, 4, 128) ones.
        let x = Array::ones::<f32>(&[1, 32, 4, 128]).unwrap();
        let y = apply_split_rotary_emb(&x, &cos, &sin).unwrap();
        assert_eq!(y.shape(), x.shape());
        let xn: f32 = mlx_rs::ops::sum(multiply(&x, &x).unwrap(), None)
            .unwrap()
            .item();
        let yn: f32 = mlx_rs::ops::sum(multiply(&y, &y).unwrap(), None)
            .unwrap()
            .item();
        assert!((xn - yn).abs() / xn < 1e-4, "norm changed: {xn} vs {yn}");
    }

    // sc-2963: the compiled split-RoPE rotation is bit-identical to eager (`max|Δ|=0`). The full
    // `apply_split_rotary_emb` computes in f32, so this gates the rotation in f32 on a real input.
    #[test]
    fn compiled_rope_bit_identical_to_eager() {
        let pos = create_position_grid(1, 1, 2, 2);
        let (cos, sin) =
            precompute_split_freqs_cis(&pos, 4096, 10000.0, &[20, 2048, 2048], 32).unwrap();
        let k = mlx_rs::random::key(0).unwrap();
        let x = mlx_rs::random::normal::<f32>(&[1, 32, 4, 128], None, None, Some(&k)).unwrap();
        crate::set_compile_glue(false);
        let eager = apply_split_rotary_emb(&x, &cos, &sin).unwrap();
        crate::set_compile_glue(true);
        let compiled = apply_split_rotary_emb(&x, &cos, &sin).unwrap();
        crate::set_compile_glue(false);
        let d = mlx_rs::ops::abs(subtract(&compiled, &eager).unwrap()).unwrap();
        let m = mlx_rs::ops::max(&d, None).unwrap().item::<f32>();
        assert_eq!(m, 0.0, "compiled split-RoPE diverged from eager");
    }
}
