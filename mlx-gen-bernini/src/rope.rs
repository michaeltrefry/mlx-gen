//! Source-id rotary embedding (`use_src_id_rotary_emb`) — the Bernini delta on top of the standard
//! 3-axis Wan RoPE ([`mlx_gen_wan::RopeTable`]).
//!
//! Upstream (`transformer_wan.py:282-289`) computes a per-source phase
//! `get_1d_rotary_pos_embed(attention_head_dim, pos=source_id)` — a complex unit-modulus vector
//! `e^{i·source_id·ω_k}` of width `head_dim/2` with `ω_k = θ^(-2k/head_dim)` (θ = 10000) — and
//! **complex-multiplies** it into the spatial RoPE `freqs` (`freqs = freqs * freqs_visual_id`). A
//! complex multiply of unit-modulus exponentials is an angle add, so per lane `k`:
//!
//! ```text
//! θ_final[p,k] = θ_spatial[p,k] + source_id · ω_k
//! ```
//!
//! We carry the spatial RoPE as the precomputed `(cos, sin)` from
//! [`mlx_gen_wan::RopeTable::precompute_cos_sin`] and apply the source phase by the same complex
//! multiply, computing the per-lane `(cos(source_id·ω_k), sin(source_id·ω_k))` in **f64** host-side to
//! match the reference's `freqs_dtype=torch.float64`, then folding it in with f32 MLX ops. `source_id
//! = 0` ⇒ phase 1 ⇒ identity (the noisy target keeps the plain spatial RoPE).

use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

/// Wan RoPE base (matches `mlx_gen_wan::rope`'s `ROPE_THETA`).
const ROPE_THETA: f64 = 10000.0;

/// Compose the source-id phase onto a precomputed spatial RoPE `(cos, sin)` (each `[seq, 1, half_d]`,
/// `half_d = head_dim/2`). Returns the per-source `(cos', sin')` for that source's tokens — feed the
/// result to [`mlx_gen_wan::rope_apply`] exactly as the spatial table would be. `source_id = 0.0`
/// returns the inputs unchanged (the noisy target).
pub fn apply_source_id(
    cos_sp: &Array,
    sin_sp: &Array,
    source_id: f64,
    head_dim: usize,
) -> Result<(Array, Array)> {
    if source_id == 0.0 {
        return Ok((cos_sp.clone(), sin_sp.clone()));
    }
    let half_d = head_dim / 2;
    // Per-lane phase (cos(source_id·ω_k), sin(source_id·ω_k)), ω_k = θ^(-2k/head_dim), computed f64.
    let mut cos_id = vec![0f32; half_d];
    let mut sin_id = vec![0f32; half_d];
    for k in 0..half_d {
        let inv = ROPE_THETA.powf(-((2 * k) as f64) / head_dim as f64);
        let ang = source_id * inv;
        cos_id[k] = ang.cos() as f32;
        sin_id[k] = ang.sin() as f32;
    }
    let shape = [1, 1, half_d as i32];
    let cos_id = Array::from_slice(&cos_id, &shape);
    let sin_id = Array::from_slice(&sin_id, &shape);
    // Complex multiply (cos_sp + i·sin_sp)·(cos_id + i·sin_id).
    let cos_out = subtract(&multiply(cos_sp, &cos_id)?, &multiply(sin_sp, &sin_id)?)?;
    let sin_out = add(&multiply(sin_sp, &cos_id)?, &multiply(cos_sp, &sin_id)?)?;
    Ok((cos_out, sin_out))
}

/// Assign source-ids to `n` conditioning sources (the noisy target separately keeps id 0). Mirrors
/// upstream `_make_sids` (`wan_diffusion.py:369-374`): ids start at 1; when `interpolate` is on and
/// `n > max_trained`, the ids are evenly spread into the trained range `[1, max_trained]` via
/// `linspace(1, max_trained, n)` (fractional) instead of extrapolating past the largest id seen in
/// training; otherwise they are the integers `1..=n`.
pub fn assign_source_ids(n: usize, max_trained: f64, interpolate: bool) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if interpolate && n as f64 > max_trained {
        if n == 1 {
            return vec![1.0];
        }
        let step = (max_trained - 1.0) / (n as f64 - 1.0);
        (0..n).map(|i| 1.0 + step * i as f64).collect()
    } else {
        (1..=n).map(|i| i as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen_wan::{rope_apply, RopeTable};
    use mlx_rs::ops::sum;

    fn max_abs_diff(a: &Array, b: &Array) -> f32 {
        let d = subtract(a, b).unwrap().abs().unwrap();
        mlx_rs::ops::max(&d, None).unwrap().item::<f32>()
    }

    #[test]
    fn source_id_zero_is_identity() {
        let table = RopeTable::new(128);
        let (cos, sin) = table.precompute_cos_sin((2, 3, 4)).unwrap();
        let (cos0, sin0) = apply_source_id(&cos, &sin, 0.0, 128).unwrap();
        assert_eq!(max_abs_diff(&cos, &cos0), 0.0);
        assert_eq!(max_abs_diff(&sin, &sin0), 0.0);
    }

    #[test]
    fn source_id_phase_is_norm_preserving() {
        // Composing a unit-modulus phase keeps cos²+sin² = 1 per lane, so rope stays an orthogonal
        // rotation (norm-preserving) for any source_id.
        let table = RopeTable::new(128);
        let (cos, sin) = table.precompute_cos_sin((1, 4, 4)).unwrap(); // seq 16
        let (cos3, sin3) = apply_source_id(&cos, &sin, 3.0, 128).unwrap();
        let unit = add(
            multiply(&cos3, &cos3).unwrap(),
            multiply(&sin3, &sin3).unwrap(),
        )
        .unwrap();
        let ones = Array::ones::<f32>(unit.shape()).unwrap();
        assert!(max_abs_diff(&unit, &ones) < 1e-5, "cos²+sin² ≠ 1");

        let x = Array::ones::<f32>(&[1, 16, 2, 128]).unwrap();
        let y = rope_apply(&x, &cos3, &sin3).unwrap();
        let xn: f32 = sum(multiply(&x, &x).unwrap(), None).unwrap().item();
        let yn: f32 = sum(multiply(&y, &y).unwrap(), None).unwrap().item();
        assert!(
            (xn - yn).abs() / xn < 1e-4,
            "rope changed norm: {xn} vs {yn}"
        );
    }

    #[test]
    fn source_id_phase_matches_manual_complex_multiply() {
        // Lane 0 has ω_0 = θ^0 = 1, so the phase for source_id s is exactly (cos s, sin s); at
        // position 0 the spatial angle is 0, so cos'[0,0]=cos(s), sin'[0,0]=sin(s).
        let table = RopeTable::new(128);
        let (cos, sin) = table.precompute_cos_sin((1, 1, 1)).unwrap(); // seq 1
        let s = 2.5_f64;
        let (cos2, sin2) = apply_source_id(&cos, &sin, s, 128).unwrap();
        let c = cos2.as_slice::<f32>()[0];
        let si = sin2.as_slice::<f32>()[0];
        assert!((c - s.cos() as f32).abs() < 1e-5, "cos lane0 = {c}");
        assert!((si - s.sin() as f32).abs() < 1e-5, "sin lane0 = {si}");
    }

    #[test]
    fn assign_source_ids_integer_and_interpolated() {
        assert_eq!(assign_source_ids(3, 5.0, true), vec![1.0, 2.0, 3.0]);
        assert_eq!(
            assign_source_ids(5, 5.0, true),
            vec![1.0, 2.0, 3.0, 4.0, 5.0]
        );
        // n > max_trained with interpolation → linspace(1, 5, n).
        let ids = assign_source_ids(9, 5.0, true);
        assert_eq!(ids.len(), 9);
        assert_eq!(ids[0], 1.0);
        assert_eq!(*ids.last().unwrap(), 5.0);
        assert!((ids[1] - 1.5).abs() < 1e-9);
        // interpolation off → integers even past max_trained.
        assert_eq!(
            assign_source_ids(7, 5.0, false),
            (1..=7).map(|i| i as f64).collect::<Vec<_>>()
        );
        assert_eq!(assign_source_ids(0, 5.0, true), Vec::<f64>::new());
    }
}
