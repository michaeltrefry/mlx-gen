//! Adaptive-Projected-Guidance (APG) — the x-space guidance core the Bernini renderer's `*_apg`
//! modes use (`wan_diffusion.py:_normalize_diff` / `normalized_guidance` / `normalized_guidance_chain`,
//! lines 91-124).
//!
//! APG runs on the **x-space** velocity prediction (`x = noisy − σ·v`) and replaces a plain CFG
//! `uncond + scale·(cond − uncond)` with a projected, momentum-smoothed, norm-clamped update:
//!
//! 1. **Momentum** (optional): `running = diff + momentum·running` (buffer persists across denoise
//!    steps; starts at 0 ⇒ first step is just `diff`).
//! 2. **Norm clamp** (when `norm_threshold > 0`): scale `diff` by `min(1, norm_threshold/‖diff‖)`.
//! 3. **Projection**: split `diff` into the component parallel to the conditional prediction `base`
//!    and the orthogonal remainder, and recombine as `orthogonal + eta·parallel`.
//!
//! The L2 norm and projection reduce over **channels + spatial** (`dim=[-1,-2,-4]` on the reference's
//! `[B,C,T,H,W]`; here `[0,2,3]` on a `[C,T,H,W]` velocity), i.e. per frame, excluding time.
//!
//! **Parity note:** the reference computes the projection in float64; MLX has no robust Metal f64, so
//! this runs in f32. Combined with the f32 source-id RoPE, this is the documented main divergence vs
//! the torch reference (the validation bar is component parity + coherent output, not bit-parity).

use mlx_rs::ops::{add, divide, maximum, minimum, multiply, sqrt, subtract};
use mlx_rs::Array;

use mlx_gen::Result;

/// APG reduction dims on a `[C, T, H, W]` velocity (channels + spatial, per frame) — the reference's
/// `dim=[-1,-2,-4]` on its `[B, C, T, H, W]` layout (which excludes the temporal axis).
const APG_DIMS: &[i32] = &[0, 2, 3];
/// `F.normalize` default eps (the denominator floor for the unit base direction).
const NORMALIZE_EPS: f32 = 1e-12;

/// Persistent momentum accumulator for one APG stream (mirrors upstream `MomentumBuffer`). One per
/// guidance term, allocated **before** the denoise loop so the running average carries across steps.
pub struct MomentumBuffer {
    momentum: f32,
    running: Option<Array>,
}

impl MomentumBuffer {
    pub fn new(momentum: f32) -> Self {
        Self {
            momentum,
            running: None,
        }
    }

    /// `running = diff + momentum·running` (the reference's `update`); the first call (running = 0)
    /// returns `diff` unchanged.
    fn update(&mut self, diff: &Array) -> Result<Array> {
        let ra = match &self.running {
            Some(r) => add(diff, &multiply(r, Array::from_f32(self.momentum))?)?,
            None => diff.clone(),
        };
        self.running = Some(ra.clone());
        Ok(ra)
    }
}

/// L2 norm over `[C, H, W]` per frame, keepdims → `[1, T, 1, 1]`.
fn l2_norm(a: &Array) -> Result<Array> {
    Ok(sqrt(&multiply(a, a)?.sum_axes(APG_DIMS, true)?)?)
}

/// The APG core: momentum → norm-clamp → orthogonal/parallel projection against `base`, returning
/// `orthogonal + eta·parallel`. `base` is the **conditional** x-prediction (the projection reference).
fn normalize_diff(
    diff: &Array,
    base: &Array,
    buf: Option<&mut MomentumBuffer>,
    eta: f32,
    norm_threshold: f32,
) -> Result<Array> {
    let mut diff = match buf {
        Some(b) => b.update(diff)?,
        None => diff.clone(),
    };
    if norm_threshold > 0.0 {
        let dn = l2_norm(&diff)?;
        let scale = minimum(
            Array::from_f32(1.0),
            &divide(Array::from_f32(norm_threshold), &dn)?,
        )?;
        diff = multiply(&diff, &scale)?;
    }
    // Unit base direction (F.normalize): base / max(‖base‖, eps).
    let bn = maximum(&l2_norm(base)?, Array::from_f32(NORMALIZE_EPS))?;
    let v1 = divide(base, &bn)?;
    // Parallel component = (diff·v1)·v1; orthogonal = diff − parallel.
    let coeff = multiply(&diff, &v1)?.sum_axes(APG_DIMS, true)?;
    let parallel = multiply(&coeff, &v1)?;
    let orthogonal = subtract(&diff, &parallel)?;
    Ok(add(
        &orthogonal,
        &multiply(&parallel, Array::from_f32(eta))?,
    )?)
}

/// Single-condition APG: `uncond + scale · normalize_diff(cond − uncond, base = cond)`
/// (`normalized_guidance`). With `eta = 1`, `norm_threshold = 0`, and no momentum this is exactly
/// plain CFG `uncond + scale·(cond − uncond)`.
pub fn normalized_guidance(
    cond: &Array,
    uncond: &Array,
    scale: f32,
    buf: Option<&mut MomentumBuffer>,
    eta: f32,
    norm_threshold: f32,
) -> Result<Array> {
    let nd = normalize_diff(&subtract(cond, uncond)?, cond, buf, eta, norm_threshold)?;
    Ok(add(uncond, &multiply(&nd, Array::from_f32(scale))?)?)
}

/// Chained APG over an ordered list of predictions (`normalized_guidance_chain`). With
/// `bases = [uncond, preds[0], preds[1], …]`, accumulates
/// `result = uncond + Σ_i scales[i] · normalize_diff(preds[i] − bases[i], base = preds[i])`, each term
/// with its own momentum buffer and norm threshold. Used by `r2v_apg` over `[x_I, x_TI]`.
pub fn normalized_guidance_chain(
    uncond: &Array,
    preds: &[Array],
    scales: &[f32],
    bufs: &mut [MomentumBuffer],
    eta: f32,
    norm_thresholds: &[f32],
) -> Result<Array> {
    let mut result = uncond.clone();
    for (i, cond) in preds.iter().enumerate() {
        let base_prev = if i == 0 { uncond } else { &preds[i - 1] };
        let nd = normalize_diff(
            &subtract(cond, base_prev)?,
            cond,
            Some(&mut bufs[i]),
            eta,
            norm_thresholds[i],
        )?;
        result = add(&result, &multiply(&nd, Array::from_f32(scales[i]))?)?;
    }
    Ok(result)
}

/// `clamp_min` floor for the `apg_delta` reference norm² (the reference's `eps=1e-8`).
const APG_DELTA_EPS: f32 = 1e-8;

/// The **v-space** APG delta projection used by the full-Bernini ViT-conditioned modes
/// (`wan_diffusion.py:apg_delta`, "veomni_editing Wan2.2"). Distinct from [`normalize_diff`]: it
/// projects `delta` onto `reference` over the **whole flattened tensor** (per batch element, not
/// per-frame) and recombines with fixed parallel/orthogonal scales (∥0.2 / ⊥1.0):
///
///   `proj = (delta·ref)/max(‖ref‖², eps)·ref`; `out = parallel_scale·proj + orthogonal_scale·(delta − proj)`.
///
/// `delta`/`reference` are `[1, n_target, C]` (the target-sliced packed-token predictions, batch 1);
/// the reduction is over `n_target·C`.
pub fn apg_delta(
    delta: &Array,
    reference: &Array,
    parallel_scale: f32,
    orthogonal_scale: f32,
) -> Result<Array> {
    // reshape(b, -1) + reduce dim=1 ≡ reduce over every axis except the batch axis 0.
    let dims: Vec<i32> = (1..delta.ndim() as i32).collect();
    let ref_norm_sq = maximum(
        &multiply(reference, reference)?.sum_axes(&dims, true)?,
        Array::from_f32(APG_DELTA_EPS),
    )?;
    let coeff = divide(
        &multiply(delta, reference)?.sum_axes(&dims, true)?,
        &ref_norm_sq,
    )?;
    let parallel = multiply(&coeff, reference)?;
    let orthogonal = subtract(delta, &parallel)?;
    Ok(add(
        &multiply(&parallel, Array::from_f32(parallel_scale))?,
        &multiply(&orthogonal, Array::from_f32(orthogonal_scale))?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs(a: &Array, b: &Array) -> f32 {
        mlx_rs::ops::max(subtract(a, b).unwrap().abs().unwrap(), None)
            .unwrap()
            .item::<f32>()
    }

    fn randish(seed: i32) -> Array {
        // Deterministic varied [C=4, T=2, H=2, W=2] tensor.
        let n = 4 * 2 * 2 * 2;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + seed * 13) % 11) as f32 - 5.0)
            .collect();
        Array::from_slice(&v, &[4, 2, 2, 2])
    }

    #[test]
    fn apg_reduces_to_plain_cfg_at_eta1_no_clamp() {
        let cond = randish(1);
        let uncond = randish(2);
        let scale = 4.0_f32;
        let got = normalized_guidance(&cond, &uncond, scale, None, 1.0, 0.0).unwrap();
        // plain CFG: uncond + scale·(cond − uncond)
        let want = add(
            &uncond,
            multiply(subtract(&cond, &uncond).unwrap(), Array::from_f32(scale)).unwrap(),
        )
        .unwrap();
        assert!(
            max_abs(&got, &want) < 1e-4,
            "eta=1/nt=0 must equal plain CFG"
        );
    }

    #[test]
    fn apg_eta0_drops_parallel_component() {
        // eta=0 ⇒ nd is purely orthogonal to `cond`, so (nd · cond) summed over C,H,W ≈ 0 per frame.
        let cond = randish(3);
        let uncond = randish(4);
        let nd = normalize_diff(&subtract(&cond, &uncond).unwrap(), &cond, None, 0.0, 0.0).unwrap();
        let dot = multiply(&nd, &cond)
            .unwrap()
            .sum_axes(APG_DIMS, true)
            .unwrap();
        let zeros = Array::zeros::<f32>(dot.shape()).unwrap();
        assert!(max_abs(&dot, &zeros) < 1e-3, "eta=0 orthogonal residual");
    }

    #[test]
    fn apg_norm_threshold_clamps_diff() {
        // A large diff with a small threshold is scaled so ‖diff‖ ≤ threshold per frame.
        let cond = multiply(randish(5), Array::from_f32(100.0)).unwrap();
        let uncond = Array::zeros::<f32>(&[4, 2, 2, 2]).unwrap();
        // Inspect the clamp directly via the single-term path with eta=1 (nd == clamped diff).
        let nd = normalize_diff(&subtract(&cond, &uncond).unwrap(), &cond, None, 1.0, 2.0).unwrap();
        let norms = l2_norm(&nd).unwrap();
        let m = mlx_rs::ops::max(&norms, None).unwrap().item::<f32>();
        assert!(m <= 2.0 + 1e-3, "clamped norm {m} must be ≤ threshold 2.0");
    }

    #[test]
    fn momentum_accumulates_across_calls() {
        let mut buf = MomentumBuffer::new(-0.5);
        let d1 = randish(6);
        let r1 = buf.update(&d1).unwrap();
        assert!(max_abs(&r1, &d1) < 1e-6, "first update returns diff");
        let d2 = randish(7);
        let r2 = buf.update(&d2).unwrap();
        // running = d2 + (-0.5)·d1
        let want = add(&d2, multiply(&d1, Array::from_f32(-0.5)).unwrap()).unwrap();
        assert!(max_abs(&r2, &want) < 1e-5, "second update accumulates");
    }
}
