//! Model-agnostic neural-net primitives — the shared `nn` layer of `mlx-gen` core.
//!
//! These are the genuinely family-independent leaf ops: dense linear, SiLU, NHWC `conv2d`,
//! pytorch-compatible `group_norm`, and nearest `upsample`. Model-specific block assemblies
//! (attention / RoPE / SwiGLU layouts) intentionally stay in their family crates — see
//! `docs/MODEL_ARCHITECTURE.md` §3.2 ("each family crate owns its blocks"). A primitive
//! graduates here only once it is provably model-agnostic; we do not lift a block to a shared
//! abstraction off a single model.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{
    add, addmm, broadcast_to, conv2d as conv2d_op, conv3d as conv3d_op, divide, erf, multiply,
    power, sigmoid, tanh,
};
use mlx_rs::transforms::compile::compile;
use mlx_rs::Array;

use crate::array::scalar;
use crate::Result;

/// `y = x · Wᵀ + b` for a stored `[out, in]` weight + bias (PyTorch `nn.Linear` convention).
///
/// FUSED `addmm(bias, x, Wᵀ)` — matching MLX's own `nn.Linear` (and the core `AdaptableLinear`
/// dense path, sc-2779). A separate `matmul` then `add` double-rounds the bias add in bf16
/// (~1.4e-3/Linear, compounding over a deep net); `addmm` rounds once. f32-invisible (`addmm ==
/// matmul+add` bit-for-bit with f32 activations, even over bf16 weights), so this is safe for the
/// current f32-activation text-encoder consumers (LTX, Qwen) and correct once they go bf16.
pub fn linear(x: &Array, w: &Array, b: &Array) -> Result<Array> {
    Ok(addmm(b, x, w.t(), 1.0, 1.0)?)
}

/// SiLU / swish activation: `x · sigmoid(x)`.
pub fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

/// GELU tanh approximation — `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` — **dtype-preserving**
/// and **bit-exact** to MLX-Python's `nn.GELU(approx="tanh")` / `mx.nn.gelu_approx`.
///
/// Two traps this avoids, both latent in f32 and surfacing only on a bf16 FFN path (sc-2779):
///   1. **Scalar dtype.** MLX weak-casts its python-float constants to the *input* dtype, so a
///      bf16 input computes in bf16. `mlx_rs::nn::gelu_approximate` (and any `scalar(..)`-based
///      hand-roll) uses f32 constants, which promote a bf16 input to f32 — a ~2.4e-3 mismatch
///      vs the golden. We cast every constant to `x.dtype()` so bf16 stays bf16 and f32 stays f32.
///   2. **The `√(2/π)` constant.** `mlx_rs::nn::gelu_approximate` builds it with an f32 MLX `sqrt`
///      op (1 ULP off MLX-Python's f64-host `math.sqrt`); over a deep f32 path that 1-ULP seed
///      amplifies (Wan UMT5: ~5e-7 → ~1e-3 over 24 layers — see [[mlx-rs-gelu-approx-f64-constant]]).
///      We compute it in f64 on the host, then cast.
///
/// Result: the f32 path is bit-exact to the reference and a bf16 input is preserved in bf16.
/// Shared across the family crates' tanh-approx FFNs (Wan today; Qwen/FLUX/FLUX.2/Z-Image as their
/// forwards move f32→bf16, sc-2718–2721). `x³` via integer-exponent `power`, as MLX does.
pub fn gelu_tanh(x: &Array) -> Result<Array> {
    let dt = x.dtype();
    let s = |v: f32| -> Result<Array> { Ok(scalar(v).as_dtype(dt)?) };
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
    let gate = add(&tanh(&inner)?, &s(1.0)?)?;
    Ok(multiply(&multiply(x, &s(0.5)?)?, &gate)?)
}

/// Exact GELU — `x · (1 + erf(x/√2)) / 2` — **dtype-preserving** and **byte-identical to MLX-Python's
/// `nn.gelu`** (the activation in the SDXL GEGLU FFN — `y_a · gelu(y_b)`).
///
/// Two things are required to match the reference bit-for-bit, both fp16-only traps (f32-invisible):
///   1. **Dtype-weak constants.** Each python-float constant is cast to `x.dtype()` so an f16 input
///      stays f16. `mlx_rs::nn::gelu` builds `1/√2` / `2` as f32 *arrays* (`array!(2f32.sqrt())`),
///      which promote an f16 input to f32 — the silent f16→f32 leak that made the SDXL fp16 U-Net run
///      (and return) f32 (sc-2721). `√2` is taken in f64 (python `math.sqrt(2)`) before the cast.
///   2. **`mx.compile`.** MLX-Python decorates `nn.gelu` with `@mx.compile`; the fused kernel rounds
///      fp16 differently from the same ops run unfused (a 1-ULP/elem gap — measured as the *sole*
///      SDXL fp16 U-Net divergence: replacing the reference's compiled gelu with an unfused literal
///      reproduces the 57.7%-byte-exact / 5.3e-4 gap exactly). Compiling the identical graph here
///      reproduces the fused rounding → bit-exact. f32-safe: an f32 input is unchanged either way.
pub fn gelu_exact(x: &Array) -> Result<Array> {
    let f = |x_: &Array| {
        let dt = x_.dtype();
        let inv_sqrt2 = scalar(std::f64::consts::SQRT_2 as f32).as_dtype(dt)?;
        let one = scalar(1.0).as_dtype(dt)?;
        let two = scalar(2.0).as_dtype(dt)?;
        let inner = erf(&divide(x_, &inv_sqrt2)?)?; // erf(x / √2)
        let gate = add(&one, &inner)?; // 1 + erf(x / √2)
        divide(&multiply(x_, &gate)?, &two) // (x · gate) / 2
    };
    Ok(compile(f, true)(x)?)
}

/// Fast-approx GELU ("quick_gelu") — `x · sigmoid(1.702·x)` — **byte-identical to MLX-Python's
/// `nn.gelu_fast_approx`** (the CLIP-L / `quick_gelu` activation). Same two requirements as
/// [`gelu_exact`]: the `1.702` constant is weak-cast to `x.dtype()` (an f32 scalar would promote an
/// f16 input to f32), and the graph is `mx.compile`'d so the fused fp16 rounding matches the
/// reference. f32-safe (no-op cast, fusion numerically identical).
pub fn gelu_quick(x: &Array) -> Result<Array> {
    let f = |x_: &Array| {
        let c = scalar(1.702).as_dtype(x_.dtype())?;
        multiply(x_, &sigmoid(&multiply(x_, &c)?)?) // x · sigmoid(1.702·x)
    };
    Ok(compile(f, true)(x)?)
}

/// 2-D conv over NHWC `x` with an mlx `[out, kH, kW, in]` weight (+ optional bias).
pub fn conv2d(x: &Array, w: &Array, b: Option<&Array>, stride: i32, padding: i32) -> Result<Array> {
    let mut y = conv2d_op(x, w, (stride, stride), (padding, padding), (1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// 3-D conv over NDHWC `x` with an mlx `[out, kD, kH, kW, in]` weight (+ optional bias).
/// `stride`/`padding` are per-axis `(depth, height, width)`. Qwen's causal-Conv3d VAE applies
/// its asymmetric temporal padding manually and calls this with `padding (0, 0, 0)`; future
/// video families (Wan2.2 / LTX) reuse it directly — hence it lives in shared core `nn`.
pub fn conv3d(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: (i32, i32, i32),
    padding: (i32, i32, i32),
) -> Result<Array> {
    let mut y = conv3d_op(x, w, stride, padding, (1, 1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// PyTorch-compatible group normalization over NHWC `x` (`weight`/`bias` are per-channel).
/// Mirrors mlx-rs `GroupNorm::pytorch_group_norm` + affine: split channels into `num_groups`,
/// layer-norm each group, then scale/shift by `weight`/`bias`.
pub fn group_norm(
    x: &Array,
    weight: &Array,
    bias: &Array,
    num_groups: i32,
    eps: f32,
) -> Result<Array> {
    let sh = x.shape();
    let batch = sh[0];
    let dims = sh[sh.len() - 1];
    let rest = &sh[1..sh.len() - 1];
    let group_size = dims / num_groups;

    let g = x
        .reshape(&[batch, -1, num_groups, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[batch, num_groups, -1])?;
    let g = layer_norm(&g, None, None, eps)?;
    let g = g
        .reshape(&[batch, num_groups, -1, group_size])?
        .transpose_axes(&[0, 2, 1, 3])?;

    let mut shape = vec![batch];
    shape.extend_from_slice(rest);
    shape.push(dims);
    let normed = g.reshape(&shape)?;
    Ok(add(&multiply(&normed, weight)?, bias)?)
}

/// Nearest-neighbor upsample of NHWC `x` by `scale` (broadcast + reshape).
pub fn upsample_nearest(x: &Array, scale: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x6 = x.reshape(&[b, h, 1, w, 1, c])?;
    let bc = broadcast_to(&x6, &[b, h, scale, w, scale, c])?;
    Ok(bc.reshape(&[b, h * scale, w * scale, c])?)
}

/// Rotary position embedding for **text encoders** — the HF "half-split" convention (distinct from
/// the DiT's interleaved RoPE, which stays family-owned). Port of the fork's `RotaryEmbedding`:
/// `inv_freq = 1/θ^(arange(0,dim,2)/dim)`; `freqs = outer(arange(seq), inv_freq)`;
/// `emb = concat([freqs, freqs])`; `cos/sin = cos/sin(emb)[None]`. Shared by the Z-Image and
/// Qwen-Image text encoders, which use the identical layout (the second-family trigger for lifting
/// this to core — F-006).
pub struct TextRope {
    inv_freq: Vec<f32>,
    dim: i32,
}

impl TextRope {
    /// `dim` = head_dim, `theta` = rope base (1e6 for both Z-Image and Qwen-Image).
    pub fn new(dim: i32, theta: f32) -> Self {
        let half = (dim / 2) as usize;
        let inv_freq = (0..half)
            .map(|i| 1.0 / theta.powf((2 * i) as f32 / dim as f32))
            .collect();
        Self { inv_freq, dim }
    }

    /// Returns `(cos, sin)`, each `[1, seq_len, dim]`, for positions `0..seq_len`.
    pub fn forward(&self, seq_len: i32) -> Result<(Array, Array)> {
        let half = self.inv_freq.len();
        // freqs[s, j] = s * inv_freq[j]  → [seq, half]
        let mut freqs = Vec::with_capacity(seq_len as usize * half);
        for s in 0..seq_len {
            for &f in &self.inv_freq {
                freqs.push(s as f32 * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[seq_len, half as i32]);
        // emb = concat([freqs, freqs], -1) → [seq, dim]
        let emb = mlx_rs::ops::concatenate_axis(&[&freqs, &freqs], 1)?;
        let cos = mlx_rs::ops::cos(&emb)?.reshape(&[1, seq_len, self.dim])?;
        let sin = mlx_rs::ops::sin(&emb)?.reshape(&[1, seq_len, self.dim])?;
        Ok((cos, sin))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, array_eq, max, subtract};
    use mlx_rs::Dtype;

    #[test]
    fn linear_is_fused_addmm() {
        // sc-2779: the shared biased-linear helper must be a FUSED addmm. In bf16, addmm differs
        // from matmul+add (single vs double rounding), so prove fusion on bf16 inputs.
        let w = Array::from_slice(
            &(0..64 * 64)
                .map(|i| (i as f32 * 0.013).sin() * 0.05)
                .collect::<Vec<_>>(),
            &[64, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let b = Array::from_slice(
            &(0..64)
                .map(|i| (i as f32 * 0.7).cos() * 0.1)
                .collect::<Vec<_>>(),
            &[64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        let x = Array::from_slice(
            &(0..4 * 64)
                .map(|i| (i as f32 * 0.031).sin() * 0.5)
                .collect::<Vec<_>>(),
            &[4, 64],
        )
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

        let got = linear(&x, &w, &b).unwrap();
        let want = addmm(&b, &x, w.t(), 1.0, 1.0).unwrap();
        assert!(array_eq(&got, &want, false).unwrap().item::<bool>());
    }

    #[test]
    fn gelu_tanh_preserves_dtype() {
        // sc-2779: a bf16 input must stay bf16 (MLX weak-casts the constants to the input dtype),
        // and an f32 input must stay f32.
        let x32 = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3]);
        assert_eq!(gelu_tanh(&x32).unwrap().dtype(), Dtype::Float32);
        let xbf = x32.as_dtype(Dtype::Bfloat16).unwrap();
        assert_eq!(gelu_tanh(&xbf).unwrap().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn gelu_tanh_f32_matches_closed_form() {
        // f32 path matches the closed form computed with the f64-host √(2/π) constant, bit-exact.
        let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3]);
        let c = scalar((2.0_f64 / std::f64::consts::PI).sqrt() as f32);
        let x3 = power(&x, Array::from_int(3)).unwrap();
        let inner = multiply(
            add(&x, multiply(&x3, scalar(0.044_715)).unwrap()).unwrap(),
            &c,
        )
        .unwrap();
        let gate = add(tanh(&inner).unwrap(), scalar(1.0)).unwrap();
        let want = multiply(multiply(&x, scalar(0.5)).unwrap(), &gate).unwrap();
        assert!(array_eq(gelu_tanh(&x).unwrap(), &want, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn gelu_tanh_bf16_is_close_to_f32_truth() {
        // Preserving bf16 must not mean *wrong*: the bf16 result tracks the f32 reference within
        // bf16 rounding.
        let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 1.0, 3.0], &[2, 3]);
        let truth = gelu_tanh(&x).unwrap();
        let bf = gelu_tanh(&x.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let d = max(abs(subtract(&bf, &truth).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert!(d < 5e-2, "bf16 gelu_tanh diverged from f32 truth: {d}");
    }

    #[test]
    fn conv3d_1x1x1_sums_input_channels_with_bias() {
        // NDHWC: a single voxel with 2 input channels [1, 2].
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 1, 1, 1, 2]);
        // weight [out=1, kD=1, kH=1, kW=1, in=2] = ones -> sums over the input channels.
        let w = Array::from_slice(&[1.0f32, 1.0], &[1, 1, 1, 1, 2]);
        let bias = Array::from_slice(&[10.0f32], &[1]);
        let y = conv3d(&x, &w, Some(&bias), (1, 1, 1), (0, 0, 0)).unwrap();
        assert_eq!(y.shape(), &[1, 1, 1, 1, 1]);
        assert_eq!(y.item::<f32>(), 13.0); // 1 + 2 + bias 10
    }
}
