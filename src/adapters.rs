//! Adapter framework — LoRA + LoKr applied as forward-time residuals over a shared
//! base. Quantized-safe: the base is never fused/mutated. Ported from the sc-2338
//! spike; mirrors the Python mflux fork's `LoKrLinear` / `FusedLoRALinear` (sc-2216).
//!
//! The base is a real `nn::Linear` *or* `nn::QuantizedLinear` (sc-2342), so quantization
//! and adapters compose: `base(x) + Σ adapter.residual(x)`. Forward is taken by `&self`
//! (we call the underlying ops directly rather than the `&mut self` `Module` trait), so a
//! whole model tree can be evaluated through shared references.
//!
//! Adapters are installed by dotted path via [`AdaptableHost`] / [`install_adapter`] — the
//! Rust stand-in for Python's dynamic `getattr`-swap, since mlx-rs flattens module params to
//! `Array` leaves and cannot replace a submodule in place.

use mlx_rs::{
    module::Param,
    nn::{Linear, QuantizedLinear},
    ops::{add, kron, matmul, multiply, quantized_matmul},
    Array, Dtype,
};

use crate::array::scalar;
use crate::Result;

pub mod loader;

/// Reconstruct a LoKr weight delta `ΔW = (alpha/rank) · kron(w1, w2)`, reshaped to the
/// base weight's logical `[out, in]` and stored at bf16. Each Kronecker factor is either
/// full (`w1` / `w2`) or a low-rank product (`w1_a @ w1_b` / `w2_a @ w2_b`). Mirrors
/// PEFT/LyCORIS `LoKrLayer.get_delta_weight` (pending the sc-2324 cross-impl parity check).
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_lokr_delta(
    alpha: f32,
    rank: f32,
    base_shape: &[i32],
    w1: Option<&Array>,
    w1_a: Option<&Array>,
    w1_b: Option<&Array>,
    w2: Option<&Array>,
    w2_a: Option<&Array>,
    w2_b: Option<&Array>,
) -> Result<Array> {
    let factor1 = match (w1, w1_a, w1_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w1 missing (need full w1 or w1_a@w1_b)".into()),
    };
    let factor2 = match (w2, w2_a, w2_b) {
        (Some(w), _, _) => w.clone(),
        (_, Some(a), Some(b)) => matmul(a, b)?,
        _ => return Err("LoKr: w2 missing (need full w2 or w2_a@w2_b)".into()),
    };
    let delta = multiply(&kron(&factor1, &factor2)?, scalar(alpha / rank))?;
    // PARITY-BF16 (sc-2609): the fork reconstructs the LoKr delta at bf16; f32 would be more precise.
    // (`residual` already reconciles this delta to the activation dtype, so flipping to f32 is local.)
    Ok(delta.reshape(base_shape)?.as_dtype(Dtype::Bfloat16)?)
}

/// One adapter's contribution WITHOUT the base, so a host can sum stacked adapters over
/// a single base application.
pub enum Adapter {
    /// LoRA: `residual = scale · x·A·B`.
    Lora { a: Array, b: Array, scale: f32 },
    /// LoKr: `residual = scale · x·ΔWᵀ`; `delta` stored bf16 (see [`reconstruct_lokr_delta`]).
    Lokr { delta: Array, scale: f32 },
}

impl Adapter {
    pub fn residual(&self, x: &Array) -> Result<Array> {
        // Adapter math runs in f32. LoRA's low-rank second matmul is `[seq,r]·[r,out]` with
        // `K = rank ≤ 512` — exactly the dense 16-bit×16-bit Metal GEMM the NAX build mis-runs
        // (M≥2 & K≤512; see `mlx-gen-qwen-image/tests/bf16_matmul_sweep.rs`). On the bf16 txt2img
        // path that GEMM would get bf16 activations and return garbage. f32 sidesteps the bug, is
        // strictly more accurate, and still matches the fork's (bug-free wheel) bf16 residual
        // within tolerance once cast back. The result is returned in the activation dtype so
        // `base(x) + residual` stays in the base dtype (PARITY-BF16 on the bf16 path; f32 base → f32).
        let xf = x.as_dtype(Dtype::Float32)?;
        let r = match self {
            Adapter::Lora { a, b, scale } => {
                let a = a.as_dtype(Dtype::Float32)?;
                let b = b.as_dtype(Dtype::Float32)?;
                multiply(&matmul(&matmul(&xf, &a)?, &b)?, scalar(*scale))?
            }
            Adapter::Lokr { delta, scale } => {
                let d = delta.as_dtype(Dtype::Float32)?;
                multiply(&matmul(&xf, d.t())?, scalar(*scale))?
            }
        };
        Ok(r.as_dtype(x.dtype())?)
    }
}

/// A linear base — dense or quantized — evaluated through a shared reference. Mirrors the
/// `forward` of mlx-rs's `nn::Linear` / `nn::QuantizedLinear` but without requiring `&mut`.
pub enum LinearBase {
    Dense(Linear),
    Quantized(QuantizedLinear),
}

impl LinearBase {
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            LinearBase::Dense(l) => {
                let mut y = matmul(x, l.weight.value.t())?;
                if let Some(b) = l.bias.value.as_ref() {
                    y = add(&y, b)?;
                }
                y
            }
            LinearBase::Quantized(q) => {
                // Belt-and-suspenders upcast for the NAX 16-bit-GEMM bug (present on both pinned
                // MLX builds, 0.30.6 and 0.31.1; see `tests/bf16_matmul_sweep.rs`). The buggy op is
                // the *dense* 16-bit×16-bit Metal GEMM (M≥2 & K≤512, or very large M) — NOT
                // `quantized_matmul`, which accumulates in fp32 (mlx#963) and is correct at every
                // shape/dtype. So this bf16→f32 upcast guards a non-bug here and is not strictly
                // load-bearing; it is kept as a cheap, uniform "16-bit activations never reach a
                // 16-bit/quant Linear" invariant (and to keep `q8_smoke.rs` green) while the
                // underlying GEMM bug persists. Weights stay Q4/Q8 throughout.
                let xf = if x.dtype() == Dtype::Bfloat16 {
                    x.as_dtype(Dtype::Float32)?
                } else {
                    x.clone()
                };
                let mut y = quantized_matmul(
                    &xf,
                    &q.inner.weight.value,
                    &q.scales.value,
                    &q.biases.value,
                    true,
                    q.group_size,
                    q.bits,
                )?;
                if let Some(b) = q.inner.bias.value.as_ref() {
                    y = add(&y, b)?;
                }
                y
            }
        })
    }
}

/// A linear base plus a stack of adapters, applied as `base(x) + Σ adapter.residual(x)`.
/// Quantized-safe: the base weight is never mutated.
pub struct AdaptableLinear {
    base: LinearBase,
    adapters: Vec<Adapter>,
}

impl AdaptableLinear {
    /// Build from a raw `[out, in]` weight (and optional bias) — the common path when
    /// loading dense (bf16/fp16/fp32) checkpoints via the `weights` module.
    pub fn dense(weight: Array, bias: Option<Array>) -> Self {
        Self::from_linear(Linear {
            weight: Param::new(weight),
            bias: Param::new(bias),
        })
    }

    /// Wrap an existing dense `nn::Linear`.
    pub fn from_linear(linear: Linear) -> Self {
        Self {
            base: LinearBase::Dense(linear),
            adapters: Vec::new(),
        }
    }

    /// Wrap an existing `nn::QuantizedLinear` (sc-2342 quantized weights).
    pub fn from_quantized(q: QuantizedLinear) -> Self {
        Self {
            base: LinearBase::Quantized(q),
            adapters: Vec::new(),
        }
    }

    /// Stack a new adapter (LoRA or LoKr) on top of any already installed.
    pub fn push(&mut self, adapter: Adapter) {
        self.adapters.push(adapter);
    }

    pub fn adapters(&self) -> &[Adapter] {
        &self.adapters
    }

    /// `true` once the base has been quantized (Q4/Q8).
    pub fn is_quantized(&self) -> bool {
        matches!(self.base, LinearBase::Quantized(_))
    }

    /// Diagnostic accessor: the quantized base's `(packed_weight, scales, biases, bias, group_size,
    /// bits)`, or `None` if the base is still dense. Used by the sc-2604 Q8 root-cause diagnostic to
    /// byte-compare the *loaded* model's quantization against the fork's `mx.quantize` (the
    /// `qmm_smallk` probe only exercised the free `quantize` op, not `try_from_linear`).
    /// Diagnostic accessor: the dense base's `(weight, bias)`, or `None` if already quantized.
    /// Used by the sc-2604 diagnostic to inspect the loaded weight dtype before quantization.
    pub fn dense_weight(&self) -> Option<(&Array, Option<&Array>)> {
        match &self.base {
            LinearBase::Dense(l) => Some((&l.weight.value, l.bias.value.as_ref())),
            LinearBase::Quantized(_) => None,
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn quantized_params(&self) -> Option<(&Array, &Array, &Array, Option<&Array>, i32, i32)> {
        match &self.base {
            LinearBase::Quantized(q) => Some((
                &q.inner.weight.value,
                &q.scales.value,
                &q.biases.value,
                q.inner.bias.value.as_ref(),
                q.group_size,
                q.bits,
            )),
            LinearBase::Dense(_) => None,
        }
    }

    /// The base weight's logical `[out, in]` shape — what a LoKr delta must reshape to.
    /// For a quantized base the packed weight is opaque, so recover it from the scales grid
    /// (`[out, in/group_size]`) times the group size.
    pub fn base_shape(&self) -> Vec<i32> {
        match &self.base {
            LinearBase::Dense(l) => l.weight.value.shape().to_vec(),
            LinearBase::Quantized(q) => {
                let s = q.scales.value.shape();
                vec![s[0], s[1] * q.group_size]
            }
        }
    }

    /// Quantize the dense base in place to Q4/Q8 (`group_size` defaults to 64), the mlx-rs
    /// equivalent of `nn.quantize` over this Linear. No-op if already quantized. Adapters are
    /// forward-time residuals over the (now quantized) base, so they are unaffected — this is
    /// why the base is never fused: fusing would force re-quantization on every adapter swap.
    pub fn quantize(&mut self, bits: i32, group_size: Option<i32>) -> Result<()> {
        if let LinearBase::Dense(l) = &self.base {
            // PARITY-BF16 (sc-2609): downcast for fork parity. f32 quantization (f32 group scales)
            // is *more* accurate; we cast to bf16 only to byte-match the fork's golden. Flip to f32
            // for quality once parity is no longer the goal — f32 is safe (the qmm path never hits
            // the bf16-GEMM bug). Rationale below.
            //
            // The fork (mflux) loads every weight at bf16 — its compute dtype — and quantizes THAT.
            // Some checkpoints (e.g. Z-Image-Turbo's transformer) ship f32 on disk; quantizing the
            // as-loaded f32 weight yields group `scales` that differ from the fork's bf16 scales by
            // ~0.13% (the integer `wq` codes and `biases` survive the perturbation, the scales do
            // not), which compounds into the base-model Q8/Q4 e2e residual (sc-2604). Cast weight +
            // bias to bf16 first so the packing is byte-identical to the fork. No-op when already
            // bf16 (e.g. Qwen, whose checkpoint is bf16-native — which is why its Q8 already matched).
            let weight = l.weight.value.as_dtype(Dtype::Bfloat16)?;
            let bias = l
                .bias
                .value
                .as_ref()
                .map(|b| b.as_dtype(Dtype::Bfloat16))
                .transpose()?;
            let linear = Linear {
                weight: Param::new(weight),
                bias: Param::new(bias),
            };
            let q = QuantizedLinear::try_from_linear(
                linear,
                group_size.unwrap_or(crate::quant::DEFAULT_GROUP_SIZE),
                bits,
            )?;
            self.base = LinearBase::Quantized(q);
        }
        Ok(())
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = self.base.forward(x)?;
        for adapter in &self.adapters {
            out = add(&out, &adapter.residual(x)?)?;
        }
        Ok(out)
    }
}

/// A module tree that can resolve a dotted parameter path (split into segments) to the
/// [`AdaptableLinear`] living there, so an adapter can be installed onto it. This is the
/// hand-written form of the macro the full adapter framework (sc-2343) will generate.
pub trait AdaptableHost {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear>;
}

/// Install an adapter onto the [`AdaptableLinear`] addressed by `dotted` (e.g.
/// `"attention.to_q"`). Errors if the path resolves to no adaptable linear.
pub fn install_adapter(
    host: &mut impl AdaptableHost,
    dotted: &str,
    adapter: Adapter,
) -> Result<()> {
    let parts: Vec<&str> = dotted.split('.').collect();
    host.adaptable_mut(&parts)
        .ok_or_else(|| format!("no adaptable linear at path: {dotted}"))?
        .push(adapter);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{all_close, array_eq};

    fn lokr_2x2() -> Array {
        reconstruct_lokr_delta(
            8.0,
            4.0,
            &[2, 2],
            Some(&Array::from_slice(&[0.5f32, 0.6], &[2, 1])),
            None,
            None,
            Some(&Array::from_slice(&[0.7f32, 0.8], &[1, 2])),
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn lokr_delta_stored_bf16() {
        assert_eq!(lokr_2x2().dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn scale_zero_lokr_is_bit_exact_noop() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        lin.push(Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.0,
        });
        let out = lin.forward(&x).unwrap();
        assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
    }

    #[test]
    fn residual_in_bf16_runs_f32_and_returns_activation_dtype() {
        // The LoRA second matmul `[seq,r]·[r,out]` (K=rank=4≤512, M=seq=4≥2) is the dense 16-bit
        // GEMM the NAX build mis-runs; `residual` must compute it in f32 and return the activation
        // dtype. So a bf16-input residual must (a) be bf16 and (b) match the f32 reference within
        // bf16 rounding — NOT diverge (which is what the buggy bf16 GEMM would produce).
        let a32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.1 - 0.4).collect::<Vec<_>>(),
            &[2, 4],
        );
        let b32 = Array::from_slice(
            &(0..8).map(|i| i as f32 * 0.05).collect::<Vec<_>>(),
            &[4, 2],
        );
        let x32 = Array::from_slice(&[1.0f32, -2.0, 0.5, 0.25, -1.0, 2.0], &[3, 2]);
        let lora = Adapter::Lora {
            a: a32.as_dtype(Dtype::Bfloat16).unwrap(),
            b: b32.as_dtype(Dtype::Bfloat16).unwrap(),
            scale: 0.5,
        };
        let got = lora
            .residual(&x32.as_dtype(Dtype::Bfloat16).unwrap())
            .unwrap();
        assert_eq!(
            got.dtype(),
            Dtype::Bfloat16,
            "residual returns the activation dtype"
        );

        // f32 reference, rounded to bf16 the way `residual` casts its result back.
        let want = multiply(
            matmul(matmul(&x32, &a32).unwrap(), &b32).unwrap(),
            scalar(0.5),
        )
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
        assert!(
            all_close(&got, &want, 5e-2, 5e-2, false)
                .unwrap()
                .item::<bool>(),
            "bf16 residual diverged from the f32 reference (bf16 GEMM bug?)"
        );
    }

    #[test]
    fn stacks_mixed_lora_and_lokr_summing_residuals() {
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]);
        let x = Array::from_slice(&[1.0f32, 2.0], &[1, 2]);
        let mut lin = AdaptableLinear::dense(w, None);
        let base = lin.forward(&x).unwrap();
        let lora = Adapter::Lora {
            a: Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[2, 2]),
            b: Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]),
            scale: 0.5,
        };
        let lokr = Adapter::Lokr {
            delta: lokr_2x2(),
            scale: 0.7,
        };
        let lora_r = lora.residual(&x).unwrap();
        let lokr_r = lokr.residual(&x).unwrap();
        lin.push(lora);
        lin.push(lokr);
        assert_eq!(lin.adapters().len(), 2);
        let expected = add(add(&base, &lora_r).unwrap(), &lokr_r).unwrap();
        assert!(
            all_close(lin.forward(&x).unwrap(), &expected, 1e-4, 1e-2, false)
                .unwrap()
                .item::<bool>()
        );
    }
}
