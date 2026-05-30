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

use crate::Result;

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

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
        Ok(match self {
            Adapter::Lora { a, b, scale } => multiply(&matmul(&matmul(x, a)?, b)?, scalar(*scale))?,
            Adapter::Lokr { delta, scale } => {
                // Reconcile the bf16 delta with the activation dtype (no-op when x is bf16).
                let d = delta.as_dtype(x.dtype())?;
                multiply(&matmul(x, d.t())?, scalar(*scale))?
            }
        })
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
                let mut y = quantized_matmul(
                    x,
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
