//! Quantization — group-wise affine Q4/Q8, the mlx-rs equivalent of the Python mflux fork's
//! `nn.quantize(model, bits=bits)` path. mflux never passes `group_size`, so it uses MLX's
//! default of **64**; `bits` is 4 or 8, resolved from the requested flag vs any pre-quantized
//! ("stored") level. The predicate quantizes every `Linear` (`hasattr(module, "to_quantized")`),
//! which here means: quantize each [`AdaptableLinear`](crate::adapters::AdaptableLinear) base.
//!
//! Byte-level packing parity vs the fork (mlx 0.31) is checked in `tests/quant_parity.rs` —
//! the version-drift risk, since the crate links an older bundled MLX (mlx-rs 0.25).

use mlx_rs::nn::{Linear, QuantizedLinear};

use crate::Result;

/// MLX's default quantization group size; mflux relies on it (never overrides).
pub const DEFAULT_GROUP_SIZE: i32 = 64;

/// Port of mflux's `QuantizationResolution.resolve`: reconcile a model's stored
/// (pre-quantized) bit-width with a requested one. Returns the effective bits — `None`
/// means run dense — plus an optional human-facing warning (stored wins on conflict).
pub fn resolve_bits(stored: Option<i32>, requested: Option<i32>) -> (Option<i32>, Option<String>) {
    match (stored, requested) {
        (None, None) => (None, None),       // none: dense
        (None, Some(r)) => (Some(r), None), // on_the_fly: quantize as requested
        (Some(s), None) => (Some(s), None), // pre_quantized: honor stored
        (Some(s), Some(r)) => {
            // conflict: stored wins; warn only if they disagree.
            let warn = (s != r)
                .then(|| format!("Model is pre-quantized at {s}-bit. Ignoring requested {r}-bit."));
            (Some(s), warn)
        }
    }
}

/// Quantize a dense `nn::Linear` to Q4/Q8 — the mlx-rs equivalent of `nn.quantize(linear)`.
/// `group_size` defaults to [`DEFAULT_GROUP_SIZE`] (matching mflux).
pub fn quantize_linear(
    linear: Linear,
    bits: i32,
    group_size: Option<i32>,
) -> Result<QuantizedLinear> {
    Ok(QuantizedLinear::try_from_linear(
        linear,
        group_size.unwrap_or(DEFAULT_GROUP_SIZE),
        bits,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_matches_mflux_rules() {
        assert_eq!(resolve_bits(None, None), (None, None)); // none
        assert_eq!(resolve_bits(None, Some(8)), (Some(8), None)); // on_the_fly
        assert_eq!(resolve_bits(Some(4), None), (Some(4), None)); // pre_quantized
                                                                  // conflict: stored wins, no warning when equal, warning when not.
        assert_eq!(resolve_bits(Some(8), Some(8)), (Some(8), None));
        let (bits, warn) = resolve_bits(Some(8), Some(4));
        assert_eq!(bits, Some(8));
        assert!(warn.unwrap().contains("8-bit"));
    }
}
