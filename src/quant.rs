//! Quantization — group-wise affine Q4/Q8, the mlx-rs equivalent of the Python mflux fork's
//! `nn.quantize(model, bits=bits)` path. mflux never passes `group_size`, so it uses MLX's
//! default of **64**. The actual quantization seam is
//! [`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize), which quantizes each
//! `Linear` base in place **with the bf16-parity cast** the fork goldens require — providers route
//! through it (or their per-family loaders), so this module owns only the shared default below.
//!
//! Byte-level packing parity vs the fork (mlx 0.31) is checked in `tests/quant_parity.rs` —
//! the version-drift risk, since the crate links an older bundled MLX (mlx-rs 0.25).

/// MLX's default quantization group size; mflux relies on it (never overrides). Used by the real
/// quantization seams ([`AdaptableLinear::quantize`](crate::adapters::AdaptableLinear::quantize) and
/// the Kolors ChatGLM3 quantizer).
pub const DEFAULT_GROUP_SIZE: i32 = 64;
