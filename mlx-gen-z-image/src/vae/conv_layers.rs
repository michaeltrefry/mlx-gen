//! Thin VAE decoder conv layers: `ConvIn`/`ConvOut` (3×3, pad 1) and `ConvNormOut`
//! (pytorch-compatible GroupNorm). NCHW I/O.

use mlx_rs::Array;

use mlx_gen::nn::{conv2d, group_norm};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

/// A 3×3 stride-1 pad-1 conv (used for both `conv_in` and `conv_out`).
pub struct ConvLayer {
    w: Array,
    b: Array,
}

impl ConvLayer {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.conv.weight"))?.clone(),
            b: w.require(&format!("{prefix}.conv.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let h = conv2d(&x, &self.w, Some(&self.b), 1, 1)?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }
}

/// Final GroupNorm before the output conv. NCHW I/O.
pub struct ConvNormOut {
    weight: Array,
    bias: Array,
}

impl ConvNormOut {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            weight: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            bias: w.require(&format!("{prefix}.norm.bias"))?.clone(),
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let h = group_norm(&x, &self.weight, &self.bias, GN_GROUPS, GN_EPS)?;
        Ok(h.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }
}
