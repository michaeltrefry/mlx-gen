//! VAE ResnetBlock2D: GroupNorm→SiLU→Conv3×3 ×2 with a residual (1×1 conv shortcut when the
//! channel count changes). NCHW I/O.

use mlx_rs::ops::add;
use mlx_rs::Array;

use super::{conv2d, group_norm};
use crate::models::z_image::timestep_embedder::silu;
use crate::weights::Weights;
use crate::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

pub struct ResnetBlock2D {
    norm1_w: Array,
    norm1_b: Array,
    conv1_w: Array,
    conv1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    conv2_w: Array,
    conv2_b: Array,
    shortcut: Option<(Array, Array)>,
}

impl ResnetBlock2D {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |s: &str| w.require(&format!("{prefix}.{s}")).cloned();
        let shortcut = match (
            w.get(&format!("{prefix}.conv_shortcut.weight")),
            w.get(&format!("{prefix}.conv_shortcut.bias")),
        ) {
            (Some(sw), Some(sb)) => Some((sw.clone(), sb.clone())),
            _ => None,
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1_w: g("conv1.weight")?,
            conv1_b: g("conv1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2_w: g("conv2.weight")?,
            conv2_b: g("conv2.bias")?,
            shortcut,
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC

        let h = group_norm(&x, &self.norm1_w, &self.norm1_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv1_w, Some(&self.conv1_b), 1, 1)?;
        let h = group_norm(&h, &self.norm2_w, &self.norm2_b, GN_GROUPS, GN_EPS)?;
        let h = conv2d(&silu(&h)?, &self.conv2_w, Some(&self.conv2_b), 1, 1)?;

        let residual = match &self.shortcut {
            Some((sw, sb)) => conv2d(&x, sw, Some(sb), 1, 0)?, // 1x1
            None => x,
        };
        Ok(add(&residual, &h)?.transpose_axes(&[0, 3, 1, 2])?) // NCHW
    }
}
