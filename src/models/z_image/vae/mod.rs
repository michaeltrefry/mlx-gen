//! Z-Image VAE decoder. The image side of the pipeline — latents → RGB. Built from the
//! convolutional primitives below (Conv2d + pytorch-compatible GroupNorm + nearest upsample),
//! which this module validates against the fork before the full `Decoder` assembly.
//!
//! Modules take/return NCHW (mirroring the fork's per-module transpose convention) and work
//! in NHWC internally, since mlx convs/norms are channels-last.

pub mod attention;
pub mod resnet_block;
pub mod up_sampler;

pub use attention::VaeAttention;
pub use resnet_block::ResnetBlock2D;
pub use up_sampler::UpSampler;

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, broadcast_to, conv2d as conv2d_op, multiply};
use mlx_rs::Array;

use crate::Result;

/// 2-D conv over NHWC `x` with an mlx `[out, kH, kW, in]` weight (+ optional bias).
pub(crate) fn conv2d(
    x: &Array,
    w: &Array,
    b: Option<&Array>,
    stride: i32,
    padding: i32,
) -> Result<Array> {
    let mut y = conv2d_op(x, w, (stride, stride), (padding, padding), (1, 1), 1)?;
    if let Some(b) = b {
        y = add(&y, b)?;
    }
    Ok(y)
}

/// PyTorch-compatible group normalization over NHWC `x` (`weight`/`bias` are per-channel).
/// Mirrors mlx-rs `GroupNorm::pytorch_group_norm` + affine: split channels into `num_groups`,
/// layer-norm each group, then scale/shift by `weight`/`bias`.
pub(crate) fn group_norm(
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

/// Nearest-neighbor upsample of NHWC `x` by `scale` (broadcast + reshape, matching the fork).
pub(crate) fn upsample_nearest(x: &Array, scale: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let x6 = x.reshape(&[b, h, 1, w, 1, c])?;
    let bc = broadcast_to(&x6, &[b, h, scale, w, scale, c])?;
    Ok(bc.reshape(&[b, h * scale, w * scale, c])?)
}
