//! SVD VAE — `AutoencoderKLTemporalDecoder` (sc-3372): a standard 2-D SD VAE **encoder** plus a
//! **spatio-temporal decoder**. Port of diffusers
//! `models/autoencoders/autoencoder_kl_temporal_decoder.py` (+ the `SpatioTemporalResBlock` /
//! `TemporalResnetBlock` / `AlphaBlender` building blocks from `models/resnet.py` and the
//! `Mid/UpBlockTemporalDecoder` from `models/unets/unet_3d_blocks.py`).
//!
//! Runs entirely NHWC for the spatial parts (`[B·F, H, W, C]`) and NDHWC (`[B, F, H, W, C]`, frame
//! axis = the temporal conv axis) for the temporal parts. Net-new rather than reusing the
//! `mlx-gen-sdxl` 2-D VAE because diffusers normalizes the SVD VAE at **eps 1e-6** (spatial /
//! encoder / `conv_norm_out`) and **1e-5** (temporal), whereas the SDXL port hardcodes 1e-5 — a gap
//! that compounds across the decoder and would miss an f32 parity gate. Core `nn` conv/group-norm/
//! silu/upsample primitives are reused throughout.
//!
//! Validated vs diffusers `encode().latent_dist.mode()` + chunked `decode(z, num_frames)` in f32
//! (`tools/dump_svd_vae_golden.py` / `tests/vae_parity.rs`).

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, multiply, pad, sigmoid, subtract};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, conv3d, group_norm, linear, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::VaeConfig;

const GN_GROUPS: i32 = 32;
/// Spatial / encoder / `conv_norm_out` GroupNorm epsilon (diffusers `resnet_eps` default).
const EPS_SPATIAL: f32 = 1e-6;
/// Temporal `TemporalResnetBlock` GroupNorm epsilon (the `SpatioTemporalResBlock` `temporal_eps`).
const EPS_TEMPORAL: f32 = 1e-5;

/// Transpose a stored NCHW conv2d weight `[out, in, kH, kW]` → mlx NHWC `[out, kH, kW, in]`.
fn conv2d_weight(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// Transpose a stored torch Conv3d weight `[out, in, kD, kH, kW]` → mlx NDHWC `[out, kD, kH, kW, in]`.
fn conv3d_weight(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 4, 1])?)
}

/// Load a conv2d as `(NHWC weight, bias)`.
fn load_conv2d(w: &Weights, name: &str) -> Result<(Array, Array)> {
    Ok((
        conv2d_weight(w.require(&format!("{name}.weight"))?)?,
        w.require(&format!("{name}.bias"))?.clone(),
    ))
}

/// Load a conv3d as `(NDHWC weight, bias)`.
fn load_conv3d(w: &Weights, name: &str) -> Result<(Array, Array)> {
    Ok((
        conv3d_weight(w.require(&format!("{name}.weight"))?)?,
        w.require(&format!("{name}.bias"))?.clone(),
    ))
}

/// Spatial `ResnetBlock2D` (temb-free in the VAE): GroupNorm→SiLU→Conv3×3 ×2 + a 1×1-conv residual
/// shortcut when channels change. NHWC `[B·F, H, W, C]`.
struct SpatialResnet {
    norm1_w: Array,
    norm1_b: Array,
    conv1: (Array, Array),
    norm2_w: Array,
    norm2_b: Array,
    conv2: (Array, Array),
    shortcut: Option<(Array, Array)>,
}

impl SpatialResnet {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        let shortcut = match w.get(&format!("{prefix}.conv_shortcut.weight")) {
            Some(_) => Some(load_conv2d(w, &format!("{prefix}.conv_shortcut"))?),
            None => None,
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: load_conv2d(w, &format!("{prefix}.conv1"))?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: load_conv2d(w, &format!("{prefix}.conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, EPS_SPATIAL)?;
        let y = conv2d(&silu(&y)?, &self.conv1.0, Some(&self.conv1.1), 1, 1)?;
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, EPS_SPATIAL)?;
        let y = conv2d(&silu(&y)?, &self.conv2.0, Some(&self.conv2.1), 1, 1)?;
        let residual = match &self.shortcut {
            Some((cw, cb)) => conv2d(x, cw, Some(cb), 1, 0)?,
            None => x.clone(),
        };
        Ok(add(&residual, &y)?)
    }
}

/// `TemporalResnetBlock` (temb-free in the VAE): GroupNorm→SiLU→Conv3d`(3,1,1)` ×2 over the frame
/// axis + an optional 1×1×1 shortcut. NDHWC `[B, F, H, W, C]`.
struct TemporalResnet {
    norm1_w: Array,
    norm1_b: Array,
    conv1: (Array, Array),
    norm2_w: Array,
    norm2_b: Array,
    conv2: (Array, Array),
    shortcut: Option<(Array, Array)>,
}

impl TemporalResnet {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        let shortcut = match w.get(&format!("{prefix}.conv_shortcut.weight")) {
            Some(_) => Some(load_conv3d(w, &format!("{prefix}.conv_shortcut"))?),
            None => None,
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            conv1: load_conv3d(w, &format!("{prefix}.conv1"))?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            conv2: load_conv3d(w, &format!("{prefix}.conv2"))?,
            shortcut,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = group_norm(x, &self.norm1_w, &self.norm1_b, GN_GROUPS, EPS_TEMPORAL)?;
        let y = conv3d(
            &silu(&y)?,
            &self.conv1.0,
            Some(&self.conv1.1),
            (1, 1, 1),
            (1, 0, 0),
        )?;
        let y = group_norm(&y, &self.norm2_w, &self.norm2_b, GN_GROUPS, EPS_TEMPORAL)?;
        let y = conv3d(
            &silu(&y)?,
            &self.conv2.0,
            Some(&self.conv2.1),
            (1, 1, 1),
            (1, 0, 0),
        )?;
        let residual = match &self.shortcut {
            Some((cw, cb)) => conv3d(x, cw, Some(cb), (1, 1, 1), (0, 0, 0))?,
            None => x.clone(),
        };
        Ok(add(&residual, &y)?)
    }
}

/// `SpatioTemporalResBlock` (VAE flavor): spatial pass on `[B·F, H, W, C]`, then the temporal pass
/// on `[B, F, H, W, C]`, blended by `AlphaBlender`. The VAE uses `merge_strategy="learned"` +
/// `switch_spatial_to_temporal_mix=True`, so (since `image_only_indicator` is unused for "learned")
/// `out = (1−σ(mix))·x_spatial + σ(mix)·x_temporal`.
struct SpatioTemporalResBlock {
    spatial: SpatialResnet,
    temporal: TemporalResnet,
    mix_factor: Array,
}

impl SpatioTemporalResBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            spatial: SpatialResnet::from_weights(w, &format!("{prefix}.spatial_res_block"))?,
            temporal: TemporalResnet::from_weights(w, &format!("{prefix}.temporal_res_block"))?,
            mix_factor: w
                .require(&format!("{prefix}.time_mixer.mix_factor"))?
                .clone(),
        })
    }

    fn forward(&self, x: &Array, num_frames: i32) -> Result<Array> {
        let spatial = self.spatial.forward(x)?; // [B·F, H, W, C_out] (the resnet may change C)
        let sh = spatial.shape();
        let (bf, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let b = bf / num_frames;
        let spatial5 = spatial.reshape(&[b, num_frames, h, w_, c])?;
        let temporal = self.temporal.forward(&spatial5)?; // [B, F, H, W, C]

        // AlphaBlender: alpha = σ(mix); switched → 1−alpha; out = (1−alpha)·spatial + alpha·temporal.
        let alpha = sigmoid(&self.mix_factor)?; // [1]
        let one_minus = subtract(scalar(1.0), &alpha)?;
        let blended = add(
            &multiply(&spatial5, &one_minus)?,
            &multiply(&temporal, &alpha)?,
        )?;
        Ok(blended.reshape(&[bf, h, w_, c])?)
    }
}

/// Single-head spatial self-attention (diffusers `Attention`, `residual_connection=True`,
/// `norm_num_groups=32`, eps 1e-6) — the VAE mid-block attention. NHWC, per-frame.
struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: (Array, Array),
    k: (Array, Array),
    v: (Array, Array),
    out: (Array, Array),
}

impl VaeAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let lin = |n: &str| -> Result<(Array, Array)> {
            Ok((
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                w.require(&format!("{prefix}.{n}.bias"))?.clone(),
            ))
        };
        Ok(Self {
            gn_w: w.require(&format!("{prefix}.group_norm.weight"))?.clone(),
            gn_b: w.require(&format!("{prefix}.group_norm.bias"))?.clone(),
            q: lin("to_q")?,
            k: lin("to_k")?,
            v: lin("to_v")?,
            out: lin("to_out.0")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.gn_w, &self.gn_b, GN_GROUPS, EPS_SPATIAL)?;
        let to_seq = |w_lin: &(Array, Array)| -> Result<Array> {
            Ok(linear(&y, &w_lin.0, &w_lin.1)?.reshape(&[b, 1, h * w_, c])?)
        };
        let q = to_seq(&self.q)?;
        let k = to_seq(&self.k)?;
        let v = to_seq(&self.v)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = linear(&o.reshape(&[b, h, w_, c])?, &self.out.0, &self.out.1)?;
        Ok(add(x, &o)?)
    }
}

/// Encoder down-block: a run of spatial resnets, then an optional stride-2 downsample.
struct EncDownBlock {
    resnets: Vec<SpatialResnet>,
    downsample: Option<(Array, Array)>,
}

impl EncDownBlock {
    fn from_weights(w: &Weights, prefix: &str, num_resnets: usize, add_down: bool) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| SpatialResnet::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            resnets,
            downsample: if add_down {
                Some(load_conv2d(w, &format!("{prefix}.downsamplers.0.conv"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            // Asymmetric (right/bottom) pad, then stride-2 / pad-0 conv (the SD downsample).
            x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            x = conv2d(&x, cw, Some(cb), 2, 0)?;
        }
        Ok(x)
    }
}

/// The standard 2-D SD VAE encoder (image → latent moments).
struct Encoder {
    conv_in: (Array, Array),
    down_blocks: Vec<EncDownBlock>,
    mid_res0: SpatialResnet,
    mid_attn: VaeAttention,
    mid_res1: SpatialResnet,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out: (Array, Array),
}

impl Encoder {
    fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        let down_blocks = (0..n)
            .map(|i| {
                EncDownBlock::from_weights(
                    w,
                    &format!("encoder.down_blocks.{i}"),
                    cfg.layers_per_block,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: load_conv2d(w, "encoder.conv_in")?,
            down_blocks,
            mid_res0: SpatialResnet::from_weights(w, "encoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "encoder.mid_block.attentions.0")?,
            mid_res1: SpatialResnet::from_weights(w, "encoder.mid_block.resnets.1")?,
            norm_out_w: w.require("encoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("encoder.conv_norm_out.bias")?.clone(),
            conv_out: load_conv2d(w, "encoder.conv_out")?,
        })
    }

    /// `x`: NHWC `[B, H, W, 3]` → moments `[B, H/8, W/8, 8]`.
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = conv2d(x, &self.conv_in.0, Some(&self.conv_in.1), 1, 1)?;
        for db in &self.down_blocks {
            x = db.forward(&x)?;
        }
        x = self.mid_res0.forward(&x)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_res1.forward(&x)?;
        let x = group_norm(
            &x,
            &self.norm_out_w,
            &self.norm_out_b,
            GN_GROUPS,
            EPS_SPATIAL,
        )?;
        conv2d(&silu(&x)?, &self.conv_out.0, Some(&self.conv_out.1), 1, 1)
    }
}

/// Decoder mid block (`MidBlockTemporalDecoder`, `num_layers=2`): res0 → spatial attn → res1.
struct DecMidBlock {
    res0: SpatioTemporalResBlock,
    attn: VaeAttention,
    res1: SpatioTemporalResBlock,
}

impl DecMidBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            res0: SpatioTemporalResBlock::from_weights(w, &format!("{prefix}.resnets.0"))?,
            attn: VaeAttention::from_weights(w, &format!("{prefix}.attentions.0"))?,
            res1: SpatioTemporalResBlock::from_weights(w, &format!("{prefix}.resnets.1"))?,
        })
    }

    fn forward(&self, x: &Array, num_frames: i32) -> Result<Array> {
        let x = self.res0.forward(x, num_frames)?;
        let x = self.attn.forward(&x)?;
        self.res1.forward(&x, num_frames)
    }
}

/// Decoder up-block (`UpBlockTemporalDecoder`): a run of spatio-temporal resnets, then an optional
/// nearest-2× + conv upsample.
struct DecUpBlock {
    resnets: Vec<SpatioTemporalResBlock>,
    upsample: Option<(Array, Array)>,
}

impl DecUpBlock {
    fn from_weights(w: &Weights, prefix: &str, num_resnets: usize, add_up: bool) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| SpatioTemporalResBlock::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            resnets,
            upsample: if add_up {
                Some(load_conv2d(w, &format!("{prefix}.upsamplers.0.conv"))?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Array, num_frames: i32) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, num_frames)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, cw, Some(cb), 1, 1)?;
        }
        Ok(x)
    }
}

/// The temporal decoder (latent → frames).
struct TemporalDecoder {
    conv_in: (Array, Array),
    mid: DecMidBlock,
    up_blocks: Vec<DecUpBlock>,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out: (Array, Array),
    time_conv_out: (Array, Array),
}

impl TemporalDecoder {
    fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        let num_resnets = cfg.layers_per_block + 1;
        let up_blocks = (0..n)
            .map(|i| {
                DecUpBlock::from_weights(
                    w,
                    &format!("decoder.up_blocks.{i}"),
                    num_resnets,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in: load_conv2d(w, "decoder.conv_in")?,
            mid: DecMidBlock::from_weights(w, "decoder.mid_block")?,
            up_blocks,
            norm_out_w: w.require("decoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("decoder.conv_norm_out.bias")?.clone(),
            conv_out: load_conv2d(w, "decoder.conv_out")?,
            time_conv_out: load_conv3d(w, "decoder.time_conv_out")?,
        })
    }

    /// `z`: NHWC `[B·F, H/8, W/8, 4]` → frames NHWC `[B·F, H, W, 3]`.
    fn forward(&self, z: &Array, num_frames: i32) -> Result<Array> {
        let mut x = conv2d(z, &self.conv_in.0, Some(&self.conv_in.1), 1, 1)?;
        x = self.mid.forward(&x, num_frames)?;
        for ub in &self.up_blocks {
            x = ub.forward(&x, num_frames)?;
        }
        let x = group_norm(
            &x,
            &self.norm_out_w,
            &self.norm_out_b,
            GN_GROUPS,
            EPS_SPATIAL,
        )?;
        let x = conv2d(&silu(&x)?, &self.conv_out.0, Some(&self.conv_out.1), 1, 1)?;

        // `time_conv_out` over the frame axis: reshape NHWC → NDHWC, Conv3d`(3,1,1)`, reshape back.
        let sh = x.shape();
        let (bf, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let b = bf / num_frames;
        let x5 = x.reshape(&[b, num_frames, h, w_, c])?;
        let x5 = conv3d(
            &x5,
            &self.time_conv_out.0,
            Some(&self.time_conv_out.1),
            (1, 1, 1),
            (1, 0, 0),
        )?;
        Ok(x5.reshape(&[bf, h, w_, c])?)
    }
}

/// The SVD `AutoencoderKLTemporalDecoder`. `encode_mode` produces the latent mean (the
/// `latent_dist.mode()` the SVD pipeline conditions on); `decode` reconstructs frames from a latent
/// (the caller divides by `scaling_factor` first, matching the diffusers pipeline — `decode` itself
/// mirrors `vae.decode`, which feeds the latent straight in: this VAE has no `post_quant_conv`).
pub struct SvdVae {
    encoder: Encoder,
    /// `quant_conv` `[8, 8, 1, 1]` → an `[8, 8]` Linear over the moment channels.
    quant: (Array, Array),
    decoder: TemporalDecoder,
    scaling_factor: f32,
}

impl SvdVae {
    pub fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let qw = w.require("quant_conv.weight")?;
        let qsh = qw.shape();
        Ok(Self {
            encoder: Encoder::from_weights(w, cfg)?,
            quant: (
                qw.reshape(&[qsh[0], qsh[1]])?,
                w.require("quant_conv.bias")?.clone(),
            ),
            decoder: TemporalDecoder::from_weights(w, cfg)?,
            scaling_factor: cfg.scaling_factor,
        })
    }

    /// The trained latent scale (`z = z · scaling_factor` for the diffusion model; the pipeline
    /// divides by it before [`decode`](Self::decode)).
    pub fn scaling_factor(&self) -> f32 {
        self.scaling_factor
    }

    /// Encode `[B, H, W, 3]` (NHWC, in roughly `[-1, 1]`) → latent **mean** `[B, H/8, W/8, 4]` (raw,
    /// **unscaled** — `latent_dist.mode()`). Multiply by [`scaling_factor`](Self::scaling_factor) for
    /// the diffusion-space conditioning latent.
    pub fn encode_mode(&self, image: &Array) -> Result<Array> {
        let moments = linear(&self.encoder.forward(image)?, &self.quant.0, &self.quant.1)?;
        // `DiagonalGaussian.mode()` = the mean = the first half of the channel axis.
        let half = moments.shape()[3] / 2;
        let idx = Array::from_slice(&(0..half).collect::<Vec<i32>>(), &[half]);
        Ok(moments.take_axis(&idx, 3)?)
    }

    /// Decode a latent `[B·F, H/8, W/8, 4]` (NHWC, already divided by `scaling_factor`) → frames
    /// `[B·F, H, W, 3]`. Mirrors diffusers `vae.decode(z, num_frames)`.
    pub fn decode(&self, z: &Array, num_frames: i32) -> Result<Array> {
        self.decoder.forward(z, num_frames)
    }
}
