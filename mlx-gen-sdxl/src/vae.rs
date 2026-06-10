//! SDXL VAE (autoencoder) — port of the vendored `_vendor/mlx_sd/vae.py`. The decoder powers T2I
//! (latents → image); the encoder powers img2img (sc-2638). Runs entirely in NHWC, reusing the
//! UNet [`ResnetBlock2D`] (temb-free here) and the core conv/group-norm primitives. SDXL's
//! `scaling_factor` is **0.13025** (not SD-2.1's 0.18215). Latents are scaled by it on encode and
//! divided by it on decode.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, multiply, pad};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, group_norm, silu, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::VaeConfig;
use crate::unet::ResnetBlock2D;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

/// Single-head spatial self-attention used in the VAE mid block (the vendored `vae.Attention`).
struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
}

impl VaeAttention {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let lin = |n: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.{n}.bias"))?.clone()),
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

    /// `x`: NHWC `[B, H, W, C]`. Single-head attention over the H·W positions, residual.
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.gn_w, &self.gn_b, GN_GROUPS, GN_EPS)?;
        let to_seq = |a: Array| -> Result<Array> { Ok(a.reshape(&[b, 1, h * w_, c])?) };
        let q = to_seq(self.q.forward(&y)?)?;
        let k = to_seq(self.k.forward(&y)?)?;
        let v = to_seq(self.v.forward(&y)?)?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = self.out.forward(&o.reshape(&[b, h, w_, c])?)?;
        Ok(add(x, &o)?)
    }
}

/// Encoder/decoder macro-block: a run of (temb-free) resnets, then an optional downsample
/// (asymmetric-pad + stride-2 conv) or upsample (nearest-2× + conv). Port of the vendored
/// `vae.EncoderDecoderBlock2D`.
struct EncoderDecoderBlock2D {
    resnets: Vec<ResnetBlock2D>,
    downsample: Option<(Array, Array)>,
    upsample: Option<(Array, Array)>,
}

impl EncoderDecoderBlock2D {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_resnets: i32,
        add_downsample: bool,
        add_upsample: bool,
    ) -> Result<Self> {
        let resnets = (0..num_resnets)
            .map(|j| ResnetBlock2D::from_weights(w, &format!("{prefix}.resnets.{j}")))
            .collect::<Result<Vec<_>>>()?;
        let conv = |which: &str| -> Result<(Array, Array)> {
            Ok((
                crate::unet::nchw_to_nhwc(w.require(&format!("{prefix}.{which}.0.conv.weight"))?)?,
                w.require(&format!("{prefix}.{which}.0.conv.bias"))?.clone(),
            ))
        };
        Ok(Self {
            resnets,
            downsample: if add_downsample {
                Some(conv("downsamplers")?)
            } else {
                None
            },
            upsample: if add_upsample {
                Some(conv("upsamplers")?)
            } else {
                None
            },
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for r in &self.resnets {
            x = r.forward(&x, None)?;
        }
        if let Some((cw, cb)) = &self.downsample {
            // Asymmetric (right/bottom) pad then stride-2, pad-0 conv (the SD downsample).
            x = pad(&x, &[(0, 0), (0, 1), (0, 1), (0, 0)][..], None, None)?;
            x = conv2d(&x, cw, Some(cb), 2, 0)?;
        }
        if let Some((cw, cb)) = &self.upsample {
            x = conv2d(&upsample_nearest(&x, 2)?, cw, Some(cb), 1, 1)?;
        }
        Ok(x)
    }
}

/// The decoder (latent → image).
struct Decoder {
    conv_in_w: Array,
    conv_in_b: Array,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<EncoderDecoderBlock2D>,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Decoder {
    fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        // decoder layers_per_block = config.layers_per_block + 1.
        let num_resnets = cfg.layers_per_block + 1;
        let up_blocks = (0..n)
            .map(|i| {
                EncoderDecoderBlock2D::from_weights(
                    w,
                    &format!("decoder.up_blocks.{i}"),
                    num_resnets,
                    false,
                    i < n - 1,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: crate::unet::nchw_to_nhwc(w.require("decoder.conv_in.weight")?)?,
            conv_in_b: w.require("decoder.conv_in.bias")?.clone(),
            mid_resnet0: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "decoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "decoder.mid_block.resnets.1")?,
            up_blocks,
            norm_out_w: w.require("decoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("decoder.conv_norm_out.bias")?.clone(),
            conv_out_w: crate::unet::nchw_to_nhwc(w.require("decoder.conv_out.weight")?)?,
            conv_out_b: w.require("decoder.conv_out.bias")?.clone(),
        })
    }

    fn forward(&self, z: &Array) -> Result<Array> {
        let mut x = conv2d(z, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        x = self.mid_resnet0.forward(&x, None)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_resnet1.forward(&x, None)?;
        for ub in &self.up_blocks {
            x = ub.forward(&x)?;
        }
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}

/// The encoder (image → latent moments).
struct Encoder {
    conv_in_w: Array,
    conv_in_b: Array,
    down_blocks: Vec<EncoderDecoderBlock2D>,
    mid_resnet0: ResnetBlock2D,
    mid_attn: VaeAttention,
    mid_resnet1: ResnetBlock2D,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl Encoder {
    fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.block_out_channels.len();
        let num_resnets = cfg.layers_per_block;
        let down_blocks = (0..n)
            .map(|i| {
                EncoderDecoderBlock2D::from_weights(
                    w,
                    &format!("encoder.down_blocks.{i}"),
                    num_resnets,
                    i < n - 1,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            conv_in_w: crate::unet::nchw_to_nhwc(w.require("encoder.conv_in.weight")?)?,
            conv_in_b: w.require("encoder.conv_in.bias")?.clone(),
            down_blocks,
            mid_resnet0: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.0")?,
            mid_attn: VaeAttention::from_weights(w, "encoder.mid_block.attentions.0")?,
            mid_resnet1: ResnetBlock2D::from_weights(w, "encoder.mid_block.resnets.1")?,
            norm_out_w: w.require("encoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("encoder.conv_norm_out.bias")?.clone(),
            conv_out_w: crate::unet::nchw_to_nhwc(w.require("encoder.conv_out.weight")?)?,
            conv_out_b: w.require("encoder.conv_out.bias")?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = conv2d(x, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;
        for db in &self.down_blocks {
            x = db.forward(&x)?;
        }
        x = self.mid_resnet0.forward(&x, None)?;
        x = self.mid_attn.forward(&x)?;
        x = self.mid_resnet1.forward(&x, None)?;
        let x = group_norm(&x, &self.norm_out_w, &self.norm_out_b, GN_GROUPS, GN_EPS)?;
        conv2d(&silu(&x)?, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }
}

/// The SDXL autoencoder. `decode` reconstructs an image from latents; `encode` produces the latent
/// mean (used to seed img2img).
pub struct Autoencoder {
    encoder: Encoder,
    decoder: Decoder,
    /// `quant_conv` [8,8,1,1] → [8,8] Linear over channels.
    quant_proj: AdaptableLinear,
    /// `post_quant_conv` [4,4,1,1] → [4,4] Linear over channels.
    post_quant_proj: AdaptableLinear,
    scaling_factor: f32,
}

impl Autoencoder {
    pub fn from_weights(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let squeeze_lin = |name: &str| -> Result<AdaptableLinear> {
            let cw = w.require(&format!("{name}.weight"))?;
            let sh = cw.shape();
            Ok(AdaptableLinear::dense(
                cw.reshape(&[sh[0], sh[1]])?,
                Some(w.require(&format!("{name}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            encoder: Encoder::from_weights(w, cfg)?,
            decoder: Decoder::from_weights(w, cfg)?,
            quant_proj: squeeze_lin("quant_conv")?,
            post_quant_proj: squeeze_lin("post_quant_conv")?,
            scaling_factor: cfg.scaling_factor,
        })
    }

    /// Decode latents `[B, H/8, W/8, 4]` (NHWC) → image tensor `[B, H, W, 3]` in roughly `[-1, 1]`.
    pub fn decode(&self, latents: &Array) -> Result<Array> {
        let z = multiply(latents, scalar(1.0 / self.scaling_factor))?;
        let z = self.post_quant_proj.forward(&z)?;
        self.decoder.forward(&z)
    }

    /// Encode an image `[B, 3, ...]`-normalized NHWC `[B, H, W, 3]` → latent **mean** `[B, H/8, W/8,
    /// 4]` (scaled by `scaling_factor`). Mirrors the vendored `Autoencoder.encode` (returns the mean;
    /// img2img seeds from the mean, not a sample).
    pub fn encode_mean(&self, x: &Array) -> Result<Array> {
        let moments = self.quant_proj.forward(&self.encoder.forward(x)?)?;
        // split into (mean, logvar) along the channel axis; keep the mean · scaling_factor.
        let c = moments.shape()[3];
        let half = c / 2;
        let idx = Array::from_slice(&(0..half).collect::<Vec<i32>>(), &[half]);
        let mean = moments.take_axis(&idx, 3)?;
        Ok(multiply(&mean, scalar(self.scaling_factor))?)
    }
}
