//! SDXL **ControlNet** branch (sc-3058) — a diffusers `ControlNetModel` (e.g. the xinsir tile-CN
//! `controlnet-tile-sdxl-1.0`). It is an *encoder copy* of the SDXL UNet: the same `conv_in` +
//! timestep/`text_time` embeddings + down/mid stack (loaded with the identical block loaders — the
//! checkpoint keys match the UNet's `down_blocks.*` / `mid_block.*`), plus three net-new pieces:
//!   - `controlnet_cond_embedding` — a tiny conv stack (3→16→32→96→256→320, three stride-2 convs)
//!     that embeds the control image to latent resolution and is *added* to `conv_in(latents)`;
//!   - `controlnet_down_blocks` — nine 1×1 "zero-conv" projections, one per down residual;
//!   - `controlnet_mid_block` — one 1×1 zero-conv for the mid output.
//!
//! `forward` returns the per-down-block + mid **residuals** (scaled by `conditioning_scale`), which
//! the main UNet adds into its skip connections + mid output. The control's `encoder_hidden_states`
//! is a **caller-supplied parameter** (text for tile-CN; the 16 face tokens for InstantID, sc-3114),
//! so the branch is a generic primitive, not tile-specific.

use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::block::{BlockSpec, UNetBlock2D};
use super::embeddings::{SinusoidalPositionalEncoding, TimestepEmbedding};
use super::nchw_to_nhwc;
use super::resnet::ResnetBlock2D;
use super::transformer::Transformer2D;
use crate::config::UNetConfig;

/// A plain (non-adapter) conv layer in NHWC. `controlnet_cond_embedding` + the zero-convs are not
/// LoRA targets, so they don't need [`mlx_gen::adapters::AdaptableConv2d`].
struct Conv2dLayer {
    weight: Array, // NHWC [out, kH, kW, in]
    bias: Array,
    stride: i32,
    pad: i32,
}

impl Conv2dLayer {
    fn load(w: &Weights, prefix: &str, stride: i32, pad: i32) -> Result<Self> {
        Ok(Self {
            weight: nchw_to_nhwc(w.require(&format!("{prefix}.weight"))?)?,
            bias: w.require(&format!("{prefix}.bias"))?.clone(),
            stride,
            pad,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        conv2d(x, &self.weight, Some(&self.bias), self.stride, self.pad)
    }
}

/// `ControlNetConditioningEmbedding`: `conv_in(3→16) → SiLU → [block → SiLU]×6 → conv_out(256→320)`.
/// The six blocks alternate stride 1 / stride 2 (three stride-2 ⇒ 8× downsample to latent res). No
/// trailing SiLU after `conv_out`.
struct CondEmbedding {
    conv_in: Conv2dLayer,
    blocks: Vec<Conv2dLayer>,
    conv_out: Conv2dLayer,
}

impl CondEmbedding {
    fn load(w: &Weights) -> Result<Self> {
        let p = "controlnet_cond_embedding";
        let conv_in = Conv2dLayer::load(w, &format!("{p}.conv_in"), 1, 1)?;
        // blocks 0..6: even = stride 1, odd = stride 2 (the down-projection convs).
        let blocks = (0..6)
            .map(|i| {
                let stride = if i % 2 == 0 { 1 } else { 2 };
                Conv2dLayer::load(w, &format!("{p}.blocks.{i}"), stride, 1)
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_out = Conv2dLayer::load(w, &format!("{p}.conv_out"), 1, 1)?;
        Ok(Self {
            conv_in,
            blocks,
            conv_out,
        })
    }

    /// `control`: NHWC `[B, H, W, 3]` in `[0,1]` → `[B, H/8, W/8, 320]`.
    fn forward(&self, control: &Array) -> Result<Array> {
        let mut e = silu(&self.conv_in.forward(control)?)?;
        for b in &self.blocks {
            e = silu(&b.forward(&e)?)?;
        }
        self.conv_out.forward(&e)
    }
}

/// The control residuals produced by one ControlNet forward, already scaled by `conditioning_scale`.
pub struct ControlResiduals {
    /// Nine down-block residuals (matching the UNet's collected skip residuals 1:1).
    pub down: Vec<Array>,
    /// The mid-block residual.
    pub mid: Array,
}

/// An SDXL ControlNet (UNet encoder copy + conditioning embedding + zero-conv heads).
pub struct ControlNet {
    conv_in: Conv2dLayer,
    timesteps: SinusoidalPositionalEncoding,
    time_embedding: TimestepEmbedding,
    add_time_proj: SinusoidalPositionalEncoding,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetBlock2D>,
    mid_resnet0: ResnetBlock2D,
    mid_transformer: Transformer2D,
    mid_resnet1: ResnetBlock2D,
    cond_embedding: CondEmbedding,
    /// Nine 1×1 zero-conv down projections (one per down residual).
    down_zero: Vec<Conv2dLayer>,
    /// The 1×1 zero-conv mid projection.
    mid_zero: Conv2dLayer,
    /// Optional context projection (diffusers `encoder_hid_proj`), present only when the checkpoint
    /// carries `encoder_hid_proj.weight` — the **Kolors** ControlNet (sc-3097), which projects the
    /// ChatGLM3 context 4096→`cross_attention_dim` (2048) up front, mirroring its own U-Net. Its own
    /// learned weights (distinct from the U-Net's). Absent for SDXL ControlNets → `None` (no-op).
    encoder_hid_proj: Option<AdaptableLinear>,
}

impl ControlNet {
    /// Load from a diffusers SDXL `ControlNetModel` checkpoint. `cfg` is [`UNetConfig::sdxl_base`]
    /// (the ControlNet shares the UNet's down/mid geometry).
    pub fn from_weights(w: &Weights, cfg: &UNetConfig) -> Result<Self> {
        let n = cfg.num_blocks();
        let boc = &cfg.block_out_channels;

        let mut down_blocks = Vec::with_capacity(n);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            down_blocks.push(UNetBlock2D::from_weights(
                w,
                &BlockSpec {
                    prefix: &format!("down_blocks.{i}"),
                    num_resnets: cfg.layers_per_block[i],
                    out_channels: boc[i],
                    num_heads: cfg.num_attention_heads[i],
                    transformer_layers: cfg.transformer_layers_per_block[i],
                    add_cross_attention: cfg.down_block_types[i].contains("CrossAttn"),
                    add_downsample: i < n - 1,
                    add_upsample: false,
                },
            )?);
        }

        let mid_resnet0 = ResnetBlock2D::from_weights(w, "mid_block.resnets.0")?;
        let mid_transformer = Transformer2D::from_weights(
            w,
            "mid_block.attentions.0",
            *boc.last().unwrap(),
            *cfg.num_attention_heads.last().unwrap(),
            *cfg.transformer_layers_per_block.last().unwrap(),
        )?;
        let mid_resnet1 = ResnetBlock2D::from_weights(w, "mid_block.resnets.1")?;

        // Nine down zero-convs (controlnet_down_blocks.{0..8}) — 1×1, stride 1, no pad.
        let down_zero = (0..9)
            .map(|i| Conv2dLayer::load(w, &format!("controlnet_down_blocks.{i}"), 1, 0))
            .collect::<Result<Vec<_>>>()?;
        let mid_zero = Conv2dLayer::load(w, "controlnet_mid_block", 1, 0)?;

        Ok(Self {
            conv_in: Conv2dLayer::load(w, "conv_in", 1, 1)?,
            timesteps: SinusoidalPositionalEncoding::timestep(boc[0])?,
            time_embedding: TimestepEmbedding::from_weights(w, "time_embedding")?,
            add_time_proj: SinusoidalPositionalEncoding::timestep(
                cfg.addition_time_embed_dim.unwrap_or(256),
            )?,
            add_embedding: TimestepEmbedding::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_resnet0,
            mid_transformer,
            mid_resnet1,
            cond_embedding: CondEmbedding::load(w)?,
            down_zero,
            mid_zero,
            // Kolors `encoder_hid_proj` (4096→2048). Auto-detected: absent for SDXL → `None`.
            encoder_hid_proj: w.get("encoder_hid_proj.weight").map(|wt| {
                AdaptableLinear::dense(wt.clone(), w.get("encoder_hid_proj.bias").cloned())
            }),
        })
    }

    /// Compute the control residuals for one denoise step.
    /// - `x`: NHWC latents `[B, H, W, 4]` (the same CFG-batched input the UNet sees).
    /// - `control`: NHWC control image `[B, H, W, 3]` in `[0,1]`.
    /// - `encoder_x`: cross-attention conditioning `[B, S, D]` — **text** for tile-CN, the face
    ///   tokens for InstantID. Generic; the branch does not assume text.
    /// - `scale`: `conditioning_scale` applied to every residual.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        control: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
        scale: f32,
    ) -> Result<ControlResiduals> {
        let batch = x.shape()[0];
        let dtype = x.dtype();

        // Kolors: project the ChatGLM3 context (4096) to `cross_attention_dim` (2048) up front, before
        // any cross-attention (diffusers applies `encoder_hid_proj` once). No-op for SDXL ControlNets.
        let projected;
        let encoder_x = match &self.encoder_hid_proj {
            Some(proj) => {
                projected = proj.forward(encoder_x)?;
                &projected
            }
            None => encoder_x,
        };

        // Timestep + SDXL `text_time` embedding (identical to the UNet).
        let t = Array::from_slice(&vec![timestep; batch as usize], &[batch]);
        let temb = self.timesteps.forward(&t)?.as_dtype(dtype)?;
        let mut temb = self.time_embedding.forward(&temb)?;
        let emb = self.add_time_proj.forward(time_ids)?;
        let es = emb.shape();
        let emb = emb.reshape(&[es[0], es[1] * es[2]])?.as_dtype(dtype)?;
        let emb = mlx_rs::ops::concatenate_axis(&[text_emb, &emb], -1)?;
        let emb = self.add_embedding.forward(&emb)?;
        temb = add(&temb, &emb)?;

        // conv_in + conditioning embedding.
        let mut x = self.conv_in.forward(x)?;
        x = add(&x, &self.cond_embedding.forward(control)?)?;

        // Down — collect skip residuals (starting with the stem+cond output).
        let mut residuals: Vec<Array> = vec![x.clone()];
        for block in &self.down_blocks {
            let (out, res) = block.forward(&x, encoder_x, &temb, None)?;
            x = out;
            residuals.extend(res);
        }

        // Mid.
        x = self.mid_resnet0.forward(&x, Some(&temb))?;
        x = self.mid_transformer.forward(&x, encoder_x)?;
        x = self.mid_resnet1.forward(&x, Some(&temb))?;

        // Zero-conv heads + scale.
        let s = scalar(scale).as_dtype(dtype)?;
        let mut down = Vec::with_capacity(residuals.len());
        for (r, z) in residuals.iter().zip(&self.down_zero) {
            down.push(multiply(&z.forward(r)?, &s)?);
        }
        let mid = multiply(&self.mid_zero.forward(&x)?, &s)?;
        Ok(ControlResiduals { down, mid })
    }

    /// Quantize the encoder-copy Linears (down/mid attention + FFN + time/add embeddings). The
    /// conv stem, conditioning embedding, and zero-convs stay dense (matching the UNet quant scope).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_embedding.quantize(bits)?;
        self.add_embedding.quantize(bits)?;
        for b in &mut self.down_blocks {
            b.quantize(bits)?;
        }
        self.mid_resnet0.quantize(bits)?;
        self.mid_transformer.quantize(bits)?;
        self.mid_resnet1.quantize(bits)?;
        if let Some(proj) = &mut self.encoder_hid_proj {
            proj.quantize(bits, None)?;
        }
        Ok(())
    }
}
