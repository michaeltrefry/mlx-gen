//! `UNet2DConditionModel` — the SDXL denoising U-Net. Port of the vendored `unet.UNetModel`: a
//! conv stem, sinusoidal timestep + SDXL `text_time` micro-conditioning embeddings, a down /
//! mid / up stack of [`UNetBlock2D`]s with cross-attention to the dual-CLIP text conditioning, and
//! a conv head. Runs entirely in NHWC. Predicts the noise (`eps`) for one denoise step.

mod block;
mod embeddings;
mod resnet;
mod transformer;

use mlx_rs::ops::{add, concatenate_axis};
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{conv2d, group_norm, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::UNetConfig;
use block::{BlockSpec, UNetBlock2D};
use embeddings::{SinusoidalPositionalEncoding, TimestepEmbedding};
use transformer::Transformer2D;

// Shared with the VAE (the vendored VAE reuses the UNet `ResnetBlock2D` without a time embedding).
pub use resnet::ResnetBlock2D;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;

/// Transpose a stored NCHW conv weight `[out, in, kH, kW]` to mlx's NHWC `[out, kH, kW, in]`.
pub(crate) fn nchw_to_nhwc(w: &Array) -> Result<Array> {
    Ok(w.transpose_axes(&[0, 2, 3, 1])?)
}

/// The SDXL conditional U-Net.
pub struct UNet2DConditionModel {
    conv_in_w: Array,
    conv_in_b: Array,
    timesteps: SinusoidalPositionalEncoding,
    time_embedding: TimestepEmbedding,
    add_time_proj: SinusoidalPositionalEncoding,
    add_embedding: TimestepEmbedding,
    down_blocks: Vec<UNetBlock2D>,
    mid_resnet0: ResnetBlock2D,
    mid_transformer: Transformer2D,
    mid_resnet1: ResnetBlock2D,
    up_blocks: Vec<UNetBlock2D>,
    conv_norm_out_w: Array,
    conv_norm_out_b: Array,
    conv_out_w: Array,
    conv_out_b: Array,
}

impl UNet2DConditionModel {
    /// Assemble the U-Net from a diffusers SDXL `unet/` checkpoint (keys read directly; conv weights
    /// transposed to NHWC on load). `cfg` is [`UNetConfig::sdxl_base`].
    pub fn from_weights(w: &Weights, cfg: &UNetConfig) -> Result<Self> {
        let n = cfg.num_blocks();
        let boc = &cfg.block_out_channels;
        let temb_dim_src = boc[0]; // sinusoidal timestep width

        // Down blocks: block i goes block_channels[i] -> block_channels[i+1].
        let mut down_blocks = Vec::with_capacity(n);
        // `i` indexes five parallel config arrays + the block prefix, not just `boc` — an
        // `enumerate()` rewrite would be strictly worse here.
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

        // Mid: resnet, transformer, resnet (the vendored mid_blocks.0/1/2).
        let mid_resnet0 = ResnetBlock2D::from_weights(w, "mid_block.resnets.0")?;
        let mid_transformer = Transformer2D::from_weights(
            w,
            "mid_block.attentions.0",
            *boc.last().unwrap(),
            *cfg.num_attention_heads.last().unwrap(),
            *cfg.transformer_layers_per_block.last().unwrap(),
        )?;
        let mid_resnet1 = ResnetBlock2D::from_weights(w, "mid_block.resnets.1")?;

        // Up blocks: checkpoint up_blocks.{k} corresponds to config index `n-1-k` (the vendored
        // builds them in reversed order). add_upsample on all but the last config index (0).
        let mut up_blocks = Vec::with_capacity(n);
        for k in 0..n {
            let ci = n - 1 - k;
            up_blocks.push(UNetBlock2D::from_weights(
                w,
                &BlockSpec {
                    prefix: &format!("up_blocks.{k}"),
                    num_resnets: cfg.layers_per_block[ci] + 1,
                    out_channels: boc[ci],
                    num_heads: cfg.num_attention_heads[ci],
                    transformer_layers: cfg.transformer_layers_per_block[ci],
                    add_cross_attention: cfg.up_block_types[ci].contains("CrossAttn"),
                    add_downsample: false,
                    add_upsample: ci > 0,
                },
            )?);
        }

        Ok(Self {
            conv_in_w: nchw_to_nhwc(w.require("conv_in.weight")?)?,
            conv_in_b: w.require("conv_in.bias")?.clone(),
            timesteps: SinusoidalPositionalEncoding::timestep(temb_dim_src)?,
            time_embedding: TimestepEmbedding::from_weights(w, "time_embedding")?,
            add_time_proj: SinusoidalPositionalEncoding::timestep(
                cfg.addition_time_embed_dim.unwrap_or(256),
            )?,
            add_embedding: TimestepEmbedding::from_weights(w, "add_embedding")?,
            down_blocks,
            mid_resnet0,
            mid_transformer,
            mid_resnet1,
            up_blocks,
            conv_norm_out_w: w.require("conv_norm_out.weight")?.clone(),
            conv_norm_out_b: w.require("conv_norm_out.bias")?.clone(),
            conv_out_w: nchw_to_nhwc(w.require("conv_out.weight")?)?,
            conv_out_b: w.require("conv_out.bias")?.clone(),
        })
    }

    /// Quantize every Linear (resnets' time/shortcut projections, attention, FFN, embeddings) to
    /// Q4/Q8. Convs (`conv_in`/`conv_out`/resnet convs/up-down samplers) stay dense.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_embedding.quantize(bits)?;
        self.add_embedding.quantize(bits)?;
        for b in &mut self.down_blocks {
            b.quantize(bits)?;
        }
        self.mid_resnet0.quantize(bits)?;
        self.mid_transformer.quantize(bits)?;
        self.mid_resnet1.quantize(bits)?;
        for b in &mut self.up_blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// Predict `eps` for one denoise step.
    /// - `x`: NHWC latents `[B, H, W, 4]`.
    /// - `timestep`: the (sigma-space) time, broadcast to the batch.
    /// - `encoder_x`: dual-CLIP text conditioning `[B, S, 2048]`.
    /// - `text_emb`: pooled conditioning `[B, 1280]`; `time_ids`: micro-conditioning `[B, 6]`.
    pub fn forward(
        &self,
        x: &Array,
        timestep: f32,
        encoder_x: &Array,
        text_emb: &Array,
        time_ids: &Array,
    ) -> Result<Array> {
        let batch = x.shape()[0];
        let dtype = x.dtype();

        // Timestep embedding (broadcast the scalar time to the batch). The sinusoidal encoding runs
        // in f32 (its `sigmas` table is f32), then the reference casts to the model dtype *before* the
        // `time_embedding` MLP (`temb = self.timesteps(t).astype(x.dtype)`), so the MLP runs in the
        // model dtype. The cast is a no-op for the f32 path.
        let t = Array::from_slice(&vec![timestep; batch as usize], &[batch]);
        let temb = self.timesteps.forward(&t)?.as_dtype(dtype)?;
        let mut temb = self.time_embedding.forward(&temb)?;

        // SDXL `text_time` added conditioning: concat(pooled_text, flattened sinusoidal time_ids).
        // `time_ids` stays f32 through its sinusoidal (the reference builds it f32), then the flattened
        // result is cast to the model dtype before concat with the (model-dtype) pooled text
        // (`...flatten(1).astype(x.dtype)`).
        let emb = self.add_time_proj.forward(time_ids)?; // [B, 6, 256]
        let es = emb.shape();
        let emb = emb.reshape(&[es[0], es[1] * es[2]])?.as_dtype(dtype)?; // flatten(1) → [B, 1536]
        let emb = concatenate_axis(&[text_emb, &emb], -1)?; // [B, 2816]
        let emb = self.add_embedding.forward(&emb)?;
        temb = add(&temb, &emb)?;

        // Conv stem.
        let mut x = conv2d(x, &self.conv_in_w, Some(&self.conv_in_b), 1, 1)?;

        // Down path — collect skip residuals (starting with the stem output).
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

        // Up path — each block pops its skip residuals.
        for block in &self.up_blocks {
            let (out, _) = block.forward(&x, encoder_x, &temb, Some(&mut residuals))?;
            x = out;
        }

        // Conv head.
        let x = group_norm(
            &x,
            &self.conv_norm_out_w,
            &self.conv_norm_out_b,
            GN_GROUPS,
            GN_EPS,
        )?;
        let x = silu(&x)?;
        conv2d(&x, &self.conv_out_w, Some(&self.conv_out_b), 1, 1)
    }

    /// Every LoRA-targetable Linear's diffusers dotted path, matching the vendored `lora.py`
    /// reachable surface (sc-2639): down/up attention (`to_q/k/v`, `to_out.0`), the `proj_in`/`proj_out`
    /// projections, and each resnet's `time_emb_proj`. **`mid_block` is intentionally omitted** — the
    /// vendored mlx-examples UNet names it `mid_blocks.1.…`, so community/diffusers LoRA keys
    /// (`mid_block.attentions.0.…`) never match and the vendored path silently drops them; this port
    /// reproduces that exactly. The correct/complete mid_block + ff coverage (strictly more than the
    /// vendored path) is sc-2671. This list also builds the kohya `flattened→dotted` lookup table.
    pub fn lora_target_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.down_blocks.iter().enumerate() {
            b.lora_target_paths(&format!("down_blocks.{i}"), &mut out);
        }
        for (k, b) in self.up_blocks.iter().enumerate() {
            b.lora_target_paths(&format!("up_blocks.{k}"), &mut out);
        }
        out
    }

    /// The **complete** LoRA-targetable surface (sc-2671), strictly larger than the vendored-faithful
    /// [`lora_target_paths`](Self::lora_target_paths): the 515 down/up attention+proj+time_emb paths
    /// **plus** `mid_block.attentions.0` (attention + `proj_in`/`proj_out`) — which the vendored
    /// mlx-examples UNet names `mid_blocks.1.…` and so silently drops — **plus** the GEGLU feed-forward
    /// (`ff.net.0.proj`, `ff.net.2`) of every cross-attention transformer (down + mid + up). Used to
    /// build the kohya lookup table when complete coverage is requested; `mid_block`/`ff` deltas are
    /// reachable through [`AdaptableHost::adaptable_mut`] (the merge layer row-splits a `ff.net.0.proj`
    /// delta into `linear1`/`linear2`). Conv-layer LoRAs remain out of scope (they are not Linears).
    pub fn lora_target_paths_complete(&self) -> Vec<String> {
        let mut out = self.lora_target_paths();
        // mid_block attention + proj (the +82 the vendored path can't reach) and the two mid resnet
        // `time_emb_proj`s (symmetric with the down/up resnet time_emb already in the faithful 515).
        self.mid_resnet0
            .lora_target_paths("mid_block.resnets.0", &mut out);
        self.mid_transformer
            .lora_target_paths("mid_block.attentions.0", &mut out);
        self.mid_resnet1
            .lora_target_paths("mid_block.resnets.1", &mut out);
        // GEGLU feed-forward across every cross-attention transformer.
        for (i, b) in self.down_blocks.iter().enumerate() {
            b.lora_target_paths_ff(&format!("down_blocks.{i}"), &mut out);
        }
        self.mid_transformer
            .lora_target_paths_ff("mid_block.attentions.0", &mut out);
        for (k, b) in self.up_blocks.iter().enumerate() {
            b.lora_target_paths_ff(&format!("up_blocks.{k}"), &mut out);
        }
        out
    }
}

impl AdaptableHost for UNet2DConditionModel {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["down_blocks", i, rest @ ..] => self
                .down_blocks
                .get_mut(i.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["up_blocks", k, rest @ ..] => self
                .up_blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            // mid_block (sc-2671 complete coverage). Routable here, but the vendored coverage path
            // gates mid_block/ff keys out so the faithful 515-module merge is unaffected; only the
            // opt-in complete coverage actually merges into these.
            ["mid_block", "attentions", "0", rest @ ..] => self.mid_transformer.adaptable_mut(rest),
            ["mid_block", "resnets", "0", rest @ ..] => self.mid_resnet0.adaptable_mut(rest),
            ["mid_block", "resnets", "1", rest @ ..] => self.mid_resnet1.adaptable_mut(rest),
            _ => None,
        }
    }
}
