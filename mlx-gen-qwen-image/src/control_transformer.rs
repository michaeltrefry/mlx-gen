//! Qwen-Image **ControlNet-Union** control transformer (epic 3401 / sc-3568). Port of the InstantX
//! `Qwen-Image-ControlNet-Union` `QwenImageControlNetModel`: a small partial copy of the base Qwen
//! MMDiT (the checkpoint ships `num_layers = 5`) that ingests the VAE-encoded control image (here a
//! DWPose skeleton) and emits one per-block residual, injected into the frozen base transformer at
//! `interval = ceil(60 / 5) = 12` (see [`crate::transformer::QwenTransformer::forward_control`]).
//!
//! Unlike the Z-Image Fun-Controlnet-Union (which *threads* a control state through the base blocks
//! at fixed places), the Qwen Union follows the standard diffusers ControlNet shape: it is an
//! **independent** mini-transformer with its own `img_in`/`txt_in`/`txt_norm`/`time_text_embed` and a
//! zero-init `controlnet_x_embedder`; each of its blocks' output is projected by a zero-init
//! `controlnet_blocks[i]` Linear into a residual. The 5 residuals are returned (pre-scale) for the
//! base transformer to add. No condition-type embedding (the checkpoint has `extra_condition_channels
//! = 0` and no control-type embed).
//!
//! The block math is the *same* [`QwenTransformerBlock`] as the base (identical on-disk keys), so the
//! loader reuses the base block remap. Adapters (the character-identity LoRA) target the **base**
//! transformer only — the control branch is never an adapter target (mirrors the Z-Image control
//! port and the fork, which trains LoRA on the base).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::transformer::time_text_embed::TimeTextEmbed;
use crate::transformer::{linear_from, QwenRope3d, QwenTransformerBlock, QwenTransformerConfig};

/// The InstantX `Qwen-Image-ControlNet-Union` config (`config.json`): a 5-block partial copy of the
/// base 60-layer MMDiT, identical inner dims (24 heads × 128 = 3072), `in_channels = 64`,
/// `extra_condition_channels = 0`.
pub struct QwenControlNetConfig {
    pub num_layers: usize,
    pub num_heads: i32,
    pub head_dim: i32,
    pub txt_norm_eps: f32,
}

impl QwenControlNetConfig {
    /// The shipped InstantX Union: `num_layers = 5`, otherwise the base Qwen-Image shape.
    pub fn qwen_image_union() -> Self {
        let base = QwenTransformerConfig::qwen_image();
        Self {
            num_layers: 5,
            num_heads: base.num_heads,
            head_dim: base.head_dim,
            txt_norm_eps: base.txt_norm_eps,
        }
    }
}

/// The Qwen ControlNet-Union control transformer (the trainable branch). Holds its own input
/// projections + 5 dual-stream blocks + 5 zero-init residual projections; emits the per-block
/// residuals for the base transformer.
pub struct QwenControlNet {
    img_in: AdaptableLinear,
    txt_norm_w: Array,
    txt_in: AdaptableLinear,
    time_text_embed: TimeTextEmbed,
    /// Zero-init projection of the packed control latent (`64 → inner_dim`), added to `img_in(x)`.
    controlnet_x_embedder: AdaptableLinear,
    blocks: Vec<QwenTransformerBlock>,
    /// Zero-init per-block residual projections (`inner_dim → inner_dim`).
    controlnet_blocks: Vec<AdaptableLinear>,
    rope: QwenRope3d,
    eps: f32,
}

impl QwenControlNet {
    /// Load from the InstantX Union checkpoint (already remapped to the base block's internal key
    /// names by the loader). `prefix` is empty for the real single-file checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &QwenControlNetConfig) -> Result<Self> {
        let p = |s: &str| {
            if prefix.is_empty() {
                s.to_string()
            } else {
                format!("{prefix}.{s}")
            }
        };
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        let mut controlnet_blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(QwenTransformerBlock::from_weights(
                w,
                &p(&format!("transformer_blocks.{i}")),
                cfg.num_heads,
                cfg.head_dim,
            )?);
            controlnet_blocks.push(linear_from(w, &p(&format!("controlnet_blocks.{i}")), true)?);
        }
        Ok(Self {
            img_in: linear_from(w, &p("img_in"), true)?,
            txt_norm_w: w.require(&p("txt_norm.weight"))?.clone(),
            txt_in: linear_from(w, &p("txt_in"), true)?,
            time_text_embed: TimeTextEmbed::from_weights(w, &p("time_text_embed"))?,
            controlnet_x_embedder: linear_from(w, &p("controlnet_x_embedder"), true)?,
            blocks,
            controlnet_blocks,
            rope: QwenRope3d::qwen_image(),
            eps: cfg.txt_norm_eps,
        })
    }

    /// Number of control residuals (= control layers); drives the base injection interval.
    pub fn num_residuals(&self) -> usize {
        self.controlnet_blocks.len()
    }

    /// Quantize the control transformer's Linears to Q4/Q8 (group_size 64), mirroring
    /// [`crate::transformer::QwenTransformer::quantize`] over the control branch. Same transformer-only
    /// scope as T2I/Edit.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_in.quantize(bits, None)?;
        self.txt_in.quantize(bits, None)?;
        self.controlnet_x_embedder.quantize(bits, None)?;
        self.time_text_embed.quantize(bits)?;
        for block in &mut self.blocks {
            block.quantize(bits)?;
        }
        for cb in &mut self.controlnet_blocks {
            cb.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Run the control branch → the per-block residuals (pre-scale), one per control layer.
    ///
    /// `hidden_states`: the current packed **noise** latents `[B, img_seq, 64]` (the controlnet sees
    /// the same latents the base does this step). `control_cond`: the packed VAE-encoded control image
    /// `[B, img_seq, 64]` (constant across steps). `encoder_hidden_states`: text features `[B, txt_seq,
    /// joint_attention_dim]`. `timestep`: the scheduler sigma (same as the base forward). The
    /// returned residuals align 1:1 with the noise token sequence (control pose is a single grid, so
    /// no `cond_grids` / `zero_cond_t`).
    pub fn forward(
        &self,
        hidden_states: &Array,
        control_cond: &Array,
        encoder_hidden_states: &Array,
        timestep: f32,
        latent_h: usize,
        latent_w: usize,
    ) -> Result<Vec<Array>> {
        let b = hidden_states.shape()[0];
        let txt_seq = encoder_hidden_states.shape()[1];

        // `img_in(x) + controlnet_x_embedder(control_cond)` (diffusers
        // `hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)`).
        let mut hidden = add(
            &self.img_in.forward(hidden_states)?,
            &self.controlnet_x_embedder.forward(control_cond)?,
        )?;
        let encoder = rms_norm(encoder_hidden_states, &self.txt_norm_w, self.eps)?;
        let mut encoder = self.txt_in.forward(&encoder)?;

        let ts = Array::from_slice(&vec![timestep; b as usize], &[b]);
        let text_emb = self.time_text_embed.forward(&ts)?;

        // Single-grid RoPE (pose control is one image; no reference / zero_cond_t path).
        let (img_cos, img_sin, txt_cos, txt_sin) =
            self.rope.forward(latent_h, latent_w, txt_seq as usize)?;

        let mut residuals = Vec::with_capacity(self.blocks.len());
        for (block, cn) in self.blocks.iter().zip(&self.controlnet_blocks) {
            let (e, h) = block.forward(
                &hidden, &encoder, &text_emb, &img_cos, &img_sin, &txt_cos, &txt_sin, None, None,
            )?;
            encoder = e;
            hidden = h;
            // residual[i] = controlnet_blocks[i](hidden_after_block_i) (diffusers zero-init proj).
            residuals.push(cn.forward(&hidden)?);
        }
        Ok(residuals)
    }
}
