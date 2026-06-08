//! XLabs FLUX IP-Adapter (image-prompt / reference conditioning) for FLUX.1 (sc-3623).
//!
//! Two pieces, both loaded from `XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`:
//!
//! 1. [`FluxImageProjModel`] (`ip_adapter_proj_model.*`) — XLabs' `ImageProjModel`: projects the
//!    CLIP-ViT-L/14 `image_embeds` `[B, 768]` (sc-3622) → `num_tokens=4` image-prompt tokens of
//!    width `cross_attention_dim=4096` (`Linear(768→4·4096)` → reshape → `LayerNorm(4096)`).
//! 2. [`FluxIpAdapter`] — the per-double-block decoupled cross-attention K/V projections
//!    (`double_blocks.{i}.processor.ip_adapter_double_stream_{k,v}_proj`, `Linear(4096→3072)`, all
//!    19 FLUX double blocks). At each double block the IP branch attends the block's **own** image
//!    query (post-RMS-norm, **pre-RoPE** — the diffusers `FluxIPAdapterAttnProcessor` semantics,
//!    which is the torch parity target: `ip_query = query` is captured *before* `apply_rotary_emb`)
//!    to the projected image tokens, and the result scaled by `ip_adapter_scale` is added **raw
//!    (ungated), after the FF residual**, to the block output — diffusers' `hidden_states =
//!    hidden_states + ip_attn_output` (transformer_flux.py:477). It deliberately bypasses the
//!    block's `gate_msa` and the FF input; folding it into the pre-gate attention output would both
//!    suppress it where the adaLN gate is small (weak resemblance) and distort the velocity
//!    (true_cfg saturation). There is **no** IP-side query or output projection in the checkpoint —
//!    the block's query is reused and the V projection output is added directly.
//!
//! Wiring into the DiT rides the [`crate::transformer::DitImageInjector`] seam via
//! [`FluxIpInjector`] (`double_block_ip`); the plain txt2img path is untouched.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::transformer::{DitImageInjector, HEADS, HEAD_DIM};

/// `ImageProjModel` LayerNorm epsilon (torch `nn.LayerNorm` default).
const PROJ_LN_EPS: f32 = 1e-5;
/// XLabs `ImageProjModel` token count + cross-attention width.
const NUM_TOKENS: i32 = 4;
const CROSS_ATTN_DIM: i32 = 4096;
/// FLUX has 19 dual-stream (double) blocks; the XLabs IP-Adapter projects into every one.
const NUM_DOUBLE_BLOCKS: usize = 19;

/// XLabs `ImageProjModel`: CLIP `image_embeds` `[B, 768]` → image-prompt tokens `[B, 4, 4096]`.
struct FluxImageProjModel {
    proj: AdaptableLinear, // 768 → 4·4096 (+bias)
    norm_w: Array,         // LayerNorm over 4096
    norm_b: Array,
}

impl FluxImageProjModel {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let proj = AdaptableLinear::dense(
            w.require(&format!("{prefix}.proj.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj.bias"))?.clone()),
        );
        Ok(Self {
            proj,
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.clone(),
        })
    }

    /// `image_embeds` `[B, 768]` → `[B, 4, 4096]` (cast to the projection's weight dtype).
    fn forward(&self, image_embeds: &Array) -> Result<Array> {
        let b = image_embeds.shape()[0];
        let dtype = self.norm_w.dtype();
        let x = self.proj.forward(&image_embeds.as_dtype(dtype)?)?; // [B, 4·4096]
        let x = x.reshape(&[b, NUM_TOKENS, CROSS_ATTN_DIM])?;
        Ok(layer_norm(
            &x,
            Some(&self.norm_w),
            Some(&self.norm_b),
            PROJ_LN_EPS,
        )?)
    }
}

/// The XLabs FLUX IP-Adapter: the image-token projector + the 19 per-double-block K/V projections.
pub struct FluxIpAdapter {
    proj_model: FluxImageProjModel,
    /// `(k_proj, v_proj)` per double block, in block order. Each `Linear(4096 → 3072)` (+bias).
    blocks: Vec<(AdaptableLinear, AdaptableLinear)>,
}

impl FluxIpAdapter {
    /// Load the whole adapter from an `XLabs-AI/flux-ip-adapter` `ip_adapter.safetensors`. The
    /// checkpoint keys are already in the natural MLX layout (`Linear` weights `[out, in]`, used by
    /// `AdaptableLinear` directly), so there is no conversion step — unlike the EVA-CLIP tower.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let proj_model = FluxImageProjModel::from_weights(w, "ip_adapter_proj_model")?;
        let mut blocks = Vec::with_capacity(NUM_DOUBLE_BLOCKS);
        for i in 0..NUM_DOUBLE_BLOCKS {
            let p = format!("double_blocks.{i}.processor.ip_adapter_double_stream");
            let k = AdaptableLinear::dense(
                w.require(&format!("{p}_k_proj.weight"))?.clone(),
                Some(w.require(&format!("{p}_k_proj.bias"))?.clone()),
            );
            let v = AdaptableLinear::dense(
                w.require(&format!("{p}_v_proj.weight"))?.clone(),
                Some(w.require(&format!("{p}_v_proj.bias"))?.clone()),
            );
            blocks.push((k, v));
        }
        if blocks.len() != NUM_DOUBLE_BLOCKS {
            return Err(Error::Msg(format!(
                "flux ip-adapter: expected {NUM_DOUBLE_BLOCKS} double-block K/V pairs, got {}",
                blocks.len()
            )));
        }
        Ok(Self { proj_model, blocks })
    }

    /// CLIP `image_embeds` `[B, 768]` → image-prompt tokens `[B, 4, 4096]`. Computed once per
    /// reference image and reused across every block and denoise step.
    pub fn tokens(&self, image_embeds: &Array) -> Result<Array> {
        self.proj_model.forward(image_embeds)
    }

    /// Number of double blocks the adapter projects into (19 for FLUX.1).
    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Decoupled cross-attention residual for double block `block_idx`:
    /// `scale · SDPA(img_q, K·tokens, V·tokens)`, reshaped to `[B, img_seq, 3072]` — the term the
    /// block adds **raw (ungated), after the FF residual** to its output (diffusers'
    /// `ip_attn_output`). `img_q` is the block's RMS-normed, **pre-RoPE** per-head image query
    /// `[B, HEADS, img_seq, HEAD_DIM]` (diffusers' `ip_query`, captured before RoPE).
    fn block_residual(
        &self,
        block_idx: usize,
        img_q: &Array,
        tokens: &Array,
        scale: f32,
    ) -> Result<Array> {
        let (k_proj, v_proj) = &self.blocks[block_idx];
        let b = img_q.shape()[0];
        let dtype = img_q.dtype();
        // K/V over the image tokens → [B, NUM_TOKENS, 3072] → [B, HEADS, NUM_TOKENS, HEAD_DIM].
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, NUM_TOKENS, HEADS, HEAD_DIM])?
                .transpose_axes(&[0, 2, 1, 3])?
                .as_dtype(dtype)?)
        };
        let k = to_heads(k_proj.forward(tokens)?)?;
        let v = to_heads(v_proj.forward(tokens)?)?;
        // ip_query has no positional rope (the image tokens carry no spatial position).
        let o =
            scaled_dot_product_attention(img_q, &k, &v, (HEAD_DIM as f32).powf(-0.5), None, None)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])? // [B, img_seq, HEADS, HEAD_DIM]
            .reshape(&[b, -1, HEADS * HEAD_DIM])?; // [B, img_seq, 3072]
        Ok(multiply(&o, &Array::from(scale).as_dtype(dtype)?)?)
    }
}

/// A [`DitImageInjector`] that drives the FLUX IP-Adapter: it owns the adapter and the
/// pre-computed image-prompt tokens for one reference image, and answers `double_block_ip` with the
/// scaled decoupled-cross-attention residual. `scale = 0` (or no reference) is a no-op.
pub struct FluxIpInjector<'a> {
    adapter: &'a FluxIpAdapter,
    tokens: Array,
    scale: f32,
}

impl<'a> FluxIpInjector<'a> {
    /// Build from a reference image's CLIP `image_embeds` `[B, 768]` and the `ip_adapter_scale`.
    pub fn new(adapter: &'a FluxIpAdapter, image_embeds: &Array, scale: f32) -> Result<Self> {
        Ok(Self {
            adapter,
            tokens: adapter.tokens(image_embeds)?,
            scale,
        })
    }

    /// A zero-strength injector (the CFG uncond row / "no image prompt"): byte-identical to plain
    /// txt2img because `double_block_ip` short-circuits on `scale == 0` and draws no RNG.
    pub fn disabled(adapter: &'a FluxIpAdapter, image_embeds: &Array) -> Result<Self> {
        Self::new(adapter, image_embeds, 0.0)
    }
}

impl DitImageInjector for FluxIpInjector<'_> {
    fn after_double(&self, _block_idx: usize, _img_hidden: &Array) -> Result<Option<Array>> {
        Ok(None)
    }
    fn injects_after_single(&self, _block_idx: usize) -> bool {
        false
    }
    fn after_single(&self, _block_idx: usize, _img_tokens: &Array) -> Result<Option<Array>> {
        Ok(None)
    }

    fn double_block_ip(&self, block_idx: usize, img_q: &Array) -> Result<Option<Array>> {
        if self.scale == 0.0 || block_idx >= self.adapter.num_blocks() {
            return Ok(None);
        }
        Ok(Some(self.adapter.block_residual(
            block_idx,
            img_q,
            &self.tokens,
            self.scale,
        )?))
    }
}
