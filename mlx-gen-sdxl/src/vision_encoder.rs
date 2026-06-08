//! CLIP **ViT-H/14 vision tower** — the image encoder for IP-Adapter (`h94/IP-Adapter`'s
//! `models/image_encoder`, an OpenCLIP `CLIPVisionModelWithProjection`). This is the one net-new
//! module the IP-Adapter spike (sc-3056) needs: mlx-gen had no image encoder before.
//!
//! The transformer body is the **same** pre-norm self-attention + MLP stack as the CLIP *text*
//! encoder ([`crate::text_encoder`]) — identical `self_attn.{q,k,v,out}_proj` / `mlp.fc1|fc2`
//! naming — so only the *front* differs: a patch-conv + class-token + learned-position embedding
//! (`vision_model.embeddings`) and a `pre_layrnorm` (note HF's spelling), and there is **no causal
//! mask** (full bidirectional attention). IP-Adapter "plus" consumes the **penultimate** hidden
//! state (`hidden_states[-2]`, raw — before `post_layernorm`/`visual_projection`), so `forward`
//! returns the full HF-style hidden-state list and the caller indexes `[-2]`.
//!
//! Config is ViT-H/14: 1280-wide, 32 layers, 16 heads (head_dim 80), patch 14, 224px → 257 tokens
//! (1 class + 16×16 patches), `gelu` (exact), LN eps 1e-5.
//!
//! NOTE (sc-3056 spike): lives in `mlx-gen-sdxl` for now; promote to a shared module when
//! FLUX/Qwen IP-Adapter reuse it (epic 3041 "build generic where cheap").

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{conv2d, gelu_exact, gelu_quick};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// CLIP's LayerNorm epsilon (shared with the text encoder).
const LN_EPS: f32 = 1e-5;

/// CLIP vision tower config. Defaults are the ViT-H/14 used by `h94/IP-Adapter`.
#[derive(Clone, Debug)]
pub struct VisionConfig {
    pub hidden: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub patch: i32,
    pub image_size: i32,
    pub num_channels: i32,
    /// MLP activation: `quick_gelu` (`x·sigmoid(1.702x)`, OpenAI CLIP-L) vs exact `gelu` (the laion
    /// OpenCLIP ViT-H tower). `config.json` `hidden_act` — they differ, and over the tower's depth
    /// the wrong one drifts the embeds (~0.93 cosine vs torch on ViT-L; sc-3622).
    pub quick_gelu: bool,
}

impl VisionConfig {
    /// OpenCLIP ViT-H/14 (`models/image_encoder/config.json`).
    pub fn vit_h_14() -> Self {
        Self {
            hidden: 1280,
            num_layers: 32,
            num_heads: 16,
            patch: 14,
            image_size: 224,
            num_channels: 3,
            quick_gelu: false,
        }
    }

    /// OpenAI CLIP ViT-L/14 (`openai/clip-vit-large-patch14`, `vision_model.*`): 1024-wide, 24
    /// layers, 16 heads (head_dim 64), patch 14, 224px → 257 tokens. The image tower the XLabs
    /// FLUX IP-Adapter conditions on (a `CLIPVisionModelWithProjection`; the 1024→768 projection
    /// head lives in the consumer, not here).
    pub fn vit_l_14() -> Self {
        Self {
            hidden: 1024,
            num_layers: 24,
            num_heads: 16,
            patch: 14,
            image_size: 224,
            num_channels: 3,
            quick_gelu: true,
        }
    }

    /// Token count = 1 class token + (image_size / patch)² patches (= 257 for ViT-H/14).
    pub fn num_positions(&self) -> i32 {
        let grid = self.image_size / self.patch;
        grid * grid + 1
    }
}

/// One CLIP vision encoder layer (pre-norm self-attention + pre-norm MLP, both residual). Mirrors
/// the text encoder's layer; the only structural difference upstream is the absence of a mask.
struct VisionEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    quick_gelu: bool,
}

impl VisionEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &VisionConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        let dense = |wn: &str, bn: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{wn}"))?.clone(),
                Some(w.require(&format!("{prefix}.{bn}"))?.clone()),
            ))
        };
        let head_dim = cfg.hidden / cfg.num_heads;
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            q: dense("self_attn.q_proj.weight", "self_attn.q_proj.bias")?,
            k: dense("self_attn.k_proj.weight", "self_attn.k_proj.bias")?,
            v: dense("self_attn.v_proj.weight", "self_attn.v_proj.bias")?,
            out: dense("self_attn.out_proj.weight", "self_attn.out_proj.bias")?,
            fc1: dense("mlp.fc1.weight", "mlp.fc1.bias")?,
            fc2: dense("mlp.fc2.weight", "mlp.fc2.bias")?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            quick_gelu: cfg.quick_gelu,
        })
    }

    /// `x`: `[B, N, D]`. No mask (bidirectional).
    fn forward(&self, x: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), LN_EPS)?;
        let y = self.attention(&y)?;
        let x = add(x, &y)?;

        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), LN_EPS)?;
        let y = self.fc1.forward(&y)?;
        let y = if self.quick_gelu {
            gelu_quick(&y)?
        } else {
            gelu_exact(&y)?
        };
        let y = self.fc2.forward(&y)?;
        Ok(add(&x, &y)?)
    }

    fn attention(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, n, self.num_heads * self.head_dim])?;
        self.out.forward(&o)
    }
}

/// A loaded CLIP ViT-H/14 vision tower (transformer body + patch/class/position embeddings).
pub struct ClipVisionEncoder {
    /// Patch conv weight in mlx NHWC layout `[hidden, patch, patch, channels]` (no bias).
    patch_embedding: Array,
    /// `[hidden]` learned class token.
    class_embedding: Array,
    /// `[num_positions, hidden]` learned position table.
    position_embedding: Array,
    pre_ln_w: Array,
    pre_ln_b: Array,
    layers: Vec<VisionEncoderLayer>,
    patch: i32,
    hidden: i32,
}

impl ClipVisionEncoder {
    /// Load from an `image_encoder` checkpoint (`vision_model.*` prefix). The patch conv weight is
    /// stored NCHW `[out, in, kH, kW]` and transposed to mlx NHWC on load.
    pub fn from_weights(w: &Weights, cfg: &VisionConfig) -> Result<Self> {
        let p = "vision_model";
        let patch_nchw = w.require(&format!("{p}.embeddings.patch_embedding.weight"))?;
        let patch_embedding = patch_nchw.transpose_axes(&[0, 2, 3, 1])?; // [O, kH, kW, in]
        let class_embedding = w
            .require(&format!("{p}.embeddings.class_embedding"))?
            .clone();
        let position_embedding = w
            .require(&format!("{p}.embeddings.position_embedding.weight"))?
            .clone();
        let pre_ln_w = w.require(&format!("{p}.pre_layrnorm.weight"))?.clone();
        let pre_ln_b = w.require(&format!("{p}.pre_layrnorm.bias"))?.clone();
        let layers = (0..cfg.num_layers)
            .map(|i| VisionEncoderLayer::from_weights(w, &format!("{p}.encoder.layers.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            pre_ln_w,
            pre_ln_b,
            layers,
            patch: cfg.patch,
            hidden: cfg.hidden,
        })
    }

    /// Embed `pixel_values` (NHWC `[B, H, W, 3]`, CLIP-normalized) → `[B, num_positions, hidden]`,
    /// then `pre_layrnorm`. This is HF's `hidden_states[0]` (the input to encoder layer 0).
    fn embed(&self, pixel_values: &Array) -> Result<Array> {
        let b = pixel_values.shape()[0];
        // Patch conv: stride = kernel = patch, no padding, no bias → [B, grid, grid, hidden].
        let patches = conv2d(pixel_values, &self.patch_embedding, None, self.patch, 0)?;
        let grid = patches.shape()[1];
        let n_patch = grid * grid;
        let patches = patches.reshape(&[b, n_patch, self.hidden])?;
        // Prepend the class token, broadcast over the batch → [B, 1+n_patch, hidden].
        let cls = broadcast_to(
            &self.class_embedding.reshape(&[1, 1, self.hidden])?,
            &[b, 1, self.hidden],
        )?;
        let x = concatenate_axis(&[&cls, &patches], 1)?;
        // Add the learned position table (one row per token).
        let pos = self.position_embedding.reshape(&[
            1,
            self.position_embedding.shape()[0],
            self.hidden,
        ])?;
        let x = add(&x, &pos)?;
        Ok(layer_norm(
            &x,
            Some(&self.pre_ln_w),
            Some(&self.pre_ln_b),
            LN_EPS,
        )?)
    }

    /// Run the tower and return the **HF-style hidden-state list**: `[pre_ln_out, L0_out, …,
    /// L{n-1}_out]` (length `num_layers + 1`). IP-Adapter "plus" reads `[-2]` (= the penultimate
    /// layer's output).
    pub fn hidden_states(&self, pixel_values: &Array) -> Result<Vec<Array>> {
        let mut x = self.embed(pixel_values)?;
        let mut states = Vec::with_capacity(self.layers.len() + 1);
        states.push(x.clone());
        for layer in &self.layers {
            x = layer.forward(&x)?;
            states.push(x.clone());
        }
        Ok(states)
    }

    /// The penultimate hidden state `[B, num_positions, hidden]` — the IP-Adapter "plus" image
    /// features fed to the Resampler.
    pub fn penultimate(&self, pixel_values: &Array) -> Result<Array> {
        let states = self.hidden_states(pixel_values)?;
        Ok(states[states.len() - 2].clone())
    }
}
