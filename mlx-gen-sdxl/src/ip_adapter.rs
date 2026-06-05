//! IP-Adapter (image-prompt conditioning) for SDXL — spike (sc-3056).
//!
//! Two pieces live here:
//! 1. [`Resampler`] — the IP-Adapter "plus" image projection (`image_proj.*` in
//!    `ip-adapter-plus_sdxl_vit-h.safetensors`). A perceiver/Resampler that maps the ViT-H
//!    penultimate hidden states `[B, 257, 1280]` → **16 image tokens × 2048** (the UNet
//!    cross-attention width). This is the *original Tencent* `Resampler`/`PerceiverAttention`
//!    layout (fused `to_kv`, `norm1`/`norm2`, bias-free projections), NOT the diffusers refactor.
//! 2. (next) the **decoupled cross-attention** primitive (`to_k_ip`/`to_v_ip` + `ip_adapter_scale`)
//!    injected into the SDXL UNet cross-attn — built generic so the token source is pluggable
//!    (ViT-H→Resampler here; an ArcFace Resampler for InstantID, sc-3113).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, broadcast_to, concatenate_axis, split};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::gelu_exact;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

const LN_EPS: f32 = 1e-5;

/// IP-Adapter "plus" Resampler config. Defaults are `ip-adapter-plus_sdxl_vit-h`.
#[derive(Clone, Debug)]
pub struct ResamplerConfig {
    /// Working width (`dim`); also the latent/query width. 1280 for plus-vit-h.
    pub dim: i32,
    /// Number of perceiver blocks (`depth`). 4 for plus-vit-h.
    pub depth: i32,
    /// Attention heads. 20 for plus-vit-h (head_dim 64).
    pub heads: i32,
    pub dim_head: i32,
    /// Output query tokens (`num_queries`). 16 for plus-vit-h.
    pub num_queries: i32,
    /// Input feature width (ViT-H hidden) feeding `proj_in`. 1280.
    pub embed_dim: i32,
    /// Output token width (= UNet cross_attention_dim). 2048 for SDXL.
    pub output_dim: i32,
}

impl ResamplerConfig {
    pub fn plus_sdxl_vit_h() -> Self {
        Self {
            dim: 1280,
            depth: 4,
            heads: 20,
            dim_head: 64,
            num_queries: 16,
            embed_dim: 1280,
            output_dim: 2048,
        }
    }
}

/// PerceiverAttention block (`layers.{i}.0`): cross-attention from the learned `latents` (queries)
/// to `cat([image_features, latents])` (keys/values), with a fused `to_kv` projection.
struct PerceiverAttention {
    norm1_w: Array, // LN on the image features (x)
    norm1_b: Array,
    norm2_w: Array, // LN on the latents
    norm2_b: Array,
    to_q: AdaptableLinear,   // bias-free
    to_kv: AdaptableLinear,  // bias-free, fused [2*inner, dim]
    to_out: AdaptableLinear, // bias-free
    heads: i32,
    dim_head: i32,
    scale: f32,
}

impl PerceiverAttention {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        let lin = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{name}"))?.clone(),
                None,
            ))
        };
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            to_q: lin("to_q.weight")?,
            to_kv: lin("to_kv.weight")?,
            to_out: lin("to_out.weight")?,
            heads: cfg.heads,
            dim_head: cfg.dim_head,
            scale: (cfg.dim_head as f32).powf(-0.5),
        })
    }

    /// `x`: image features `[B, Nx, dim]`; `latents`: `[B, Nq, dim]`. Returns `[B, Nq, dim]` (the
    /// to_out projection; the Resampler adds the residual outside).
    fn forward(&self, x: &Array, latents: &Array) -> Result<Array> {
        let x = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), LN_EPS)?;
        let latents = layer_norm(latents, Some(&self.norm2_w), Some(&self.norm2_b), LN_EPS)?;
        let b = latents.shape()[0];
        let nq = latents.shape()[1];

        let q = self.to_q.forward(&latents)?;
        let kv_input = concatenate_axis(&[&x, &latents], 1)?; // [B, Nx+Nq, dim]
        let kv = self.to_kv.forward(&kv_input)?; // [B, S, 2*inner]
        let parts = split(&kv, 2, -1)?;
        let (k, v) = (&parts[0], &parts[1]);

        let to_heads = |a: &Array| -> Result<Array> {
            let n = a.shape()[1];
            Ok(a.reshape(&[b, n, self.heads, self.dim_head])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(&q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, nq, self.heads * self.dim_head])?;
        self.to_out.forward(&o)
    }
}

/// FeedForward block (`layers.{i}.1`): LayerNorm → Linear(dim, 4·dim) → GELU → Linear(4·dim, dim),
/// the two Linears bias-free. The Resampler adds the residual outside.
struct ResamplerFeedForward {
    ln_w: Array,
    ln_b: Array,
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl ResamplerFeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln_w: w.require(&format!("{prefix}.0.weight"))?.clone(),
            ln_b: w.require(&format!("{prefix}.0.bias"))?.clone(),
            fc1: AdaptableLinear::dense(w.require(&format!("{prefix}.1.weight"))?.clone(), None),
            fc2: AdaptableLinear::dense(w.require(&format!("{prefix}.3.weight"))?.clone(), None),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let y = layer_norm(x, Some(&self.ln_w), Some(&self.ln_b), LN_EPS)?;
        let y = self.fc1.forward(&y)?;
        let y = gelu_exact(&y)?;
        self.fc2.forward(&y)
    }
}

/// The IP-Adapter "plus" image projection (`image_proj.*`): ViT-H penultimate hidden states →
/// `[B, num_queries, output_dim]` image tokens.
pub struct Resampler {
    /// `[1, num_queries, dim]` learned query latents.
    latents: Array,
    proj_in: AdaptableLinear,  // embed_dim → dim (+bias)
    proj_out: AdaptableLinear, // dim → output_dim (+bias)
    norm_out_w: Array,
    norm_out_b: Array,
    layers: Vec<(PerceiverAttention, ResamplerFeedForward)>,
    dim: i32,
    num_queries: i32,
}

impl Resampler {
    /// Load from the `image_proj` namespace of an IP-Adapter-plus checkpoint.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ResamplerConfig) -> Result<Self> {
        let latents = w.require(&format!("{prefix}.latents"))?.clone();
        let proj_in = AdaptableLinear::dense(
            w.require(&format!("{prefix}.proj_in.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj_in.bias"))?.clone()),
        );
        let proj_out = AdaptableLinear::dense(
            w.require(&format!("{prefix}.proj_out.weight"))?.clone(),
            Some(w.require(&format!("{prefix}.proj_out.bias"))?.clone()),
        );
        let layers = (0..cfg.depth)
            .map(|i| -> Result<_> {
                let attn =
                    PerceiverAttention::from_weights(w, &format!("{prefix}.layers.{i}.0"), cfg)?;
                let ff = ResamplerFeedForward::from_weights(w, &format!("{prefix}.layers.{i}.1"))?;
                Ok((attn, ff))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latents,
            proj_in,
            proj_out,
            norm_out_w: w.require(&format!("{prefix}.norm_out.weight"))?.clone(),
            norm_out_b: w.require(&format!("{prefix}.norm_out.bias"))?.clone(),
            layers,
            dim: cfg.dim,
            num_queries: cfg.num_queries,
        })
    }

    /// `image_features`: ViT-H penultimate `[B, Nx, embed_dim]` → image tokens
    /// `[B, num_queries, output_dim]`.
    pub fn forward(&self, image_features: &Array) -> Result<Array> {
        let b = image_features.shape()[0];
        if self.latents.shape() != [1, self.num_queries, self.dim] {
            return Err(Error::Msg(format!(
                "resampler latents shape {:?} != [1, {}, {}]",
                self.latents.shape(),
                self.num_queries,
                self.dim
            )));
        }
        let mut latents = broadcast_to(&self.latents, &[b, self.num_queries, self.dim])?;
        let x = self.proj_in.forward(image_features)?;
        for (attn, ff) in &self.layers {
            latents = add(&attn.forward(&x, &latents)?, &latents)?;
            latents = add(&ff.forward(&latents)?, &latents)?;
        }
        let out = self.proj_out.forward(&latents)?;
        Ok(layer_norm(
            &out,
            Some(&self.norm_out_w),
            Some(&self.norm_out_b),
            LN_EPS,
        )?)
    }
}
