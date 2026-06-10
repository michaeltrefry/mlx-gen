//! IDFormer (sc-3071) — the PuLID perceiver-resampler that fuses the ArcFace embedding + the EVA
//! visual features into the 32-token `id_embedding` the FLUX cross-attn (sc-3072) injects. Port of
//! `pulid/encoders_transformer.py IDFormer`.
//!
//! Structure (dim=1024, depth=10, heads=16, dim_head=64, 5 id tokens, 32 queries, out 2048):
//!   * `latents` [1,32,1024] learned queries; `proj_out` [1024,2048] learned param (raw matmul).
//!   * `id_embedding_mapping`: 1280→1024→(LN,LeakyReLU)→1024→(LN,LeakyReLU)→1024×5  (the 5 id tokens).
//!   * `mapping_0..4`: 1024→1024→(LN,LeakyReLU)→1024→(LN,LeakyReLU)→1024  (projects each EVA scale).
//!   * 10 × (PerceiverAttention + FeedForward), grouped 5 scales × 2 layers.
//!
//! All LayerNorms are `nn.LayerNorm` default **ε=1e-5** (distinct from the EVA tower's 1e-6). The
//! perceiver attention factors the scale as `(q·s)@(k·s)`, s=dim_head^-0.25 (≡ scale dim_head^-0.5)
//! with softmax in f32 — reproduced by MLX SDPA(scale=dim_head^-0.5).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{broadcast_to, concatenate_axis, matmul, maximum, multiply};
use mlx_rs::Array;

use mlx_gen::array::scalar;
use mlx_gen::nn::{gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// nn.LayerNorm default epsilon (the IDFormer/PuLID modules; NOT the EVA 1e-6).
const EPS: f32 = 1e-5;
/// nn.LeakyReLU default negative slope.
const LEAKY: f32 = 0.01;

fn join(p: &str, leaf: &str) -> String {
    format!("{p}.{leaf}")
}

fn leaky_relu(x: &Array) -> Result<Array> {
    Ok(maximum(
        x,
        &multiply(x, &scalar(LEAKY).as_dtype(x.dtype())?)?,
    )?)
}

/// The shared `Linear→LN→LeakyReLU→Linear→LN→LeakyReLU→Linear` mapping head (Sequential indices
/// 0,1,3,4,6 in the checkpoint). All three linears are biased.
struct MappingMlp {
    l0_w: Array,
    l0_b: Array,
    ln1_w: Array,
    ln1_b: Array,
    l3_w: Array,
    l3_b: Array,
    ln4_w: Array,
    ln4_b: Array,
    l6_w: Array,
    l6_b: Array,
}

impl MappingMlp {
    fn from_weights(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            l0_w: w.require(&join(p, "0.weight"))?.clone(),
            l0_b: w.require(&join(p, "0.bias"))?.clone(),
            ln1_w: w.require(&join(p, "1.weight"))?.clone(),
            ln1_b: w.require(&join(p, "1.bias"))?.clone(),
            l3_w: w.require(&join(p, "3.weight"))?.clone(),
            l3_b: w.require(&join(p, "3.bias"))?.clone(),
            ln4_w: w.require(&join(p, "4.weight"))?.clone(),
            ln4_b: w.require(&join(p, "4.bias"))?.clone(),
            l6_w: w.require(&join(p, "6.weight"))?.clone(),
            l6_b: w.require(&join(p, "6.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = linear(x, &self.l0_w, &self.l0_b)?;
        let h = layer_norm(&h, Some(&self.ln1_w), Some(&self.ln1_b), EPS)?;
        let h = leaky_relu(&h)?;
        let h = linear(&h, &self.l3_w, &self.l3_b)?;
        let h = layer_norm(&h, Some(&self.ln4_w), Some(&self.ln4_b), EPS)?;
        let h = leaky_relu(&h)?;
        linear(&h, &self.l6_w, &self.l6_b)
    }
}

/// PerceiverAttention: q from `latents`, k/v from `cat(ctx, latents)`. bias-free linears.
///
/// **Shares its structure with [`crate::ca`]'s `PerceiverAttentionCA`** (same `norm1`/`norm2` +
/// `to_q`/`to_kv`/`to_out` weight keys, head split, `EPS = 1e-5`, SDPA `scale = dim_head^-0.5`, and
/// out-projection). They are kept as separate 1:1 mirrors of the two distinct upstream Python classes
/// (F-085). The **only** behavioral difference is the k/v source: here k/v come from
/// `cat(ctx, latents)`, whereas the CA variant uses `x` (the id_embedding) alone. Any change to the
/// shared plumbing — EPS, the SDPA scale, the bias-free linears, a future quantization path — must be
/// applied to **both** to preserve parity.
struct PerceiverAttention {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    to_q: Array,
    to_kv: Array,
    to_out: Array,
    heads: i32,
    dim_head: i32,
}

impl PerceiverAttention {
    fn from_weights(w: &Weights, p: &str, heads: i32, dim_head: i32) -> Result<Self> {
        Ok(Self {
            norm1_w: w.require(&join(p, "norm1.weight"))?.clone(),
            norm1_b: w.require(&join(p, "norm1.bias"))?.clone(),
            norm2_w: w.require(&join(p, "norm2.weight"))?.clone(),
            norm2_b: w.require(&join(p, "norm2.bias"))?.clone(),
            to_q: w.require(&join(p, "to_q.weight"))?.clone(),
            to_kv: w.require(&join(p, "to_kv.weight"))?.clone(),
            to_out: w.require(&join(p, "to_out.weight"))?.clone(),
            heads,
            dim_head,
        })
    }

    /// `ctx`: `[B, n_ctx, dim]` (image/id features); `latents`: `[B, n_lat, dim]` (queries).
    fn forward(&self, ctx: &Array, latents: &Array) -> Result<Array> {
        let x = layer_norm(ctx, Some(&self.norm1_w), Some(&self.norm1_b), EPS)?;
        let lat = layer_norm(latents, Some(&self.norm2_w), Some(&self.norm2_b), EPS)?;
        let (b, n_lat) = (lat.shape()[0], lat.shape()[1]);
        let (h, hd) = (self.heads, self.dim_head);

        let q = matmul(&lat, self.to_q.t())?;
        let kv = matmul(&concatenate_axis(&[&x, &lat], 1)?, self.to_kv.t())?;
        let n_kv = kv.shape()[1];
        let kv_parts = mlx_rs::ops::split(&kv, 2, 2)?; // along last (dim) axis: [k | v]
        let to_heads = |t: &Array, n: i32| -> Result<Array> {
            Ok(t.reshape(&[b, n, h, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(&q, n_lat)?;
        let k = to_heads(&kv_parts[0], n_kv)?;
        let v = to_heads(&kv_parts[1], n_kv)?;

        let scale = (hd as f32).powf(-0.5);
        let attn = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let out = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, n_lat, h * hd])?;
        Ok(matmul(&out, self.to_out.t())?)
    }
}

/// FeedForward: `LN → Linear(no bias) → GELU(exact) → Linear(no bias)` (Sequential 0,1,3).
struct FeedForward {
    ln_w: Array,
    ln_b: Array,
    l1: Array,
    l3: Array,
}

impl FeedForward {
    fn from_weights(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            ln_w: w.require(&join(p, "0.weight"))?.clone(),
            ln_b: w.require(&join(p, "0.bias"))?.clone(),
            l1: w.require(&join(p, "1.weight"))?.clone(),
            l3: w.require(&join(p, "3.weight"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = layer_norm(x, Some(&self.ln_w), Some(&self.ln_b), EPS)?;
        let h = matmul(&h, self.l1.t())?;
        let h = gelu_exact(&h)?;
        Ok(matmul(&h, self.l3.t())?)
    }
}

#[derive(Clone, Debug)]
pub struct IdFormerConfig {
    pub dim: i32,
    pub depth: i32,
    pub heads: i32,
    pub dim_head: i32,
    pub num_id_token: i32,
    pub num_queries: i32,
    pub output_dim: i32,
}

impl Default for IdFormerConfig {
    fn default() -> Self {
        Self {
            dim: 1024,
            depth: 10,
            heads: 16,
            dim_head: 64,
            num_id_token: 5,
            num_queries: 32,
            output_dim: 2048,
        }
    }
}

pub struct IdFormer {
    latents: Array,  // [1, num_queries, dim]
    proj_out: Array, // [dim, output_dim]
    id_embedding_mapping: MappingMlp,
    mapping: Vec<MappingMlp>, // 5
    layers: Vec<(PerceiverAttention, FeedForward)>,
    per_scale: i32, // depth / 5
    cfg: IdFormerConfig,
}

impl IdFormer {
    /// `prefix` is the top-level module name (`"pulid_encoder"`).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: IdFormerConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let mapping = (0..5)
            .map(|i| MappingMlp::from_weights(w, &p(&format!("mapping_{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let layers = (0..cfg.depth)
            .map(|i| {
                Ok((
                    PerceiverAttention::from_weights(
                        w,
                        &p(&format!("layers.{i}.0")),
                        cfg.heads,
                        cfg.dim_head,
                    )?,
                    FeedForward::from_weights(w, &p(&format!("layers.{i}.1")))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            latents: w.require(&p("latents"))?.clone(),
            proj_out: w.require(&p("proj_out"))?.clone(),
            id_embedding_mapping: MappingMlp::from_weights(w, &p("id_embedding_mapping"))?,
            mapping,
            layers,
            per_scale: cfg.depth / 5,
            cfg,
        })
    }

    /// `id_cond`: `[B, 1280]` (cat of ArcFace 512 + id_cond_vit 768).
    /// `id_vit_hidden`: 5 × `[B, 577, 1024]` (the EVA hidden states).
    /// Returns `id_embedding` `[B, num_queries, output_dim]` (32×2048).
    pub fn forward(&self, id_cond: &Array, id_vit_hidden: &[Array]) -> Result<Array> {
        // `pub` runtime path: report the EVA-capture-count invariant via `Result`, not a panic that
        // would cross the `Generator::generate` boundary for a future caller with a different EVA
        // capture schedule (F-080).
        if id_vit_hidden.len() != 5 {
            return Err(Error::Msg(format!(
                "IDFormer expects 5 EVA hidden states, got {}",
                id_vit_hidden.len()
            )));
        }
        let b = id_cond.shape()[0];
        let dim = self.cfg.dim;

        let mut latents = broadcast_to(&self.latents, &[b, self.cfg.num_queries, dim])?;
        // id tokens: 1280 -> 1024*5 -> [B, 5, 1024]
        let x = self.id_embedding_mapping.forward(id_cond)?.reshape(&[
            b,
            self.cfg.num_id_token,
            dim,
        ])?;
        latents = concatenate_axis(&[&latents, &x], 1)?; // [B, 37, 1024]

        for i in 0..5 {
            let vit = self.mapping[i as usize].forward(&id_vit_hidden[i as usize])?;
            let ctx = concatenate_axis(&[&x, &vit], 1)?; // [B, 5+577, 1024]
            for l in (i * self.per_scale)..((i + 1) * self.per_scale) {
                let (attn, ff) = &self.layers[l as usize];
                latents = mlx_rs::ops::add(&latents, &attn.forward(&ctx, &latents)?)?;
                latents = mlx_rs::ops::add(&latents, &ff.forward(&latents)?)?;
            }
        }

        // take the query tokens, project to output_dim via the raw param matmul
        let idx = Array::from_slice(
            &(0..self.cfg.num_queries).collect::<Vec<i32>>(),
            &[self.cfg.num_queries],
        );
        let q = latents.take_axis(&idx, 1)?; // [B, 32, 1024]
        Ok(matmul(&q, &self.proj_out)?)
    }
}
