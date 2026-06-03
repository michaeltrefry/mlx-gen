//! Wan DiT — port of `models/wan/{model.py,transformer.py,attention.py}` (`WanModel`,
//! `WanAttentionBlock`, `WanSelfAttention`, `WanCrossAttention`, `Head`).
//!
//! The backbone is dimension-parametric; this slice targets the dense 5B (dim 3072, 30 layers,
//! 24 heads, head_dim 128, in/out 48). A latent `[C, F, H, W]` is 3-D patchified + linearly embedded,
//! refined by `num_layers` blocks (self-attn with qk-RMSNorm + 3-axis RoPE, cross-attn over the text
//! context, adaLN-6vec modulation, gated-GELU FFN), then projected by the modulated `Head` and
//! unpatchified back to `[out_dim, F, H, W]`.
//!
//! ## Dtype regime — mirrors the reference exactly (the parity contract)
//! Weights load **as stored** (bf16; modulation tables upcast to f32, equivalently). Every
//! projection / attention matmul runs **bf16** (the reference's `x.astype(_linear_dtype)` before
//! each), but the **residual stream and modulation are f32**: the modulation tables + the time
//! embedding `e` are f32, so `norm(x)·(1+e_scale)+e_shift` and `x + gate·y` promote the stream to f32
//! from the first block on (the reference's `autocast(float32)` residual). qk-RMSNorm runs on the
//! bf16 q/k; RoPE applies in f32 on **bf16-precision** cos/sin (the reference's `prepare_rope` builds
//! them in the weight dtype) then casts back to bf16. The patch embedding and the head run
//! f32-promoted matmuls (bf16 weight × f32 activation), as the reference does (no `.astype` there).
//!
//! This is byte-faithful to the production bf16 reference now that **both** NAX 16-bit kernel bugs
//! are patched on the pinned build: the dense GEMM (sc-2714, `matmul.cpp`) and `fast` SDPA (sc-2770,
//! `scaled_dot_product_attention.cpp`) — see [[pmetal-mlx-bf16-matmul-bug]]. (An earlier f32-activation
//! version was a workaround for the then-broken bf16 SDPA; it's obsolete.)

use mlx_gen::array::scalar;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, cos, matmul, multiply, sigmoid, sin, split};
use mlx_rs::{Array, Dtype};

use crate::config::WanModelConfig;
use crate::patchify::{patchify, unpatchify};
use crate::rope::RopeTable;
use crate::text_encoder::gelu_tanh;

/// A `[out, in]` weight + bias (every DiT `nn.Linear` is biased). `forward` is dtype-agnostic — the
/// result dtype follows `x` (bf16 in × bf16 weight → bf16; f32 in → f32-promoted), so callers cast
/// `x` to mirror the reference's explicit `.astype` placement.
struct Linear {
    w: Array,
    b: Array,
}

impl Linear {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(add(&matmul(x, self.w.t())?, &self.b)?)
    }
}

/// SiLU `x·σ(x)` (the reference `nn.SiLU`), bit-exact and dtype-preserving.
fn silu(x: &Array) -> Result<Array> {
    Ok(multiply(x, &sigmoid(x)?)?)
}

fn f32(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

fn bf16(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Bfloat16)?)
}

/// `WanLayerNorm` with `elementwise_affine=False` — `mx.fast.layer_norm(x, None, None, eps)`.
fn ln(x: &Array, eps: f32) -> Result<Array> {
    Ok(layer_norm(x, None, None, eps)?)
}

struct SelfAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Array, // qk-RMSNorm over the full dim
    norm_k: Array,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
}

impl SelfAttention {
    fn load(w: &Weights, prefix: &str, cfg: &WanModelConfig) -> Result<Self> {
        let head_dim = cfg.dim / cfg.num_heads;
        Ok(Self {
            q: Linear::load(w, &format!("{prefix}.q"))?,
            k: Linear::load(w, &format!("{prefix}.k"))?,
            v: Linear::load(w, &format!("{prefix}.v"))?,
            o: Linear::load(w, &format!("{prefix}.o"))?,
            norm_q: w.require(&format!("{prefix}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.norm_k.weight"))?.clone(),
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.eps as f32,
        })
    }

    /// `x_mod`: `[1, L, dim]` (f32). `cos`/`sin`: `[L, 1, half_d]` (bf16). Returns `[1, L, dim]` bf16.
    fn forward(&self, x_mod: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        // Matmuls run bf16 (the reference's `x.astype(w_dtype)`); the f32 residual is restored by the
        // block's modulation. q/k get full-dim bf16 RMSNorm before the head split; RoPE applies in
        // f32 on bf16 cos/sin then casts back to bf16 for the bf16 SDPA.
        let xw = bf16(x_mod)?;
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let s = x_mod.shape()[1];

        let q = rms_norm(&self.q.forward(&xw)?, &self.norm_q, self.eps)?;
        let k = rms_norm(&self.k.forward(&xw)?, &self.norm_k, self.eps)?;
        let q = bf16(&crate::rope::rope_apply(
            &f32(&q.reshape(&[1, s, n, d])?)?,
            cos,
            sin,
        )?)?
        .transpose_axes(&[0, 2, 1, 3])?;
        let k = bf16(&crate::rope::rope_apply(
            &f32(&k.reshape(&[1, s, n, d])?)?,
            cos,
            sin,
        )?)?
        .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&xw)?
            .reshape(&[1, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let out = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, s, n * d])?;
        self.o.forward(&out)
    }
}

struct CrossAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    norm_q: Array,
    norm_k: Array,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    eps: f32,
}

impl CrossAttention {
    fn load(w: &Weights, prefix: &str, cfg: &WanModelConfig) -> Result<Self> {
        let head_dim = cfg.dim / cfg.num_heads;
        Ok(Self {
            q: Linear::load(w, &format!("{prefix}.q"))?,
            k: Linear::load(w, &format!("{prefix}.k"))?,
            v: Linear::load(w, &format!("{prefix}.v"))?,
            o: Linear::load(w, &format!("{prefix}.o"))?,
            norm_q: w.require(&format!("{prefix}.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.norm_k.weight"))?.clone(),
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps: cfg.eps as f32,
        })
    }

    /// Cached K/V from the (bf16) text context `[1, L_ctx, dim]` — computed once, reused per step.
    fn prepare_kv(&self, context: &Array) -> Result<(Array, Array)> {
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let ctx = bf16(context)?;
        let k = rms_norm(&self.k.forward(&ctx)?, &self.norm_k, self.eps)?
            .reshape(&[1, -1, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(&ctx)?
            .reshape(&[1, -1, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        Ok((k, v))
    }

    /// `x`: `[1, L, dim]` (f32). `(k, v)`: cached (bf16). Returns `[1, L, dim]` bf16.
    fn forward(&self, x: &Array, kv: &(Array, Array)) -> Result<Array> {
        let (n, d) = (self.num_heads as i32, self.head_dim as i32);
        let s = x.shape()[1];
        let q = rms_norm(&self.q.forward(&bf16(x)?)?, &self.norm_q, self.eps)?
            .reshape(&[1, s, n, d])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let out = scaled_dot_product_attention(&q, &kv.0, &kv.1, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, s, n * d])?;
        self.o.forward(&out)
    }
}

struct Block {
    modulation: Array, // [1, 6, dim]
    self_attn: SelfAttention,
    cross_attn: CrossAttention,
    norm3_w: Array, // cross-attn norm (affine LayerNorm)
    norm3_b: Array,
    ffn_fc1: Linear,
    ffn_fc2: Linear,
    eps: f32,
}

impl Block {
    fn load(w: &Weights, i: usize, cfg: &WanModelConfig) -> Result<Self> {
        let p = format!("blocks.{i}");
        Ok(Self {
            modulation: f32(w.require(&format!("{p}.modulation"))?)?,
            self_attn: SelfAttention::load(w, &format!("{p}.self_attn"), cfg)?,
            cross_attn: CrossAttention::load(w, &format!("{p}.cross_attn"), cfg)?,
            norm3_w: f32(w.require(&format!("{p}.norm3.weight"))?)?,
            norm3_b: f32(w.require(&format!("{p}.norm3.bias"))?)?,
            ffn_fc1: Linear::load(w, &format!("{p}.ffn.fc1"))?,
            ffn_fc2: Linear::load(w, &format!("{p}.ffn.fc2"))?,
            eps: cfg.eps as f32,
        })
    }

    fn prepare_kv(&self, context: &Array) -> Result<(Array, Array)> {
        self.cross_attn.prepare_kv(context)
    }

    /// `x`: `[1, L, dim]` (f32). `e`: `[1, 1, 6, dim]` f32 time modulation.
    fn forward(
        &self,
        x: &Array,
        e: &Array,
        kv: &(Array, Array),
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        // adaLN-6vec modulation, f32 (promotes the residual stream). [1,6,dim] + [1,1,6,dim] →
        // [1,1,6,dim], split into 6 × [1,1,dim] (shift/scale/gate for self-attn, then FFN).
        let dim = self.self_attn.num_heads as i32 * self.self_attn.head_dim as i32;
        let m = add(&self.modulation, e)?; // [1, 1, 6, dim] f32
        let p = split(&m, 6, 2)?;
        let v = |i: usize| -> Result<Array> { Ok(p[i].reshape(&[1, 1, dim])?) };
        let (e0, e1, e2) = (v(0)?, v(1)?, v(2)?);
        let (e3, e4, e5) = (v(3)?, v(4)?, v(5)?);

        // Self-attention.
        let x_mod = add(&multiply(&ln(x, self.eps)?, &add(&e1, scalar(1.0))?)?, &e0)?;
        let y = self.self_attn.forward(&x_mod, cos, sin)?;
        let x = add(x, &multiply(&y, &e2)?)?;

        // Cross-attention (affine LayerNorm on context-side query, no modulation).
        let x_cross = layer_norm(&x, Some(&self.norm3_w), Some(&self.norm3_b), self.eps)?;
        let x = add(&x, &self.cross_attn.forward(&x_cross, kv)?)?;

        // Gated-GELU FFN (bf16 matmuls; the reference's `x.astype(w_dtype)`).
        let x_mod = add(&multiply(&ln(&x, self.eps)?, &add(&e4, scalar(1.0))?)?, &e3)?;
        let y = gelu_tanh(&self.ffn_fc1.forward(&bf16(&x_mod)?)?)?;
        let y = self.ffn_fc2.forward(&y)?;
        Ok(add(&x, &multiply(&y, &e5)?)?)
    }
}

/// The Wan DiT (5B dense T2V). Holds the loaded weights + the precomputed RoPE table.
pub struct WanTransformer {
    patch_embedding: Linear,
    text_embedding_0: Linear,
    text_embedding_1: Linear,
    time_embedding_0: Linear,
    time_embedding_1: Linear,
    time_projection: Linear,
    blocks: Vec<Block>,
    head_modulation: Array, // [1, 2, dim]
    head: Linear,
    rope: RopeTable,
    inv_freq: Array, // [freq_dim/2] f32, for the sinusoidal time embedding
    cfg: WanModelConfig,
}

impl WanTransformer {
    pub fn from_weights(w: &Weights, cfg: &WanModelConfig) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            blocks.push(Block::load(w, i, cfg)?);
        }
        let half = cfg.freq_dim / 2;
        let inv: Vec<f32> = (0..half)
            .map(|j| (10000.0_f64.powf(-(j as f64) / half as f64)) as f32)
            .collect();
        Ok(Self {
            patch_embedding: Linear::load(w, "patch_embedding_proj")?,
            text_embedding_0: Linear::load(w, "text_embedding_0")?,
            text_embedding_1: Linear::load(w, "text_embedding_1")?,
            time_embedding_0: Linear::load(w, "time_embedding_0")?,
            time_embedding_1: Linear::load(w, "time_embedding_1")?,
            time_projection: Linear::load(w, "time_projection")?,
            blocks,
            head_modulation: f32(w.require("head.modulation")?)?,
            head: Linear::load(w, "head.head")?,
            rope: RopeTable::new(cfg.dim / cfg.num_heads),
            inv_freq: Array::from_slice(&inv, &[half as i32]),
            cfg: cfg.clone(),
        })
    }

    /// Embed a single T5 prompt embedding `[L_text, text_dim]` → `[1, text_len, dim]` (bf16),
    /// zero-padded to `text_len`. Mirrors `WanModel.embed_text` for one prompt.
    pub fn embed_text(&self, t5_embed: &Array) -> Result<Array> {
        let text_len = self.cfg.text_len as i32;
        let l = t5_embed.shape()[0];
        let dim_text = t5_embed.shape()[1];
        let ctx = if l < text_len {
            let pad = Array::zeros::<f32>(&[text_len - l, dim_text])?.as_dtype(t5_embed.dtype())?;
            concatenate_axis(&[t5_embed, &pad], 0)?
        } else {
            t5_embed.clone()
        };
        let ctx = ctx.reshape(&[1, text_len, dim_text])?;
        let h = self.text_embedding_0.forward(&ctx)?;
        let h = gelu_tanh(&h)?;
        // Cast to bf16 (the reference's `.astype(model_dtype)`); the cross-attn K/V run bf16.
        bf16(&self.text_embedding_1.forward(&h)?)
    }

    /// Sinusoidal time embedding `e` `[1, dim]` (f32) + the 6-vector modulation `e0` `[1,1,6,dim]`.
    fn time_embed(&self, t: f32) -> Result<(Array, Array)> {
        let pos = Array::from_slice(&[t], &[1, 1]); // [1, 1]
        let sinusoid = multiply(&pos, &self.inv_freq)?; // [1, half] f32
        let sin_emb = concatenate_axis(&[&cos(&sinusoid)?, &sin(&sinusoid)?], 1)?; // [1, freq_dim]
        let e = self
            .time_embedding_1
            .forward(&silu(&self.time_embedding_0.forward(&sin_emb)?)?)?; // [1, dim] f32
        let e0 =
            self.time_projection
                .forward(&silu(&e)?)?
                .reshape(&[1, 1, 6, self.cfg.dim as i32])?; // [1,1,6,dim] f32
        Ok((e, e0))
    }

    /// Output head: modulated LayerNorm + projection → `[1, L, out_dim·∏patch]` (f32).
    fn apply_head(&self, x: &Array, e: &Array) -> Result<Array> {
        let dim = self.cfg.dim as i32;
        // head.modulation [1,2,dim] + e [1,1,1,dim] → [1,1,2,dim], split into shift e0 / scale e1.
        let m = add(&self.head_modulation, &e.reshape(&[1, 1, 1, dim])?)?;
        let p = split(&m, 2, 2)?;
        let e0 = p[0].reshape(&[1, 1, dim])?;
        let e1 = p[1].reshape(&[1, 1, dim])?;
        let x_mod = add(
            &multiply(&ln(x, self.cfg.eps as f32)?, &add(&e1, scalar(1.0))?)?,
            &e0,
        )?;
        self.head.forward(&x_mod)
    }

    /// Per-stage capture for parity bisection: `(x_embed, e_time, e0_mod, x_block0, x_blocks,
    /// x_head)`. Mirrors [`forward`](Self::forward) but returns the intermediate hiddens.
    pub fn forward_capture(
        &self,
        latent: &Array,
        t: f32,
        context_embed: &Array,
    ) -> Result<Vec<Array>> {
        let (tokens, grid) = patchify(latent, self.cfg.patch_size)?;
        let l = (grid.0 * grid.1 * grid.2) as i32;
        let x_embed =
            bf16(&self.patch_embedding.forward(&tokens)?)?.reshape(&[1, l, self.cfg.dim as i32])?;
        let (e, e0) = self.time_embed(t)?;
        // bf16 cos/sin (the reference's `prepare_rope` builds them in the weight dtype = bf16).
        let (cos_t, sin_t) = self.rope.precompute_cos_sin(grid)?;
        let (cos_t, sin_t) = (bf16(&cos_t)?, bf16(&sin_t)?);
        let mut x = x_embed.clone();
        let mut x_block0 = x_embed.clone();
        for (i, block) in self.blocks.iter().enumerate() {
            let kv = block.prepare_kv(context_embed)?;
            x = block.forward(&x, &e0, &kv, &cos_t, &sin_t)?;
            if i == 0 {
                x_block0 = x.clone();
            }
        }
        let x_head = self.apply_head(&x, &e)?;
        Ok(vec![x_embed, e, e0, x_block0, x, x_head])
    }

    /// Full DiT forward for a single latent (B=1, the cfg-disabled path). `latent`: `[C, F, H, W]`
    /// (f32). `t`: integer-valued timestep. `context_embed`: `[1, text_len, dim]` (bf16) from
    /// [`embed_text`](Self::embed_text). Returns the denoised `[out_dim, F, H, W]` (f32).
    ///
    /// The CFG B=2 path (S4) calls this once per branch — bit-identical to the reference's batched
    /// forward, since attention/matmuls never mix batch elements.
    pub fn forward(&self, latent: &Array, t: f32, context_embed: &Array) -> Result<Array> {
        // Patchify + embed; cast to bf16 to start the block stream (reference casts patches to w_dtype).
        let (tokens, grid) = patchify(latent, self.cfg.patch_size)?;
        let mut x = bf16(&self.patch_embedding.forward(&tokens)?)?.reshape(&[
            1,
            (grid.0 * grid.1 * grid.2) as i32,
            self.cfg.dim as i32,
        ])?;

        let (e, e0) = self.time_embed(t)?;
        let (cos_t, sin_t) = self.rope.precompute_cos_sin(grid)?;
        let (cos_t, sin_t) = (bf16(&cos_t)?, bf16(&sin_t)?);

        for block in &self.blocks {
            let kv = block.prepare_kv(context_embed)?;
            x = block.forward(&x, &e0, &kv, &cos_t, &sin_t)?;
        }

        let x = self.apply_head(&x, &e)?; // [1, L, out_dim·∏patch] f32
        let l = x.shape()[1];
        let x = x.reshape(&[l, x.shape()[2]])?;
        unpatchify(&x, grid, self.cfg.out_dim, self.cfg.patch_size)
    }
}
