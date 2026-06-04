//! UNet spatial transformer: `Transformer2D` (GroupNorm → linear `proj_in` → N `TransformerBlock`s
//! → linear `proj_out`, residual) and its `TransformerBlock` (self-attn → cross-attn → GEGLU FFN).
//! Port of the vendored `unet.Transformer2D` / `TransformerBlock`. SDXL uses linear `proj_in/out`
//! (`use_linear_projection`), no attention masks, and exact `gelu` in the GEGLU. NHWC I/O.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::{gelu_exact, group_norm};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5;

/// Multi-head attention as the vendored `nn.MultiHeadAttention`: q/k/v projections without bias,
/// output projection with bias, no mask. Used for both self-attention (context = `x`) and
/// cross-attention (context = the text `memory`).
struct AttentionMHA {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl AttentionMHA {
    fn from_weights(w: &Weights, prefix: &str, model_dims: i32, num_heads: i32) -> Result<Self> {
        let no_bias = |n: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{n}.weight"))?.clone(),
                None,
            ))
        };
        let head_dim = model_dims / num_heads;
        Ok(Self {
            q: no_bias("to_q")?,
            k: no_bias("to_k")?,
            v: no_bias("to_v")?,
            out: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_out.0.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.to_out.0.bias"))?.clone()),
            ),
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.q, &mut self.k, &mut self.v, &mut self.out] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// `x`: `[B, L, D]` (queries); `context`: `[B, S, Dctx]` (keys/values; == `x` for self-attn).
    /// Fused `scaled_dot_product_attention` (mathematically the reference's `nn.MultiHeadAttention`;
    /// an explicit softmax matmul was tried and gave no measurable parity gain at large e2e cost).
    /// The four LoRA-targetable attention projections, by diffusers leaf name (the `.to_out.0`
    /// dot is the GEGLU-style indexed leaf the kohya flattener turns into `to_out_0`).
    fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        for leaf in ["to_q", "to_k", "to_v", "to_out.0"] {
            out.push(format!("{prefix}.{leaf}"));
        }
    }

    fn forward(&self, x: &Array, context: &Array) -> Result<Array> {
        let (b, l) = (x.shape()[0], x.shape()[1]);
        let s = context.shape()[1];
        let q = self
            .q
            .forward(x)?
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k
            .forward(context)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v
            .forward(context)?
            .reshape(&[b, s, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, l, self.num_heads * self.head_dim])?;
        self.out.forward(&o)
    }
}

/// One spatial-transformer block: pre-norm self-attn, pre-norm cross-attn to the text memory, and a
/// pre-norm GEGLU FFN (`linear1(y) * gelu(linear2(y)) → linear3`). All residual.
struct TransformerBlock {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    norm3_w: Array,
    norm3_b: Array,
    attn1: AttentionMHA,
    attn2: AttentionMHA,
    /// GEGLU value half (`ff.net.0.proj` rows `[0:hidden]`).
    linear1: AdaptableLinear,
    /// GEGLU gate half (`ff.net.0.proj` rows `[hidden:2*hidden]`).
    linear2: AdaptableLinear,
    /// FFN output (`ff.net.2`).
    linear3: AdaptableLinear,
}

impl TransformerBlock {
    fn from_weights(w: &Weights, prefix: &str, model_dims: i32, num_heads: i32) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}")).cloned();
        // GEGLU: split `ff.net.0.proj` (weight [2*hidden, D], bias [2*hidden]) into value/gate halves.
        let proj_w = g("ff.net.0.proj.weight")?;
        let proj_b = g("ff.net.0.proj.bias")?;
        let two_h = proj_w.shape()[0];
        let hidden = two_h / 2;
        let split_row = |a: &Array, lo: i32, hi: i32| -> Result<Array> {
            let idx = Array::from_slice(&(lo..hi).collect::<Vec<i32>>(), &[hi - lo]);
            Ok(a.take_axis(&idx, 0)?)
        };
        let linear1 = AdaptableLinear::dense(
            split_row(&proj_w, 0, hidden)?,
            Some(split_row(&proj_b, 0, hidden)?),
        );
        let linear2 = AdaptableLinear::dense(
            split_row(&proj_w, hidden, two_h)?,
            Some(split_row(&proj_b, hidden, two_h)?),
        );
        let linear3 = AdaptableLinear::dense(g("ff.net.2.weight")?, Some(g("ff.net.2.bias")?));
        Ok(Self {
            norm1_w: g("norm1.weight")?,
            norm1_b: g("norm1.bias")?,
            norm2_w: g("norm2.weight")?,
            norm2_b: g("norm2.bias")?,
            norm3_w: g("norm3.weight")?,
            norm3_b: g("norm3.bias")?,
            attn1: AttentionMHA::from_weights(
                w,
                &format!("{prefix}.attn1"),
                model_dims,
                num_heads,
            )?,
            attn2: AttentionMHA::from_weights(
                w,
                &format!("{prefix}.attn2"),
                model_dims,
                num_heads,
            )?,
            linear1,
            linear2,
            linear3,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn1.quantize(bits)?;
        self.attn2.quantize(bits)?;
        self.linear1.quantize(bits, None)?;
        self.linear2.quantize(bits, None)?;
        self.linear3.quantize(bits, None)?;
        Ok(())
    }

    fn forward(&self, x: &Array, memory: &Array) -> Result<Array> {
        // Self-attention.
        let y = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), LN_EPS)?;
        let x = add(x, &self.attn1.forward(&y, &y)?)?;
        // Cross-attention to the text memory.
        let y = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), LN_EPS)?;
        let x = add(&x, &self.attn2.forward(&y, memory)?)?;
        // GEGLU FFN.
        let y = layer_norm(&x, Some(&self.norm3_w), Some(&self.norm3_b), LN_EPS)?;
        let y = multiply(
            &self.linear1.forward(&y)?,
            &gelu_exact(&self.linear2.forward(&y)?)?,
        )?;
        let y = self.linear3.forward(&y)?;
        Ok(add(&x, &y)?)
    }
}

/// A 2-D spatial transformer over NHWC features, cross-attending to the text `encoder_x`.
pub struct Transformer2D {
    norm_w: Array,
    norm_b: Array,
    proj_in: AdaptableLinear,
    blocks: Vec<TransformerBlock>,
    proj_out: AdaptableLinear,
}

impl Transformer2D {
    /// `prefix` addresses the `attentions.{i}` module. `num_layers` = transformer blocks.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        model_dims: i32,
        num_heads: i32,
        num_layers: i32,
    ) -> Result<Self> {
        let blocks = (0..num_layers)
            .map(|i| {
                TransformerBlock::from_weights(
                    w,
                    &format!("{prefix}.transformer_blocks.{i}"),
                    model_dims,
                    num_heads,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            norm_w: w.require(&format!("{prefix}.norm.weight"))?.clone(),
            norm_b: w.require(&format!("{prefix}.norm.bias"))?.clone(),
            proj_in: AdaptableLinear::dense(
                w.require(&format!("{prefix}.proj_in.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.proj_in.bias"))?.clone()),
            ),
            blocks,
            proj_out: AdaptableLinear::dense(
                w.require(&format!("{prefix}.proj_out.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.proj_out.bias"))?.clone()),
            ),
        })
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.proj_in.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        for b in &mut self.blocks {
            b.quantize(bits)?;
        }
        Ok(())
    }

    /// `x`: NHWC `[B, H, W, C]`; `encoder_x`: text memory `[B, S, Dctx]`.
    pub fn forward(&self, x: &Array, encoder_x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w_, c) = (sh[0], sh[1], sh[2], sh[3]);
        let y = group_norm(x, &self.norm_w, &self.norm_b, GN_GROUPS, GN_EPS)?;
        let mut y = self.proj_in.forward(&y.reshape(&[b, h * w_, c])?)?;
        for block in &self.blocks {
            y = block.forward(&y, encoder_x)?;
        }
        let y = self.proj_out.forward(&y)?.reshape(&[b, h, w_, c])?;
        Ok(add(&y, x)?)
    }

    /// LoRA-targetable Linears under this `attentions.{i}` module, by diffusers path: `proj_in`,
    /// `proj_out`, and each transformer block's attention projections. The GEGLU FF (`linear1/2/3`)
    /// is intentionally excluded — the vendored `lora.py` can't reach it (mlx-examples renames it),
    /// so faithfully porting that path omits it too (sc-2671 adds it).
    pub fn lora_target_paths(&self, prefix: &str, out: &mut Vec<String>) {
        out.push(format!("{prefix}.proj_in"));
        out.push(format!("{prefix}.proj_out"));
        for (k, b) in self.blocks.iter().enumerate() {
            b.attn1
                .lora_target_paths(&format!("{prefix}.transformer_blocks.{k}.attn1"), out);
            b.attn2
                .lora_target_paths(&format!("{prefix}.transformer_blocks.{k}.attn2"), out);
        }
    }

    /// The GEGLU feed-forward LoRA targets (diffusers naming) under this `attentions.{i}` module:
    /// each block's `ff.net.0.proj` (the fused value+gate proj, row-split across `linear1`/`linear2`
    /// at merge) and `ff.net.2`. Kept separate from [`Transformer2D::lora_target_paths`] because the
    /// vendored `lora.py` can't reach the FF (mlx-examples renames it) — complete coverage (sc-2671)
    /// adds it on top of the faithful surface.
    pub fn lora_target_paths_ff(&self, prefix: &str, out: &mut Vec<String>) {
        for k in 0..self.blocks.len() {
            out.push(format!("{prefix}.transformer_blocks.{k}.ff.net.0.proj"));
            out.push(format!("{prefix}.transformer_blocks.{k}.ff.net.2"));
        }
    }
}

// LoRA key→module routing (sc-2639). Diffusers leaf naming; the GEGLU FF and (for the U-Net)
// mid_block are intentionally unreachable here to mirror the vendored `lora.py` surface.
impl AdaptableHost for AttentionMHA {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(&mut self.q),
            ["to_k"] => Some(&mut self.k),
            ["to_v"] => Some(&mut self.v),
            ["to_out", "0"] => Some(&mut self.out),
            _ => None,
        }
    }
}

impl AdaptableHost for TransformerBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            // GEGLU FF (sc-2671 complete coverage). The diffusers `ff.net.0.proj` is row-split into
            // `linear1` (value) + `linear2` (gate) and `ff.net.2` is `linear3`; the SDXL adapter
            // merge translates those diffusers FF keys into these internal `ff.linearN` names (and
            // row-splits a `ff.net.0.proj` delta across linear1/linear2). Unreachable under the
            // vendored coverage (the merge gates FF keys out there).
            ["ff", "linear1"] => Some(&mut self.linear1),
            ["ff", "linear2"] => Some(&mut self.linear2),
            ["ff", "linear3"] => Some(&mut self.linear3),
            _ => None,
        }
    }
}

impl AdaptableHost for Transformer2D {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["proj_in"] => Some(&mut self.proj_in),
            ["proj_out"] => Some(&mut self.proj_out),
            ["transformer_blocks", k, rest @ ..] => self
                .blocks
                .get_mut(k.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }
}
