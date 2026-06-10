//! ChatGLM3-6B forward — Kolors' text encoder (sc-3091). **Encoder-only**: no LM head, no
//! generation, no KV cache. Faithful port of the diffusers `KolorsPipeline` reference
//! (`diffusers/pipelines/kolors/text_encoder.py`, `ChatGLMModel`), driven by `encode_prompt`
//! with `output_hidden_states=True`.
//!
//! Patterned on [`mlx_gen_ltx::gemma`] (Config / Quant / Linear-enum / Layer / Model, a
//! `forward → Vec<Array>` of hidden states), with the ChatGLM3-specific pieces:
//!
//!  - **Half-dim interleaved RoPE.** Rotary applies to the **first `rotary_dim` (64)** of the
//!    128-wide head dim, with **adjacent-pair** interleaving `(x[2i], x[2i+1])` (NOT the HF
//!    half-split); the trailing 64 dims pass through unrotated. θ = 10000, constant across layers.
//!  - **Fused, biased `query_key_value`.** One `[4608, 4096]` Linear (with bias) →
//!    q (32·128) + k (2·128) + v (2·128); **multi-query attention, 2 KV groups** broadcast to 32
//!    query heads by the GQA-aware SDPA. The output `dense` proj is bias-less.
//!  - **RMSNorm = plain `weight · x̂`** (eps 1e-5) — NOT Gemma's `(1 + weight)`.
//!  - **GLMBlock pre-norm residual**: `h = x + dense(attn(input_ln(x)))`;
//!    `out = h + mlp(post_attn_ln(h))`. MLP = `dense_4h_to_h(silu(g) · u)` where `dense_h_to_4h`
//!    fuses gate+up (out `2·13696`); the activation is **SiLU** (`torch.chunk(·,2)[0]` gated).
//!  - **`apply_query_key_layer_scaling` cancels out** on the torch≥2 SDPA path the reference takes,
//!    leaving the standard `1/√head_dim` scale + plain softmax (the coeff multiply only exists in the
//!    legacy eager fallback, which does not run). So attention is ordinary scaled-dot-product.
//!
//! ### Output contract (what Kolors consumes)
//! [`forward`](ChatGlmModel::forward) returns the **`num_layers + 1` (29)** hidden states exactly as
//! the reference `all_hidden_states`: `[embedding] + [output of layer 0 .. layer 27]`, the last entry
//! taken **before** `final_layernorm`. Each is `[B, S, hidden]` (the reference's `[S, B, hidden]`
//! transposed). Kolors uses **`hidden_states[-2]`** (penultimate, layer-26 output) as the context and
//! **`hidden_states[-1]` at the last sequence position** as the pooled embedding — neither is
//! final-normed. [`encode_prompt`](ChatGlmModel::encode_prompt) extracts both. `final_layernorm` is
//! loaded and applied to expose the conventional `last_hidden_state`, but Kolors does not use it.

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::module::Param;
use mlx_rs::nn::{Linear, QuantizedLinear};
use mlx_rs::ops::{
    add, addmm, concatenate_axis, cos as cos_op, dequantize, matmul, multiply, quantized_matmul,
    sin as sin_op, split, subtract,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::silu;
use mlx_gen::quant::DEFAULT_GROUP_SIZE;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

/// ChatGLM3-6B text config (the Kolors `text_encoder/config.json` values).
#[derive(Clone, Copy, Debug)]
pub struct ChatGlmConfig {
    pub hidden_size: i32,
    pub num_layers: usize,
    /// Query heads (32).
    pub num_heads: i32,
    /// Multi-query KV groups (2). Broadcast to `num_heads` by the GQA-aware SDPA.
    pub num_kv_groups: i32,
    /// Per-head dim (`kv_channels` = 128).
    pub head_dim: i32,
    /// FFN inner width (13696). `dense_h_to_4h` emits `2 ·` this (fused gate+up).
    pub ffn_hidden: i32,
    pub rms_eps: f32,
    /// RoPE base θ (10000).
    pub rope_base: f32,
    /// Rotated head-dim prefix (`kv_channels / 2` = 64); the remaining dims pass through.
    pub rotary_dim: i32,
    pub vocab_size: i32,
}

impl ChatGlmConfig {
    /// The Kolors ChatGLM3-6B values.
    pub fn chatglm3_6b() -> Self {
        Self {
            hidden_size: 4096,
            num_layers: 28,
            num_heads: 32,
            num_kv_groups: 2,
            head_dim: 128,
            ffn_hidden: 13696,
            rms_eps: 1e-5,
            rope_base: 10_000.0,
            rotary_dim: 64,
            vocab_size: 65024,
        }
    }
}

/// Quantization geometry (group size / bits), consumed by sc-3096. `None` ⇒ the dense default.
#[derive(Clone, Copy, Debug)]
pub struct ChatGlmQuant {
    pub group: i32,
    pub bits: i32,
}

/// A ChatGLM projection — dense or affine-quantized — with an **optional Linear bias** (the fused
/// `query_key_value` carries one; `dense` / MLP projections do not, per `add_bias_linear=false`).
enum ChatGlmLinear {
    Dense {
        w: Array, // [out, in]
        bias: Option<Array>,
    },
    Quant {
        q: Array,      // [out, in_packed] u32
        scales: Array, // [out, in/group]
        qbias: Array,  // [out, in/group] (affine zero-point — NOT a Linear bias)
        group: i32,
        bits: i32,
        bias: Option<Array>,
    },
}

impl ChatGlmLinear {
    /// Load `{key}.weight` (+ `.scales`/`.biases` when quantized) and, when `with_bias`, `{key}.bias`.
    /// Quantized iff `quant` is `Some` and `{key}.scales` is present. All non-packed tensors cast to
    /// `dtype` (the compute dtype).
    fn load(
        w: &Weights,
        key: &str,
        quant: Option<ChatGlmQuant>,
        with_bias: bool,
        dtype: Dtype,
    ) -> Result<Self> {
        let bias = if with_bias {
            Some(w.require(&format!("{key}.bias"))?.as_dtype(dtype)?)
        } else {
            None
        };
        match (quant, w.get(&format!("{key}.scales"))) {
            (Some(qz), Some(scales)) => Ok(ChatGlmLinear::Quant {
                q: w.require(&format!("{key}.weight"))?.clone(),
                scales: scales.as_dtype(dtype)?,
                qbias: w.require(&format!("{key}.biases"))?.as_dtype(dtype)?,
                group: qz.group,
                bits: qz.bits,
                bias,
            }),
            _ => Ok(ChatGlmLinear::Dense {
                w: w.require(&format!("{key}.weight"))?.as_dtype(dtype)?,
                bias,
            }),
        }
    }

    /// `y = x · Wᵀ (+ bias)`. Dense bias add is the FUSED `addmm` (single rounding, matching the core
    /// [`mlx_gen::nn::linear`]); quant uses `quantized_matmul` (transpose, fp32 accumulation).
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            ChatGlmLinear::Dense { w, bias } => match bias {
                Some(b) => Ok(addmm(b, x, w.t(), 1.0, 1.0)?),
                None => Ok(matmul(x, w.t())?),
            },
            ChatGlmLinear::Quant {
                q,
                scales,
                qbias,
                group,
                bits,
                bias,
            } => {
                let y = quantized_matmul(x, q, scales, qbias, true, *group, *bits)?;
                match bias {
                    Some(b) => Ok(add(&y, b)?),
                    None => Ok(y),
                }
            }
        }
    }

    /// Load-time quantization (the mlx-gen-sdxl sc-2641 path, NOT checkpoint-driven): pack a `Dense`
    /// projection's weight to Q4/Q8 in-memory via the same `QuantizedLinear` path the rest of the repo
    /// uses (so the affine packing is byte-identical), keeping the Linear `bias` as a separate add. The
    /// weight is cast to **bf16 before packing** to match the repo's group-scale convention
    /// (`AdaptableLinear::quantize`). A `Quant` variant is left unchanged (idempotent).
    fn quantize(&mut self, bits: i32, group: Option<i32>) -> Result<()> {
        if let ChatGlmLinear::Dense { w, bias } = self {
            let group = group.unwrap_or(DEFAULT_GROUP_SIZE);
            let linear = Linear {
                weight: Param::new(w.as_dtype(Dtype::Bfloat16)?),
                bias: Param::new(None),
            };
            let ql = QuantizedLinear::try_from_linear(linear, group, bits)?;
            *self = ChatGlmLinear::Quant {
                q: ql.inner.weight.value,
                scales: ql.scales.value,
                qbias: ql.biases.value,
                group,
                bits,
                bias: bias.take(),
            };
        }
        Ok(())
    }
}

struct GlmBlock {
    input_ln: Array,
    post_attn_ln: Array,
    qkv: ChatGlmLinear, // fused query_key_value, biased
    dense: ChatGlmLinear,
    h_to_4h: ChatGlmLinear, // fused gate+up
    h4_to_h: ChatGlmLinear,
}

/// The ChatGLM3-6B backbone used as the Kolors text encoder.
pub struct ChatGlmModel {
    embed: Array, // [vocab, hidden]
    layers: Vec<GlmBlock>,
    final_ln: Array,
    cfg: ChatGlmConfig,
    dtype: Dtype,
}

impl ChatGlmModel {
    /// Build from a `Weights` map holding the Kolors `text_encoder` tensors (the
    /// `embedding.word_embeddings.*` / `encoder.layers.{i}.*` / `encoder.final_layernorm.*` layout).
    /// `quant` selectively quantizes the projections (sc-3096; `None` is the dense default). `dtype`
    /// is the compute dtype — `Float32` for the near-bit parity gate.
    pub fn from_weights(
        w: &Weights,
        cfg: ChatGlmConfig,
        quant: Option<ChatGlmQuant>,
        dtype: Dtype,
    ) -> Result<Self> {
        let norm = |key: &str| -> Result<Array> { Ok(w.require(key)?.as_dtype(dtype)?) };
        let lin = |key: &str, with_bias: bool| -> Result<ChatGlmLinear> {
            ChatGlmLinear::load(w, key, quant, with_bias, dtype)
        };

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for i in 0..cfg.num_layers {
            let b = format!("encoder.layers.{i}.");
            layers.push(GlmBlock {
                input_ln: norm(&format!("{b}input_layernorm.weight"))?,
                post_attn_ln: norm(&format!("{b}post_attention_layernorm.weight"))?,
                qkv: lin(&format!("{b}self_attention.query_key_value"), true)?,
                dense: lin(&format!("{b}self_attention.dense"), false)?,
                h_to_4h: lin(&format!("{b}mlp.dense_h_to_4h"), false)?,
                h4_to_h: lin(&format!("{b}mlp.dense_4h_to_h"), false)?,
            });
        }

        Ok(Self {
            embed: load_embedding(w, "embedding.word_embeddings", quant, dtype)?,
            layers,
            final_ln: norm("encoder.final_layernorm.weight")?,
            cfg,
            dtype,
        })
    }

    /// Load-time quantize the encoder to Q4/Q8 (the memory driver — the 28 GLM blocks dominate the 6B
    /// footprint). Quantizes every block projection (fused `query_key_value` + `dense` + the two MLP
    /// linears); the token embedding stays dense (the gather stays exact, mirroring the mlx-gen-sdxl
    /// text-encoder `quantize`, which also leaves its embedding dense). `final_layernorm` /
    /// `*_layernorm` weights are norms, not quantized. Idempotent.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.qkv.quantize(bits, None)?;
            layer.dense.quantize(bits, None)?;
            layer.h_to_4h.quantize(bits, None)?;
            layer.h4_to_h.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Rotary `(cos, sin)`, each `[1, seq, 1, rotary_dim/2]`, for the given absolute `positions` (one
    /// per sequence slot). Computed in f32 then cast to the compute dtype — the reference's
    /// `forward_impl` (idx_theta in f32, `cos`/`sin`, cast to the model dtype). Kolors left-pads, so
    /// `positions` are the tokenizer's `position_ids` (pad slots → 0, real tokens → 0..L-1), NOT a
    /// plain arange (see [`forward`](Self::forward)).
    fn rope_tables(&self, positions: &[i32]) -> Result<(Array, Array)> {
        let seq = positions.len() as i32;
        let half = (self.cfg.rotary_dim / 2) as usize;
        let rot = self.cfg.rotary_dim as f32;
        let inv_freq: Vec<f32> = (0..half)
            .map(|j| 1.0 / self.cfg.rope_base.powf((2 * j) as f32 / rot))
            .collect();
        let mut freqs = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            for &f in &inv_freq {
                freqs.push(p as f32 * f);
            }
        }
        let freqs = Array::from_slice(&freqs, &[1, seq, 1, half as i32]);
        let cos = cos_op(&freqs)?.as_dtype(self.dtype)?;
        let sin = sin_op(&freqs)?.as_dtype(self.dtype)?;
        Ok((cos, sin))
    }

    /// GLM interleaved half-dim RoPE on `x` `[b, s, heads, head_dim]`. Rotate the first `rotary_dim`
    /// dims as adjacent pairs `(x[2i], x[2i+1])` against `(cos, sin)` `[1,s,1,rotary_dim/2]`; pass the
    /// trailing `head_dim - rotary_dim` dims through.
    fn apply_rope(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h) = (sh[0], sh[1], sh[2]);
        let half = self.cfg.rotary_dim / 2;
        // Equal split because rotary_dim == head_dim / 2 (64 of 128).
        let parts = split(x, 2, 3)?;
        let (x_rot, x_pass) = (&parts[0], &parts[1]); // each [b,s,h,rotary_dim or rest]
        let xr = x_rot.reshape(&[b, s, h, half, 2])?;
        let pair = split(&xr, 2, 4)?; // 2 × [b,s,h,half,1]
        let x0 = pair[0].reshape(&[b, s, h, half])?; // even lane
        let x1 = pair[1].reshape(&[b, s, h, half])?; // odd lane
        let out0 = subtract(&multiply(&x0, cos)?, &multiply(&x1, sin)?)?;
        let out1 = add(&multiply(&x1, cos)?, &multiply(&x0, sin)?)?;
        let rot = concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
            .reshape(&[b, s, h, self.cfg.rotary_dim])?;
        Ok(concatenate_axis(&[&rot, x_pass], 3)?)
    }

    fn attn(
        &self,
        layer: &GlmBlock,
        x: &Array,
        mask: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let (nh, kv, d) = (
            self.cfg.num_heads,
            self.cfg.num_kv_groups,
            self.cfg.head_dim,
        );
        // Fused QKV → [b, s, (nh + 2·kv), d]; head-major, so heads 0..nh = q, nh..nh+kv = k, … = v.
        let qkv = layer.qkv.forward(x)?.reshape(&[b, s, nh + 2 * kv, d])?;
        let q = take_heads(&qkv, 0, nh)?;
        let k = take_heads(&qkv, nh, kv)?;
        let v = take_heads(&qkv, nh + kv, kv)?;

        let q = self.apply_rope(&q, cos, sin)?;
        let k = self.apply_rope(&k, cos, sin)?;

        // [b,s,heads,d] → [b,heads,s,d]
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let scale = (d as f32).powf(-0.5);
        let mask = mask.as_dtype(q.dtype())?;
        let out = scaled_dot_product_attention(&q, &k, &v, scale, &mask, None)?; // GQA-aware
        let out = out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, nh * d])?;
        layer.dense.forward(&out)
    }

    fn mlp(&self, layer: &GlmBlock, x: &Array) -> Result<Array> {
        // dense_h_to_4h fuses gate+up (out 2·ffn); swiglu = silu(chunk0) · chunk1.
        let gu = layer.h_to_4h.forward(x)?;
        let parts = split(&gu, 2, -1)?;
        let gated = multiply(&silu(&parts[0])?, &parts[1])?;
        layer.h4_to_h.forward(&gated)
    }

    fn block(
        &self,
        layer: &GlmBlock,
        x: &Array,
        mask: &Array,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        let r = self.attn(
            layer,
            &rms_norm(x, &layer.input_ln, self.cfg.rms_eps)?,
            mask,
            cos,
            sin,
        )?;
        let h = add(x, &r)?;
        let r = self.mlp(layer, &rms_norm(&h, &layer.post_attn_ln, self.cfg.rms_eps)?)?;
        Ok(add(&h, &r)?)
    }

    /// Run the encoder, returning the **`num_layers + 1` (29)** hidden states the reference
    /// `all_hidden_states` exposes: `[embedding, out(layer 0), …, out(layer 27)]`, each `[B,S,hidden]`,
    /// the last entry **pre**-`final_layernorm`. `input_ids` / `attention_mask` are `[B, S]` (i32);
    /// `attention_mask` is 1 for valid tokens, 0 for padding. RoPE positions are a plain `0..S` arange
    /// (the reference default when `position_ids=None`); use [`forward_with_positions`] for Kolors'
    /// left-padded `position_ids`.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Vec<Array>> {
        self.forward_with_positions(input_ids, attention_mask, None)
    }

    /// Like [`forward`](Self::forward) but with explicit RoPE `position_ids` `[B, S]` (i32). Kolors
    /// passes the tokenizer's left-padded `position_ids` (pad slots 0, real tokens 0..L-1) for both
    /// the positive and negative prompt; `None` falls back to a `0..S` arange.
    pub fn forward_with_positions(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        position_ids: Option<&Array>,
    ) -> Result<Vec<Array>> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);
        let ids = input_ids.reshape(&[-1])?;
        let mut h = self
            .embed
            .take_axis(&ids, 0)?
            .reshape(&[b, s, self.cfg.hidden_size])?;

        let positions: Vec<i32> = match position_ids {
            Some(p) => p
                .reshape(&[-1])?
                .as_dtype(Dtype::Int32)?
                .as_slice::<i32>()
                .to_vec(),
            None => (0..s).collect(),
        };
        let mask = self.causal_padding_mask(attention_mask, b, s)?;
        let (cos, sin) = self.rope_tables(&positions)?;

        let mut hiddens = Vec::with_capacity(self.cfg.num_layers + 1);
        hiddens.push(h.clone()); // state 0 = embedding (input to layer 0)
        for layer in &self.layers {
            h = self.block(layer, &h, &mask, &cos, &sin)?;
            hiddens.push(h.clone()); // output of this layer
        }
        Ok(hiddens)
    }

    /// Extract Kolors conditioning: `(context, pooled)`. `context` = `hidden_states[-2]` `[B,S,hidden]`
    /// (penultimate layer); `pooled` = `hidden_states[-1]` at the **last sequence position** `[B,hidden]`.
    /// `position_ids` is the tokenizer's left-padded RoPE positions (`None` ⇒ `0..S` arange).
    pub fn encode_prompt(
        &self,
        input_ids: &Array,
        attention_mask: &Array,
        position_ids: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let hs = self.forward_with_positions(input_ids, attention_mask, position_ids)?;
        let n = hs.len();
        let context = hs[n - 2].clone();
        let last = &hs[n - 1];
        let lsh = last.shape();
        let (b, s, hidden) = (lsh[0], lsh[1], lsh[2]);
        let idx = Array::from_slice(&[s - 1], &[1]);
        let pooled = last.take_axis(&idx, 1)?.reshape(&[b, hidden])?;
        Ok((context, pooled))
    }

    /// The conventional `last_hidden_state` = `final_layernorm(hidden_states[-1])`. Not used by Kolors
    /// conditioning; exposed for completeness / GLM-family reuse.
    pub fn last_hidden_state(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let hs = self.forward(input_ids, attention_mask)?;
        Ok(rms_norm(
            &hs[hs.len() - 1],
            &self.final_ln,
            self.cfg.rms_eps,
        )?)
    }

    /// Additive `(B, 1, S, S)` mask in the compute dtype, mirroring the reference `get_masks`:
    /// a **real** query row `i` (`mask[i]=1`) attends key `j` iff causal (`j ≤ i`) and the key is not
    /// padding (`mask[j]=1`); a **padding** query row (`mask[i]=0`) attends everything (the reference's
    /// `-= padding_mask - 1` adjustment) — so its hidden state is deterministic and matches, even though
    /// Kolors ignores it. Disallowed → a large finite negative (avoids the all-`-inf` softmax NaN).
    ///
    /// Built per batch row from the `[B, S]` `attention_mask`, so a `B > 1` call applies each row's own
    /// padding rather than batch-0's to everyone (F-127). `[B, 1, S, S]` broadcasts over heads and is
    /// bit-identical to the old `[1, 1, S, S]` when `B = 1`.
    fn causal_padding_mask(&self, attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
        let m = attention_mask.reshape(&[-1])?.as_dtype(Dtype::Int32)?;
        let m = m.as_slice::<i32>();
        let (bl, sl) = (b as usize, s as usize);
        if m.len() != bl * sl {
            return Err(Error::Msg(format!(
                "chatglm3 attention mask has {} entries, expected {bl}×{sl}",
                m.len()
            )));
        }
        let data = causal_padding_data(m, bl, sl);
        Array::from_slice(&data, &[b, 1, s, s])
            .as_dtype(self.dtype)
            .map_err(Error::from)
    }
}

/// Flat `[B·S·S]` additive-mask data: for each batch row `bi`, query `i` is masked off key `j`
/// (value `-1e30`) unless that row's own padding allows it. Pulled out of `causal_padding_mask` so
/// the per-row batch logic (F-127) is unit-testable without a loaded encoder. Assumes `m.len()==b*s`.
fn causal_padding_data(m: &[i32], b: usize, s: usize) -> Vec<f32> {
    let neg = -1e30f32;
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        let row = &m[bi * s..(bi + 1) * s];
        for i in 0..s {
            let pad_query = row[i] == 0;
            for j in 0..s {
                let allowed = pad_query || (j <= i && row[j] != 0);
                if !allowed {
                    data[(bi * s + i) * s + j] = neg;
                }
            }
        }
    }
    data
}

/// Take `count` consecutive heads starting at `start` along the head axis (axis 2) of a
/// `[b, s, heads, d]` tensor — the unequal fused-QKV split (q=32, k=2, v=2 heads).
fn take_heads(x: &Array, start: i32, count: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..start + count).collect();
    let idx = Array::from_slice(&idx, &[count]);
    Ok(x.take_axis(&idx, 2)?)
}

/// Load the token-embedding table as a dense matrix (dequantizing if the snapshot is quantized —
/// per-group affine dequant is row-independent, so dequant-then-gather == the reference's
/// gather-then-dequant).
fn load_embedding(
    w: &Weights,
    key: &str,
    quant: Option<ChatGlmQuant>,
    dtype: Dtype,
) -> Result<Array> {
    match (quant, w.get(&format!("{key}.scales"))) {
        (Some(qz), Some(scales)) => {
            let q = w.require(&format!("{key}.weight"))?;
            let qbias = w.require(&format!("{key}.biases"))?;
            Ok(
                dequantize(q, scales, Some(qbias), Some(qz.group), Some(qz.bits))?
                    .as_dtype(dtype)?,
            )
        }
        _ => Ok(w.require(&format!("{key}.weight"))?.as_dtype(dtype)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_padding_data_uses_per_row_padding() {
        // F-127: a B=2 mask must apply each row's own padding, not batch-0's to everyone.
        // Row 0: all real ([1,1,1]). Row 1: key 0 padded ([0,1,1]) — a causally-visible position so
        // the two rows genuinely differ (the old flatten-first-S bug would have made them identical).
        let s = 3;
        let m = [1, 1, 1, /* row 1 */ 0, 1, 1];
        let data = causal_padding_data(&m, 2, s);
        let neg = -1e30f32;
        let at = |bi: usize, i: usize, j: usize| data[(bi * s + i) * s + j];

        // Row 0 is a plain causal mask.
        assert_eq!(at(0, 0, 1), neg); // future key masked
        assert_eq!(at(0, 1, 0), 0.0); // past real key attended

        // Row 1, query 1 (real): key 0 is padding ⇒ masked, even though it's causal.
        assert_eq!(at(1, 1, 0), neg);
        assert_eq!(at(1, 1, 1), 0.0);
        // The same cell differs between the rows — the core of the bug.
        assert_ne!(at(0, 1, 0), at(1, 1, 0));

        // Row 1, query 0 is itself padding ⇒ attends everything (reference `-= padding_mask - 1`).
        assert_eq!(at(1, 0, 0), 0.0);
        assert_eq!(at(1, 0, 2), 0.0);
        // ...whereas row 0's real query 0 is strictly causal.
        assert_ne!(at(0, 0, 2), at(1, 0, 2));
    }
}
