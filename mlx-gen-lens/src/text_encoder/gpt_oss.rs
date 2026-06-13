//! gpt-oss-20b attention core (sc-3165): GQA + learned **attention sinks** + alternating
//! sliding/full causal masks + **YaRN RoPE** + RMSNorm — a faithful port of
//! `transformers.models.gpt_oss.modeling_gpt_oss` (`GptOssAttention` / `eager_attention_forward` /
//! `GptOssRotaryEmbedding`).
//!
//! ## Parity-critical details (from the reference)
//! - **RoPE is NeoX "half-split"** (`_apply_rotary_emb` chunks the head_dim in two; cos/sin have
//!   length `head_dim/2`) with the YaRN `attention_scaling` folded into cos/sin. mlx
//!   `fast::rope` does **not** reproduce this layout with custom `freqs` (verified: both
//!   `traditional` settings diverge ~1.7), so the rotation is applied explicitly here — cheap, since
//!   the encoder runs a single short forward.
//! - **Attention sinks**: per-head learnable logit appended as an extra softmax column, then dropped
//!   after the softmax. The reference subtracts the row-wise max *over the combined scores+sink* for
//!   bf16 stability; we reproduce that exactly with an explicit `−max` / exp / denominator softmax
//!   (`softmax([scores, sink])[..., :L]` ≡ `exp(scores−m) / (Σ exp(scores−m) + exp(sink−m))`).
//! - **No q/k-norm** (unlike Gemma). attention scale = `head_dim^-0.5`. Projections **carry biases**.
//! - **GQA**: 64 query heads over 8 KV heads (`repeat_kv`, n_rep = 8).
//!
//! The MoE feed-forward + decoder-layer/residual assembly is sc-3166; this module is the attention
//! sub-block only (it consumes an already-RMSNorm'd hidden state, exactly like the reference
//! `GptOssAttention.forward`).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, cos as cos_op, divide, matmul, max_axes, maximum, minimum,
    multiply, quantize, quantized_matmul, sigmoid, sin as sin_op, split, split_sections, subtract,
    sum_axes,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Quant, Result};

use crate::config::GptOssConfig;
use crate::text_encoder::mxfp4::dequantize_mxfp4;

/// A scalar `[1]` array for broadcasting multiplies.
fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// `y = x · Wᵀ + b` for a stored `[out, in]` weight and `[out]` bias (the gpt-oss attention
/// projections all have biases — `attention_bias: true`).
struct LinearBias {
    w: Array, // [out, in]
    b: Array, // [out]
}

impl LinearBias {
    fn load(w: &Weights, key: &str, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{key}.weight"))?.as_dtype(dtype)?,
            b: w.require(&format!("{key}.bias"))?.as_dtype(dtype)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(add(&matmul(x, self.w.t())?, &self.b)?)
    }
}

/// One gpt-oss decoder layer's attention (`self_attn`). Consumes the RMSNorm'd hidden state and
/// returns the attention output *before* the residual add (matching `GptOssAttention.forward`).
pub struct GptOssAttention {
    q_proj: LinearBias,
    k_proj: LinearBias,
    v_proj: LinearBias,
    o_proj: LinearBias,
    /// Per-head sink logits, `[num_heads]`.
    sinks: Array,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl GptOssAttention {
    /// Load `self_attn` at `{prefix}` (e.g. `model.layers.0.self_attn`) at `dtype` (bf16 production /
    /// f32 for the correctness gate). The attention weights are dense in the checkpoint
    /// (`modules_to_not_convert` keeps `self_attn` out of MXFP4).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &GptOssConfig,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            q_proj: LinearBias::load(w, &format!("{prefix}.q_proj"), dtype)?,
            k_proj: LinearBias::load(w, &format!("{prefix}.k_proj"), dtype)?,
            v_proj: LinearBias::load(w, &format!("{prefix}.v_proj"), dtype)?,
            o_proj: LinearBias::load(w, &format!("{prefix}.o_proj"), dtype)?,
            sinks: w.require(&format!("{prefix}.sinks"))?.as_dtype(dtype)?,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f32).powf(-0.5),
        })
    }

    /// `x`: `[B, L, hidden]` RMSNorm'd hidden state. `inv_freq`: the YaRN frequencies `[head_dim/2]`.
    /// `attn_scaling`: the YaRN mscale. `mask`: additive `[1, 1, L, L]` (or broadcastable) causal /
    /// sliding mask. Returns `[B, L, hidden]`.
    pub fn forward(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        mask: &Array,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, l) = (sh[0], sh[1]);
        let (h, kv, d) = (self.num_heads, self.num_kv_heads, self.head_dim);

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, l, h, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,H,L,d]
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,kv,L,d]
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, l, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,kv,L,d]

        // RoPE: the reference uses a NeoX **half-split** rotation (`_apply_rotary_emb` chunks the
        // head_dim in two; cos/sin have length head_dim/2) with the YaRN `attention_scaling` folded
        // into cos/sin. mlx `fast::rope` does not reproduce this layout with custom `freqs`, so we
        // apply it explicitly (cheap: encoder-only, short sequence).
        let (cos, sin) = yarn_cos_sin(l, inv_freq, attn_scaling, x.dtype())?;
        let q = apply_half_rope(&q, &cos, &sin)?;
        let k = apply_half_rope(&k, &cos, &sin)?;

        // GQA: repeat K/V from `kv` heads to `h` heads (n_rep = h/kv).
        let k = repeat_kv(&k, h)?; // [B,H,L,d]
        let v = repeat_kv(&v, h)?; // [B,H,L,d]

        // scores = (q·kᵀ)·scale + mask   → [B,H,L,L]
        let scores = multiply(
            &matmul(&q, &k.transpose_axes(&[0, 1, 3, 2])?)?,
            scalar(self.scale),
        )?;
        let scores = add(&scores, mask)?;

        // Sink column: sinks[h] → [1,H,1,1] → broadcast [B,H,L,1].
        let sink = broadcast_to(&self.sinks.reshape(&[1, h, 1, 1])?, &[b, h, l, 1])?;

        // Softmax over [scores, sink] with the reference's −(row-max incl. sink) stabilization, then
        // drop the sink column: probs = exp(scores−m) / (Σ exp(scores−m) + exp(sink−m)).
        let row_max = max_axes(&scores, &[-1], true)?; // [B,H,L,1]
        let m = maximum(&row_max, &sink)?; // [B,H,L,1]
        let exp_scores = subtract(&scores, &m)?.exp()?; // [B,H,L,L]
        let exp_sink = subtract(&sink, &m)?.exp()?; // [B,H,L,1]
        let denom = add(&sum_axes(&exp_scores, &[-1], true)?, &exp_sink)?; // [B,H,L,1]
        let probs = divide(&exp_scores, &denom)?; // [B,H,L,L]

        let out = matmul(&probs, &v)?; // [B,H,L,d]
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, l, h * d])?;
        self.o_proj.forward(&out)
    }

    /// Incremental (cached) attention for autoregressive generation (sc-3176). Processes the `T` new
    /// tokens in `x` `[B, T, hidden]` at **absolute** positions `position..position+T`, appends their
    /// (post-RoPE) K / (pre-repeat) V to `cache`, attends the new queries over the whole cache, then —
    /// for a sliding-window layer — evicts the cache to the last `window` keys for the next step.
    /// `mask` is the additive `[1, 1, T, cache_len]` causal(+sliding) mask for the prefill (`T > 1`);
    /// for a single decode token (`T == 1`) every cached key is valid, so `mask` is `None`.
    ///
    /// `position` is the **true** sequence offset of the first new token, passed explicitly because a
    /// sliding layer's `cache.len()` is capped at `window` and so does *not* track the absolute
    /// position — the RoPE rotation must use the real position (a bug if derived from `cache.len()`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        position: i32,
        cache: &mut KvCache,
        sliding_window: Option<i32>,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let sh = x.shape();
        let (b, t) = (sh[0], sh[1]);
        let (h, kv, d) = (self.num_heads, self.num_kv_heads, self.head_dim);
        let past = position;

        let q = self
            .q_proj
            .forward(x)?
            .reshape(&[b, t, h, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,H,T,d]
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, t, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?; // [B,kv,T,d]
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, t, kv, d])?
            .transpose_axes(&[0, 2, 1, 3])?;

        // RoPE the new q/k at their absolute positions; cache the post-RoPE K so relative rotations
        // stay correct under sliding-window eviction.
        let (cos, sin) = yarn_cos_sin_at(past, t, inv_freq, attn_scaling, x.dtype())?;
        let q = apply_half_rope(&q, &cos, &sin)?;
        let k = apply_half_rope(&k, &cos, &sin)?;

        cache.append(&k, &v)?;
        // Sliding window: a **decode** query (`T == 1`) attends to exactly the last `window` keys
        // (positions `p-window+1..=p`), so evict the stale key BEFORE attending — appending the new
        // key made the cache `window+1`. (A `T > 1` prefill instead carries the sliding **mask** over
        // the full prompt, so it evicts *after* attending, leaving the window primed for the next
        // step.) This keeps the cached decode bit-identical to a masked full recompute.
        let prefill = t > 1;
        if !prefill {
            if let Some(w) = sliding_window {
                cache.truncate_last(w)?;
            }
        }
        let k_all = repeat_kv(cache.k.as_ref().unwrap(), h)?; // [B,H,cache_len,d]
        let v_all = repeat_kv(cache.v.as_ref().unwrap(), h)?;

        let mut scores = multiply(
            &matmul(&q, &k_all.transpose_axes(&[0, 1, 3, 2])?)?,
            scalar(self.scale),
        )?; // [B,H,T,cache_len]
        if let Some(m) = mask {
            scores = add(&scores, m)?;
        }
        let sink = broadcast_to(&self.sinks.reshape(&[1, h, 1, 1])?, &[b, h, t, 1])?;
        let row_max = max_axes(&scores, &[-1], true)?;
        let m = maximum(&row_max, &sink)?;
        let exp_scores = subtract(&scores, &m)?.exp()?;
        let exp_sink = subtract(&sink, &m)?.exp()?;
        let denom = add(&sum_axes(&exp_scores, &[-1], true)?, &exp_sink)?;
        let probs = divide(&exp_scores, &denom)?;

        let out = matmul(&probs, &v_all)?; // [B,H,T,d]
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, t, h * d])?;

        // Prefill: prime the sliding cache to the last `window` keys for the next (decode) step.
        if prefill {
            if let Some(w) = sliding_window {
                cache.truncate_last(w)?;
            }
        }
        self.o_proj.forward(&out)
    }
}

/// A per-layer key/value cache for incremental decode (sc-3176). Stores the **post-RoPE K** and **V**
/// at `[B, kv_heads, seq, head_dim]` (pre-`repeat_kv`); a sliding-window layer truncates to the last
/// `window` after each step.
#[derive(Default)]
pub struct KvCache {
    k: Option<Array>,
    v: Option<Array>,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of cached key positions so far.
    pub fn len(&self) -> i32 {
        self.k.as_ref().map(|k| k.shape()[2]).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append the new `[B, kv, T, d]` K / V and return the full cached `(K, V)`.
    fn append(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let k_all = match &self.k {
            Some(prev) => concatenate_axis(&[prev, k], 2)?,
            None => k.clone(),
        };
        let v_all = match &self.v {
            Some(prev) => concatenate_axis(&[prev, v], 2)?,
            None => v.clone(),
        };
        self.k = Some(k_all.clone());
        self.v = Some(v_all.clone());
        Ok((k_all, v_all))
    }

    /// Keep only the last `max` key positions (sliding-window eviction).
    fn truncate_last(&mut self, max: i32) -> Result<()> {
        let len = self.len();
        if len <= max {
            return Ok(());
        }
        // `[:, :, len-max:, :]` — split at `len-max` along the sequence axis, keep the tail.
        let tail =
            |a: &Array| -> Result<Array> { Ok(split_sections(a, &[len - max], 2)?[1].clone()) };
        self.k = self.k.as_ref().map(tail).transpose()?;
        self.v = self.v.as_ref().map(tail).transpose()?;
        Ok(())
    }
}

/// Build the YaRN RoPE `cos`/`sin` for positions `0..l`, each `[1, 1, l, head_dim/2]`, with the
/// `attention_scaling` (mscale) folded in (`cos = cos(p·inv_freq)·scaling`), matching
/// `GptOssRotaryEmbedding.forward`. Cast to `dtype` so they multiply cleanly against q/k.
fn yarn_cos_sin(l: i32, inv_freq: &Array, scaling: f32, dtype: Dtype) -> Result<(Array, Array)> {
    yarn_cos_sin_at(0, l, inv_freq, scaling, dtype)
}

/// As [`yarn_cos_sin`] but for the absolute positions `start..start+l` — used by the incremental
/// decode path (sc-3176), where the `l` new tokens sit at positions offset by the cache length.
fn yarn_cos_sin_at(
    start: i32,
    l: i32,
    inv_freq: &Array,
    scaling: f32,
    dtype: Dtype,
) -> Result<(Array, Array)> {
    let half = inv_freq.shape()[0];
    let pos: Vec<f32> = (start..start + l).map(|i| i as f32).collect();
    let pos = Array::from_slice(&pos, &[l, 1]);
    let freqs = multiply(&pos, &inv_freq.reshape(&[1, half])?)?; // [l, half]
    let s = scalar(scaling);
    let cos = multiply(&cos_op(&freqs)?, &s)?
        .reshape(&[1, 1, l, half])?
        .as_dtype(dtype)?;
    let sin = multiply(&sin_op(&freqs)?, &s)?
        .reshape(&[1, 1, l, half])?
        .as_dtype(dtype)?;
    Ok((cos, sin))
}

/// Apply the NeoX half-split rotation to `[B, H, L, d]` given `cos`/`sin` `[1, 1, L, d/2]`:
/// `out = cat(first·cos − second·sin, second·cos + first·sin)` where `first`/`second` are the two
/// halves of the head dim. Bit-identical to `transformers`' `_apply_rotary_emb`.
fn apply_half_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let parts = split(x, 2, -1)?;
    let (first, second) = (&parts[0], &parts[1]);
    let out_first = subtract(&multiply(first, cos)?, &multiply(second, sin)?)?;
    let out_second = add(&multiply(second, cos)?, &multiply(first, sin)?)?;
    Ok(concatenate_axis(&[out_first, out_second], -1)?)
}

/// `repeat_kv`: expand `[B, kv, L, d]` to `[B, H, L, d]` by repeat-interleaving each KV head
/// `H/kv` times (matching `transformers.repeat_kv`).
fn repeat_kv(x: &Array, num_heads: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, kv, l, d) = (sh[0], sh[1], sh[2], sh[3]);
    if kv == num_heads {
        return Ok(x.clone());
    }
    let n_rep = num_heads / kv;
    let expanded = broadcast_to(&x.reshape(&[b, kv, 1, l, d])?, &[b, kv, n_rep, l, d])?;
    Ok(expanded.reshape(&[b, num_heads, l, d])?)
}

/// Build the additive attention mask `[1, 1, L, L]` for a single un-padded sequence: causal, and —
/// for sliding-window (local) layers — additionally masking keys older than `window` (`i − j ≥
/// window`). Matches `create_causal_mask` / `create_sliding_window_causal_mask` for the no-padding
/// case the Lens encoder runs.
pub fn attention_mask(l: i32, sliding_window: Option<i32>, dtype: Dtype) -> Result<Array> {
    let l = l as usize;
    let neg = f32::MIN / 2.0;
    let mut data = vec![0f32; l * l];
    for i in 0..l {
        for j in 0..l {
            let causal_ok = j <= i;
            let window_ok = match sliding_window {
                Some(w) => (i as i64 - j as i64) < w as i64,
                None => true,
            };
            data[i * l + j] = if causal_ok && window_ok { 0.0 } else { neg };
        }
    }
    Array::from_slice(&data, &[1, 1, l as i32, l as i32])
        .as_dtype(dtype)
        .map_err(Error::from)
}

// =====================================================================================================
// MoE feed-forward + decoder-layer assembly (sc-3166)
// =====================================================================================================

/// One expert projection — either the dense MXFP4-dequantized weight or an MLX-quantized (Q4/Q8)
/// pack (sc-3172). The dense forward is `x · w + b` for a stored `[in, out]` weight (the eager
/// `GptOssExperts` layout, where the expert matmul is `x · gate_up` / `gated · down`, **not** the
/// `x · Wᵀ` of a `nn.Linear`). The quantized forward is the same product via `quantized_matmul` on
/// the `[out, in]` pack (so `transpose = true` recovers `x · w`).
enum Proj {
    /// `w`: `[in, out]`; forward `x · w + b` (byte-identical to the sc-3166 dense MoE path).
    Dense { w: Array, b: Array },
    /// MLX group-wise affine pack of `wᵀ` (`[out, in]`); forward
    /// `quantized_matmul(x, wq, scales, biases, transpose=true) + b`.
    Quant {
        wq: Array,
        scales: Array,
        biases: Array,
        b: Array,
        group_size: i32,
        bits: i32,
    },
}

impl Proj {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Proj::Dense { w, b } => Ok(add(&matmul(x, w)?, b)?),
            Proj::Quant {
                wq,
                scales,
                biases,
                b,
                group_size,
                bits,
            } => {
                let y = quantized_matmul(x, wq, scales, biases, true, *group_size, *bits)?;
                Ok(add(&y, b)?)
            }
        }
    }

    /// Quantize a dense proj to `bits`-bit MLX affine (group 64). The dense `w` is `[in, out]`; MLX
    /// `quantize` expects `[out, in]` (it groups along the last/`in` axis), so transpose first. The
    /// weight + bias are cast to bf16 before packing — the fork-parity convention shared with
    /// [`AdaptableLinear::quantize`] (`quantized_matmul` accumulates in fp32 regardless). No-op if
    /// already quantized.
    fn into_quantized(self, bits: i32, group_size: i32) -> Result<Self> {
        match self {
            Proj::Dense { w, b } => {
                let w_oi = w.t().as_dtype(Dtype::Bfloat16)?; // [out, in]
                let (wq, scales, biases) = quantize(&w_oi, group_size, bits)?;
                Ok(Proj::Quant {
                    wq,
                    scales,
                    biases,
                    b: b.as_dtype(Dtype::Bfloat16)?,
                    group_size,
                    bits,
                })
            }
            q => Ok(q),
        }
    }

    /// The arrays to `eval` so the dense bf16 dequant transient frees once the pack is materialized.
    fn quant_arrays(&self) -> Option<[&Array; 3]> {
        match self {
            Proj::Quant {
                wq, scales, biases, ..
            } => Some([wq, scales, biases]),
            Proj::Dense { .. } => None,
        }
    }
}

/// One expert's two projections (`gate_up`, `down`) — dense or quantized.
struct Expert {
    gate_up: Proj,
    down: Proj,
}

/// gpt-oss MoE feed-forward: a top-k linear router + 32 **clamped-SwiGLU** experts. Faithful port of
/// `GptOssTopKRouter` + `GptOssExperts`: router → top-`k` softmax over the selected logits; each
/// expert computes `(up+1)·(gate·σ(α·gate))` with `gate` clamped to `≤limit` and `up` clamped to
/// `±limit`, weighted by its router score. Correctness-first: the experts are evaluated densely (all
/// 32) and combined by a masked routing-weight matrix (the encoder runs short prompts; a gather/
/// grouped-GEMM path can follow).
pub struct GptOssMoe {
    router_w: Array, // [E, hidden]
    router_b: Array, // [E]
    experts: Vec<Expert>,
    num_experts: i32,
    top_k: i32,
    inter: i32,
    alpha: f32,
    limit: f32,
}

impl GptOssMoe {
    /// Load `mlp` at `{prefix}` (e.g. `model.layers.0.mlp`). The router stays dense bf16; the experts
    /// are MXFP4 (`*_blocks`/`*_scales`) and are dequantized to `dtype` via [`dequantize_mxfp4`].
    ///
    /// When `quant` is `Some`, each expert projection is immediately re-quantized to MLX Q4/Q8 (the
    /// `~12 GB` path, sc-3172): the per-layer bf16 dequant is the only transient — it is `eval`'d into
    /// the Q4/Q8 pack and freed before the next layer loads, so the full bf16 expert stack
    /// (`~38 GB` across 24 layers) never co-resides. The router/attention/embedding stay dense.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        let e = cfg.num_experts;
        let (hidden, inter) = (cfg.hidden_size, cfg.intermediate);
        let req = |k: &str| -> Result<Array> { Ok(w.require(k)?.as_dtype(dtype)?) };

        let gate_up = dequantize_mxfp4(
            w.require(&format!("{prefix}.experts.gate_up_proj_blocks"))?,
            w.require(&format!("{prefix}.experts.gate_up_proj_scales"))?,
            dtype,
        )?; // [E, hidden, 2*inter]
        let down = dequantize_mxfp4(
            w.require(&format!("{prefix}.experts.down_proj_blocks"))?,
            w.require(&format!("{prefix}.experts.down_proj_scales"))?,
            dtype,
        )?; // [E, inter, hidden]
        let gate_up_b = req(&format!("{prefix}.experts.gate_up_proj_bias"))?; // [E, 2*inter]
        let down_b = req(&format!("{prefix}.experts.down_proj_bias"))?; // [E, hidden]

        // Split the per-expert stacks into individual [.,.] weights (drops the leading E axis).
        let gu = split(&gate_up, e, 0)?;
        let gub = split(&gate_up_b, e, 0)?;
        let dn = split(&down, e, 0)?;
        let dnb = split(&down_b, e, 0)?;
        let mut experts = Vec::with_capacity(e as usize);
        for i in 0..e as usize {
            let mut expert = Expert {
                gate_up: Proj::Dense {
                    w: gu[i].reshape(&[hidden, 2 * inter])?,
                    b: gub[i].reshape(&[2 * inter])?,
                },
                down: Proj::Dense {
                    w: dn[i].reshape(&[inter, hidden])?,
                    b: dnb[i].reshape(&[hidden])?,
                },
            };
            if let Some(q) = quant {
                let (bits, gs) = (q.bits(), mlx_gen::quant::DEFAULT_GROUP_SIZE);
                expert.gate_up = expert.gate_up.into_quantized(bits, gs)?;
                expert.down = expert.down.into_quantized(bits, gs)?;
            }
            experts.push(expert);
        }

        // Force the packs so the layer's bf16 dequant transient frees before the next layer (the
        // memory win is only realized if the bf16 stack does not stay alive in the lazy graph).
        if quant.is_some() {
            let mut to_eval: Vec<&Array> = Vec::with_capacity(e as usize * 6);
            for expert in &experts {
                if let Some(a) = expert.gate_up.quant_arrays() {
                    to_eval.extend_from_slice(&a);
                }
                if let Some(a) = expert.down.quant_arrays() {
                    to_eval.extend_from_slice(&a);
                }
            }
            mlx_rs::transforms::eval(to_eval)?;
        }

        Ok(Self {
            router_w: req(&format!("{prefix}.router.weight"))?,
            router_b: req(&format!("{prefix}.router.bias"))?,
            experts,
            num_experts: e,
            top_k: cfg.experts_per_tok,
            inter,
            alpha: cfg.swiglu_alpha,
            limit: cfg.swiglu_limit,
        })
    }

    /// `x`: `[B, L, hidden]`. Returns `[B, L, hidden]`.
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, l, hidden) = (sh[0], sh[1], sh[2]);
        let n = b * l;
        let xf = x.reshape(&[n, hidden])?;

        // Router logits → dense top-k softmax routing-weight matrix [n, E] (zero off the top-k).
        let logits = add(&matmul(&xf, self.router_w.t())?, &self.router_b)?; // [n, E]
        let routing = self.routing_weights(&logits, n)?; // [n, E]
        let routing_cols = split(&routing, self.num_experts, 1)?; // E × [n, 1]

        let limit = scalar(self.limit);
        let neg_limit = scalar(-self.limit);
        let alpha = scalar(self.alpha);
        let one = scalar(1.0);

        let mut acc: Option<Array> = None;
        for (e, expert) in self.experts.iter().enumerate() {
            let gate_up = expert.gate_up.forward(&xf)?; // [n, 2*inter]
                                                        // Interleaved gate/up: reshape [n, inter, 2] → split → gate = [..,0], up = [..,1].
            let gu = gate_up.reshape(&[n, self.inter, 2])?;
            let halves = split(&gu, 2, -1)?;
            let gate = halves[0].reshape(&[n, self.inter])?;
            let up = halves[1].reshape(&[n, self.inter])?;
            // clamp: gate ≤ limit; up ∈ [−limit, limit].
            let gate = minimum(&gate, &limit)?;
            let up = maximum(&minimum(&up, &limit)?, &neg_limit)?;
            // glu = gate · σ(α·gate); gated = (up + 1) · glu.
            let glu = multiply(&gate, &sigmoid(&multiply(&gate, &alpha)?)?)?;
            let gated = multiply(&add(&up, &one)?, &glu)?; // [n, inter]
            let out_e = expert.down.forward(&gated)?; // [n, hidden]
            let weighted = multiply(&out_e, &routing_cols[e])?; // [n, hidden] · [n, 1]
            acc = Some(match acc {
                None => weighted,
                Some(a) => add(&a, &weighted)?,
            });
        }
        // No expert hit is impossible (top_k ≥ 1); unwrap is safe.
        Ok(acc.expect("at least one expert").reshape(&[b, l, hidden])?)
    }

    /// Build the dense `[n, E]` routing-weight matrix: per row, softmax over the top-`k` logits and
    /// scatter to the selected expert indices (zero elsewhere). Host-side (exact `torch.topk`
    /// tie-by-index semantics); `n·E` is small for an encoder pass.
    fn routing_weights(&self, logits: &Array, n: i32) -> Result<Array> {
        let e = self.num_experts as usize;
        let k = self.top_k as usize;
        let l32 = logits.as_dtype(Dtype::Float32)?;
        let data = l32.as_slice::<f32>(); // [n*E]
        let mut out = vec![0f32; n as usize * e];
        for row in 0..n as usize {
            let s = &data[row * e..row * e + e];
            let mut idx: Vec<usize> = (0..e).collect();
            // descending value, ties broken by lower index (matches torch.topk).
            // `total_cmp` is NaN-safe (a NaN router logit from bf16 overflow would panic
            // `partial_cmp().unwrap()`); identical to the prior order for finite values, so the
            // descending-value, tie-by-lower-index `torch.topk` semantics are unchanged (sc-5251/F-001).
            idx.sort_by(|&a, &b| s[b].total_cmp(&s[a]).then(a.cmp(&b)));
            let top = &idx[..k];
            let maxv = top.iter().map(|&i| s[i]).fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            let exps: Vec<f32> = top
                .iter()
                .map(|&i| {
                    let ev = (s[i] - maxv).exp();
                    denom += ev;
                    ev
                })
                .collect();
            for (j, &i) in top.iter().enumerate() {
                out[row * e + i] = exps[j] / denom;
            }
        }
        Array::from_slice(&out, &[n, e as i32])
            .as_dtype(logits.dtype())
            .map_err(Error::from)
    }
}

/// One full gpt-oss decoder layer: pre-norm sandwich `h + attn(rms(h))` then `h + moe(rms(h))`
/// (`GptOssDecoderLayer.forward`).
pub struct GptOssDecoderLayer {
    input_ln: Array,
    post_attn_ln: Array,
    attn: GptOssAttention,
    moe: GptOssMoe,
    eps: f32,
}

impl GptOssDecoderLayer {
    /// Load the layer at `{prefix}` (e.g. `model.layers.0`). `quant` (when `Some`) quantizes only the
    /// MoE experts to Q4/Q8 (sc-3172); attention/router/norms stay dense `dtype`.
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &GptOssConfig,
        dtype: Dtype,
        quant: Option<Quant>,
    ) -> Result<Self> {
        Ok(Self {
            input_ln: w
                .require(&format!("{prefix}.input_layernorm.weight"))?
                .as_dtype(dtype)?,
            post_attn_ln: w
                .require(&format!("{prefix}.post_attention_layernorm.weight"))?
                .as_dtype(dtype)?,
            attn: GptOssAttention::from_weights(w, &format!("{prefix}.self_attn"), cfg, dtype)?,
            moe: GptOssMoe::from_weights(w, &format!("{prefix}.mlp"), cfg, dtype, quant)?,
            eps: cfg.rms_eps,
        })
    }

    /// The MoE sub-block (exposed for isolated validation).
    pub fn moe(&self) -> &GptOssMoe {
        &self.moe
    }

    /// `x`: `[B, L, hidden]`. `inv_freq`/`attn_scaling`: YaRN constants. `mask`: additive attention
    /// mask. Returns `[B, L, hidden]`.
    pub fn forward(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        mask: &Array,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self.attn.forward(&normed, inv_freq, attn_scaling, mask)?,
        )?;
        let normed = rms_norm(&h, &self.post_attn_ln, self.eps)?;
        Ok(add(&h, &self.moe.forward(&normed)?)?)
    }

    /// Incremental (cached) decoder layer for generation (sc-3176): the same pre-norm sandwich, with
    /// the attention sub-block running [`GptOssAttention::forward_cached`] over `cache`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_cached(
        &self,
        x: &Array,
        inv_freq: &Array,
        attn_scaling: f32,
        position: i32,
        cache: &mut KvCache,
        sliding_window: Option<i32>,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let normed = rms_norm(x, &self.input_ln, self.eps)?;
        let h = add(
            x,
            &self.attn.forward_cached(
                &normed,
                inv_freq,
                attn_scaling,
                position,
                cache,
                sliding_window,
                mask,
            )?,
        )?;
        let normed = rms_norm(&h, &self.post_attn_ln, self.eps)?;
        Ok(add(&h, &self.moe.forward(&normed)?)?)
    }
}
