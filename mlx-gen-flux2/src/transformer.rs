//! FLUX.2 MMDiT transformer — 8 double (joint img+txt) blocks + 24 single (fused parallel
//! attention+SwiGLU) blocks, shared per-stream modulation, 4-axis interleaved RoPE, and an
//! `AdaLayerNormContinuous` output. Port of `models/flux2/model/flux2_transformer/`.
//!
//! Runs f32 activations (matmul(f32 act, bf16 weight)→f32): the `x_embedder` (K=128, M=seq≥2) is
//! the dense 16-bit Metal GEMM bug shape, so the whole stack must run f32 — which is also the
//! quality target. Linears are bias-less core [`AdaptableLinear`]s so `spec.quantize` can pack
//! every projection to Q4/Q8 in place (sc-2643; the fork quantizes every transformer `nn.Linear`).
//! RMSNorm/LayerNorm weights stay full precision. With f32 activations the quantized forward feeds
//! `quantized_matmul` f32 inputs (no bf16 upcast needed). LoRA over these bases = sc-2646.

use mlx_rs::error::Exception;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, sigmoid, split};
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};
use std::f32::consts::LN_10;

use mlx_gen::adapters::loader::{BflTarget, LoraRowSlice};
use mlx_gen::adapters::{prefixed_paths, AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Flux2Config;
use crate::kv_cache::{Flux2KvCache, Stream};
use crate::pos_embed::Flux2PosEmbed;

/// Per-call KV-cache binding handed to an attention layer: `(cache, layer_idx_within_stream)`.
/// `None` on the dense path (txt2img, plain edit) and inside parity tests.
type CacheSlot<'a> = Option<(&'a Flux2KvCache, usize)>;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

// sc-2963 compiled-glue toggle + the `modulate`/`gated`/`rope_rotate` helpers are hoisted into core
// (F-101) so FLUX.1/FLUX.2 share one implementation. Re-export the toggle as this crate's public API;
// FLUX.2's modulate keeps a strong f32 `1` via `one_matches_scale = false`. SwiGLU stays crate-specific.
use mlx_gen::nn::compile_glue;
pub use mlx_gen::nn::set_compile_glue;

/// Wrap a stored `[out, in]` weight as a bias-less dense [`AdaptableLinear`] (every FLUX.2
/// transformer projection). Dense forward = `matmul(x, wᵀ)`, bit-identical to the prior raw
/// `matmul_t`; `quantize` swaps the base to a Q4/Q8 `quantized_matmul`.
fn lin(w: &Weights, key: &str) -> Result<AdaptableLinear> {
    Ok(AdaptableLinear::dense(w.require(key)?.clone(), None))
}

fn require_f32_input(x: &Array) -> Result<Array> {
    Ok(x.as_dtype(Dtype::Float32)?)
}

/// `[B,S,H·D]` → `[B,H,S,D]`, with per-head q/k RMSNorm (f32). Port of `AttentionUtils.process_qkv`.
#[allow(clippy::too_many_arguments)]
fn process_qkv(
    x: &Array,
    q_w: &AdaptableLinear,
    k_w: &AdaptableLinear,
    v_w: &AdaptableLinear,
    norm_q: &Array,
    norm_k: &Array,
    heads: i32,
    head_dim: i32,
) -> Result<(Array, Array, Array)> {
    let sh = x.shape();
    let (b, s) = (sh[0], sh[1]);
    let to_bhsd = |a: Array| -> Result<Array> {
        Ok(a.reshape(&[b, s, heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?)
    };
    let q = to_bhsd(q_w.forward(x)?)?;
    let k = to_bhsd(k_w.forward(x)?)?;
    let v = to_bhsd(v_w.forward(x)?)?;
    let q = rms_norm(&q, norm_q, RMS_EPS)?;
    let k = rms_norm(&k, norm_k, RMS_EPS)?;
    Ok((q, k, v))
}

/// The complex RoPE rotation `(real + imag·i)·(cos + sin·i)` → `(out_real, out_imag)`. Forwards to
/// the shared [`mlx_gen::nn::rope_rotate`].
fn rope_rotate(real: &Array, imag: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    mlx_gen::nn::rope_rotate(real, imag, cos, sin)
}

/// Interleaved RoPE (`AttentionUtils.apply_rope_bshd`): pairs `(x[2i], x[2i+1])` rotated by
/// `cos/sin[i]`. `cos`/`sin`: `[S, head_dim/2]`; `q`/`k`: `[B,H,S,head_dim]`.
fn apply_rope(q: &Array, k: &Array, cos: &Array, sin: &Array) -> Result<(Array, Array)> {
    let s = cos.shape()[0];
    let half = cos.shape()[1];
    let cos = cos.reshape(&[1, 1, s, half])?;
    let sin = sin.reshape(&[1, 1, s, half])?;
    let one = |x: &Array| -> Result<Array> {
        let sh = x.shape();
        let (b, h, seq, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let x5 = x.reshape(&[b, h, seq, hd / 2, 2])?;
        let p = split(&x5, 2, 4)?;
        let real = p[0].reshape(&[b, h, seq, hd / 2])?;
        let imag = p[1].reshape(&[b, h, seq, hd / 2])?;
        let (out0, out1) = rope_rotate(&real, &imag, &cos, &sin)?;
        Ok(
            concatenate_axis(&[&out0.expand_dims(4)?, &out1.expand_dims(4)?], 4)?
                .reshape(&[b, h, seq, hd])?,
        )
    };
    Ok((one(q)?, one(k)?))
}

/// SDPA over `[B,H,S,D]` → `[B,S,H·D]`.
fn attention(q: &Array, k: &Array, v: &Array, head_dim: i32) -> Result<Array> {
    let b = q.shape()[0];
    let scale = (head_dim as f32).powf(-0.5);
    let o = scaled_dot_product_attention(q, k, v, scale, None, None)?;
    Ok(o.transpose_axes(&[0, 2, 1, 3])?
        .reshape(&[b, -1, q.shape()[1] * head_dim])?)
}

/// SwiGLU: split last axis in half, `silu(x1) · x2`. The `split` runs eagerly (a shapeless
/// `mx.compile` can't infer a split's output shapes); the fusable `silu(x1)·x2` arithmetic is
/// compiled into one kernel when the sc-2963 glue toggle is on. Bit-exact to the eager
/// `multiply(silu(x1), x2)` — the inline `a·sigmoid(a)` mirrors [`mlx_gen::nn::silu`] op-for-op.
fn swiglu(x: &Array) -> Result<Array> {
    let p = split(x, 2, -1)?;
    let f = |(a, b): (&Array, &Array)| -> std::result::Result<Array, Exception> {
        multiply(&multiply(a, &sigmoid(a)?)?, b) // silu(a)·b
    };
    if compile_glue() {
        Ok(compile(f, true)((&p[0], &p[1]))?)
    } else {
        Ok(f((&p[0], &p[1]))?)
    }
}

/// `(1 + scale) · norm(x) + shift` — FLUX.2 keeps a strong f32 `1`. Forwards to the shared
/// [`mlx_gen::nn::modulate`] with `one_matches_scale = false`.
fn modulate(norm: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    mlx_gen::nn::modulate(norm, scale, shift, false)
}

/// Gated residual `x + gate·y`. Forwards to the shared [`mlx_gen::nn::gated`].
fn gated(x: &Array, gate: &Array, y: &Array) -> Result<Array> {
    mlx_gen::nn::gated(x, gate, y)
}

/// Sinusoidal timestep embedding (diffusers `_timestep_embedding`, flip_sin_to_cos): `[B]` → `[B,
/// dim]` = `concat([cos(args), sin(args)])`.
fn timestep_embedding(t: &Array, dim: usize) -> Result<Array> {
    let half = dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-LN_10 * 4.0 * i as f32 / half as f32).exp())
        .collect();
    // ln(10000) = 4·ln(10).
    let freqs = Array::from_slice(&freqs, &[1, half as i32]);
    let t = t.reshape(&[t.shape()[0], 1])?.as_dtype(Dtype::Float32)?;
    let args = multiply(&t, &freqs)?;
    Ok(concatenate_axis(&[&args.cos()?, &args.sin()?], 1)?)
}

struct FeedForward {
    linear_in: AdaptableLinear,
    linear_out: AdaptableLinear,
}

impl FeedForward {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_in: lin(w, &format!("{prefix}.linear_in.weight"))?,
            linear_out: lin(w, &format!("{prefix}.linear_out.weight"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let x = self.linear_in.forward(x)?;
        let x = swiglu(&x)?;
        self.linear_out.forward(&x)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear_in.quantize(bits, None)?;
        self.linear_out.quantize(bits, None)?;
        Ok(())
    }
}

struct DoubleBlock {
    attn: DoubleAttention,
    ff: FeedForward,
    ff_context: FeedForward,
}

struct DoubleAttention {
    to_q: AdaptableLinear,
    to_k: AdaptableLinear,
    to_v: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    add_q: AdaptableLinear,
    add_k: AdaptableLinear,
    add_v: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_added_q: Array,
    norm_added_k: Array,
    heads: i32,
    head_dim: i32,
}

impl DoubleAttention {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, head_dim: i32) -> Result<Self> {
        let g = |n: &str| w.require(&format!("{prefix}.{n}.weight")).cloned();
        let l = |n: &str| lin(w, &format!("{prefix}.{n}.weight"));
        Ok(Self {
            to_q: l("to_q")?,
            to_k: l("to_k")?,
            to_v: l("to_v")?,
            to_out: l("to_out")?,
            norm_q: g("norm_q")?,
            norm_k: g("norm_k")?,
            add_q: l("add_q_proj")?,
            add_k: l("add_k_proj")?,
            add_v: l("add_v_proj")?,
            to_add_out: l("to_add_out")?,
            norm_added_q: g("norm_added_q")?,
            norm_added_k: g("norm_added_k")?,
            heads,
            head_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for p in [
            &mut self.to_q,
            &mut self.to_k,
            &mut self.to_v,
            &mut self.to_out,
            &mut self.add_q,
            &mut self.add_k,
            &mut self.add_v,
            &mut self.to_add_out,
        ] {
            p.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Joint attention. Returns `(img_attn_out, txt_attn_out)`. `cache` (the 9b-kv edit path)
    /// stores/splices the trailing reference K/V for this double-stream layer post-RoPE.
    fn forward(
        &self,
        img: &Array,
        txt: &Array,
        cos: &Array,
        sin: &Array,
        cache: CacheSlot<'_>,
    ) -> Result<(Array, Array)> {
        let (iq, ik, iv) = process_qkv(
            img,
            &self.to_q,
            &self.to_k,
            &self.to_v,
            &self.norm_q,
            &self.norm_k,
            self.heads,
            self.head_dim,
        )?;
        let (tq, tk, tv) = process_qkv(
            txt,
            &self.add_q,
            &self.add_k,
            &self.add_v,
            &self.norm_added_q,
            &self.norm_added_k,
            self.heads,
            self.head_dim,
        )?;
        // [txt, img] order along the sequence (axis 2 in BHSD).
        let q = concatenate_axis(&[&tq, &iq], 2)?;
        let k = concatenate_axis(&[&tk, &ik], 2)?;
        let v = concatenate_axis(&[&tv, &iv], 2)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        // KV-cache hook (post-RoPE, pre-SDPA): extract stores the trailing ref K/V; cached splices
        // it back so the `[txt, target]` queries attend over `[txt, target, ref]`.
        let (k, v) = match cache {
            Some((c, idx)) => c.apply(Stream::Double, idx, k, v)?,
            None => (k, v),
        };
        let o = attention(&q, &k, &v, self.head_dim)?;
        let txt_seq = txt.shape()[1];
        let txt_idx = Array::from_slice(&(0..txt_seq).collect::<Vec<i32>>(), &[txt_seq]);
        let img_idx = Array::from_slice(
            &(txt_seq..o.shape()[1]).collect::<Vec<i32>>(),
            &[o.shape()[1] - txt_seq],
        );
        let txt_out = self.to_add_out.forward(&o.take_axis(&txt_idx, 1)?)?;
        let img_out = self.to_out.forward(&o.take_axis(&img_idx, 1)?)?;
        Ok((img_out, txt_out))
    }
}

impl DoubleBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            attn: DoubleAttention::from_weights(w, &format!("{prefix}.attn"), heads, head_dim)?,
            ff: FeedForward::from_weights(w, &format!("{prefix}.ff"))?,
            ff_context: FeedForward::from_weights(w, &format!("{prefix}.ff_context"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.ff.quantize(bits)?;
        self.ff_context.quantize(bits)?;
        Ok(())
    }

    /// `img_mod` / `txt_mod`: `[(shift_msa,scale_msa,gate_msa),(shift_mlp,scale_mlp,gate_mlp)]`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        mut img: Array,
        mut txt: Array,
        img_mod: &[(Array, Array, Array); 2],
        txt_mod: &[(Array, Array, Array); 2],
        cos: &Array,
        sin: &Array,
        cache: CacheSlot<'_>,
    ) -> Result<(Array, Array)> {
        let (shift_msa, scale_msa, gate_msa) = &img_mod[0];
        let (shift_mlp, scale_mlp, gate_mlp) = &img_mod[1];
        let (c_shift_msa, c_scale_msa, c_gate_msa) = &txt_mod[0];
        let (c_shift_mlp, c_scale_mlp, c_gate_mlp) = &txt_mod[1];

        let norm_img = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_msa, shift_msa)?;
        let norm_txt = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_msa,
            c_shift_msa,
        )?;

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin, cache)?;
        img = gated(&img, gate_msa, &img_attn)?;
        txt = gated(&txt, c_gate_msa, &txt_attn)?;

        let norm_img2 = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_mlp, shift_mlp)?;
        let img_ff = self.ff.forward(&norm_img2)?;
        img = gated(&img, gate_mlp, &img_ff)?;

        let norm_txt2 = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_mlp,
            c_shift_mlp,
        )?;
        let txt_ff = self.ff_context.forward(&norm_txt2)?;
        txt = gated(&txt, c_gate_mlp, &txt_ff)?;

        Ok((txt, img))
    }
}

struct SingleBlock {
    to_qkv_mlp: AdaptableLinear,
    to_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    heads: i32,
    head_dim: i32,
    inner: i32,
}

impl SingleBlock {
    fn from_weights(w: &Weights, prefix: &str, heads: i32, head_dim: i32) -> Result<Self> {
        Ok(Self {
            to_qkv_mlp: lin(w, &format!("{prefix}.attn.to_qkv_mlp_proj.weight"))?,
            to_out: lin(w, &format!("{prefix}.attn.to_out.weight"))?,
            norm_q: w.require(&format!("{prefix}.attn.norm_q.weight"))?.clone(),
            norm_k: w.require(&format!("{prefix}.attn.norm_k.weight"))?.clone(),
            heads,
            head_dim,
            inner: heads * head_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.to_qkv_mlp.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        Ok(())
    }

    /// `mod`: `(shift, scale, gate)`. `cache` (9b-kv edit) stores/splices the trailing reference
    /// K/V for this single-stream layer post-RoPE.
    fn forward(
        &self,
        hidden: &Array,
        m: &(Array, Array, Array),
        cos: &Array,
        sin: &Array,
        cache: CacheSlot<'_>,
    ) -> Result<Array> {
        let (shift, scale, gate) = m;
        let norm = modulate(&layer_norm(hidden, None, None, LN_EPS)?, scale, shift)?;
        let proj = self.to_qkv_mlp.forward(&norm)?;

        let sh = proj.shape();
        let (b, s) = (sh[0], sh[1]);
        let take = |start: i32, end: i32| -> Result<Array> {
            let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
            Ok(proj.take_axis(&idx, 2)?)
        };
        let q = take(0, self.inner)?;
        let k = take(self.inner, 2 * self.inner)?;
        let v = take(2 * self.inner, 3 * self.inner)?;
        let mlp = take(3 * self.inner, sh[2])?;

        let to_bhsd = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, s, self.heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = rms_norm(&to_bhsd(q)?, &self.norm_q, RMS_EPS)?;
        let k = rms_norm(&to_bhsd(k)?, &self.norm_k, RMS_EPS)?;
        let v = to_bhsd(v)?;
        let (q, k) = apply_rope(&q, &k, cos, sin)?;
        let (k, v) = match cache {
            Some((c, idx)) => c.apply(Stream::Single, idx, k, v)?,
            None => (k, v),
        };
        let attn = attention(&q, &k, &v, self.head_dim)?;

        let mlp = swiglu(&mlp)?;
        let cat = concatenate_axis(&[&attn, &mlp], -1)?;
        let attn_output = self.to_out.forward(&cat)?;
        gated(hidden, gate, &attn_output)
    }
}

/// Per-stream modulation producer: `silu(temb) → linear → split into `sets` × (shift,scale,gate)`.
struct Modulation {
    linear: AdaptableLinear,
    sets: usize,
}

impl Modulation {
    fn from_weights(w: &Weights, prefix: &str, sets: usize) -> Result<Self> {
        Ok(Self {
            linear: lin(w, &format!("{prefix}.linear.weight"))?,
            sets,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.linear.quantize(bits, None)
    }

    /// `temb`: `[B, dim]` → `Vec<(shift,scale,gate)>` of length `sets`, each `[B,1,dim]`.
    fn forward(&self, temb: &Array) -> Result<Vec<(Array, Array, Array)>> {
        let mod_ = self.linear.forward(&silu(temb)?)?.expand_dims(1)?;
        let chunks = split(&mod_, (3 * self.sets) as i32, -1)?;
        Ok((0..self.sets)
            .map(|i| {
                (
                    chunks[3 * i].clone(),
                    chunks[3 * i + 1].clone(),
                    chunks[3 * i + 2].clone(),
                )
            })
            .collect())
    }
}

/// The FLUX.2 MMDiT transformer.
pub struct Flux2Transformer {
    pos_embed: Flux2PosEmbed,
    time_linear1: AdaptableLinear,
    time_linear2: AdaptableLinear,
    mod_img: Modulation,
    mod_txt: Modulation,
    mod_single: Modulation,
    x_embedder: AdaptableLinear,
    context_embedder: AdaptableLinear,
    double_blocks: Vec<DoubleBlock>,
    single_blocks: Vec<SingleBlock>,
    norm_out_linear: AdaptableLinear,
    proj_out: AdaptableLinear,
    time_channels: usize,
}

impl Flux2Transformer {
    pub fn from_weights(w: &Weights, cfg: &Flux2Config) -> Result<Self> {
        let heads = cfg.num_heads as i32;
        let head_dim = cfg.head_dim as i32;
        let double_blocks = (0..cfg.num_double_layers)
            .map(|i| {
                DoubleBlock::from_weights(w, &format!("transformer_blocks.{i}"), heads, head_dim)
            })
            .collect::<Result<Vec<_>>>()?;
        let single_blocks = (0..cfg.num_single_layers)
            .map(|i| {
                SingleBlock::from_weights(
                    w,
                    &format!("single_transformer_blocks.{i}"),
                    heads,
                    head_dim,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pos_embed: Flux2PosEmbed::new(cfg.rope_theta, cfg.axes_dim),
            time_linear1: lin(w, "time_guidance_embed.linear_1.weight")?,
            time_linear2: lin(w, "time_guidance_embed.linear_2.weight")?,
            mod_img: Modulation::from_weights(w, "double_stream_modulation_img", 2)?,
            mod_txt: Modulation::from_weights(w, "double_stream_modulation_txt", 2)?,
            mod_single: Modulation::from_weights(w, "single_stream_modulation", 1)?,
            x_embedder: lin(w, "x_embedder.weight")?,
            context_embedder: lin(w, "context_embedder.weight")?,
            double_blocks,
            single_blocks,
            norm_out_linear: lin(w, "norm_out.linear.weight")?,
            proj_out: lin(w, "proj_out.weight")?,
            time_channels: cfg.timestep_channels,
        })
    }

    /// Quantize every transformer `nn.Linear` to Q4/Q8 (group_size 64) in place — the mlx-rs
    /// equivalent of the fork's `nn.quantize(transformer, predicate=hasattr to_quantized, bits)`.
    /// RMSNorm/LayerNorm weights are not Linears, so they stay full precision (as in the fork).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.time_linear1.quantize(bits, None)?;
        self.time_linear2.quantize(bits, None)?;
        self.mod_img.quantize(bits)?;
        self.mod_txt.quantize(bits)?;
        self.mod_single.quantize(bits)?;
        self.x_embedder.quantize(bits, None)?;
        self.context_embedder.quantize(bits, None)?;
        for b in &mut self.double_blocks {
            b.quantize(bits)?;
        }
        for b in &mut self.single_blocks {
            b.quantize(bits)?;
        }
        self.norm_out_linear.quantize(bits, None)?;
        self.proj_out.quantize(bits, None)?;
        Ok(())
    }

    /// Test-only (sc-2643 byte-parity gate): the quantized `(wq, scales, biases, group_size, bits)`
    /// of `transformer_blocks.0.attn.to_q` — a representative bias-less, bf16-native Linear. `None`
    /// if the transformer is still dense.
    #[doc(hidden)]
    pub fn probe_quant_to_q(&self) -> Option<(&Array, &Array, &Array, i32, i32)> {
        let (wq, sc, bi, _bias, gs, b) = self.double_blocks[0].attn.to_q.quantized_params()?;
        Some((wq, sc, bi, gs, b))
    }

    fn temb(&self, timestep: f32) -> Result<Array> {
        // klein has no guidance embedding; timestep is fed as sigma·1000 (>1) so no rescale.
        let t = Array::from_slice(&[timestep], &[1]);
        let emb = timestep_embedding(&t, self.time_channels)?;
        let h = self.time_linear1.forward(&emb)?;
        self.time_linear2.forward(&silu(&h)?)
    }

    fn norm_out(&self, x: &Array, temb: &Array) -> Result<Array> {
        let p = self.norm_out_linear.forward(&silu(temb)?)?; // [B, 2·dim]
        let parts = split(&p, 2, 1)?;
        let scale = parts[0].expand_dims(1)?; // [B,1,dim]
        let shift = parts[1].expand_dims(1)?;
        let normed = layer_norm(x, None, None, LN_EPS)?;
        Ok(add(
            &multiply(&normed, &add(&scale, scalar(1.0))?)?,
            &shift,
        )?)
    }

    /// `hidden_states`: `[B, seq_img, in_channels]`; `encoder_hidden_states`: `[B, seq_txt,
    /// joint_attention_dim]`; `img_ids`/`txt_ids`: `[seq, 4]` (or `[1, seq, 4]`). `timestep` is the
    /// scaled sigma (×1000). Returns the velocity `[B, seq_img, out_channels]`. Dense path: no cache.
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
    ) -> Result<Array> {
        self.forward_with_cache(
            hidden_states,
            encoder_hidden_states,
            img_ids,
            txt_ids,
            timestep,
            None,
        )
    }

    /// As [`Self::forward`], with an optional 9b-kv [`Flux2KvCache`] threaded through every
    /// attention layer (the double + single stacks indexed independently from 0). On the
    /// [`crate::kv_cache::CacheMode::Extract`] step the `img_ids` carry the reference tokens
    /// (`[target, ref]`); on [`crate::kv_cache::CacheMode::Cached`] steps they carry `[target]`
    /// only and the cached ref K/V are spliced back inside each attention.
    pub fn forward_with_cache(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
        cache: Option<&Flux2KvCache>,
    ) -> Result<Array> {
        let temb = self.temb(timestep)?;
        let mut img = self
            .x_embedder
            .forward(&require_f32_input(hidden_states)?)?;
        let mut txt = self
            .context_embedder
            .forward(&require_f32_input(encoder_hidden_states)?)?;

        let drop_batch = |ids: &Array| -> Result<Array> {
            if ids.shape().len() == 3 {
                Ok(ids.reshape(&[ids.shape()[1], ids.shape()[2]])?)
            } else {
                Ok(ids.clone())
            }
        };
        let (img_cos, img_sin) = self.pos_embed.forward(&drop_batch(img_ids)?)?;
        let (txt_cos, txt_sin) = self.pos_embed.forward(&drop_batch(txt_ids)?)?;
        let cos = concatenate_axis(&[&txt_cos, &img_cos], 0)?;
        let sin = concatenate_axis(&[&txt_sin, &img_sin], 0)?;

        let mi = self.mod_img.forward(&temb)?;
        let mt = self.mod_txt.forward(&temb)?;
        let img_mod = [mi[0].clone(), mi[1].clone()];
        let txt_mod = [mt[0].clone(), mt[1].clone()];

        for (idx, block) in self.double_blocks.iter().enumerate() {
            (txt, img) = block.forward(
                img,
                txt,
                &img_mod,
                &txt_mod,
                &cos,
                &sin,
                cache.map(|c| (c, idx)),
            )?;
        }

        let txt_seq = txt.shape()[1];
        let mut hidden = concatenate_axis(&[&txt, &img], 1)?;
        let ms = self.mod_single.forward(&temb)?;
        for (idx, block) in self.single_blocks.iter().enumerate() {
            hidden = block.forward(&hidden, &ms[0], &cos, &sin, cache.map(|c| (c, idx)))?;
        }

        // Keep only the image tokens.
        let img_seq = hidden.shape()[1] - txt_seq;
        let img_idx = Array::from_slice(
            &(txt_seq..hidden.shape()[1]).collect::<Vec<i32>>(),
            &[img_seq],
        );
        let hidden = hidden.take_axis(&img_idx, 1)?;
        let hidden = self.norm_out(&hidden, &temb)?;
        self.proj_out.forward(&hidden)
    }
}

// ---- LoRA/LoKr adapter routing (sc-2646) ------------------------------------------------------
//
// The Rust analog of the fork's `Flux2LoRAMapping`: map the trained-file (diffusers) module paths
// to the crate's `AdaptableLinear` fields, across the FULL transformer-only surface (globals +
// double + single blocks). VAE + Qwen3 TE are NOT LoRA targets. The fork's standard/diffusers
// naming is what these resolve (bare / `transformer.` / `diffusion_model.` prefixes are stripped by
// the core loader before the path reaches here); the BFL/ComfyUI fused-qkv-split + kohya `lora_unet_`
// namings are a separate cross-family format (sc-2618), not handled here.

impl AdaptableHost for Modulation {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["linear"] => Some(&mut self.linear),
            _ => None,
        }
    }
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["linear_in"] => Some(&mut self.linear_in),
            ["linear_out"] => Some(&mut self.linear_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["linear_in", "linear_out"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for DoubleAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file (diffusers) naming → fields: image stream `to_q/k/v`/`to_out`; text stream
        // `add_{q,k,v}_proj` → `add_{q,k,v}` and `to_add_out`.
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            // The fork accepts both the bare `to_out` and the HF-style `to_out.0` (diffusers wraps
            // the output projection in a `Sequential[Linear, Dropout]`); both address this Linear.
            ["to_out"] | ["to_out", "0"] => Some(&mut self.to_out),
            ["add_q_proj"] => Some(&mut self.add_q),
            ["add_k_proj"] => Some(&mut self.add_k),
            ["add_v_proj"] => Some(&mut self.add_v),
            ["to_add_out"] => Some(&mut self.to_add_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        // Both `to_out` and the HF-style `to_out.0` alias resolve to the output projection, and the
        // fork carries a `lora_unet_…_attn_to_out` *and* `…_attn_to_out_0` kohya pattern — emit both
        // so either flattened spelling resolves.
        [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "to_out.0",
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

impl AdaptableHost for DoubleBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            ["ff_context", rest @ ..] => self.ff_context.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = prefixed_paths("attn", &self.attn);
        out.extend(prefixed_paths("ff", &self.ff));
        out.extend(prefixed_paths("ff_context", &self.ff_context));
        out
    }
}

impl AdaptableHost for SingleBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // The fused `to_qkv_mlp_proj` takes a single LoRA covering q/k/v/mlp jointly (the fork maps
        // it as one target); `to_out` is the output projection.
        match path {
            ["attn", "to_qkv_mlp_proj"] => Some(&mut self.to_qkv_mlp),
            ["attn", "to_out"] => Some(&mut self.to_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["attn.to_qkv_mlp_proj", "attn.to_out"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl AdaptableHost for Flux2Transformer {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            // Globals.
            ["x_embedder"] => Some(&mut self.x_embedder),
            ["context_embedder"] => Some(&mut self.context_embedder),
            ["proj_out"] => Some(&mut self.proj_out),
            ["norm_out", "linear"] => Some(&mut self.norm_out_linear),
            ["double_stream_modulation_img", rest @ ..] => self.mod_img.adaptable_mut(rest),
            ["double_stream_modulation_txt", rest @ ..] => self.mod_txt.adaptable_mut(rest),
            ["single_stream_modulation", rest @ ..] => self.mod_single.adaptable_mut(rest),
            ["time_guidance_embed", "linear_1"] => Some(&mut self.time_linear1),
            ["time_guidance_embed", "linear_2"] => Some(&mut self.time_linear2),
            // klein is distilled (no guidance embedding), so `guidance_linear_{1,2}` don't exist
            // here — a LoRA targeting them is correctly surfaced as unmatched.
            ["transformer_blocks", n, rest @ ..] => self
                .double_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            ["single_transformer_blocks", n, rest @ ..] => self
                .single_blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    /// kohya-reachable targets (sc-2618): the diffusers-named double + single block linears. Globals
    /// (`x_embedder`/`context_embedder`/`proj_out`/`norm_out`/the modulations/`time_guidance_embed`)
    /// carry no `lora_unet_` pattern in the fork mapping, so they are excluded (reachable via the
    /// dotted form). The fused→split BFL convention (`double_blocks_*`/`single_blocks_*`) is a
    /// different format (sc-2743) and is intentionally NOT enumerated → such keys surface as unmatched.
    fn adaptable_paths(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (i, b) in self.double_blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("transformer_blocks.{i}"), b));
        }
        for (i, b) in self.single_blocks.iter().enumerate() {
            out.extend(prefixed_paths(&format!("single_transformer_blocks.{i}"), b));
        }
        out
    }

    /// BFL / ComfyUI fused→split targets (sc-2743), the Rust analog of the fork's
    /// `Flux2LoRAMapping._get_bfl_*` + the `base_model.model.` global renames. Three things sc-2618's
    /// diffusers/peft/kohya paths can't do, all here:
    /// - **fused-qkv split**: the BFL `double_blocks.{n}.{img,txt}_attn.qkv` linear is one fused
    ///   `[3·inner, …]` projection; FLUX.2's model keeps q/k/v SEPARATE (`attn.to_q/to_k/to_v`,
    ///   `add_{q,k,v}_proj`), so each destination row-slices its third (equal 3-way; `inner`-independent).
    /// - **BFL module renames**: `img_attn.proj`→`to_out`, `txt_attn.proj`→`to_add_out`,
    ///   `{img,txt}_mlp.{0,2}`→`ff{_context}.linear_{in,out}`, `single_blocks.{n}.linear{1,2}`→
    ///   `attn.{to_qkv_mlp_proj,to_out}` (linear1 stays FUSED in FLUX.2 → no split), and the global
    ///   `base_model.model.` renames (`img_in`→`x_embedder`, `final_layer.linear`→`proj_out`, …).
    /// - **the `diffusion_model.` / `base_model.model.` dotted prefixes** carrying BFL module names.
    ///
    /// klein-absent globals (`norm_out`, `guidance_linear_*`) have no `base_model.model.` BFL spelling
    /// in the fork, so they're omitted; their diffusers-named forms stay peft-reachable. The 4-way
    /// qkv-mlp split (`_split_qkv_mlp_up`) is FLUX.1's (separate `proj_mlp`) and lands with sc-2657.
    fn bfl_targets(&self) -> Vec<BflTarget> {
        let mut out = Vec::new();

        // Globals: `base_model.model.` BFL renames only (the diffusers-named globals — bare /
        // `transformer.` / `diffusion_model.` — are already covered by the peft loader).
        for (bfl, tgt) in [
            ("img_in", "x_embedder"),
            ("txt_in", "context_embedder"),
            ("time_in.in_layer", "time_guidance_embed.linear_1"),
            ("time_in.out_layer", "time_guidance_embed.linear_2"),
            (
                "double_stream_modulation_img.lin",
                "double_stream_modulation_img.linear",
            ),
            (
                "double_stream_modulation_txt.lin",
                "double_stream_modulation_txt.linear",
            ),
            (
                "single_stream_modulation.lin",
                "single_stream_modulation.linear",
            ),
            ("final_layer.linear", "proj_out"),
        ] {
            let (up, down, alpha) = bfl_global_keys(bfl);
            out.push(rename_target(tgt, up, down, alpha));
        }

        // Double blocks.
        for i in 0..self.double_blocks.len() {
            // Fused qkv → split: img → to_{q,k,v}; txt → add_{q,k,v}_proj.
            for (stream, dst) in [
                ("img", ["to_q", "to_k", "to_v"]),
                ("txt", ["add_q_proj", "add_k_proj", "add_v_proj"]),
            ] {
                let flat = format!("double_blocks_{i}_{stream}_attn_qkv");
                let dotted = format!("double_blocks.{i}.{stream}_attn.qkv");
                let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                for idx in 0..3i32 {
                    out.push(BflTarget {
                        target_path: format!("transformer_blocks.{i}.attn.{}", dst[idx as usize]),
                        up_keys: up.clone(),
                        down_keys: down.clone(),
                        alpha_keys: alpha.clone(),
                        up_slice: Some(LoraRowSlice::Chunk { n: 3, index: idx }),
                        down_slice: Some(LoraRowSlice::ChunkIfDivisible { n: 3, index: idx }),
                    });
                }
            }
            // attn output proj (rename, no split): img.proj → to_out; txt.proj → to_add_out.
            for (stream, tgt) in [("img", "to_out"), ("txt", "to_add_out")] {
                let flat = format!("double_blocks_{i}_{stream}_attn_proj");
                let dotted = format!("double_blocks.{i}.{stream}_attn.proj");
                let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                out.push(rename_target(
                    &format!("transformer_blocks.{i}.attn.{tgt}"),
                    up,
                    down,
                    alpha,
                ));
            }
            // MLP (rename): img_mlp.{0,2} → ff.linear_{in,out}; txt_mlp.{0,2} → ff_context.linear_{in,out}.
            for (stream, ff) in [("img", "ff"), ("txt", "ff_context")] {
                for (n, lin) in [("0", "linear_in"), ("2", "linear_out")] {
                    let flat = format!("double_blocks_{i}_{stream}_mlp_{n}");
                    let dotted = format!("double_blocks.{i}.{stream}_mlp.{n}");
                    let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                    out.push(rename_target(
                        &format!("transformer_blocks.{i}.{ff}.{lin}"),
                        up,
                        down,
                        alpha,
                    ));
                }
            }
        }

        // Single blocks (rename, FUSED — no split): linear1 → attn.to_qkv_mlp_proj; linear2 → attn.to_out.
        for i in 0..self.single_blocks.len() {
            for (which, tgt) in [("linear1", "to_qkv_mlp_proj"), ("linear2", "to_out")] {
                let flat = format!("single_blocks_{i}_{which}");
                let dotted = format!("single_blocks.{i}.{which}");
                let (up, down, alpha) = bfl_block_keys(&flat, &dotted);
                out.push(rename_target(
                    &format!("single_transformer_blocks.{i}.attn.{tgt}"),
                    up,
                    down,
                    alpha,
                ));
            }
        }

        out
    }
}

/// A non-split BFL target (a plain module rename): the source factors are copied through, no slice.
fn rename_target(
    target_path: &str,
    up_keys: Vec<String>,
    down_keys: Vec<String>,
    alpha_keys: Vec<String>,
) -> BflTarget {
    BflTarget {
        target_path: target_path.to_string(),
        up_keys,
        down_keys,
        alpha_keys,
        up_slice: None,
        down_slice: None,
    }
}

/// Every BFL source-key spelling for one *block* linear, across the three prefix conventions: kohya
/// `lora_unet_<flat>` (flattened module path) and the dotted `diffusion_model.<dotted>` /
/// `base_model.model.<dotted>` (both BFL-named for block layers), partitioned into (up, down, alpha)
/// — `lora_up`≡`lora_B`, `lora_down`≡`lora_A`. Mirrors the fork's BFL `possible_*_patterns`.
fn bfl_block_keys(flat: &str, dotted: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let up = vec![
        format!("lora_unet_{flat}.lora_up.weight"),
        format!("diffusion_model.{dotted}.lora_B.weight"),
        format!("diffusion_model.{dotted}.lora_up.weight"),
        format!("base_model.model.{dotted}.lora_B.weight"),
        format!("base_model.model.{dotted}.lora_up.weight"),
    ];
    let down = vec![
        format!("lora_unet_{flat}.lora_down.weight"),
        format!("diffusion_model.{dotted}.lora_A.weight"),
        format!("diffusion_model.{dotted}.lora_down.weight"),
        format!("base_model.model.{dotted}.lora_A.weight"),
        format!("base_model.model.{dotted}.lora_down.weight"),
    ];
    let alpha = vec![
        format!("lora_unet_{flat}.alpha"),
        format!("diffusion_model.{dotted}.alpha"),
        format!("base_model.model.{dotted}.alpha"),
    ];
    (up, down, alpha)
}

/// BFL source-key spellings for a *global* linear: only the `base_model.model.<bfl_name>` form adds
/// new coverage (the diffusers-named globals are peft-reachable), so the fork carries no `lora_unet_`
/// or `diffusion_model.` BFL-named global pattern. (up, down, alpha).
fn bfl_global_keys(bfl: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let up = vec![
        format!("base_model.model.{bfl}.lora_B.weight"),
        format!("base_model.model.{bfl}.lora_up.weight"),
    ];
    let down = vec![
        format!("base_model.model.{bfl}.lora_A.weight"),
        format!("base_model.model.{bfl}.lora_down.weight"),
    ];
    let alpha = vec![format!("base_model.model.{bfl}.alpha")];
    (up, down, alpha)
}

/// Configuration glue so callers can keep the transformer's dims in one place.
pub type Flux2TransformerConfig = Flux2Config;

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::adapters::{install_adapter, Adapter};

    #[test]
    fn timestep_embedding_shape_and_flip() {
        let t = Array::from_slice(&[1000.0f32], &[1]);
        let emb = timestep_embedding(&t, 256).unwrap();
        assert_eq!(emb.shape(), &[1, 256]);
    }

    // ---- sc-2646 adapter routing (diffusers-name → field translation) -------------------------

    fn dummy_lin() -> AdaptableLinear {
        AdaptableLinear::dense(Array::from_slice(&[0.0f32], &[1, 1]), None)
    }
    fn dummy_arr() -> Array {
        Array::from_slice(&[1.0f32], &[1])
    }
    fn noop_adapter() -> Adapter {
        Adapter::Lora {
            a: Array::from_slice(&[0.0f32], &[1, 1]),
            b: Array::from_slice(&[0.0f32], &[1, 1]),
            scale: 0.0,
        }
    }
    /// Path resolves iff installing a no-op adapter there succeeds.
    fn resolves(host: &mut impl AdaptableHost, path: &str) -> bool {
        install_adapter(host, path, noop_adapter()).is_ok()
    }

    fn double_attn() -> DoubleAttention {
        DoubleAttention {
            to_q: dummy_lin(),
            to_k: dummy_lin(),
            to_v: dummy_lin(),
            to_out: dummy_lin(),
            norm_q: dummy_arr(),
            norm_k: dummy_arr(),
            add_q: dummy_lin(),
            add_k: dummy_lin(),
            add_v: dummy_lin(),
            to_add_out: dummy_lin(),
            norm_added_q: dummy_arr(),
            norm_added_k: dummy_arr(),
            heads: 1,
            head_dim: 1,
        }
    }

    #[test]
    fn double_attention_routes_diffusers_names() {
        let mut attn = double_attn();
        for p in [
            "to_q",
            "to_k",
            "to_v",
            "to_out",
            "to_out.0", // HF-style diffusers Sequential alias (the fork accepts both)
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
        ] {
            assert!(resolves(&mut attn, p), "{p} should resolve");
        }
        // Internal field names + off-surface must not resolve.
        for p in ["add_q", "add_k", "add_v", "to_add_out.0", "qkv"] {
            assert!(!resolves(&mut attn, p), "{p} must not resolve");
        }
    }

    #[test]
    fn double_block_routes_attn_and_ffs() {
        let mut block = DoubleBlock {
            attn: double_attn(),
            ff: FeedForward {
                linear_in: dummy_lin(),
                linear_out: dummy_lin(),
            },
            ff_context: FeedForward {
                linear_in: dummy_lin(),
                linear_out: dummy_lin(),
            },
        };
        for p in [
            "attn.to_q",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.linear_in",
            "ff.linear_out",
            "ff_context.linear_in",
            "ff_context.linear_out",
        ] {
            assert!(resolves(&mut block, p), "{p} should resolve");
        }
        for p in ["ff.net.0.proj", "mlp.linear_in", "attn.to_qkv_mlp_proj"] {
            assert!(!resolves(&mut block, p), "{p} must not resolve");
        }
    }

    #[test]
    fn single_block_routes_fused_qkv_mlp() {
        let mut block = SingleBlock {
            to_qkv_mlp: dummy_lin(),
            to_out: dummy_lin(),
            norm_q: dummy_arr(),
            norm_k: dummy_arr(),
            heads: 1,
            head_dim: 1,
            inner: 1,
        };
        // The fused projection is addressed by its checkpoint name `attn.to_qkv_mlp_proj`.
        assert!(resolves(&mut block, "attn.to_qkv_mlp_proj"));
        assert!(resolves(&mut block, "attn.to_out"));
        // The internal field name + split q/k/v must NOT resolve (single LoRA covers them jointly).
        for p in ["to_qkv_mlp", "attn.to_q", "attn.to_qkv_mlp_proj.0"] {
            assert!(!resolves(&mut block, p), "{p} must not resolve");
        }
    }

    #[test]
    fn modulation_and_feed_forward_route_leaf_names() {
        let mut m = Modulation {
            linear: dummy_lin(),
            sets: 1,
        };
        assert!(resolves(&mut m, "linear"));
        assert!(!resolves(&mut m, "weight"));

        let mut ff = FeedForward {
            linear_in: dummy_lin(),
            linear_out: dummy_lin(),
        };
        assert!(resolves(&mut ff, "linear_in"));
        assert!(resolves(&mut ff, "linear_out"));
        assert!(!resolves(&mut ff, "net.0.proj"));
    }

    fn ff() -> FeedForward {
        FeedForward {
            linear_in: dummy_lin(),
            linear_out: dummy_lin(),
        }
    }
    fn modulation(sets: usize) -> Modulation {
        Modulation {
            linear: dummy_lin(),
            sets,
        }
    }

    /// A minimal transformer (1 double + 1 single block) for the top-level key→module routing —
    /// the globals' diffusers-name translations + the block-index parse.
    fn tiny_transformer() -> Flux2Transformer {
        Flux2Transformer {
            pos_embed: Flux2PosEmbed::new(2000.0, [32, 32, 32, 32]),
            time_linear1: dummy_lin(),
            time_linear2: dummy_lin(),
            mod_img: modulation(2),
            mod_txt: modulation(2),
            mod_single: modulation(1),
            x_embedder: dummy_lin(),
            context_embedder: dummy_lin(),
            double_blocks: vec![DoubleBlock {
                attn: double_attn(),
                ff: ff(),
                ff_context: ff(),
            }],
            single_blocks: vec![SingleBlock {
                to_qkv_mlp: dummy_lin(),
                to_out: dummy_lin(),
                norm_q: dummy_arr(),
                norm_k: dummy_arr(),
                heads: 1,
                head_dim: 1,
                inner: 1,
            }],
            norm_out_linear: dummy_lin(),
            proj_out: dummy_lin(),
            time_channels: 256,
        }
    }

    #[test]
    fn transformer_routes_full_diffusers_surface() {
        let mut t = tiny_transformer();
        // Globals (diffusers names → internal fields).
        for p in [
            "x_embedder",
            "context_embedder",
            "proj_out",
            "norm_out.linear",
            "double_stream_modulation_img.linear",
            "double_stream_modulation_txt.linear",
            "single_stream_modulation.linear",
            "time_guidance_embed.linear_1",
            "time_guidance_embed.linear_2",
            // Double block 0.
            "transformer_blocks.0.attn.to_q",
            "transformer_blocks.0.attn.add_k_proj",
            "transformer_blocks.0.attn.to_add_out",
            "transformer_blocks.0.ff.linear_in",
            "transformer_blocks.0.ff_context.linear_out",
            // Single block 0.
            "single_transformer_blocks.0.attn.to_qkv_mlp_proj",
            "single_transformer_blocks.0.attn.to_out",
        ] {
            assert!(resolves(&mut t, p), "{p} should resolve");
        }
        // Off-surface / wrong index / klein-absent guidance linears must NOT resolve.
        for p in [
            "norm_out_linear",
            "time_guidance_embed.guidance_linear_1",
            "transformer_blocks.1.attn.to_q", // only 1 double block here
            "single_transformer_blocks.5.attn.to_out",
            "transformer_blocks.0.attn.qkv",
            "vae.encoder",
        ] {
            assert!(!resolves(&mut t, p), "{p} must not resolve");
        }
    }

    // ---- sc-2618 kohya `lora_unet_` routing (no real weights) ---------------------------------

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("mlx_gen_flux2_kohya_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// `adaptable_paths()` is the kohya-reachable surface; every entry must resolve via
    /// `adaptable_mut` (drift guard) and flatten to a collision-free stem (so the table is 1:1).
    #[test]
    fn adaptable_paths_resolve_and_flatten_uniquely() {
        let mut t = tiny_transformer();
        let paths = t.adaptable_paths();
        assert!(!paths.is_empty());
        // Drift guard: each enumerated path resolves through the matcher.
        for p in &paths {
            assert!(
                resolves(&mut t, p),
                "enumerated {p} does not resolve via adaptable_mut"
            );
        }
        // Globals are excluded from the kohya surface.
        for g in [
            "x_embedder",
            "proj_out",
            "norm_out.linear",
            "time_guidance_embed.linear_1",
        ] {
            assert!(
                !paths.iter().any(|p| p == g),
                "global {g} must be excluded from kohya"
            );
        }
        // Collision-free flattening.
        let flat: std::collections::BTreeSet<String> =
            paths.iter().map(|p| p.replace('.', "_")).collect();
        assert_eq!(
            flat.len(),
            paths.len(),
            "two paths flattened to the same kohya stem"
        );
        // The `to_out` / `to_out.0` aliases both appear (the fork emits both kohya spellings).
        assert!(paths
            .iter()
            .any(|p| p == "transformer_blocks.0.attn.to_out"));
        assert!(paths
            .iter()
            .any(|p| p == "transformer_blocks.0.attn.to_out.0"));
    }

    /// A diffusers-named kohya file applies through the strict provider seam (every stem resolves).
    #[test]
    fn kohya_diffusers_applies() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};

        let small = Array::from_slice(&[0.01f32], &[1, 1]); // [r=1,in=1] / [out=1,r=1]
        let meta = None as Option<&std::collections::HashMap<String, String>>;

        // One kohya key pair per reachable stem.
        let mut t = tiny_transformer();
        let n = t.adaptable_paths().len();
        let mut arrays: Vec<(String, &Array)> = Vec::new();
        for stem in t.adaptable_paths().iter().map(|p| p.replace('.', "_")) {
            arrays.push((format!("lora_unet_{stem}.lora_down.weight"), &small));
            arrays.push((format!("lora_unet_{stem}.lora_up.weight"), &small));
        }
        let refs: Vec<(&str, &Array)> = arrays.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let path = tmp("flux2_kohya_diffusers.safetensors");
        Array::save_safetensors(refs, meta, &path).unwrap();
        let report = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(
            report.applied, n,
            "every diffusers-named kohya stem should resolve"
        );
        assert!(report.unmatched_paths.is_empty());
    }

    // ---- sc-2743 BFL / ComfyUI fused→split routing (no real weights) --------------------------

    /// The full `bfl_targets()` surface: drift guard (every target resolves), count, collision-free
    /// target paths, the fused-qkv 3-way fan-out, and the FLUX.2 single-block `linear1` staying FUSED.
    #[test]
    fn bfl_targets_resolve_full_surface() {
        let mut t = tiny_transformer();
        let targets = t.bfl_targets();
        // 8 globals + (1 double block × 12) + (1 single block × 2) = 22.
        assert_eq!(
            targets.len(),
            22,
            "BFL target count for a 1-double + 1-single tiny transformer"
        );
        // Drift guard: every BFL target path resolves through the matcher.
        for tg in &targets {
            let segs: Vec<&str> = tg.target_path.split('.').collect();
            assert!(
                AdaptableHost::adaptable_mut(&mut t, &segs).is_some(),
                "BFL target {} does not resolve via adaptable_mut",
                tg.target_path
            );
        }
        // Distinct destinations (the qkv fan-out is across DIFFERENT targets, never a collision).
        let distinct: std::collections::BTreeSet<&String> =
            targets.iter().map(|tg| &tg.target_path).collect();
        assert_eq!(distinct.len(), targets.len(), "two BFL targets collide");

        // Fused img-qkv up key feeds exactly to_q/to_k/to_v with Chunk index 0/1/2.
        let qkv_up = "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight";
        let mut fanned: Vec<(String, i32)> = targets
            .iter()
            .filter(|tg| tg.up_keys.iter().any(|k| k == qkv_up))
            .map(|tg| {
                let idx = match &tg.up_slice {
                    Some(LoraRowSlice::Chunk { index, .. }) => *index,
                    _ => panic!("qkv target {} lacks a Chunk up-slice", tg.target_path),
                };
                (tg.target_path.clone(), idx)
            })
            .collect();
        fanned.sort();
        assert_eq!(
            fanned,
            vec![
                ("transformer_blocks.0.attn.to_k".to_string(), 1),
                ("transformer_blocks.0.attn.to_q".to_string(), 0),
                ("transformer_blocks.0.attn.to_v".to_string(), 2),
            ]
        );

        // FLUX.2 single-block `linear1` stays FUSED → maps to `to_qkv_mlp_proj` with NO slice.
        let l1 = targets
            .iter()
            .find(|tg| tg.target_path == "single_transformer_blocks.0.attn.to_qkv_mlp_proj")
            .expect("single linear1 target");
        assert!(
            l1.up_slice.is_none() && l1.down_slice.is_none(),
            "FLUX.2 single linear1 must not split (it is fused in the model)"
        );
    }

    /// sc-2743 gate at the FLUX.2 dispatch level: a BFL *fused* qkv kohya file resolves and installs
    /// the BYTE-IDENTICAL `to_q/to_k/to_v` adapters as the equivalent diffusers split-target file
    /// (the diffusers path is fork-verified, sc-2646 → transitively the BFL path matches the fork).
    #[test]
    fn bfl_fused_qkv_resolves_and_splits_like_diffusers() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::adapters::Adapter;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let meta = None as Option<&std::collections::HashMap<String, String>>;

        // out=2 per head, 3 heads → fused up [6,1]; r=1 (not ÷3) → shared down [1,in=2]; alpha=4.
        let (inner, inp, r) = (2i32, 2i32, 1i32);
        let bq = [0.10f32, 0.11];
        let bk = [0.20f32, 0.21];
        let bv = [0.30f32, 0.31];
        let mut fused = bq.to_vec();
        fused.extend_from_slice(&bk);
        fused.extend_from_slice(&bv);
        let b_fused = Array::from_slice(&fused, &[3 * inner, r]);
        let b_q = Array::from_slice(&bq, &[inner, r]);
        let b_k = Array::from_slice(&bk, &[inner, r]);
        let b_v = Array::from_slice(&bv, &[inner, r]);
        let a = Array::from_slice(&[0.5f32, -0.5], &[r, inp]);
        let alpha = Array::from_slice(&[4.0f32], &[1]);

        let bpath = tmp("flux2_bfl_qkv.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_up.weight",
                    &b_fused,
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_qkv.lora_down.weight",
                    &a,
                ),
                ("lora_unet_double_blocks_0_img_attn_qkv.alpha", &alpha),
            ],
            meta,
            &bpath,
        )
        .unwrap();
        let mut tb = tiny_transformer();
        let rb = apply_flux2_adapters(
            &mut tb,
            &[AdapterSpec {
                path: bpath,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(rb.applied, 3, "one fused qkv → three split targets");
        assert!(rb.unmatched_paths.is_empty());

        // Equivalent diffusers split-target file: per-head up, SHARED down, same alpha.
        let ppath = tmp("flux2_bfl_split_peft.safetensors");
        Array::save_safetensors(
            vec![
                (
                    "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
                    &b_q,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
                    &a,
                ),
                ("transformer.transformer_blocks.0.attn.to_q.alpha", &alpha),
                (
                    "transformer.transformer_blocks.0.attn.to_k.lora_B.weight",
                    &b_k,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_k.lora_A.weight",
                    &a,
                ),
                ("transformer.transformer_blocks.0.attn.to_k.alpha", &alpha),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_B.weight",
                    &b_v,
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_v.lora_A.weight",
                    &a,
                ),
                ("transformer.transformer_blocks.0.attn.to_v.alpha", &alpha),
            ],
            meta,
            &ppath,
        )
        .unwrap();
        let mut tp = tiny_transformer();
        apply_flux2_adapters(
            &mut tp,
            &[AdapterSpec {
                path: ppath,
                scale: 0.8,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();

        for tgt in ["to_q", "to_k", "to_v"] {
            let segs = ["transformer_blocks", "0", "attn", tgt];
            let pull = |t: &mut Flux2Transformer| match AdaptableHost::adaptable_mut(t, &segs)
                .unwrap()
                .adapters()
            {
                [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
                _ => panic!("expected one LoRA at {tgt}"),
            };
            let (ba, bb) = pull(&mut tb);
            let (pa, pb) = pull(&mut tp);
            assert!(
                mlx_rs::ops::array_eq(&ba, &pa, false)
                    .unwrap()
                    .item::<bool>()
                    && mlx_rs::ops::array_eq(&bb, &pb, false)
                        .unwrap()
                        .item::<bool>(),
                "BFL split and diffusers split installed different adapters at {tgt}"
            );
        }
    }

    /// sc-2743: BFL plain renames resolve across all three prefix conventions — `base_model.model.`
    /// globals (`img_in`→`x_embedder`, `final_layer.linear`→`proj_out`), a `diffusion_model.` dotted
    /// block (`…img_attn.proj`→`to_out`), and a `base_model.model.` dotted single block
    /// (`…linear1`→`to_qkv_mlp_proj`).
    #[test]
    fn bfl_renames_and_prefixes_resolve() {
        use crate::adapters::apply_flux2_adapters;
        use mlx_gen::runtime::{AdapterKind, AdapterSpec};
        let meta = None as Option<&std::collections::HashMap<String, String>>;
        let s = Array::from_slice(&[0.01f32], &[1, 1]);
        let path = tmp("flux2_bfl_renames.safetensors");
        Array::save_safetensors(
            vec![
                ("base_model.model.img_in.lora_A.weight", &s),
                ("base_model.model.img_in.lora_B.weight", &s),
                ("base_model.model.final_layer.linear.lora_A.weight", &s),
                ("base_model.model.final_layer.linear.lora_B.weight", &s),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight",
                    &s,
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight",
                    &s,
                ),
                ("base_model.model.single_blocks.0.linear1.lora_A.weight", &s),
                ("base_model.model.single_blocks.0.linear1.lora_B.weight", &s),
            ],
            meta,
            &path,
        )
        .unwrap();
        let mut t = tiny_transformer();
        let rep = apply_flux2_adapters(
            &mut t,
            &[AdapterSpec {
                path,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .unwrap();
        assert_eq!(
            rep.applied, 4,
            "all four BFL renames across the prefix conventions resolve"
        );
        assert!(rep.unmatched_paths.is_empty());
    }
}
