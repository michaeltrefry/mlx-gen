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

use std::f32::consts::LN_10;

use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, subtract};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::array::scalar;
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Flux2Config;
use crate::pos_embed::Flux2PosEmbed;

const LN_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-5;

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
        let out0 = subtract(&multiply(&real, &cos)?, &multiply(&imag, &sin)?)?;
        let out1 = add(&multiply(&imag, &cos)?, &multiply(&real, &sin)?)?;
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

/// SwiGLU: split last axis in half, `silu(x1) · x2`.
fn swiglu(x: &Array) -> Result<Array> {
    let p = split(x, 2, -1)?;
    Ok(multiply(&silu(&p[0])?, &p[1])?)
}

/// `(1 + scale) · norm(x) + shift` with `scale`/`shift` broadcast `[B,1,D]`.
fn modulate(norm: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(&multiply(norm, &add(scale, scalar(1.0))?)?, shift)?)
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

    /// Joint attention. Returns `(img_attn_out, txt_attn_out)`.
    fn forward(
        &self,
        img: &Array,
        txt: &Array,
        cos: &Array,
        sin: &Array,
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

        let (img_attn, txt_attn) = self.attn.forward(&norm_img, &norm_txt, cos, sin)?;
        img = add(&img, &multiply(gate_msa, &img_attn)?)?;
        txt = add(&txt, &multiply(c_gate_msa, &txt_attn)?)?;

        let norm_img2 = modulate(&layer_norm(&img, None, None, LN_EPS)?, scale_mlp, shift_mlp)?;
        img = add(&img, &multiply(gate_mlp, &self.ff.forward(&norm_img2)?)?)?;

        let norm_txt2 = modulate(
            &layer_norm(&txt, None, None, LN_EPS)?,
            c_scale_mlp,
            c_shift_mlp,
        )?;
        txt = add(
            &txt,
            &multiply(c_gate_mlp, &self.ff_context.forward(&norm_txt2)?)?,
        )?;

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

    /// `mod`: `(shift, scale, gate)`.
    fn forward(
        &self,
        hidden: &Array,
        m: &(Array, Array, Array),
        cos: &Array,
        sin: &Array,
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
        let attn = attention(&q, &k, &v, self.head_dim)?;

        let mlp = swiglu(&mlp)?;
        let cat = concatenate_axis(&[&attn, &mlp], -1)?;
        let attn_output = self.to_out.forward(&cat)?;
        Ok(add(hidden, &multiply(gate, &attn_output)?)?)
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
    /// scaled sigma (×1000). Returns the velocity `[B, seq_img, out_channels]`.
    pub fn forward(
        &self,
        hidden_states: &Array,
        encoder_hidden_states: &Array,
        img_ids: &Array,
        txt_ids: &Array,
        timestep: f32,
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

        for block in &self.double_blocks {
            (txt, img) = block.forward(img, txt, &img_mod, &txt_mod, &cos, &sin)?;
        }

        let txt_seq = txt.shape()[1];
        let mut hidden = concatenate_axis(&[&txt, &img], 1)?;
        let ms = self.mod_single.forward(&temb)?;
        for block in &self.single_blocks {
            hidden = block.forward(&hidden, &ms[0], &cos, &sin)?;
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
}
