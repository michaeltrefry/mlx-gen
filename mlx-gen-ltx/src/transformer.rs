//! S3 — the LTX-2.3 **DiT** (video stack): the preprocessor (patchify + adaLN-single) → 48 ×
//! `BasicAVTransformerBlock` (video-only path) → output projection → velocity. Port of the
//! `mlx_video` reference `models/ltx/{transformer,attention,adaln,feed_forward,ltx}.py`.
//!
//! Per-block math (S3a): **gated attention** (`to_gate_logits → 2·sigmoid`, zero-init identity), q/k
//! **RMSNorm** over the full inner_dim (pre-head, learned), **SPLIT 3-D RoPE** on q/k (reusing the S0
//! [`crate::rope`]), SDPA, `to_out`; **adaLN-single** with the 9-row `scale_shift_table` (gated 2.3
//! family: MSA rows 0..3, FF rows 3..6, text-cross-attn rows 6..9); **prompt adaLN** modulating the
//! text context; **FeedForward** = `proj_in → gelu(tanh) → proj_out`.
//!
//! Full forward (S3b): patchify_proj → adaLN-single (timestep → 9·dim) + prompt-adaLN (→ 2·dim) →
//! caption projection (Identity for 2.3) → SPLIT RoPE from the position grid → 48 blocks → output
//! (`LayerNorm` affine-false + final 2-row `scale_shift_table` modulated by the embedded timestep) →
//! `proj_out → 128` velocity. `denoised = latent − σ·velocity`.
//!
//! **Quant.** The shipped `base_q8` transformer stores the attn/ff Linears Q8-quantized (U32 +
//! `scales` + `biases`, group 64) — there is no dense bf16 checkpoint. [`Precision::F32Q8`] is the
//! production path: **f32 activations × Q8 `quantized_matmul`** (the port's quality target; a single
//! block is bit-exact to the reference at matched mlx 0.31.2). [`Precision::F32`] additionally
//! dequantizes the Q8 weights to dense f32 — the S3a block math gate. [`Precision::Bf16Q8`] mirrors
//! the reference's own bf16 compute, retained for diagnostics.

use mlx_rs::fast::{layer_norm, rms_norm as fast_rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{
    add, concatenate_axis, dequantize, divide, multiply, quantized_matmul, sigmoid, subtract,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{gelu_tanh, linear};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::Result;

use crate::config::LtxConfig;
use crate::rope::{apply_split_rotary_emb, precompute_split_freqs_cis};

/// Q8 quant config of the shipped transformer (`split_model.json`: bits 8, group 64).
const QUANT_BITS: i32 = 8;
const QUANT_GROUP: i32 = 64;
/// adaLN-single sinusoidal timestep projection width (PixArt `Timesteps`).
const TIME_PROJ_DIM: i32 = 256;

/// Compute precision for the DiT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// f32 activations, Q8 weights **dequantized** to dense f32 — the S3a block math gate (small).
    F32,
    /// **f32 activations × Q8 `quantized_matmul`** (dense weights f32) — the production path and the
    /// port's quality target. The **full 48-layer velocity is bit-exact** to the reference (mlx
    /// 0.31.2) — required because the distilled stage-1 sampler is chaos-sensitive (any per-forward
    /// seed amplifies to a large latent divergence; sc-2842). The bf16 alternative drifts ~3e-2 over
    /// the 48-layer residual stream, amplified by the output LayerNorm.
    F32Q8,
    /// bf16 activations × Q8 `quantized_matmul` (dense bf16 elsewhere) — matches the reference's own
    /// compute dtype; retained for reference/diagnostics.
    Bf16Q8,
}

impl Precision {
    fn dtype(self) -> Dtype {
        match self {
            Precision::F32 | Precision::F32Q8 => Dtype::Float32,
            Precision::Bf16Q8 => Dtype::Bfloat16,
        }
    }

    /// Whether Q8 weights are kept quantized (`quantized_matmul`) vs dequantized to dense f32.
    fn keep_quant(self) -> bool {
        matches!(self, Precision::F32Q8 | Precision::Bf16Q8)
    }
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

/// Load a non-Linear param (norm weight, scale-shift table) cast to the compute dtype.
fn param(w: &Weights, key: &str, prec: Precision) -> Result<Array> {
    to_dtype(w.require(key)?, prec.dtype())
}

/// `x · (1 + scale) + shift` (adaLN modulation), broadcasting `scale`/`shift` `(B, S', dim)` over the
/// token axis.
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    Ok(add(
        &multiply(x, &add(scale, scalar(1.0).as_dtype(scale.dtype())?)?)?,
        shift,
    )?)
}

/// A Linear — dense or Q8-quantized, selected by [`Precision`] at load.
enum Linear {
    Dense {
        w: Array, // [out, in]
        b: Array, // [out]
    },
    Quant {
        q: Array,      // [out, in_packed] U32
        scales: Array, // [out, in/group]
        biases: Array,
        b: Array,
        group: i32,
        bits: i32,
    },
}

impl Linear {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        let dt = prec.dtype();
        let b = to_dtype(w.require(&format!("{prefix}.bias"))?, dt)?;
        match w.get(&format!("{prefix}.scales")) {
            Some(scales) => {
                let q = w.require(&format!("{prefix}.weight"))?;
                let biases = w.require(&format!("{prefix}.biases"))?;
                if prec.keep_quant() {
                    // Keep Q8; `quantized_matmul` dequantizes on the fly with fp32 accumulation and
                    // is correct for f32 *or* bf16 activations (the Z-Image/Qwen Q8 path). Scales /
                    // biases are cast to the compute dtype so the on-the-fly dequant matches the
                    // reference's (f32 for F32Q8 — a lossless upcast of the bf16 file scales).
                    Ok(Linear::Quant {
                        q: q.clone(),
                        scales: to_dtype(scales, dt)?,
                        biases: to_dtype(biases, dt)?,
                        b,
                        group: QUANT_GROUP,
                        bits: QUANT_BITS,
                    })
                } else {
                    // Dequantize to dense f32 (bit-identical to the reference's mx.dequantize).
                    let dense =
                        dequantize(q, scales, Some(biases), Some(QUANT_GROUP), Some(QUANT_BITS))?;
                    Ok(Linear::Dense {
                        w: to_dtype(&dense, Dtype::Float32)?,
                        b,
                    })
                }
            }
            None => Ok(Linear::Dense {
                w: to_dtype(w.require(&format!("{prefix}.weight"))?, dt)?,
                b,
            }),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Linear::Dense { w, b } => linear(x, w, b),
            Linear::Quant {
                q,
                scales,
                biases,
                b,
                group,
                bits,
            } => Ok(add(
                &quantized_matmul(x, q, scales, biases, true, *group, *bits)?,
                b,
            )?),
        }
    }
}

/// `mx.fast.rms_norm(x, ones, eps)` — the block's weightless pre-norm (feature RMS over the last axis).
fn rms_norm_noweight(x: &Array, eps: f32) -> Result<Array> {
    let dim = *x.shape().last().unwrap();
    let ones = Array::ones::<f32>(&[dim])?.as_dtype(x.dtype())?;
    Ok(fast_rms_norm(x, &ones, eps)?)
}

/// Multi-head attention with q/k RMSNorm, optional SPLIT RoPE, optional per-head gating. Self-attn
/// when `context` is `None`; cross-attn otherwise. RoPE `(cos, sin)` applies to q **and** k (self-attn
/// only; cross-attn passes `pe = None`).
struct Attention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    q_norm: Array,
    k_norm: Array,
    to_out: Linear,
    gate: Option<Linear>,
    heads: i32,
    dim_head: i32,
    eps: f32,
}

impl Attention {
    fn load(w: &Weights, prefix: &str, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let gate = if w.get(&format!("{prefix}.to_gate_logits.weight")).is_some() {
            Some(Linear::load(w, &format!("{prefix}.to_gate_logits"), prec)?)
        } else {
            None
        };
        Ok(Self {
            to_q: Linear::load(w, &format!("{prefix}.to_q"), prec)?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"), prec)?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"), prec)?,
            q_norm: param(w, &format!("{prefix}.q_norm.weight"), prec)?,
            k_norm: param(w, &format!("{prefix}.k_norm.weight"), prec)?,
            to_out: Linear::load(w, &format!("{prefix}.to_out"), prec)?,
            gate,
            heads: cfg.num_attention_heads,
            dim_head: cfg.attention_head_dim,
            eps: cfg.norm_eps as f32,
        })
    }

    /// `(B, S, inner)` → `(B, H, S, head_dim)`.
    fn to_heads(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        Ok(x.reshape(&[b, s, self.heads, self.dim_head])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }

    fn forward(
        &self,
        x: &Array,
        context: Option<&Array>,
        mask: Option<&Array>,
        pe: Option<(&Array, &Array)>,
    ) -> Result<Array> {
        let ctx = context.unwrap_or(x);
        let q = fast_rms_norm(&self.to_q.forward(x)?, &self.q_norm, self.eps)?;
        let k = fast_rms_norm(&self.to_k.forward(ctx)?, &self.k_norm, self.eps)?;
        let v = self.to_v.forward(ctx)?;

        let mut qh = self.to_heads(&q)?;
        let mut kh = self.to_heads(&k)?;
        let vh = self.to_heads(&v)?;
        if let Some((cos, sin)) = pe {
            qh = apply_split_rotary_emb(&qh, cos, sin)?;
            kh = apply_split_rotary_emb(&kh, cos, sin)?;
        }

        // Match the reference's Python `1.0 / math.sqrt(dim_head)` (f64 → f32), not `d^-0.5` in f32.
        let scale = (1.0f64 / (self.dim_head as f64).sqrt()) as f32;
        let out = match mask {
            Some(m) => scaled_dot_product_attention(&qh, &kh, &vh, scale, m, None)?,
            None => scaled_dot_product_attention(&qh, &kh, &vh, scale, None, None)?,
        };
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let inner = self.heads * self.dim_head;
        let mut out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, inner])?;

        if let Some(gate) = &self.gate {
            // Per-head gate: 2·sigmoid(logits) (zero-init → identity), broadcast over head_dim.
            let logits = gate.forward(x)?;
            let gates = multiply(&sigmoid(&logits)?, scalar(2.0).as_dtype(logits.dtype())?)?;
            let gates = gates.reshape(&[b, s, self.heads, 1])?;
            out = multiply(&out.reshape(&[b, s, self.heads, self.dim_head])?, &gates)?
                .reshape(&[b, s, inner])?;
        }
        self.to_out.forward(&out)
    }
}

/// `proj_in → gelu(tanh) → proj_out`.
struct FeedForward {
    proj_in: Linear,
    proj_out: Linear,
}

impl FeedForward {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        Ok(Self {
            proj_in: Linear::load(w, &format!("{prefix}.proj_in"), prec)?,
            proj_out: Linear::load(w, &format!("{prefix}.proj_out"), prec)?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.proj_out
            .forward(&gelu_tanh(&self.proj_in.forward(x)?)?)
    }
}

/// `table[row] + timestep_proj[row]` for `row ∈ [lo, hi)`. `table` is `(num_ada, dim)`; `timestep` is
/// `(B, S', num_ada·dim)`. Returns the `hi−lo` modulation tensors, each `(B, S', dim)`.
fn ada_values(table: &Array, timestep: &Array, lo: i32, hi: i32) -> Result<Vec<Array>> {
    let num_ada = table.shape()[0];
    let dim = table.shape()[1];
    let ts = timestep.shape();
    let (b, s) = (ts[0], ts[1]);
    let ts4 = timestep.reshape(&[b, s, num_ada, dim])?;
    let mut out = Vec::with_capacity((hi - lo) as usize);
    for row in lo..hi {
        let trow = table.index_axis(row, 0)?.reshape(&[1, 1, dim])?;
        let tsrow = ts4.index_axis(row, 2)?;
        out.push(add(&trow, &tsrow)?);
    }
    Ok(out)
}

/// Index a single position `i` along `axis`, dropping that axis.
trait IndexAxis {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array>;
}
impl IndexAxis for Array {
    fn index_axis(&self, i: i32, axis: i32) -> Result<Array> {
        Ok(self.take_axis(Array::from_int(i), axis)?)
    }
}

/// One video transformer block (`BasicAVTransformerBlock`, video-only / gated 2.3 path).
pub struct VideoBlock {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    scale_shift_table: Array,        // (9, inner)
    prompt_scale_shift_table: Array, // (2, inner)
    eps: f32,
}

impl VideoBlock {
    /// Load a block (`prefix` e.g. `transformer_blocks.0`) at the given [`Precision`].
    pub fn load(w: &Weights, prefix: &str, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), cfg, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), cfg, prec)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"), prec)?,
            scale_shift_table: param(w, &format!("{prefix}.scale_shift_table"), prec)?,
            prompt_scale_shift_table: param(
                w,
                &format!("{prefix}.prompt_scale_shift_table"),
                prec,
            )?,
            eps: cfg.norm_eps as f32,
        })
    }

    /// Forward (gated, 9-row table): MSA(self, RoPE) → text cross-attn (prompt-modulated context) →
    /// FeedForward, each adaLN-modulated and gated.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x: &Array,
        timesteps: &Array,
        prompt_timestep: Option<&Array>,
        context: &Array,
        mask: Option<&Array>,
        cos: &Array,
        sin: &Array,
    ) -> Result<Array> {
        // --- MSA (self-attention) ---
        let msa = ada_values(&self.scale_shift_table, timesteps, 0, 3)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = self.attn1.forward(&norm, None, None, Some((cos, sin)))?;
        let mut x = add(x, &multiply(&attn, &msa[2])?)?;

        // --- prompt-adaLN on the text context ---
        let v_context = {
            let (p_shift, p_scale) = match prompt_timestep {
                Some(pt) => {
                    let p = ada_values(&self.prompt_scale_shift_table, pt, 0, 2)?;
                    (p[0].clone(), p[1].clone())
                }
                None => (
                    self.prompt_scale_shift_table.index_axis(0, 0)?,
                    self.prompt_scale_shift_table.index_axis(1, 0)?,
                ),
            };
            modulate(context, &p_scale, &p_shift)?
        };

        // --- text cross-attention (adaLN rows 6..9) ---
        let ca = ada_values(&self.scale_shift_table, timesteps, 6, 9)?;
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = self.attn2.forward(&norm_ca, Some(&v_context), mask, None)?;
        x = add(&x, &multiply(&cross, &ca[2])?)?;

        // --- FeedForward (adaLN rows 3..6) ---
        let mlp = ada_values(&self.scale_shift_table, timesteps, 3, 6)?;
        let norm_mlp = modulate(&rms_norm_noweight(&x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff = self.ff.forward(&norm_mlp)?;
        x = add(&x, &multiply(&ff, &mlp[2])?)?;

        Ok(x)
    }
}

/// PixArt sinusoidal timestep embedding (`flip_sin_to_cos`, `downscale_freq_shift = 0`, max_period
/// 10000): `concat([cos(t·f), sin(t·f)])` with `f[i] = exp(−ln(10000)·i/half)`. `timesteps` is `(N,)`
/// f32; returns `(N, TIME_PROJ_DIM)` f32.
///
/// The log-spaced freqs are computed in **MLX float32** (`arange → ×(−ln θ) → ÷half → exp`), mirroring
/// the reference `get_timestep_embedding` op-for-op. A host-f64 table (the obvious shortcut) diverges
/// ~1e-7 per element from the MLX-f32 kernels (88/128 freqs differ; the projection differs up to
/// ~5e-5 after ×1000 + cos/sin) — invisible in bf16 but, in the F32Q8 path, this adaLN timestep
/// embedding modulates **every** block and the sub-ULP seed compounds across the 48-layer residual
/// stream into a percent-level velocity divergence that the distilled stage-1 sampler then amplifies.
/// (RoPE, by contrast, follows the reference's own numpy-f64 path — see [`crate::rope`].)
fn timestep_embedding(timesteps: &Array) -> Result<Array> {
    let half = TIME_PROJ_DIM / 2; // 128
    let neg_ln = -(10000f64).ln() as f32;
    let exponent = divide(
        &multiply(&Array::arange::<_, f32>(None, half, None)?, scalar(neg_ln))?,
        scalar(half as f32),
    )?;
    let freq = exponent.exp()?.reshape(&[1, half])?; // (1, half)
    let emb = multiply(&timesteps.reshape(&[-1, 1])?, &freq)?; // (N, half)
    Ok(concatenate_axis(&[&emb.cos()?, &emb.sin()?], 1)?) // (N, dim), cos first
}

/// adaLN-single (`AdaLayerNormSingle`): `timestep → sinusoidal(256) → MLP(silu) → embedded`, then
/// `linear(silu(embedded)) → coeff·dim` scale-shift parameters.
struct AdaLayerNormSingle {
    ts_lin1: Linear, // 256 → dim
    ts_lin2: Linear, // dim → dim
    linear: Linear,  // dim → coeff·dim
}

impl AdaLayerNormSingle {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        Ok(Self {
            ts_lin1: Linear::load(w, &format!("{prefix}.emb.timestep_embedder.linear1"), prec)?,
            ts_lin2: Linear::load(w, &format!("{prefix}.emb.timestep_embedder.linear2"), prec)?,
            linear: Linear::load(w, &format!("{prefix}.linear"), prec)?,
        })
    }

    /// `timestep` is the already-scaled `(N,)` f32. Returns `(scale_shift (N, coeff·dim), embedded
    /// (N, dim))` in `dt`.
    fn forward(&self, timestep: &Array, dt: Dtype) -> Result<(Array, Array)> {
        let proj = timestep_embedding(timestep)?.as_dtype(dt)?;
        let h = mlx_gen::nn::silu(&self.ts_lin1.forward(&proj)?)?;
        let embedded = self.ts_lin2.forward(&h)?;
        let scale_shift = self.linear.forward(&mlx_gen::nn::silu(&embedded)?)?;
        Ok((scale_shift, embedded))
    }
}

/// The LTX-2.3 video DiT: preprocessor + 48 blocks + output projection. Predicts velocity.
pub struct LtxDiT {
    patchify_proj: Linear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: Option<AdaLayerNormSingle>,
    blocks: Vec<VideoBlock>,
    scale_shift_table: Array, // (2, inner)
    proj_out: Linear,
    cfg: LtxConfig,
    prec: Precision,
}

impl LtxDiT {
    pub fn from_weights(w: &Weights, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let blocks = (0..cfg.num_layers)
            .map(|i| VideoBlock::load(w, &format!("transformer_blocks.{i}"), cfg, prec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patchify_proj: Linear::load(w, "patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "adaln_single", prec)?,
            prompt_adaln: if cfg.apply_gated_attention {
                Some(AdaLayerNormSingle::load(w, "prompt_adaln_single", prec)?)
            } else {
                None
            },
            blocks,
            scale_shift_table: param(w, "scale_shift_table", prec)?,
            proj_out: Linear::load(w, "proj_out", prec)?,
            cfg: cfg.clone(),
            prec,
        })
    }

    /// The preprocessor (mirrors the reference `TransformerArgsPreprocessor.prepare`): patchify_proj →
    /// adaLN-single timestep projection + prompt-adaLN → caption_projection (Identity, 2.3) → SPLIT
    /// RoPE tables. Shared by [`forward`](Self::forward) and [`block_hidden`](Self::block_hidden).
    fn preprocess(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        positions: &Array,
    ) -> Result<Preprocessed> {
        let dt = self.prec.dtype();
        let b = latent.shape()[0];
        let inner = self.cfg.inner_dim();
        let coeff = self.cfg.adaln_embedding_coefficient;

        let x = self.patchify_proj.forward(&latent.as_dtype(dt)?)?;

        // adaLN-single timestep projection. The `× timestep_scale_multiplier` runs in the **input
        // dtype** (matching `denoise_av`, which feeds a latent-dtype timestep): the adaLN sinusoid
        // upcasts to f32 internally, but a bf16 timestep must round `bf16(σ·1000)` *first* — pre-
        // upcasting to f32 would change the high-frequency sinusoid phase (~33% velocity divergence
        // in the bf16 path). f32 input is unaffected (`f32(σ)·1000` either way).
        let mult = scalar(self.cfg.timestep_scale_multiplier as f32).as_dtype(timestep.dtype())?;
        let ts_flat = multiply(timestep, &mult)?.reshape(&[-1])?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, dt)?;
        let ts_emb = ts_emb.reshape(&[b, -1, coeff * inner])?;
        let emb_ts = emb_ts.reshape(&[b, -1, inner])?;

        // prompt-adaLN (gated family): one shared modulation per sample.
        let prompt_ts = match &self.prompt_adaln {
            Some(padaln) => {
                let src = if timestep.ndim() > 1 {
                    timestep.index_axis(0, 1)?.reshape(&[b, 1])?
                } else {
                    timestep.clone()
                };
                let src = multiply(&src, &mult)?.reshape(&[-1])?;
                let (pts, _) = padaln.forward(&src, dt)?;
                Some(pts.reshape(&[b, -1, 2 * inner])?)
            }
            None => None,
        };

        // caption_projection = Identity (2.3): context enters cross-attn as-is.
        let context = context.as_dtype(dt)?;

        // SPLIT RoPE from the position grid (f32 tables; the block casts per input dtype).
        let (cos, sin) = precompute_split_freqs_cis(
            positions,
            inner,
            self.cfg.positional_embedding_theta,
            self.cfg.positional_embedding_max_pos,
            self.cfg.num_attention_heads,
        )?;

        Ok(Preprocessed {
            x,
            ts_emb,
            emb_ts,
            prompt_ts,
            context,
            cos,
            sin,
        })
    }

    /// Velocity forward.
    ///
    /// * `latent` — `(B, S, in_channels=128)` patchified latent tokens.
    /// * `timestep` — `(B, 1)` (or `(B,)`) per-sample sigma (T2V; broadcast over tokens).
    /// * `context` — `(B, ctx, inner)` text embeddings (connector output); `mask` its additive mask.
    /// * `positions` — `(B, 3, S, 2)` position grid (from [`crate::positions`]).
    pub fn forward(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
    ) -> Result<Array> {
        let p = self.preprocess(latent, timestep, context, positions)?;
        let mut h = p.x;
        for block in &self.blocks {
            h = block.forward(
                &h,
                &p.ts_emb,
                p.prompt_ts.as_ref(),
                &p.context,
                mask,
                &p.cos,
                &p.sin,
            )?;
        }
        self.output_head(&h, &p.emb_ts)
    }

    /// Diagnostic: run the preprocessor + the first `n` blocks and return the hidden state (for the
    /// per-block bisection of the e2e residual). `n == blocks.len()` is the full pre-output hidden.
    #[doc(hidden)]
    pub fn block_hidden(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        mask: Option<&Array>,
        positions: &Array,
        n: usize,
    ) -> Result<Array> {
        let p = self.preprocess(latent, timestep, context, positions)?;
        let mut h = p.x;
        for block in self.blocks.iter().take(n) {
            h = block.forward(
                &h,
                &p.ts_emb,
                p.prompt_ts.as_ref(),
                &p.context,
                mask,
                &p.cos,
                &p.sin,
            )?;
        }
        Ok(h)
    }
}

/// The [`LtxDiT::preprocess`] outputs threaded into the block stack + output head.
struct Preprocessed {
    x: Array,
    ts_emb: Array,
    emb_ts: Array,
    prompt_ts: Option<Array>,
    context: Array,
    cos: Array,
    sin: Array,
}

impl LtxDiT {
    /// The output head in isolation (LayerNorm-affine-false → final scale-shift → proj_out), for the
    /// S3b bisection: feed the reference post-block hidden + embedded timestep, compare the velocity.
    pub fn output_head(&self, h: &Array, emb_ts: &Array) -> Result<Array> {
        let b = h.shape()[0];
        let inner = self.cfg.inner_dim();
        let table = self.scale_shift_table.reshape(&[1, 1, 2, inner])?;
        let ss = add(&table, &emb_ts.reshape(&[b, -1, 1, inner])?)?;
        let shift = ss.index_axis(0, 2)?;
        let scale = ss.index_axis(1, 2)?;
        let normed = layer_norm(h, None, None, self.cfg.norm_eps as f32)?;
        let out = modulate(&normed, &scale, &shift)?;
        self.proj_out.forward(&out)
    }
}

/// `denoised = latent − σ·velocity` (`to_denoised`): velocity → x₀.
pub fn to_denoised(latent: &Array, velocity: &Array, sigma: &Array) -> Result<Array> {
    Ok(subtract(latent, &multiply(velocity, sigma)?)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modulate_closed_form() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let scale = Array::from_slice(&[0.0f32, 1.0, 0.0, 1.0], &[1, 1, 4]);
        let shift = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0], &[1, 1, 4]);
        let got = modulate(&x, &scale, &shift).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[2.0, 4.0, 2.0, 8.0]);
    }

    #[test]
    fn ada_values_splits_rows() {
        let table = Array::from_slice(&(0..18).map(|v| v as f32).collect::<Vec<_>>(), &[9, 2]);
        let ts = Array::zeros::<f32>(&[1, 1, 18]).unwrap();
        let vals = ada_values(&table, &ts, 0, 3).unwrap();
        assert_eq!(vals.len(), 3);
        assert_eq!(vals[0].as_slice::<f32>(), &[0.0, 1.0]);
        assert_eq!(vals[2].as_slice::<f32>(), &[4.0, 5.0]);
    }

    #[test]
    fn timestep_embedding_shape_and_pad() {
        // (N=2,) → (2, 256), cos-first (t=0 → cos=1, sin=0).
        let t = Array::from_slice(&[0.0f32, 1.0], &[2]);
        let emb = timestep_embedding(&t).unwrap();
        assert_eq!(emb.shape(), &[2, 256]);
        let row0 = emb.index_axis(0, 0).unwrap();
        let s = row0.as_slice::<f32>();
        assert!((s[0] - 1.0).abs() < 1e-6); // cos(0)
        assert!(s[128].abs() < 1e-6); // sin(0)
    }
}
