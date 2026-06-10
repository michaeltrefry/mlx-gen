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
//! **Quant.** The shipped transformer stores the attn/ff Linears selectively quantized (U32 +
//! `scales` + `biases`) — there is no dense bf16 checkpoint. The **bits/group ride on the checkpoint's
//! `split_model.json`** ([`crate::config::SplitModel`]): `base_q8` at 8 bits, `base_q4` at 4 bits,
//! group 64 — read into [`Precision`], never hardcoded (sc-2686). The per-Linear predicate
//! (quantize iff the weights carry `.scales`) mirrors `generate_av.py`'s `_should_quantize`.
//!
//! [`Precision::quant_f32`] is the production quality target: **f32 activations × `quantized_matmul`**
//! (a single block is bit-exact to the reference at matched mlx 0.31.2). [`Precision::quant_bf16`]
//! mirrors the reference's own bf16 compute (the production-speed path). [`Precision::dense_f32`]
//! additionally dequantizes the weights to dense f32 — the S3a block-math gate.

use std::cell::{Cell, RefCell};

use mlx_rs::error::Exception;
use mlx_rs::fast::{layer_norm, rms_norm as fast_rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{
    add, concatenate_axis, dequantize, divide, matmul, multiply, power, quantized_matmul, sigmoid,
    subtract, tanh,
};
use mlx_rs::transforms::compile::compile;
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::{gelu_tanh, linear};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::Result;

use crate::config::LtxConfig;
use crate::rope::{apply_split_rotary_emb, precompute_split_freqs_cis};

/// adaLN-single sinusoidal timestep projection width (PixArt `Timesteps`).
const TIME_PROJ_DIM: i32 = 256;

/// How to run the (selectively quantized) DiT: the activation/compute dtype, whether quantized
/// weights stay packed (`quantized_matmul`) or are dequantized to dense, and the **checkpoint's**
/// quant geometry (`bits`/`group` from `split_model.json` — so Q4 and Q8 both load without a code
/// change; sc-2686). Construct via [`quant_f32`](Self::quant_f32) / [`quant_bf16`](Self::quant_bf16)
/// / [`dense_f32`](Self::dense_f32).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Precision {
    mode: Mode,
    bits: i32,
    group: i32,
}

/// The compute mode (independent of the quant bit-width, which rides alongside in [`Precision`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// f32 activations, quantized weights **dequantized** to dense f32 — the S3a block-math gate.
    DenseF32,
    /// **f32 activations × `quantized_matmul`** — the production path / quality target. The full
    /// 48-layer velocity is bit-exact to the reference (mlx 0.31.2), required because the distilled
    /// stage-1 sampler is chaos-sensitive (sc-2842).
    QuantF32,
    /// bf16 activations × `quantized_matmul` — the reference's own compute dtype (production speed).
    QuantBf16,
}

impl Precision {
    /// f32 activations, quantized weights dequantized to dense f32 (the block-math gate).
    pub fn dense_f32(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::DenseF32,
            bits,
            group,
        }
    }

    /// f32 activations × `quantized_matmul` (the production quality target).
    pub fn quant_f32(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::QuantF32,
            bits,
            group,
        }
    }

    /// bf16 activations × `quantized_matmul` (the reference's native production-speed path).
    pub fn quant_bf16(bits: i32, group: i32) -> Self {
        Self {
            mode: Mode::QuantBf16,
            bits,
            group,
        }
    }

    fn dtype(self) -> Dtype {
        match self.mode {
            Mode::DenseF32 | Mode::QuantF32 => Dtype::Float32,
            Mode::QuantBf16 => Dtype::Bfloat16,
        }
    }

    /// Whether quantized weights are kept packed (`quantized_matmul`) vs dequantized to dense f32.
    fn keep_quant(self) -> bool {
        matches!(self.mode, Mode::QuantF32 | Mode::QuantBf16)
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
/// token axis. One fused kernel when the sc-2963 glue toggle is on (the `1` is cast to `scale`'s dtype
/// inside, as before — bit-identical and dtype-preserving).
fn modulate(x: &Array, scale: &Array, shift: &Array) -> Result<Array> {
    let f = |(x, sc, sh): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(
            &multiply(x, &add(sc, &scalar(1.0).as_dtype(sc.dtype())?)?)?,
            sh,
        )
    };
    if crate::compile_glue() {
        Ok(compile(f, true)((x, scale, shift))?)
    } else {
        Ok(f((x, scale, shift))?)
    }
}

/// Gated residual `x + out·gate` — one fused kernel (multiply + add) when the sc-2963 glue toggle is
/// on; bit-identical to the eager `add(x, out·gate)`, dtype-preserving.
fn gated(x: &Array, out: &Array, gate: &Array) -> Result<Array> {
    let f = |(x, o, g): (&Array, &Array, &Array)| -> std::result::Result<Array, Exception> {
        add(x, &multiply(o, g)?)
    };
    if crate::compile_glue() {
        Ok(compile(f, true)((x, out, gate))?)
    } else {
        Ok(f((x, out, gate))?)
    }
}

/// The tanh-GELU FFN activation. Body mirrors [`mlx_gen::nn::gelu_tanh`] exactly (dtype-preserving,
/// f64-host `√(2/π)`); when the sc-2963 glue toggle is on, MLX fuses its ~8 elementwise ops into one
/// kernel — by far the biggest per-step glue cost at video sequence (the FFN expansion runs on
/// `[B, S, ffn]` tensors of tens-to-hundreds of millions of elements). Off ⇒ defers to core
/// `gelu_tanh`, so the eager path is byte-for-byte the previous behaviour.
fn gelu_ffn(x: &Array) -> Result<Array> {
    if !crate::compile_glue() {
        return gelu_tanh(x);
    }
    let f = |x_: &Array| -> std::result::Result<Array, Exception> {
        let dt = x_.dtype();
        let s = |v: f32| -> std::result::Result<Array, Exception> { scalar(v).as_dtype(dt) };
        let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
        let x3 = power(x_, Array::from_int(3))?;
        let inner = multiply(&add(x_, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
        let gate = add(&tanh(&inner)?, &s(1.0)?)?;
        multiply(&multiply(x_, &s(0.5)?)?, &gate)
    };
    Ok(compile(f, true)(x)?)
}

/// A Linear's base weight — dense or Q8-quantized, selected by [`Precision`] at load.
enum LinearKind {
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

impl LinearKind {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            LinearKind::Dense { w, b } => linear(x, w, b),
            LinearKind::Quant {
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

    /// The base weight's logical `[out, in]` — what a LoKr delta reshapes to. The quantized packed
    /// weight is opaque, so recover `in` from the scales grid (`[out, in/group]`) × the group size
    /// (mirrors the core `AdaptableLinear::base_shape`).
    fn base_shape(&self) -> Vec<i32> {
        match self {
            LinearKind::Dense { w, .. } => w.shape().to_vec(),
            LinearKind::Quant { scales, group, .. } => {
                let s = scales.shape();
                vec![s[0], s[1] * group]
            }
        }
    }
}

/// One adapter as a forward-time residual — the reference `lora/apply.py::LoRALinear`
/// (`out + scale·strength·(x·Aᵀ·Bᵀ)`), NOT a merged weight. Residual (vs the reference's
/// alternative `apply_loras_to_model` merge→bf16) keeps the shipped Q8 base intact — a full
/// attn+ff adapter over this 22B Q8 transformer would dequantize ~15 GB to bf16 if merged, and
/// per-pass strength would double it — and leaves the bit-exact base forward (sc-2842) untouched.
///
/// **LoRA** keeps its factors at their loaded (bf16) dtype so a bf16 `x` (Bf16Q8) runs bf16 and an
/// f32 `x` (F32Q8) promotes to f32, exactly as the reference does (the sc-2772 toolchain fix means
/// the low-rank bf16 GEMM is correct, so no f32 forcing is needed). **LoKr** (sc-2393 — net-new, the
/// reference `lora/` has no LoKr) carries a precomputed `[out,in]` delta (`alpha/rank` already folded
/// in by `reconstruct_lokr_delta`); the residual is `x·ΔWᵀ` with `ΔW` cast to the activation dtype,
/// mirroring the fork's `LoKrLinear` / the core `Adapter::Lokr`. Both variants apply a per-pass scale.
enum LtxAdapter {
    /// `residual = pass_scale[pass] · (x·Aᵀ)·Bᵀ`. `pass_scale` already folds `(alpha/rank)·strength`.
    Lora {
        a: Array, // Aᵀ : [in, rank] (residual form; `lora_A` transposed)
        b: Array, // Bᵀ : [rank, out] (`lora_B` transposed)
        pass_scale: Vec<f32>,
    },
    /// `residual = pass_scale[pass] · x·ΔWᵀ`. `ΔW` ([out,in], bf16) has `alpha/rank` baked in, so
    /// `pass_scale` is the user `strength[pass]` alone (no further alpha/rank fold).
    Lokr {
        delta: Array, // ΔW : [out, in] (bf16; cast to the activation dtype at forward)
        pass_scale: Vec<f32>,
    },
}

impl LtxAdapter {
    /// Per-pass effective strengths (one entry per distilled stage, or a length-1 uniform vec).
    /// `Linear::forward` clamps the active pass index into this.
    fn pass_scale(&self) -> &[f32] {
        match self {
            LtxAdapter::Lora { pass_scale, .. } | LtxAdapter::Lokr { pass_scale, .. } => pass_scale,
        }
    }

    /// The unscaled residual `x·Aᵀ·Bᵀ` (LoRA) or `x·ΔWᵀ` (LoKr), before the per-pass scale.
    fn residual(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            LtxAdapter::Lora { a, b, .. } => matmul(&matmul(x, a)?, b)?,
            LtxAdapter::Lokr { delta, .. } => matmul(x, delta.as_dtype(x.dtype())?.t())?,
        })
    }
}

/// The adapter overlay on a [`Linear`]: the stacked residuals plus the active denoise-pass index,
/// set by [`LtxDiT::set_lora_pass`] before each stage. The pass lives in a `Cell` so a shared
/// `&self` forward reads it and the pipeline switches passes without `&mut` (the crate runs
/// single-device, one job per thread — see the runtime docs; `Cell<usize>` is `Send`).
struct LoraStack {
    adapters: Vec<LtxAdapter>,
    pass: Cell<usize>,
}

/// A Linear: a base weight ([`LinearKind`]) plus an optional forward-time LoRA overlay. With no
/// adapters the forward is byte-identical to the pre-sc-2687 path (the Q8/dense base only).
pub struct Linear {
    kind: LinearKind,
    lora: Option<LoraStack>,
}

impl Linear {
    fn load(w: &Weights, prefix: &str, prec: Precision) -> Result<Self> {
        let dt = prec.dtype();
        let b = to_dtype(w.require(&format!("{prefix}.bias"))?, dt)?;
        let kind = match w.get(&format!("{prefix}.scales")) {
            Some(scales) => {
                let q = w.require(&format!("{prefix}.weight"))?;
                let biases = w.require(&format!("{prefix}.biases"))?;
                if prec.keep_quant() {
                    // Keep the weights packed; `quantized_matmul` dequantizes on the fly with fp32
                    // accumulation and is correct for f32 *or* bf16 activations at either bit-width
                    // (the Z-Image/Qwen Q8 path). Scales / biases are cast to the compute dtype so the
                    // on-the-fly dequant matches the reference's (f32 for quant_f32 — a lossless upcast
                    // of the bf16 file scales). bits/group come from the checkpoint's split_model.json
                    // via `prec` (sc-2686), so Q4 and Q8 both load unchanged.
                    LinearKind::Quant {
                        q: q.clone(),
                        scales: to_dtype(scales, dt)?,
                        biases: to_dtype(biases, dt)?,
                        b,
                        group: prec.group,
                        bits: prec.bits,
                    }
                } else {
                    // Dequantize to dense f32 (bit-identical to the reference's mx.dequantize).
                    let dense =
                        dequantize(q, scales, Some(biases), Some(prec.group), Some(prec.bits))?;
                    LinearKind::Dense {
                        w: to_dtype(&dense, Dtype::Float32)?,
                        b,
                    }
                }
            }
            None => LinearKind::Dense {
                w: to_dtype(w.require(&format!("{prefix}.weight"))?, dt)?,
                b,
            },
        };
        Ok(Linear { kind, lora: None })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut out = self.kind.forward(x)?;
        if let Some(stack) = &self.lora {
            let pass = stack.pass.get();
            for ad in &stack.adapters {
                // residual = (LoRA: (x·Aᵀ)·Bᵀ | LoKr: x·ΔWᵀ) · scale[pass], factors/delta at the
                // activation dtype (MLX promotes a bf16 factor against an f32 activation), the scale
                // through a dtype-matched scalar so the multiply preserves the residual dtype (the
                // reference's weak Python-float `scale·…`).
                let r = ad.residual(x)?;
                let ps = ad.pass_scale();
                let s = ps[pass.min(ps.len() - 1)];
                out = add(&out, &multiply(&r, &scalar(s).as_dtype(r.dtype())?)?)?;
            }
        }
        Ok(out)
    }

    /// Stack a LoRA residual (the loader installs one per resolved target). `a`/`b` are the raw
    /// `lora_A [rank,in]` / `lora_B [out,rank]` transposed to residual form by the caller.
    pub(crate) fn push_lora(&mut self, a: Array, b: Array, pass_scale: Vec<f32>) {
        self.lora_stack()
            .adapters
            .push(LtxAdapter::Lora { a, b, pass_scale });
    }

    /// Stack a LoKr residual (sc-2393). `delta` is the precomputed `[out,in]` bf16 weight delta
    /// (`alpha/rank` already folded in by `reconstruct_lokr_delta`); `pass_scale` is the per-pass
    /// user strength alone. Net-new — the reference `lora/` is LoRA-only.
    pub(crate) fn push_lokr(&mut self, delta: Array, pass_scale: Vec<f32>) {
        self.lora_stack()
            .adapters
            .push(LtxAdapter::Lokr { delta, pass_scale });
    }

    /// **Training seam** (sc-3047): replace the adapter stack with a single trainable LoRA residual.
    /// `a`/`b` are the residual-form factors (`lora_A.t()` `[in,rank]` and the `alpha/rank`-scaled
    /// `lora_B.t()` `[rank,out]`), with a single unit pass-scale, so the forward adds
    /// `(x·a)·b` = `(x·Aᵀ·Bᵀ)·(alpha/rank)` — the same residual the reference `_MlxLoRALinear` applies.
    /// The trainer re-injects the freshly-traced factors each optimizer step (functional autograd),
    /// the LTX analog of the core `AdaptableLinear::set_adapters`. Unlike [`push_lora`](Self::push_lora)
    /// this *replaces* the stack (one residual at a time), not append.
    pub(crate) fn set_train_lora(&mut self, a: Array, b: Array) {
        self.lora = Some(LoraStack {
            adapters: vec![LtxAdapter::Lora {
                a,
                b,
                pass_scale: vec![1.0],
            }],
            pass: Cell::new(0),
        });
    }

    /// The adapter stack, created empty (pass 0) on first install.
    fn lora_stack(&mut self) -> &mut LoraStack {
        self.lora.get_or_insert_with(|| LoraStack {
            adapters: Vec::new(),
            pass: Cell::new(0),
        })
    }

    /// Select the active denoise pass for this linear's LoRA residuals (no-op without adapters).
    fn set_lora_pass(&self, pass: usize) {
        if let Some(stack) = &self.lora {
            stack.pass.set(pass);
        }
    }

    /// The base weight's logical `[out, in]` — the shape a LoKr delta reshapes to (sc-2393).
    pub(crate) fn base_shape(&self) -> Vec<i32> {
        self.kind.base_shape()
    }
}

/// A model tree that routes a LoRA target's dotted path (post LTX key normalization) to its
/// [`Linear`] and selects the active denoise pass for all installed residuals. Implemented by the
/// video-only [`LtxDiT`] building block and the production dual-modality [`AvDiT`], so the adapter
/// loader ([`crate::adapters::apply_ltx_adapters`]) is generic over both (sc-2687).
pub trait LtxAdaptable {
    /// Resolve a target path to its [`Linear`], or `None` if it names no adaptable module (reported
    /// skipped by the loader, never silently dropped).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear>;
    /// Select the active distilled denoise pass for every installed residual (no-op without adapters).
    fn set_lora_pass(&self, pass: usize);
}

impl LtxAdaptable for LtxDiT {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        LtxDiT::adaptable_mut(self, path)
    }
    fn set_lora_pass(&self, pass: usize) {
        LtxDiT::set_lora_pass(self, pass)
    }
}

impl LtxAdaptable for AvDiT {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        AvDiT::adaptable_mut(self, path)
    }
    fn set_lora_pass(&self, pass: usize) {
        AvDiT::set_lora_pass(self, pass)
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
    /// Load an attention with explicit `heads`/`dim_head` (the *inner* dims = `heads·dim_head`,
    /// which the q/k/v project to; cross-modal attns project a different query/context dim into the
    /// same inner). `eps` is the q/k-RMSNorm epsilon.
    fn load(
        w: &Weights,
        prefix: &str,
        heads: i32,
        dim_head: i32,
        eps: f32,
        prec: Precision,
    ) -> Result<Self> {
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
            heads,
            dim_head,
            eps,
        })
    }

    /// `(B, S, inner)` → `(B, H, S, head_dim)`.
    fn to_heads(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        Ok(x.reshape(&[b, s, self.heads, self.dim_head])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }

    /// `pe` rotates the query (and the key if `k_pe` is `None`); `k_pe` rotates the key separately
    /// (cross-modal: video-positioned q, audio-positioned k, or vice-versa). `pe == None` ⇒ no RoPE
    /// on either (text cross-attention). Mirrors `attention.py::Attention.__call__`.
    fn forward(
        &self,
        x: &Array,
        context: Option<&Array>,
        mask: Option<&Array>,
        pe: Option<(&Array, &Array)>,
        k_pe: Option<(&Array, &Array)>,
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
            let (kc, ks) = k_pe.unwrap_or((cos, sin));
            kh = apply_split_rotary_emb(&kh, kc, ks)?;
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

    /// LoRA key→module map (diffusers/peft naming, post LTX normalization). `to_out.0`→`to_out`
    /// is done by the loader; `to_gate_logits` resolves only when the gated branch is present.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out"] => Some(&mut self.to_out),
            ["to_gate_logits"] => self.gate.as_mut(),
            _ => None,
        }
    }

    fn set_lora_pass(&self, pass: usize) {
        self.to_q.set_lora_pass(pass);
        self.to_k.set_lora_pass(pass);
        self.to_v.set_lora_pass(pass);
        self.to_out.set_lora_pass(pass);
        if let Some(g) = &self.gate {
            g.set_lora_pass(pass);
        }
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
        self.proj_out.forward(&gelu_ffn(&self.proj_in.forward(x)?)?)
    }

    /// LoRA targets: `ff.net.0.proj`→`ff.proj_in`, `ff.net.2`→`ff.proj_out` (renamed by the loader).
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["proj_in"] => Some(&mut self.proj_in),
            ["proj_out"] => Some(&mut self.proj_out),
            _ => None,
        }
    }

    fn set_lora_pass(&self, pass: usize) {
        self.proj_in.set_lora_pass(pass);
        self.proj_out.set_lora_pass(pass);
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
        let (h, dh, eps) = (
            cfg.num_attention_heads,
            cfg.attention_head_dim,
            cfg.norm_eps as f32,
        );
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), h, dh, eps, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), h, dh, eps, prec)?,
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
        let attn = self
            .attn1
            .forward(&norm, None, None, Some((cos, sin)), None)?;
        let mut x = gated(x, &attn, &msa[2])?;

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
        let cross = self
            .attn2
            .forward(&norm_ca, Some(&v_context), mask, None, None)?;
        x = gated(&x, &cross, &ca[2])?;

        // --- FeedForward (adaLN rows 3..6) ---
        let mlp = ada_values(&self.scale_shift_table, timesteps, 3, 6)?;
        let norm_mlp = modulate(&rms_norm_noweight(&x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff = self.ff.forward(&norm_mlp)?;
        x = gated(&x, &ff, &mlp[2])?;

        Ok(x)
    }

    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            _ => None,
        }
    }

    pub(crate) fn set_lora_pass(&self, pass: usize) {
        self.attn1.set_lora_pass(pass);
        self.attn2.set_lora_pass(pass);
        self.ff.set_lora_pass(pass);
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

    /// LoRA targets in diffusers naming: `emb.timestep_embedder.linear1/linear2` and `linear`. (A
    /// trained file spelling the embedder `linear_1`/`linear_2` — the PixArt convention — does NOT
    /// resolve here and is reported skipped, matching the reference `_normalize_ltx_lora_key`, which
    /// has no such rename; the base checkpoint names them `linear1`/`linear2`.)
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["emb", "timestep_embedder", "linear1"] => Some(&mut self.ts_lin1),
            ["emb", "timestep_embedder", "linear2"] => Some(&mut self.ts_lin2),
            ["linear"] => Some(&mut self.linear),
            _ => None,
        }
    }

    fn set_lora_pass(&self, pass: usize) {
        self.ts_lin1.set_lora_pass(pass);
        self.ts_lin2.set_lora_pass(pass);
        self.linear.set_lora_pass(pass);
    }
}

/// The LTX-2.3 video DiT: preprocessor + 48 blocks + output projection. Predicts velocity.
/// Memoizes a SPLIT-RoPE `(cos, sin)` table pair keyed on the `positions` content (F-048). The tables
/// depend only on `positions` (+ the model's fixed dims), which are **constant across every denoise
/// step of a stage**, so the expensive single-threaded f64 host trig in [`precompute_split_freqs_cis`]
/// runs once per stage instead of once per step. Bit-identical: the recomputed tables are byte-equal,
/// so a cache hit returns exactly what a recompute would. Inference is one job per thread (like the
/// LoRA-pass [`Cell`]), so a [`RefCell`] is sufficient.
#[derive(Default)]
struct RopeMemo {
    cached: RefCell<Option<(Vec<f32>, Array, Array)>>,
}

impl RopeMemo {
    /// Return the cached `(cos, sin)` if `positions` is unchanged since the last call, else run
    /// `compute`, cache, and return it. The key is the positions content (small relative to the trig).
    fn get_or_compute(
        &self,
        positions: &Array,
        compute: impl FnOnce() -> Result<(Array, Array)>,
    ) -> Result<(Array, Array)> {
        let key: Vec<f32> = positions.as_slice::<f32>().to_vec();
        if let Some((k, cos, sin)) = self.cached.borrow().as_ref() {
            if *k == key {
                return Ok((cos.clone(), sin.clone()));
            }
        }
        let (cos, sin) = compute()?;
        *self.cached.borrow_mut() = Some((key, cos.clone(), sin.clone()));
        Ok((cos, sin))
    }
}

pub struct LtxDiT {
    patchify_proj: Linear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: Option<AdaLayerNormSingle>,
    blocks: Vec<VideoBlock>,
    scale_shift_table: Array, // (2, inner)
    proj_out: Linear,
    cfg: LtxConfig,
    prec: Precision,
    /// Per-stage SPLIT-RoPE table cache (F-048): the tables are constant across denoise steps.
    rope_memo: RopeMemo,
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
            rope_memo: RopeMemo::default(),
        })
    }

    /// Resolve a LoRA target (diffusers/peft dotted path, post LTX key normalization) to its
    /// [`Linear`], so the loader ([`crate::adapters`]) can install a residual onto it. The video-only
    /// surface: the 48 blocks' `attn{1,2}` + `ff` leaves, the two adaLN-single modules, and the
    /// global `patchify_proj`/`proj_out`. Audio / `av_ca` / `a2v` targets and the PixArt-spelled adaLN
    /// embedder (`linear_1/2`) resolve to `None` here → reported skipped, never silently dropped.
    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["patchify_proj"] => Some(&mut self.patchify_proj),
            ["proj_out"] => Some(&mut self.proj_out),
            ["adaln_single", rest @ ..] => self.adaln.adaptable_mut(rest),
            ["prompt_adaln_single", rest @ ..] => self.prompt_adaln.as_mut()?.adaptable_mut(rest),
            ["transformer_blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            _ => None,
        }
    }

    /// Select the active distilled denoise pass for every installed LoRA residual (sc-2687 per-pass
    /// strength). The pipeline calls this before each stage; a no-adapter model is unaffected.
    pub fn set_lora_pass(&self, pass: usize) {
        self.patchify_proj.set_lora_pass(pass);
        self.proj_out.set_lora_pass(pass);
        self.adaln.set_lora_pass(pass);
        if let Some(p) = &self.prompt_adaln {
            p.set_lora_pass(pass);
        }
        for b in &self.blocks {
            b.set_lora_pass(pass);
        }
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

        // SPLIT RoPE from the position grid (f32 tables; the block casts per input dtype). Cached on
        // `positions` — constant across the stage's denoise steps, so computed once not per step (F-048).
        let (cos, sin) = self.rope_memo.get_or_compute(positions, || {
            precompute_split_freqs_cis(
                positions,
                inner,
                self.cfg.positional_embedding_theta,
                &self.cfg.positional_embedding_max_pos,
                self.cfg.num_attention_heads,
            )
        })?;

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

// ===================================================================================================
// AudioVideo DiT (sc-2684) — the dual-modality `BasicAVTransformerBlock` / `LTXModel`.
// ===================================================================================================

/// `positions[:, 0:1, :, :]` — the time axis as a `(B, 1, T, 2)` grid (the cross-modal RoPE input;
/// `MultiModalTransformerArgsPreprocessor.prepare`). For the audio grid (already `(B, 1, T, 2)`) this
/// is a no-op slice.
fn time_axis(positions: &Array) -> Result<Array> {
    let sh = positions.shape();
    let (b, t) = (sh[0], sh[2]);
    Ok(positions
        .take_axis(Array::from_int(0), 1)? // (B, T, 2)
        .reshape(&[b, 1, t, 2])?)
}

/// One modality's non-block modules + dims — the video or audio half of the AV DiT. Carries the
/// patchify projection, adaLN-single (timestep → coeff·dim) + prompt-adaLN, the two cross-modal
/// adaLN-single modules (4-coeff scale-shift + 1-coeff gate), the output scale-shift table, and the
/// output projection, plus the dims that drive RoPE.
struct Stream {
    patchify: Linear,
    adaln: AdaLayerNormSingle,
    prompt_adaln: AdaLayerNormSingle,
    cross_ss_adaln: AdaLayerNormSingle,
    cross_gate_adaln: AdaLayerNormSingle,
    scale_shift_table: Array, // (2, inner) output head
    proj_out: Linear,
    inner: i32,
    heads: i32,
    coeff: i32, // adaLN row count (9 gated)
    self_max_pos: Vec<i32>,
    cross_inner: i32, // audio_cross_attention_dim (2048) — the cross-modal RoPE inner
    cross_max_pos: i32,
    theta: f64,
    ts_mult: i32,
    av_ca_ts_mult: i32,
    eps: f32,
    prec: Precision,
    /// Per-stage caches for this stream's self-attention + cross-modal SPLIT-RoPE tables (F-048).
    self_rope_memo: RopeMemo,
    cross_rope_memo: RopeMemo,
}

/// The per-modality preprocessed args threaded into the block stack + output head.
struct StreamPrep {
    x: Array,
    ts_emb: Array,
    emb_ts: Array,
    prompt_ts: Array,
    context: Array,
    cos: Array,
    sin: Array,
    cross_cos: Array,
    cross_sin: Array,
    cross_ss_ts: Array,
    cross_gate_ts: Array,
}

/// Borrowed view of a [`StreamPrep`] passed to [`AvBlock::forward`].
struct StreamArgs<'a> {
    ts_emb: &'a Array,
    prompt_ts: &'a Array,
    context: &'a Array,
    mask: Option<&'a Array>,
    cos: &'a Array,
    sin: &'a Array,
    cross_cos: &'a Array,
    cross_sin: &'a Array,
    cross_ss_ts: &'a Array,
    cross_gate_ts: &'a Array,
}

impl StreamPrep {
    fn args<'a>(&'a self, mask: Option<&'a Array>) -> StreamArgs<'a> {
        StreamArgs {
            ts_emb: &self.ts_emb,
            prompt_ts: &self.prompt_ts,
            context: &self.context,
            mask,
            cos: &self.cos,
            sin: &self.sin,
            cross_cos: &self.cross_cos,
            cross_sin: &self.cross_sin,
            cross_ss_ts: &self.cross_ss_ts,
            cross_gate_ts: &self.cross_gate_ts,
        }
    }
}

impl Stream {
    /// `latent` `(B, S, in)`, per-token `timestep` `(B, S)`, text `context`, `positions` grid.
    /// Reproduces `TransformerArgsPreprocessor.prepare` + the multimodal cross-PE / cross-timesteps.
    fn prepare(
        &self,
        latent: &Array,
        timestep: &Array,
        context: &Array,
        positions: &Array,
    ) -> Result<StreamPrep> {
        let dt = self.prec.dtype();
        let b = latent.shape()[0];
        let (inner, coeff) = (self.inner, self.coeff);

        let x = self.patchify.forward(&latent.as_dtype(dt)?)?;

        // adaLN-single timestep projection (the `× ts_mult` runs in the input dtype; see the
        // video-only path's note — bf16 must round `bf16(σ·1000)` first).
        let mult = scalar(self.ts_mult as f32).as_dtype(timestep.dtype())?;
        let ts_flat = multiply(timestep, &mult)?.reshape(&[-1])?;
        let (ts_emb, emb_ts) = self.adaln.forward(&ts_flat, dt)?;
        let ts_emb = ts_emb.reshape(&[b, -1, coeff * inner])?;
        let emb_ts = emb_ts.reshape(&[b, -1, inner])?;

        // prompt-adaLN: one shared modulation per sample (timestep[:, :1]).
        let src = if timestep.ndim() > 1 {
            timestep.index_axis(0, 1)?.reshape(&[b, 1])?
        } else {
            timestep.clone()
        };
        let src = multiply(&src, &mult)?.reshape(&[-1])?;
        let (pts, _) = self.prompt_adaln.forward(&src, dt)?;
        let prompt_ts = pts.reshape(&[b, -1, 2 * inner])?;

        // Cross-modal scale-shift (4·dim) + gate (1·dim) timesteps. The gate timestep carries the
        // extra `av_ca_factor = av_ca_ts_mult / ts_mult` (1.0 for 2.3, an exact f32 no-op).
        let (cross_ss, _) = self.cross_ss_adaln.forward(&ts_flat, dt)?;
        let cross_ss_ts = cross_ss.reshape(&[b, -1, 4 * inner])?;
        let factor =
            scalar(self.av_ca_ts_mult as f32 / self.ts_mult as f32).as_dtype(ts_flat.dtype())?;
        let gate_in = multiply(&ts_flat, &factor)?;
        let (cross_gate, _) = self.cross_gate_adaln.forward(&gate_in, dt)?;
        let cross_gate_ts = cross_gate.reshape(&[b, -1, inner])?;

        // caption_projection = Identity (2.3): context enters cross-attn as-is.
        let context = context.as_dtype(dt)?;

        // Self-attention SPLIT RoPE (modality inner dim, modality max_pos). Both tables are constant
        // across the stage's denoise steps, so cache them on `positions` (computed once, not per
        // step — F-048; the cross table's `time_axis(positions)` is derived from the same key).
        let (cos, sin) = self.self_rope_memo.get_or_compute(positions, || {
            precompute_split_freqs_cis(positions, inner, self.theta, &self.self_max_pos, self.heads)
        })?;
        // Cross-modal SPLIT RoPE: the time axis only, at the cross inner dim (2048) / cross max_pos.
        let (cross_cos, cross_sin) = self.cross_rope_memo.get_or_compute(positions, || {
            precompute_split_freqs_cis(
                &time_axis(positions)?,
                self.cross_inner,
                self.theta,
                &[self.cross_max_pos],
                self.heads,
            )
        })?;

        Ok(StreamPrep {
            x,
            ts_emb,
            emb_ts,
            prompt_ts,
            context,
            cos,
            sin,
            cross_cos,
            cross_sin,
            cross_ss_ts,
            cross_gate_ts,
        })
    }

    /// Output head (LayerNorm-affine-false → final 2-row scale-shift → proj_out). Mirrors
    /// `LTXModel._process_output`.
    fn output_head(&self, h: &Array, emb_ts: &Array) -> Result<Array> {
        let b = h.shape()[0];
        let table = self.scale_shift_table.reshape(&[1, 1, 2, self.inner])?;
        let ss = add(&table, &emb_ts.reshape(&[b, -1, 1, self.inner])?)?;
        let shift = ss.index_axis(0, 2)?;
        let scale = ss.index_axis(1, 2)?;
        let normed = layer_norm(h, None, None, self.eps)?;
        self.proj_out.forward(&modulate(&normed, &scale, &shift)?)
    }

    /// Select the active denoise pass for this stream's LoRA-targetable Linears (the patchify/out
    /// projections + the four adaLN-single modules). The `scale_shift_table`/gate params carry no
    /// Linear, so they are unaffected.
    fn set_lora_pass(&self, pass: usize) {
        self.patchify.set_lora_pass(pass);
        self.proj_out.set_lora_pass(pass);
        self.adaln.set_lora_pass(pass);
        self.prompt_adaln.set_lora_pass(pass);
        self.cross_ss_adaln.set_lora_pass(pass);
        self.cross_gate_adaln.set_lora_pass(pass);
    }
}

/// `4·scale-shift + 1·gate` cross-modal adaLN values from the pre-split tables. Returns
/// `(scale_a2v, shift_a2v, scale_v2a, shift_v2a, gate)` — the row layout of
/// `scale_shift_table_a2v_ca_{audio,video}` (`get_av_ca_ada_values`).
fn av_ca_ada(
    ss_table: &Array,
    gate_table: &Array,
    ss_ts: &Array,
    gate_ts: &Array,
) -> Result<(Array, Array, Array, Array, Array)> {
    let ss = ada_values(ss_table, ss_ts, 0, 4)?;
    let g = ada_values(gate_table, gate_ts, 0, 1)?;
    Ok((
        ss[0].clone(),
        ss[1].clone(),
        ss[2].clone(),
        ss[3].clone(),
        g[0].clone(),
    ))
}

/// One AudioVideo transformer block: the video stack + the audio stack + bidirectional cross-modal
/// attention (`BasicAVTransformerBlock`). Per-block order: video self+text-CA → audio self+text-CA →
/// cross-modal (a2v updates video, v2a updates audio) → video FF → audio FF.
struct AvBlock {
    // Video.
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    v_sst: Array, // (9, 4096)
    v_pst: Array, // (2, 4096)
    // Audio.
    a_attn1: Attention,
    a_attn2: Attention,
    a_ff: FeedForward,
    a_sst: Array, // (9, 2048)
    a_pst: Array, // (2, 2048)
    // Cross-modal.
    a2v: Attention,       // audio_to_video_attn (Q video, K/V audio)
    v2a: Attention,       // video_to_audio_attn (Q audio, K/V video)
    ca_audio_ss: Array,   // (4, 2048)
    ca_audio_gate: Array, // (1, 2048)
    ca_video_ss: Array,   // (4, 4096)
    ca_video_gate: Array, // (1, 4096)
    eps: f32,
}

impl AvBlock {
    fn load(w: &Weights, prefix: &str, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let eps = cfg.norm_eps as f32;
        let (vh, vdh) = (cfg.num_attention_heads, cfg.attention_head_dim);
        let (ah, adh) = (cfg.audio_num_attention_heads, cfg.audio_attention_head_dim);
        // Split a (5, dim) cross table into the 4-row scale-shift block + the 1-row gate.
        let split = |key: &str| -> Result<(Array, Array)> {
            let t = param(w, &format!("{prefix}.{key}"), prec)?;
            let ss = t.take_axis(Array::from_slice(&[0, 1, 2, 3], &[4]), 0)?;
            let gate = t.take_axis(Array::from_slice(&[4], &[1]), 0)?;
            Ok((ss, gate))
        };
        let (ca_audio_ss, ca_audio_gate) = split("scale_shift_table_a2v_ca_audio")?;
        let (ca_video_ss, ca_video_gate) = split("scale_shift_table_a2v_ca_video")?;
        Ok(Self {
            attn1: Attention::load(w, &format!("{prefix}.attn1"), vh, vdh, eps, prec)?,
            attn2: Attention::load(w, &format!("{prefix}.attn2"), vh, vdh, eps, prec)?,
            ff: FeedForward::load(w, &format!("{prefix}.ff"), prec)?,
            v_sst: param(w, &format!("{prefix}.scale_shift_table"), prec)?,
            v_pst: param(w, &format!("{prefix}.prompt_scale_shift_table"), prec)?,
            a_attn1: Attention::load(w, &format!("{prefix}.audio_attn1"), ah, adh, eps, prec)?,
            a_attn2: Attention::load(w, &format!("{prefix}.audio_attn2"), ah, adh, eps, prec)?,
            a_ff: FeedForward::load(w, &format!("{prefix}.audio_ff"), prec)?,
            a_sst: param(w, &format!("{prefix}.audio_scale_shift_table"), prec)?,
            a_pst: param(w, &format!("{prefix}.audio_prompt_scale_shift_table"), prec)?,
            // Cross-modal attns run at the audio inner dim (heads 32 × head_dim 64 = 2048).
            a2v: Attention::load(
                w,
                &format!("{prefix}.audio_to_video_attn"),
                ah,
                adh,
                eps,
                prec,
            )?,
            v2a: Attention::load(
                w,
                &format!("{prefix}.video_to_audio_attn"),
                ah,
                adh,
                eps,
                prec,
            )?,
            ca_audio_ss,
            ca_audio_gate,
            ca_video_ss,
            ca_video_gate,
            eps,
        })
    }

    /// Self+text-CA for one modality (`run_vx`/`run_ax` body, sans FF): MSA (RoPE) → prompt-modulated
    /// text cross-attention. Returns the updated stream hidden.
    #[allow(clippy::too_many_arguments)]
    fn self_and_text(
        &self,
        x: &Array,
        attn1: &Attention,
        attn2: &Attention,
        sst: &Array,
        pst: &Array,
        a: &StreamArgs,
    ) -> Result<Array> {
        let msa = ada_values(sst, a.ts_emb, 0, 3)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &msa[1], &msa[0])?;
        let attn = attn1.forward(&norm, None, None, Some((a.cos, a.sin)), None)?;
        let mut x = gated(x, &attn, &msa[2])?;

        let p = ada_values(pst, a.prompt_ts, 0, 2)?;
        let context = modulate(a.context, &p[1], &p[0])?;

        let ca = ada_values(sst, a.ts_emb, 6, 9)?;
        let norm_ca = modulate(&rms_norm_noweight(&x, self.eps)?, &ca[1], &ca[0])?;
        let cross = attn2.forward(&norm_ca, Some(&context), a.mask, None, None)?;
        x = gated(&x, &cross, &ca[2])?;
        Ok(x)
    }

    /// FeedForward (adaLN rows 3..6) for one modality.
    fn feed_forward(
        &self,
        x: &Array,
        ff: &FeedForward,
        sst: &Array,
        ts_emb: &Array,
    ) -> Result<Array> {
        let mlp = ada_values(sst, ts_emb, 3, 6)?;
        let norm = modulate(&rms_norm_noweight(x, self.eps)?, &mlp[1], &mlp[0])?;
        let ff_out = ff.forward(&norm)?;
        gated(x, &ff_out, &mlp[2])
    }

    /// Joint forward: `(vx, ax)` in, `(vx, ax)` out.
    fn forward(
        &self,
        vx: &Array,
        ax: &Array,
        v: &StreamArgs,
        a: &StreamArgs,
    ) -> Result<(Array, Array)> {
        // Video / audio self-attention + text cross-attention.
        let mut vx =
            self.self_and_text(vx, &self.attn1, &self.attn2, &self.v_sst, &self.v_pst, v)?;
        let mut ax = self.self_and_text(
            ax,
            &self.a_attn1,
            &self.a_attn2,
            &self.a_sst,
            &self.a_pst,
            a,
        )?;

        // Cross-modal attention — both directions read the pre-update rms_norm snapshots.
        let vx_n3 = rms_norm_noweight(&vx, self.eps)?;
        let ax_n3 = rms_norm_noweight(&ax, self.eps)?;
        let (sca_a2v, sha_a2v, sca_v2a, sha_v2a, gate_v2a) = av_ca_ada(
            &self.ca_audio_ss,
            &self.ca_audio_gate,
            a.cross_ss_ts,
            a.cross_gate_ts,
        )?;
        let (scv_a2v, shv_a2v, scv_v2a, shv_v2a, gate_a2v) = av_ca_ada(
            &self.ca_video_ss,
            &self.ca_video_gate,
            v.cross_ss_ts,
            v.cross_gate_ts,
        )?;

        // Audio-to-Video: Q from video (video cross-PE), K/V from audio (audio cross-PE).
        let a2v = self.a2v.forward(
            &modulate(&vx_n3, &scv_a2v, &shv_a2v)?,
            Some(&modulate(&ax_n3, &sca_a2v, &sha_a2v)?),
            None,
            Some((v.cross_cos, v.cross_sin)),
            Some((a.cross_cos, a.cross_sin)),
        )?;
        vx = gated(&vx, &a2v, &gate_a2v)?;

        // Video-to-Audio: Q from audio (audio cross-PE), K/V from video (video cross-PE).
        let v2a = self.v2a.forward(
            &modulate(&ax_n3, &sca_v2a, &sha_v2a)?,
            Some(&modulate(&vx_n3, &scv_v2a, &shv_v2a)?),
            None,
            Some((a.cross_cos, a.cross_sin)),
            Some((v.cross_cos, v.cross_sin)),
        )?;
        ax = gated(&ax, &v2a, &gate_v2a)?;

        // FeedForward.
        vx = self.feed_forward(&vx, &self.ff, &self.v_sst, v.ts_emb)?;
        ax = self.feed_forward(&ax, &self.a_ff, &self.a_sst, a.ts_emb)?;
        Ok((vx, ax))
    }

    /// LoRA key→module map for one AV block: the video self/text attns + ff, the audio analogues, and
    /// the two cross-modal attns (`audio_to_video_attn`/`video_to_audio_attn`). `audio_ff.net.*` →
    /// `audio_ff.proj_*` is renamed by the loader before reaching here.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["attn1", rest @ ..] => self.attn1.adaptable_mut(rest),
            ["attn2", rest @ ..] => self.attn2.adaptable_mut(rest),
            ["ff", rest @ ..] => self.ff.adaptable_mut(rest),
            ["audio_attn1", rest @ ..] => self.a_attn1.adaptable_mut(rest),
            ["audio_attn2", rest @ ..] => self.a_attn2.adaptable_mut(rest),
            ["audio_ff", rest @ ..] => self.a_ff.adaptable_mut(rest),
            ["audio_to_video_attn", rest @ ..] => self.a2v.adaptable_mut(rest),
            ["video_to_audio_attn", rest @ ..] => self.v2a.adaptable_mut(rest),
            _ => None,
        }
    }

    fn set_lora_pass(&self, pass: usize) {
        for attn in [
            &self.attn1,
            &self.attn2,
            &self.a_attn1,
            &self.a_attn2,
            &self.a2v,
            &self.v2a,
        ] {
            attn.set_lora_pass(pass);
        }
        self.ff.set_lora_pass(pass);
        self.a_ff.set_lora_pass(pass);
    }
}

/// The LTX-2.3 **AudioVideo** DiT (`LTXModel` with both stacks). Predicts `(video_velocity,
/// audio_velocity)` from the two latent token streams + shared text conditioning.
pub struct AvDiT {
    video: Stream,
    audio: Stream,
    blocks: Vec<AvBlock>,
}

impl AvDiT {
    pub fn from_weights(w: &Weights, cfg: &LtxConfig, prec: Precision) -> Result<Self> {
        let video = Stream {
            patchify: Linear::load(w, "patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "adaln_single", prec)?,
            prompt_adaln: AdaLayerNormSingle::load(w, "prompt_adaln_single", prec)?,
            cross_ss_adaln: AdaLayerNormSingle::load(
                w,
                "av_ca_video_scale_shift_adaln_single",
                prec,
            )?,
            cross_gate_adaln: AdaLayerNormSingle::load(w, "av_ca_a2v_gate_adaln_single", prec)?,
            scale_shift_table: param(w, "scale_shift_table", prec)?,
            proj_out: Linear::load(w, "proj_out", prec)?,
            inner: cfg.inner_dim(),
            heads: cfg.num_attention_heads,
            coeff: cfg.adaln_embedding_coefficient,
            self_max_pos: cfg.positional_embedding_max_pos.to_vec(),
            cross_inner: cfg.audio_cross_attention_dim,
            cross_max_pos: cfg.cross_pe_max_pos(),
            theta: cfg.positional_embedding_theta,
            ts_mult: cfg.timestep_scale_multiplier,
            av_ca_ts_mult: cfg.av_ca_timestep_scale_multiplier,
            eps: cfg.norm_eps as f32,
            prec,
            self_rope_memo: RopeMemo::default(),
            cross_rope_memo: RopeMemo::default(),
        };
        let audio = Stream {
            patchify: Linear::load(w, "audio_patchify_proj", prec)?,
            adaln: AdaLayerNormSingle::load(w, "audio_adaln_single", prec)?,
            prompt_adaln: AdaLayerNormSingle::load(w, "audio_prompt_adaln_single", prec)?,
            cross_ss_adaln: AdaLayerNormSingle::load(
                w,
                "av_ca_audio_scale_shift_adaln_single",
                prec,
            )?,
            cross_gate_adaln: AdaLayerNormSingle::load(w, "av_ca_v2a_gate_adaln_single", prec)?,
            scale_shift_table: param(w, "audio_scale_shift_table", prec)?,
            proj_out: Linear::load(w, "audio_proj_out", prec)?,
            inner: cfg.audio_inner_dim(),
            heads: cfg.audio_num_attention_heads,
            coeff: cfg.adaln_embedding_coefficient,
            self_max_pos: vec![cfg.audio_positional_embedding_max_pos],
            cross_inner: cfg.audio_cross_attention_dim,
            cross_max_pos: cfg.cross_pe_max_pos(),
            theta: cfg.positional_embedding_theta,
            ts_mult: cfg.timestep_scale_multiplier,
            av_ca_ts_mult: cfg.av_ca_timestep_scale_multiplier,
            eps: cfg.norm_eps as f32,
            prec,
            self_rope_memo: RopeMemo::default(),
            cross_rope_memo: RopeMemo::default(),
        };
        let blocks = (0..cfg.num_layers)
            .map(|i| AvBlock::load(w, &format!("transformer_blocks.{i}"), cfg, prec))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            video,
            audio,
            blocks,
        })
    }

    /// Resolve a LoRA target (diffusers/peft dotted path, post LTX key normalization) to its
    /// [`Linear`] across the **full** AudioVideo surface (sc-2687): the per-block video/audio self +
    /// text attns, the two cross-modal attns, both stacks' feed-forwards, and every stream-global
    /// projection / adaLN-single (video + audio + the four `av_ca_*` cross modules). The module name
    /// itself selects the stream (`audio_*` / `av_ca_audio_*` / `av_ca_v2a_*` → audio). A target that
    /// names no module (e.g. the PixArt-spelled adaLN embedder `linear_1/2`, or an out-of-range block)
    /// resolves to `None` → reported skipped, never silently dropped.
    pub(crate) fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut Linear> {
        match path {
            ["transformer_blocks", n, rest @ ..] => self
                .blocks
                .get_mut(n.parse::<usize>().ok()?)?
                .adaptable_mut(rest),
            // Video-stream globals.
            ["patchify_proj"] => Some(&mut self.video.patchify),
            ["proj_out"] => Some(&mut self.video.proj_out),
            ["adaln_single", rest @ ..] => self.video.adaln.adaptable_mut(rest),
            ["prompt_adaln_single", rest @ ..] => self.video.prompt_adaln.adaptable_mut(rest),
            ["av_ca_video_scale_shift_adaln_single", rest @ ..] => {
                self.video.cross_ss_adaln.adaptable_mut(rest)
            }
            ["av_ca_a2v_gate_adaln_single", rest @ ..] => {
                self.video.cross_gate_adaln.adaptable_mut(rest)
            }
            // Audio-stream globals.
            ["audio_patchify_proj"] => Some(&mut self.audio.patchify),
            ["audio_proj_out"] => Some(&mut self.audio.proj_out),
            ["audio_adaln_single", rest @ ..] => self.audio.adaln.adaptable_mut(rest),
            ["audio_prompt_adaln_single", rest @ ..] => self.audio.prompt_adaln.adaptable_mut(rest),
            ["av_ca_audio_scale_shift_adaln_single", rest @ ..] => {
                self.audio.cross_ss_adaln.adaptable_mut(rest)
            }
            ["av_ca_v2a_gate_adaln_single", rest @ ..] => {
                self.audio.cross_gate_adaln.adaptable_mut(rest)
            }
            _ => None,
        }
    }

    /// Select the active distilled denoise pass for every installed LoRA residual (sc-2687 per-pass
    /// strength) across both streams and all blocks. The pipeline calls this before each stage; a
    /// no-adapter model is unaffected.
    pub fn set_lora_pass(&self, pass: usize) {
        self.video.set_lora_pass(pass);
        self.audio.set_lora_pass(pass);
        for b in &self.blocks {
            b.set_lora_pass(pass);
        }
    }

    /// Joint velocity forward.
    ///
    /// * `*_latent` — `(B, S, in_channels)` patchified tokens (video 128, audio 128).
    /// * `*_timestep` — `(B, S)` per-token sigma.
    /// * `*_context` — text embeddings (video 4096, audio 2048); `*_mask` their additive masks.
    /// * `*_positions` — the position grids (video `(B,3,T,2)`, audio `(B,1,T,2)`).
    ///
    /// Returns `(video_velocity (B, S_v, 128), audio_velocity (B, S_a, 128))`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        video_latent: &Array,
        video_timestep: &Array,
        video_context: &Array,
        video_mask: Option<&Array>,
        video_positions: &Array,
        audio_latent: &Array,
        audio_timestep: &Array,
        audio_context: &Array,
        audio_mask: Option<&Array>,
        audio_positions: &Array,
    ) -> Result<(Array, Array)> {
        let vp =
            self.video
                .prepare(video_latent, video_timestep, video_context, video_positions)?;
        let ap =
            self.audio
                .prepare(audio_latent, audio_timestep, audio_context, audio_positions)?;
        let (mut vx, mut ax) = (vp.x.clone(), ap.x.clone());
        let (va, aa) = (vp.args(video_mask), ap.args(audio_mask));
        for block in &self.blocks {
            let (nv, na) = block.forward(&vx, &ax, &va, &aa)?;
            vx = nv;
            ax = na;
        }
        let v_vel = self.video.output_head(&vx, &vp.emb_ts)?;
        let a_vel = self.audio.output_head(&ax, &ap.emb_ts)?;
        Ok((v_vel, a_vel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_memo_caches_on_positions_and_recomputes_on_change() {
        // F-048: the per-stage RoPE tables are recomputed only when `positions` changes. A constant
        // positions (every denoise step) hits the cache; a new positions (next stage) recomputes.
        let memo = RopeMemo::default();
        let calls = std::cell::Cell::new(0);
        let compute = |tag: f32| {
            // Distinct table content per "stage" so we can tell a hit from a recompute.
            (
                Array::from_slice(&[tag, tag + 1.0], &[2]),
                Array::from_slice(&[tag + 2.0, tag + 3.0], &[2]),
            )
        };

        let pos_a = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]);
        let pos_a2 = Array::from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]); // same content
        let pos_b = Array::from_slice(&[9.0f32, 1.0, 2.0, 3.0], &[1, 1, 2, 2]); // different

        let (c0, _) = memo
            .get_or_compute(&pos_a, || {
                calls.set(calls.get() + 1);
                Ok(compute(10.0))
            })
            .unwrap();
        // Same positions content → cache hit, compute not re-run, identical table returned.
        let (c1, _) = memo
            .get_or_compute(&pos_a2, || {
                calls.set(calls.get() + 1);
                Ok(compute(99.0)) // would differ if (wrongly) recomputed
            })
            .unwrap();
        assert_eq!(calls.get(), 1, "same positions must hit the cache");
        assert_eq!(c0.as_slice::<f32>(), c1.as_slice::<f32>());

        // Different positions → recompute.
        let (c2, _) = memo
            .get_or_compute(&pos_b, || {
                calls.set(calls.get() + 1);
                Ok(compute(20.0))
            })
            .unwrap();
        assert_eq!(calls.get(), 2, "changed positions must recompute");
        assert_eq!(c2.as_slice::<f32>(), &[20.0, 21.0]);
    }

    #[test]
    fn modulate_closed_form() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let scale = Array::from_slice(&[0.0f32, 1.0, 0.0, 1.0], &[1, 1, 4]);
        let shift = Array::from_slice(&[1.0f32, 0.0, -1.0, 0.0], &[1, 1, 4]);
        let got = modulate(&x, &scale, &shift).unwrap();
        assert_eq!(got.as_slice::<f32>(), &[2.0, 4.0, 2.0, 8.0]);
    }

    // sc-2963: the compiled glue helpers (`modulate`/`gated`/`gelu_ffn`) the AvDiT forward composes
    // are bit-identical to eager (`max|Δ|=0`) at both LTX compute dtypes — f32 (the `quant_f32`
    // quality target) and bf16 (the `quant_bf16` speed path). The block forward has no committed
    // weights (the Q8 block lives in the 20 GB checkpoint), so this gates the helpers directly.
    #[test]
    fn compiled_glue_bit_identical_to_eager() {
        use mlx_rs::random;
        let rnd = |shape: &[i32], dt: Dtype| -> Array {
            let k = random::key(0).unwrap();
            random::normal::<f32>(shape, None, None, Some(&k))
                .unwrap()
                .as_dtype(dt)
                .unwrap()
        };
        let max_abs = |a: &Array, b: &Array| -> f32 {
            let d = mlx_rs::ops::abs(subtract(a, b).unwrap()).unwrap();
            mlx_rs::ops::max(&d, None)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
                .item::<f32>()
        };
        for dt in [Dtype::Float32, Dtype::Bfloat16] {
            let (b, s, dim, ffn) = (1i32, 32i32, 64i32, 256i32);
            // modulate: x·(1+scale)+shift, scale/shift broadcast [B,1,dim].
            let x = rnd(&[b, s, dim], dt);
            let scale = rnd(&[b, 1, dim], dt);
            let shift = rnd(&[b, 1, dim], dt);
            crate::set_compile_glue(false);
            let me = modulate(&x, &scale, &shift).unwrap();
            crate::set_compile_glue(true);
            let mc = modulate(&x, &scale, &shift).unwrap();
            crate::set_compile_glue(false);
            assert_eq!(mc.dtype(), dt, "modulate dtype {dt:?}");
            assert_eq!(max_abs(&mc, &me), 0.0, "modulate {dt:?}");

            // gated: x + out·gate.
            let out = rnd(&[b, s, dim], dt);
            let gate = rnd(&[b, 1, dim], dt);
            crate::set_compile_glue(false);
            let ge = gated(&x, &out, &gate).unwrap();
            crate::set_compile_glue(true);
            let gc = gated(&x, &out, &gate).unwrap();
            crate::set_compile_glue(false);
            assert_eq!(max_abs(&gc, &ge), 0.0, "gated {dt:?}");

            // gelu_ffn: the tanh-GELU FFN activation (eager defers to core gelu_tanh).
            let h = rnd(&[b, s, ffn], dt);
            crate::set_compile_glue(false);
            let fe = gelu_ffn(&h).unwrap();
            crate::set_compile_glue(true);
            let fc = gelu_ffn(&h).unwrap();
            crate::set_compile_glue(false);
            assert_eq!(fc.dtype(), dt, "gelu_ffn dtype {dt:?}");
            assert_eq!(max_abs(&fc, &fe), 0.0, "gelu_ffn {dt:?}");
        }
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

    use mlx_rs::ops::{all_close, array_eq};

    /// A bare dense Linear (no bias contribution) for the residual-math gates.
    fn dense(w: Array) -> Linear {
        let out = w.shape()[0];
        Linear {
            kind: LinearKind::Dense {
                w,
                b: Array::zeros::<f32>(&[out]).unwrap(),
            },
            lora: None,
        }
    }

    #[test]
    fn lora_scale_zero_is_bit_exact_noop_and_nonzero_has_effect() {
        // base [out=2, in=3]; LoRA a=[in=3, rank=2], b=[rank=2, out=2] (residual form, alpha/rank
        // already folded into the per-pass scale by the loader — here it's the raw scale).
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let base = dense(w.clone()).forward(&x).unwrap();

        // scale 0 → bit-exact no-op (`out + 0·residual`).
        let mut lin0 = dense(w.clone());
        lin0.push_lora(a.clone(), b.clone(), vec![0.0]);
        assert!(array_eq(lin0.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());

        // scale 0.5 → differs from base and equals `base + 0.5·(x·a)·b` exactly.
        let mut lin1 = dense(w);
        lin1.push_lora(a.clone(), b.clone(), vec![0.5]);
        let got = lin1.forward(&x).unwrap();
        assert!(!array_eq(&got, &base, false).unwrap().item::<bool>());
        let resid = multiply(matmul(matmul(&x, &a).unwrap(), &b).unwrap(), scalar(0.5)).unwrap();
        let want = add(&base, &resid).unwrap();
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn lora_per_pass_selects_strength() {
        // pass_scale [0.0, 1.0]: pass 0 is a no-op, pass 1 applies the full residual (sc-2687).
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let base = dense(w.clone()).forward(&x).unwrap();

        let mut lin = dense(w);
        lin.push_lora(a, b, vec![0.0, 1.0]);
        lin.set_lora_pass(0);
        assert!(
            array_eq(lin.forward(&x).unwrap(), &base, false)
                .unwrap()
                .item::<bool>(),
            "pass 0 (strength 0) must be a no-op"
        );
        lin.set_lora_pass(1);
        assert!(
            !array_eq(lin.forward(&x).unwrap(), &base, false)
                .unwrap()
                .item::<bool>(),
            "pass 1 (strength 1) must change the output"
        );
    }

    #[test]
    fn lora_pass_index_clamps_into_uniform_scale() {
        // A uniform (length-1) pass_scale is used regardless of the selected pass — the clamp keeps a
        // uniform adapter working even when the pipeline advances to pass 1.
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let mut lin = dense(w);
        lin.push_lora(a, b, vec![0.5]);
        lin.set_lora_pass(0);
        let p0 = lin.forward(&x).unwrap();
        lin.set_lora_pass(1);
        let p1 = lin.forward(&x).unwrap();
        assert!(array_eq(&p0, &p1, false).unwrap().item::<bool>());
    }

    #[test]
    fn lokr_scale_zero_is_noop_nonzero_matches_delta_and_per_pass() {
        // sc-2393: a LoKr residual carries a precomputed `[out,in]` delta (alpha/rank baked in). The
        // forward is `out + pass_scale · x·ΔWᵀ` — scale 0 a bit-exact no-op, scale s = `base + s·x·ΔWᵀ`,
        // and the per-pass strength selects exactly like LoRA. `base [out=2,in=3]`, `ΔW [2,3]`.
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let delta = Array::from_slice(&[0.05f32, -0.1, 0.2, 0.15, -0.25, 0.3], &[2, 3]);
        let base = dense(w.clone()).forward(&x).unwrap();

        // scale 0 → bit-exact no-op.
        let mut lin0 = dense(w.clone());
        lin0.push_lokr(delta.clone(), vec![0.0]);
        assert!(array_eq(lin0.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());

        // scale 0.5 → `base + 0.5·x·ΔWᵀ` exactly, and differs from base.
        let mut lin = dense(w.clone());
        lin.push_lokr(delta.clone(), vec![0.5]);
        let got = lin.forward(&x).unwrap();
        assert!(!array_eq(&got, &base, false).unwrap().item::<bool>());
        let resid = multiply(matmul(&x, delta.t()).unwrap(), scalar(0.5)).unwrap();
        let want = add(&base, &resid).unwrap();
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());

        // Per-pass [0.0, 1.0]: pass 0 no-op, pass 1 applies.
        let mut linp = dense(w);
        linp.push_lokr(delta, vec![0.0, 1.0]);
        linp.set_lora_pass(0);
        assert!(array_eq(linp.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());
        linp.set_lora_pass(1);
        assert!(!array_eq(linp.forward(&x).unwrap(), &base, false)
            .unwrap()
            .item::<bool>());
    }

    #[test]
    fn lora_and_lokr_stack_on_one_linear() {
        // Both adapter kinds sum onto a single base: `out + lora_resid + lokr_resid`.
        let w = Array::from_slice(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &[2, 3]);
        let x = Array::from_slice(&[1.0f32, -2.0, 0.5], &[1, 3]);
        let a = Array::from_slice(&[0.1f32, 0.2, 0.3, -0.1, -0.2, 0.4], &[3, 2]);
        let b = Array::from_slice(&[0.5f32, -0.5, 0.25, 0.75], &[2, 2]);
        let delta = Array::from_slice(&[0.05f32, -0.1, 0.2, 0.15, -0.25, 0.3], &[2, 3]);
        let base = dense(w.clone()).forward(&x).unwrap();

        let mut lin = dense(w);
        lin.push_lora(a.clone(), b.clone(), vec![0.5]);
        lin.push_lokr(delta.clone(), vec![0.25]);
        let got = lin.forward(&x).unwrap();

        let lora_r = multiply(matmul(matmul(&x, &a).unwrap(), &b).unwrap(), scalar(0.5)).unwrap();
        let lokr_r = multiply(matmul(&x, delta.t()).unwrap(), scalar(0.25)).unwrap();
        let want = add(add(&base, &lora_r).unwrap(), &lokr_r).unwrap();
        assert!(all_close(&got, &want, 1e-6, 1e-6, false)
            .unwrap()
            .item::<bool>());
    }
}
