//! Lens joint (dual-stream) attention (`LensJointAttention`). **Fused** `img_qkv`/`txt_qkv`
//! projections (bias) split into per-stream q/k/v, per-head q/k RMSNorm, interleaved-complex RoPE on
//! both streams, then SDPA over the **`[img, txt]`**-concatenated sequence (matching the Lens
//! `_build_joint_attention_mask` which orders image tokens first), split back and projected
//! (`to_out.0` for image, `to_add_out` for text).

use mlx_rs::error::Result as MlxResult;
use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::ops::{add, concatenate_axis, multiply, split, split_sections, subtract};
use mlx_rs::transforms::checkpoint;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, load_weight};

/// QK-RMSNorm + block-norm epsilon (the Lens block builds `LensJointAttention(eps=1e-6)` via its own
/// `eps` default).
const RMS_EPS: f32 = 1e-6;

/// Load a biased diffusers `[out, in]` projection as an [`AdaptableLinear`] (the LoRA/LoKr adapter
/// targets, sc-3174). The dense forward is `x·Wᵀ + b`, identical to the sc-3168 [`super::Linear`].
fn load_adaptable(w: &Weights, prefix: &str, dtype: Dtype) -> Result<AdaptableLinear> {
    let weight = w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?;
    let bias = w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?;
    Ok(AdaptableLinear::dense(weight, Some(bias)))
}

#[derive(Clone)]
pub struct LensJointAttention {
    img_qkv: AdaptableLinear,
    txt_qkv: AdaptableLinear,
    to_out: AdaptableLinear,
    to_add_out: AdaptableLinear,
    norm_q: Array,
    norm_k: Array,
    norm_added_q: Array,
    norm_added_k: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    /// sc-5170 — run the joint SDPA inside an `mlx::checkpoint` so its backward recomputes the
    /// attention instead of retaining the `[heads, joint, joint]` probability matrix (the grad
    /// through `fast::scaled_dot_product_attention` decomposes to naive attention — MLX has no fused
    /// SDPA backward — and that one retained seq² array per block dominates the dense training
    /// working set). Numerically identical (same math, recomputed); inference never sets it (default
    /// off, zero cost), the trainer enables it unconditionally (LoRA + LoKr).
    ckpt_sdpa: bool,
}

impl LensJointAttention {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            img_qkv: load_adaptable(w, &join(prefix, "img_qkv"), dtype)?,
            txt_qkv: load_adaptable(w, &join(prefix, "txt_qkv"), dtype)?,
            to_out: load_adaptable(w, &join(prefix, "to_out.0"), dtype)?,
            to_add_out: load_adaptable(w, &join(prefix, "to_add_out"), dtype)?,
            norm_q: load_weight(w, &join(prefix, "norm_q"), dtype)?,
            norm_k: load_weight(w, &join(prefix, "norm_k"), dtype)?,
            norm_added_q: load_weight(w, &join(prefix, "norm_added_q"), dtype)?,
            norm_added_k: load_weight(w, &join(prefix, "norm_added_k"), dtype)?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            ckpt_sdpa: false,
        })
    }

    /// Toggle SDPA-segment gradient checkpointing (sc-5170). Training-only knob — see `ckpt_sdpa`.
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.ckpt_sdpa = on;
    }

    /// Quantize the four projections to Q4/Q8 (sc-3175). Call **after** any adapter merge — the
    /// adapters are forward-time residuals over the (now quantized) base, exactly as the shared seam
    /// intends, so a quantized base + LoRA residual compose. The QK-norm weights stay full precision.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.img_qkv.quantize(bits, None)?;
        self.txt_qkv.quantize(bits, None)?;
        self.to_out.quantize(bits, None)?;
        self.to_add_out.quantize(bits, None)?;
        Ok(())
    }

    /// `img`/`txt`: `[B, seq, dim]`; rope tables `[seq, head_dim/2]`; `mask`: optional additive
    /// `[B, 1, 1, img+txt]`. Returns `(img_attn, txt_attn)`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        img: &Array,
        txt: &Array,
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let (b, img_seq) = (img.shape()[0], img.shape()[1]);
        let txt_seq = txt.shape()[1];
        let (h, hd) = (self.num_heads, self.head_dim);

        // Fused QKV per stream → [B, seq, 3, heads, head_dim] → q/k/v each [B, seq, heads, head_dim].
        let qkv = |lin: &AdaptableLinear, x: &Array, seq: i32| -> Result<(Array, Array, Array)> {
            let t = lin.forward(x)?.reshape(&[b, seq, 3, h, hd])?;
            let parts = split(&t, 3, 2)?; // 3 × [B, seq, 1, heads, head_dim]
            let q = parts[0].reshape(&[b, seq, h, hd])?;
            let k = parts[1].reshape(&[b, seq, h, hd])?;
            let v = parts[2].reshape(&[b, seq, h, hd])?;
            Ok((q, k, v))
        };
        let (img_q, img_k, img_v) = qkv(&self.img_qkv, img, img_seq)?;
        let (txt_q, txt_k, txt_v) = qkv(&self.txt_qkv, txt, txt_seq)?;

        // QK RMSNorm over head_dim.
        let img_q = rms_norm(&img_q, &self.norm_q, RMS_EPS)?;
        let img_k = rms_norm(&img_k, &self.norm_k, RMS_EPS)?;
        let txt_q = rms_norm(&txt_q, &self.norm_added_q, RMS_EPS)?;
        let txt_k = rms_norm(&txt_k, &self.norm_added_k, RMS_EPS)?;

        // Per-stream interleaved-complex RoPE.
        let img_q = apply_rope(&img_q, img_cos, img_sin)?;
        let img_k = apply_rope(&img_k, img_cos, img_sin)?;
        let txt_q = apply_rope(&txt_q, txt_cos, txt_sin)?;
        let txt_k = apply_rope(&txt_k, txt_cos, txt_sin)?;

        // Joint [img, txt] over the sequence axis, then [B, heads, seq, head_dim] for SDPA.
        let q = concatenate_axis(&[&img_q, &txt_q], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = concatenate_axis(&[&img_k, &txt_k], 1)?.transpose_axes(&[0, 2, 1, 3])?;
        let v = concatenate_axis(&[&img_v, &txt_v], 1)?.transpose_axes(&[0, 2, 1, 3])?;

        let o = if self.ckpt_sdpa {
            // sc-5170: checkpoint just the joint SDPA. q/k/v are the threaded inputs (grads to the
            // QKV projections — and their LoRA — flow back through them); the f32 scale and the
            // additive mask are captured constants (the mask carries no trainable graph). The
            // backward recomputes the decomposed attention for THIS block alone, so the
            // `[heads, joint, joint]` probability matrix is a per-block transient, never 48×
            // retained.
            let scale = self.scale;
            let m = mask.cloned();
            let mut seg = checkpoint(move |inp: &[Array]| -> MlxResult<Vec<Array>> {
                let o = match m.as_ref() {
                    Some(mm) => {
                        scaled_dot_product_attention(&inp[0], &inp[1], &inp[2], scale, mm, None)?
                    }
                    None => {
                        scaled_dot_product_attention(&inp[0], &inp[1], &inp[2], scale, None, None)?
                    }
                };
                Ok(vec![o])
            });
            seg(&[q, k, v])?
                .into_iter()
                .next()
                .expect("one sdpa output")
        } else {
            match mask {
                Some(m) => scaled_dot_product_attention(&q, &k, &v, self.scale, m, None)?,
                None => scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?,
            }
        };
        let joint = img_seq + txt_seq;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, joint, h * hd])?;

        // Split back at the image/text boundary (image first).
        let parts = split_sections(&o, &[img_seq], 1)?;
        let img_attn = self.to_out.forward(&parts[0])?;
        let txt_attn = self.to_add_out.forward(&parts[1])?;
        Ok((img_attn, txt_attn))
    }
}

impl AdaptableHost for LensJointAttention {
    /// Trained-file (diffusers/peft) module names → the fused attention projections (sc-3174). The
    /// Lens trainer's `DEFAULT_LORA_TARGET_MODULES` = `img_qkv` / `txt_qkv` / `to_out.0` / `to_add_out`
    /// (the QKV are fused `[3·inner, in]`, so a LoRA on them merges whole — no q/k/v split). `to_out`
    /// is a `ModuleList([Linear, Identity])`, addressed `to_out.0`; accept the bare `to_out` alias too.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["img_qkv"] => Some(&mut self.img_qkv),
            ["txt_qkv"] => Some(&mut self.txt_qkv),
            ["to_out"] | ["to_out", "0"] => Some(&mut self.to_out),
            ["to_add_out"] => Some(&mut self.to_add_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

/// Interleaved complex RoPE: pairs `(x_2i, x_2i+1)` rotated by `(cos_i, sin_i)`, reproducing the
/// reference `view_as_complex(...)·freqs_cis`. `x`: `[B, seq, heads, head_dim]`; `cos`/`sin`:
/// `[seq, head_dim/2]` (f32). The rotation is computed in the promoted dtype and cast back to `x`'s
/// dtype (`type_as(x)`).
fn apply_rope(x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
    let sh = x.shape();
    let (b, seq, heads, hd) = (sh[0], sh[1], sh[2], sh[3]);
    let half = hd / 2;
    let x5 = x.reshape(&[b, seq, heads, half, 2])?;
    let parts = split(&x5, 2, 4)?; // even/odd lanes
    let xr = parts[0].reshape(&[b, seq, heads, half])?;
    let xi = parts[1].reshape(&[b, seq, heads, half])?;
    let cos = cos.reshape(&[1, seq, 1, half])?;
    let sin = sin.reshape(&[1, seq, 1, half])?;
    let out_r = subtract(&multiply(&xr, &cos)?, &multiply(&xi, &sin)?)?;
    let out_i = add(&multiply(&xr, &sin)?, &multiply(&xi, &cos)?)?;
    let stacked = concatenate_axis(&[&out_r.expand_dims(4)?, &out_i.expand_dims(4)?], 4)?;
    Ok(stacked.reshape(&[b, seq, heads, hd])?.as_dtype(x.dtype())?)
}
