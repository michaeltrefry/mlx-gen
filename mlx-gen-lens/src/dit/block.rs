//! Lens dual-stream MMDiT block (`LensTransformerBlock`). Each stream (image, text) gets two AdaLN
//! modulations from the timestep embedding — `mod1` around the joint attention, `mod2` around the
//! **SwiGLU** MLP — with gated residuals. Norms are affine **RMSNorm** (`rms_norm=True`, eps 1e-6).

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::{add, multiply, split};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, load_weight, LensJointAttention, Linear};

const NORM_EPS: f32 = 1e-6;

/// SwiGLU MLP (`GateMLP`): `w2(silu(w1(x)) · w3(x))`, all bias-less. Hidden width `dim/3·8`. The
/// three projections are [`AdaptableLinear`] so they can be Q4/Q8-quantized (sc-3175).
#[derive(Clone)]
struct GateMlp {
    w1: AdaptableLinear,
    w2: AdaptableLinear,
    w3: AdaptableLinear,
}

impl GateMlp {
    fn from_weights(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Self> {
        let load = |name: &str| -> Result<AdaptableLinear> {
            let weight = load_weight(w, &join(prefix, name), dtype)?;
            Ok(AdaptableLinear::dense(weight, None))
        };
        Ok(Self {
            w1: load("w1")?,
            w2: load("w2")?,
            w3: load("w3")?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let gate = silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&multiply(&gate, &up)?)
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.w1.quantize(bits, None)?;
        self.w2.quantize(bits, None)?;
        self.w3.quantize(bits, None)?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct LensTransformerBlock {
    img_mod: Linear,
    txt_mod: Linear,
    img_norm1: Array,
    img_norm2: Array,
    txt_norm1: Array,
    txt_norm2: Array,
    attn: LensJointAttention,
    img_mlp: GateMlp,
    txt_mlp: GateMlp,
}

impl LensTransformerBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        Ok(Self {
            img_mod: Linear::load(w, &join(prefix, "img_mod.1"), true, dtype)?,
            txt_mod: Linear::load(w, &join(prefix, "txt_mod.1"), true, dtype)?,
            img_norm1: load_weight(w, &join(prefix, "img_norm1"), dtype)?,
            img_norm2: load_weight(w, &join(prefix, "img_norm2"), dtype)?,
            txt_norm1: load_weight(w, &join(prefix, "txt_norm1"), dtype)?,
            txt_norm2: load_weight(w, &join(prefix, "txt_norm2"), dtype)?,
            attn: LensJointAttention::from_weights(
                w,
                &join(prefix, "attn"),
                num_heads,
                head_dim,
                dtype,
            )?,
            img_mlp: GateMlp::from_weights(w, &join(prefix, "img_mlp"), dtype)?,
            txt_mlp: GateMlp::from_weights(w, &join(prefix, "txt_mlp"), dtype)?,
        })
    }

    /// Quantize the block's compute-heavy linears to Q4/Q8 (sc-3175): the joint-attention projections
    /// and both SwiGLU MLPs. The AdaLN modulations (`img_mod`/`txt_mod`) and the RMSNorm weights stay
    /// full precision (small and precision-sensitive).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.img_mlp.quantize(bits)?;
        self.txt_mlp.quantize(bits)?;
        Ok(())
    }

    /// Toggle SDPA-segment gradient checkpointing on this block's joint attention (sc-5170).
    pub fn set_sdpa_checkpoint(&mut self, on: bool) {
        self.attn.set_sdpa_checkpoint(on);
    }

    /// Returns `(encoder_hidden_states, hidden_states)` (text, image) — the reference block's order.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        hidden_states: &Array,         // image [B, img_seq, dim]
        encoder_hidden_states: &Array, // text  [B, txt_seq, dim]
        temb: &Array,                  // [B, dim]
        img_cos: &Array,
        img_sin: &Array,
        txt_cos: &Array,
        txt_sin: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        // SiLU'd timestep → per-stream 6·dim modulation, split into mod1 (around attn) / mod2 (MLP).
        let act = silu(temb)?;
        let img_mod = split(&self.img_mod.forward(&act)?, 2, 1)?; // [mod1, mod2], each [B, 3·dim]
        let txt_mod = split(&self.txt_mod.forward(&act)?, 2, 1)?;

        let (img_modulated, img_gate1) = modulate(
            &rms_norm(hidden_states, &self.img_norm1, NORM_EPS)?,
            &img_mod[0],
        )?;
        let (txt_modulated, txt_gate1) = modulate(
            &rms_norm(encoder_hidden_states, &self.txt_norm1, NORM_EPS)?,
            &txt_mod[0],
        )?;

        let (img_attn, txt_attn) = self.attn.forward(
            &img_modulated,
            &txt_modulated,
            img_cos,
            img_sin,
            txt_cos,
            txt_sin,
            mask,
        )?;

        let hidden_states = add(hidden_states, &multiply(&img_gate1, &img_attn)?)?;
        let encoder_hidden_states = add(encoder_hidden_states, &multiply(&txt_gate1, &txt_attn)?)?;

        let (img_modulated2, img_gate2) = modulate(
            &rms_norm(&hidden_states, &self.img_norm2, NORM_EPS)?,
            &img_mod[1],
        )?;
        let hidden_states = add(
            &hidden_states,
            &multiply(&img_gate2, &self.img_mlp.forward(&img_modulated2)?)?,
        )?;

        let (txt_modulated2, txt_gate2) = modulate(
            &rms_norm(&encoder_hidden_states, &self.txt_norm2, NORM_EPS)?,
            &txt_mod[1],
        )?;
        let encoder_hidden_states = add(
            &encoder_hidden_states,
            &multiply(&txt_gate2, &self.txt_mlp.forward(&txt_modulated2)?)?,
        )?;

        Ok((encoder_hidden_states, hidden_states))
    }
}

impl AdaptableHost for LensTransformerBlock {
    /// The adapter targets live in the joint attention (`attn.{img_qkv,txt_qkv,to_out.0,to_add_out}`);
    /// the modulations and SwiGLU MLPs are not in the Lens trainer's target set.
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attn", rest @ ..] => self.attn.adaptable_mut(rest),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        mlx_gen::adapters::prefixed_paths("attn", &self.attn)
    }
}

/// `(x·(1+scale) + shift, gate)` from a `[B, 3·dim]` modulation laid out **(shift, scale, gate)**
/// (the reference `_modulate`). Scale/shift/gate broadcast over the sequence axis.
fn modulate(x: &Array, mod_params: &Array) -> Result<(Array, Array)> {
    let p = split(mod_params, 3, 1)?; // shift, scale, gate — each [B, dim]
    let shift = p[0].expand_dims(1)?; // [B, 1, dim]
    let scale = p[1].expand_dims(1)?;
    let gate = p[2].expand_dims(1)?;
    let one = Array::from_slice(&[1.0f32], &[1]);
    let out = add(&multiply(x, &add(&scale, &one)?)?, &shift)?;
    Ok((out, gate))
}
