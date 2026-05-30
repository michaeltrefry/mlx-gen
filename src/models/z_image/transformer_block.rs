//! Z-Image DiT block: adaLN modulation → modulated attention residual → modulated SwiGLU
//! FFN residual. Port of the Python fork's `transformer_block.py`, dimension-parametric.
//! Whole-block fp32 parity vs the Python reference is covered by `tests/z_image_block.rs`.

use mlx_rs::{
    fast::rms_norm,
    ops::{add, multiply, split, tanh},
    Array,
};

use crate::adapters::{AdaptableHost, AdaptableLinear};
use crate::models::z_image::attention::ZImageAttention;
use crate::models::z_image::feed_forward::FeedForward;
use crate::weights::Weights;
use crate::Result;

/// Shape of one Z-Image transformer block.
#[derive(Debug, Clone, Copy)]
pub struct ZImageBlockConfig {
    pub dim: i32,
    pub n_heads: i32,
    pub norm_eps: f32,
}

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

pub struct ZImageTransformerBlock {
    pub attention: ZImageAttention,
    pub feed_forward: FeedForward,
    attention_norm1: Array,
    attention_norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    ada_ln: AdaptableLinear,
    eps: f32,
}

impl ZImageTransformerBlock {
    /// Load a block from weights under `prefix` (e.g. `"transformer.layers.0"`, or `"w"` for
    /// the standalone parity fixture). Keys mirror the Python `tree_flatten` layout.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: ZImageBlockConfig) -> Result<Self> {
        let ada_w = w
            .require(&format!("{prefix}.adaLN_modulation.0.weight"))?
            .clone();
        let ada_b = w.get(&format!("{prefix}.adaLN_modulation.0.bias")).cloned();
        Ok(Self {
            attention: ZImageAttention::from_weights(
                w,
                &format!("{prefix}.attention"),
                cfg.dim,
                cfg.n_heads,
                cfg.norm_eps,
            )?,
            feed_forward: FeedForward::from_weights(w, &format!("{prefix}.feed_forward"))?,
            attention_norm1: w
                .require(&format!("{prefix}.attention_norm1.weight"))?
                .clone(),
            attention_norm2: w
                .require(&format!("{prefix}.attention_norm2.weight"))?
                .clone(),
            ffn_norm1: w.require(&format!("{prefix}.ffn_norm1.weight"))?.clone(),
            ffn_norm2: w.require(&format!("{prefix}.ffn_norm2.weight"))?.clone(),
            ada_ln: AdaptableLinear::dense(ada_w, ada_b),
            eps: cfg.norm_eps,
        })
    }

    pub fn forward(&self, x: &Array, freqs_cis: &Array, t_emb: &Array) -> Result<Array> {
        // adaLN modulation: (1, 4*dim) -> (1, 1, 4*dim) -> 4 × (1, 1, dim)
        let modulation = self.ada_ln.forward(t_emb)?.expand_dims(1)?;
        let p = split(&modulation, 4, 2)?;
        let scale_msa = add(&p[0], scalar(1.0))?;
        let gate_msa = tanh(&p[1])?;
        let scale_mlp = add(&p[2], scalar(1.0))?;
        let gate_mlp = tanh(&p[3])?;

        // Modulated attention residual.
        let s1 = multiply(&rms_norm(x, &self.attention_norm1, self.eps)?, &scale_msa)?;
        let attn_out = self.attention.forward(&s1, freqs_cis)?;
        let x1 = add(
            x,
            &multiply(
                &gate_msa,
                &rms_norm(&attn_out, &self.attention_norm2, self.eps)?,
            )?,
        )?;

        // Modulated SwiGLU FFN residual.
        let s2 = multiply(&rms_norm(&x1, &self.ffn_norm1, self.eps)?, &scale_mlp)?;
        let ffn_out = self.feed_forward.forward(&s2)?;
        Ok(add(
            &x1,
            &multiply(&gate_mlp, &rms_norm(&ffn_out, &self.ffn_norm2, self.eps)?)?,
        )?)
    }
}

impl AdaptableHost for ZImageTransformerBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["attention", rest @ ..] => self.attention.adaptable_mut(rest),
            ["feed_forward", rest @ ..] => self.feed_forward.adaptable_mut(rest),
            ["adaLN_modulation", "0"] => Some(&mut self.ada_ln),
            _ => None,
        }
    }
}
