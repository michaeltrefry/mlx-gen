//! Z-Image caption-refiner block. Port of `ZImageContextBlock`: the same attention + SwiGLU
//! FFN as the main block, but with plain pre-norm residuals and **no** timestep (adaLN)
//! modulation — it refines the text/caption stream, which carries no timestep.

use mlx_rs::fast::rms_norm;
use mlx_rs::ops::add;
use mlx_rs::Array;

use super::attention::ZImageAttention;
use super::feed_forward::FeedForward;
use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct ZImageContextBlock {
    pub attention: ZImageAttention,
    pub feed_forward: FeedForward,
    attention_norm1: Array,
    attention_norm2: Array,
    ffn_norm1: Array,
    ffn_norm2: Array,
    eps: f32,
}

impl ZImageContextBlock {
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dim: i32,
        n_heads: i32,
        norm_eps: f32,
    ) -> Result<Self> {
        Ok(Self {
            attention: ZImageAttention::from_weights(
                w,
                &format!("{prefix}.attention"),
                dim,
                n_heads,
                norm_eps,
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
            eps: norm_eps,
        })
    }

    /// Quantize the block's Linears to Q4/Q8 (group_size 64). The context block has no adaLN
    /// modulation (it carries no timestep), so only attention + FFN have Linears.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attention.quantize(bits)?;
        self.feed_forward.quantize(bits)?;
        Ok(())
    }

    pub fn forward(&self, x: &Array, freqs_cis: &Array) -> Result<Array> {
        let attn_out = self
            .attention
            .forward(&rms_norm(x, &self.attention_norm1, self.eps)?, freqs_cis)?;
        let x = add(x, &rms_norm(&attn_out, &self.attention_norm2, self.eps)?)?;
        let ffn_out = self
            .feed_forward
            .forward(&rms_norm(&x, &self.ffn_norm1, self.eps)?)?;
        Ok(add(&x, &rms_norm(&ffn_out, &self.ffn_norm2, self.eps)?)?)
    }
}

impl AdaptableHost for ZImageContextBlock {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Context blocks carry no timestep, so there is no `adaLN_modulation` target (the fork's
        // mapping lists one uniformly across layer types, but the context-refiner file never
        // populates it → it simply resolves to None here and the loader reports it unmatched).
        match path {
            ["attention", rest @ ..] => self.attention.adaptable_mut(rest),
            ["feed_forward", rest @ ..] => self.feed_forward.adaptable_mut(rest),
            _ => None,
        }
    }
}
