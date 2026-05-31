//! Full Z-Image text encoder: token embedding → N pre-norm decoder layers → the **second-to-
//! last** layer's hidden states (no final norm), cast to f32. Port of the fork's `TextEncoder`.
//! These hidden states are the DiT's `cap_feats` conditioning (after slicing to valid tokens).

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, EncoderLayer, TextRope};

/// Z-Image text-encoder dimensions (Qwen3-style decoder LM).
pub struct ZTextEncoderConfig {
    pub vocab_size: i32,
    pub hidden_size: i32,
    pub n_layers: usize,
    pub n_heads: i32,
    pub n_kv_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl ZTextEncoderConfig {
    /// The production Z-Image-turbo text encoder (`Tongyi-MAI/Z-Image` `text_encoder/`).
    pub fn z_image() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 2560,
            n_layers: 36,
            n_heads: 32,
            n_kv_heads: 8,
            head_dim: 128,
            intermediate_size: 9728,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
        }
    }
}

pub struct TextEncoder {
    embed_tokens: Array,
    layers: Vec<EncoderLayer>,
    rope: TextRope,
}

impl TextEncoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ZTextEncoderConfig) -> Result<Self> {
        let embed_tokens = w.require(&join(prefix, "embed_tokens.weight"))?.clone();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lp = join(prefix, &format!("layers.{i}"));
            layers.push(EncoderLayer::from_weights(
                w,
                &lp,
                cfg.n_heads,
                cfg.n_kv_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
            )?);
        }
        // The fork also has a final `norm`, but it is never applied (the returned [-2] layer is
        // un-normed), so we don't load it.
        Ok(Self {
            embed_tokens,
            layers,
            rope: TextRope::new(cfg.head_dim, cfg.rope_theta),
        })
    }

    /// `input_ids` / `attention_mask`: `[b, s]` int32. Returns `[b, s, hidden]` (f32) — the
    /// second-to-last layer's hidden states, matching the fork's `all_hidden_states[-2]`.
    pub fn forward(&self, input_ids: &Array, attention_mask: &Array) -> Result<Array> {
        let sh = input_ids.shape();
        let (b, s) = (sh[0], sh[1]);

        let embed = self
            .embed_tokens
            .take_axis(input_ids, 0)?
            .as_dtype(Dtype::Float32)?; // [b, s, hidden]
        let (cos, sin) = self.rope.forward(s)?;
        let mask = build_mask(attention_mask, b, s)?;

        // all_hidden_states = [embed, out(L0), out(L1), ...]; return the second-to-last.
        let mut hidden = vec![embed];
        for layer in &self.layers {
            let h = layer.forward(hidden.last().unwrap(), &cos, &sin, &mask)?;
            hidden.push(h);
        }
        Ok(hidden[hidden.len() - 2].clone())
    }
}

/// Additive attention mask `[b, 1, s, s]`: `0` where a query may attend (key is causal **and**
/// not padding), `-inf` otherwise — the fork's causal ⊕ padding combination.
fn build_mask(attention_mask: &Array, b: i32, s: i32) -> Result<Array> {
    let am = attention_mask.as_slice::<i32>();
    let (b, s) = (b as usize, s as usize);
    let mut data = vec![0f32; b * s * s];
    for bi in 0..b {
        for i in 0..s {
            for j in 0..s {
                let allowed = j <= i && am[bi * s + j] == 1;
                if !allowed {
                    data[(bi * s + i) * s + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    Ok(Array::from_slice(&data, &[b as i32, 1, s as i32, s as i32]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_is_causal_and_masks_padding() {
        // b=1, s=3, last token padded.
        let am = Array::from_slice(&[1i32, 1, 0], &[1, 3]);
        let m = build_mask(&am, 1, 3).unwrap();
        let v = m.as_slice::<f32>(); // [1,1,3,3] -> 9 values, row-major [query][key]
        let neg = f32::NEG_INFINITY;
        // query 0: key0 allowed, key1 future, key2 future+pad
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], neg);
        assert_eq!(v[2], neg);
        // query 1: key0,key1 allowed, key2 future
        assert_eq!(v[3], 0.0);
        assert_eq!(v[4], 0.0);
        assert_eq!(v[5], neg);
        // query 2: key0,key1 allowed (causal), key2 padded -> masked
        assert_eq!(v[6], 0.0);
        assert_eq!(v[7], 0.0);
        assert_eq!(v[8], neg);
    }
}
