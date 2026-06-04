//! CLIP text encoders — a Rust port of the vendored `_vendor/mlx_sd/clip.py` (`CLIPTextModel` +
//! `CLIPEncoderLayer`). SDXL conditions on **two** of these: CLIP-L (`text_encoder`, 768-wide, no
//! projection) and OpenCLIP-bigG (`text_encoder_2`, 1280-wide, with a final projection feeding the
//! pooled micro-conditioning). Both run **fp16**, matching the production reference
//! (`StableDiffusionXL(float16=True)`); `load_text_encoder_*_dtype` keeps an f32 path for stage gates.
//!
//! The SDXL conditioning is `concat(te1.hidden_states[-2], te2.hidden_states[-2])` (the
//! penultimate-layer hidden states, BEFORE the final layer-norm) over the full padded sequence, plus
//! `te2.pooled` (the projected EOS token of the final-layer-norm'd output). The encoder applies a
//! causal mask only — there is no padding mask, matching the reference (padding positions flow into
//! the UNet cross-attention).

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::array::host_i32;
use mlx_gen::nn::{gelu_exact, gelu_quick};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::{ClipActivation, ClipTextConfig};

/// CLIP's LayerNorm epsilon (HF `layer_norm_eps`, mlx `nn.LayerNorm` default).
const LN_EPS: f32 = 1e-5;

/// The outputs the SDXL pipeline reads off a CLIP encoder.
pub struct ClipOutput {
    /// Final-layer-norm'd output `[B, N, D]`.
    pub last_hidden_state: Array,
    /// Per-encoder-layer outputs (before the final layer-norm); `hidden_states[-2]` feeds the
    /// SDXL conditioning.
    pub hidden_states: Vec<Array>,
    /// The (optionally projected) EOS-token pooled output `[B, proj_or_D]`.
    pub pooled: Array,
}

/// One CLIP transformer encoder layer (pre-norm self-attention + pre-norm MLP, both residual).
struct ClipEncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    out: AdaptableLinear,
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    act: ClipActivation,
}

impl ClipEncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &ClipTextConfig) -> Result<Self> {
        let g = |name: &str| w.require(&format!("{prefix}.{name}")).cloned();
        let dense = |wn: &str, bn: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{wn}"))?.clone(),
                Some(w.require(&format!("{prefix}.{bn}"))?.clone()),
            ))
        };
        let head_dim = cfg.model_dims / cfg.num_heads;
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            q: dense("self_attn.q_proj.weight", "self_attn.q_proj.bias")?,
            k: dense("self_attn.k_proj.weight", "self_attn.k_proj.bias")?,
            v: dense("self_attn.v_proj.weight", "self_attn.v_proj.bias")?,
            out: dense("self_attn.out_proj.weight", "self_attn.out_proj.bias")?,
            linear1: dense("mlp.fc1.weight", "mlp.fc1.bias")?,
            linear2: dense("mlp.fc2.weight", "mlp.fc2.bias")?,
            num_heads: cfg.num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            act: cfg.hidden_act,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [
            &mut self.q,
            &mut self.k,
            &mut self.v,
            &mut self.out,
            &mut self.linear1,
            &mut self.linear2,
        ] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    fn activation(&self, x: &Array) -> Result<Array> {
        match self.act {
            // CLIP-L "quick_gelu" = `x · sigmoid(1.702·x)` — the vendored `mlx.nn.gelu_fast_approx`
            // (NOT mlx-rs `gelu_fast_approximate`, which uses 1.773). The core `gelu_quick` matches it
            // byte-for-byte: dtype-weak `1.702` (an f32 scalar promotes f16→f32) + `mx.compile` (the
            // fused fp16 rounding — see `gelu_exact`/sc-2721). CLIP-bigG uses exact `gelu`.
            ClipActivation::QuickGelu => gelu_quick(x),
            ClipActivation::Gelu => gelu_exact(x),
        }
    }

    /// `x`: `[B, N, D]`; `mask`: additive causal `[1, 1, N, N]`.
    fn forward(&self, x: &Array, mask: &Array) -> Result<Array> {
        // Self-attention (pre-norm, residual).
        let y = layer_norm(x, Some(&self.ln1_w), Some(&self.ln1_b), LN_EPS)?;
        let y = self.attention(&y, mask)?;
        let x = add(x, &y)?;

        // MLP (pre-norm, residual).
        let y = layer_norm(&x, Some(&self.ln2_w), Some(&self.ln2_b), LN_EPS)?;
        let y = self.linear1.forward(&y)?;
        let y = self.activation(&y)?;
        let y = self.linear2.forward(&y)?;
        Ok(add(&x, &y)?)
    }

    fn attention(&self, x: &Array, mask: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, n) = (sh[0], sh[1]);
        let to_heads = |a: Array| -> Result<Array> {
            Ok(a.reshape(&[b, n, self.num_heads, self.head_dim])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = to_heads(self.q.forward(x)?)?;
        let k = to_heads(self.k.forward(x)?)?;
        let v = to_heads(self.v.forward(x)?)?;
        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, mask, None)?;
        let o =
            o.transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[b, n, self.num_heads * self.head_dim])?;
        self.out.forward(&o)
    }
}

/// A loaded CLIP text encoder.
pub struct ClipTextEncoder {
    token_embedding: Array,
    position_embedding: Array,
    layers: Vec<ClipEncoderLayer>,
    final_ln_w: Array,
    final_ln_b: Array,
    /// TE2 only (`CLIPTextModelWithProjection`): projects the pooled EOS token. Bias-free.
    text_projection: Option<AdaptableLinear>,
}

impl ClipTextEncoder {
    /// Load from a CLIP-text checkpoint. `prefix` is the encoder namespace (`text_model`); the
    /// projection (TE2) is read from the bare top-level `text_projection.weight`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &ClipTextConfig) -> Result<Self> {
        let token_embedding = w
            .require(&format!("{prefix}.embeddings.token_embedding.weight"))?
            .clone();
        let position_embedding = w
            .require(&format!("{prefix}.embeddings.position_embedding.weight"))?
            .clone();
        let layers = (0..cfg.num_layers)
            .map(|i| {
                ClipEncoderLayer::from_weights(w, &format!("{prefix}.encoder.layers.{i}"), cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let final_ln_w = w
            .require(&format!("{prefix}.final_layer_norm.weight"))?
            .clone();
        let final_ln_b = w
            .require(&format!("{prefix}.final_layer_norm.bias"))?
            .clone();
        let text_projection = match cfg.projection_dim {
            Some(_) => Some(AdaptableLinear::dense(
                w.require("text_projection.weight")?.clone(),
                None,
            )),
            None => None,
        };
        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_ln_w,
            final_ln_b,
            text_projection,
        })
    }

    /// Quantize every Linear (the per-layer projections + the text projection) to Q4/Q8. The token /
    /// position embeddings stay dense (gather lookups, not matmuls).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for l in &mut self.layers {
            l.quantize(bits)?;
        }
        if let Some(p) = &mut self.text_projection {
            p.quantize(bits, None)?;
        }
        Ok(())
    }

    /// Run the encoder over `input_ids` `[B, N]` (int32). Returns the last hidden state, the
    /// per-layer hidden states, and the pooled EOS token (projected for TE2).
    pub fn forward(&self, input_ids: &Array) -> Result<ClipOutput> {
        let sh = input_ids.shape();
        let (b, n) = (sh[0], sh[1]);
        let dim = self.token_embedding.shape()[1];

        // Token + position embeddings. `position_embedding.weight[:N]` broadcast over the batch.
        let ids_flat = input_ids.reshape(&[b * n])?;
        let tok = self
            .token_embedding
            .take_axis(&ids_flat, 0)?
            .reshape(&[b, n, dim])?;
        let pos_idx = Array::from_slice(&(0..n).collect::<Vec<i32>>(), &[n]);
        let pos = self.position_embedding.take_axis(&pos_idx, 0)?; // [N, D]
        let mut x = add(&tok, &pos.reshape(&[1, n, dim])?)?;

        let mask = causal_mask(n, x.dtype())?;
        let mut hidden_states = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
            hidden_states.push(x.clone());
        }
        let last_hidden_state =
            layer_norm(&x, Some(&self.final_ln_w), Some(&self.final_ln_b), LN_EPS)?;

        // Pooled: the hidden state at each row's EOS token (argmax of the ids — EOS is the max id),
        // optionally projected. Compute EOS indices on the host (ids are small).
        let ids_host = host_i32(input_ids)?;
        let mut gather = Vec::with_capacity(b as usize);
        for row in 0..b as usize {
            let r = &ids_host[row * n as usize..(row + 1) * n as usize];
            let eos = r
                .iter()
                .enumerate()
                .max_by_key(|(_, &v)| v)
                .map(|(i, _)| i as i32)
                .unwrap_or(0);
            gather.push(row as i32 * n + eos);
        }
        let gather = Array::from_slice(&gather, &[b]);
        let pooled = last_hidden_state
            .reshape(&[b * n, dim])?
            .take_axis(&gather, 0)?;
        let pooled = match &self.text_projection {
            Some(p) => p.forward(&pooled)?,
            None => pooled,
        };

        Ok(ClipOutput {
            last_hidden_state,
            hidden_states,
            pooled,
        })
    }
}

/// Build the additive causal attention mask `[1, 1, N, N]` in the model `dtype`: `0` where a query
/// may attend to a key (`j <= i`), a large negative above the diagonal. Mirrors the vendored
/// `CLIPTextModel._get_mask(N, dtype)` exactly, including its dtype-specific masked value —
/// **`-6e4` for fp16** (≈ f16's finite floor; `-1e9` would overflow to `-inf`) and `-1e9` for f32.
/// The mask must share the q/k/v dtype or `scaled_dot_product_attention` rejects it.
fn causal_mask(n: i32, dtype: Dtype) -> Result<Array> {
    let nu = n as usize;
    let masked = if dtype == Dtype::Float16 {
        -6.0e4
    } else {
        -1.0e9
    };
    let mut m = vec![0f32; nu * nu];
    for i in 0..nu {
        for (j, slot) in m[i * nu..(i + 1) * nu].iter_mut().enumerate() {
            if j > i {
                *slot = masked;
            }
        }
    }
    Ok(Array::from_slice(&m, &[1, 1, n, n]).as_dtype(dtype)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_mask_is_lower_triangular() {
        let m = causal_mask(3, Dtype::Float32).unwrap();
        let v = m.reshape(&[9]).unwrap();
        let s = v.as_slice::<f32>();
        // row 0 attends only to 0; row 1 to {0,1}; row 2 to all.
        assert_eq!(s[0], 0.0);
        assert_eq!(s[1], -1e9);
        assert_eq!(s[2], -1e9);
        assert_eq!(s[3], 0.0);
        assert_eq!(s[4], 0.0);
        assert_eq!(s[5], -1e9);
        assert_eq!(s[6], 0.0);
        assert_eq!(s[8], 0.0);
    }
}
