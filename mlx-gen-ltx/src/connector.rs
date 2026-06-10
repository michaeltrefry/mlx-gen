//! `Embeddings1DConnector` — the LTX-2.3 text-feature connector (S1).
//!
//! Port of `text_encoder.py`'s `Embeddings1DConnector` as configured for the LTX-2.3 models
//! (`connector.safetensors`): an **8-layer** pre-norm transformer over the Gemma feature-extractor
//! output, dim **4096** (32 heads × 128), **gated** attention with q/k RMSNorm, a plain
//! gelu MLP (inner 16384), **128 learnable registers** that replace left-padding, and a
//! connector-specific **1-D SPLIT RoPE** (positions `arange(seq)/4096`, double-precision).
//!
//! Two connectors exist in the checkpoint (`video_embeddings_connector.*`,
//! `audio_embeddings_connector.*`); this core uses the video one. Compute dtype is a parameter:
//! **bf16** to match the reference pipeline end-to-end, **f32** for the isolated bit-exact gate.
//! The fused SDPA is always run in **f32** regardless — the pmetal bf16 maskless-SDPA kernel
//! returns garbage at this shape (see `tests/bf16_sdpa_bug.rs`); the reference's wheel MLX has a
//! correct bf16 SDPA, so f32 matches it to bf16 rounding.

use std::f64::consts::PI;

use mlx_rs::fast::{rms_norm, scaled_dot_product_attention};
use mlx_rs::nn::gelu;
use mlx_rs::ops::{add, concatenate_axis, multiply, sigmoid, sum, tile};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::LtxConfig;
use crate::rope::apply_split_rotary_emb;

const CONNECTOR_EPS: f32 = 1e-6;

/// One connector transformer block (attn1 + gelu FF, both pre-normed with a unit-weight RMSNorm).
struct ConnectorBlock {
    to_q_w: Array,
    to_q_b: Array,
    to_k_w: Array,
    to_k_b: Array,
    to_v_w: Array,
    to_v_b: Array,
    to_out_w: Array,
    to_out_b: Array,
    q_norm_w: Array,
    k_norm_w: Array,
    gate_w: Array,
    gate_b: Array,
    ff_in_w: Array,
    ff_in_b: Array,
    ff_out_w: Array,
    ff_out_b: Array,
}

/// The video text-feature connector.
pub struct Connector {
    blocks: Vec<ConnectorBlock>,
    registers: Array, // (num_registers, dim)
    num_heads: i32,
    head_dim: i32,
    theta: f64,
    max_pos: i32,
    ones: Array,  // unit RMSNorm weight (dim,)
    dtype: Dtype, // compute dtype: bf16 to match the reference pipeline; f32 for the isolated gate
}

impl Connector {
    /// Build the **video** connector from a `Weights` map (e.g. `connector.safetensors`) under
    /// `prefix` (`"video_embeddings_connector."`). Weights are cast to `dtype` (bf16 to match the
    /// reference pipeline end-to-end; f32 for the isolated bit-exact gate).
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &LtxConfig, dtype: Dtype) -> Result<Self> {
        Self::from_weights_dims(
            w,
            prefix,
            cfg.connector_num_layers,
            cfg.connector_num_attention_heads,
            cfg.connector_attention_head_dim,
            cfg.positional_embedding_theta,
            cfg.connector_positional_embedding_max_pos,
            dtype,
        )
    }

    /// Build a connector with **explicit** dims — used for both the video connector (32×128) and the
    /// audio connector (32×64, sc-2684), which share the checkpoint's layer count / theta / max_pos
    /// but differ in `head_dim` (hence `dim`).
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights_dims(
        w: &Weights,
        prefix: &str,
        num_layers: i32,
        num_heads: i32,
        head_dim: i32,
        theta: f64,
        max_pos: i32,
        dtype: Dtype,
    ) -> Result<Self> {
        let n = num_layers as usize;
        let dim = num_heads * head_dim;
        let f32w = |key: &str| -> Result<Array> {
            w.get(key)
                .ok_or_else(|| Error::MissingTensor(key.into()))?
                .as_dtype(dtype)
                .map_err(Error::from)
        };
        let mut blocks = Vec::with_capacity(n);
        for i in 0..n {
            let b = format!("{prefix}transformer_1d_blocks.{i}.");
            blocks.push(ConnectorBlock {
                to_q_w: f32w(&format!("{b}attn1.to_q.weight"))?,
                to_q_b: f32w(&format!("{b}attn1.to_q.bias"))?,
                to_k_w: f32w(&format!("{b}attn1.to_k.weight"))?,
                to_k_b: f32w(&format!("{b}attn1.to_k.bias"))?,
                to_v_w: f32w(&format!("{b}attn1.to_v.weight"))?,
                to_v_b: f32w(&format!("{b}attn1.to_v.bias"))?,
                to_out_w: f32w(&format!("{b}attn1.to_out.0.weight"))?,
                to_out_b: f32w(&format!("{b}attn1.to_out.0.bias"))?,
                q_norm_w: f32w(&format!("{b}attn1.q_norm.weight"))?,
                k_norm_w: f32w(&format!("{b}attn1.k_norm.weight"))?,
                gate_w: f32w(&format!("{b}attn1.to_gate_logits.weight"))?,
                gate_b: f32w(&format!("{b}attn1.to_gate_logits.bias"))?,
                ff_in_w: f32w(&format!("{b}ff.net.0.proj.weight"))?,
                ff_in_b: f32w(&format!("{b}ff.net.0.proj.bias"))?,
                ff_out_w: f32w(&format!("{b}ff.net.2.weight"))?,
                ff_out_b: f32w(&format!("{b}ff.net.2.bias"))?,
            });
        }
        let registers = f32w(&format!("{prefix}learnable_registers"))?;
        Ok(Self {
            blocks,
            registers,
            num_heads,
            head_dim,
            theta,
            max_pos,
            ones: Array::ones::<f32>(&[dim])?.as_dtype(dtype)?,
            dtype,
        })
    }

    /// Connector-specific 1-D SPLIT RoPE (double-precision): positions `arange(seq)`, scaled by
    /// `max_pos`. Returns `(cos, sin)`, each `(1, num_heads, seq, head_dim/2)`.
    fn rope(&self, seq: usize) -> Result<(Array, Array)> {
        let heads = self.num_heads as usize;
        let head_half = (self.head_dim / 2) as usize;
        let dim = heads * (self.head_dim as usize);
        let n_elem = 2usize; // 2 * len([max_pos])
        let num_indices = dim / n_elem; // 2048 (= heads * head_half, no padding)
        let step = if num_indices == 1 {
            0.0
        } else {
            1.0 / (num_indices - 1) as f64
        };
        let indices: Vec<f64> = (0..num_indices)
            .map(|i| self.theta.powf(i as f64 * step) * (PI / 2.0))
            .collect();

        let mut cos = vec![0f32; heads * seq * head_half];
        let mut sin = vec![0f32; heads * seq * head_half];
        for t in 0..seq {
            let scaled = (t as f64 / self.max_pos as f64) * 2.0 - 1.0;
            for h in 0..heads {
                for p in 0..head_half {
                    let ang = scaled * indices[h * head_half + p];
                    let o = (h * seq + t) * head_half + p;
                    cos[o] = ang.cos() as f32;
                    sin[o] = ang.sin() as f32;
                }
            }
        }
        let shape = [1, heads as i32, seq as i32, head_half as i32];
        Ok((
            Array::from_slice(&cos, &shape),
            Array::from_slice(&sin, &shape),
        ))
    }

    /// Replace left-padding with learnable registers (batch=1). Valid tokens (the trailing
    /// `num_valid` of a left-padded sequence) move to the front; registers fill the tail.
    fn replace_with_registers(&self, x: &Array, mask01: &Array) -> Result<Array> {
        let sh = x.shape();
        let (s, dim) = (sh[1], sh[2]);
        let nv = sum(mask01, None)?.item::<i32>();
        let num_reg = self.registers.shape()[0];
        let num_tiles = s / num_reg;
        let reg_full = tile(&self.registers, &[num_tiles, 1])? // (s, dim)
            .reshape(&[1, s, dim])?
            .as_dtype(x.dtype())?;
        if nv >= s {
            return Ok(x.clone());
        }
        let valid_idx: Vec<i32> = (s - nv..s).collect();
        let tail_idx: Vec<i32> = (nv..s).collect();
        let valid = x.take_axis(Array::from_slice(&valid_idx, &[valid_idx.len() as i32]), 1)?;
        let reg_tail =
            reg_full.take_axis(Array::from_slice(&tail_idx, &[tail_idx.len() as i32]), 1)?;
        Ok(concatenate_axis(&[&valid, &reg_tail], 1)?)
    }

    fn attn(&self, blk: &ConnectorBlock, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let (h, d) = (self.num_heads, self.head_dim);
        let q = rms_norm(
            &linear(x, &blk.to_q_w, &blk.to_q_b)?,
            &blk.q_norm_w,
            CONNECTOR_EPS,
        )?;
        let k = rms_norm(
            &linear(x, &blk.to_k_w, &blk.to_k_b)?,
            &blk.k_norm_w,
            CONNECTOR_EPS,
        )?;
        let v = linear(x, &blk.to_v_w, &blk.to_v_b)?;
        let q = q.reshape(&[b, s, h, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.reshape(&[b, s, h, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.reshape(&[b, s, h, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let q = apply_split_rotary_emb(&q, cos, sin)?;
        let k = apply_split_rotary_emb(&k, cos, sin)?;
        let scale = 1.0 / (d as f32).sqrt();
        // SDPA in f32: the pmetal bf16 fused-SDPA kernel returns garbage at this shape (mask=None,
        // 32 heads, head_dim 128) — a sibling of the bf16-GEMM bug, NOT fixed by sc-2714 (which
        // patched matmul.cpp only). The reference's wheel MLX has a correct bf16 SDPA, so an f32
        // SDPA matches it to bf16 rounding. (No-op when the connector already runs f32.) See
        // tests/bf16_sdpa_bug.rs.
        let out = scaled_dot_product_attention(
            &q.as_dtype(Dtype::Float32)?,
            &k.as_dtype(Dtype::Float32)?,
            &v.as_dtype(Dtype::Float32)?,
            scale,
            None,
            None,
        )?
        .as_dtype(self.dtype)?; // (b,h,s,d)
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, -1])?;
        // Gated: out_head *= sigmoid(to_gate_logits(x)).
        let gates = sigmoid(&linear(x, &blk.gate_w, &blk.gate_b)?)?.reshape(&[b, s, h, 1])?;
        let out = multiply(&out.reshape(&[b, s, h, d])?, &gates)?.reshape(&[b, s, -1])?;
        linear(&out, &blk.to_out_w, &blk.to_out_b)
    }

    fn block(&self, blk: &ConnectorBlock, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let n = rms_norm(x, &self.ones, CONNECTOR_EPS)?;
        let x = add(x, &self.attn(blk, &n, cos, sin)?)?;
        let n = rms_norm(&x, &self.ones, CONNECTOR_EPS)?;
        let ff = linear(
            &gelu(&linear(&n, &blk.ff_in_w, &blk.ff_in_b)?)?,
            &blk.ff_out_w,
            &blk.ff_out_b,
        )?;
        Ok(add(&x, &ff)?)
    }

    /// Run the connector. `x` = `(1, seq, dim)` feature-extractor output (f32); `mask01` = `(1, seq)`
    /// 1/0 attention mask (1 = valid; left-padded). Returns video embeddings `(1, seq, dim)`.
    pub fn forward(&self, x: &Array, mask01: &Array) -> Result<Array> {
        let mut h = self.replace_with_registers(&x.as_dtype(self.dtype)?, mask01)?;
        let (cos, sin) = self.rope(h.shape()[1] as usize)?;
        for blk in &self.blocks {
            h = self.block(blk, &h, &cos, &sin)?;
        }
        rms_norm(&h, &self.ones, CONNECTOR_EPS).map_err(Error::from)
    }
}
