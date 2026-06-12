//! SAM3 DETR detector — encoder + decoder + presence token + dot-product scoring, porting the
//! `Sam3DetrEncoder` / `Sam3DetrDecoder` / `Sam3DotProductScoring` path (epic 4910, sc-4921).
//!
//! Consumes the finest FPN vision feature (72²) + the projected text features (SAM3-B) and produces,
//! for 200 object queries: open-vocabulary concept logits (`pred_logits`), refined boxes
//! (`pred_boxes`, xyxy∈[0,1]), and the global `presence_logits`. All standard attention — no
//! deformable attention, no NMS, no Hungarian (set prediction). The decoder's vision cross-attention
//! is biased by a **BoxRPB** relative-position bias (log-scale-encoded box↔grid deltas), and boxes
//! are refined iteratively across the 6 layers. Token layout `[B, seq, C]`.

use std::f32::consts::PI;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::relu;
use mlx_rs::ops::{
    abs, add, concatenate_axis, cos, divide, log, maximum, minimum, multiply, pad, sigmoid, sign,
    sin, stack_axis,
};
use mlx_rs::Array;

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Sam3DetrConfig;

const SCALE_2PI: f32 = 2.0 * PI;
const NUM_POS: i32 = 128; // sine position features per axis (hidden_size / 2)

fn join(prefix: &str, leaf: &str) -> String {
    format!("{prefix}.{leaf}")
}

/// `sin(even)/cos(odd)` interleave of a `[.., 128]` raw angle tensor → `[.., 128]`
/// (`stack(sin(x[0::2]), cos(x[1::2])).flatten`), the SAM3 sine-embedding convention.
fn sincos_interleave(raw: &Array) -> Result<Array> {
    let sh = raw.shape();
    let half = sh[sh.len() - 1] / 2;
    let mut paired: Vec<i32> = sh[..sh.len() - 1].to_vec();
    paired.push(half);
    paired.push(2);
    let r = raw.reshape(&paired)?;
    let last = paired.len() as i32 - 1;
    let even = r.take_axis(Array::from_int(0), last)?;
    let odd = r.take_axis(Array::from_int(1), last)?;
    let stacked = stack_axis(&[&sin(&even)?, &cos(&odd)?], last)?;
    Ok(stacked.reshape(sh)?)
}

/// `dim_t[i] = temperature^(2·(i/2)/NUM_POS)` (host constant).
fn dim_t() -> Vec<f32> {
    (0..NUM_POS)
        .map(|i| 10000f32.powf(2.0 * ((i / 2) as f32) / NUM_POS as f32))
        .collect()
}

/// Additive SDPA over `[b, nh, *, hd]` q/k/v with an optional additive `mask`.
fn attend(q: &Array, k: &Array, v: &Array, scale: f32, mask: Option<&Array>) -> Result<Array> {
    Ok(match mask {
        Some(m) => scaled_dot_product_attention(q, k, v, scale, m, None)?,
        None => scaled_dot_product_attention(q, k, v, scale, None, None)?,
    })
}

/// Generic multi-head attention (`Sam3Attention`): separate q/k/v/o, optional additive mask.
/// Shared by the DETR stack (sc-4921) and the geometry encoder (sc-4923).
pub(crate) struct Attn {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    o_b: Array,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
}

impl Attn {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        Self::from_dims(w, prefix, cfg.num_attention_heads, cfg.head_dim())
    }

    /// Construct from explicit head geometry — the geometry encoder reuses the same `Sam3Attention`
    /// shape but carries its own (numerically identical) config.
    pub(crate) fn from_dims(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
    ) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        Ok(Self {
            q_w: g("q_proj.weight")?,
            q_b: g("q_proj.bias")?,
            k_w: g("k_proj.weight")?,
            k_b: g("k_proj.bias")?,
            v_w: g("v_proj.weight")?,
            v_b: g("v_proj.bias")?,
            o_w: g("o_proj.weight")?,
            o_b: g("o_proj.bias")?,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub(crate) fn forward(
        &self,
        query: &Array,
        key: &Array,
        value: &Array,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let b = query.shape()[0];
        let ql = query.shape()[1];
        let kl = key.shape()[1];
        let (nh, hd) = (self.num_heads, self.head_dim);
        let heads = |t: Array, n: i32| -> Result<Array> {
            Ok(t.reshape(&[b, n, nh, hd])?.transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = heads(linear(query, &self.q_w, &self.q_b)?, ql)?;
        let k = heads(linear(key, &self.k_w, &self.k_b)?, kl)?;
        let v = heads(linear(value, &self.v_w, &self.v_b)?, kl)?;
        let o = attend(&q, &k, &v, self.scale, mask)?;
        let o = o
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, ql, nh * hd])?;
        linear(&o, &self.o_w, &self.o_b)
    }
}

/// `Sam3MLP` (DETR enc/dec FFN): `fc1` → **ReLU** → `fc2` (`hidden_act = "relu"`).
/// Shared with the geometry encoder (sc-4923).
pub(crate) struct Ffn {
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
}

impl Ffn {
    pub(crate) fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        Ok(Self {
            fc1_w: g("fc1.weight")?,
            fc1_b: g("fc1.bias")?,
            fc2_w: g("fc2.weight")?,
            fc2_b: g("fc2.bias")?,
        })
    }
    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let h = relu(&linear(x, &self.fc1_w, &self.fc1_b)?)?;
        linear(&h, &self.fc2_w, &self.fc2_b)
    }
}

/// `Sam3DecoderMLP`: a 2- or 3-layer ReLU MLP (relu between layers, no final activation).
struct DecoderMlp {
    layers: Vec<(Array, Array)>,
}

impl DecoderMlp {
    fn from_weights(w: &Weights, prefix: &str, num_layers: usize) -> Result<Self> {
        let layers = (1..=num_layers)
            .map(|i| -> Result<(Array, Array)> {
                Ok((
                    w.require(&join(prefix, &format!("layer{i}.weight")))?
                        .clone(),
                    w.require(&join(prefix, &format!("layer{i}.bias")))?.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        let n = self.layers.len();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            h = linear(&h, w, b)?;
            if i + 1 < n {
                h = relu(&h)?;
            }
        }
        Ok(h)
    }
}

fn ln(x: &Array, w: &Array, b: &Array, eps: f32) -> Result<Array> {
    Ok(layer_norm(x, Some(w), Some(b), eps)?)
}

/// `clamp(x, lo, hi)` via `minimum(maximum(x, lo), hi)`.
fn clamp(x: &Array, lo: f32, hi: f32) -> Result<Array> {
    Ok(minimum(
        &maximum(x, Array::from_f32(lo))?,
        Array::from_f32(hi),
    )?)
}

/// `inverse_sigmoid(x, eps=1e-3)` = `log(clamp(x,eps,1) / clamp(1-x,eps,1))`.
fn inverse_sigmoid(x: &Array) -> Result<Array> {
    let eps = 1e-3;
    let x = clamp(x, 0.0, 1.0)?;
    let x1 = clamp(&x, eps, 1.0)?;
    let x2 = clamp(&Array::from_f32(1.0).subtract(&x)?, eps, 1.0)?;
    Ok(log(&divide(&x1, &x2)?)?)
}

/// `(cx,cy,w,h) → (x1,y1,x2,y2)` over the last axis.
fn cxcywh_to_xyxy(b: &Array) -> Result<Array> {
    let g = |i: i32| b.take_axis(Array::from_int(i), -1);
    let (cx, cy, w, h) = (g(0)?, g(1)?, g(2)?, g(3)?);
    let half_w = multiply(&w, Array::from_f32(0.5))?;
    let half_h = multiply(&h, Array::from_f32(0.5))?;
    let x1 = cx.subtract(&half_w)?;
    let y1 = cy.subtract(&half_h)?;
    let x2 = add(&cx, &half_w)?;
    let y2 = add(&cy, &half_h)?;
    Ok(stack_axis(&[&x1, &y1, &x2, &y2], -1)?)
}

/// `Sam3DotProductScoring`: open-vocab logit = `scale · ⟨query_proj(q), text_proj(meanpool(text))⟩`.
struct DotScoring {
    text_mlp: DecoderMlp,
    text_mlp_out_w: Array,
    text_mlp_out_b: Array,
    text_proj_w: Array,
    text_proj_b: Array,
    query_proj_w: Array,
    query_proj_b: Array,
    scale: f32,
    clamp: f32,
    eps: f32,
}

impl DotScoring {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        Ok(Self {
            text_mlp: DecoderMlp::from_weights(w, &join(prefix, "text_mlp"), 2)?,
            text_mlp_out_w: g("text_mlp_out_norm.weight")?,
            text_mlp_out_b: g("text_mlp_out_norm.bias")?,
            text_proj_w: g("text_proj.weight")?,
            text_proj_b: g("text_proj.bias")?,
            query_proj_w: g("query_proj.weight")?,
            query_proj_b: g("query_proj.bias")?,
            scale: 1.0 / (cfg.hidden_size as f32).sqrt(),
            clamp: cfg.score_clamp,
            eps: cfg.layer_norm_eps,
        })
    }

    /// `queries`: `[1, Q, D]`; `text`: `[1, L, D]`; `text_mask`: per-token validity.
    /// Returns `pred_logits` `[1, Q]`.
    fn forward(&self, queries: &Array, text: &Array, text_mask: &[i32]) -> Result<Array> {
        // text_mlp residual + out-norm
        let t = self.text_mlp.forward(text)?;
        let t = add(&t, text)?;
        let t = ln(&t, &self.text_mlp_out_w, &self.text_mlp_out_b, self.eps)?;
        // masked mean over valid tokens. NOTE: valid positions need not be contiguous — the PVS
        // path (sc-4923) concatenates valid geometry-prompt tokens *after* the text padding, so we
        // weight by the mask rather than assuming a leading valid run. Equivalent to a leading-run
        // take for the text-only PCS path.
        let l = text_mask.len() as i32;
        let isv: Vec<f32> = text_mask
            .iter()
            .map(|&m| if m == 1 { 1.0 } else { 0.0 })
            .collect();
        let n_valid = isv.iter().sum::<f32>().max(1.0);
        let is_valid = Array::from_slice(&isv, &[1, l, 1]);
        let pooled = divide(
            &multiply(&t, &is_valid)?.sum_axis(1, false)?,
            Array::from_f32(n_valid),
        )?; // [1, D]
        let proj_text = linear(&pooled, &self.text_proj_w, &self.text_proj_b)?; // [1, D]
        let proj_q = linear(queries, &self.query_proj_w, &self.query_proj_b)?; // [1, Q, D]
                                                                               // ⟨q, text⟩ over D → [1, Q]
        let scores = multiply(&proj_q, &proj_text.reshape(&[1, 1, -1])?)?.sum_axis(-1, false)?;
        let scores = multiply(&scores, Array::from_f32(self.scale))?;
        clamp(&scores, -self.clamp, self.clamp)
    }
}

/// One pre-norm DETR encoder layer: vision self-attn + text cross-attn + FFN.
struct EncoderLayer {
    ln1_w: Array,
    ln1_b: Array,
    ln2_w: Array,
    ln2_b: Array,
    ln3_w: Array,
    ln3_b: Array,
    self_attn: Attn,
    cross_attn: Attn,
    ffn: Ffn,
    eps: f32,
}

impl EncoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            ln3_w: g("layer_norm3.weight")?,
            ln3_b: g("layer_norm3.bias")?,
            self_attn: Attn::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            cross_attn: Attn::from_weights(w, &join(prefix, "cross_attn"), cfg)?,
            ffn: Ffn::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    fn forward(
        &self,
        x: &Array,
        vis_pos: &Array,
        text: &Array,
        text_mask: Option<&Array>,
    ) -> Result<Array> {
        // vision self-attention (pos added to q/k, not v)
        let h = ln(x, &self.ln1_w, &self.ln1_b, self.eps)?;
        let hp = add(&h, vis_pos)?;
        let a = self.self_attn.forward(&hp, &hp, &h, None)?;
        let x = add(x, &a)?;
        // text cross-attention
        let h = ln(&x, &self.ln2_w, &self.ln2_b, self.eps)?;
        let a = self.cross_attn.forward(&h, text, text, text_mask)?;
        let x = add(&x, &a)?;
        // FFN
        let h = ln(&x, &self.ln3_w, &self.ln3_b, self.eps)?;
        let a = self.ffn.forward(&h)?;
        Ok(add(&x, &a)?)
    }
}

/// One post-norm DETR decoder layer: query self-attn + text cross-attn + vision cross-attn (BoxRPB)
/// + FFN. `hidden` is `[1, 1+Q, D]` (presence token at index 0); `query_pos` is `[1, Q, D]`.
struct DecoderLayer {
    self_attn: Attn,
    self_ln_w: Array,
    self_ln_b: Array,
    text_attn: Attn,
    text_ln_w: Array,
    text_ln_b: Array,
    vis_attn: Attn,
    vis_ln_w: Array,
    vis_ln_b: Array,
    ffn: Ffn,
    mlp_ln_w: Array,
    mlp_ln_b: Array,
    eps: f32,
}

impl DecoderLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        Ok(Self {
            self_attn: Attn::from_weights(w, &join(prefix, "self_attn"), cfg)?,
            self_ln_w: g("self_attn_layer_norm.weight")?,
            self_ln_b: g("self_attn_layer_norm.bias")?,
            text_attn: Attn::from_weights(w, &join(prefix, "text_cross_attn"), cfg)?,
            text_ln_w: g("text_cross_attn_layer_norm.weight")?,
            text_ln_b: g("text_cross_attn_layer_norm.bias")?,
            vis_attn: Attn::from_weights(w, &join(prefix, "vision_cross_attn"), cfg)?,
            vis_ln_w: g("vision_cross_attn_layer_norm.weight")?,
            vis_ln_b: g("vision_cross_attn_layer_norm.bias")?,
            ffn: Ffn::from_weights(w, &join(prefix, "mlp"))?,
            mlp_ln_w: g("mlp_layer_norm.weight")?,
            mlp_ln_b: g("mlp_layer_norm.bias")?,
            eps: cfg.layer_norm_eps,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Array,
        query_pos_padded: &Array, // [1, 1+Q, D] (presence row = 0)
        text: &Array,
        text_mask: Option<&Array>,
        vision: &Array,
        vis_pos: &Array,
        rpb: &Array, // [1, nh, 1+Q, HW]
    ) -> Result<Array> {
        // self-attention
        let qp = add(hidden, query_pos_padded)?;
        let a = self.self_attn.forward(&qp, &qp, hidden, None)?;
        let x = add(hidden, &a)?;
        let x = ln(&x, &self.self_ln_w, &self.self_ln_b, self.eps)?;
        // text cross-attention
        let qp = add(&x, query_pos_padded)?;
        let a = self.text_attn.forward(&qp, text, text, text_mask)?;
        let x = add(&x, &a)?;
        let x = ln(&x, &self.text_ln_w, &self.text_ln_b, self.eps)?;
        // vision cross-attention with BoxRPB bias
        let qp = add(&x, query_pos_padded)?;
        let kp = add(vision, vis_pos)?;
        let a = self.vis_attn.forward(&qp, &kp, vision, Some(rpb))?;
        let x = add(&x, &a)?;
        let x = ln(&x, &self.vis_ln_w, &self.vis_ln_b, self.eps)?;
        // FFN (no pre-norm; post-norm)
        let a = self.ffn.forward(&x)?;
        let x = add(&x, &a)?;
        ln(&x, &self.mlp_ln_w, &self.mlp_ln_b, self.eps)
    }
}

/// The DETR detector head: encoder + decoder + presence + scoring. Produces concept logits, boxes,
/// and presence from the 72² FPN feature + projected text features.
pub struct Sam3Detector {
    enc_layers: Vec<EncoderLayer>,
    dec_layers: Vec<DecoderLayer>,
    output_ln_w: Array,
    output_ln_b: Array,
    box_head: DecoderMlp,
    query_embed: Array,      // [Q, D]
    reference_points: Array, // [Q, 4]
    presence_token: Array,   // [1, D]
    presence_head: DecoderMlp,
    presence_ln_w: Array,
    presence_ln_b: Array,
    presence_clamp: f32,
    ref_point_head: DecoderMlp,
    box_rpb_x: DecoderMlp,
    box_rpb_y: DecoderMlp,
    scoring: DotScoring,
    cfg: Sam3DetrConfig,
}

/// The detector outputs needed downstream (SAM3-D adds masks).
pub struct DetectorOutput {
    /// `[1, Q]` concept logits (pre-sigmoid).
    pub pred_logits: Array,
    /// `[1, Q, 4]` boxes in xyxy ∈ [0, 1].
    pub pred_boxes: Array,
    /// `[1, 1]` global presence logit.
    pub presence_logits: Array,
    /// `[1, Q, D]` final decoder query hidden states (output-LN'd) — the mask head consumes these.
    pub query_hidden: Array,
    /// `[1, H·W, D]` DETR encoder output (the encoded 72² level) — the mask head consumes this.
    pub encoder_hidden_states: Array,
}

impl Sam3Detector {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3DetrConfig) -> Result<Self> {
        let enc_prefix = join(prefix, "detr_encoder");
        let dec_prefix = join(prefix, "detr_decoder");
        let enc_layers = (0..cfg.num_encoder_layers)
            .map(|i| EncoderLayer::from_weights(w, &join(&enc_prefix, &format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let dec_layers = (0..cfg.num_decoder_layers)
            .map(|i| DecoderLayer::from_weights(w, &join(&dec_prefix, &format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let d = |n: &str| -> Result<Array> { Ok(w.require(&join(&dec_prefix, n))?.clone()) };
        Ok(Self {
            enc_layers,
            dec_layers,
            output_ln_w: d("output_layer_norm.weight")?,
            output_ln_b: d("output_layer_norm.bias")?,
            box_head: DecoderMlp::from_weights(w, &join(&dec_prefix, "box_head"), 3)?,
            query_embed: d("query_embed.weight")?,
            reference_points: d("reference_points.weight")?,
            presence_token: d("presence_token.weight")?,
            presence_head: DecoderMlp::from_weights(w, &join(&dec_prefix, "presence_head"), 3)?,
            presence_ln_w: d("presence_layer_norm.weight")?,
            presence_ln_b: d("presence_layer_norm.bias")?,
            presence_clamp: cfg.presence_clamp,
            ref_point_head: DecoderMlp::from_weights(w, &join(&dec_prefix, "ref_point_head"), 2)?,
            box_rpb_x: DecoderMlp::from_weights(w, &join(&dec_prefix, "box_rpb_embed_x"), 2)?,
            box_rpb_y: DecoderMlp::from_weights(w, &join(&dec_prefix, "box_rpb_embed_y"), 2)?,
            scoring: DotScoring::from_weights(w, &join(prefix, "dot_product_scoring"), cfg)?,
            cfg: cfg.clone(),
        })
    }

    /// `vision_feature`: the 72² FPN feature **NHWC** `[1, H, W, 256]`; `text`: `[1, L, 256]`.
    /// `text_mask`: per-text-token validity (`1`/`0`). Returns concept logits, boxes, presence.
    pub fn forward(
        &self,
        vision_feature: &Array,
        text: &Array,
        text_mask: &[i32],
    ) -> Result<DetectorOutput> {
        let sh = vision_feature.shape();
        let (h, w) = (sh[1], sh[2]);
        let hw = h * w;
        let vision = vision_feature.reshape(&[1, hw, self.cfg.hidden_size])?;
        let vis_pos = sine_position_embedding_flat(h, w, self.cfg.hidden_size)?; // [1, HW, D]
        let text_key_mask = text_key_mask(text_mask);

        // --- encoder ---
        let mut enc = vision;
        for layer in &self.enc_layers {
            enc = layer.forward(&enc, &vis_pos, text, Some(&text_key_mask))?;
        }

        // --- decoder ---
        let q = self.cfg.num_queries;
        let d = self.cfg.hidden_size;
        let query_embeds = self.query_embed.reshape(&[1, q, d])?;
        let presence = self.presence_token.reshape(&[1, 1, d])?;
        let mut hidden = concatenate_axis(&[&presence, &query_embeds], 1)?; // [1, 1+Q, D]
        let mut reference_boxes = sigmoid(&self.reference_points.reshape(&[1, q, 4])?)?;

        let mut last_query_hidden = None;
        let mut last_ref_input = None;
        let mut last_presence = None;

        for layer in &self.dec_layers {
            // conditional query positions from the current reference boxes
            let query_sine = self.encode_boxes(&reference_boxes)?; // [1, Q, 4*128]
            let query_pos = self.ref_point_head.forward(&query_sine)?; // [1, Q, D]
            let query_pos_padded = pad(
                &query_pos,
                &[(0, 0), (1, 0), (0, 0)][..],
                Some(Array::from_f32(0.0)),
                None,
            )?; // presence row = 0
                // BoxRPB bias, padded with a zero row for the presence query
            let rpb = self.box_rpb(&reference_boxes, h, w)?; // [1, nh, Q, HW]
            let rpb = pad(
                &rpb,
                &[(0, 0), (0, 0), (1, 0), (0, 0)][..],
                Some(Array::from_f32(0.0)),
                None,
            )?;

            hidden = layer.forward(
                &hidden,
                &query_pos_padded,
                text,
                Some(&text_key_mask),
                &enc,
                &vis_pos,
                &rpb,
            )?;

            // query hidden (drop presence) → output LN
            let qidx = Array::from_slice(&(1..=q).collect::<Vec<i32>>(), &[q]);
            let query_hidden = hidden.take_axis(&qidx, 1)?; // [1, Q, D]
            let query_hidden = ln(
                &query_hidden,
                &self.output_ln_w,
                &self.output_ln_b,
                self.cfg.layer_norm_eps,
            )?;

            // record this layer's reference-box input + outputs (final layer is what we return)
            last_ref_input = Some(reference_boxes.clone());
            last_query_hidden = Some(query_hidden.clone());

            // iterative box refinement for the next layer
            let delta = self.box_head.forward(&query_hidden)?;
            reference_boxes = sigmoid(&add(&delta, &inverse_sigmoid(&reference_boxes)?)?)?;

            // presence
            let presence_hidden = hidden.take_axis(Array::from_slice(&[0], &[1]), 1)?; // [1,1,D]
            let p = ln(
                &presence_hidden,
                &self.presence_ln_w,
                &self.presence_ln_b,
                self.cfg.layer_norm_eps,
            )?;
            let p = self.presence_head.forward(&p)?.reshape(&[1, 1])?;
            last_presence = Some(clamp(&p, -self.presence_clamp, self.presence_clamp)?);
        }

        let query_hidden = last_query_hidden.unwrap();
        let ref_input = last_ref_input.unwrap();
        let presence_logits = last_presence.unwrap();

        // final boxes: sigmoid(inv_sigmoid(ref_input) + box_head(query_hidden)) → xyxy
        let offsets = self.box_head.forward(&query_hidden)?;
        let boxes_cxcywh = sigmoid(&add(&inverse_sigmoid(&ref_input)?, &offsets)?)?;
        let pred_boxes = cxcywh_to_xyxy(&boxes_cxcywh)?;
        let pred_logits = self.scoring.forward(&query_hidden, text, text_mask)?;

        Ok(DetectorOutput {
            pred_logits,
            pred_boxes,
            presence_logits,
            query_hidden,
            encoder_hidden_states: enc,
        })
    }

    /// `encode_boxes` (sine box embedding): `[1, Q, 4]` cxcywh → `[1, Q, 4·128]` (pos_y, x, w, h).
    fn encode_boxes(&self, boxes: &Array) -> Result<Array> {
        let dim_t = Array::from_slice(&dim_t(), &[NUM_POS]);
        let q = boxes.shape()[1];
        let comp = |idx: i32| -> Result<Array> {
            let e = multiply(
                &boxes.take_axis(Array::from_int(idx), -1)?,
                Array::from_f32(SCALE_2PI),
            )?; // [1,Q]
            let raw = divide(&e.reshape(&[1, q, 1])?, &dim_t.reshape(&[1, 1, NUM_POS])?)?; // [1,Q,128]
            sincos_interleave(&raw)
        };
        // reference order: cat(pos_y, pos_x, pos_w, pos_h) → indices (1,0,2,3)
        let pos_y = comp(1)?;
        let pos_x = comp(0)?;
        let pos_w = comp(2)?;
        let pos_h = comp(3)?;
        Ok(concatenate_axis(&[&pos_y, &pos_x, &pos_w, &pos_h], -1)?)
    }

    /// BoxRPB relative-position bias `[1, nh, Q, H·W]` (log-scale-encoded box↔grid deltas).
    fn box_rpb(&self, reference_boxes: &Array, h: i32, w: i32) -> Result<Array> {
        let q = reference_boxes.shape()[1];
        let nh = self.cfg.num_attention_heads;
        let boxes_xyxy = cxcywh_to_xyxy(reference_boxes)?; // [1,Q,4]
        let coords_h = Array::from_slice(
            &(0..h).map(|i| i as f32 / h as f32).collect::<Vec<f32>>(),
            &[h],
        );
        let coords_w = Array::from_slice(
            &(0..w).map(|i| i as f32 / w as f32).collect::<Vec<f32>>(),
            &[w],
        );
        // y deltas from box (y1,y2) = indices [1,3]; x deltas from (x1,x2) = [0,2]
        let by = boxes_xyxy.take_axis(Array::from_slice(&[1, 3], &[2]), 2)?; // [1,Q,2]
        let bx = boxes_xyxy.take_axis(Array::from_slice(&[0, 2], &[2]), 2)?; // [1,Q,2]
        let dy = coords_h
            .reshape(&[1, 1, h, 1])?
            .subtract(&by.reshape(&[1, q, 1, 2])?)?; // [1,Q,H,2]
        let dx = coords_w
            .reshape(&[1, 1, w, 1])?
            .subtract(&bx.reshape(&[1, q, 1, 2])?)?; // [1,Q,W,2]
        let dy = self.box_rpb_y.forward(&log_scale(&dy)?)?; // [1,Q,H,nh]
        let dx = self.box_rpb_x.forward(&log_scale(&dx)?)?; // [1,Q,W,nh]
                                                            // rpb[b,q,h,w,head] = dy[b,q,h,head] + dx[b,q,w,head]
        let rpb = add(
            &dy.reshape(&[1, q, h, 1, nh])?,
            &dx.reshape(&[1, q, 1, w, nh])?,
        )?; // [1,Q,H,W,nh]
        let rpb = rpb
            .reshape(&[1, q, h * w, nh])?
            .transpose_axes(&[0, 3, 1, 2])?; // [1,nh,Q,HW]
        Ok(rpb)
    }
}

/// Log-scale delta encoding: `d8 = d·8; sign(d8)·log2(|d8|+1)/log2(8)`.
fn log_scale(d: &Array) -> Result<Array> {
    let d8 = multiply(d, Array::from_f32(8.0))?;
    let inv_log2_8 = 1.0 / 3.0; // log2(8) = 3
    let mag = log(&add(&abs(&d8)?, Array::from_f32(1.0))?)?; // ln(|d8|+1)
    let log2 = multiply(&mag, Array::from_f32(std::f32::consts::LOG2_E * inv_log2_8))?; // ln·log2(e)/3 = log2/3
    multiply(&sign(&d8)?, &log2).map_err(Into::into)
}

/// Build a key-padding additive mask `[1, 1, 1, L]` (0 valid, −1e9 padded), broadcast over heads/queries.
fn text_key_mask(text_mask: &[i32]) -> Array {
    let row: Vec<f32> = text_mask
        .iter()
        .map(|&m| if m == 1 { 0.0 } else { -1e9 })
        .collect();
    Array::from_slice(&row, &[1, 1, 1, row.len() as i32])
}

/// Sine position embedding (normalize=True), flattened to `[1, H·W, D]` (host-computed constant for
/// a fixed grid). Mirrors `Sam3SinePositionEmbedding.build_sine_position_embedding` for the neck.
pub(crate) fn sine_position_embedding_flat(h: i32, w: i32, d: i32) -> Result<Array> {
    let np = (d / 2) as usize; // 128
    let dt = dim_t();
    let eps = 1e-6f32;
    let mut out = vec![0f32; (h * w * d) as usize];
    // sin(even)/cos(odd) interleave of v/dim_t into a 128-slice
    let fill = |buf: &mut [f32], v: f32| {
        for j in 0..np / 2 {
            buf[2 * j] = (v / dt[2 * j]).sin();
            buf[2 * j + 1] = (v / dt[2 * j + 1]).cos();
        }
    };
    for hi in 0..h as usize {
        let yv = (hi as f32 + 1.0) / (h as f32 + eps) * SCALE_2PI;
        for wi in 0..w as usize {
            let xv = (wi as f32 + 1.0) / (w as f32 + eps) * SCALE_2PI;
            let base = (hi * w as usize + wi) * d as usize;
            fill(&mut out[base..base + np], yv); // pos_y first
            fill(&mut out[base + np..base + 2 * np], xv); // then pos_x
        }
    }
    Ok(Array::from_slice(&out, &[1, h * w, d]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::array_eq;

    #[test]
    fn cxcywh_to_xyxy_roundtrips_a_unit_box() {
        // center (0.5,0.5) size (0.4,0.2) → (0.3,0.4,0.7,0.6)
        let b = Array::from_slice(&[0.5f32, 0.5, 0.4, 0.2], &[1, 1, 4]);
        let xyxy = cxcywh_to_xyxy(&b).unwrap().reshape(&[4]).unwrap();
        let v = xyxy.as_slice::<f32>();
        for (got, want) in v.iter().zip([0.3f32, 0.4, 0.7, 0.6]) {
            assert!((got - want).abs() < 1e-6, "got {got} want {want}");
        }
    }

    #[test]
    fn inverse_sigmoid_inverts_sigmoid() {
        let x = Array::from_slice(&[-2.0f32, -0.5, 0.0, 1.3, 3.0], &[5]);
        let round = inverse_sigmoid(&sigmoid(&x).unwrap()).unwrap();
        let d = max(abs(subtract(&round, &x).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert!(d < 1e-3, "inverse_sigmoid∘sigmoid drift {d}");
    }

    #[test]
    fn sine_position_embedding_has_expected_shape() {
        let p = sine_position_embedding_flat(72, 72, 256).unwrap();
        assert_eq!(p.shape(), &[1, 5184, 256]);
    }

    #[test]
    fn sincos_interleave_places_sin_even_cos_odd() {
        // raw all-zero → sin=0 on even lanes, cos=1 on odd lanes.
        let raw = Array::from_slice(&[0f32; 8], &[1, 8]);
        let out = sincos_interleave(&raw).unwrap().reshape(&[8]).unwrap();
        let want = Array::from_slice(&[0f32, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], &[8]);
        assert!(array_eq(&out, &want, None).unwrap().item::<bool>());
    }

    use mlx_rs::ops::{abs, max, subtract};
}
