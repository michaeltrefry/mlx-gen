//! SAM3 tracker — single-frame path (epic 4910, sc-4924, Phase F).
//!
//! The SAM3 tracker is **SAM2.1** architecture (mask decoder 256/2L/8H + dynamic-multimask via
//! stability; prompt encoder; memory bank for video). It is fed by the **shared PE vision encoder**
//! ([`crate::vision`]) + its own `tracker_neck` FPN (NOT a separate Hiera trunk). The checkpoint
//! stores it under `tracker_model.*` / `tracker_neck.*` in **`transformers`-5 module naming**, which
//! diverges from the original `facebookresearch/sam2` `.pt` naming `mlx-gen-sam2` was ported against
//! (`o_proj` vs `out_proj`, `upscale_conv1/2` vs `output_upscaling_*`, `mlp.proj_in/out` vs
//! `mlp.layers.*`, `feature_projection` is net-new, …). So this is a direct port mirroring the public
//! Apache-2.0 `transformers` reference (`modeling_sam3_tracker_video.py`), reusing this crate's own
//! NHWC primitives, rather than a key-translation over `mlx-gen-sam2`.
//!
//! This module is the **single-frame box-prompt** path (`tracker_neck` → prompt encoder → two-way
//! mask decoder), the SAM2-`Sam2Segmenter`-equivalent that the memory/video layer (F2) builds on.
//! Layout is NHWC end-to-end (matching [`crate::vision`] / [`crate::mask`]).

use std::f32::consts::PI;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::relu;
use mlx_rs::ops::{
    add, broadcast_to, concatenate_axis, conv_transpose2d, multiply, sigmoid, stack_axis,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{conv2d, gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use std::rc::Rc;

use crate::config::Sam3VisionConfig;
use crate::util::{conv_transpose_w, conv_w_ohwi, join};
use crate::vision::{quantize_backbone_rc, Backbone, FpnLayer};

/// Take a single index `i` along `axis`, dropping that axis.
fn take1(x: &Array, i: i32, axis: i32) -> Result<Array> {
    Ok(x.take_axis(Array::from_int(i), axis)?)
}

/// Slice `[start, end)` along `axis` (keeps the axis).
fn slice_axis(x: &Array, axis: i32, start: i32, end: i32) -> Result<Array> {
    let idx: Vec<i32> = (start..end).collect();
    Ok(x.take_axis(Array::from_slice(&idx, &[idx.len() as i32]), axis)?)
}

// --- fixed facebook/sam3 tracker hyperparameters (Sam3TrackerVideoMaskDecoderConfig) -------------
const HIDDEN: i32 = 256;
const NUM_HEADS: i32 = 8;
const NUM_MASK_TOKENS: i32 = 4; // num_multimask_outputs (3) + 1
const ATTN_DOWNSAMPLE: i32 = 2;
const LN_EPS: f32 = 1e-6;
const INPUT_SIZE: f32 = 1008.0; // image_size
const STABILITY_DELTA: f32 = 0.05; // dynamic_multimask_stability_delta
const STABILITY_THRESH: f32 = 0.98; // dynamic_multimask_stability_thresh
const NO_OBJ_SCORE: f32 = -1024.0; // logit for "object absent" frames
const MASK_INPUT_SIZE: i32 = 288; // prompt encoder mask_input_size (4·1008/14)

fn weight_bias(w: &Weights, prefix: &str) -> Result<(Array, Array)> {
    Ok((
        w.require(&join(prefix, "weight"))?.clone(),
        w.require(&join(prefix, "bias"))?.clone(),
    ))
}

fn ln(x: &Array, p: &(Array, Array)) -> Result<Array> {
    Ok(layer_norm(x, Some(&p.0), Some(&p.1), LN_EPS)?)
}

// --- FeedForward (Sam3TrackerVideoFeedForward) ---------------------------------------------------

/// `proj_in → act → (layers.i → act)* → proj_out → [sigmoid]`. `act` is ReLU throughout the tracker.
struct FeedForward {
    proj_in: AdaptableLinear,
    layers: Vec<AdaptableLinear>,
    proj_out: AdaptableLinear,
    sigmoid_output: bool,
}

impl FeedForward {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: i32,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let layers = (0..num_layers - 2)
            .map(|i| crate::load_linear(w, &join(prefix, &format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            proj_in: crate::load_linear(w, &join(prefix, "proj_in"))?,
            layers,
            proj_out: crate::load_linear(w, &join(prefix, "proj_out"))?,
            sigmoid_output,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        crate::quantize_linear(&mut self.proj_in, bits)?;
        for l in &mut self.layers {
            crate::quantize_linear(l, bits)?;
        }
        crate::quantize_linear(&mut self.proj_out, bits)?;
        Ok(())
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = relu(&self.proj_in.forward(x)?)?;
        for l in &self.layers {
            h = relu(&l.forward(&h)?)?;
        }
        h = self.proj_out.forward(&h)?;
        if self.sigmoid_output {
            h = sigmoid(&h)?;
        }
        Ok(h)
    }
}

// --- Attention (Sam3TrackerVideoAttention, with q/k/v down-projection) ---------------------------

/// MHA on `[b, n, hidden]` tokens; q/k/v project to `internal = hidden / downsample`, split into
/// `NUM_HEADS`, SDPA, then `o_proj` back to `hidden`.
struct Attention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    num_heads: i32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, downsample: i32) -> Result<Self> {
        let _ = downsample; // head split is derived from the loaded projection width at forward time
        let l = |n: &str| crate::load_linear(w, &join(prefix, n));
        Ok(Self {
            q: l("q_proj")?,
            k: l("k_proj")?,
            v: l("v_proj")?,
            o: l("o_proj")?,
            num_heads: NUM_HEADS,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        crate::quantize_linear(&mut self.q, bits)?;
        crate::quantize_linear(&mut self.k, bits)?;
        crate::quantize_linear(&mut self.v, bits)?;
        crate::quantize_linear(&mut self.o, bits)?;
        Ok(())
    }

    fn forward(&self, q: &Array, k: &Array, v: &Array) -> Result<Array> {
        let sep = |x: &Array| -> Result<Array> {
            let sh = x.shape();
            let (b, n, c) = (sh[0], sh[1], sh[2]);
            Ok(x.reshape(&[b, n, self.num_heads, c / self.num_heads])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = sep(&self.q.forward(q)?)?;
        let k = sep(&self.k.forward(k)?)?;
        let v = sep(&self.v.forward(v)?)?;
        let scale = 1.0 / (q.shape()[3] as f32).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let sh = out.shape();
        let (b, h, n, c) = (sh[0], sh[1], sh[2], sh[3]);
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, h * c])?;
        self.o.forward(&out)
    }
}

// --- Two-way attention block + transformer (Sam3TrackerVideoTwoWayTransformer) -------------------

struct TwoWayBlock {
    self_attn: Attention,
    norm1: (Array, Array),
    cross_t2i: Attention,
    norm2: (Array, Array),
    mlp: FeedForward,
    norm3: (Array, Array),
    cross_i2t: Attention,
    norm4: (Array, Array),
    skip_first_layer_pe: bool,
}

impl TwoWayBlock {
    fn from_weights(w: &Weights, prefix: &str, skip_first_layer_pe: bool) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::from_weights(w, &join(prefix, "self_attn"), 1)?,
            norm1: weight_bias(w, &join(prefix, "layer_norm1"))?,
            cross_t2i: Attention::from_weights(
                w,
                &join(prefix, "cross_attn_token_to_image"),
                ATTN_DOWNSAMPLE,
            )?,
            norm2: weight_bias(w, &join(prefix, "layer_norm2"))?,
            mlp: FeedForward::from_weights(w, &join(prefix, "mlp"), 2, false)?,
            norm3: weight_bias(w, &join(prefix, "layer_norm3"))?,
            cross_i2t: Attention::from_weights(
                w,
                &join(prefix, "cross_attn_image_to_token"),
                ATTN_DOWNSAMPLE,
            )?,
            norm4: weight_bias(w, &join(prefix, "layer_norm4"))?,
            skip_first_layer_pe,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.self_attn.quantize(bits)?;
        self.cross_t2i.quantize(bits)?;
        self.mlp.quantize(bits)?;
        self.cross_i2t.quantize(bits)?;
        Ok(())
    }

    /// `queries`/`keys`: `[b, nq, D]` / `[b, nk, D]`; `query_pe`/`key_pe`: same shapes.
    fn forward(
        &self,
        queries: &Array,
        keys: &Array,
        query_pe: &Array,
        key_pe: &Array,
    ) -> Result<(Array, Array)> {
        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = add(queries, query_pe)?;
            add(queries, &self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = ln(&queries, &self.norm1)?;

        let q = add(&queries, query_pe)?;
        let k = add(keys, key_pe)?;
        queries = ln(
            &add(&queries, &self.cross_t2i.forward(&q, &k, keys)?)?,
            &self.norm2,
        )?;
        queries = ln(&add(&queries, &self.mlp.forward(&queries)?)?, &self.norm3)?;

        let q = add(&queries, query_pe)?;
        let k = add(keys, key_pe)?;
        let keys = ln(
            &add(keys, &self.cross_i2t.forward(&k, &q, &queries)?)?,
            &self.norm4,
        )?;
        Ok((queries, keys))
    }
}

struct TwoWayTransformer {
    layers: Vec<TwoWayBlock>,
    final_attn: Attention,
    norm_final: (Array, Array),
}

impl TwoWayTransformer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            layers: vec![
                TwoWayBlock::from_weights(w, &join(prefix, "layers.0"), true)?,
                TwoWayBlock::from_weights(w, &join(prefix, "layers.1"), false)?,
            ],
            final_attn: Attention::from_weights(
                w,
                &join(prefix, "final_attn_token_to_image"),
                ATTN_DOWNSAMPLE,
            )?,
            norm_final: weight_bias(w, &join(prefix, "layer_norm_final_attn"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        self.final_attn.quantize(bits)?;
        Ok(())
    }

    /// `image_embedding`/`image_pe`: token-flattened `[b, hw, D]`; `point_embedding`: `[b, n, D]`.
    fn forward(
        &self,
        image_embedding: &Array,
        image_pe: &Array,
        point_embedding: &Array,
    ) -> Result<(Array, Array)> {
        let mut queries = point_embedding.clone();
        let mut keys = image_embedding.clone();
        for layer in &self.layers {
            let (q, k) = layer.forward(&queries, &keys, point_embedding, image_pe)?;
            queries = q;
            keys = k;
        }
        let q = add(&queries, point_embedding)?;
        let k = add(&keys, image_pe)?;
        queries = ln(
            &add(&queries, &self.final_attn.forward(&q, &k, &keys)?)?,
            &self.norm_final,
        )?;
        Ok((queries, keys))
    }
}

// --- Prompt encoder (Sam3TrackerVideoPromptEncoder, box path) ------------------------------------

/// `PositionEmbeddingRandom` (`shared_embedding`): `cat[sin, cos](2π · (2·coord−1) @ gaussian)`.
/// `gaussian` is `[2, HIDDEN/2]`. Coords are normalized to `[0,1]`.
struct PositionEmbeddingRandom {
    gaussian: Array, // [2, 128]
}

impl PositionEmbeddingRandom {
    /// `coords_norm`: `[..., 2]` in `[0,1]`. Returns `[..., HIDDEN]`.
    fn forward(&self, coords_norm: &Array) -> Result<Array> {
        let c = add(
            &coords_norm.multiply(Array::from_f32(2.0))?,
            Array::from_f32(-1.0),
        )?;
        // [..., 2] @ [2, 128] → [..., 128]
        let proj = c.matmul(&self.gaussian)?;
        let proj = proj.multiply(Array::from_f32(2.0 * PI))?;
        Ok(concatenate_axis(&[&proj.sin()?, &proj.cos()?], -1)?)
    }

    /// Dense positional grid for a `g×g` feature map → NHWC `[1, g, g, HIDDEN]` (each cell at its
    /// pixel center `(i+0.5)/g`).
    fn dense_pe(&self, g: i32) -> Result<Array> {
        let n = (g * g) as usize;
        let mut coords = vec![0f32; n * 2];
        for y in 0..g {
            for x in 0..g {
                let idx = (y * g + x) as usize * 2;
                coords[idx] = (x as f32 + 0.5) / g as f32;
                coords[idx + 1] = (y as f32 + 0.5) / g as f32;
            }
        }
        let coords = Array::from_slice(&coords, &[1, g, g, 2]);
        self.forward(&coords)
    }
}

/// SAM3 tracker prompt encoder — box path only (single-frame PVS). `point_embed[2]`/`[3]` are the
/// box-corner embeddings; the padding point uses `not_a_point_embed`. Dense embedding for a box-only
/// prompt is the broadcast `no_mask_embed`.
struct PromptEncoder {
    pe: PositionEmbeddingRandom,
    point_embed: Array,   // [4, 256]
    not_a_point: Array,   // [1, 256]
    no_mask_embed: Array, // [1, 256]
    /// `mask_embed` dense-mask path (`Sam3TrackerVideoMaskEmbedding`): conv1 1→4 k2s2, LN, gelu,
    /// conv2 4→16 k2s2, LN, gelu, conv3 16→256 k1 → dense `[1, 72, 72, 256]`. Convs are OHWI; the
    /// channels-first LayerNorms are a plain LN over the (last, NHWC) channel axis.
    mask_conv1: (Array, Array),
    mask_ln1: (Array, Array),
    mask_conv2: (Array, Array),
    mask_ln2: (Array, Array),
    mask_conv3: (Array, Array),
}

impl PromptEncoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let me = join(prefix, "mask_embed");
        let (c1, c1b) = weight_bias(w, &join(&me, "conv1"))?;
        let (c2, c2b) = weight_bias(w, &join(&me, "conv2"))?;
        let (c3, c3b) = weight_bias(w, &join(&me, "conv3"))?;
        Ok(Self {
            pe: PositionEmbeddingRandom {
                gaussian: w
                    .require(&join(prefix, "shared_embedding.positional_embedding"))?
                    .clone(),
            },
            point_embed: w.require(&join(prefix, "point_embed.weight"))?.clone(),
            not_a_point: w
                .require(&join(prefix, "not_a_point_embed.weight"))?
                .clone(),
            no_mask_embed: w.require(&join(prefix, "no_mask_embed.weight"))?.clone(),
            mask_conv1: (conv_w_ohwi(&c1)?, c1b),
            mask_ln1: weight_bias(w, &join(&me, "layer_norm1"))?,
            mask_conv2: (conv_w_ohwi(&c2)?, c2b),
            mask_ln2: weight_bias(w, &join(&me, "layer_norm2"))?,
            mask_conv3: (conv_w_ohwi(&c3)?, c3b),
        })
    }

    /// `mask_embed` forward + empty-point sparse, for a mask-conditioned (detection-seeded) frame.
    /// `mask_288`: NHWC `[1, 288, 288, 1]` (the prompt-encoder `mask_input_size`). Returns
    /// `(sparse [1, 1, 2, 256] (2× not_a_point), dense [1, 72, 72, 256])`.
    fn encode_mask_prompt(&self, mask_288: &Array) -> Result<(Array, Array)> {
        let h = conv2d(mask_288, &self.mask_conv1.0, Some(&self.mask_conv1.1), 2, 0)?;
        let h = gelu_exact(&ln(&h, &self.mask_ln1)?)?;
        let h = conv2d(&h, &self.mask_conv2.0, Some(&self.mask_conv2.1), 2, 0)?;
        let h = gelu_exact(&ln(&h, &self.mask_ln2)?)?;
        let dense = conv2d(&h, &self.mask_conv3.0, Some(&self.mask_conv3.1), 1, 0)?;
        let nap = self.not_a_point.reshape(&[1, 1, 1, HIDDEN])?;
        let sparse = concatenate_axis(&[&nap, &nap], 2)?;
        Ok((sparse, dense))
    }

    /// `box_xyxy` in **1008-input** pixel space → `(sparse [1, 1, 3, 256], dense [1, g, g, 256])`.
    /// Mirrors `_embed_boxes`: corners +0.5, normalize by the input size, pos-encode, add the
    /// corner-2/3 point embeddings, set the pad point to `not_a_point`.
    fn encode_box(&self, box_xyxy: [f32; 4], g: i32) -> Result<(Array, Array)> {
        // 3 corner coords (x0,y0),(x1,y1),(pad 0,0), normalized; pad stays 0.
        let norm = |v: f32| (v + 0.5) / INPUT_SIZE;
        let coords = [
            norm(box_xyxy[0]),
            norm(box_xyxy[1]),
            norm(box_xyxy[2]),
            norm(box_xyxy[3]),
            0.0,
            0.0,
        ];
        let coords = Array::from_slice(&coords, &[1, 1, 3, 2]);
        let emb = self.pe.forward(&coords)?; // [1,1,3,256]
                                             // corner 0 += point_embed[2], corner 1 += point_embed[3], corner 2 = not_a_point.
        let pe2 = take1(&self.point_embed, 2, 0)?.reshape(&[1, 1, 1, HIDDEN])?;
        let pe3 = take1(&self.point_embed, 3, 0)?.reshape(&[1, 1, 1, HIDDEN])?;
        let rows: Vec<Array> = vec![
            add(&take1(&emb, 0, 2)?.reshape(&[1, 1, 1, HIDDEN])?, &pe2)?,
            add(&take1(&emb, 1, 2)?.reshape(&[1, 1, 1, HIDDEN])?, &pe3)?,
            self.not_a_point.reshape(&[1, 1, 1, HIDDEN])?,
        ];
        let sparse = concatenate_axis(&rows.iter().collect::<Vec<_>>(), 2)?; // [1,1,3,256]

        let dense = broadcast_to(
            &self.no_mask_embed.reshape(&[1, 1, 1, HIDDEN])?,
            &[1, g, g, HIDDEN],
        )?;
        Ok((sparse, dense))
    }

    /// Empty-prompt encoding for a no-prompt (memory-conditioned) tracking frame: `_single_frame_forward`
    /// pads a single empty point with label −1 then `_embed_points(pad=True)` pads one more, so both
    /// resulting tokens collapse to `not_a_point_embed` (label −1 overwrites the point PE). Dense is the
    /// broadcast `no_mask_embed`. Returns `(sparse [1, 1, 2, 256], dense [1, g, g, 256])`.
    fn encode_empty_point(&self, g: i32) -> Result<(Array, Array)> {
        let nap = self.not_a_point.reshape(&[1, 1, 1, HIDDEN])?;
        let sparse = concatenate_axis(&[&nap, &nap], 2)?; // [1,1,2,256]
        let dense = broadcast_to(
            &self.no_mask_embed.reshape(&[1, 1, 1, HIDDEN])?,
            &[1, g, g, HIDDEN],
        )?;
        Ok((sparse, dense))
    }
}

/// `_dynamic_multimask_via_stability` (~1550): on a single-mask request, keep mask token 0 if its
/// stability score — the IoU between the `±delta`-thresholded mask areas — is `≥ thresh`; otherwise
/// fall back to the best-predicted-IoU multimask candidate (tokens 1..). `masks`:
/// `[1, NUM_MASK_TOKENS, mg, mg]`; `ious`: `[1, NUM_MASK_TOKENS]`. Returns `(mask [1,1,mg,mg],
/// iou [1,1])`. B=P=1, so the reference's `torch.where` is a scalar choice (evaluated host-side).
fn dynamic_multimask_via_stability(masks: &Array, ious: &Array) -> Result<(Array, Array)> {
    let iou_v = ious
        .reshape(&[-1])?
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec();
    // best multimask candidate (tokens 1..) by predicted IoU.
    let best = 1 + argmax(&iou_v[1..]) as i32;
    // stability score of the single mask (token 0).
    let single = slice_axis(masks, 1, 0, 1)?;
    let sv = single
        .reshape(&[-1])?
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec();
    let area_i = sv.iter().filter(|&&x| x > STABILITY_DELTA).count() as f32;
    let area_u = sv.iter().filter(|&&x| x > -STABILITY_DELTA).count() as f32;
    let stability = if area_u > 0.0 { area_i / area_u } else { 1.0 };
    if stability >= STABILITY_THRESH {
        Ok((single, slice_axis(ious, 1, 0, 1)?))
    } else {
        Ok((
            slice_axis(masks, 1, best, best + 1)?,
            slice_axis(ious, 1, best, best + 1)?,
        ))
    }
}

// --- Mask decoder (Sam3TrackerVideoMaskDecoder) --------------------------------------------------

struct MaskDecoder {
    transformer: TwoWayTransformer,
    iou_token: Array,       // [1, 256]
    mask_tokens: Array,     // [4, 256]
    obj_score_token: Array, // [1, 256]
    upscale_conv1: (Array, Array),
    upscale_layer_norm: (Array, Array),
    upscale_conv2: (Array, Array),
    hypernet: Vec<FeedForward>,
    iou_head: FeedForward,
    obj_score_head: FeedForward,
}

impl MaskDecoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let hypernet = (0..NUM_MASK_TOKENS)
            .map(|i| {
                FeedForward::from_weights(
                    w,
                    &join(prefix, &format!("output_hypernetworks_mlps.{i}")),
                    3,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let (uc1w, uc1b) = weight_bias(w, &join(prefix, "upscale_conv1"))?;
        let (uc2w, uc2b) = weight_bias(w, &join(prefix, "upscale_conv2"))?;
        Ok(Self {
            transformer: TwoWayTransformer::from_weights(w, &join(prefix, "transformer"))?,
            iou_token: w.require(&join(prefix, "iou_token.weight"))?.clone(),
            mask_tokens: w.require(&join(prefix, "mask_tokens.weight"))?.clone(),
            obj_score_token: w.require(&join(prefix, "obj_score_token.weight"))?.clone(),
            upscale_conv1: (conv_transpose_w(&uc1w)?, uc1b),
            upscale_layer_norm: weight_bias(w, &join(prefix, "upscale_layer_norm"))?,
            upscale_conv2: (conv_transpose_w(&uc2w)?, uc2b),
            hypernet,
            iou_head: FeedForward::from_weights(w, &join(prefix, "iou_prediction_head"), 3, true)?,
            obj_score_head: FeedForward::from_weights(
                w,
                &join(prefix, "pred_obj_score_head"),
                3,
                false,
            )?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.transformer.quantize(bits)?;
        for h in &mut self.hypernet {
            h.quantize(bits)?;
        }
        self.iou_head.quantize(bits)?;
        self.obj_score_head.quantize(bits)?;
        Ok(())
    }

    /// `image_embedding`: NHWC `[1, g, g, 256]`; `image_pe`: NHWC `[1, g, g, 256]`; `sparse`:
    /// `[1, 1, n, 256]`; `dense`: NHWC `[1, g, g, 256]`; `high_res`: `[feat_s0 (NHWC, 4g², 32),
    /// feat_s1 (NHWC, 2g², 64)]`. Returns `(masks [1, k, mg, mg], ious [1, k], obj_score [1, 1],
    /// mask_tokens_out [1, 4, 256])` with `multimask_output` selecting the 3 multimask candidates
    /// (slice 1..). The full (unsliced) `mask_tokens_out` is returned so the caller can extract the
    /// object-pointer source token (`sam_output_token`): for a multimask request that is the
    /// best-IoU candidate (`1 + argmax`); for the single/dynamic request it is always token 0.
    fn forward(
        &self,
        image_embedding: &Array,
        image_pe: &Array,
        sparse: &Array,
        dense: &Array,
        high_res: &[Array; 2],
        multimask_output: bool,
    ) -> Result<(Array, Array, Array, Array)> {
        let g = image_embedding.shape()[1];
        // output tokens: [obj_score, iou, mask_tokens(4)] → [1, 6, 256]
        let out_tokens = concatenate_axis(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?
        .reshape(&[1, 1, 2 + NUM_MASK_TOKENS, HIDDEN])?;
        let sparse2 = sparse.reshape(&[1, 1, sparse.shape()[2], HIDDEN])?;
        let tokens = concatenate_axis(&[&out_tokens, &sparse2], 2)?.reshape(&[1, -1, HIDDEN])?; // [1, n_tok, 256]

        // image + dense, flattened to tokens [1, g*g, 256].
        let img = add(image_embedding, dense)?.reshape(&[1, g * g, HIDDEN])?;
        let img_pe = image_pe.reshape(&[1, g * g, HIDDEN])?;

        let (hs, src) = self.transformer.forward(&img, &img_pe, &tokens)?;
        let iou_token_out = take1(&hs, 1, 1)?.reshape(&[1, HIDDEN])?;
        let mask_tokens_out = slice_axis(&hs, 1, 2, 2 + NUM_MASK_TOKENS)?; // [1, 4, 256]
        let obj_score = self
            .obj_score_head
            .forward(&take1(&hs, 0, 1)?.reshape(&[1, HIDDEN])?)?; // [1, 1]

        // upscale: NHWC src [1, g, g, 256] → +feat_s1 (2g²) → +feat_s0 (4g²).
        let src = src.reshape(&[1, g, g, HIDDEN])?;
        let mut up = conv_transpose2d(&src, &self.upscale_conv1.0, (2, 2), None, None, None, None)?;
        up = add(&up, &self.upscale_conv1.1)?;
        up = add(&up, &high_res[1])?;
        up = gelu_exact(&ln(&up, &self.upscale_layer_norm)?)?;
        let mut up2 = conv_transpose2d(&up, &self.upscale_conv2.0, (2, 2), None, None, None, None)?;
        up2 = add(&up2, &self.upscale_conv2.1)?;
        up2 = gelu_exact(&add(&up2, &high_res[0])?)?; // [1, 4g, 4g, 32]
        let (mg, ch) = (up2.shape()[1], up2.shape()[3]);

        // hypernetwork: per mask token MLP → [1, 4, 32]; mask = hyper @ upscaled.
        let hyper: Vec<Array> = (0..NUM_MASK_TOKENS as usize)
            .map(|i| {
                self.hypernet[i]
                    .forward(&take1(&mask_tokens_out, i as i32, 1)?.reshape(&[1, HIDDEN])?)
            })
            .collect::<Result<Vec<_>>>()?;
        let hyper =
            stack_axis(&hyper.iter().collect::<Vec<_>>(), 1)?.reshape(&[1, NUM_MASK_TOKENS, ch])?;
        let up_flat = up2
            .reshape(&[1, (mg * mg), ch])?
            .transpose_axes(&[0, 2, 1])?; // [1, ch, mg²]
        let masks = hyper
            .matmul(&up_flat)?
            .reshape(&[1, NUM_MASK_TOKENS, mg, mg])?;
        let ious = self.iou_head.forward(&iou_token_out)?; // [1, 4]

        let (masks, ious) = if multimask_output {
            (
                slice_axis(&masks, 1, 1, NUM_MASK_TOKENS)?,
                slice_axis(&ious, 1, 1, NUM_MASK_TOKENS)?,
            )
        } else {
            // single-mask request (no-prompt video frames): dynamic_multimask_via_stability keeps
            // mask 0 if stable, else falls back to the best-IoU multimask candidate.
            dynamic_multimask_via_stability(&masks, &ious)?
        };
        Ok((masks, ious, obj_score, mask_tokens_out))
    }
}

// --- tracker neck (Sam3VisionNeck over tracker_neck.* + conv_s0/s1) ------------------------------

/// The tracker's FPN neck (same `FpnLayer` pyramid as the detector neck, separate `tracker_neck.*`
/// weights) plus the `conv_s0`/`conv_s1` high-res projections (which live under the mask decoder).
struct TrackerNeck {
    fpn_layers: Vec<FpnLayer>,
    conv_s0: (Array, Array), // 1×1 256→32
    conv_s1: (Array, Array), // 1×1 256→64
}

impl TrackerNeck {
    fn from_weights(
        w: &Weights,
        neck_prefix: &str,
        decoder_prefix: &str,
        cfg: &Sam3VisionConfig,
    ) -> Result<Self> {
        let fpn_layers = cfg
            .scale_factors
            .iter()
            .enumerate()
            .map(|(i, &scale)| {
                FpnLayer::from_weights(w, &join(neck_prefix, &format!("fpn_layers.{i}")), scale)
            })
            .collect::<Result<Vec<_>>>()?;
        let (s0w, s0b) = weight_bias(w, &join(decoder_prefix, "conv_s0"))?;
        let (s1w, s1b) = weight_bias(w, &join(decoder_prefix, "conv_s1"))?;
        Ok(Self {
            fpn_layers,
            conv_s0: (conv_w_ohwi(&s0w)?, s0b),
            conv_s1: (conv_w_ohwi(&s1w)?, s1b),
        })
    }

    /// `backbone`: NHWC `[1, g0, g0, C]` PE features. Returns `(image_embedding [1, g, g, 256],
    /// high_res [feat_s0 (4g²,32), feat_s1 (2g²,64)])`. The coarsest FPN level (36²) is dropped
    /// (`fpn_hidden_states[:-1]`).
    fn forward(&self, backbone: &Array) -> Result<(Array, [Array; 2])> {
        let fpn: Vec<Array> = self
            .fpn_layers
            .iter()
            .map(|l| l.forward(backbone))
            .collect::<Result<Vec<_>>>()?; // [288²,144²,72²,36²]
        let feat_s0 = conv2d(&fpn[0], &self.conv_s0.0, Some(&self.conv_s0.1), 1, 0)?; // 288², 32
        let feat_s1 = conv2d(&fpn[1], &self.conv_s1.0, Some(&self.conv_s1.1), 1, 0)?; // 144², 64
        let image_embedding = fpn[2].clone(); // 72², 256
        Ok((image_embedding, [feat_s0, feat_s1]))
    }
}

// --- memory encoder (Sam3TrackerVideoMemoryEncoder, F2) ------------------------------------------
//
// Encodes a frame's image features + its predicted mask into a 64-channel spatial memory + sine
// position encoding, stored in the memory bank for later frames. Mirrors `_encode_new_memory`
// (modeling_sam3_tracker_video.py ~2658) and the encoder forward (~1136). NHWC throughout; the
// reference is NCHW (channels-first `LayerNorm` ⇒ a plain LayerNorm over the channel axis here).

const SIGMOID_SCALE_FOR_MEM: f32 = 20.0;
const SIGMOID_BIAS_FOR_MEM: f32 = -10.0;
const MEM_OUT_CHANNELS: i32 = 64; // memory_encoder_output_channels
const MEM_POS_FEATS: i32 = MEM_OUT_CHANNELS / 2; // PositionEmbeddingSine num_position_features (32)
const MEM_SINE_TEMPERATURE: f32 = 10000.0;
const NUM_MASKMEM: i32 = 7; // num_maskmem (memory bank depth)

/// Grouped 2-D conv on an **NHWC** input with an OHWI weight (+ channel bias). `groups == channels`
/// is the depthwise case (`memory_fuser` 7×7); everything else is `groups = 1`.
fn conv2d_g(
    x: &Array,
    w_ohwi: &Array,
    b: &Array,
    stride: i32,
    pad: i32,
    groups: i32,
) -> Result<Array> {
    let y = mlx_rs::ops::conv2d(x, w_ohwi, (stride, stride), (pad, pad), (1, 1), groups)?;
    Ok(add(&y, b)?)
}

/// Separable bilinear-resize weight matrix `W` `[out, in]` for `align_corners=False`
/// (`out = W @ in @ Wᵀ`). Matches `torch.nn.functional.interpolate(mode="bilinear")`; the SAM3 mask
/// prep is 1008²→1152² (**upsampling**, so the reference `antialias=True` is a documented no-op).
fn bilinear_resize_matrix(in_size: i32, out_size: i32) -> Array {
    let mut data = vec![0f32; (out_size * in_size) as usize];
    let scale = in_size as f32 / out_size as f32;
    for o in 0..out_size {
        // area_pixel source index (align_corners=False), clamped to [0, in-1].
        let src = ((o as f32 + 0.5) * scale - 0.5).clamp(0.0, (in_size - 1) as f32);
        let x0 = src.floor() as i32;
        let x1 = (x0 + 1).min(in_size - 1);
        let frac = src - x0 as f32;
        data[(o * in_size + x0) as usize] += 1.0 - frac;
        data[(o * in_size + x1) as usize] += frac;
    }
    Array::from_slice(&data, &[out_size, in_size])
}

/// `PositionEmbeddingSine(num_position_features=N, normalize=True)` over a `g×g` grid → NHWC
/// `[1, g, g, 2N]`. Channel layout is `cat(pos_y[N], pos_x[N])`; within each half the `2k`/`2k+1` pair
/// is `(sin, cos)` at frequency `10000^(k/(N/2))`. Host-computed (weight-free). `N=32` is the memory
/// encoder's `maskmem_pos_enc`; `N=128` is the neck's `current_vision_pos` (`Sam3SinePositionEmbedding`).
fn position_embedding_sine(g: i32, num_pos: i32) -> Array {
    let half = (num_pos / 2) as usize;
    let scale = 2.0 * PI;
    let eps = 1e-6f32;
    let denom = g as f32 + eps;
    let freqs: Vec<f32> = (0..half)
        .map(|k| MEM_SINE_TEMPERATURE.powf((2.0 * k as f32) / num_pos as f32))
        .collect();
    let ch = (2 * num_pos) as usize; // 64
    let mut data = vec![0f32; (g * g) as usize * ch];
    for y in 0..g {
        let ye = (y as f32 + 1.0) / denom * scale;
        for x in 0..g {
            let xe = (x as f32 + 1.0) / denom * scale;
            let base = ((y * g + x) as usize) * ch;
            for k in 0..half {
                let (py, px) = (ye / freqs[k], xe / freqs[k]);
                data[base + 2 * k] = py.sin();
                data[base + 2 * k + 1] = py.cos();
                data[base + num_pos as usize + 2 * k] = px.sin();
                data[base + num_pos as usize + 2 * k + 1] = px.cos();
            }
        }
    }
    Array::from_slice(&data, &[1, g, g, ch as i32])
}

/// `MaskDownSampler`: 4× (k3/s2/p1 conv → channels-first LayerNorm → GELU), channels 1→4→16→64→256,
/// then a 1×1 `final_conv` (256→256). Shrinks `[1,1152,1152,1]` → `[1,72,72,256]`. NHWC.
struct MaskDownSampler {
    layers: Vec<((Array, Array), (Array, Array))>, // (conv (OHWI,bias), layer_norm (w,b))
    final_conv: (Array, Array),
}

impl MaskDownSampler {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let layers = (0..4)
            .map(|i| -> Result<((Array, Array), (Array, Array))> {
                let lp = join(prefix, &format!("layers.{i}"));
                let (cw, cb) = weight_bias(w, &join(&lp, "conv"))?;
                Ok((
                    (conv_w_ohwi(&cw)?, cb),
                    weight_bias(w, &join(&lp, "layer_norm"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let (fw, fb) = weight_bias(w, &join(prefix, "final_conv"))?;
        Ok(Self {
            layers,
            final_conv: (conv_w_ohwi(&fw)?, fb),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        for ((cw, cb), norm) in &self.layers {
            x = conv2d_g(&x, cw, cb, 2, 1, 1)?;
            x = gelu_exact(&ln(&x, norm)?)?;
        }
        conv2d_g(&x, &self.final_conv.0, &self.final_conv.1, 1, 0, 1)
    }
}

/// `MemoryFuserCXBlock`: ConvNeXt-style residual — 7×7 depthwise conv → channels-first LayerNorm →
/// 1×1 expand (256→1024) → GELU → 1×1 project (1024→256) → per-channel `scale` → +input. NHWC, so
/// the pointwise convs are last-axis linears with no permute.
struct CxBlock {
    depthwise: (Array, Array), // OHWI [256,7,7,1], depthwise (groups=256)
    norm: (Array, Array),
    pw1: (Array, Array),
    pw2: (Array, Array),
    scale: Array, // [256]
}

impl CxBlock {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let (dw, db) = weight_bias(w, &join(prefix, "depthwise_conv"))?;
        Ok(Self {
            depthwise: (conv_w_ohwi(&dw)?, db),
            norm: weight_bias(w, &join(prefix, "layer_norm"))?,
            pw1: weight_bias(w, &join(prefix, "pointwise_conv1"))?,
            pw2: weight_bias(w, &join(prefix, "pointwise_conv2"))?,
            scale: w.require(&join(prefix, "scale"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = conv2d_g(x, &self.depthwise.0, &self.depthwise.1, 1, 3, HIDDEN)?; // groups == channels
        let h = ln(&h, &self.norm)?;
        let h = linear(
            &gelu_exact(&linear(&h, &self.pw1.0, &self.pw1.1)?)?,
            &self.pw2.0,
            &self.pw2.1,
        )?;
        let h = multiply(&self.scale, &h)?;
        Ok(add(x, &h)?)
    }
}

/// `Sam3TrackerVideoMemoryEncoder`: `mask_downsampler` + `feature_projection` (1×1 256→256) +
/// `memory_fuser` (2 CXBlocks) + `projection` (1×1 256→64), returning the 64-channel memory map +
/// its sine position encoding.
struct MemoryEncoder {
    mask_downsampler: MaskDownSampler,
    feature_projection: (Array, Array), // 1×1 256→256
    fuser: Vec<CxBlock>,
    projection: (Array, Array), // 1×1 256→64
}

impl MemoryEncoder {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let (fpw, fpb) = weight_bias(w, &join(prefix, "feature_projection"))?;
        let (pw, pb) = weight_bias(w, &join(prefix, "projection"))?;
        Ok(Self {
            mask_downsampler: MaskDownSampler::from_weights(w, &join(prefix, "mask_downsampler"))?,
            feature_projection: (conv_w_ohwi(&fpw)?, fpb),
            fuser: vec![
                CxBlock::from_weights(w, &join(prefix, "memory_fuser.layers.0"))?,
                CxBlock::from_weights(w, &join(prefix, "memory_fuser.layers.1"))?,
            ],
            projection: (conv_w_ohwi(&pw)?, pb),
        })
    }

    /// `pix_feat`: NHWC `[1,72,72,256]` raw image embedding; `mask_for_mem`: NHWC `[1,1152,1152,1]`
    /// scaled mask. Returns `(features [1,72,72,64], pos_enc [1,72,72,64])`.
    fn forward(&self, pix_feat: &Array, mask_for_mem: &Array) -> Result<(Array, Array)> {
        let masks = self.mask_downsampler.forward(mask_for_mem)?; // [1,72,72,256]
        let mut x = conv2d_g(
            pix_feat,
            &self.feature_projection.0,
            &self.feature_projection.1,
            1,
            0,
            1,
        )?;
        x = add(&x, &masks)?;
        for layer in &self.fuser {
            x = layer.forward(&x)?;
        }
        let features = conv2d_g(&x, &self.projection.0, &self.projection.1, 1, 0, 1)?; // [1,72,72,64]
        let pos = position_embedding_sine(features.shape()[1], MEM_POS_FEATS);
        Ok((features, pos))
    }
}

// --- memory attention (Sam3TrackerVideoMemoryAttention, F2) --------------------------------------
//
// 4 RoPE-attention layers that fuse a frame's vision features (self-attn) with the temporal memory
// bank (cross-attn). Mirrors `Sam3TrackerVideoMemoryAttention` (~965), `…MemoryAttentionLayer`
// (~914), `…RoPEAttention` (~841), `…VisionRotaryEmbedding` (~733). hidden 256, 1 head, head_dim 256,
// FFN 2048 (**relu**), axial 2-D RoPE θ1e4 over the 72² grid. Cross-attn keys/values come from the
// 64-dim memory; object-pointer tokens (the trailing `num_k_exclude_rope`) skip RoPE.

const MEM_ATTN_LAYERS: i32 = 4;
const MEM_ATTN_HEADS: i32 = 1;
const MEM_ATTN_HEAD_DIM: i32 = HIDDEN / MEM_ATTN_HEADS; // 256 (downsample_rate 1); FFN 2048 from weights
const ROPE_THETA: f32 = 10000.0;

/// `VisionRotaryEmbedding.create_inv_freq` → `(cos, sin)` tables `[g·g, 256]` for axial 2-D RoPE.
/// `inv_freq[p] = repeat_interleave(cat(x_pos·freqs, y_pos·freqs), 2)` with
/// `freqs[j] = θ^(−4j/256)`, `j∈0..64`. Host-computed (weight-free), validated vs the oracle tables.
fn build_rope_tables(grid: i32) -> (Array, Array) {
    let dim = MEM_ATTN_HEAD_DIM; // 256
    let nf = (dim / 4) as usize; // 64 frequencies per axis
    let freqs: Vec<f32> = (0..nf)
        .map(|j| ROPE_THETA.powf(-(4.0 * j as f32) / dim as f32))
        .collect();
    let seq = (grid * grid) as usize;
    let d = dim as usize;
    let (mut cosd, mut sind) = (vec![0f32; seq * d], vec![0f32; seq * d]);
    for p in 0..seq {
        let x = (p as i32 % grid) as f32;
        let y = (p as i32 / grid) as f32;
        for (j, &f) in freqs.iter().enumerate() {
            // x-half occupies base indices 0..64 → interleaved positions 2j, 2j+1.
            let (fx, fy) = (x * f, y * f);
            let bx = 2 * j;
            let by = 2 * (nf + j);
            for (pos, ang) in [(bx, fx), (by, fy)] {
                let (c, s) = (ang.cos(), ang.sin());
                cosd[p * d + pos] = c;
                cosd[p * d + pos + 1] = c;
                sind[p * d + pos] = s;
                sind[p * d + pos + 1] = s;
            }
        }
    }
    let shape = [seq as i32, dim];
    (
        Array::from_slice(&cosd, &shape),
        Array::from_slice(&sind, &shape),
    )
}

/// `rotate_pairwise`: pairs `(a, b)` → `(−b, a)` along the last (head_dim) axis.
fn rotate_pairwise(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let d = sh[sh.len() - 1];
    let mut paired: Vec<i32> = sh[..sh.len() - 1].to_vec();
    paired.push(d / 2);
    paired.push(2);
    let xr = x.reshape(&paired)?;
    let x1 = take1(&xr, 0, -1)?; // even lanes
    let x2 = take1(&xr, 1, -1)?; // odd lanes
    let stacked = stack_axis(&[&x2.multiply(Array::from_f32(-1.0))?, &x1], -1)?;
    Ok(stacked.reshape(sh)?)
}

/// `apply_rotary_pos_emb_2d`: rotate all of `q`; rotate the leading `seq_k − num_k_exclude` keys
/// (object-pointer tokens pass through). `q`/`k`: `[1, heads, seq, head_dim]`; `cos`/`sin`:
/// `[seq, head_dim]`. `repeat_freqs_k` tiles the tables when the key length is a multiple of `q`.
fn apply_rope_2d(
    q: &Array,
    k: &Array,
    cos: &Array,
    sin: &Array,
    num_k_exclude: i32,
    repeat_freqs_k: bool,
) -> Result<(Array, Array)> {
    let q_embed = add(&multiply(q, cos)?, &multiply(&rotate_pairwise(q)?, sin)?)?;
    let seq_k = k.shape()[2];
    let n_rot = seq_k - num_k_exclude;
    let k_rot = slice_axis(k, 2, 0, n_rot)?;
    let q_seq = q.shape()[2];
    let (cos_k, sin_k) = if repeat_freqs_k && n_rot != q_seq {
        // `repeat_interleave` builds `rf·q_seq` frequency rows; if `n_rot` isn't an exact multiple of
        // `q_seq` the table won't line up with `k_rot`'s `n_rot` positions → a Metal shape failure.
        // Canonical window/feature sizes satisfy the invariant; reject a non-canonical one cleanly
        // instead of producing a wrong-shape RoPE table (F-016).
        if q_seq == 0 || n_rot % q_seq != 0 {
            return Err(Error::Msg(format!(
                "sam3 tracker: rope key length {n_rot} is not a multiple of query length {q_seq}"
            )));
        }
        let rf = (n_rot / q_seq) as usize;
        (
            concatenate_axis(&vec![cos; rf], 0)?,
            concatenate_axis(&vec![sin; rf], 0)?,
        )
    } else {
        (cos.clone(), sin.clone())
    };
    let k_embed = add(
        &multiply(&k_rot, &cos_k)?,
        &multiply(&rotate_pairwise(&k_rot)?, &sin_k)?,
    )?;
    let k_out = if num_k_exclude > 0 {
        let k_pass = slice_axis(k, 2, n_rot, seq_k)?;
        concatenate_axis(&[&k_embed, &k_pass], 2)?
    } else {
        k_embed
    };
    Ok((q_embed, k_out))
}

/// `RoPEAttention`: q/k/v project to `internal = 256` (downsample 1), split into `MEM_ATTN_HEADS`,
/// axial RoPE on q + the rotated keys, SDPA, then `o_proj`. `kv_in_dim` is 256 (self-attn) or 64
/// (cross-attn over the memory bank).
struct RoPEAttention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    rope_k_repeat: bool,
}

impl RoPEAttention {
    fn from_weights(w: &Weights, prefix: &str, rope_k_repeat: bool) -> Result<Self> {
        let l = |n: &str| crate::load_linear(w, &join(prefix, n));
        Ok(Self {
            q: l("q_proj")?,
            k: l("k_proj")?,
            v: l("v_proj")?,
            o: l("o_proj")?,
            rope_k_repeat,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        crate::quantize_linear(&mut self.q, bits)?;
        crate::quantize_linear(&mut self.k, bits)?;
        crate::quantize_linear(&mut self.v, bits)?;
        crate::quantize_linear(&mut self.o, bits)?;
        Ok(())
    }

    /// `query`: `[1, seq, 256]`; `key`/`value`: `[1, seq_k, kv_in]`. Returns `[1, seq, 256]`.
    fn forward(
        &self,
        query: &Array,
        key: &Array,
        value: &Array,
        cos: &Array,
        sin: &Array,
        num_k_exclude: i32,
    ) -> Result<Array> {
        let to_heads = |x: &Array| -> Result<Array> {
            let sh = x.shape();
            Ok(
                x.reshape(&[sh[0], sh[1], MEM_ATTN_HEADS, MEM_ATTN_HEAD_DIM])?
                    .transpose_axes(&[0, 2, 1, 3])?,
            ) // [1, heads, seq, head_dim]
        };
        let q = to_heads(&self.q.forward(query)?)?;
        let k = to_heads(&self.k.forward(key)?)?;
        let v = to_heads(&self.v.forward(value)?)?;
        let (q, k) = apply_rope_2d(&q, &k, cos, sin, num_k_exclude, self.rope_k_repeat)?;
        let scale = 1.0 / (MEM_ATTN_HEAD_DIM as f32).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let sh = out.shape();
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            sh[0],
            sh[2],
            MEM_ATTN_HEADS * MEM_ATTN_HEAD_DIM,
        ])?;
        self.o.forward(&out)
    }
}

/// One memory-attention layer: pre-norm self-attn → pre-norm cross-attn over `keys + key_pos` →
/// pre-norm FFN (linear1 → relu → linear2), each a residual add.
struct MemAttnLayer {
    self_attn: RoPEAttention,
    cross_attn: RoPEAttention,
    norm1: (Array, Array),
    norm2: (Array, Array),
    norm3: (Array, Array),
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
}

impl MemAttnLayer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            self_attn: RoPEAttention::from_weights(w, &join(prefix, "self_attn"), false)?,
            cross_attn: RoPEAttention::from_weights(w, &join(prefix, "cross_attn_image"), true)?,
            norm1: weight_bias(w, &join(prefix, "layer_norm1"))?,
            norm2: weight_bias(w, &join(prefix, "layer_norm2"))?,
            norm3: weight_bias(w, &join(prefix, "layer_norm3"))?,
            linear1: crate::load_linear(w, &join(prefix, "linear1"))?,
            linear2: crate::load_linear(w, &join(prefix, "linear2"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.self_attn.quantize(bits)?;
        self.cross_attn.quantize(bits)?;
        crate::quantize_linear(&mut self.linear1, bits)?;
        crate::quantize_linear(&mut self.linear2, bits)?;
        Ok(())
    }

    /// `queries`: `[1, seq, 256]`; `keys`/`key_pos`: `[1, seq_k, 64]`. `num_k_exclude` = object-pointer
    /// token count (skip RoPE on the cross-attn keys). `cos`/`sin` are the RoPE tables.
    fn forward(
        &self,
        queries: &Array,
        keys: &Array,
        key_pos: &Array,
        cos: &Array,
        sin: &Array,
        num_k_exclude: i32,
    ) -> Result<Array> {
        // self-attention (no excluded keys; RoPE on full q/k).
        let q = ln(queries, &self.norm1)?;
        let q = self.self_attn.forward(&q, &q, &q, cos, sin, 0)?;
        let queries = add(queries, &q)?;
        // cross-attention over the memory bank (keys offset by their positional embedding).
        let q = ln(&queries, &self.norm2)?;
        let key = add(keys, key_pos)?;
        let q = self
            .cross_attn
            .forward(&q, &key, keys, cos, sin, num_k_exclude)?;
        let queries = add(&queries, &q)?;
        // FFN.
        let q = ln(&queries, &self.norm3)?;
        let q = self.linear2.forward(&relu(&self.linear1.forward(&q)?)?)?;
        Ok(add(&queries, &q)?)
    }
}

/// `Sam3TrackerVideoMemoryAttention`: `no_memory`-frame conditioning is handled by the caller; this
/// is the memory-conditioned path — 4 layers + a final LayerNorm, over precomputed RoPE tables.
struct MemoryAttention {
    layers: Vec<MemAttnLayer>,
    norm: (Array, Array),
    rope_cos: Array,
    rope_sin: Array,
}

impl MemoryAttention {
    fn from_weights(w: &Weights, prefix: &str, grid: i32) -> Result<Self> {
        let layers = (0..MEM_ATTN_LAYERS)
            .map(|i| MemAttnLayer::from_weights(w, &join(prefix, &format!("layers.{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let (rope_cos, rope_sin) = build_rope_tables(grid);
        Ok(Self {
            layers,
            norm: weight_bias(w, &join(prefix, "layer_norm"))?,
            rope_cos,
            rope_sin,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// `current_vision_features`/`current_vision_pos`: seq-first `[seq, 1, 256]`; `memory`/`memory_pos`:
    /// seq-first `[seq_k, 1, 64]`. `num_object_pointer_tokens` are the trailing memory tokens excluded
    /// from RoPE. Returns the conditioned features batch-first `[1, seq, 256]`.
    fn forward(
        &self,
        current_vision_features: &Array,
        current_vision_pos: &Array,
        memory: &Array,
        memory_pos: &Array,
        num_object_pointer_tokens: i32,
    ) -> Result<Array> {
        // output = vision + 0.1·pos, then to batch-first [1, seq, C].
        let output = add(
            current_vision_features,
            &current_vision_pos.multiply(Array::from_f32(0.1))?,
        )?;
        let mut output = output.transpose_axes(&[1, 0, 2])?; // [1, seq, 256]
        let mem = memory.transpose_axes(&[1, 0, 2])?; // [1, seq_k, 64]
        let mem_pos = memory_pos.transpose_axes(&[1, 0, 2])?;
        for layer in &self.layers {
            output = layer.forward(
                &output,
                &mem,
                &mem_pos,
                &self.rope_cos,
                &self.rope_sin,
                num_object_pointer_tokens,
            )?;
        }
        ln(&output, &self.norm)
    }
}

// --- single-frame tracker -----------------------------------------------------------------------

/// SAM3 single-frame box-prompt tracker (PVS path). Reuses the shared PE [`Backbone`]; loads the
/// tracker neck + prompt encoder + mask decoder from `tracker_neck.*` / `tracker_model.*`.
pub struct Sam3Tracker {
    backbone: Rc<Backbone>,
    neck: TrackerNeck,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    /// `tracker_model.shared_image_embedding` — a **separate** random-Gaussian table from the prompt
    /// encoder's `shared_embedding`; supplies the dense image positional encoding (the two are not
    /// the same matrix in the checkpoint — `get_image_wide_positional_embeddings`).
    image_pe_embed: PositionEmbeddingRandom,
    /// `tracker_model.no_memory_embedding` `[1, 1, 256]` — the learned "no memory yet" bias added to
    /// the image embedding on a frame with no memory conditioning (the single-frame / init path).
    /// On a memory-conditioned video frame this is replaced by memory attention (F2).
    no_memory_embedding: Array,
    /// `tracker_model.memory_encoder` — encodes a frame's image features + predicted mask into the
    /// 64-channel spatial memory stored in the memory bank (F2).
    memory_encoder: MemoryEncoder,
    /// `tracker_model.occlusion_spatial_embedding_parameter` `[1, 64]` — added to the spatial memory
    /// when the object is predicted absent (`object_score_logits ≤ 0`).
    occlusion: Array,
    /// `tracker_model.memory_attention` — 4 RoPE layers fusing a frame's vision features with the
    /// temporal memory bank (F2). The per-object memory-bank assembly that feeds this is F2.4
    /// ([`Sam3Tracker::prepare_memory_conditioned_features`]).
    memory_attention: MemoryAttention,
    /// `tracker_model.object_pointer_proj` — 3-layer FeedForward (256→256) projecting the SAM mask
    /// output token into the object pointer stored in the memory bank (F2.4).
    object_pointer_proj: FeedForward,
    /// `tracker_model.no_object_pointer` `[1, 256]` — the object pointer used when the object is
    /// predicted absent (`object_score ≤ 0`).
    no_object_pointer: Array,
    /// `tracker_model.memory_temporal_positional_encoding` `[7, 1, 1, 64]` — per-temporal-offset
    /// learned bias added to a memory frame's spatial position encoding (indexed `offset − 1`, where
    /// a conditioning frame's `offset = 0` wraps to the last entry, mirroring Python negative
    /// indexing).
    mem_temporal_pos_enc: Array,
    /// `tracker_model.temporal_positional_encoding_projection_layer` — Linear 256→64 projecting an
    /// object pointer's 1-D sine temporal encoding down to the memory dimension (F2.4).
    tpos_proj: AdaptableLinear,
    /// `tracker_model.mask_downsample` — a single conv (1→1, k4s4) shrinking the input mask before the
    /// prompt encoder's `mask_embed` on the `_use_mask_as_output` object-pointer path (F2.5a-ii).
    mask_downsample: (Array, Array),
}

/// A frame's encoded memory: the 64-channel spatial feature map + its sine position encoding, both
/// NHWC `[1, 72, 72, 64]` (f32). The reference casts the features to bf16 for storage; that cast is
/// applied by the memory-bank layer, not here.
pub struct MemoryFeatures {
    pub features: Array,
    pub pos: Array,
}

const MASK_MEM_SIZE: i32 = 1152; // mask_input_size (4·72) · 4

/// A no-prompt (memory-conditioned) tracking-frame prediction (`_run_single_frame_inference` output):
/// the low-res (288²) and high-res (1008²) mask logits, the object pointer stored in the memory bank,
/// and the object-score logit.
pub struct TrackerFrameOutput {
    /// Low-res mask logits `[1, 1, 288, 288]` (= `low_res_mask_size`).
    pub low_res: Array,
    /// High-res mask logits `[1, 1, 1008, 1008]` (bilinear-upsampled; fed to the memory encoder).
    pub high_res: Array,
    /// Object pointer `[1, 256]` (`object_pointer_proj` of the SAM mask token, occlusion-gated).
    pub object_pointer: Array,
    /// Object-score logit (`> 0` ⇒ object present).
    pub object_score: f32,
}

/// A single-frame tracker prediction: the best (argmax-IoU) low-res mask logits + its IoU + the
/// object-score logit.
pub struct TrackerMask {
    /// Low-res mask logits `[mg, mg]` (f32) for the best candidate.
    pub low_res: Array,
    pub iou: f32,
    pub object_score: f32,
}

impl Sam3Tracker {
    /// Load from a `facebook/sam3` weight map (PE backbone + `tracker_neck` + `tracker_model`). The
    /// backbone is loaded from `detector_model.vision_encoder.backbone` — the **same** keys the
    /// detector's vision encoder uses. In the video pipeline use
    /// [`Self::from_weights_with_backbone`] to share one backbone with the detector instead of
    /// loading a second copy (F-028).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let cfg = Sam3VisionConfig::sam3();
        let backbone = Rc::new(Backbone::from_weights(
            w,
            "detector_model.vision_encoder.backbone",
            &cfg,
        )?);
        Self::from_weights_with_backbone(w, backbone)
    }

    /// Load the tracker reusing an already-loaded (and possibly shared) PE [`Backbone`]. Lets the
    /// video model share one backbone between the detector segmenter and the tracker (F-028).
    pub(crate) fn from_weights_with_backbone(w: &Weights, backbone: Rc<Backbone>) -> Result<Self> {
        let cfg = Sam3VisionConfig::sam3();
        Ok(Self {
            backbone,
            neck: TrackerNeck::from_weights(w, "tracker_neck", "tracker_model.mask_decoder", &cfg)?,
            prompt: PromptEncoder::from_weights(w, "tracker_model.prompt_encoder")?,
            decoder: MaskDecoder::from_weights(w, "tracker_model.mask_decoder")?,
            image_pe_embed: PositionEmbeddingRandom {
                gaussian: w
                    .require("tracker_model.shared_image_embedding.positional_embedding")?
                    .clone(),
            },
            no_memory_embedding: w.require("tracker_model.no_memory_embedding")?.clone(),
            memory_encoder: MemoryEncoder::from_weights(w, "tracker_model.memory_encoder")?,
            occlusion: w
                .require("tracker_model.occlusion_spatial_embedding_parameter")?
                .clone(),
            memory_attention: MemoryAttention::from_weights(
                w,
                "tracker_model.memory_attention",
                (INPUT_SIZE as i32) / 14, // 72² RoPE grid
            )?,
            object_pointer_proj: FeedForward::from_weights(
                w,
                "tracker_model.object_pointer_proj",
                3,
                false,
            )?,
            no_object_pointer: w.require("tracker_model.no_object_pointer")?.clone(),
            mem_temporal_pos_enc: w
                .require("tracker_model.memory_temporal_positional_encoding")?
                .clone(),
            tpos_proj: crate::load_linear(
                w,
                "tracker_model.temporal_positional_encoding_projection_layer",
            )?,
            mask_downsample: {
                let (mw, mb) = weight_bias(w, "tracker_model.mask_downsample")?;
                (conv_w_ohwi(&mw)?, mb)
            },
        })
    }

    /// Quantize the tracker's linear projections (Q8/Q4): the shared PE backbone, the mask decoder's
    /// two-way transformer + hypernet/IoU/obj-score MLPs, the memory-attention RoPE attention + FFN,
    /// the object-pointer projection, and the temporal-pos projection. Convs (tracker neck, memory
    /// encoder, ConvNeXt pointwise convs, upscale/mask-downsample), GroupNorms, the prompt encoder's
    /// embeddings, and the random-Gaussian position tables stay dense (sc-4925).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        quantize_backbone_rc(&mut self.backbone, bits)?;
        self.quantize_except_backbone(bits)
    }

    /// Quantize everything **except** the shared PE backbone (the mask decoder, memory attention,
    /// object-pointer projection, and temporal-pos projection). The video model calls this after
    /// quantizing the single shared backbone once, so the backbone isn't quantized twice (F-028).
    pub(crate) fn quantize_except_backbone(&mut self, bits: i32) -> Result<()> {
        self.decoder.quantize(bits)?;
        self.memory_attention.quantize(bits)?;
        self.object_pointer_proj.quantize(bits)?;
        crate::quantize_linear(&mut self.tpos_proj, bits)?;
        Ok(())
    }

    /// The shared PE [`Backbone`] handle (clone of the `Rc`), for the video model to quantize once
    /// and reinstall into both consumers (F-028).
    pub(crate) fn backbone_rc(&self) -> Rc<Backbone> {
        self.backbone.clone()
    }

    /// Replace the PE backbone with a (typically pre-quantized, shared) one.
    pub(crate) fn set_backbone(&mut self, backbone: Rc<Backbone>) {
        self.backbone = backbone;
    }

    /// The axial 2-D RoPE `(cos, sin)` tables `[72², 256]` the memory attention uses (exposed for
    /// parity validation against the reference `VisionRotaryEmbedding`).
    pub fn memory_attention_rope_tables(&self) -> (Array, Array) {
        (
            self.memory_attention.rope_cos.clone(),
            self.memory_attention.rope_sin.clone(),
        )
    }

    /// Memory-conditioned features (`_prepare_memory_conditioned_features` non-init branch): fuse a
    /// frame's vision features with the assembled memory bank via memory attention. `current_vision_*`:
    /// seq-first `[seq, 1, 256]`; `memory`/`memory_pos`: seq-first `[seq_k, 1, 64]` (spatial memory +
    /// trailing `num_object_pointer_tokens` object-pointer tokens). Returns batch-first `[1, seq, 256]`.
    /// The per-object bank assembly that produces `memory` lands with F2.4.
    pub fn condition_with_memory(
        &self,
        current_vision_features: &Array,
        current_vision_pos: &Array,
        memory: &Array,
        memory_pos: &Array,
        num_object_pointer_tokens: i32,
    ) -> Result<Array> {
        self.memory_attention.forward(
            current_vision_features,
            current_vision_pos,
            memory,
            memory_pos,
            num_object_pointer_tokens,
        )
    }

    /// `_single_frame_forward` object-pointer tail: project the SAM mask output token through
    /// `object_pointer_proj`, gated by the object-appearing flag — present (`object_score > 0`) keeps
    /// the projection, absent replaces it with the learned `no_object_pointer`. Returns `[1, 256]`.
    pub fn compute_object_pointer(&self, mask_token: &Array, object_score: f32) -> Result<Array> {
        let token = mask_token.reshape(&[1, HIDDEN])?;
        if object_score > 0.0 {
            self.object_pointer_proj.forward(&token)
        } else {
            Ok(self.no_object_pointer.reshape(&[1, HIDDEN])?)
        }
    }

    /// `_prepare_memory_conditioned_features` (non-init branch) — assemble the per-object memory bank
    /// and fuse it with the current frame's vision features via memory attention. This is the F2.4
    /// bank-assembly that feeds [`Self::condition_with_memory`].
    ///
    /// - `current_vision_features` / `current_vision_pos`: seq-first `[seq, 1, 256]` (the 72² image
    ///   embedding from [`Self::encode_frame`], flattened HW-first).
    /// - `spatial_mem`: the gathered memory frames as `(relative_temporal_offset, features, pos)`,
    ///   each `features`/`pos` seq-first `[seq, 1, 64]` (the stored `maskmem_features`/`maskmem_pos_enc`).
    ///   `_build_memory_attention_inputs` adds `memory_temporal_positional_encoding[offset − 1]` to each
    ///   frame's spatial pos (offset 0 → last entry, Python negative-index wrap).
    /// - `object_pointers`: `(temporal_offset, pointer [1, 256])` — `_get_object_pointers` selection;
    ///   `_process_object_pointers` adds a 1-D sine temporal PE (normalized by
    ///   `max_object_pointers_to_use − 1`, projected 256→64), then splits each 256-d pointer into
    ///   `256/64 = 4` consecutive 64-d memory tokens appended after the spatial memory.
    ///
    /// Returns the conditioned feature map NHWC `[1, g, g, 256]` for the SAM decoder.
    pub fn prepare_memory_conditioned_features(
        &self,
        current_vision_features: &Array,
        current_vision_pos: &Array,
        spatial_mem: &[(i32, Array, Array)],
        object_pointers: &[(i32, Array)],
        max_object_pointers_to_use: i32,
    ) -> Result<Array> {
        // Recover the square grid side from the flat sequence length. A bare `sqrt() as i32` would
        // truncate one short on a non-perfect-square length, mis-shaping the `[g,g,…]` reshape below
        // → a panic. Round and verify the length really is a perfect square (F-018).
        let seq = current_vision_features.shape()[0];
        let g = (seq as f64).sqrt().round() as i32;
        if g * g != seq {
            return Err(Error::Msg(format!(
                "sam3 tracker: vision feature length {seq} is not a perfect square (g={g})"
            )));
        }

        // Spatial memory: concat each gathered frame's features + (pos + temporal-pos[offset−1]).
        let mut mem_feats: Vec<Array> = Vec::new();
        let mut mem_pos: Vec<Array> = Vec::new();
        for (offset, feat, pos) in spatial_mem {
            let idx = (offset - 1).rem_euclid(NUM_MASKMEM); // offset 0 → 6 (negative-index wrap)
            let tpos =
                take1(&self.mem_temporal_pos_enc, idx, 0)?.reshape(&[1, 1, MEM_OUT_CHANNELS])?;
            mem_feats.push(feat.clone());
            mem_pos.push(add(pos, &tpos)?);
        }

        // Object pointers: sine temporal PE → project → split 256 into 4×64 memory tokens.
        let mut num_object_pointer_tokens = 0;
        if !object_pointers.is_empty() {
            let num_splits = HIDDEN / MEM_OUT_CHANNELS; // 4
            let max_temporal_diff = (max_object_pointers_to_use - 1).max(1) as f32;
            let offsets: Vec<f32> = object_pointers
                .iter()
                .map(|(o, _)| *o as f32 / max_temporal_diff)
                .collect();
            let sine_pe = sine_pe_1d(&offsets, HIDDEN as usize, MEM_SINE_TEMPERATURE); // [P, 256]
            let proj = self.tpos_proj.forward(&sine_pe)?; // [P, 64]
            let p = object_pointers.len() as i32;
            // pointer tokens stacked [P, 256] → split each into 4×64 contiguous → [P·4, 1, 64].
            let mut rows: Vec<Array> = Vec::with_capacity(object_pointers.len());
            for (_, t) in object_pointers {
                rows.push(t.reshape(&[1, HIDDEN])?);
            }
            let stacked = concatenate_axis(&rows.iter().collect::<Vec<_>>(), 0)?; // [P, 256]
            let split = stacked.reshape(&[p * num_splits, 1, MEM_OUT_CHANNELS])?; // [P·4, 1, 64]
                                                                                  // pos embed [P, 64] → [P, 1, 64] → repeat_interleave(4) → [P·4, 1, 64].
            let pe = proj.reshape(&[p, 1, 1, MEM_OUT_CHANNELS])?;
            let pe = broadcast_to(&pe, &[p, num_splits, 1, MEM_OUT_CHANNELS])?.reshape(&[
                p * num_splits,
                1,
                MEM_OUT_CHANNELS,
            ])?;
            mem_feats.push(split);
            mem_pos.push(pe);
            num_object_pointer_tokens = p * num_splits;
        }

        let combined_memory = concatenate_axis(&mem_feats.iter().collect::<Vec<_>>(), 0)?;
        let combined_pos = concatenate_axis(&mem_pos.iter().collect::<Vec<_>>(), 0)?;
        let conditioned = self.condition_with_memory(
            current_vision_features,
            current_vision_pos,
            &combined_memory,
            &combined_pos,
            num_object_pointer_tokens,
        )?; // [1, seq, 256] batch-first
        Ok(conditioned.reshape(&[1, g, g, HIDDEN])?)
    }

    /// Mask prep for `_encode_new_memory`: resize the image-resolution mask logits to the 1152² mask
    /// memory size (separable bilinear, `align_corners=False`), then `sigmoid` (or `>0` binarize for
    /// point/box-prompted frames), then `·20 −10`. Returns NHWC `[1, 1152, 1152, 1]`.
    pub fn prepare_mask_for_mem(
        &self,
        pred_high_res: &Array,
        is_mask_from_pts: bool,
    ) -> Result<Array> {
        let sh = pred_high_res.shape();
        let (in_h, in_w) = (sh[sh.len() - 2], sh[sh.len() - 1]);
        let m = pred_high_res.reshape(&[in_h, in_w])?;
        let resized = if in_h == MASK_MEM_SIZE && in_w == MASK_MEM_SIZE {
            m
        } else {
            let wh = bilinear_resize_matrix(in_h, MASK_MEM_SIZE);
            let ww = bilinear_resize_matrix(in_w, MASK_MEM_SIZE);
            wh.matmul(&m)?.matmul(&ww.transpose_axes(&[1, 0])?)?
        };
        let prob = if is_mask_from_pts {
            resized.gt(Array::from_f32(0.0))?.as_dtype(Dtype::Float32)?
        } else {
            sigmoid(&resized)?
        };
        let scaled = add(
            &prob.multiply(Array::from_f32(SIGMOID_SCALE_FOR_MEM))?,
            Array::from_f32(SIGMOID_BIAS_FOR_MEM),
        )?;
        Ok(scaled.reshape(&[1, MASK_MEM_SIZE, MASK_MEM_SIZE, 1])?)
    }

    /// `_encode_new_memory`: encode a frame's raw image embedding + its predicted mask into spatial
    /// memory. `pix_feat`: NHWC `[1, 72, 72, 256]` raw image embedding (the [`Self::encode_frame`]
    /// first output — **no** `no_memory_embedding` bias). `pred_high_res`: `[1, 1, 1008, 1008]`
    /// image-resolution mask logits. `object_score`: the decoder object-score logit (drives the
    /// occlusion add). `is_mask_from_pts`: binarize vs sigmoid the mask (true for point/box frames).
    pub fn encode_new_memory(
        &self,
        pix_feat: &Array,
        pred_high_res: &Array,
        object_score: f32,
        is_mask_from_pts: bool,
    ) -> Result<MemoryFeatures> {
        let mask_for_mem = self.prepare_mask_for_mem(pred_high_res, is_mask_from_pts)?;
        let (mut features, pos) = self.memory_encoder.forward(pix_feat, &mask_for_mem)?;
        if object_score <= 0.0 {
            // object predicted absent → add the occlusion spatial embedding over the grid.
            features = add(
                &features,
                &self.occlusion.reshape(&[1, 1, 1, MEM_OUT_CHANNELS])?,
            )?;
        }
        Ok(MemoryFeatures { features, pos })
    }

    /// Encode a frame's pixels `[1, 3, 1008, 1008]` → `(image_embedding, high_res)`. Runs the shared
    /// PE backbone once; the detector path can run its own neck over the same backbone separately.
    pub fn encode_frame(&self, pixel_values: &Array) -> Result<(Array, [Array; 2])> {
        let backbone = self.backbone.forward(pixel_values)?;
        self.neck.forward(&backbone)
    }

    /// The neck's 72² sine position encoding (`Sam3SinePositionEmbedding`, `num_position_features=128`,
    /// `normalize=True`) flattened seq-first `[g², 1, 256]` — the `current_vision_pos` that memory
    /// attention adds (`+0.1·pos`) to the conditioned features. Weight-free; depends only on the grid.
    pub fn frame_position_encoding(&self, g: i32) -> Result<Array> {
        Ok(position_embedding_sine(g, HIDDEN / 2).reshape(&[g * g, 1, HIDDEN])?)
    }

    /// Box-prompt a pre-encoded frame: `box_xyxy` in **1008-input** space → best low-res mask.
    pub fn segment_encoded(
        &self,
        image_embedding: &Array,
        high_res: &[Array; 2],
        box_xyxy_1008: [f32; 4],
    ) -> Result<TrackerMask> {
        self.segment_encoded_multimask(image_embedding, high_res, box_xyxy_1008, true)
    }

    /// Like [`Self::segment_encoded`] but choosing the mask-output policy: `true` requests the 3
    /// multimask candidates (box-prompt PVS path), `false` requests a single mask via
    /// `dynamic_multimask_via_stability` (the no-prompt video-frame decode path). Exposed for F2.
    pub fn segment_encoded_multimask(
        &self,
        image_embedding: &Array,
        high_res: &[Array; 2],
        box_xyxy_1008: [f32; 4],
        multimask: bool,
    ) -> Result<TrackerMask> {
        let g = image_embedding.shape()[1];
        // No-memory single-frame path: add the learned no-memory bias to the image embedding
        // (broadcast over the spatial grid). A memory-conditioned video frame skips this (F2).
        let image_embedding = add(
            image_embedding,
            &self.no_memory_embedding.reshape(&[1, 1, 1, HIDDEN])?,
        )?;
        let (sparse, dense) = self.prompt.encode_box(box_xyxy_1008, g)?;
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        let (masks, ious, obj_score, _mask_tokens) = self.decoder.forward(
            &image_embedding,
            &image_pe,
            &sparse,
            &dense,
            high_res,
            multimask,
        )?;
        // argmax IoU over the returned candidates (host).
        let iv = ious
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec();
        let best = argmax(&iv);
        let mg = masks.shape()[2];
        let low_res = take1(&take1(&masks, 0, 0)?, best as i32, 0)?
            .reshape(&[mg, mg])?
            .as_dtype(Dtype::Float32)?;
        Ok(TrackerMask {
            low_res,
            iou: iv[best],
            object_score: obj_score
                .reshape(&[-1])?
                .as_dtype(Dtype::Float32)?
                .as_slice::<f32>()[0],
        })
    }

    /// Decode a no-prompt (memory-conditioned) tracking frame — `_run_single_frame_inference` with no
    /// point/mask inputs (the video-propagation path). `conditioned_embedding`: NHWC `[1, 72, 72, 256]`
    /// from [`Self::prepare_memory_conditioned_features`] (already memory-conditioned — **no**
    /// `no_memory_embedding` add). `high_res`: the frame's `[feat_s0, feat_s1]`. Empty-point prompt;
    /// **multimask decode** (`_use_multimask` is true for tracking frames — `multimask_min_pt_num ≤ 0`)
    /// → the best-predicted-IoU candidate selects both the output mask AND the object-pointer token.
    /// Absent objects (`object_score ≤ 0`) get `NO_OBJ_SCORE` masks. Returns the low-res/high-res masks
    /// + object pointer + score.
    pub fn decode_tracked_frame(
        &self,
        conditioned_embedding: &Array,
        high_res: &[Array; 2],
    ) -> Result<TrackerFrameOutput> {
        let g = conditioned_embedding.shape()[1];
        let (sparse, dense) = self.prompt.encode_empty_point(g)?;
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        // multimask=true ⇒ the decoder returns the 3 multimask candidates (masks[1,3,mg,mg],
        // ious[1,3]) + the full mask_tokens[1,4,256].
        let (masks, ious, obj_score, mask_tokens) = self.decoder.forward(
            conditioned_embedding,
            &image_pe,
            &sparse,
            &dense,
            high_res,
            true,
        )?;
        let object_score = obj_score
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()[0];
        // best-IoU candidate over the 3 multimask outputs.
        let iv = ious
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec();
        let best = argmax(&iv) as i32;
        let mg = masks.shape()[2];
        let best_mask = take1(&masks, best, 1)?.reshape(&[1, 1, mg, mg])?;
        // is_obj_appearing: present keeps the mask, absent replaces it with NO_OBJ_SCORE everywhere.
        let low_res = if object_score > 0.0 {
            best_mask
        } else {
            broadcast_to(Array::from_f32(NO_OBJ_SCORE), &[1, 1, mg, mg])?
        };
        // high-res: separable bilinear 288→1008 (align_corners=False), for the memory encoder.
        let m = low_res.reshape(&[mg, mg])?;
        let up = bilinear_resize_matrix(mg, INPUT_SIZE as i32);
        let high = up
            .matmul(&m)?
            .matmul(&up.transpose_axes(&[1, 0])?)?
            .reshape(&[1, 1, INPUT_SIZE as i32, INPUT_SIZE as i32])?;
        // object pointer: multimask ⇒ sam_output_token is the best-IoU candidate (token `best + 1` in
        // the full set, since the multimask slice drops token 0).
        let token = take1(&mask_tokens, best + 1, 1)?;
        let object_pointer = self.compute_object_pointer(&token, object_score)?;
        Ok(TrackerFrameOutput {
            low_res,
            high_res: high,
            object_pointer,
            object_score,
        })
    }

    /// Decode a mask-conditioned (detection-seeded) frame — `_use_mask_as_output` — producing the
    /// high-res mask logits (for the memory encoder) + the object pointer (for the bank). `raw_embedding`:
    /// NHWC `[1, 72, 72, 256]` — the **raw** frame image embedding ([`Self::encode_frame`] first output;
    /// the mask-as-output path does **not** add `no_memory_embedding` and does **not** memory-condition).
    /// `mask_det`: NHWC `[1, 288, 288, 1]` binary detection mask (`det ≥ 0.5`). The detector mask is
    /// upsampled to 1008² and turned into `±` logits (`·20 −10`); the object pointer comes from the SAM
    /// decoder prompted with `mask_embed(mask_downsample(mask))` (multimask ⇒ best-IoU token), gated by
    /// the decoder's object score and then by mask presence (`any(mask > 0)`).
    ///
    /// `low_res` here is the simple 288² detection-mask logits (the reference's antialiased
    /// 1008→288 downsample is not bit-reproduced — it is unused for output/memory on this path; the
    /// memory encoder consumes `high_res`).
    pub fn decode_mask_conditioning_frame(
        &self,
        raw_embedding: &Array,
        high_res: &[Array; 2],
        mask_det: &Array,
    ) -> Result<TrackerFrameOutput> {
        let g = raw_embedding.shape()[1];
        let in_sz = mask_det.shape()[1];
        let big = INPUT_SIZE as i32;
        // detection mask presence (drives the object score + outer pointer gate).
        let is_appearing = mask_det
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .iter()
            .any(|&v| v > 0.0);
        // upsample the binary mask to 1008² and turn it into ± logits.
        let md = mask_det.reshape(&[in_sz, in_sz])?;
        let up = bilinear_resize_matrix(in_sz, big);
        let mask_big = up.matmul(&md)?.matmul(&up.transpose_axes(&[1, 0])?)?; // [1008,1008]
        let high = add(
            &mask_big.multiply(Array::from_f32(SIGMOID_SCALE_FOR_MEM))?,
            Array::from_f32(SIGMOID_BIAS_FOR_MEM),
        )?
        .reshape(&[1, 1, big, big])?;
        // mask prompt: mask_downsample (k4s4 → 252²) then bilinear up to mask_input_size 288².
        let mask_big_nhwc = mask_big.reshape(&[1, big, big, 1])?;
        let mds = conv2d(
            &mask_big_nhwc,
            &self.mask_downsample.0,
            Some(&self.mask_downsample.1),
            4,
            0,
        )?; // [1,252,252,1]
        let ds = mds.shape()[1];
        let mds2 = mds.reshape(&[ds, ds])?;
        let up2 = bilinear_resize_matrix(ds, MASK_INPUT_SIZE);
        let mask_288 = up2
            .matmul(&mds2)?
            .matmul(&up2.transpose_axes(&[1, 0])?)?
            .reshape(&[1, MASK_INPUT_SIZE, MASK_INPUT_SIZE, 1])?;
        let (sparse, dense) = self.prompt.encode_mask_prompt(&mask_288)?;
        // decoder on the RAW image embedding (no no_memory_embedding), multimask=true.
        let image_pe = self.image_pe_embed.dense_pe(g)?;
        let (masks, ious, obj_score, mask_tokens) =
            self.decoder
                .forward(raw_embedding, &image_pe, &sparse, &dense, high_res, true)?;
        let decoder_score = obj_score
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()[0];
        let iv = ious
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec();
        let best = argmax(&iv) as i32;
        // object pointer: best-IoU token, inner-gated by the decoder score, outer-gated by mask presence.
        let token = take1(&mask_tokens, best + 1, 1)?;
        let inner = self.compute_object_pointer(&token, decoder_score)?;
        let object_pointer = if is_appearing {
            inner
        } else {
            self.no_object_pointer.reshape(&[1, HIDDEN])?
        };
        let object_score = if is_appearing {
            SIGMOID_SCALE_FOR_MEM + SIGMOID_BIAS_FOR_MEM // 20 − 10 = 10
        } else {
            SIGMOID_BIAS_FOR_MEM // −10
        };
        let _ = masks; // mask output on this path comes from the detection, not the decoder.
        let low_res = add(
            &md.multiply(Array::from_f32(SIGMOID_SCALE_FOR_MEM))?,
            Array::from_f32(SIGMOID_BIAS_FOR_MEM),
        )?
        .reshape(&[1, 1, in_sz, in_sz])?;
        Ok(TrackerFrameOutput {
            low_res,
            high_res: high,
            object_pointer,
            object_score,
        })
    }

    /// End-to-end single-frame: pixels + box (1008-input space) → best low-res mask.
    pub fn segment(&self, pixel_values: &Array, box_xyxy_1008: [f32; 4]) -> Result<TrackerMask> {
        let (emb, high_res) = self.encode_frame(pixel_values)?;
        self.segment_encoded(&emb, &high_res, box_xyxy_1008)
    }
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// `get_1d_sine_pe` (modeling_sam3_tracker_video.py ~1592): 1-D sinusoidal positional encoding for a
/// set of (already-normalized) positions. `dim` must be even; the first `dim/2` outputs are the sines
/// and the last `dim/2` the cosines, with paired frequencies `temperature^(2·(j//2)/(dim/2))`.
/// Host-computed (the inputs are a handful of object-pointer temporal offsets). Returns `[P, dim]`.
fn sine_pe_1d(positions: &[f32], dim: usize, temperature: f32) -> Array {
    let pe_dim = dim / 2;
    let mut out = vec![0f32; positions.len() * dim];
    for (i, &p) in positions.iter().enumerate() {
        for j in 0..pe_dim {
            let dim_t = temperature.powf((2 * (j / 2)) as f32 / pe_dim as f32);
            let v = p / dim_t;
            out[i * dim + j] = v.sin();
            out[i * dim + pe_dim + j] = v.cos();
        }
    }
    Array::from_slice(&out, &[positions.len() as i32, dim as i32])
}
