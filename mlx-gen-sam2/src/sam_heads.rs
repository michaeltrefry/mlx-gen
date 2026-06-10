//! SAM2 prompt encoder + two-way mask decoder — port of `mlx_sam/models/sam_heads.py`.
//!
//! The box→mask half of the segmenter that sits on top of the image encoder ([`crate::image_encoder`]):
//!   * [`PromptEncoder`] — turns a box (two corner "points", labels 2/3) + a pad point into the
//!     sparse prompt tokens, plus the dense (no-mask) embedding and the grid position encoding.
//!   * [`MaskDecoder`] — a two-way (token↔image) transformer + transposed-conv upscaling +
//!     hypernetwork mask MLPs producing 4 candidate masks, their IoU predictions, and the object
//!     score. The box-prompt segmenter takes `multimask_output` (3 masks) and argmaxes the IoU.
//!
//! Layout: feature maps stay NCHW (the reference's canonical layout); convs/transposed-convs are
//! wrapped to transpose to MLX's NHWC and back. Token tensors `[b, n, c]` are layout-agnostic.

use std::f32::consts::PI;
use std::sync::OnceLock;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::nn::relu;
use mlx_rs::ops::{
    self, broadcast_to, concatenate_axis, matmul, mean_axes, multiply, rsqrt, sigmoid, stack_axis,
};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// LayerNorm epsilon (`nn.LayerNorm` and `LayerNorm2d` both use 1e-6).
const EPS: f32 = 1e-6;

fn join(prefix: &str, leaf: &str) -> String {
    if prefix.is_empty() {
        leaf.to_string()
    } else {
        format!("{prefix}.{leaf}")
    }
}

/// 2-D conv over an **NCHW** input with an OHWI weight (+ bias): transpose → `conv2d` → transpose.
fn conv2d_nchw(x: &Array, w: &Array, b: &Array, stride: i32, pad: i32) -> Result<Array> {
    let y = conv2d(&x.transpose_axes(&[0, 2, 3, 1])?, w, Some(b), stride, pad)?;
    Ok(y.transpose_axes(&[0, 3, 1, 2])?)
}

/// Transposed 2-D conv over an **NCHW** input with an OHWI weight (+ bias), `stride`=kernel.
fn conv_transpose2d_nchw(x: &Array, w: &Array, b: &Array, stride: i32) -> Result<Array> {
    let y = ops::conv_transpose2d(
        x.transpose_axes(&[0, 2, 3, 1])?,
        w,
        (stride, stride),
        None,
        None,
        None,
        None,
    )?;
    let y = ops::add(&y, b)?; // bias over the last (channel) axis, NHWC
    Ok(y.transpose_axes(&[0, 3, 1, 2])?)
}

/// `LayerNorm2d`: normalize an **NCHW** tensor over the channel axis (per spatial position).
fn layer_norm_2d(x: &Array, weight: &Array, bias: &Array) -> Result<Array> {
    let mean = mean_axes(x, &[1], true)?;
    let centered = ops::subtract(x, &mean)?;
    let var = mean_axes(&ops::square(&centered)?, &[1], true)?;
    let normed = multiply(&centered, &rsqrt(&ops::add(&var, Array::from_f32(EPS))?)?)?;
    let w = weight.reshape(&[1, -1, 1, 1])?;
    let b = bias.reshape(&[1, -1, 1, 1])?;
    Ok(ops::add(&multiply(&normed, &w)?, &b)?)
}

/// An N-layer MLP (`SamMLP`): linear · (relu|gelu) between layers, optional final sigmoid.
pub(crate) struct SamMlp {
    layers: Vec<(Array, Array)>,
    gelu: bool,
    sigmoid_output: bool,
}

impl SamMlp {
    pub(crate) fn from_weights(
        w: &Weights,
        prefix: &str,
        num_layers: usize,
        gelu: bool,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let layers = (0..num_layers)
            .map(|i| -> Result<(Array, Array)> {
                Ok((
                    w.require(&join(prefix, &format!("layers.{i}.weight")))?
                        .clone(),
                    w.require(&join(prefix, &format!("layers.{i}.bias")))?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            layers,
            gelu,
            sigmoid_output,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let mut x = x.clone();
        let n = self.layers.len();
        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = linear(&x, w, b)?;
            if i < n - 1 {
                x = if self.gelu {
                    gelu_exact(&x)?
                } else {
                    relu(&x)?
                };
            }
        }
        if self.sigmoid_output {
            x = sigmoid(&x)?;
        }
        Ok(x)
    }
}

/// Gaussian random position encoding (`PositionEmbeddingRandom`) for point coords + the dense grid PE.
struct PositionEmbeddingRandom {
    /// `positional_encoding_gaussian_matrix`, `[2, num_pos_feats]`.
    gaussian: Array,
}

impl PositionEmbeddingRandom {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            gaussian: w
                .require(&join(prefix, "positional_encoding_gaussian_matrix"))?
                .clone(),
        })
    }

    /// `coords`: `[..., 2]` in [0,1] → `[..., 2·num_pos_feats]` (sin‖cos of `2π·((2c-1)·G)`).
    fn pe_encoding(&self, coords: &Array) -> Result<Array> {
        let two = Array::from_f32(2.0);
        let c = ops::subtract(&multiply(coords, &two)?, Array::from_f32(1.0))?; // 2c - 1
        let c = matmul(&c, &self.gaussian)?; // [..., npf]
        let c = multiply(&c, Array::from_f32(2.0 * PI))?;
        Ok(concatenate_axis(&[&ops::sin(&c)?, &ops::cos(&c)?], -1)?)
    }

    /// Dense grid PE for an `h×w` feature map → NCHW `[1, 2·npf, h, w]`.
    fn dense_pe(&self, h: i32, w: i32) -> Result<Array> {
        let ys: Vec<f32> = (0..h).map(|i| (i as f32 + 0.5) / h as f32).collect();
        let xs: Vec<f32> = (0..w).map(|j| (j as f32 + 0.5) / w as f32).collect();
        let y = broadcast_to(Array::from_slice(&ys, &[h, 1]), &[h, w])?;
        let x = broadcast_to(Array::from_slice(&xs, &[1, w]), &[h, w])?;
        let coords = stack_axis(&[&x, &y], -1)?; // [h, w, 2]
        let pe = self.pe_encoding(&coords)?; // [h, w, 2·npf]
        Ok(pe.transpose_axes(&[2, 0, 1])?.expand_dims(0)?) // [1, 2·npf, h, w]
    }

    /// Point coords in pixel space → `[..., 2·npf]` (normalize by `image_size`, then encode).
    fn forward_with_coords(&self, coords: &Array, image_size: i32) -> Result<Array> {
        let scale = Array::from_f32(image_size as f32);
        self.pe_encoding(&ops::divide(coords, &scale)?)
    }
}

/// Prompt encoder: box/point sparse tokens + dense (no-mask) embedding (`PromptEncoder`).
pub struct PromptEncoder {
    pe_layer: PositionEmbeddingRandom,
    /// `point_embeddings[0..4]`: 0/1 = neg/pos point, 2/3 = box top-left / bottom-right.
    point_embeddings: Vec<Array>,
    not_a_point_embed: Array,
    no_mask_embed: Array,
    /// Mask-prompt downscaling (`mask_downscaling_{0,1,3,4,6}`): conv k2/s2, LN2d, conv k2/s2,
    /// LN2d, conv k1 — turns a `[b,1,256,256]` mask prompt into the dense `[b,256,64,64]` embedding.
    mask_downscaling: MaskDownscaling,
    embed_dim: i32,
    image_embedding: i32,  // square grid side (64)
    input_image_size: i32, // 1024
    /// Cached dense grid PE — a constant of `image_embedding` recomputed per frame otherwise (F-167).
    dense_pe_cache: OnceLock<Array>,
}

/// The five mask-prompt downscaling layers (`PromptEncoder._embed_masks`).
struct MaskDownscaling {
    conv0: (Array, Array), // k2 s2: 1→4
    norm1: (Array, Array), // LayerNorm2d(4)
    conv3: (Array, Array), // k2 s2: 4→16
    norm4: (Array, Array), // LayerNorm2d(16)
    conv6: (Array, Array), // k1: 16→256
}

impl PromptEncoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let point_embeddings = (0..4)
            .map(|i| -> Result<Array> {
                Ok(w.require(&p(&format!("point_embeddings.{i}.weight")))?
                    .clone())
            })
            .collect::<Result<Vec<_>>>()?;
        let wb = |name: &str| ln(w, &p(name));
        Ok(Self {
            pe_layer: PositionEmbeddingRandom::from_weights(w, &p("pe_layer"))?,
            point_embeddings,
            not_a_point_embed: w.require(&p("not_a_point_embed.weight"))?.clone(),
            no_mask_embed: w.require(&p("no_mask_embed.weight"))?.clone(),
            mask_downscaling: MaskDownscaling {
                conv0: wb("mask_downscaling_0")?,
                norm1: wb("mask_downscaling_1")?,
                conv3: wb("mask_downscaling_3")?,
                norm4: wb("mask_downscaling_4")?,
                conv6: wb("mask_downscaling_6")?,
            },
            embed_dim: 256,
            image_embedding: 64,
            input_image_size: 1024,
            dense_pe_cache: OnceLock::new(),
        })
    }

    /// Dense grid position encoding `[1, embed_dim, grid, grid]` (the decoder's `image_pe`). Constant
    /// for a loaded model, so it is built once and cached — the image segmenter and every frame of the
    /// video loop reuse the same tensor instead of recomputing it (F-167).
    pub fn dense_pe(&self) -> Result<Array> {
        if let Some(pe) = self.dense_pe_cache.get() {
            return Ok(pe.clone());
        }
        let pe = self
            .pe_layer
            .dense_pe(self.image_embedding, self.image_embedding)?;
        // A race-loser's `set` returns Err; both computed the identical constant, so ignore it.
        let _ = self.dense_pe_cache.set(pe.clone());
        Ok(pe)
    }

    /// Embed `points` `[b, n, 2]` (pixel space) with int `labels` `[b, n]`, padding one
    /// "not a point" if `pad`. Returns sparse tokens `[b, n(+1), embed_dim]`.
    fn embed_points(&self, points: &Array, labels: &Array, pad: bool) -> Result<Array> {
        let mut points = ops::add(points, Array::from_f32(0.5))?;
        let mut labels = labels.clone();
        if pad {
            let b = points.shape()[0];
            let pad_pt = ops::zeros::<f32>(&[b, 1, 2])?;
            let pad_lbl = ops::multiply(&ops::ones::<i32>(&[b, 1])?, Array::from_int(-1))?;
            points = concatenate_axis(&[&points, &pad_pt], 1)?;
            labels = concatenate_axis(&[&labels, &pad_lbl], 1)?;
        }
        let pe = self
            .pe_layer
            .forward_with_coords(&points, self.input_image_size)?; // [b,n,256]
        let lab = labels.expand_dims(-1)?; // [b,n,1]
        let pick = |val: i32| -> Result<Array> { Ok(lab.eq(Array::from_int(val))?) };

        // label == -1 ⇒ the "not a point" embedding (padding).
        let na = broadcast_to(&self.not_a_point_embed, pe.shape())?;
        let mut out = ops::r#where(&pick(-1)?, &na, &pe)?;
        // labels 0..3 ⇒ add the corresponding point embedding.
        for i in 0..4 {
            let added = ops::add(&out, &self.point_embeddings[i as usize])?;
            out = ops::r#where(&pick(i)?, &added, &out)?;
        }
        Ok(out)
    }

    /// Box (1024-space corners) → sparse tokens `[1, 3, embed_dim]` (corners labelled 2/3 + pad),
    /// plus the dense no-mask embedding `[1, embed_dim, grid, grid]`.
    pub fn encode_box(&self, box_xyxy: [f32; 4]) -> Result<(Array, Array)> {
        let [x1, y1, x2, y2] = box_xyxy;
        let points = Array::from_slice(&[x1, y1, x2, y2], &[1, 2, 2]);
        let labels = Array::from_slice(&[2i32, 3], &[1, 2]);
        self.encode(Some(&points), Some(&labels), None)
    }

    /// General prompt encode (`PromptEncoder.__call__`): optional `points` `[b,n,2]` (1024-space)
    /// with int `labels` `[b,n]`, and an optional dense `mask_input` `[b,1,256,256]`. Returns the
    /// sparse tokens `[b, n(+1), embed_dim]` (one pad point appended when points are present) and the
    /// dense embedding `[b, embed_dim, grid, grid]` (from the mask prompt, else the no-mask embed).
    pub(crate) fn encode(
        &self,
        points: Option<&Array>,
        labels: Option<&Array>,
        masks: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let bs = points
            .map(|p| p.shape()[0])
            .or_else(|| masks.map(|m| m.shape()[0]))
            .unwrap_or(1);
        let sparse = match (points, labels) {
            (Some(p), Some(l)) => self.embed_points(p, l, true)?,
            _ => ops::zeros::<f32>(&[bs, 0, self.embed_dim])?,
        };
        let dense = match masks {
            Some(m) => self.embed_masks(m)?,
            None => self.dense_embedding(bs)?,
        };
        Ok((sparse, dense))
    }

    /// `_embed_masks`: a `[b,1,256,256]` mask prompt → dense `[b,256,64,64]` embedding.
    fn embed_masks(&self, masks: &Array) -> Result<Array> {
        let d = &self.mask_downscaling;
        let x = conv2d_nchw(masks, &d.conv0.0, &d.conv0.1, 2, 0)?;
        let x = gelu_exact(&layer_norm_2d(&x, &d.norm1.0, &d.norm1.1)?)?;
        let x = conv2d_nchw(&x, &d.conv3.0, &d.conv3.1, 2, 0)?;
        let x = gelu_exact(&layer_norm_2d(&x, &d.norm4.0, &d.norm4.1)?)?;
        conv2d_nchw(&x, &d.conv6.0, &d.conv6.1, 1, 0)
    }

    /// The dense embedding when no mask prompt is given: `no_mask_embed` broadcast over the grid.
    fn dense_embedding(&self, bs: i32) -> Result<Array> {
        let e = self.no_mask_embed.reshape(&[1, self.embed_dim, 1, 1])?;
        Ok(broadcast_to(
            &e,
            &[
                bs,
                self.embed_dim,
                self.image_embedding,
                self.image_embedding,
            ],
        )?)
    }
}

/// Multi-head attention with an optional q/k/v down-projection (`Attention`).
struct Attention {
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    o_w: Array,
    o_b: Array,
    num_heads: i32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, num_heads: i32) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            q_w: w.require(&p("q_proj.weight"))?.clone(),
            q_b: w.require(&p("q_proj.bias"))?.clone(),
            k_w: w.require(&p("k_proj.weight"))?.clone(),
            k_b: w.require(&p("k_proj.bias"))?.clone(),
            v_w: w.require(&p("v_proj.weight"))?.clone(),
            v_b: w.require(&p("v_proj.bias"))?.clone(),
            o_w: w.require(&p("out_proj.weight"))?.clone(),
            o_b: w.require(&p("out_proj.bias"))?.clone(),
            num_heads,
        })
    }

    fn forward(&self, q: &Array, k: &Array, v: &Array) -> Result<Array> {
        let sep = |x: &Array| -> Result<Array> {
            let sh = x.shape();
            let (b, n, c) = (sh[0], sh[1], sh[2]);
            Ok(x.reshape(&[b, n, self.num_heads, c / self.num_heads])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = sep(&linear(q, &self.q_w, &self.q_b)?)?;
        let k = sep(&linear(k, &self.k_w, &self.k_b)?)?;
        let v = sep(&linear(v, &self.v_w, &self.v_b)?)?;
        let scale = 1.0 / (q.shape()[3] as f32).sqrt();
        let out = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let sh = out.shape();
        let (b, h, n, c) = (sh[0], sh[1], sh[2], sh[3]);
        let out = out.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, n, h * c])?;
        linear(&out, &self.o_w, &self.o_b)
    }
}

/// One two-way block: token self-attn, token→image cross-attn, MLP, image→token cross-attn.
struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: (Array, Array),
    cross_t2i: Attention,
    norm2: (Array, Array),
    mlp: SamMlp,
    norm3: (Array, Array),
    cross_i2t: Attention,
    norm4: (Array, Array),
    skip_first_layer_pe: bool,
}

fn ln(w: &Weights, prefix: &str) -> Result<(Array, Array)> {
    Ok((
        w.require(&join(prefix, "weight"))?.clone(),
        w.require(&join(prefix, "bias"))?.clone(),
    ))
}

impl TwoWayAttentionBlock {
    fn from_weights(w: &Weights, prefix: &str, skip_first_layer_pe: bool) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            self_attn: Attention::from_weights(w, &p("self_attn"), 8)?,
            norm1: ln(w, &p("norm1"))?,
            cross_t2i: Attention::from_weights(w, &p("cross_attn_token_to_image"), 8)?,
            norm2: ln(w, &p("norm2"))?,
            mlp: SamMlp::from_weights(w, &p("mlp"), 2, false, false)?,
            norm3: ln(w, &p("norm3"))?,
            cross_i2t: Attention::from_weights(w, &p("cross_attn_image_to_token"), 8)?,
            norm4: ln(w, &p("norm4"))?,
            skip_first_layer_pe,
        })
    }

    fn forward(
        &self,
        queries: &Array,
        keys: &Array,
        query_pe: &Array,
        key_pe: &Array,
    ) -> Result<(Array, Array)> {
        let lnf = |x: &Array, n: &(Array, Array)| layer_norm(x, Some(&n.0), Some(&n.1), EPS);

        let mut queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)?
        } else {
            let q = ops::add(queries, query_pe)?;
            ops::add(queries, &self.self_attn.forward(&q, &q, queries)?)?
        };
        queries = lnf(&queries, &self.norm1)?;

        let q = ops::add(&queries, query_pe)?;
        let k = ops::add(keys, key_pe)?;
        queries = lnf(
            &ops::add(&queries, &self.cross_t2i.forward(&q, &k, keys)?)?,
            &self.norm2,
        )?;
        queries = lnf(
            &ops::add(&queries, &self.mlp.forward(&queries)?)?,
            &self.norm3,
        )?;

        let q = ops::add(&queries, query_pe)?;
        let k = ops::add(keys, key_pe)?;
        let keys = lnf(
            &ops::add(keys, &self.cross_i2t.forward(&k, &q, &queries)?)?,
            &self.norm4,
        )?;
        Ok((queries, keys))
    }
}

/// The two-way transformer (`TwoWayTransformer`): 2 blocks + a final token→image attention.
struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn: Attention,
    norm_final: (Array, Array),
}

impl TwoWayTransformer {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        Ok(Self {
            layers: vec![
                TwoWayAttentionBlock::from_weights(w, &p("layers.0"), true)?,
                TwoWayAttentionBlock::from_weights(w, &p("layers.1"), false)?,
            ],
            final_attn: Attention::from_weights(w, &p("final_attn_token_to_image"), 8)?,
            norm_final: ln(w, &p("norm_final_attn"))?,
        })
    }

    /// `image_embedding`/`image_pe`: NCHW `[b,c,h,w]`; `point_embedding`: `[b, n, c]`.
    fn forward(
        &self,
        image_embedding: &Array,
        image_pe: &Array,
        point_embedding: &Array,
    ) -> Result<(Array, Array)> {
        let sh = image_embedding.shape();
        let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);
        let flat = |x: &Array| -> Result<Array> {
            Ok(x.reshape(&[b, c, h * w])?.transpose_axes(&[0, 2, 1])?)
        };
        let image_embedding = flat(image_embedding)?;
        let image_pe = flat(image_pe)?;

        let mut queries = point_embedding.clone();
        let mut keys = image_embedding;
        for layer in &self.layers {
            let (q, k) = layer.forward(&queries, &keys, point_embedding, &image_pe)?;
            queries = q;
            keys = k;
        }
        let q = ops::add(&queries, point_embedding)?;
        let k = ops::add(&keys, &image_pe)?;
        queries = layer_norm(
            &ops::add(&queries, &self.final_attn.forward(&q, &k, &keys)?)?,
            Some(&self.norm_final.0),
            Some(&self.norm_final.1),
            EPS,
        )?;
        Ok((queries, keys))
    }
}

/// The full mask decoder (`MaskDecoder`): two-way transformer + upscaling + hypernetwork masks.
pub struct MaskDecoder {
    transformer: TwoWayTransformer,
    iou_token: Array,
    mask_tokens: Array,
    obj_score_token: Array,
    upscale0: (Array, Array), // ConvTranspose2d 256→64
    upscale1: (Array, Array), // LayerNorm2d(64)
    upscale3: (Array, Array), // ConvTranspose2d 64→32
    conv_s0: (Array, Array),  // 1×1 256→32 (high-res feat 0)
    conv_s1: (Array, Array),  // 1×1 256→64 (high-res feat 1)
    hypernet: Vec<SamMlp>,
    iou_head: SamMlp,
    obj_score_head: SamMlp,
    dynamic_multimask_via_stability: bool,
    stability_delta: f32,
    stability_thresh: f32,
}

impl MaskDecoder {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);
        let conv = |name: &str| -> Result<(Array, Array)> {
            Ok((
                w.require(&p(&format!("{name}.weight")))?.clone(),
                w.require(&p(&format!("{name}.bias")))?.clone(),
            ))
        };
        let hypernet = (0..4)
            .map(|i| {
                SamMlp::from_weights(
                    w,
                    &p(&format!("output_hypernetworks_mlps.{i}")),
                    3,
                    false,
                    false,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            transformer: TwoWayTransformer::from_weights(w, &p("transformer"))?,
            iou_token: w.require(&p("iou_token.weight"))?.clone(),
            mask_tokens: w.require(&p("mask_tokens.weight"))?.clone(),
            obj_score_token: w.require(&p("obj_score_token.weight"))?.clone(),
            upscale0: conv("output_upscaling_0")?,
            upscale1: ln(w, &p("output_upscaling_1"))?,
            upscale3: conv("output_upscaling_3")?,
            conv_s0: conv("conv_s0")?,
            conv_s1: conv("conv_s1")?,
            hypernet,
            iou_head: SamMlp::from_weights(w, &p("iou_prediction_head"), 3, false, true)?,
            obj_score_head: SamMlp::from_weights(w, &p("pred_obj_score_head"), 3, false, false)?,
            dynamic_multimask_via_stability: true,
            stability_delta: 0.05,
            stability_thresh: 0.98,
        })
    }

    /// Project the two finest FPN maps to the high-res features the upscaler adds (`conv_s0/s1`).
    pub fn project_high_res(&self, fpn: &[Array]) -> Result<Vec<Array>> {
        let s0 = conv2d_nchw(&fpn[0], &self.conv_s0.0, &self.conv_s0.1, 1, 0)?;
        let s1 = conv2d_nchw(&fpn[1], &self.conv_s1.0, &self.conv_s1.1, 1, 0)?;
        Ok(vec![s0, s1])
    }

    /// Core forward (pre mask-selection). Returns `(masks[b,4,H,W], iou_pred[b,4],
    /// mask_tokens_out[b,4,256], object_score_logits[b,1])`.
    fn predict_masks(
        &self,
        image_embeddings: &Array,
        image_pe: &Array,
        sparse: &Array,
        dense: &Array,
        high_res: &[Array],
    ) -> Result<(Array, Array, Array, Array)> {
        // tokens = [obj_score, iou, mask_tokens(4)] ++ sparse prompt tokens.
        let output_tokens = concatenate_axis(
            &[&self.obj_score_token, &self.iou_token, &self.mask_tokens],
            0,
        )?;
        let bs = sparse.shape()[0];
        let nt = output_tokens.shape()[0];
        let output_tokens = broadcast_to(&output_tokens.expand_dims(0)?, &[bs, nt, 256])?;
        let tokens = concatenate_axis(&[&output_tokens, sparse], 1)?;

        let src = ops::add(image_embeddings, dense)?;
        let pos_src = broadcast_to(image_pe, src.shape())?;
        let sh = src.shape();
        let (b, c, h, w) = (sh[0], sh[1], sh[2], sh[3]);

        let (hs, src_tokens) = self.transformer.forward(&src, &pos_src, &tokens)?;
        let iou_token_out = hs.take_axis(Array::from_int(1), 1)?; // [b,1,256] -> squeeze later
        let iou_token_out = iou_token_out.reshape(&[b, 256])?;
        let mask_tokens_out = slice_tokens(&hs, 2, 6)?; // [b,4,256]

        // back to NCHW image grid
        let src = src_tokens
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[b, c, h, w])?;

        let feat_s0 = &high_res[0];
        let feat_s1 = &high_res[1];
        let x = conv_transpose2d_nchw(&src, &self.upscale0.0, &self.upscale0.1, 2)?;
        let x = gelu_exact(&layer_norm_2d(
            &ops::add(&x, feat_s1)?,
            &self.upscale1.0,
            &self.upscale1.1,
        )?)?;
        let x = conv_transpose2d_nchw(&x, &self.upscale3.0, &self.upscale3.1, 2)?;
        let upscaled = gelu_exact(&ops::add(&x, feat_s0)?)?; // [b,32,H,W]

        // hypernetwork: per mask-token MLP → [b,4,32]; mask = hyper @ upscaled.
        let hyper = stack_axis(
            &(0..4)
                .map(|i| {
                    let tok = mask_tokens_out
                        .take_axis(Array::from_int(i), 1)?
                        .reshape(&[b, 256])?;
                    self.hypernet[i as usize].forward(&tok)
                })
                .collect::<Result<Vec<_>>>()?
                .iter()
                .collect::<Vec<_>>(),
            1,
        )?; // [b,4,32]
        let su = upscaled.shape();
        let (uc, uh, uw) = (su[1], su[2], su[3]);
        let masks =
            matmul(&hyper, &upscaled.reshape(&[b, uc, uh * uw])?)?.reshape(&[b, -1, uh, uw])?;

        let iou_pred = self.iou_head.forward(&iou_token_out)?; // [b,4]
        let obj_tok = hs.take_axis(Array::from_int(0), 1)?.reshape(&[b, 256])?;
        let object_score_logits = self.obj_score_head.forward(&obj_tok)?; // [b,1]
        Ok((masks, iou_pred, mask_tokens_out, object_score_logits))
    }

    /// Full decoder (`MaskDecoder.__call__`). `multimask_output` true ⇒ the 3 multi-masks
    /// (tokens 1-3) + their IoUs + their SAM tokens; false ⇒ the single mask (dynamic stability
    /// fallback) + token 0. Returns `(masks, ious, sam_tokens, object_score_logits)` — `sam_tokens`
    /// and `object_score_logits` are what the video predictor needs to build the object pointer.
    pub(crate) fn predict(
        &self,
        image_embeddings: &Array,
        image_pe: &Array,
        sparse: &Array,
        dense: &Array,
        multimask_output: bool,
        high_res: &[Array],
    ) -> Result<(Array, Array, Array, Array)> {
        let (masks, iou_pred, tokens, obj_logits) =
            self.predict_masks(image_embeddings, image_pe, sparse, dense, high_res)?;

        // Zero-out (to -1024) the masks when the object score says "no object".
        let is_obj = obj_logits.gt(Array::from_f32(0.0))?; // [b,1]
        let is_obj4 = is_obj.reshape(&[is_obj.shape()[0], 1, 1, 1])?;
        let neg = ops::full::<f32>(masks.shape(), &Array::from_f32(-1024.0))?;
        let masks = ops::r#where(&is_obj4, &masks, &neg)?;

        let (masks_out, iou_out, sam_tokens) = if multimask_output {
            (
                slice_masks(&masks, 1, 4)?,
                slice_iou(&iou_pred, 1, 4)?,
                slice_tokens(&tokens, 1, 4)?,
            )
        } else if self.dynamic_multimask_via_stability {
            let (m, i) = self.dynamic_multimask(&masks, &iou_pred)?;
            (m, i, slice_tokens(&tokens, 0, 1)?)
        } else {
            (
                slice_masks(&masks, 0, 1)?,
                slice_iou(&iou_pred, 0, 1)?,
                slice_tokens(&tokens, 0, 1)?,
            )
        };
        Ok((masks_out, iou_out, sam_tokens, obj_logits))
    }

    /// Image-path convenience: drop the SAM tokens / object score, return just `(masks, ious)`.
    pub fn forward(
        &self,
        image_embeddings: &Array,
        image_pe: &Array,
        sparse: &Array,
        dense: &Array,
        multimask_output: bool,
        high_res: &[Array],
    ) -> Result<(Array, Array)> {
        let (masks, ious, _tokens, _obj) = self.predict(
            image_embeddings,
            image_pe,
            sparse,
            dense,
            multimask_output,
            high_res,
        )?;
        Ok((masks, ious))
    }

    fn stability_scores(&self, masks: &Array) -> Result<Array> {
        let sh = masks.shape();
        let flat = masks.reshape(&[sh[0], sh[1], -1])?;
        let area_i = mean_count_gt(&flat, self.stability_delta)?;
        let area_u = mean_count_gt(&flat, -self.stability_delta)?;
        let ratio = ops::divide(&area_i, &area_u)?;
        let ones = ops::ones::<f32>(area_u.shape())?;
        Ok(ops::r#where(
            &area_u.gt(Array::from_f32(0.0))?,
            &ratio,
            &ones,
        )?)
    }

    fn dynamic_multimask(&self, masks: &Array, iou_pred: &Array) -> Result<(Array, Array)> {
        let multi_logits = slice_masks(masks, 1, 4)?; // [b,3,H,W]
        let multi_iou = slice_iou(iou_pred, 1, 4)?; // [b,3]
        let best = ops::indexing::argmax_axis(&multi_iou, 1, false)?; // [b]
        let arange = Array::from_slice(&[0i32, 1, 2], &[1, 3]);
        let one_hot = arange.eq(&best.expand_dims(1)?)?.as_dtype(masks.dtype())?; // [b,3]
        let oh_m = one_hot.reshape(&[one_hot.shape()[0], 3, 1, 1])?;
        let best_logits = ops::sum_axes(&multiply(&multi_logits, &oh_m)?, &[1], true)?;
        let best_iou = ops::sum_axes(&multiply(&multi_iou, &one_hot)?, &[1], true)?;

        let single_logits = slice_masks(masks, 0, 1)?;
        let single_iou = slice_iou(iou_pred, 0, 1)?;
        let stable = self
            .stability_scores(&single_logits)?
            .ge(Array::from_f32(self.stability_thresh))?; // [b,1]
        let stable_m = stable.reshape(&[stable.shape()[0], 1, 1, 1])?;
        Ok((
            ops::r#where(&stable_m, &single_logits, &best_logits)?,
            ops::r#where(&stable, &single_iou, &best_iou)?,
        ))
    }
}

/// `count(x > thresh)` over the last axis as f32 (helper for the stability score).
fn mean_count_gt(x: &Array, thresh: f32) -> Result<Array> {
    let mask = x
        .gt(Array::from_f32(thresh))?
        .as_dtype(mlx_rs::Dtype::Float32)?;
    Ok(ops::sum_axes(&mask, &[-1], false)?)
}

/// `hs[:, start..end, :]` over the token axis.
fn slice_tokens(hs: &Array, start: i32, end: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
    Ok(hs.take_axis(&idx, 1)?)
}

/// `masks[:, start..end, :, :]` over the mask axis.
fn slice_masks(masks: &Array, start: i32, end: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
    Ok(masks.take_axis(&idx, 1)?)
}

/// `iou[:, start..end]` over the mask axis.
fn slice_iou(iou: &Array, start: i32, end: i32) -> Result<Array> {
    let idx = Array::from_slice(&(start..end).collect::<Vec<i32>>(), &[end - start]);
    Ok(iou.take_axis(&idx, 1)?)
}
