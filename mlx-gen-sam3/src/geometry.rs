//! SAM3 geometry/exemplar prompt encoder — the box/point **PVS** (Promptable Visual Segmentation)
//! prompt path (epic 4910, sc-4923). Ports `Sam3GeometryEncoder` directly from the public
//! Apache-2.0 `transformers` reference.
//!
//! A box prompt (normalized cxcywh) is encoded three ways and summed: a direct linear projection of
//! the coordinates, ROI-align pooling of the 72² FPN feature at the box (`roi_align` then a 7×7
//! conv), and a sine position encoding of the box center (+ raw h/w). A positive/negative label
//! embedding is added, a CLS token is appended, and the result is refined by 3 pre-norm transformer
//! layers that cross-attend to the 72² vision feature. The output prompt tokens `[1, N+1, 256]` are
//! concatenated with the text features and fed to the detector + mask decoder as
//! `combined_prompt_features` (see [`crate::model`]).
//!
//! `roi_align` has no native mlx-rs op: it is realized as a host-built bilinear sampling matrix —
//! faithful to `torchvision.ops.roi_align` (`spatial_scale=1`, `sampling_ratio=-1`, `aligned=False`)
//! — applied on the CPU stream so the pooling + conv contraction keep full fp32 precision (Metal
//! matmul is reduced-precision).

use std::f32::consts::PI;

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, concatenate_axis, matmul_device};
use mlx_rs::{Array, Dtype, StreamOrDevice};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::Sam3GeometryConfig;
use crate::detr::{Attn, Ffn};

const SCALE_2PI: f32 = 2.0 * PI;

fn join(prefix: &str, leaf: &str) -> String {
    format!("{prefix}.{leaf}")
}

fn ln(x: &Array, w: &Array, b: &Array, eps: f32) -> Result<Array> {
    Ok(layer_norm(x, Some(w), Some(b), eps)?)
}

/// One pre-norm geometry-encoder layer (`Sam3GeometryEncoderLayer`): prompt self-attn → vision
/// cross-attn (key = vision + pos, value = vision) → ReLU FFN, each residual-added.
struct GeometryLayer {
    ln1_w: Array,
    ln1_b: Array,
    self_attn: Attn,
    ln2_w: Array,
    ln2_b: Array,
    cross_attn: Attn,
    ln3_w: Array,
    ln3_b: Array,
    ffn: Ffn,
    eps: f32,
}

impl GeometryLayer {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3GeometryConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        let (nh, hd) = (cfg.num_attention_heads, cfg.head_dim());
        Ok(Self {
            ln1_w: g("layer_norm1.weight")?,
            ln1_b: g("layer_norm1.bias")?,
            self_attn: Attn::from_dims(w, &join(prefix, "self_attn"), nh, hd)?,
            ln2_w: g("layer_norm2.weight")?,
            ln2_b: g("layer_norm2.bias")?,
            cross_attn: Attn::from_dims(w, &join(prefix, "cross_attn"), nh, hd)?,
            ln3_w: g("layer_norm3.weight")?,
            ln3_b: g("layer_norm3.bias")?,
            ffn: Ffn::from_weights(w, &join(prefix, "mlp"))?,
            eps: cfg.layer_norm_eps,
        })
    }

    /// `prompt`: `[1, P, C]`; `vision`: `[1, H·W, C]` (raw 72² feature, flattened); `vision_pos`:
    /// `[1, H·W, C]`. All prompt tokens are valid (PVS path), so attention runs unmasked.
    fn forward(&self, prompt: &Array, vision: &Array, vision_pos: &Array) -> Result<Array> {
        let h = ln(prompt, &self.ln1_w, &self.ln1_b, self.eps)?;
        let a = self.self_attn.forward(&h, &h, &h, None)?;
        let x = add(prompt, &a)?;

        let h = ln(&x, &self.ln2_w, &self.ln2_b, self.eps)?;
        let key = add(vision, vision_pos)?;
        let a = self.cross_attn.forward(&h, &key, vision, None)?;
        let x = add(&x, &a)?;

        let h = ln(&x, &self.ln3_w, &self.ln3_b, self.eps)?;
        let a = self.ffn.forward(&h)?;
        Ok(add(&x, &a)?)
    }
}

/// SAM3 geometry/exemplar prompt encoder (`Sam3GeometryEncoder`).
pub struct Sam3GeometryEncoder {
    label_embed: Array, // [2, C]
    cls_embed: Array,   // [1, C]
    boxes_direct_w: Array,
    boxes_direct_b: Array, // Linear(4, C)
    boxes_pool_w: Array,   // Conv2d(C, C, roi_size) weight [C, C, R, R]
    boxes_pool_b: Array,   // [C]
    boxes_pos_w: Array,
    boxes_pos_b: Array, // Linear(C + 2, C)
    vision_ln_w: Array,
    vision_ln_b: Array,
    final_proj_w: Array,
    final_proj_b: Array,
    prompt_ln_w: Array,
    prompt_ln_b: Array,
    layers: Vec<GeometryLayer>,
    output_ln_w: Array,
    output_ln_b: Array,
    cfg: Sam3GeometryConfig,
}

impl Sam3GeometryEncoder {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3GeometryConfig) -> Result<Self> {
        let g = |n: &str| -> Result<Array> { Ok(w.require(&join(prefix, n))?.clone()) };
        let layers = (0..cfg.num_layers)
            .map(|i| GeometryLayer::from_weights(w, &join(prefix, &format!("layers.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            label_embed: g("label_embed.weight")?,
            cls_embed: g("cls_embed.weight")?,
            boxes_direct_w: g("boxes_direct_project.weight")?,
            boxes_direct_b: g("boxes_direct_project.bias")?,
            boxes_pool_w: g("boxes_pool_project.weight")?,
            boxes_pool_b: g("boxes_pool_project.bias")?,
            boxes_pos_w: g("boxes_pos_enc_project.weight")?,
            boxes_pos_b: g("boxes_pos_enc_project.bias")?,
            vision_ln_w: g("vision_layer_norm.weight")?,
            vision_ln_b: g("vision_layer_norm.bias")?,
            final_proj_w: g("final_proj.weight")?,
            final_proj_b: g("final_proj.bias")?,
            prompt_ln_w: g("prompt_layer_norm.weight")?,
            prompt_ln_b: g("prompt_layer_norm.bias")?,
            layers,
            output_ln_w: g("output_layer_norm.weight")?,
            output_ln_b: g("output_layer_norm.bias")?,
            cfg: cfg.clone(),
        })
    }

    /// Encode box prompts into prompt tokens.
    ///
    /// * `boxes`: `[1, N, 4]` normalized cxcywh ∈ [0, 1] (relative to the model input).
    /// * `box_labels`: length `N` (`1` = positive, `0` = negative).
    /// * `vision`: the 72² FPN feature, **NHWC** `[1, H, W, C]` (the level the detector consumes).
    /// * `vision_pos`: the matching flattened sine position embedding `[1, H·W, C]`.
    ///
    /// Returns the geometry prompt tokens `[1, N + 1, C]` (boxes followed by the CLS token); all
    /// tokens are valid.
    pub fn forward(
        &self,
        boxes: &Array,
        box_labels: &[i32],
        vision: &Array,
        vision_pos: &Array,
    ) -> Result<Array> {
        let sh = vision.shape();
        let (h, w) = (sh[1], sh[2]);
        let c = self.cfg.hidden_size;
        let n = boxes.shape()[1];

        // (1) direct projection of the box coordinates
        let direct = linear(boxes, &self.boxes_direct_w, &self.boxes_direct_b)?; // [1, N, C]

        // (2) ROI-align pooling of the channel-normalized 72² feature, then the 7×7 conv
        let norm_feat = ln(
            vision,
            &self.vision_ln_w,
            &self.vision_ln_b,
            self.cfg.layer_norm_eps,
        )?; // [1, H, W, C]
        let boxes_host = boxes.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec(); // N·4 cxcywh
        let pooled = self.roi_pool(&boxes_host, n, &norm_feat, h, w)?; // [1, N, C]

        // (3) sine position encoding of the box center (+ raw h/w), projected to C
        let pos_enc = box_pos_encoding(&boxes_host, n, c); // [1, N, C+2]
        let pos = linear(&pos_enc, &self.boxes_pos_w, &self.boxes_pos_b)?; // [1, N, C]

        // label (positive/negative) embedding + the three box encodings
        let lbl_idx = Array::from_slice(box_labels, &[n]);
        let label = self
            .label_embed
            .take_axis(&lbl_idx, 0)?
            .reshape(&[1, n, c])?;
        let boxes_embed = add(&add(&add(&direct, &pooled)?, &pos)?, &label)?; // [1, N, C]

        // append the always-valid CLS token
        let cls = self.cls_embed.reshape(&[1, 1, c])?;
        let prompt = concatenate_axis(&[&boxes_embed, &cls], 1)?; // [1, N+1, C]
        let prompt = ln(
            &linear(&prompt, &self.final_proj_w, &self.final_proj_b)?,
            &self.prompt_ln_w,
            &self.prompt_ln_b,
            self.cfg.layer_norm_eps,
        )?;

        // refine with transformer layers cross-attending to the raw vision feature
        let vision_flat = vision.reshape(&[1, h * w, c])?;
        let mut x = prompt;
        for layer in &self.layers {
            x = layer.forward(&x, &vision_flat, vision_pos)?;
        }
        ln(
            &x,
            &self.output_ln_w,
            &self.output_ln_b,
            self.cfg.layer_norm_eps,
        )
    }

    /// `roi_align` (bilinear ROI pool, `roi_size`²) + the `boxes_pool_project` 7×7 conv, fused as
    /// two CPU-stream matmuls: a host-built sampling matrix `[N·R², H·W]` gathers the pooled grid,
    /// then the conv weight `[C, C·R²]` contracts it to `[1, N, C]`.
    fn roi_pool(
        &self,
        boxes_host: &[f32],
        n: i32,
        norm_feat: &Array,
        h: i32,
        w: i32,
    ) -> Result<Array> {
        let c = self.cfg.hidden_size;
        let r = self.cfg.roi_size;
        let cpu = StreamOrDevice::cpu();

        let s = roi_align_matrix(boxes_host, n as usize, h, w, r);
        let s = Array::from_slice(&s, &[n * r * r, h * w]);
        let vflat = norm_feat.reshape(&[h * w, c])?;
        // sampled grid: [N·R², C] → [N, C, R²] (the conv's (in, kH, kW) order) → [N, C·R²]
        let sampled = matmul_device(&s, &vflat, &cpu)?
            .reshape(&[n, r * r, c])?
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[n, c * r * r])?;
        // boxes_pool_project conv weight [C_out, C_in, R, R] → [C_out, C_in·R²]
        let wflat = self.boxes_pool_w.reshape(&[c, c * r * r])?;
        let pooled = matmul_device(&sampled, &wflat.transpose_axes(&[1, 0])?, &cpu)?; // [N, C]
        Ok(add(&pooled, &self.boxes_pool_b)?.reshape(&[1, n, c])?)
    }
}

/// Sine position encoding of box centers (+ raw height/width), `[1, N, C+2]`. Mirrors
/// `Sam3GeometryEncoder._encode_box_coordinates`: `cat(pos_y, pos_x, height, width)` where each
/// `pos_*` is the `sin(even)/cos(odd)` interleave of `center·2π / dim_t`.
fn box_pos_encoding(boxes_cxcywh: &[f32], n: i32, c: i32) -> Array {
    let npf = (c / 2) as usize; // 128
    let dim_t: Vec<f32> = (0..npf)
        .map(|i| 10000f32.powf(2.0 * ((i / 2) as f32) / npf as f32))
        .collect();
    let total = 2 * npf + 2; // 258
    let mut out = vec![0f32; n as usize * total];
    let enc = |v: f32, dst: &mut [f32]| {
        let e = v * SCALE_2PI;
        for j in 0..npf / 2 {
            dst[2 * j] = (e / dim_t[2 * j]).sin();
            dst[2 * j + 1] = (e / dim_t[2 * j + 1]).cos();
        }
    };
    for bi in 0..n as usize {
        let (cx, cy, bw, bh) = (
            boxes_cxcywh[bi * 4],
            boxes_cxcywh[bi * 4 + 1],
            boxes_cxcywh[bi * 4 + 2],
            boxes_cxcywh[bi * 4 + 3],
        );
        let base = bi * total;
        enc(cy, &mut out[base..base + npf]); // pos_y first
        enc(cx, &mut out[base + npf..base + 2 * npf]); // then pos_x
        out[base + 2 * npf] = bh; // raw box height
        out[base + 2 * npf + 1] = bw; // raw box width
    }
    Array::from_slice(&out, &[1, n, total as i32])
}

/// Host-built `torchvision.ops.roi_align` sampling matrix `[N·R², H·W]` for boxes in normalized
/// cxcywh. Each output cell row holds the bilinear interpolation weights (averaged over the adaptive
/// sample grid) over the flattened H·W feature. `spatial_scale=1`, `sampling_ratio=-1`,
/// `aligned=False`.
fn roi_align_matrix(boxes_cxcywh: &[f32], n: usize, h: i32, w: i32, r: i32) -> Vec<f32> {
    let hw = (h * w) as usize;
    let o = r as usize;
    let (hf, wf) = (h as f32, w as f32);
    let mut s = vec![0f32; n * o * o * hw];
    for bi in 0..n {
        let (cx, cy, bw, bh) = (
            boxes_cxcywh[bi * 4],
            boxes_cxcywh[bi * 4 + 1],
            boxes_cxcywh[bi * 4 + 2],
            boxes_cxcywh[bi * 4 + 3],
        );
        // normalized cxcywh → xyxy → feature coordinates (× W, H)
        let start_w = (cx - 0.5 * bw) * wf;
        let start_h = (cy - 0.5 * bh) * hf;
        let roi_w = ((cx + 0.5 * bw) * wf - start_w).max(1.0); // !aligned → min size 1
        let roi_h = ((cy + 0.5 * bh) * hf - start_h).max(1.0);
        let bin_w = roi_w / r as f32;
        let bin_h = roi_h / r as f32;
        let grid_w = (roi_w / r as f32).ceil().max(1.0) as i32;
        let grid_h = (roi_h / r as f32).ceil().max(1.0) as i32;
        let count = (grid_h * grid_w).max(1) as f32;
        for ph in 0..o {
            for pw in 0..o {
                let row = (bi * o * o + ph * o + pw) * hw;
                let srow = &mut s[row..row + hw];
                for iy in 0..grid_h {
                    let y = start_h + ph as f32 * bin_h + (iy as f32 + 0.5) * bin_h / grid_h as f32;
                    for ix in 0..grid_w {
                        let x =
                            start_w + pw as f32 * bin_w + (ix as f32 + 0.5) * bin_w / grid_w as f32;
                        bilinear_acc(srow, h, w, y, x, 1.0 / count);
                    }
                }
            }
        }
    }
    s
}

/// Accumulate one bilinear sample's four corner weights into a `[H·W]` sampling row. Out-of-range
/// samples contribute nothing; edge handling matches torchvision's `bilinear_interpolate`.
fn bilinear_acc(row: &mut [f32], h: i32, w: i32, y: f32, x: f32, weight: f32) {
    let (hf, wf) = (h as f32, w as f32);
    if y < -1.0 || y > hf || x < -1.0 || x > wf {
        return;
    }
    let mut y = if y <= 0.0 { 0.0 } else { y };
    let mut x = if x <= 0.0 { 0.0 } else { x };
    let mut y_low = y as i32;
    let mut x_low = x as i32;
    let y_high;
    let x_high;
    if y_low >= h - 1 {
        y_low = h - 1;
        y_high = h - 1;
        y = y_low as f32;
    } else {
        y_high = y_low + 1;
    }
    if x_low >= w - 1 {
        x_low = w - 1;
        x_high = w - 1;
        x = x_low as f32;
    } else {
        x_high = x_low + 1;
    }
    let (ly, lx) = (y - y_low as f32, x - x_low as f32);
    let (hy, hx) = (1.0 - ly, 1.0 - lx);
    let idx = |yy: i32, xx: i32| (yy * w + xx) as usize;
    row[idx(y_low, x_low)] += hy * hx * weight;
    row[idx(y_low, x_high)] += hy * lx * weight;
    row[idx(y_high, x_low)] += ly * hx * weight;
    row[idx(y_high, x_high)] += ly * lx * weight;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roi_align_matrix_rows_are_normalized() {
        // A box covering the whole 4×4 feature → every output cell's weights sum to 1.
        let boxes = [0.5f32, 0.5, 1.0, 1.0]; // cxcywh covering [0,1]²
        let s = roi_align_matrix(&boxes, 1, 4, 4, 7);
        let hw = 16;
        for cell in 0..49 {
            let sum: f32 = s[cell * hw..(cell + 1) * hw].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "cell {cell} weight sum {sum}");
        }
    }

    #[test]
    fn bilinear_acc_hits_exact_pixel() {
        // Sampling exactly at an interior pixel center puts all weight on that pixel.
        let mut row = vec![0f32; 16];
        bilinear_acc(&mut row, 4, 4, 2.0, 1.0, 1.0);
        assert!((row[2 * 4 + 1] - 1.0).abs() < 1e-6);
        let total: f32 = row.iter().sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn box_pos_encoding_has_expected_shape_and_tail() {
        // Tail two columns are the raw box height then width.
        let boxes = [0.3f32, 0.4, 0.2, 0.6];
        let enc = box_pos_encoding(&boxes, 1, 256);
        assert_eq!(enc.shape(), &[1, 1, 258]);
        let v = enc.as_slice::<f32>();
        assert!((v[256] - 0.6).abs() < 1e-6, "height tail");
        assert!((v[257] - 0.2).abs() < 1e-6, "width tail");
    }
}
