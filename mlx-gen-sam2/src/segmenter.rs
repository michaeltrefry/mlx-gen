//! SAM2 box-prompt segmenter — orchestration of encoder → prompt encoder → mask decoder, plus the
//! preprocessing and mask post-processing (`mlx_sam/models/segmenter.py` image path + the
//! `_Sam2Segmenter` contract). Image mode only (no memory); the video layer is Phase B.
//!
//! Pipeline: preprocess (1024² square stretch, /255, ImageNet norm) → encode → project high-res
//! features → encode the box (two corner points + a pad point) → decode (3 candidate masks) →
//! argmax IoU → bilinear-upsample the best low-res logits to the original size → threshold > 0 →
//! binary `L` mask.

use mlx_rs::ops::add;
use mlx_rs::{Array, Dtype};

use mlx_gen::image::resize_bilinear_u8;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::{Sam2ImageEncoderConfig, Sam2ModelSize};
use crate::image_encoder::Sam2ImageEncoder;
use crate::sam_heads::{MaskDecoder, PromptEncoder};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const SIZE: usize = 1024;

/// SAM2 preprocessing: resize an RGB8 HWC image to 1024² (square stretch, bilinear), `/255`, then
/// ImageNet-normalize → NCHW `[1, 3, 1024, 1024]` f32. Byte-faithful to the spike's `preprocess`.
pub fn preprocess(rgb: &[u8], in_h: usize, in_w: usize) -> Array {
    assert_eq!(rgb.len(), in_h * in_w * 3, "rgb must be HWC RGB8");
    let resized = resize_bilinear_u8(rgb, in_h, in_w, SIZE, SIZE); // HWC f32 in [0,255]
    let mut nchw = vec![0f32; 3 * SIZE * SIZE];
    for y in 0..SIZE {
        for x in 0..SIZE {
            for c in 0..3 {
                let v = resized[(y * SIZE + x) * 3 + c] / 255.0;
                nchw[c * SIZE * SIZE + y * SIZE + x] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
            }
        }
    }
    Array::from_slice(&nchw, &[1, 3, SIZE as i32, SIZE as i32])
}

/// Scale a box from original pixel space to the 1024² input space (`orig · 1024/W, · 1024/H`).
pub fn box_to_1024(box_xyxy: [f32; 4], orig_w: u32, orig_h: u32) -> [f32; 4] {
    let sx = SIZE as f32 / orig_w as f32;
    let sy = SIZE as f32 / orig_h as f32;
    [
        box_xyxy[0] * sx,
        box_xyxy[1] * sy,
        box_xyxy[2] * sx,
        box_xyxy[3] * sy,
    ]
}

/// Bilinear upsample a single-channel f32 map (`align_corners=False`, half-pixel centers — matches
/// `torch.nn.functional.interpolate(mode="bilinear")`, the spike's mask post-process).
fn upsample_bilinear_f32(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; out_h * out_w];
    let sy = in_h as f32 / out_h as f32;
    let sx = in_w as f32 / out_w as f32;
    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).clamp(0.0, (in_h - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;
        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).clamp(0.0, (in_w - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;
            let v00 = src[y0 * in_w + x0];
            let v01 = src[y0 * in_w + x1];
            let v10 = src[y1 * in_w + x0];
            let v11 = src[y1 * in_w + x1];
            let top = v00 + (v01 - v00) * wx;
            let bot = v10 + (v11 - v10) * wx;
            out[oy * out_w + ox] = top + (bot - top) * wy;
        }
    }
    out
}

/// SAM2 image segmenter: encoder + prompt encoder + mask decoder (+ the `no_mem_embed` bias the
/// image path adds to the vision features).
pub struct Sam2Segmenter {
    encoder: Sam2ImageEncoder,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    no_mem_embed: Array, // [1, 1, 256]
}

impl Sam2Segmenter {
    pub fn from_weights(w: &Weights, cfg: &Sam2ImageEncoderConfig) -> Result<Self> {
        Ok(Self {
            encoder: Sam2ImageEncoder::from_weights(w, cfg)?,
            prompt: PromptEncoder::from_weights(w, "sam_prompt_encoder")?,
            decoder: MaskDecoder::from_weights(w, "sam_mask_decoder")?,
            no_mem_embed: w.require("no_mem_embed")?.clone(),
        })
    }

    /// Convenience: build for a named model size.
    pub fn from_weights_for_size(w: &Weights, size: Sam2ModelSize) -> Result<Self> {
        Self::from_weights(w, &Sam2ImageEncoderConfig::for_size(size))
    }

    /// Box prompt (already in 1024² space) → the best low-res mask logits `[256, 256]` (f32) plus
    /// its predicted IoU. The encoder/decoder run on GPU; the caller upsamples + thresholds.
    pub fn best_low_res_mask(
        &self,
        pixel_values: &Array,
        box_xyxy_1024: [f32; 4],
    ) -> Result<(Array, f32)> {
        let encoded = self.encoder.forward(pixel_values)?;
        let high_res = self.decoder.project_high_res(&encoded.backbone_fpn)?;
        let (sparse, dense) = self.prompt.encode_box(box_xyxy_1024)?;

        // image path: vision_features + no_mem_embed (broadcast over the spatial grid).
        let image_embeddings = add(
            &encoded.vision_features,
            &self.no_mem_embed.reshape(&[1, 256, 1, 1])?,
        )?;
        let image_pe = self.prompt.dense_pe()?;

        let (masks, ious) = self.decoder.forward(
            &image_embeddings,
            &image_pe,
            &sparse,
            &dense,
            true, // multimask_output → 3 candidates
            &high_res,
        )?; // masks [1,3,256,256], ious [1,3]

        // argmax IoU on the host (3 values) → select the best candidate.
        let iou_vec = ious
            .reshape(&[-1])?
            .as_dtype(Dtype::Float32)?
            .as_slice::<f32>()
            .to_vec();
        let best = iou_vec
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as i32)
            .unwrap_or(0);
        let best_mask = masks
            .take_axis(Array::from_int(best), 1)?
            .reshape(&[256, 256])?;
        Ok((best_mask, iou_vec[best as usize]))
    }

    /// Full box-prompt segmentation: box in 1024² space → binary `L` mask at the original size.
    /// Values are 0 / 255 (`u8`), shape `[orig_h, orig_w]`.
    pub fn segment_from_pixels(
        &self,
        pixel_values: &Array,
        box_xyxy_1024: [f32; 4],
        orig_w: u32,
        orig_h: u32,
    ) -> Result<Array> {
        let (low, _iou) = self.best_low_res_mask(pixel_values, box_xyxy_1024)?;
        let logits = low.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let up = upsample_bilinear_f32(&logits, 256, 256, orig_h as usize, orig_w as usize);
        let mask: Vec<u8> = up.iter().map(|&v| if v > 0.0 { 255 } else { 0 }).collect();
        Ok(Array::from_slice(&mask, &[orig_h as i32, orig_w as i32]))
    }

    /// End-to-end: an RGB8 HWC image + a box in original pixel space → binary `L` mask `[H, W]`.
    pub fn segment(&self, rgb: &[u8], in_h: u32, in_w: u32, box_xyxy: [f32; 4]) -> Result<Array> {
        let pixels = preprocess(rgb, in_h as usize, in_w as usize);
        let box_1024 = box_to_1024(box_xyxy, in_w, in_h);
        self.segment_from_pixels(&pixels, box_1024, in_w, in_h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_to_1024_scales_each_axis() {
        // 640×480 image, box → scaled by (1024/640, 1024/480) = (1.6, ~2.133).
        let b = box_to_1024([100.0, 50.0, 200.0, 150.0], 640, 480);
        assert!((b[0] - 160.0).abs() < 1e-3);
        assert!((b[1] - 106.6667).abs() < 1e-2);
        assert!((b[2] - 320.0).abs() < 1e-3);
        assert!((b[3] - 320.0).abs() < 1e-2);
    }

    #[test]
    fn preprocess_shape_and_normalization() {
        // A flat mid-gray image: every normalized value = (128/255 - mean)/std, per channel.
        let (h, w) = (8usize, 5usize);
        let rgb = vec![128u8; h * w * 3];
        let pix = preprocess(&rgb, h, w);
        assert_eq!(pix.shape(), &[1, 3, 1024, 1024]);
        let v = pix.as_slice::<f32>();
        let expect_c0 = (128.0 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
        assert!(
            (v[0] - expect_c0).abs() < 1e-4,
            "channel-0 norm {} vs {expect_c0}",
            v[0]
        );
        // channel 1 starts at offset 1024*1024.
        let expect_c1 = (128.0 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
        assert!((v[1024 * 1024] - expect_c1).abs() < 1e-4);
    }

    #[test]
    fn upsample_bilinear_doubles_and_interpolates() {
        // 2×2 → 4×4: corners preserved, midpoints interpolated (monotone), align_corners=False.
        let src = vec![0.0f32, 1.0, 1.0, 2.0];
        let up = upsample_bilinear_f32(&src, 2, 2, 4, 4);
        assert_eq!(up.len(), 16);
        // top-left stays the min, bottom-right the max; overall monotone increasing corner-to-corner.
        assert!(up[0] <= up[15]);
        assert!((up[0] - 0.0).abs() < 1e-6);
        assert!((up[15] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn threshold_is_strictly_positive() {
        // The post-process keeps logits strictly > 0 (matches the spike's `> 0` threshold).
        let logits = [-0.001f32, 0.0, 0.001];
        let bin: Vec<u8> = logits
            .iter()
            .map(|&v| if v > 0.0 { 255 } else { 0 })
            .collect();
        assert_eq!(bin, vec![0, 0, 255]);
    }
}
