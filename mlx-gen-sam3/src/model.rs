//! SAM3 still-image concept segmenter — assembles the PE vision encoder (A), CLIP text encoder (B),
//! DETR detector (C), and mask head (D) into the end-to-end `Sam3Model` image path (epic 4910).
//!
//! `pixel_values[1,3,1008,1008] + "person" → per-instance masks`. Mirrors `Sam3Model.forward` for
//! both the text-only **PCS** path ([`Sam3ImageSegmenter::forward`]) and the box-prompted **PVS**
//! path ([`Sam3ImageSegmenter::forward_with_boxes`], sc-4923) — the latter prepends geometry prompt
//! tokens to the text features as the reference's `combined_prompt_features`.

use std::rc::Rc;

use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::{Sam3DetrConfig, Sam3GeometryConfig, Sam3TextConfig, Sam3VisionConfig};
use crate::detr::sine_position_embedding_flat;
use crate::mask::{post_process_instances, Instance, Sam3MaskHead};
use crate::vision::Backbone;
use crate::{Sam3Detector, Sam3GeometryEncoder, Sam3TextEncoder, Sam3VisionEncoder};

/// Full raw outputs of the image segmenter (pre-post-process).
pub struct SegmentationOutput {
    /// `[1, Q]` concept logits.
    pub pred_logits: Array,
    /// `[1, Q, 4]` boxes xyxy ∈ [0, 1].
    pub pred_boxes: Array,
    /// `[1, 1]` presence logit.
    pub presence_logits: Array,
    /// `[1, Q, 288, 288]` per-query mask logits.
    pub pred_masks: Array,
    /// `[1, 288, 288, 1]` semantic-segmentation logits.
    pub semantic_seg: Array,
}

/// End-to-end SAM3 still-image concept segmenter.
pub struct Sam3ImageSegmenter {
    vision: Sam3VisionEncoder,
    text: Sam3TextEncoder,
    geometry: Sam3GeometryEncoder,
    detector: Sam3Detector,
    mask_head: Sam3MaskHead,
}

impl Sam3ImageSegmenter {
    /// Load every stage from a `facebook/sam3` weight map.
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let vision = Sam3VisionEncoder::from_weights(
            w,
            "detector_model.vision_encoder",
            &Sam3VisionConfig::sam3(),
        )?;
        Self::from_weights_with_vision(w, vision)
    }

    /// Load the segmenter reusing an already-loaded (and possibly shared) PE [`Backbone`]. Lets the
    /// video model share one backbone between this detector segmenter and the tracker (F-028).
    pub(crate) fn from_weights_with_backbone(w: &Weights, backbone: Rc<Backbone>) -> Result<Self> {
        let vision = Sam3VisionEncoder::from_weights_with_backbone(
            w,
            "detector_model.vision_encoder",
            &Sam3VisionConfig::sam3(),
            backbone,
        )?;
        Self::from_weights_with_vision(w, vision)
    }

    fn from_weights_with_vision(w: &Weights, vision: Sam3VisionEncoder) -> Result<Self> {
        let detr_cfg = Sam3DetrConfig::sam3();
        Ok(Self {
            vision,
            text: Sam3TextEncoder::from_weights(
                w,
                "detector_model.text_encoder.text_model",
                "detector_model.text_projection",
                &Sam3TextConfig::sam3(),
            )?,
            geometry: Sam3GeometryEncoder::from_weights(
                w,
                "detector_model.geometry_encoder",
                &Sam3GeometryConfig::sam3(),
            )?,
            detector: Sam3Detector::from_weights(w, "detector_model", &detr_cfg)?,
            mask_head: Sam3MaskHead::from_weights(w, "detector_model", &detr_cfg)?,
        })
    }

    /// Affine-quantize the segmenter's linear projections to `bits` (`8` = Q8 near-lossless, `4` =
    /// Q4) — the PE ViT backbone, CLIP text tower + projection, DETR encoder/decoder + scoring, the
    /// geometry encoder, and the mask head's prompt attention + embedder. Convs, GroupNorms,
    /// embeddings, and the small/odd projections stay dense (see [`crate::quantize_linear`]).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.vision.quantize(bits)?;
        self.quantize_except_backbone(bits)
    }

    /// Quantize everything **except** the PE backbone (the text tower + projection, geometry encoder,
    /// DETR detector, and mask head). The video model calls this after quantizing the single shared
    /// backbone once, so the backbone isn't quantized twice (F-028). (The FPN neck is conv-only and
    /// stays dense, so the vision encoder has nothing else to quantize here.)
    pub(crate) fn quantize_except_backbone(&mut self, bits: i32) -> Result<()> {
        self.text.quantize(bits)?;
        self.geometry.quantize(bits)?;
        self.detector.quantize(bits)?;
        self.mask_head.quantize(bits)?;
        Ok(())
    }

    /// The shared PE [`Backbone`] handle (clone of the `Rc`). Used by the F-028 sharing test to
    /// assert pointer-identity with the tracker's backbone.
    #[cfg(test)]
    pub(crate) fn vision_backbone_rc(&self) -> Rc<Backbone> {
        self.vision.backbone_rc()
    }

    /// Replace the vision encoder's PE backbone with a (typically pre-quantized, shared) one.
    pub(crate) fn set_vision_backbone(&mut self, backbone: Rc<Backbone>) {
        self.vision.set_backbone(backbone);
    }

    /// `pixel_values`: NCHW `[1, 3, 1008, 1008]`; `input_ids`: `[1, 32]`; `text_mask`: per-token
    /// validity (`1`/`0`). Text-only PCS — runs the full detector + mask head.
    pub fn forward(
        &self,
        pixel_values: &Array,
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<SegmentationOutput> {
        let fpn = self.vision.forward(pixel_values)?; // NHWC [288²,144²,72²,36²]
        let text = self.text.forward(input_ids, text_mask)?; // [1,32,256]
        self.detect_and_segment(&fpn, &text, text_mask)
    }

    /// Box-prompted **PVS** path (sc-4923): the geometry encoder turns `boxes` (normalized cxcywh,
    /// `[1, N, 4]`) + `box_labels` (length `N`, `1`=positive/`0`=negative) into `N + 1` prompt
    /// tokens, which are concatenated *after* the text features as the reference's
    /// `combined_prompt_features` and drive the detector + mask head.
    pub fn forward_with_boxes(
        &self,
        pixel_values: &Array,
        input_ids: &Array,
        text_mask: &[i32],
        boxes: &Array,
        box_labels: &[i32],
    ) -> Result<SegmentationOutput> {
        let fpn = self.vision.forward(pixel_values)?;
        let text = self.text.forward(input_ids, text_mask)?; // [1,32,256]

        // geometry prompt tokens cross-attend to the 72² feature (fpn[2]) + its sine pos embed.
        let sh = fpn[2].shape();
        let vision_pos =
            sine_position_embedding_flat(sh[1], sh[2], Sam3GeometryConfig::sam3().hidden_size)?;
        let geo = self
            .geometry
            .forward(boxes, box_labels, &fpn[2], &vision_pos)?; // [1,N+1,256]

        let combined = concatenate_axis(&[&text, &geo], 1)?;
        let mut combined_mask = text_mask.to_vec();
        combined_mask.extend(std::iter::repeat_n(1, geo.shape()[1] as usize));

        self.detect_and_segment(&fpn, &combined, &combined_mask)
    }

    /// Shared detector + mask-head tail. `prompt`/`prompt_mask` are the text features (PCS) or the
    /// text⊕geometry `combined_prompt_features` (PVS).
    fn detect_and_segment(
        &self,
        fpn: &[Array],
        prompt: &Array,
        prompt_mask: &[i32],
    ) -> Result<SegmentationOutput> {
        let det = self.detector.forward(&fpn[2], prompt, prompt_mask)?;
        let backbone = [fpn[0].clone(), fpn[1].clone(), fpn[2].clone()];
        let masks = self.mask_head.forward(
            &det.query_hidden,
            &backbone,
            &det.encoder_hidden_states,
            prompt,
            &prompt_key_mask(prompt_mask),
        )?;
        Ok(SegmentationOutput {
            pred_logits: det.pred_logits,
            pred_boxes: det.pred_boxes,
            presence_logits: det.presence_logits,
            pred_masks: masks.pred_masks,
            semantic_seg: masks.semantic_seg,
        })
    }

    /// Convenience: full forward + instance post-process. `target_wh` is the original image size
    /// (for box scaling); masks come back at the native 288² resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn segment(
        &self,
        pixel_values: &Array,
        input_ids: &Array,
        text_mask: &[i32],
        target_wh: (f32, f32),
        threshold: f32,
        mask_threshold: f32,
    ) -> Result<Vec<Instance>> {
        let out = self.forward(pixel_values, input_ids, text_mask)?;
        post_process_instances(
            &out.pred_logits,
            &out.pred_boxes,
            &out.presence_logits,
            &out.pred_masks,
            target_wh,
            threshold,
            mask_threshold,
        )
    }

    /// Box-prompted PVS convenience: [`Self::forward_with_boxes`] + instance post-process.
    #[allow(clippy::too_many_arguments)]
    pub fn segment_with_boxes(
        &self,
        pixel_values: &Array,
        input_ids: &Array,
        text_mask: &[i32],
        boxes: &Array,
        box_labels: &[i32],
        target_wh: (f32, f32),
        threshold: f32,
        mask_threshold: f32,
    ) -> Result<Vec<Instance>> {
        let out = self.forward_with_boxes(pixel_values, input_ids, text_mask, boxes, box_labels)?;
        post_process_instances(
            &out.pred_logits,
            &out.pred_boxes,
            &out.presence_logits,
            &out.pred_masks,
            target_wh,
            threshold,
            mask_threshold,
        )
    }
}

/// Additive key-padding mask `[1, 1, 1, L]` (0 valid, −1e9 padded) for the mask head's prompt attn.
fn prompt_key_mask(text_mask: &[i32]) -> Array {
    let row: Vec<f32> = text_mask
        .iter()
        .map(|&m| if m == 1 { 0.0 } else { -1e9 })
        .collect();
    Array::from_slice(&row, &[1, 1, 1, row.len() as i32])
}
