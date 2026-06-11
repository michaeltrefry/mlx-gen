//! SAM2 **video propagation predictor** — port of `mlx_sam/video_predictor.py` +
//! `Sam2ImageSegmenter`'s video methods (`mlx_sam/models/segmenter.py`). This is the Phase-B layer
//! (sc-3714) that drives the memory bank ([`crate::memory`]) across a clip: prompt on a frame with a
//! box, then propagate temporally-consistent masks forward, conditioning each frame on the memory of
//! the frames behind it.
//!
//! Scope is the **single-object box-prompted** path the person pipeline needs (one tracked object —
//! `replace_person`'s selected person — so the per-object bookkeeping the reference keeps for
//! multi-object tracking collapses to a single output dict). Re-prompt / correction is supported via
//! [`Sam2VideoPredictor::add_correction_points`] (a point edit on an already-tracked frame threads
//! the previous mask back in as a dense prompt, exactly like the reference's `prev_low` path).
//!
//! The model-level methods ported here (the heavy lifting the predictor delegates to the model):
//!   * [`Sam2VideoModel::predict_from_encoded`] — prompt encode + mask decode, returning the low-res
//!     masks, IoUs, the **object pointer** (a 256-vec summary of the tracked object, gated by the
//!     object-score) and the object score itself.
//!   * [`Sam2VideoModel::encode_memory`] — turn a frame's `(features, predicted mask)` into the
//!     64-channel memory feature map stored in the bank.
//!   * [`Sam2VideoModel::condition_with_memories`] — assemble the memory bank (spatial memories with
//!     temporal-position encodings + object-pointer tokens) and run memory attention, producing the
//!     memory-conditioned features the decoder consumes on a tracked frame.
//!
//! Parity: `tests/video_parity.rs` (`#[ignore]`, real weights) runs the reference predictor's golden
//! clip and asserts the per-frame masks agree.
//!
//! **Direction — forward only (F-176).** The single public propagation entry point,
//! [`Sam2VideoPredictor::propagate`], tracks strictly *forward* and records every frame as forward
//! (`frames_tracked` ← `false`). The reverse-tracking plumbing it threads — the `track_in_reverse`
//! flag, the `frames_tracked` bool, and the reverse arms in `maskmem_prev_frame` and the
//! in-past / strided-frame arithmetic of [`Sam2VideoModel::condition_with_memories`] — mirrors the
//! reference (`propagate_in_video(reverse=True)`) and is covered by unit tests, but is **currently
//! unreachable from the public API**: nothing ever inserts `true`, and there is no
//! `propagate_reverse` method. The arithmetic is kept (not deleted) because it is the faithful
//! port and the seam a future reverse entry point plugs into — but exposing `propagate_reverse` as
//! production API is gated on a reverse end-to-end golden (the forward path is only verified via the
//! real-weights `video_parity.rs`; reverse needs the equivalent). Tracked on sc-4151; do not assume
//! reverse propagation is callable today.

use std::collections::BTreeMap;

use mlx_rs::ops::{
    self, add, broadcast_to, concatenate_axis, maximum, minimum, multiply, sigmoid, stack_axis,
    subtract,
};
use mlx_rs::{Array, Dtype};

use mlx_gen::nn::linear;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::{Sam2ImageEncoderConfig, Sam2ModelSize};
use crate::image_encoder::Sam2ImageEncoder;
use crate::memory::{MemoryAttention, MemoryEncoder};
use crate::sam_heads::{MaskDecoder, PromptEncoder, SamMlp};
use crate::segmenter::preprocess;

const IMAGE_SIZE: i32 = 1024;
const NUM_MASKMEM: i32 = 7;
const MAX_OBJ_PTRS: i32 = 16;
const MEM_STRIDE: i32 = 1; // memory_temporal_stride_for_eval

/// A frame's encoded image features (`Sam2ImageSegmenter.encode_image`).
#[derive(Clone)]
struct Encoded {
    /// Coarsest backbone-FPN map `[1, 256, 64, 64]` — the decoder image embedding + memory seq.
    vision_features: Array,
    /// Position encoding for the coarsest map (`vision_pos_enc[-1]`), `[1, 256, 64, 64]`.
    vision_pos_enc: Array,
    /// The two finest projected high-res features the mask upscaler adds.
    high_res: Vec<Array>,
}

/// One frame's tracked output (the reference's per-frame `current_out` dict, single object).
#[derive(Clone)]
struct FrameOut {
    /// Low-res mask logits `[1, 1, 256, 256]`.
    pred_masks: Array,
    /// Object pointer `[1, 256]`.
    obj_ptr: Array,
    /// Object-score logits `[1, 1]`.
    object_score_logits: Array,
    /// Encoded memory `[1, 64, 64, 64]` + its position encoding (None until memory-encoded).
    maskmem_features: Option<Array>,
    maskmem_pos_enc: Option<Array>,
}

/// Output of [`Sam2VideoModel::predict_from_encoded`].
struct PredictOut {
    low_res_masks: Array,       // [1, k, 256, 256] (k = 3 multimask, else 1)
    ious: Array,                // [1, k]
    obj_ptr: Array,             // [1, 256]
    object_score_logits: Array, // [1, 1]
}

/// The SAM2 model with the video heads (`Sam2ImageSegmenter`): image encoder + prompt encoder +
/// mask decoder + memory encoder/attention + the object-pointer projections and memory globals.
struct Sam2VideoModel {
    encoder: Sam2ImageEncoder,
    prompt: PromptEncoder,
    decoder: MaskDecoder,
    memory_encoder: MemoryEncoder,
    memory_attention: MemoryAttention,
    obj_ptr_proj: SamMlp,              // SamMLP(256,256,256,3) relu
    obj_ptr_tpos_proj: (Array, Array), // Linear 256→64
    no_mem_embed: Array,               // [1, 1, 256]
    no_obj_ptr: Array,                 // [1, 256]
    no_obj_embed_spatial: Array,       // [1, 64]
    maskmem_tpos_enc: Array,           // [7, 1, 1, 64]
}

impl Sam2VideoModel {
    fn from_weights(w: &Weights, cfg: &Sam2ImageEncoderConfig) -> Result<Self> {
        Ok(Self {
            encoder: Sam2ImageEncoder::from_weights(w, cfg)?,
            prompt: PromptEncoder::from_weights(w, "sam_prompt_encoder")?,
            decoder: MaskDecoder::from_weights(w, "sam_mask_decoder")?,
            memory_encoder: MemoryEncoder::from_weights(w, "memory_encoder")?,
            memory_attention: MemoryAttention::from_weights(w, "memory_attention")?,
            obj_ptr_proj: SamMlp::from_weights(w, "obj_ptr_proj", 3, false, false)?,
            obj_ptr_tpos_proj: (
                w.require("obj_ptr_tpos_proj.weight")?.clone(),
                w.require("obj_ptr_tpos_proj.bias")?.clone(),
            ),
            no_mem_embed: w.require("no_mem_embed")?.clone(),
            no_obj_ptr: w.require("no_obj_ptr")?.clone(),
            no_obj_embed_spatial: w.require("no_obj_embed_spatial")?.clone(),
            maskmem_tpos_enc: w.require("maskmem_tpos_enc")?.clone(),
        })
    }

    /// `encode_image`: pixels `[1,3,1024,1024]` → coarsest features + pos + high-res features.
    fn encode_image(&self, pixels: &Array) -> Result<Encoded> {
        let out = self.encoder.forward(pixels)?;
        let high_res = self.decoder.project_high_res(&out.backbone_fpn)?;
        let vision_pos_enc = out
            .vision_pos_enc
            .last()
            .expect("at least one pos level")
            .clone();
        Ok(Encoded {
            vision_features: out.vision_features,
            vision_pos_enc,
            high_res,
        })
    }

    /// `predict_from_encoded`: prompt encode + mask decode → masks/ious + object pointer + score.
    /// `point_coords` `[1,n,2]` (1024-space) with `point_labels` `[1,n]`; both `None` ⇒ a single
    /// `(0,0)` point with label −1 (the "tracking with no new prompt" case). `mask_input`
    /// `[1,1,256,256]` threads a previous mask in as a dense prompt. `add_no_mem_embed` adds the
    /// no-memory bias (true for the very first frame, false once memory conditioning is applied).
    #[allow(clippy::too_many_arguments)]
    fn predict_from_encoded(
        &self,
        enc: &Encoded,
        conditioned_features: Option<&Array>,
        point_coords: Option<&Array>,
        point_labels: Option<&Array>,
        mask_input: Option<&Array>,
        multimask_output: bool,
        add_no_mem_embed: bool,
    ) -> Result<PredictOut> {
        let features = conditioned_features.unwrap_or(&enc.vision_features);
        let batch = features.shape()[0];

        let default_coords;
        let default_labels;
        let (coords, labels) = match (point_coords, point_labels) {
            (Some(c), Some(l)) => (c, l),
            _ => {
                default_coords = ops::zeros::<f32>(&[batch, 1, 2])?;
                default_labels = multiply(&ops::ones::<i32>(&[batch, 1])?, Array::from_int(-1))?;
                (&default_coords, &default_labels)
            }
        };
        let (sparse, dense) = self.prompt.encode(Some(coords), Some(labels), mask_input)?;

        let image_embeddings = if add_no_mem_embed {
            add(features, &self.no_mem_embed.reshape(&[1, 256, 1, 1])?)?
        } else {
            features.clone()
        };
        let image_pe = self.prompt.dense_pe()?;

        let (masks, ious, sam_tokens, object_score_logits) = self.decoder.predict(
            &image_embeddings,
            &image_pe,
            &sparse,
            &dense,
            multimask_output,
            &enc.high_res,
        )?;

        let token = select_obj_ptr_token(&sam_tokens, &ious, multimask_output)?;
        let obj_ptr = self.project_object_pointer(&token, &object_score_logits)?;
        Ok(PredictOut {
            low_res_masks: masks.as_dtype(Dtype::Float32)?,
            ious,
            obj_ptr,
            object_score_logits,
        })
    }

    /// `project_object_pointer`: MLP-project the selected mask token, then gate by the object score
    /// (no-object frames fall back to the learned `no_obj_ptr`).
    fn project_object_pointer(&self, token: &Array, object_score_logits: &Array) -> Result<Array> {
        let obj_ptr = self.obj_ptr_proj.forward(token)?; // [b,256]
        let is_obj = object_score_logits
            .gt(Array::from_f32(0.0))?
            .as_dtype(obj_ptr.dtype())?; // [b,1]
        let kept = multiply(&is_obj, &obj_ptr)?;
        let dropped = multiply(&subtract(Array::from_f32(1.0), &is_obj)?, &self.no_obj_ptr)?;
        Ok(add(&kept, &dropped)?)
    }

    /// `encode_memory`: a frame's `(vision_features, low-res mask, object score)` → the 64-channel
    /// memory feature map + position encoding stored in the bank.
    fn encode_memory(
        &self,
        vision_features: &Array,
        low_res_mask: &Array,
        object_score_logits: &Array,
        is_mask_from_points: bool,
    ) -> Result<(Array, Array)> {
        let high_res = upsample_mask_1024(low_res_mask)?; // [1,1,1024,1024]
        let mask_for_mem = if is_mask_from_points {
            high_res
                .gt(Array::from_f32(0.0))?
                .as_dtype(Dtype::Float32)?
        } else {
            sigmoid(&high_res)?
        };
        // scale 0/1 → −10/+10 (the reference's `* 20 - 10`).
        let mask_for_mem = subtract(
            &multiply(&mask_for_mem, Array::from_f32(20.0))?,
            Array::from_f32(10.0),
        )?;
        let mem = self
            .memory_encoder
            .forward(vision_features, &mask_for_mem, true)?;

        // No-object frames add the learned spatial bias (kept = 0 when the object is present).
        let is_obj = object_score_logits
            .gt(Array::from_f32(0.0))?
            .as_dtype(mem.vision_features.dtype())?; // [1,1]
        let gate = subtract(Array::from_f32(1.0), &is_obj)?.reshape(&[1, 1, 1, 1])?;
        let spatial = self.no_obj_embed_spatial.reshape(&[1, 64, 1, 1])?;
        let features = add(&mem.vision_features, &multiply(&gate, &spatial)?)?;
        Ok((features, mem.vision_pos_enc))
    }

    /// `condition_with_memories` (single object, propagation contract — `current_frame_idx` always
    /// set). Assemble the memory bank from the cond + non-cond frame memories with temporal-position
    /// encodings and object-pointer tokens, then run memory attention → memory-conditioned features.
    fn condition_with_memories(
        &self,
        enc: &Encoded,
        non_cond: &BTreeMap<i32, FrameOut>,
        cond: &BTreeMap<i32, FrameOut>,
        current_frame_idx: i32,
        track_in_reverse: bool,
    ) -> Result<Array> {
        let feat = &enc.vision_features;
        let (b, c) = (feat.shape()[0], feat.shape()[1]);
        let seq = feat.reshape(&[b, c, -1])?.transpose_axes(&[2, 0, 1])?; // [HW,b,256]
        let pos = &enc.vision_pos_enc;
        let seq_pos = pos.reshape(&[b, c, -1])?.transpose_axes(&[2, 0, 1])?;

        let mut mem_parts: Vec<Array> = Vec::new();
        let mut pos_parts: Vec<Array> = Vec::new();

        // Spatial memories: cond frames at t_pos 0, then the temporally-strided recent frames.
        let mut append_memory = |out: &FrameOut, t_pos: i32| -> Result<()> {
            // A cond frame added via `add_points_internal` stores `run_mem_encoder: false` — its memory
            // is encoded lazily in `preflight`. Correcting a prompted frame before `propagate`
            // (`add_new_box(f0)` → `add_correction_points(f0)`) reaches here with memory not yet
            // encoded. Skip such a frame rather than panicking: `preflight` encodes it before
            // propagation, and if no frame has memory the empty-mem path below handles it (F-166).
            let (Some(mf), Some(mp)) =
                (out.maskmem_features.as_ref(), out.maskmem_pos_enc.as_ref())
            else {
                return Ok(());
            };
            let (mb, mc) = (mf.shape()[0], mf.shape()[1]);
            let mem = mf.reshape(&[mb, mc, -1])?.transpose_axes(&[2, 0, 1])?; // [HW,b,64]
            let mem_pos = mp.reshape(&[mb, mc, -1])?.transpose_axes(&[2, 0, 1])?;
            // maskmem_tpos_enc[7 - t_pos - 1] → [1,1,64], broadcast over the spatial tokens.
            let tpos = self
                .maskmem_tpos_enc
                .take_axis(Array::from_int(NUM_MASKMEM - t_pos - 1), 0)?
                .reshape(&[1, 1, 64])?;
            mem_parts.push(mem);
            pos_parts.push(add(&mem_pos, &tpos)?);
            Ok(())
        };

        for out in cond.values() {
            append_memory(out, 0)?;
        }
        for t_pos in 1..NUM_MASKMEM {
            let prev = maskmem_prev_frame(current_frame_idx, t_pos, track_in_reverse);
            if let Some(out) = non_cond.get(&prev) {
                append_memory(out, t_pos)?;
            }
        }

        if mem_parts.is_empty() {
            return Ok(add(feat, &self.no_mem_embed.reshape(&[1, 256, 1, 1])?)?);
        }

        let mut mem = concatenate_axis(&mem_parts.iter().collect::<Vec<_>>(), 0)?;
        let mut mem_pos = concatenate_axis(&pos_parts.iter().collect::<Vec<_>>(), 0)?;

        // Object-pointer tokens: cond frames in the past, then the recent strided frames.
        let mut distances: Vec<i32> = Vec::new();
        let mut ptrs: Vec<Array> = Vec::new();
        for (&frame_idx, out) in cond.iter() {
            let in_past = if track_in_reverse {
                frame_idx >= current_frame_idx
            } else {
                frame_idx <= current_frame_idx
            };
            if in_past {
                let signed =
                    (current_frame_idx - frame_idx) * if track_in_reverse { -1 } else { 1 };
                distances.push(signed);
                ptrs.push(out.obj_ptr.clone());
            }
        }
        for t_diff in 1..MAX_OBJ_PTRS {
            let frame_idx = if track_in_reverse {
                current_frame_idx + t_diff
            } else {
                current_frame_idx - t_diff
            };
            if let Some(out) = non_cond.get(&frame_idx) {
                distances.push(t_diff);
                ptrs.push(out.obj_ptr.clone());
            }
        }

        let mut num_obj = 0;
        if !ptrs.is_empty() {
            let take = (ptrs.len()).min(MAX_OBJ_PTRS as usize);
            distances.truncate(take);
            ptrs.truncate(take);
            let p = ptrs.len() as i32;
            // [P,b,256] → [P,b,4,64] → [P,4,b,64] → [4P,b,64].
            let ptr_src = stack_axis(&ptrs.iter().collect::<Vec<_>>(), 0)?;
            let pb = ptr_src.shape()[1];
            let ptr = ptr_src
                .reshape(&[p, pb, 4, 64])?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[p * 4, pb, 64])?;
            // Temporal position per pointer → projected → repeated 4× (one per split token).
            let tpos_tbl = obj_ptr_pos_table(&distances, MAX_OBJ_PTRS - 1); // [P,256]
            let ptr_pos = linear(
                &tpos_tbl,
                &self.obj_ptr_tpos_proj.0,
                &self.obj_ptr_tpos_proj.1,
            )?; // [P,64]
            let ptr_pos = ptr_pos.reshape(&[p, 1, 64])?; // broadcast batch (b=1)
            let ptr_pos = broadcast_to(&ptr_pos.reshape(&[p, 1, 1, 64])?, &[p, 4, 1, 64])?
                .reshape(&[p * 4, 1, 64])?;
            mem = concatenate_axis(&[&mem, &ptr], 0)?;
            mem_pos = concatenate_axis(&[&mem_pos, &ptr_pos], 0)?;
            num_obj = ptr.shape()[0];
        }

        let out = self
            .memory_attention
            .forward(&seq, &seq_pos, &mem, &mem_pos, num_obj)?;
        // [HW,b,256] → [b,256,HW] → feat shape.
        Ok(out.transpose_axes(&[1, 2, 0])?.reshape(feat.shape())?)
    }
}

/// Select the object-pointer source token: token 0 when single, else the argmax-IoU multimask token.
fn select_obj_ptr_token(tokens: &Array, ious: &Array, multimask_output: bool) -> Result<Array> {
    let b = tokens.shape()[0];
    let t = tokens.shape()[1];
    if !multimask_output || t == 1 {
        return Ok(tokens
            .take_axis(Array::from_int(0), 1)?
            .reshape(&[b, 256])?);
    }
    // host argmax over the t IoUs (b == 1 in the single-object path).
    let iv = ious
        .reshape(&[-1])?
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec();
    let best = crate::util::argmax_f32(&iv) as i32;
    Ok(tokens
        .take_axis(Array::from_int(best), 1)?
        .reshape(&[b, 256])?)
}

/// The previous frame a spatial memory at temporal slot `t_pos` (1..7) is pulled from, matching the
/// reference's stride arithmetic (`condition_with_memories`, `memory_temporal_stride_for_eval = 1`).
/// `t_rel = num_maskmem − t_pos`: the most-recent slot (`t_rel == 1`) is the immediately prior frame;
/// the rest step back one strided frame each. Forward tracking subtracts, reverse adds.
fn maskmem_prev_frame(current: i32, t_pos: i32, track_in_reverse: bool) -> i32 {
    let t_rel = NUM_MASKMEM - t_pos;
    if t_rel == 1 {
        if track_in_reverse {
            current + t_rel
        } else {
            current - t_rel
        }
    } else if track_in_reverse {
        let base = -(-(current + 2) / MEM_STRIDE) * MEM_STRIDE;
        base + (t_rel - 2) * MEM_STRIDE
    } else {
        let base = ((current - 2) / MEM_STRIDE) * MEM_STRIDE;
        base - (t_rel - 2) * MEM_STRIDE
    }
}

/// `_obj_ptr_pos` (pre-projection): per-pointer signed distance → a `[P, 256]` sinusoidal table
/// (`sin‖cos` of `pos / dim_t`, `dim_t[j] = 10000^(2⌊j/2⌋/128)`). Host-built (deterministic).
fn obj_ptr_pos_table(distances: &[i32], max_distance: i32) -> Array {
    const PE_DIM: usize = 128;
    let denom = max_distance.max(1) as f32;
    let dim_t: Vec<f32> = (0..PE_DIM)
        .map(|j| 10000f32.powf(2.0 * ((j / 2) as f32) / PE_DIM as f32))
        .collect();
    let p = distances.len();
    let mut tbl = vec![0f32; p * 2 * PE_DIM];
    for (i, &d) in distances.iter().enumerate() {
        let pos = d as f32 / denom;
        for (j, &dt) in dim_t.iter().enumerate() {
            let v = pos / dt;
            tbl[i * 2 * PE_DIM + j] = v.sin();
            tbl[i * 2 * PE_DIM + PE_DIM + j] = v.cos();
        }
    }
    Array::from_slice(&tbl, &[p as i32, 2 * PE_DIM as i32])
}

/// Bilinear-upsample a `[1,1,256,256]` low-res mask to `[1,1,1024,1024]`, matching the reference's
/// separable `_resize_1d` (`align_corners=False`, half-pixel). Host-computed for exact parity.
fn upsample_mask_1024(low: &Array) -> Result<Array> {
    const IN: usize = 256;
    const OUT: usize = 1024;
    let src = low.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
    debug_assert_eq!(src.len(), IN * IN);
    // 1-D coefficients (shared by both axes): lo/hi indices + weight per output position.
    let scale = IN as f32 / OUT as f32;
    let mut lo = [0usize; OUT];
    let mut hi = [0usize; OUT];
    let mut wt = [0f32; OUT];
    for (o, ((l, h), wv)) in lo
        .iter_mut()
        .zip(hi.iter_mut())
        .zip(wt.iter_mut())
        .enumerate()
    {
        let s = (o as f32 + 0.5) * scale - 0.5;
        let lo_f = s.floor();
        *l = lo_f.clamp(0.0, (IN - 1) as f32) as usize;
        *h = (lo_f + 1.0).clamp(0.0, (IN - 1) as f32) as usize;
        *wv = s - lo_f;
    }
    // Horizontal pass: [IN, IN] → [IN, OUT].
    let mut tmp = vec![0f32; IN * OUT];
    for y in 0..IN {
        let row = &src[y * IN..y * IN + IN];
        for ox in 0..OUT {
            tmp[y * OUT + ox] = row[lo[ox]] * (1.0 - wt[ox]) + row[hi[ox]] * wt[ox];
        }
    }
    // Vertical pass: [IN, OUT] → [OUT, OUT].
    let mut out = vec![0f32; OUT * OUT];
    for oy in 0..OUT {
        let (l, h, w) = (lo[oy], hi[oy], wt[oy]);
        for ox in 0..OUT {
            out[oy * OUT + ox] = tmp[l * OUT + ox] * (1.0 - w) + tmp[h * OUT + ox] * w;
        }
    }
    Ok(Array::from_slice(&out, &[1, 1, OUT as i32, OUT as i32]))
}

/// SAM2 single-object video predictor: prompt a frame with a box, then propagate masks across the
/// clip using the memory bank ([`mlx_sam.video_predictor.SAM2VideoPredictor`], single-object path).
pub struct Sam2VideoPredictor {
    model: Sam2VideoModel,
}

/// Tracking state for one clip (single object). Holds the preprocessed frames + the per-frame
/// outputs the memory bank replays (`cond` = prompted frames, `non_cond` = propagated frames).
pub struct VideoState {
    images: Array, // [T, 3, 1024, 1024]
    num_frames: i32,
    video_h: u32,
    video_w: u32,
    cond: BTreeMap<i32, FrameOut>,
    non_cond: BTreeMap<i32, FrameOut>,
    /// Per-frame point prompts (1024-space coords `[1,n,2]`, labels `[1,n]`).
    points: BTreeMap<i32, (Array, Array)>,
    frames_tracked: BTreeMap<i32, bool>, // frame → reverse
    /// Per-frame image-encoder output cached for prompted/corrected (cond) frames, so `preflight`'s
    /// memory encode reuses the Hiera+FPN forward `run_single_frame` already ran instead of redoing
    /// it (F-168, the torch reference's `cached_features`). Evicted in `preflight` after consumption,
    /// so it only ever holds the handful of un-memory-encoded cond frames.
    encoded: BTreeMap<i32, Encoded>,
}

impl Sam2VideoPredictor {
    pub fn from_weights(w: &Weights, cfg: &Sam2ImageEncoderConfig) -> Result<Self> {
        Ok(Self {
            model: Sam2VideoModel::from_weights(w, cfg)?,
        })
    }

    pub fn from_weights_for_size(w: &Weights, size: Sam2ModelSize) -> Result<Self> {
        Self::from_weights(w, &Sam2ImageEncoderConfig::for_size(size))
    }

    /// `init_state` from already-preprocessed frames `[T, 3, 1024, 1024]` (the reference's
    /// `state["images"]`) plus the original video size (for prompt coordinate scaling).
    pub fn init_state_from_pixels(&self, images: Array, video_h: u32, video_w: u32) -> VideoState {
        let num_frames = images.shape()[0];
        VideoState {
            images,
            num_frames,
            video_h,
            video_w,
            cond: BTreeMap::new(),
            non_cond: BTreeMap::new(),
            points: BTreeMap::new(),
            frames_tracked: BTreeMap::new(),
            encoded: BTreeMap::new(),
        }
    }

    /// `init_state` from raw RGB8 HWC frames (each `video_h × video_w × 3`): preprocess them into the
    /// `[T,3,1024,1024]` clip tensor, then build the state.
    pub fn init_state_from_frames(
        &self,
        frames: &[&[u8]],
        video_h: u32,
        video_w: u32,
    ) -> Result<VideoState> {
        let per = frames
            .iter()
            .map(|f| preprocess(f, video_h as usize, video_w as usize))
            .collect::<Result<Vec<_>>>()?;
        let images = concatenate_axis(&per.iter().collect::<Vec<_>>(), 0)?;
        Ok(self.init_state_from_pixels(images, video_h, video_w))
    }

    fn frame_pixels(&self, state: &VideoState, frame_idx: i32) -> Result<Array> {
        Ok(state
            .images
            .take_axis(Array::from_int(frame_idx), 0)?
            .reshape(&[1, 3, IMAGE_SIZE, IMAGE_SIZE])?)
    }

    fn encode_frame(&self, state: &VideoState, frame_idx: i32) -> Result<Encoded> {
        let pixels = self.frame_pixels(state, frame_idx)?;
        self.model.encode_image(&pixels)
    }

    /// `add_new_points_or_box` (box). Prompt frame `frame_idx` with a box in **original** pixel
    /// space; produces the initial conditioned mask (memory is encoded lazily at propagation time).
    pub fn add_new_box(
        &self,
        state: &mut VideoState,
        frame_idx: i32,
        box_xyxy: [f32; 4],
    ) -> Result<Array> {
        let sx = IMAGE_SIZE as f32 / state.video_w as f32;
        let sy = IMAGE_SIZE as f32 / state.video_h as f32;
        let pts = Array::from_slice(
            &[
                box_xyxy[0] * sx,
                box_xyxy[1] * sy,
                box_xyxy[2] * sx,
                box_xyxy[3] * sy,
            ],
            &[1, 2, 2],
        );
        let labels = Array::from_slice(&[2i32, 3], &[1, 2]);
        self.add_points_internal(state, frame_idx, pts, labels)
    }

    /// `add_new_points` (correction). Add positive/negative point(s) in original pixel space to an
    /// (already-tracked or new) frame; when the frame was previously tracked, its prior mask is fed
    /// back as a dense prompt (the reference's `prev_low` path) so the correction refines the track.
    pub fn add_correction_points(
        &self,
        state: &mut VideoState,
        frame_idx: i32,
        points: &[[f32; 2]],
        labels: &[i32],
    ) -> Result<Array> {
        if points.len() != labels.len() {
            return Err(Error::Msg(format!(
                "sam2 add_correction_points: points/labels length mismatch ({} vs {})",
                points.len(),
                labels.len()
            )));
        }
        let sx = IMAGE_SIZE as f32 / state.video_w as f32;
        let sy = IMAGE_SIZE as f32 / state.video_h as f32;
        let flat: Vec<f32> = points.iter().flat_map(|p| [p[0] * sx, p[1] * sy]).collect();
        let n = points.len() as i32;
        let pts = Array::from_slice(&flat, &[1, n, 2]);
        let lab = Array::from_slice(labels, &[1, n]);
        self.add_points_internal(state, frame_idx, pts, lab)
    }

    /// Shared prompt-edit path (`add_new_points_or_box` body): run single-frame inference (no memory
    /// encode — that happens in the propagation preflight) and store the frame as a cond frame.
    fn add_points_internal(
        &self,
        state: &mut VideoState,
        frame_idx: i32,
        pts: Array,
        labels: Array,
    ) -> Result<Array> {
        let is_init_cond = !state.frames_tracked.contains_key(&frame_idx);
        let reverse = state
            .frames_tracked
            .get(&frame_idx)
            .copied()
            .unwrap_or(false);
        state
            .points
            .insert(frame_idx, (pts.clone(), labels.clone()));

        // The previous mask (if this frame was already tracked) seeds the dense prompt, clipped to
        // ±32 (`mx.clip(prev["pred_masks"], -32, 32)`).
        let prev_low = state
            .cond
            .get(&frame_idx)
            .or_else(|| state.non_cond.get(&frame_idx))
            .map(|o| -> Result<Array> {
                Ok(minimum(
                    &maximum(&o.pred_masks, Array::from_f32(-32.0))?,
                    Array::from_f32(32.0),
                )?)
            })
            .transpose()?;

        let (out, enc) = self.run_single_frame(
            state,
            frame_idx,
            is_init_cond,
            Some((&pts, &labels)),
            reverse,
            prev_low.as_ref(),
            false,
        )?;
        let mask = out.pred_masks.clone();
        state.cond.insert(frame_idx, out);
        state.non_cond.remove(&frame_idx);
        // Cache this cond frame's encoder output so `preflight` reuses it instead of re-encoding
        // (F-168). `preflight` removes it once the memory is encoded.
        state.encoded.insert(frame_idx, enc);
        Ok(mask)
    }

    /// `_run_single_frame_inference` (single object). Predict the frame's mask — from the prompt
    /// directly on an init cond frame, else conditioned on the memory bank — and optionally
    /// memory-encode it.
    /// Returns the frame output and the `Encoded` image features it computed, so the caller can cache
    /// them for `preflight` (F-168) rather than re-running the Hiera+FPN forward.
    #[allow(clippy::too_many_arguments)]
    fn run_single_frame(
        &self,
        state: &VideoState,
        frame_idx: i32,
        is_init_cond: bool,
        point_inputs: Option<(&Array, &Array)>,
        reverse: bool,
        prev_low: Option<&Array>,
        run_mem_encoder: bool,
    ) -> Result<(FrameOut, Encoded)> {
        let enc = self.encode_frame(state, frame_idx)?;
        let is_mask_from_points = point_inputs.is_some();
        let init_prompt = match (is_init_cond, prev_low.is_none(), point_inputs) {
            (true, true, Some(pts)) => Some(pts),
            _ => None,
        };
        let out = if let Some((c, l)) = init_prompt {
            self.model
                .predict_from_encoded(&enc, None, Some(c), Some(l), None, true, true)?
        } else {
            let conditioned = self.model.condition_with_memories(
                &enc,
                &state.non_cond,
                &state.cond,
                frame_idx,
                reverse,
            )?;
            let (coords, labels) = match point_inputs {
                Some((c, l)) => (Some(c), Some(l)),
                None => (None, None),
            };
            self.model.predict_from_encoded(
                &enc,
                Some(&conditioned),
                coords,
                labels,
                prev_low,
                false,
                false,
            )?
        };

        let low = best_low_mask(&out.low_res_masks, &out.ious)?;
        let mut frame = FrameOut {
            pred_masks: low.clone(),
            obj_ptr: out.obj_ptr.clone(),
            object_score_logits: out.object_score_logits.clone(),
            maskmem_features: None,
            maskmem_pos_enc: None,
        };
        if run_mem_encoder {
            let (mf, mp) = self.model.encode_memory(
                &enc.vision_features,
                &low,
                &out.object_score_logits,
                is_mask_from_points,
            )?;
            frame.maskmem_features = Some(mf);
            frame.maskmem_pos_enc = Some(mp);
        }
        Ok((frame, enc))
    }

    /// Ensure every cond frame's memory is encoded (`_propagate_preflight`).
    fn preflight(&self, state: &mut VideoState) -> Result<()> {
        let to_encode: Vec<i32> = state
            .cond
            .iter()
            .filter(|(_, o)| o.maskmem_features.is_none())
            .map(|(&f, _)| f)
            .collect();
        for frame_idx in to_encode {
            // Reuse the encoder output `add_points_internal` cached for this cond frame; only re-encode
            // if it isn't there (e.g. a cond frame inserted by another path). Evicting frees it (F-168).
            let enc = match state.encoded.remove(&frame_idx) {
                Some(enc) => enc,
                None => self.encode_frame(state, frame_idx)?,
            };
            let out = state.cond.get(&frame_idx).unwrap();
            let is_mask_from_points = state.points.contains_key(&frame_idx);
            let (mf, mp) = self.model.encode_memory(
                &enc.vision_features,
                &out.pred_masks,
                &out.object_score_logits,
                is_mask_from_points,
            )?;
            let out = state.cond.get_mut(&frame_idx).unwrap();
            out.maskmem_features = Some(mf);
            out.maskmem_pos_enc = Some(mp);
        }
        Ok(())
    }

    /// `propagate_in_video` (forward, single object). Returns each frame's low-res mask logits
    /// `[1,1,256,256]` in propagation order, starting from the prompted frame.
    ///
    /// Forward only: every frame is recorded as forward (`frames_tracked` ← `false`). There is no
    /// `propagate_reverse` yet — the reverse plumbing is wired and unit-tested but unreachable; see
    /// the module-level "Direction" note (F-176 / sc-4151).
    pub fn propagate(&self, state: &mut VideoState) -> Result<Vec<(i32, Array)>> {
        self.preflight(state)?;
        let start = *state.cond.keys().min().ok_or_else(|| {
            Error::Msg(
                "sam2 propagate: a prompt is required before propagation (no conditioning frames \
                 — call add_box/add_correction_points first)"
                    .into(),
            )
        })?;
        let mut results = Vec::new();
        for frame_idx in start..state.num_frames {
            if let Some(out) = state.cond.get(&frame_idx) {
                results.push((frame_idx, out.pred_masks.clone()));
            } else {
                // Non-cond propagate frames are encoded once and never revisited, so their `Encoded`
                // is dropped rather than cached (keeps the cache bounded to cond frames — F-168).
                let (out, _enc) =
                    self.run_single_frame(state, frame_idx, false, None, false, None, true)?;
                results.push((frame_idx, out.pred_masks.clone()));
                state.non_cond.insert(frame_idx, out);
            }
            state.frames_tracked.insert(frame_idx, false);
        }
        Ok(results)
    }

    /// Threshold a frame's low-res logits to a binary `L` mask at the original video resolution
    /// (`> 0`), bilinear-resized (half-pixel). Convenience for callers that want display masks.
    pub fn mask_to_video_res(&self, state: &VideoState, low: &Array) -> Result<Array> {
        let logits = low.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let (h, w) = (state.video_h as usize, state.video_w as usize);
        let up = resize_bilinear_2d(&logits, 256, 256, h, w);
        let bin: Vec<u8> = up.iter().map(|&v| if v > 0.0 { 255 } else { 0 }).collect();
        Ok(Array::from_slice(&bin, &[h as i32, w as i32]))
    }
}

/// Best low-res mask by IoU (`_best_low_mask`): argmax over the candidate axis (host, k ≤ 3).
fn best_low_mask(low: &Array, ious: &Array) -> Result<Array> {
    let k = low.shape()[1];
    if k == 1 {
        return Ok(low.reshape(&[1, 1, 256, 256])?);
    }
    let iv = ious
        .reshape(&[-1])?
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec();
    let best = crate::util::argmax_f32(&iv) as i32;
    Ok(low
        .take_axis(Array::from_int(best), 1)?
        .reshape(&[1, 1, 256, 256])?)
}

/// The shared host bilinear resampler (display helper), aliased to its historical name here (F-171).
use crate::util::bilinear_resize_f32 as resize_bilinear_2d;

#[cfg(test)]
mod tests {
    use super::*;

    /// Forward tracking pulls the spatial memory bank from the consecutive frames immediately behind
    /// the current one: slots t_pos 6..1 → frames `current-1 .. current-6` (stride 1).
    #[test]
    fn maskmem_prev_frame_forward_is_consecutive_recent() {
        let current = 5;
        let frames: Vec<i32> = (1..NUM_MASKMEM)
            .map(|t_pos| maskmem_prev_frame(current, t_pos, false))
            .collect();
        // t_pos = 1,2,3,4,5,6  →  current-6,-5,-4,-3,-2,-1.
        assert_eq!(frames, vec![-1, 0, 1, 2, 3, 4]);
    }

    /// Reverse tracking mirrors it: the bank comes from the frames immediately ahead.
    #[test]
    fn maskmem_prev_frame_reverse_is_consecutive_ahead() {
        let current = 5;
        let frames: Vec<i32> = (1..NUM_MASKMEM)
            .map(|t_pos| maskmem_prev_frame(current, t_pos, true))
            .collect();
        assert_eq!(frames, vec![11, 10, 9, 8, 7, 6]);
    }

    /// The object-pointer temporal table is `[P, 256]`, and each `(sin, cos)` column pair for the
    /// same frequency lies on the unit circle (`sin² + cos² = 1`).
    #[test]
    fn obj_ptr_pos_table_is_unit_circle() {
        let tbl = obj_ptr_pos_table(&[0, 1, -3, 7], MAX_OBJ_PTRS - 1);
        assert_eq!(tbl.shape(), &[4, 256]);
        let v = tbl.as_slice::<f32>();
        for row in 0..4 {
            for j in 0..128 {
                let s = v[row * 256 + j];
                let c = v[row * 256 + 128 + j];
                assert!((s * s + c * c - 1.0).abs() < 1e-5, "row {row} col {j}");
            }
        }
        // Distance 0 → all sin = 0, all cos = 1.
        for j in 0..128 {
            assert!(v[j].abs() < 1e-6);
            assert!((v[128 + j] - 1.0).abs() < 1e-6);
        }
    }

    /// Upsampling a constant low-res mask to 1024² yields a constant map of the same value
    /// (interpolation of equal endpoints), with the expected shape.
    #[test]
    fn upsample_mask_1024_is_constant_preserving() {
        let low = Array::from_slice(&vec![0.7f32; 256 * 256], &[1, 1, 256, 256]);
        let up = upsample_mask_1024(&low).unwrap();
        assert_eq!(up.shape(), &[1, 1, 1024, 1024]);
        let v = up.as_slice::<f32>();
        let max_dev = v.iter().fold(0f32, |m, &x| m.max((x - 0.7).abs()));
        assert!(max_dev < 1e-6, "constant upsample deviated by {max_dev:e}");
    }

    /// A monotone low-res ramp upsamples to a monotone (non-decreasing along each axis) map —
    /// catches an axis swap or flipped interpolation weight.
    #[test]
    fn upsample_mask_1024_preserves_monotone_ramp() {
        let mut src = vec![0f32; 256 * 256];
        for (y, row) in src.chunks_mut(256).enumerate() {
            for (x, v) in row.iter_mut().enumerate() {
                *v = y as f32 + x as f32;
            }
        }
        let low = Array::from_slice(&src, &[1, 1, 256, 256]);
        let up = upsample_mask_1024(&low).unwrap();
        let v = up.as_slice::<f32>();
        // Corner ordering: top-left smallest, bottom-right largest.
        assert!(v[0] < v[1024 * 1024 - 1]);
        // Row 0 is non-decreasing left→right.
        assert!(v[0] <= v[500] && v[500] <= v[1023]);
    }
}
