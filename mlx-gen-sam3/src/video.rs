//! SAM3 multi-object video PCS pipeline (`Sam3VideoModel`) — epic 4910, sc-4924, Phase F2.5/F2.6.
//!
//! Pure-host orchestration over the (parity-green) tracker neural primitives in [`crate::tracker`]:
//! per frame, detect concept instances ([`crate::Sam3ImageSegmenter`]), propagate existing identities
//! through the per-object memory bank ([`Sam3Tracker::decode_tracked_frame`]), associate detections to
//! tracklets, seed new identities from unmatched detections
//! ([`Sam3Tracker::decode_mask_conditioning_frame`]), and encode each frame's masks into memory.
//!
//! Mirrors `transformers` `sam3_video/modeling_sam3_video.py` `_det_track_one_frame`. Matches the
//! reference's **no-`kernels`** configuration: NMS (`det_nms_thresh`) and hole-filling are no-ops.
//! Masks flow as raw 288² logits (the processor sigmoids for display).

use std::collections::BTreeMap;
use std::rc::Rc;

use mlx_rs::ops::sigmoid;
use mlx_rs::{Array, Dtype};

use mlx_gen::Result;

use crate::config::Sam3VisionConfig;
use crate::tracker::TrackerFrameOutput;
use crate::vision::Backbone;
use crate::{Sam3ImageSegmenter, Sam3Tracker};

// --- config (Sam3VideoConfig defaults) -----------------------------------------------------------
const LOW_RES: i32 = 288; // low_res_mask_size
const SCORE_THRESH_DET: f32 = 0.5; // score_threshold_detection
const NEW_DET_THRESH: f32 = 0.7;
const ASSOC_IOU_THRESH: f32 = 0.1;
const TRK_ASSOC_IOU_THRESH: f32 = 0.5;
const HIGH_CONF_THRESH: f32 = 0.8;
const HIGH_IOU_THRESH: f32 = 0.8;
const NUM_MASKMEM: i32 = 7;
const MAX_COND_FRAME_NUM: i32 = 4;
const MAX_OBJ_PTRS: i32 = 16; // max_object_pointers_in_encoder
const RECONDITION_EVERY: i32 = 16;
const INIT_KEEP_ALIVE: i32 = 30;
const MAX_KEEP_ALIVE: i32 = 30;
const MIN_KEEP_ALIVE: i32 = -1;
const HOTSTART_DELAY: i32 = 15;
const HOTSTART_UNMATCH: usize = 8;
const HOTSTART_DUP: usize = 8;
const SUPPRESS_OCC_THRESH: f32 = 0.7; // suppress_overlapping_based_on_recent_occlusion_threshold
const NEVER_OCCLUDED: i32 = -1;
const ALWAYS_OCCLUDED: i32 = 100_000;
const NO_OBJ_LOGIT: f32 = -10.0;

/// Gathered spatial memory: `(relative_temporal_offset, maskmem_features, maskmem_pos_enc)` per frame.
type SpatialMem = Vec<(i32, Array, Array)>;
/// Gathered object pointers: `(temporal_offset, pointer [1,256])`.
type ObjPointers = Vec<(i32, Array)>;

/// A detection on a frame: raw 288² mask logits + score + box, plus the prompt that produced it.
struct Detection {
    mask: Vec<f32>, // [288·288] logits
    score: f32,
    prompt_id: i32,
}

/// One stored per-frame output for an object (the memory-bank entry).
#[derive(Clone)]
struct FrameMem {
    maskmem_features: Option<Array>, // [5184,1,64] seq-first (bf16-cast); None until memory-encoded
    maskmem_pos_enc: Option<Array>,  // [5184,1,64]
    object_pointer: Array,           // [1,256]
    object_score: f32,
}

/// Per-object memory bank: conditioning-frame outputs (user/detection-seeded) + tracked-frame outputs.
#[derive(Default, Clone)]
struct ObjectBank {
    cond: BTreeMap<i32, FrameMem>,
    non_cond: BTreeMap<i32, FrameMem>,
}

/// The per-frame segmentation result: object id → 288² mask logits, in id order.
pub struct VideoFrameOutput {
    pub obj_ids: Vec<i32>,
    pub masks: Vec<Vec<f32>>, // each [288·288] logits, parallel to obj_ids
}

/// `Sam3VideoModel`: the detector + the tracker, driving the multi-object PCS pipeline.
pub struct Sam3VideoModel {
    segmenter: Sam3ImageSegmenter,
    tracker: Sam3Tracker,
    // --- per-session state ---
    obj_ids: Vec<i32>,      // ordered; index = obj_idx
    banks: Vec<ObjectBank>, // parallel to obj_ids
    obj_prompt: Vec<i32>,   // prompt id per obj_idx
    max_obj_id: i32,
    num_frames: i32,
    // hotstart metadata (keyed by obj_id)
    first_frame: BTreeMap<i32, i32>,
    unmatched_frames: BTreeMap<i32, Vec<i32>>,
    keep_alive: BTreeMap<i32, i32>,
    overlap_pairs: BTreeMap<(i32, i32), Vec<i32>>,
    removed: std::collections::BTreeSet<i32>,
    last_occluded: BTreeMap<i32, i32>,
}

impl Sam3VideoModel {
    pub fn from_weights(w: &mlx_gen::weights::Weights) -> Result<Self> {
        // One PE backbone, shared between the detector segmenter and the tracker. Both load it from
        // the same `detector_model.vision_encoder.backbone` keys, so loading it twice would carry two
        // identical ~445M-param copies resident at video time (F-028).
        let cfg = Sam3VisionConfig::sam3();
        let backbone = Rc::new(Backbone::from_weights(
            w,
            "detector_model.vision_encoder.backbone",
            &cfg,
        )?);
        Ok(Self {
            segmenter: Sam3ImageSegmenter::from_weights_with_backbone(w, backbone.clone())?,
            tracker: Sam3Tracker::from_weights_with_backbone(w, backbone)?,
            obj_ids: Vec::new(),
            banks: Vec::new(),
            obj_prompt: Vec::new(),
            max_obj_id: -1,
            num_frames: 0,
            first_frame: BTreeMap::new(),
            unmatched_frames: BTreeMap::new(),
            keep_alive: BTreeMap::new(),
            overlap_pairs: BTreeMap::new(),
            removed: std::collections::BTreeSet::new(),
            last_occluded: BTreeMap::new(),
        })
    }

    /// Affine-quantize the whole video model to `bits` (Q8/Q4): the single shared PE backbone plus
    /// the detector segmenter's and the tracker's own heads. Convs/norms/embeddings stay dense
    /// (sc-4925).
    ///
    /// The backbone is shared (one `Rc`) between the segmenter and the tracker (F-028), so it is
    /// quantized **once** and the same quantized `Rc` reinstalled into both — otherwise each side
    /// would quantize into a separate copy and re-duplicate the weights we just deduplicated.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        let mut backbone = (*self.tracker.backbone_rc()).clone();
        backbone.quantize(bits)?;
        let backbone = Rc::new(backbone);
        self.segmenter.set_vision_backbone(backbone.clone());
        self.tracker.set_backbone(backbone);
        self.segmenter.quantize_except_backbone(bits)?;
        self.tracker.quantize_except_backbone(bits)?;
        Ok(())
    }

    /// Process a whole video (forward, non-streaming): `frames[f]` = NCHW `[1,3,1008,1008]`; one text
    /// prompt (`input_ids[1,32]` + `text_mask`). Returns per-frame `obj_id → 288² mask logits`.
    pub fn propagate(
        &mut self,
        frames: &[Array],
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<Vec<VideoFrameOutput>> {
        self.num_frames = frames.len() as i32;
        let mut outputs = Vec::new();
        for (f, px) in frames.iter().enumerate() {
            outputs.push(self.process_frame(f as i32, px, input_ids, text_mask)?);
        }
        Ok(outputs)
    }

    fn process_frame(
        &mut self,
        frame_idx: i32,
        pixels: &Array,
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<VideoFrameOutput> {
        // --- Step 1: vision + detection ---
        let (img_emb, high_res) = self.tracker.encode_frame(pixels)?; // [1,72,72,256], [s0,s1]
        let g = img_emb.shape()[1];
        let cvf = img_emb.reshape(&[g * g, 1, 256])?;
        let cvp = self.tracker.frame_position_encoding(g)?;
        let det = self.run_detection(pixels, input_ids, text_mask)?;

        // --- Step 2: propagate existing identities (run_mem_encoder = false) ---
        let num_existing = self.obj_ids.len();
        let mut trk_masks: Vec<Vec<f32>> = Vec::with_capacity(num_existing); // [288²] logits per obj
        let mut trk_scores: Vec<f32> = Vec::with_capacity(num_existing);
        for obj_idx in 0..num_existing {
            let (spatial, pointers, max_optr) = self.gather_memory(obj_idx, frame_idx);
            let conditioned = self
                .tracker
                .prepare_memory_conditioned_features(&cvf, &cvp, &spatial, &pointers, max_optr)?;
            let out = self.tracker.decode_tracked_frame(&conditioned, &high_res)?;
            let low = to_vec(&out.low_res)?;
            self.banks[obj_idx].non_cond.insert(
                frame_idx,
                FrameMem {
                    maskmem_features: None,
                    maskmem_pos_enc: None,
                    object_pointer: out.object_pointer.clone(),
                    object_score: out.object_score,
                },
            );
            trk_scores.push(out.object_score);
            trk_masks.push(low);
        }

        // --- Step 3: associate + new-object ids + hotstart ---
        let assoc = self.associate(&det, &trk_masks);
        let new_obj_ids: Vec<i32> = (0..assoc.new_det_inds.len() as i32)
            .map(|i| self.max_obj_id + 1 + i)
            .collect();
        for (&oid, &di) in new_obj_ids.iter().zip(&assoc.new_det_inds) {
            // prompt id assigned at creation (recorded when the object is added below)
            let _ = (oid, di);
        }
        let removed_now = self.process_hotstart(frame_idx, &assoc, &new_obj_ids);

        // recondition every Nth frame: confidently re-detected tracks become conditioning frames
        // (recondition_on_trk_masks = True → "validate" mode keeps the tracker mask).
        let mut reconditioned_obj_ids: Vec<i32> = Vec::new();
        if RECONDITION_EVERY > 0
            && frame_idx % RECONDITION_EVERY == 0
            && !assoc.trk_id_to_max_iou_high_conf_det.is_empty()
        {
            for &trk_oid in assoc.trk_id_to_max_iou_high_conf_det.keys() {
                if let Some(obj_idx) = self.obj_ids.iter().position(|&o| o == trk_oid) {
                    if trk_scores.get(obj_idx).copied().unwrap_or(f32::MIN) > HIGH_CONF_THRESH {
                        reconditioned_obj_ids.push(trk_oid);
                    }
                }
            }
        }

        // --- Step 4 (planning tail): suppress overlaps + encode memory for existing objects ---
        if num_existing > 0 {
            self.suppress_overlapping_recent_occlusion(frame_idx, &mut trk_masks, &removed_now);
            self.tracker_update_memories(frame_idx, &img_emb, &trk_masks)?;
            // move reconditioned frames from non_cond → cond so they seed future memory selection.
            for &oid in &reconditioned_obj_ids {
                if let Some(obj_idx) = self.obj_ids.iter().position(|&o| o == oid) {
                    if let Some(fm) = self.banks[obj_idx].non_cond.remove(&frame_idx) {
                        self.banks[obj_idx].cond.insert(frame_idx, fm);
                    }
                }
            }
        }

        // --- Step 5 (execution): add new objects from unmatched detections ---
        for (&oid, &di) in new_obj_ids.iter().zip(&assoc.new_det_inds) {
            self.add_object(oid, det.dets[di].prompt_id);
            let obj_idx = self.obj_ids.len() - 1;
            // binarize the detection logits at 0.5 (reference: det_mask >= 0.5) → mask prompt.
            let mask_bin: Vec<f32> = det.dets[di]
                .mask
                .iter()
                .map(|&v| if v >= 0.5 { 1.0 } else { 0.0 })
                .collect();
            let mask_nhwc = Array::from_slice(&mask_bin, &[1, LOW_RES, LOW_RES, 1]);
            let out: TrackerFrameOutput = self
                .tracker
                .decode_mask_conditioning_frame(&img_emb, &high_res, &mask_nhwc)?;
            let mem =
                self.tracker
                    .encode_new_memory(&img_emb, &out.high_res, out.object_score, true)?;
            self.banks[obj_idx].cond.insert(
                frame_idx,
                FrameMem {
                    maskmem_features: Some(seq_first(&mem.features, true)?),
                    maskmem_pos_enc: Some(seq_first(&mem.pos, false)?),
                    object_pointer: out.object_pointer,
                    object_score: out.object_score,
                },
            );
        }
        // remove objects flagged by hotstart
        for oid in &removed_now {
            self.remove_object(*oid);
        }

        // --- build outputs ---
        self.build_outputs(
            &det,
            &assoc,
            &new_obj_ids,
            &trk_masks,
            &reconditioned_obj_ids,
        )
    }

    // ----- detection (run_detection, single prompt, NMS off) -----
    fn run_detection(
        &self,
        pixels: &Array,
        input_ids: &Array,
        text_mask: &[i32],
    ) -> Result<DetFrame> {
        let seg = self.segmenter.forward(pixels, input_ids, text_mask)?;
        let presence = sigmoid(&seg.presence_logits)?.item::<f32>();
        let probs: Vec<f32> = sigmoid(&seg.pred_logits)?
            .as_slice::<f32>()
            .iter()
            .map(|&s| s * presence)
            .collect();
        let q = probs.len();
        let masks = seg.pred_masks.reshape(&[q as i32, LOW_RES * LOW_RES])?;
        let masks_v = masks.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let mut dets = Vec::new();
        for (qi, &p) in probs.iter().enumerate() {
            if p <= SCORE_THRESH_DET {
                continue;
            }
            let m = masks_v
                [qi * (LOW_RES * LOW_RES) as usize..(qi + 1) * (LOW_RES * LOW_RES) as usize]
                .to_vec();
            dets.push(Detection {
                mask: m,
                score: p,
                prompt_id: 0,
            });
        }
        Ok(DetFrame { dets })
    }

    // ----- memory bank gather (F2.4 selection logic) -----
    fn gather_memory(&self, obj_idx: usize, frame_idx: i32) -> (SpatialMem, ObjPointers, i32) {
        let bank = &self.banks[obj_idx];
        // spatial memory: closest cond frames (offset 0) + non-cond at offsets [num_maskmem-1..1].
        let (selected_cond, unselected_cond) =
            select_closest_cond_frames(frame_idx, &bank.cond, MAX_COND_FRAME_NUM);
        let mut spatial: Vec<(i32, Array, Array)> = Vec::new();
        for f in &selected_cond {
            if let Some(m) = bank.cond.get(f) {
                if let (Some(feat), Some(pos)) = (&m.maskmem_features, &m.maskmem_pos_enc) {
                    spatial.push((0, feat.clone(), pos.clone()));
                }
            }
        }
        for rel in (1..NUM_MASKMEM).rev() {
            let prev = frame_idx - rel;
            let out = bank.non_cond.get(&prev).or_else(|| {
                if unselected_cond.contains(&prev) {
                    bank.cond.get(&prev)
                } else {
                    None
                }
            });
            if let Some(m) = out {
                if let (Some(feat), Some(pos)) = (&m.maskmem_features, &m.maskmem_pos_enc) {
                    spatial.push((rel, feat.clone(), pos.clone()));
                }
            }
        }
        // object pointers: eligible cond frames (t <= frame_idx) + non-cond up to max_optr-1.
        let max_optr = self.num_frames.min(MAX_OBJ_PTRS);
        let mut pointers: Vec<(i32, Array)> = Vec::new();
        for (&t, m) in &bank.cond {
            if t <= frame_idx {
                pointers.push((frame_idx - t, m.object_pointer.clone()));
            }
        }
        for t_diff in 1..max_optr {
            let r = frame_idx - t_diff;
            if r < 0 || r >= self.num_frames {
                break;
            }
            if let Some(m) = bank.non_cond.get(&r) {
                pointers.push((t_diff, m.object_pointer.clone()));
            }
        }
        (spatial, pointers, max_optr)
    }

    // ----- association (_associate_det_trk; mask-IoU, no Hungarian) -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (iou / obj_ids / dets)
    fn associate(&self, det: &DetFrame, trk_masks: &[Vec<f32>]) -> Assoc {
        let n = det.dets.len();
        let m = trk_masks.len();
        let mut a = Assoc::default();
        if m == 0 {
            a.new_det_inds = (0..n).collect();
            return a;
        }
        let det_bin: Vec<Vec<bool>> = det.dets.iter().map(|d| binarize(&d.mask)).collect();
        let trk_bin: Vec<Vec<bool>> = trk_masks.iter().map(|t| binarize(t)).collect();
        let trk_nonempty: Vec<bool> = trk_bin.iter().map(|t| t.iter().any(|&x| x)).collect();
        // IoU[n][m], zeroed across prompt groups.
        let mut iou = vec![vec![0f32; m]; n];
        for (i, db) in det_bin.iter().enumerate() {
            for (j, tb) in trk_bin.iter().enumerate() {
                if det.dets[i].prompt_id == self.obj_prompt[j] {
                    iou[i][j] = mask_iou(db, tb);
                }
            }
        }
        // tracks: unmatched if non-empty and no det IoU >= trk_assoc; empty if zero-area.
        for j in 0..m {
            let matched = (0..n).any(|i| iou[i][j] >= TRK_ASSOC_IOU_THRESH);
            if !trk_nonempty[j] {
                a.empty_trk.push(self.obj_ids[j]);
            } else if !matched {
                a.unmatched_trk.push(self.obj_ids[j]);
            }
        }
        // detections: new if score >= new_det_thresh and no track IoU >= assoc_iou.
        for i in 0..n {
            let matches_any = (0..m).any(|j| iou[i][j] >= ASSOC_IOU_THRESH);
            if det.dets[i].score >= NEW_DET_THRESH && !matches_any {
                a.new_det_inds.push(i);
            }
            let matched: Vec<i32> = (0..m)
                .filter(|&j| iou[i][j] >= ASSOC_IOU_THRESH)
                .map(|j| self.obj_ids[j])
                .collect();
            // det → max-IoU track for high-conf/high-iou recondition candidates.
            let is_new = det.dets[i].score >= NEW_DET_THRESH && !matches_any;
            let (best_j, best_iou) = (0..m).fold((0usize, -1f32), |(bj, bi), j| {
                if iou[i][j] > bi {
                    (j, iou[i][j])
                } else {
                    (bj, bi)
                }
            });
            if det.dets[i].score >= HIGH_CONF_THRESH
                && !is_new
                && best_iou >= HIGH_IOU_THRESH
                && m > 0
            {
                a.trk_id_to_max_iou_high_conf_det
                    .insert(self.obj_ids[best_j], i);
            }
            a.det_to_matched_trk.push(matched);
        }
        a
    }

    // ----- hotstart (_process_hotstart) -----
    fn process_hotstart(&mut self, frame_idx: i32, a: &Assoc, new_obj_ids: &[i32]) -> Vec<i32> {
        let mut newly_removed = Vec::new();
        let hotstart_diff = frame_idx - HOTSTART_DELAY;
        // log first-appearance + init keep-alive for new objects.
        for &oid in new_obj_ids {
            self.first_frame.entry(oid).or_insert(frame_idx);
            self.keep_alive.insert(oid, INIT_KEEP_ALIVE);
        }
        // matched tracks bump keep-alive; unmatched decrement + log.
        let mut matched: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
        for trks in &a.det_to_matched_trk {
            matched.extend(trks.iter().copied());
        }
        for &oid in &matched {
            let k = self
                .keep_alive
                .get(&oid)
                .copied()
                .unwrap_or(INIT_KEEP_ALIVE);
            self.keep_alive.insert(oid, MAX_KEEP_ALIVE.min(k + 1));
        }
        for &oid in &a.unmatched_trk {
            self.unmatched_frames
                .entry(oid)
                .or_default()
                .push(frame_idx);
            let k = self
                .keep_alive
                .get(&oid)
                .copied()
                .unwrap_or(INIT_KEEP_ALIVE);
            self.keep_alive.insert(oid, MIN_KEEP_ALIVE.max(k - 1));
        }
        // removal: unmatched for >= unmatch_thresh frames within hotstart.
        let unmatched_snapshot: Vec<(i32, usize, i32)> = self
            .unmatched_frames
            .iter()
            .map(|(&oid, fs)| (oid, fs.len(), *self.first_frame.get(&oid).unwrap_or(&0)))
            .collect();
        for (oid, count, first) in unmatched_snapshot {
            if self.removed.contains(&oid) || newly_removed.contains(&oid) {
                continue;
            }
            if count >= HOTSTART_UNMATCH && first > hotstart_diff {
                newly_removed.push(oid);
            }
        }
        // duplicate-overlap tracking + removal.
        for trks in &a.det_to_matched_trk {
            if trks.len() < 2 {
                continue;
            }
            let first_appear = *trks
                .iter()
                .min_by_key(|&&o| *self.first_frame.get(&o).unwrap_or(&0))
                .unwrap();
            for &oid in trks {
                if oid != first_appear {
                    self.overlap_pairs
                        .entry((first_appear, oid))
                        .or_default()
                        .push(frame_idx);
                }
            }
        }
        let overlap_snapshot: Vec<(i32, usize, i32)> = self
            .overlap_pairs
            .iter()
            .map(|(&(_f, oid), fs)| (oid, fs.len(), *self.first_frame.get(&oid).unwrap_or(&0)))
            .collect();
        for (oid, count, first) in overlap_snapshot {
            if self.removed.contains(&oid) || newly_removed.contains(&oid) {
                continue;
            }
            if first > hotstart_diff && count >= HOTSTART_DUP {
                newly_removed.push(oid);
            }
        }
        for &oid in &newly_removed {
            self.removed.insert(oid);
        }
        newly_removed
    }

    // ----- occlusion-based overlap suppression (_suppress_overlapping_based_on_recent_occlusion) -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (masks / obj_ids / last_occ)
    fn suppress_overlapping_recent_occlusion(
        &mut self,
        frame_idx: i32,
        trk_masks: &mut [Vec<f32>],
        removed_now: &[i32],
    ) {
        let n = trk_masks.len();
        if n == 0 {
            return;
        }
        let bin: Vec<Vec<bool>> = trk_masks.iter().map(|t| binarize(t)).collect();
        // last-occluded per object (NEVER if unseen, ALWAYS if removed this frame).
        let last_occ: Vec<i32> = (0..n)
            .map(|j| {
                let oid = self.obj_ids[j];
                self.last_occluded
                    .get(&oid)
                    .copied()
                    .unwrap_or(if removed_now.contains(&oid) {
                        ALWAYS_OCCLUDED
                    } else {
                        NEVER_OCCLUDED
                    })
            })
            .collect();
        let mut to_suppress = vec![false; n];
        // within each prompt group, suppress the more-recently-occluded of an overlapping pair.
        for pg in unique(&self.obj_prompt[0..n]) {
            let idxs: Vec<usize> = (0..n).filter(|&j| self.obj_prompt[j] == pg).collect();
            if idxs.len() <= 1 {
                continue;
            }
            for ai in 0..idxs.len() {
                for bj in (ai + 1)..idxs.len() {
                    let (i, j) = (idxs[ai], idxs[bj]);
                    if mask_iou(&bin[i], &bin[j]) < SUPPRESS_OCC_THRESH {
                        continue;
                    }
                    // suppress i if it was occluded more recently (and j was previously occluded).
                    if last_occ[i] > last_occ[j] && last_occ[j] > NEVER_OCCLUDED {
                        to_suppress[i] = true;
                    }
                    if last_occ[j] > last_occ[i] && last_occ[i] > NEVER_OCCLUDED {
                        to_suppress[j] = true;
                    }
                }
            }
        }
        // update last-occluded for occluded-or-suppressed objects; zero out suppressed masks.
        for j in 0..n {
            let occluded = !bin[j].iter().any(|&x| x);
            let oid = self.obj_ids[j];
            let new_lo = if occluded || to_suppress[j] {
                frame_idx
            } else {
                last_occ[j]
            };
            self.last_occluded.insert(oid, new_lo);
            if to_suppress[j] {
                for v in trk_masks[j].iter_mut() {
                    *v = NO_OBJ_LOGIT;
                }
            }
        }
    }

    // ----- memory encode for existing objects (_tracker_update_memories) -----
    #[allow(clippy::needless_range_loop)] // index into parallel banks / constrained masks
    fn tracker_update_memories(
        &mut self,
        frame_idx: i32,
        img_emb: &Array,
        trk_masks: &[Vec<f32>],
    ) -> Result<()> {
        let n = trk_masks.len();
        if n == 0 {
            return Ok(());
        }
        // non-overlapping constraints (per prompt group): pixel-wise argmax keep + shrink suppression.
        let constrained = suppress_pw_area_shrinkage(trk_masks, &self.obj_prompt[0..n]);
        for obj_idx in 0..n {
            let mask = &constrained[obj_idx];
            let appearing = mask.iter().any(|&v| v > 0.0);
            let object_score = if appearing { 10.0 } else { -10.0 };
            // high-res mask for the encoder = the 288² logits (encode_new_memory resizes to 1152²).
            let mask_arr = Array::from_slice(mask, &[1, 1, LOW_RES, LOW_RES]);
            let mem = self
                .tracker
                .encode_new_memory(img_emb, &mask_arr, object_score, false)?;
            // store into whichever (cond/non_cond) holds this frame for the object.
            let feat = seq_first(&mem.features, true)?;
            let pos = seq_first(&mem.pos, false)?;
            if let Some(fm) = self.banks[obj_idx].cond.get_mut(&frame_idx) {
                fm.maskmem_features = Some(feat);
                fm.maskmem_pos_enc = Some(pos);
                fm.object_score = object_score;
            } else if let Some(fm) = self.banks[obj_idx].non_cond.get_mut(&frame_idx) {
                fm.maskmem_features = Some(feat);
                fm.maskmem_pos_enc = Some(pos);
                fm.object_score = object_score;
            }
        }
        Ok(())
    }

    // ----- build outputs -----
    #[allow(clippy::needless_range_loop)] // parallel-array indexing (trk_masks / obj_ids)
    fn build_outputs(
        &self,
        det: &DetFrame,
        a: &Assoc,
        new_obj_ids: &[i32],
        trk_masks: &[Vec<f32>],
        reconditioned_obj_ids: &[i32],
    ) -> Result<VideoFrameOutput> {
        let mut obj_ids = Vec::new();
        let mut masks = Vec::new();
        // existing identities → propagated tracker masks, except reconditioned ones use the detection.
        let num_existing = trk_masks.len();
        for j in 0..num_existing {
            let oid = self.obj_ids[j];
            obj_ids.push(oid);
            if reconditioned_obj_ids.contains(&oid) {
                if let Some(&di) = a.trk_id_to_max_iou_high_conf_det.get(&oid) {
                    masks.push(det.dets[di].mask.clone());
                    continue;
                }
            }
            masks.push(trk_masks[j].clone());
        }
        // new identities → raw detection logits (hole-fill skipped: no kernels).
        for (&oid, &di) in new_obj_ids.iter().zip(&a.new_det_inds) {
            obj_ids.push(oid);
            masks.push(det.dets[di].mask.clone());
        }
        Ok(VideoFrameOutput { obj_ids, masks })
    }

    fn add_object(&mut self, obj_id: i32, prompt_id: i32) {
        self.obj_ids.push(obj_id);
        self.banks.push(ObjectBank::default());
        self.obj_prompt.push(prompt_id);
        self.max_obj_id = self.max_obj_id.max(obj_id);
    }

    fn remove_object(&mut self, obj_id: i32) {
        if let Some(idx) = self.obj_ids.iter().position(|&o| o == obj_id) {
            self.obj_ids.remove(idx);
            self.banks.remove(idx);
            self.obj_prompt.remove(idx);
        }
    }
}

// --- helpers -------------------------------------------------------------------------------------

struct DetFrame {
    dets: Vec<Detection>,
}

#[derive(Default)]
struct Assoc {
    new_det_inds: Vec<usize>,
    unmatched_trk: Vec<i32>,
    empty_trk: Vec<i32>,
    det_to_matched_trk: Vec<Vec<i32>>,
    trk_id_to_max_iou_high_conf_det: BTreeMap<i32, usize>,
}

/// `_select_closest_cond_frames`: ≤ `max` cond frames closest to `frame_idx`. Returns
/// (selected frame indices, unselected frame indices).
fn select_closest_cond_frames(
    frame_idx: i32,
    cond: &BTreeMap<i32, FrameMem>,
    max: i32,
) -> (Vec<i32>, std::collections::BTreeSet<i32>) {
    let keys: Vec<i32> = cond.keys().copied().collect();
    if max == -1 || keys.len() as i32 <= max {
        return (keys, std::collections::BTreeSet::new());
    }
    let mut selected: std::collections::BTreeSet<i32> = std::collections::BTreeSet::new();
    if let Some(&before) = keys.iter().filter(|&&t| t < frame_idx).max() {
        selected.insert(before);
    }
    if let Some(&after) = keys.iter().filter(|&&t| t >= frame_idx).min() {
        selected.insert(after);
    }
    let mut remaining: Vec<i32> = keys
        .iter()
        .copied()
        .filter(|t| !selected.contains(t))
        .collect();
    remaining.sort_by_key(|&t| (t - frame_idx).abs());
    for t in remaining
        .into_iter()
        .take((max - selected.len() as i32).max(0) as usize)
    {
        selected.insert(t);
    }
    let unselected: std::collections::BTreeSet<i32> = keys
        .iter()
        .copied()
        .filter(|t| !selected.contains(t))
        .collect();
    (selected.into_iter().collect(), unselected)
}

/// `_apply_non_overlapping_constraints` + `_suppress_shrinked_masks` per prompt group.
#[allow(clippy::needless_range_loop)] // pixel-wise argmax over parallel grouped masks
fn suppress_pw_area_shrinkage(masks: &[Vec<f32>], prompts: &[i32]) -> Vec<Vec<f32>> {
    let n = masks.len();
    let mut out = masks.to_vec();
    for pg in unique(prompts) {
        let idxs: Vec<usize> = (0..n).filter(|&j| prompts[j] == pg).collect();
        if idxs.len() <= 1 {
            continue;
        }
        let len = masks[0].len();
        // pixel-wise argmax over the group; keep only the max object's logit, clamp others to ≤ -10.
        let mut constrained: Vec<Vec<f32>> = idxs.iter().map(|&j| masks[j].clone()).collect();
        for p in 0..len {
            let (mut best, mut bv) = (0usize, f32::NEG_INFINITY);
            for (gi, &j) in idxs.iter().enumerate() {
                if masks[j][p] > bv {
                    bv = masks[j][p];
                    best = gi;
                }
            }
            for gi in 0..idxs.len() {
                if gi != best && constrained[gi][p] > NO_OBJ_LOGIT {
                    constrained[gi][p] = NO_OBJ_LOGIT;
                }
            }
        }
        // shrink suppression: if area drops below 30% after constraints, fully suppress.
        for (gi, &j) in idxs.iter().enumerate() {
            let before = masks[j].iter().filter(|&&v| v > 0.0).count().max(1) as f32;
            let after = constrained[gi].iter().filter(|&&v| v > 0.0).count() as f32;
            if after / before >= 0.3 {
                out[j] = constrained[gi].clone();
            } else {
                out[j] = masks[j].iter().map(|&v| v.min(NO_OBJ_LOGIT)).collect();
            }
        }
    }
    out
}

fn unique(v: &[i32]) -> Vec<i32> {
    let mut s: Vec<i32> = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}

/// Threshold a mask (logits or probabilities) at 0 → per-pixel bool. (Tracker masks are logits, det
/// masks are already centered at 0, so the same `> 0` rule applies to both — previously duplicated as
/// a separate `binarize_gt0`, F-071.)
fn binarize(m: &[f32]) -> Vec<bool> {
    m.iter().map(|&v| v > 0.0).collect()
}

fn mask_iou(a: &[bool], b: &[bool]) -> f32 {
    let mut inter = 0u32;
    let mut uni = 0u32;
    for (&x, &y) in a.iter().zip(b) {
        if x && y {
            inter += 1;
        }
        if x || y {
            uni += 1;
        }
    }
    inter as f32 / (uni.max(1) as f32)
}

fn to_vec(a: &Array) -> Result<Vec<f32>> {
    Ok(a.reshape(&[-1])?
        .as_dtype(Dtype::Float32)?
        .as_slice::<f32>()
        .to_vec())
}

/// Flatten the memory encoder's NHWC `[1,72,72,64]` output to seq-first `[5184,1,64]`. The reference
/// stores `maskmem_features` as **bfloat16** (`bf16 = true`) but `maskmem_pos_enc` stays f32
/// (`to(pred_masks.dtype)`), so the two must round-trip differently.
fn seq_first(a: &Array, bf16: bool) -> Result<Array> {
    let sh = a.shape();
    let (g, c) = (sh[1], sh[3]);
    let flat = a.reshape(&[g * g, 1, c])?;
    if bf16 {
        Ok(flat.as_dtype(Dtype::Bfloat16)?.as_dtype(Dtype::Float32)?)
    } else {
        Ok(flat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::weights::Weights;

    /// F-028: the detector segmenter and the tracker must share **one** PE backbone instance — both
    /// at load and after quantization — rather than each holding its own ~445M-param copy. Checks
    /// `Rc` pointer-identity of the two backbones (the cheapest, most direct proof that the weights
    /// are not duplicated). Weights-gated (no torch fixture needed — only the real `facebook/sam3`
    /// weights).
    #[test]
    #[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors>"]
    fn backbone_is_shared_not_duplicated() {
        let weights_path = std::env::var("SAM3_WEIGHTS")
            .expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
        let w = Weights::from_file(&weights_path).expect("load sam3 weights");

        let mut model = Sam3VideoModel::from_weights(&w).expect("build video model");
        assert!(
            Rc::ptr_eq(
                &model.segmenter.vision_backbone_rc(),
                &model.tracker.backbone_rc(),
            ),
            "at load: segmenter and tracker must point at one shared PE backbone",
        );

        model.quantize(8).expect("quantize q8");
        assert!(
            Rc::ptr_eq(
                &model.segmenter.vision_backbone_rc(),
                &model.tracker.backbone_rc(),
            ),
            "after quantize: the shared backbone must stay a single quantized copy",
        );
    }
}
