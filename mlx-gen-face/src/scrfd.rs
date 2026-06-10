//! SCRFD-10g face detector — a native MLX port of antelopev2 `scrfd_10g_bnkps` (sc-3082).
//!
//! Produces, per face, a bounding box + 5-point landmarks (the kps that drive both the ArcFace
//! `norm_crop` alignment and InstantID's `draw_kps`). The network is a ResNet-style backbone + a
//! PAFPN neck + per-stride detection heads; the decode (anchor centres + `distance2bbox` /
//! `distance2kps` + NMS) is plain host math. Weights come from [`tools/convert_scrfd.py`].
//!
//! ## Architecture (verified from the onnx graph; BatchNorm folded into biased convs)
//! - **stem**: `Conv(3→28,3×3,s2)+Relu → Conv(28→28)+Relu → Conv(28→56)+Relu → MaxPool2×2 s2`.
//! - **backbone** blocks `[3,4,2,3]` (channels 56/88/88/224). Block = `Conv(c1,3×3,stride)+Relu →
//!   Conv(c2,3×3,s1) → + identity → Relu`; stages 2–4 block 0 has stride 2 + a downsample
//!   (`AvgPool2×2 s2 → Conv1×1`) on the identity (stage 1 has none). Taps: C2 (s8), C3 (s16), C4 (s32).
//! - **neck (PAFPN)**: lateral 1×1 (→56) on C2/C3/C4; top-down ×2 nearest-upsample + add; fpn 3×3
//!   convs; bottom-up downsample 3×3 s2 + add; pafpn 3×3 convs. Head inputs: s8 = fpn0(P3),
//!   s16 = pafpn0(N4), s32 = pafpn1(N5).
//! - **heads** (per stride): `3×(Conv 3×3 +Relu →80)` then `cls(→2)` (sigmoid), `reg(→8)` (× learned
//!   per-level scale), `kps(→20)`; reshaped to `[-1,1]/[-1,4]/[-1,10]`. num_anchors=2.
//!
//! Run at a fixed **640×640** input (NHWC, normalized `(rgb-127.5)/128`); the reference's dynamic
//! Resize collapses to exact ×2 upsamples there. fp32; conv weights MLX-native OHWI.

use mlx_gen::array::scalar;
use mlx_gen::nn;
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::ops::{add, max_axes, maximum, mean_axes, multiply, sigmoid};
use mlx_rs::Array;

/// Backbone residual-block counts per stage.
const STAGE_BLOCKS: [usize; 4] = [3, 4, 2, 3];
/// Fixed detector input side.
pub const DET_SIZE: i32 = 640;
const NUM_ANCHORS: usize = 2;

/// A single detected face (640-space coords unless rescaled): box, 5 landmarks, score.
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: [f32; 4], // x1, y1, x2, y2
    pub kps: [[f32; 2]; 5],
    pub score: f32,
}

fn relu(x: &Array) -> Result<Array> {
    Ok(maximum(x, scalar(0.0))?)
}

/// 2×2 stride-2 pooling over NHWC via reshape + reduce over the two size-2 axes (exact for even dims).
fn pool2x2(x: &Array, avg: bool) -> Result<Array> {
    let s = x.shape();
    let (n, h, w, c) = (s[0], s[1], s[2], s[3]);
    let r = x.reshape(&[n, h / 2, 2, w / 2, 2, c])?;
    Ok(if avg {
        mean_axes(&r, &[2, 4], false)?
    } else {
        max_axes(&r, &[2, 4], false)?
    })
}

struct Conv {
    w: Array,
    b: Array,
}

impl Conv {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }
    fn forward(&self, x: &Array, stride: i32, padding: i32) -> Result<Array> {
        nn::conv2d(x, &self.w, Some(&self.b), stride, padding)
    }
}

/// `Conv(c1,3×3,stride)+Relu → Conv(c2,3×3,s1) → + identity(/downsample) → Relu`.
struct Block {
    conv1: Conv,
    stride: i32,
    conv2: Conv,
    downsample: Option<Conv>, // AvgPool2×2 already applied to its input
}

impl Block {
    fn forward(&self, x: &Array) -> Result<Array> {
        let t = relu(&self.conv1.forward(x, self.stride, 1)?)?;
        let t = self.conv2.forward(&t, 1, 1)?;
        let identity = match &self.downsample {
            Some(ds) => ds.forward(&pool2x2(x, true)?, 1, 0)?,
            None => x.clone(),
        };
        relu(&add(&t, &identity)?)
    }
}

struct Head {
    stride: i32,
    scale: f32,
    stem: [Conv; 3],
    cls: Conv,
    reg: Conv,
    kps: Conv,
}

/// Raw per-stride head outputs: scores `[N,1]` (sigmoid), bbox `[N,4]` (× scale), kps `[N,10]`.
struct StrideOut {
    stride: i32,
    scores: Array,
    bbox: Array,
    kps: Array,
}

impl Head {
    fn load(w: &Weights, stride: i32) -> Result<Self> {
        let p = format!("head{stride}");
        Ok(Self {
            stride,
            scale: w.require(&format!("{p}.scale"))?.item::<f32>(),
            stem: [
                Conv::load(w, &format!("{p}.stem0"))?,
                Conv::load(w, &format!("{p}.stem1"))?,
                Conv::load(w, &format!("{p}.stem2"))?,
            ],
            cls: Conv::load(w, &format!("{p}.cls"))?,
            reg: Conv::load(w, &format!("{p}.reg"))?,
            kps: Conv::load(w, &format!("{p}.kps"))?,
        })
    }

    fn forward(&self, x: &Array) -> Result<StrideOut> {
        let mut h = x.clone();
        for c in &self.stem {
            h = relu(&c.forward(&h, 1, 1)?)?;
        }
        // NHWC [1,H,W,C] → [-1,K]; (h,w,anchor) row order matches the onnx transpose+reshape.
        let scores = sigmoid(&self.cls.forward(&h, 1, 1)?.reshape(&[-1, 1])?)?;
        let bbox = multiply(
            &self.reg.forward(&h, 1, 1)?.reshape(&[-1, 4])?,
            scalar(self.scale),
        )?;
        let kps = self.kps.forward(&h, 1, 1)?.reshape(&[-1, 10])?;
        Ok(StrideOut {
            stride: self.stride,
            scores,
            bbox,
            kps,
        })
    }
}

/// SCRFD-10g detector.
pub struct Scrfd {
    stem: [Conv; 3],
    stages: Vec<Vec<Block>>,
    lateral: [Conv; 3],
    fpn: [Conv; 3],
    down: [Conv; 2],
    pafpn: [Conv; 2],
    heads: [Head; 3],
}

impl Scrfd {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let mut stages = Vec::with_capacity(STAGE_BLOCKS.len());
        for (si, &nb) in STAGE_BLOCKS.iter().enumerate() {
            let l = si + 1;
            let mut blocks = Vec::with_capacity(nb);
            for b in 0..nb {
                let p = format!("stage{l}.{b}");
                // stages 2-4 block 0: stride 2 + downsample; everything else stride 1, no downsample.
                let has_ds = b == 0 && l > 1;
                blocks.push(Block {
                    conv1: Conv::load(w, &format!("{p}.conv1"))?,
                    stride: if has_ds { 2 } else { 1 },
                    conv2: Conv::load(w, &format!("{p}.conv2"))?,
                    downsample: if has_ds {
                        Some(Conv::load(w, &format!("{p}.downsample"))?)
                    } else {
                        None
                    },
                });
            }
            stages.push(blocks);
        }
        Ok(Self {
            stem: [
                Conv::load(w, "stem.conv0")?,
                Conv::load(w, "stem.conv1")?,
                Conv::load(w, "stem.conv2")?,
            ],
            stages,
            lateral: [
                Conv::load(w, "neck.lateral0")?,
                Conv::load(w, "neck.lateral1")?,
                Conv::load(w, "neck.lateral2")?,
            ],
            fpn: [
                Conv::load(w, "neck.fpn0")?,
                Conv::load(w, "neck.fpn1")?,
                Conv::load(w, "neck.fpn2")?,
            ],
            down: [Conv::load(w, "neck.down0")?, Conv::load(w, "neck.down1")?],
            pafpn: [Conv::load(w, "neck.pafpn0")?, Conv::load(w, "neck.pafpn1")?],
            heads: [Head::load(w, 8)?, Head::load(w, 16)?, Head::load(w, 32)?],
        })
    }

    /// Backbone → (C2 s8, C3 s16, C4 s32).
    fn backbone(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let mut h = relu(&self.stem[0].forward(x, 2, 1)?)?;
        h = relu(&self.stem[1].forward(&h, 1, 1)?)?;
        h = relu(&self.stem[2].forward(&h, 1, 1)?)?;
        h = pool2x2(&h, false)?; // maxpool
        let mut taps = Vec::new();
        for stage in &self.stages {
            for blk in stage {
                h = blk.forward(&h)?;
            }
            taps.push(h.clone());
        }
        // taps = [stage1, stage2(C2), stage3(C3), stage4(C4)]
        Ok((taps[1].clone(), taps[2].clone(), taps[3].clone()))
    }

    /// Full network → the 3 per-stride raw outputs.
    fn forward(&self, x: &Array) -> Result<[StrideOut; 3]> {
        let (c2, c3, c4) = self.backbone(x)?;
        let l0 = self.lateral[0].forward(&c2, 1, 0)?;
        let l1 = self.lateral[1].forward(&c3, 1, 0)?;
        let l2 = self.lateral[2].forward(&c4, 1, 0)?;
        // top-down (×2 nearest upsample + add)
        let p4 = add(&l1, &nn::upsample_nearest(&l2, 2)?)?;
        let p3 = add(&l0, &nn::upsample_nearest(&p4, 2)?)?;
        let f0 = self.fpn[0].forward(&p3, 1, 1)?;
        let f1 = self.fpn[1].forward(&p4, 1, 1)?;
        let f2 = self.fpn[2].forward(&l2, 1, 1)?;
        // bottom-up (downsample 3×3 s2 + add)
        let n4 = add(&f1, &self.down[0].forward(&f0, 2, 1)?)?;
        let n5 = add(&f2, &self.down[1].forward(&n4, 2, 1)?)?;
        let out8 = f0;
        let out16 = self.pafpn[0].forward(&n4, 1, 1)?;
        let out32 = self.pafpn[1].forward(&n5, 1, 1)?;
        Ok([
            self.heads[0].forward(&out8)?,
            self.heads[1].forward(&out16)?,
            self.heads[2].forward(&out32)?,
        ])
    }

    /// Test/debug hook: the raw per-stride `(stride, scores[N,1], bbox[N,4], kps[N,10])` outputs,
    /// matching the onnx graph outputs (scores sigmoided, bbox scaled). Used by the network-parity test.
    pub fn raw_outputs(&self, x: &Array) -> Result<Vec<(i32, Array, Array, Array)>> {
        Ok(self
            .forward(x)?
            .into_iter()
            .map(|o| (o.stride, o.scores, o.bbox, o.kps))
            .collect())
    }

    /// Detect faces in a preprocessed `[1,640,640,3]` NHWC f32 image (`(rgb-127.5)/128`).
    ///
    /// `det_scale` maps 640-space coords back to the original image (divide); pass 1.0 to keep
    /// 640-space. Returns NMS-filtered detections sorted by score (descending).
    pub fn detect(
        &self,
        x: &Array,
        det_scale: f32,
        score_thresh: f32,
        nms_thresh: f32,
    ) -> Result<Vec<Detection>> {
        let outs = self.forward(x)?;
        let mut dets: Vec<Detection> = Vec::new();
        for out in &outs {
            let s = out.stride;
            let w = (DET_SIZE / s) as usize;
            let readback = |a: &Array| -> Result<Vec<f32>> {
                Ok(a.try_as_slice::<f32>()
                    .map_err(|e| format!("scrfd output readback: {e}"))?
                    .to_vec())
            };
            let scores = readback(&out.scores)?;
            let bbox = readback(&out.bbox)?;
            let kps = readback(&out.kps)?;
            let sf = s as f32;
            for (r, &score) in scores.iter().enumerate() {
                // Drop non-finite AND below-threshold scores: a NaN off a corrupted checkpoint or
                // numeric blow-up passes `score < thresh` (NaN comparisons are false) and would later
                // panic the NMS sort, crashing every consumer (F-078).
                if !score.is_finite() || score < score_thresh {
                    continue;
                }
                // anchor centre: row r → (cell, anchor); cell = r / num_anchors; (h,w) = (cell/W, cell%W)
                let cell = r / NUM_ANCHORS;
                let cx = (cell % w) as f32 * sf;
                let cy = (cell / w) as f32 * sf;
                let d = &bbox[r * 4..r * 4 + 4];
                let bb = [
                    cx - d[0] * sf,
                    cy - d[1] * sf,
                    cx + d[2] * sf,
                    cy + d[3] * sf,
                ];
                let kp = &kps[r * 10..r * 10 + 10];
                let mut pts = [[0.0f32; 2]; 5];
                for (i, p) in pts.iter_mut().enumerate() {
                    *p = [cx + kp[i * 2] * sf, cy + kp[i * 2 + 1] * sf];
                }
                dets.push(Detection {
                    bbox: bb,
                    kps: pts,
                    score,
                });
            }
        }
        let mut kept = nms(dets, nms_thresh);
        if det_scale != 1.0 {
            let inv = 1.0 / det_scale;
            for d in &mut kept {
                for v in &mut d.bbox {
                    *v *= inv;
                }
                for p in &mut d.kps {
                    p[0] *= inv;
                    p[1] *= inv;
                }
            }
        }
        Ok(kept)
    }
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = a[2].min(b[2]);
    let y2 = a[3].min(b[3]);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0);
    let area_b = (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Greedy NMS by descending score (insightface uses IoU threshold 0.4).
fn nms(mut dets: Vec<Detection>, thresh: f32) -> Vec<Detection> {
    // `total_cmp` (not `partial_cmp().unwrap()`) so a NaN score can never panic the sort — scores
    // come straight off a model readback (F-078). Decode already drops non-finite scores; this is the
    // belt-and-suspenders guard for the production runtime path.
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut keep: Vec<Detection> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| iou(&k.bbox, &d.bbox) <= thresh) {
            keep.push(d);
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(score: f32, x: f32) -> Detection {
        // Disjoint 10×10 boxes (stride 100) so NMS keeps them all — isolates the score sort.
        Detection {
            bbox: [x, 0.0, x + 10.0, 10.0],
            kps: [[0.0; 2]; 5],
            score,
        }
    }

    /// F-078: a NaN score must not panic the NMS sort (`total_cmp`, not `partial_cmp().unwrap()`),
    /// and the finite scores must still come out in descending order.
    #[test]
    fn nms_sorts_descending_and_survives_nan() {
        let dets = vec![
            det(0.5, 0.0),
            det(f32::NAN, 100.0),
            det(0.9, 200.0),
            det(0.7, 300.0),
        ];
        let kept = nms(dets, 0.4); // disjoint boxes ⇒ all retained, just reordered
        assert_eq!(kept.len(), 4);
        // The three finite scores are descending; the NaN sorts deterministically (total order) and
        // never panics — that's the regression being guarded.
        let finite: Vec<f32> = kept
            .iter()
            .map(|d| d.score)
            .filter(|s| s.is_finite())
            .collect();
        assert_eq!(finite, vec![0.9, 0.7, 0.5]);
    }
}
