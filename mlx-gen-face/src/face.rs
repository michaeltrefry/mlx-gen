//! Unified `FaceAnalysis` — the one entry point that orchestrates the four native sub-models
//! (sc-3085), mirroring insightface `app.get()` so the PuLID-FLUX (epic 3069) and InstantID
//! (epic 3061) ports map 1:1. This is the payoff that removes the torch/onnx face stack.
//!
//! Pipeline (zero Python):
//! - [`FaceAnalysis::analyze`]: detector blob (cv2-faithful resize-to-fit 640 + pad + normalize) →
//!   SCRFD detect ([`crate::scrfd`]) → 5-pt `norm_crop` 112² ([`crate::align`]) → glintr100 embedding
//!   ([`crate::iresnet`]) → `Vec<Face>` sorted largest-first.
//! - [`FaceAnalysis::face_features_image`] (PuLID only): facexlib 512² align → BiSeNet parse
//!   ([`crate::bisenet`]) → background-whitened grayscale.
//!
//! ## Parity note (sc-3131)
//! Detection (bbox/kps) and parsing reproduce the reference exactly. The ArcFace embedding matches
//! onnx's *canonical* pure-numpy `ReferenceEvaluator` at cos ≥ 0.998 — i.e. the native forward is the
//! numerically-correct one; insightface's *ONNX Runtime* (CPU/MLAS) output is the one that diverges
//! (~0.98, deep-conv only). The parity bar is therefore the canonical-math one (sc-3131).

use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};
use mlx_rs::Array;

use crate::bisenet::BiSeNet;
use crate::iresnet::ArcFace;
use crate::scrfd::Scrfd;
use crate::{align, bisenet};

/// Fixed SCRFD detector input (square, matches [`crate::scrfd`]).
const DET_SIZE: i32 = 640;

/// One detected face — mirrors insightface's `Face` fields the consumers use.
#[derive(Clone, Debug)]
pub struct Face {
    /// `[x1, y1, x2, y2]` in original-image pixels.
    pub bbox: [f32; 4],
    /// 5 landmarks (L-eye, R-eye, nose, L-mouth, R-mouth) in original-image pixels.
    pub kps: [[f32; 2]; 5],
    /// SCRFD detection confidence.
    pub det_score: f32,
    /// Raw 512-d glintr100 recognition embedding (un-normalized; L2-normalize for cosine).
    pub embedding: Vec<f32>,
}

impl Face {
    fn area(&self) -> f32 {
        (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    }
}

/// cv2 `resize` `INTER_LINEAR` for an RGB `u8` HWC image — the SCRFD detector preprocessing.
/// Faithful fixed-point bilinear: half-pixel source coords `(d+0.5)·scale − 0.5`, weights quantized
/// to `INTER_RESIZE_COEF_BITS = 11`, two integer passes, `>>22` with rounding (the "interpolation is
/// a family, match the variant" discipline — cv2's resize fixed-point, distinct from PIL/warpAffine).
pub fn resize_bilinear_cv2(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
) -> Vec<u8> {
    const C: usize = 3;
    // `src` is indexed by `in_h`/`in_w`; reject a mismatched `(buf, h, w)` triple at the entry rather
    // than panic out-of-bounds in the horizontal pass (F-081).
    assert!(
        src.len() >= in_h * in_w * C,
        "resize_bilinear_cv2: src buffer of {} bytes too small for {in_h}×{in_w}×3",
        src.len()
    );
    const BITS: i64 = 11;
    const SCALE: f64 = (1i64 << BITS) as f64; // 2048

    // Per-axis taps + 11-bit weights (cvRound, ties-to-even = saturate_cast<short>).
    let coeffs = |in_n: usize, out_n: usize| {
        let scale = in_n as f64 / out_n as f64;
        let mut ofs = Vec::with_capacity(out_n);
        let mut a = Vec::with_capacity(out_n);
        for d in 0..out_n {
            let f = (d as f64 + 0.5) * scale - 0.5;
            let mut s = f.floor() as i64;
            let mut fr = f - s as f64;
            if s < 0 {
                s = 0;
                fr = 0.0;
            }
            if s >= in_n as i64 - 1 {
                s = in_n as i64 - 1;
                fr = 0.0;
            }
            let s1 = (s + 1).min(in_n as i64 - 1);
            let w1 = (fr * SCALE).round_ties_even() as i64;
            let w0 = ((1.0 - fr) * SCALE).round_ties_even() as i64;
            ofs.push((s as usize, s1 as usize));
            a.push((w0, w1));
        }
        (ofs, a)
    };

    let (xofs, xa) = coeffs(in_w, out_w);
    let (yofs, ya) = coeffs(in_h, out_h);

    // Horizontal pass over every source row → int (value·2048).
    let mut hbuf = vec![0i64; in_h * out_w * C];
    for sy in 0..in_h {
        for (dx, (&(sx, sx1), &(w0, w1))) in xofs.iter().zip(&xa).enumerate() {
            for ch in 0..C {
                hbuf[(sy * out_w + dx) * C + ch] = src[(sy * in_w + sx) * C + ch] as i64 * w0
                    + src[(sy * in_w + sx1) * C + ch] as i64 * w1;
            }
        }
    }

    // Vertical pass → uint8, (acc + 2^21) >> 22.
    let mut out = vec![0u8; out_h * out_w * C];
    for (dy, (&(sy0, sy1), &(v0, v1))) in yofs.iter().zip(&ya).enumerate() {
        for dx in 0..out_w {
            for ch in 0..C {
                let acc =
                    hbuf[(sy0 * out_w + dx) * C + ch] * v0 + hbuf[(sy1 * out_w + dx) * C + ch] * v1;
                out[(dy * out_w + dx) * C + ch] = (((acc + (1 << 21)) >> 22).clamp(0, 255)) as u8;
            }
        }
    }
    out
}

/// Build the SCRFD detector blob from an RGB `u8` image: insightface-faithful resize-to-fit 640
/// (aspect-preserving) → top-left pad to 640² → `(rgb − 127.5) / 128`. Returns the NHWC
/// `[1,640,640,3]` f32 blob and `det_scale` (= `new_h / h`).
pub fn detector_blob(img: &[u8], h: usize, w: usize) -> (Array, f32) {
    // `img` is resized/indexed by `h`/`w`; reject a mismatched `(buf, h, w)` triple at the entry
    // (F-081). `FaceAnalysis::analyze` already returns a typed error for this; the assert guards
    // direct callers of this pub primitive.
    assert!(
        img.len() >= h * w * 3,
        "detector_blob: img buffer of {} bytes too small for {h}×{w}×3",
        img.len()
    );
    let det = DET_SIZE as usize;
    let im_ratio = h as f64 / w as f64;
    let (new_w, new_h) = if im_ratio > 1.0 {
        ((det as f64 / im_ratio) as usize, det)
    } else {
        (det, (det as f64 * im_ratio) as usize)
    };
    let det_scale = new_h as f32 / h as f32;
    let resized = resize_bilinear_cv2(img, h, w, new_h, new_w);

    // top-left into a 640² canvas; normalize (rgb-127.5)/128.
    let mut blob = vec![0f32; det * det * 3];
    let norm = |v: u8| (v as f32 - 127.5) / 128.0;
    let pad0 = norm(0); // padded region = normalized 0
    for v in blob.iter_mut() {
        *v = pad0;
    }
    for y in 0..new_h {
        for x in 0..new_w {
            for ch in 0..3 {
                blob[(y * det + x) * 3 + ch] = norm(resized[(y * new_w + x) * 3 + ch]);
            }
        }
    }
    (
        Array::from_slice(&blob, &[1, det as i32, det as i32, 3]),
        det_scale,
    )
}

/// The native face-analysis stack: SCRFD + ArcFace (+ optional BiSeNet for the PuLID crop path).
pub struct FaceAnalysis {
    scrfd: Scrfd,
    arcface: ArcFace,
    parser: Option<BiSeNet>,
    /// Detection score / NMS thresholds (insightface defaults: 0.5 / 0.4).
    pub det_thresh: f32,
    pub nms_thresh: f32,
}

impl FaceAnalysis {
    /// Load the detection + recognition stack (the InstantID / ArcFace path). For the PuLID crop
    /// path, add the parser with [`FaceAnalysis::with_parser`].
    pub fn load(scrfd_weights: &Weights, arcface_weights: &Weights) -> Result<Self> {
        Ok(Self {
            scrfd: Scrfd::from_weights(scrfd_weights)?,
            arcface: ArcFace::from_weights(arcface_weights)?,
            parser: None,
            det_thresh: 0.5,
            nms_thresh: 0.4,
        })
    }

    /// Attach the BiSeNet parser (enables [`FaceAnalysis::face_features_image`]).
    pub fn with_parser(mut self, bisenet_weights: &Weights) -> Result<Self> {
        self.parser = Some(BiSeNet::from_weights(bisenet_weights)?);
        Ok(self)
    }

    /// Detect → align → embed every face in an RGB `u8` image, sorted **largest-first** (insightface
    /// `app.get()` order — PuLID uses `[0]`, the max face). `h`/`w` are the image dimensions.
    pub fn analyze(&self, img: &[u8], h: usize, w: usize) -> Result<Vec<Face>> {
        // The worker hands us a decoded buffer plus `(h, w)`; a mismatch would index out of bounds in
        // the detector/align primitives. Reject it with a typed error rather than crash generation
        // (F-081) — this is the public worker entry, so it owns the diagnosable check.
        if img.len() < h * w * 3 {
            return Err(Error::Msg(format!(
                "face analyze: img buffer of {} bytes too small for {h}×{w}×3",
                img.len()
            )));
        }
        let (blob, det_scale) = detector_blob(img, h, w);
        let dets = self
            .scrfd
            .detect(&blob, det_scale, self.det_thresh, self.nms_thresh)?;

        let mut faces = Vec::with_capacity(dets.len());
        for d in &dets {
            let crop = align::norm_crop(img, h, w, &d.kps);
            let emb = self.arcface.forward(&align::to_arcface_input(&[crop])?)?;
            faces.push(Face {
                bbox: d.bbox,
                kps: d.kps,
                det_score: d.score,
                embedding: emb
                    .try_as_slice::<f32>()
                    .map_err(|e| format!("embedding readback: {e}"))?
                    .to_vec(),
            });
        }
        faces.sort_by(|a, b| b.area().total_cmp(&a.area()));
        Ok(faces)
    }

    /// PuLID `face_features_image`: facexlib 512² align of `face` → BiSeNet parse → background
    /// whitened, foreground grayscale. NHWC `[1,512,512,3]` f32. Requires [`FaceAnalysis::with_parser`].
    pub fn face_features_image(
        &self,
        img: &[u8],
        h: usize,
        w: usize,
        face: &Face,
    ) -> Result<Array> {
        let parser = self.parser.as_ref().ok_or_else(|| {
            "face_features_image requires a BiSeNet parser (with_parser)".to_string()
        })?;
        let crop = align::align_face_512(img, h, w, &face.kps); // 512² RGB u8
        let rgb01 = u8_to_rgb01(&crop, 512, 512);
        let mask = parser.parse_mask(&bisenet::to_parse_input(&rgb01)?)?;
        bisenet::face_features_image(&rgb01, &mask)
    }
}

/// RGB `u8` HWC → NHWC `[1,H,W,3]` f32 in `[0,1]`.
fn u8_to_rgb01(crop: &[u8], h: i32, w: i32) -> Array {
    let data: Vec<f32> = crop.iter().map(|&v| v as f32 / 255.0).collect();
    Array::from_slice(&data, &[1, h, w, 3])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F-081: `resize_bilinear_cv2` indexes `src` by `in_h`/`in_w`; a buffer too small for the claimed
    /// dims is caught at the entry, not as an opaque out-of-bounds in the horizontal pass.
    #[test]
    #[should_panic(expected = "too small for 4×4×3")]
    fn resize_rejects_undersized_buffer() {
        let src = vec![0u8; 4 * 4 * 3 - 1]; // one byte short of 4×4×3
        let _ = resize_bilinear_cv2(&src, 4, 4, 8, 8);
    }

    /// A correctly-sized buffer resizes to the requested `out_h × out_w × 3`.
    #[test]
    fn resize_accepts_matched_buffer() {
        let src = vec![128u8; 4 * 4 * 3];
        let out = resize_bilinear_cv2(&src, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }
}
