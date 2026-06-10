//! 5-point landmark alignment — the classical-CV glue (sc-3083) between the SCRFD detector
//! ([`crate::scrfd`]) and the ArcFace embedding ([`crate::iresnet`]).
//!
//! Two crops, both a *similarity transform + warp*:
//! - [`norm_crop`] (112²) — **insightface-faithful**, the fidelity-critical crop ArcFace was
//!   trained on. `glintr100`/PuLID/InstantID expect *this exact* alignment, so it must reproduce
//!   `insightface.utils.face_align.norm_crop` numerically (the gate is the downstream embedding
//!   cosine ≥ 0.9999). insightface uses `skimage` `SimilarityTransform` (Umeyama) → 2×3 `M` →
//!   `cv2.warpAffine(img, M, (112,112), borderValue=0)`.
//! - [`align_face_512`] (512²) — the facexlib `align_warp_face` equivalent for the EVA-CLIP /
//!   BiSeNet-parsing path (PuLID `face_features_image`). Per the sc-3080 spike this path is
//!   *tolerant / swappable*: facexlib aligns its own RetinaFace landmarks to an FFHQ template
//!   with `cv2.estimateAffinePartial2D` (a similarity fit ≡ Umeyama for 5 clean points). We reuse
//!   the **SCRFD** landmarks against the same template — authorized by the spike, since identity is
//!   carried by the (faithful) ArcFace crop, not this one.
//!
//! Both share one [`umeyama`] similarity solver and one [`warp_affine`] that **bit-matches**
//! `cv2.warpAffine`'s `INTER_LINEAR` path: the inverse affine in `f64`, per-pixel coordinates in
//! `AB_BITS`/`INTER_BITS` fixed-point, and the integer bilinear MAC with the 15-bit weight table.
//! Matching cv2's *fixed-point* arithmetic (not "a bilinear") is the same discipline that gives the
//! PIL resampler ([`mlx_gen::image`]) pixel-parity — an `f64` warp drifts ±1 LSB at edges.

use mlx_gen::Result;
use mlx_rs::Array;

/// insightface `arcface_dst` 112² template (L-eye, R-eye, nose, L-mouth, R-mouth).
pub const ARCFACE_DST: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// facexlib FFHQ 512² template (`FaceRestoreHelper`, `face_size=512`, `crop_ratio=(1,1)`).
/// Same 5-point order as [`ARCFACE_DST`].
pub const FACEXLIB_DST_512: [[f64; 2]; 5] = [
    [192.98138, 239.94708],
    [318.90277, 240.1936],
    [256.63416, 314.01935],
    [201.26117, 371.41043],
    [313.08905, 371.15118],
];

/// facexlib `align_warp_face` constant border `(135,133,132)` BGR → RGB `[132,133,135]`.
pub const FACEXLIB_BORDER_RGB: [u8; 3] = [132, 133, 135];

/// A 2×3 affine, row-major `[a,b,c, d,e,f]`: `x' = a·x + b·y + c`, `y' = d·x + e·y + f`.
#[derive(Clone, Copy, Debug)]
pub struct Affine2x3(pub [f64; 6]);

impl Affine2x3 {
    /// cv2 `invertAffineTransform` (the inverse map `warpAffine` applies internally when the
    /// supplied `M` is *forward*, src→dst, without `WARP_INVERSE_MAP`).
    pub fn invert(&self) -> Affine2x3 {
        let m = &self.0;
        let det = m[0] * m[4] - m[1] * m[3];
        let d = if det != 0.0 { 1.0 / det } else { 0.0 };
        let a11 = m[4] * d;
        let a22 = m[0] * d;
        let a12 = -m[1] * d;
        let a21 = -m[3] * d;
        let b1 = -a11 * m[2] - a12 * m[5];
        let b2 = -a21 * m[2] - a22 * m[5];
        Affine2x3([a11, a12, b1, a21, a22, b2])
    }
}

/// 2×2 symmetric eigendecomposition of `[[p,q],[q,r]]` → eigenvalues (descending) and the two
/// orthonormal eigenvectors as columns. Robust closed form.
fn sym_eig2(p: f64, q: f64, r: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let tr = p + r;
    let disc = ((tr * tr / 4.0 - (p * r - q * q)).max(0.0)).sqrt();
    let l1 = tr / 2.0 + disc;
    let l2 = tr / 2.0 - disc;
    let v1 = if q.abs() > 1e-30 {
        let v = [q, l1 - p];
        let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
        [v[0] / n, v[1] / n]
    } else if p >= r {
        [1.0, 0.0]
    } else {
        [0.0, 1.0]
    };
    let v2 = [-v1[1], v1[0]]; // orthonormal complement
    ([l1, l2], [v1, v2])
}

/// Estimate the least-squares 2-D similarity transform `src → dst` (rotation + uniform scale +
/// translation), faithful to `skimage.transform._umeyama(src, dst, estimate_scale=True)` (what
/// insightface `estimate_norm` calls). Full-rank case (always, for real face landmarks):
/// `R = U·diag(d)·Vᵀ` with `d = [1, sign(det A)]`, `scale = (S·d)/var`.
pub fn umeyama(src: &[[f64; 2]], dst: &[[f64; 2]]) -> Affine2x3 {
    let n = src.len() as f64;
    let mean = |p: &[[f64; 2]]| {
        let mut m = [0.0; 2];
        for q in p {
            m[0] += q[0];
            m[1] += q[1];
        }
        [m[0] / n, m[1] / n]
    };
    let sm = mean(src);
    let dm = mean(dst);

    // A = dst_demean^T @ src_demean / n  (2×2); var = mean |src_demean|^2.
    let mut a = [[0.0f64; 2]; 2];
    let mut var = 0.0f64;
    for (s, d) in src.iter().zip(dst) {
        let sx = s[0] - sm[0];
        let sy = s[1] - sm[1];
        let dx = d[0] - dm[0];
        let dy = d[1] - dm[1];
        a[0][0] += dx * sx;
        a[0][1] += dx * sy;
        a[1][0] += dy * sx;
        a[1][1] += dy * sy;
        var += sx * sx + sy * sy;
    }
    for row in &mut a {
        for v in row {
            *v /= n;
        }
    }
    var /= n;

    let det_a = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    let d = [1.0, if det_a < 0.0 { -1.0 } else { 1.0 }];

    // SVD of A via the symmetric eigendecomposition of AᵀA: V = eigenvectors, σ = sqrt(λ),
    // U = A·V·diag(1/σ). The Umeyama result is gauge-invariant, so any consistent SVD works.
    let p = a[0][0] * a[0][0] + a[1][0] * a[1][0];
    let q = a[0][0] * a[0][1] + a[1][0] * a[1][1];
    let r = a[0][1] * a[0][1] + a[1][1] * a[1][1];
    let (lam, v) = sym_eig2(p, q, r); // v[i] is the i-th eigenvector (a column of V)
    let sigma = [lam[0].max(0.0).sqrt(), lam[1].max(0.0).sqrt()];

    // U columns u_i = A v_i / σ_i.
    let mut u = [[0.0f64; 2]; 2];
    for i in 0..2 {
        let av = [
            a[0][0] * v[i][0] + a[0][1] * v[i][1],
            a[1][0] * v[i][0] + a[1][1] * v[i][1],
        ];
        let s = if sigma[i] > 1e-30 { sigma[i] } else { 1.0 };
        u[i] = [av[0] / s, av[1] / s];
    }

    // R = U diag(d) Vᵀ = Σ_i d_i · u_i ⊗ v_i   (u_i, v_i are columns).
    let mut rot = [[0.0f64; 2]; 2];
    for i in 0..2 {
        for (a_idx, rrow) in rot.iter_mut().enumerate() {
            for (b_idx, rv) in rrow.iter_mut().enumerate() {
                *rv += d[i] * u[i][a_idx] * v[i][b_idx];
            }
        }
    }

    let scale = if var > 0.0 {
        (sigma[0] * d[0] + sigma[1] * d[1]) / var
    } else {
        1.0
    };

    let m00 = scale * rot[0][0];
    let m01 = scale * rot[0][1];
    let m10 = scale * rot[1][0];
    let m11 = scale * rot[1][1];
    let tx = dm[0] - (m00 * sm[0] + m01 * sm[1]);
    let ty = dm[1] - (m10 * sm[0] + m11 * sm[1]);
    Affine2x3([m00, m01, tx, m10, m11, ty])
}

/// Similarity transform from 5 detected landmarks to a fixed template.
pub fn estimate_similarity(kps: &[[f32; 2]; 5], dst: &[[f64; 2]; 5]) -> Affine2x3 {
    let src: Vec<[f64; 2]> = kps.iter().map(|p| [p[0] as f64, p[1] as f64]).collect();
    umeyama(&src, dst)
}

/// insightface `estimate_norm` for the 112² ArcFace crop (the `arcface_dst` template).
pub fn estimate_norm(kps: &[[f32; 2]; 5]) -> Affine2x3 {
    estimate_similarity(kps, &ARCFACE_DST)
}

// cv2 warpAffine INTER_LINEAR fixed-point constants.
const AB_BITS: i64 = 10; // matrix → fixed point
const INTER_BITS: i64 = 5; // sub-pixel fraction (INTER_TAB_SIZE = 32)
const INTER_TAB: i64 = 1 << INTER_BITS; // 32
const AB_SCALE: f64 = (1i64 << AB_BITS) as f64; // 1024
const ROUND_DELTA: i64 = (1 << AB_BITS) / (1 << INTER_BITS) / 2; // 16
const REMAP_BITS: i64 = 15; // bilinear weight table scale (1<<15)
const REMAP_HALF: i64 = 1 << (REMAP_BITS - 1); // 16384 (output rounding bias)
const WSCALE: i64 = 1 << (REMAP_BITS - 2 * INTER_BITS); // 32: per-axis 1/32 weights → 1<<15 table

/// cvRound: round to nearest, ties to even (matches `saturate_cast<int>(double)`).
#[inline]
fn cv_round(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Warp an RGB `u8` HWC image by a *forward* (src→dst) 2×3 affine into `out_h × out_w`,
/// **bit-matching** `cv2.warpAffine(..., INTER_LINEAR, BORDER_CONSTANT=border)`. Returns RGB `u8`.
pub fn warp_affine(
    src: &[u8],
    in_h: usize,
    in_w: usize,
    forward: &Affine2x3,
    out_h: usize,
    out_w: usize,
    border: [u8; 3],
) -> Vec<u8> {
    // `src` is indexed as `(y·in_w + x)·3 + ch` with bounds from `in_h`/`in_w` only; a `(buf,h,w)`
    // mismatch would otherwise index out of bounds deep in the sample loop. Check the contract at the
    // entry so the failure is labeled here (F-081). Callers that decode their own buffers uphold it.
    assert!(
        src.len() >= in_h * in_w * 3,
        "warp_affine: src buffer of {} bytes too small for {in_h}×{in_w}×3",
        src.len()
    );
    let im = forward.invert().0; // src_x = im0·x+im1·y+im2 ; src_y = im3·x+im4·y+im5
    let adelta: Vec<i64> = (0..out_w as i64)
        .map(|x| cv_round(im[0] * x as f64 * AB_SCALE))
        .collect();
    let bdelta: Vec<i64> = (0..out_w as i64)
        .map(|x| cv_round(im[3] * x as f64 * AB_SCALE))
        .collect();

    let sample = |px: i64, py: i64, ch: usize| -> i64 {
        if px >= 0 && (px as usize) < in_w && py >= 0 && (py as usize) < in_h {
            src[(py as usize * in_w + px as usize) * 3 + ch] as i64
        } else {
            border[ch] as i64
        }
    };

    let mut out = vec![0u8; out_h * out_w * 3];
    for y in 0..out_h as i64 {
        let x0 = cv_round((im[1] * y as f64 + im[2]) * AB_SCALE) + ROUND_DELTA;
        let y0 = cv_round((im[4] * y as f64 + im[5]) * AB_SCALE) + ROUND_DELTA;
        for x in 0..out_w {
            let xq = (x0 + adelta[x]) >> (AB_BITS - INTER_BITS);
            let yq = (y0 + bdelta[x]) >> (AB_BITS - INTER_BITS);
            let sx = xq >> INTER_BITS;
            let sy = yq >> INTER_BITS;
            let fx = xq & (INTER_TAB - 1);
            let fy = yq & (INTER_TAB - 1);
            let w_tl = (INTER_TAB - fy) * (INTER_TAB - fx) * WSCALE;
            let w_tr = (INTER_TAB - fy) * fx * WSCALE;
            let w_bl = fy * (INTER_TAB - fx) * WSCALE;
            let w_br = fy * fx * WSCALE;
            let base = (y as usize * out_w + x) * 3;
            for ch in 0..3 {
                let acc = sample(sx, sy, ch) * w_tl
                    + sample(sx + 1, sy, ch) * w_tr
                    + sample(sx, sy + 1, ch) * w_bl
                    + sample(sx + 1, sy + 1, ch) * w_br;
                out[base + ch] = ((acc + REMAP_HALF) >> REMAP_BITS).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// insightface `norm_crop`: 5-pt kps → 112² aligned RGB `u8` crop (the ArcFace input image,
/// `borderValue=0`). Feed [`to_arcface_input`] then [`crate::ArcFace::forward`].
pub fn norm_crop(src: &[u8], in_h: usize, in_w: usize, kps: &[[f32; 2]; 5]) -> Vec<u8> {
    warp_affine(src, in_h, in_w, &estimate_norm(kps), 112, 112, [0, 0, 0])
}

/// facexlib `align_warp_face` equivalent: 5-pt kps → 512² aligned RGB `u8` crop against the FFHQ
/// template (gray border). The EVA-CLIP / parsing crop for PuLID (tolerant path — see module docs).
pub fn align_face_512(src: &[u8], in_h: usize, in_w: usize, kps: &[[f32; 2]; 5]) -> Vec<u8> {
    let m = estimate_similarity(kps, &FACEXLIB_DST_512);
    warp_affine(src, in_h, in_w, &m, 512, 512, FACEXLIB_BORDER_RGB)
}

/// Pack 112² RGB `u8` crops into the ArcFace input batch: NHWC `[N,112,112,3]` f32, normalized
/// `(rgb - 127.5) / 127.5` (the antelopev2 ArcFace blob, swapRB already folded in by warping RGB).
pub fn to_arcface_input(crops: &[Vec<u8>]) -> Result<Array> {
    let n = crops.len() as i32;
    let mut data = Vec::with_capacity(crops.len() * 112 * 112 * 3);
    for crop in crops {
        debug_assert_eq!(crop.len(), 112 * 112 * 3, "crop must be 112×112×3");
        data.extend(crop.iter().map(|&v| (v as f32 - 127.5) / 127.5));
    }
    Ok(Array::from_slice(&data, &[n, 112, 112, 3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The similarity solver must agree with skimage on a known transform: map the template by a
    /// fixed scale·R + translation to synthesize landmarks (in f64, isolating the solver from the
    /// f32 kps quantization), then recover the inverse `(1/s)·Rᵀ` to f64 precision. (Real f32 kps
    /// add ~1e-6; the authoritative tolerance lives in the golden test vs insightface `estimate_norm`.)
    #[test]
    fn umeyama_recovers_a_known_similarity() {
        let (s, ct, st, tx, ty) = (1.7f64, 0.8f64, 0.6f64, 12.0f64, -5.0f64); // scale, cosθ, sinθ
        let src: Vec<[f64; 2]> = ARCFACE_DST
            .iter()
            .map(|p| {
                [
                    s * (ct * p[0] - st * p[1]) + tx,
                    s * (st * p[0] + ct * p[1]) + ty,
                ]
            })
            .collect();
        // fit src -> template, i.e. the inverse of the map above.
        let m = umeyama(&src, &ARCFACE_DST).0;
        let inv_s = 1.0 / s;
        let expect = [
            inv_s * ct,
            inv_s * st,
            -inv_s * (ct * tx + st * ty),
            -inv_s * st,
            inv_s * ct,
            inv_s * (st * tx - ct * ty),
        ];
        for (g, e) in m.iter().zip(&expect) {
            assert!((g - e).abs() < 1e-9, "umeyama {m:?} vs {expect:?}");
        }
    }

    /// Warping by the identity must return the source unchanged (sanity for the fixed-point path).
    #[test]
    fn warp_identity_is_lossless() {
        let mut src = vec![0u8; 8 * 8 * 3];
        for (i, v) in src.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let id = Affine2x3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let out = warp_affine(&src, 8, 8, &id, 8, 8, [0, 0, 0]);
        assert_eq!(out, src, "identity warp must be lossless");
    }

    /// F-081: a `(buf, h, w)` mismatch (buffer too small for the claimed dims) is caught at the entry
    /// with a labeled message, not an opaque out-of-bounds index deep in the sample loop.
    #[test]
    #[should_panic(expected = "too small for 8×8×3")]
    fn warp_affine_rejects_undersized_buffer() {
        let src = vec![0u8; 8 * 8 * 3 - 1]; // one byte short of 8×8×3
        let id = Affine2x3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let _ = warp_affine(&src, 8, 8, &id, 8, 8, [0, 0, 0]);
    }
}
