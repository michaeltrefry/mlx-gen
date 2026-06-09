//! Native OpenPose (COCO-18) skeleton rasterizer (sc-3379).
//!
//! A **bit-exact** Rust port of the worker's `openpose_skeleton.py::draw_bodypose`
//! (`apps/worker/scene_worker/openpose_skeleton.py`), which renders the body-skeleton control image
//! the xinsir OpenPose-SDXL ControlNet was trained on (the `controlnet_aux draw_bodypose` format, no
//! `controlnet_aux` dependency). InstantID pose mode (sc-3117) drives the body pose with this while
//! IdentityNet anchors the face.
//!
//! The render is the same OpenCV integer rasterization as [`crate::kps::draw_kps`] — limb "sticks"
//! (rotated filled ellipses via `ellipse2Poly` + `fillConvexPoly`) then joint circles — so it reuses
//! the OpenCV primitives ported there. The two differences from `draw_kps`: the canvas is `uint8`
//! throughout (**no** ×0.6 stick dimming), and the placement/trig run in `f64` (the Python uses plain
//! float math, not `np.float32`), so the geometry is computed in `f64` here to match byte-for-byte.
//!
//! Keypoints are normalized `[0,1]` to a centered square (`square_fit`) — the poses are stored
//! square-canonical, so a non-square canvas letterboxes the margins (black = no control signal); a
//! square canvas maps 1:1. The COCO-18 order is:
//!   0 nose 1 neck 2 r_sho 3 r_elb 4 r_wri 5 l_sho 6 l_elb 7 l_wri
//!   8 r_hip 9 r_kne 10 r_ank 11 l_hip 12 l_kne 13 l_ank 14 r_eye 15 l_eye 16 r_ear 17 l_ear
//!
//! Only `draw_bodypose` is ported here: the InstantID render path
//! (`instantid_adapter.py`) imports exactly `draw_bodypose` + `face_box_from_keypoints` +
//! `normalize_keypoints`, and the bundled gallery poses are body-only. The whole-body
//! (`draw_wholebody`/`draw_handpose`/`draw_facepose`) extension is the **Z-Image strict-pose** tier's
//! renderer (sc-2257), a different consumer; it is not part of the InstantID surface and is not ported
//! here.

use mlx_gen::media::Image;

use crate::kps::{circle_filled, ellipse2poly, fill_convex_poly};

/// The number of COCO-18 body keypoints.
pub const NUM_BODY_KEYPOINTS: usize = 18;

/// One normalized `[0,1]` body keypoint, or `None` when that joint is absent/occluded.
pub type BodyPoint = Option<(f64, f64)>;

/// `limbSeq` — the 17 COCO-18 bone connections (`(from, to)` keypoint indices).
const LIMB_SEQ: [(usize, usize); 17] = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
];

/// The 18 per-limb / per-joint colors (RGB), matching `controlnet_aux`.
const COLORS: [[u8; 3]; 18] = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
];

/// The default stick width / joint radius (`openpose_skeleton.py` default `stickwidth=4`).
pub const STICKWIDTH: i32 = 4;

/// Centered-square placement for normalized `[0,1]` keypoints, returning `(side, offset_x,
/// offset_y)`. Poses are stored square-canonical, so render them into the largest centered square of
/// a (possibly non-square) canvas and letterbox the margins. A square canvas (`w == h`) maps 1:1.
/// Mirrors `openpose_skeleton.py::square_fit` (integer floor division).
pub fn square_fit(canvas_w: u32, canvas_h: u32) -> (i64, i64, i64) {
    let side = canvas_w.min(canvas_h) as i64;
    (
        side,
        (canvas_w as i64 - side) / 2,
        (canvas_h as i64 - side) / 2,
    )
}

/// Coerce a slice of optional `(x, y[, conf])` entries into exactly [`NUM_BODY_KEYPOINTS`] normalized
/// points — the `openpose_skeleton.py::normalize_keypoints` rule: `conf <= 0` drops the point to
/// `None`, the list is truncated/padded to 18. `conf` is the optional third element.
pub fn normalize_keypoints(raw: &[Option<(f64, f64, Option<f64>)>]) -> Vec<BodyPoint> {
    let mut points: Vec<BodyPoint> = raw
        .iter()
        .map(|entry| match entry {
            Some((x, y, conf)) => match conf {
                Some(c) if *c <= 0.0 => None,
                _ => Some((*x, *y)),
            },
            None => None,
        })
        .take(NUM_BODY_KEYPOINTS)
        .collect();
    points.resize(NUM_BODY_KEYPOINTS, None);
    points
}

/// Render an OpenPose (COCO-18) skeleton — a black `canvas_w × canvas_h` RGB [`Image`]: colored limb
/// sticks (rotated filled ellipses) then colored joint circles. `keypoints` are normalized `[0,1]`
/// (placed into the centered square via [`square_fit`]); absent joints (`None`) are skipped, dropping
/// any limb that touches them. Bit-exact to `openpose_skeleton.py::draw_bodypose`.
pub fn draw_bodypose(
    canvas_w: u32,
    canvas_h: u32,
    keypoints: &[BodyPoint],
    stickwidth: i32,
) -> Image {
    let (w, h) = (canvas_w as i32, canvas_h as i32);
    let mut canvas = vec![0u8; (w as usize) * (h as usize) * 3];
    let (side, ox, oy) = square_fit(canvas_w, canvas_h);

    // Place keypoints into the centered square (f64, matching the Python float math).
    let side_f = side as f64;
    let (ox_f, oy_f) = (ox as f64, oy as f64);
    let pts: Vec<Option<(f64, f64)>> = keypoints
        .iter()
        .map(|p| p.map(|(x, y)| (ox_f + x * side_f, oy_f + y * side_f)))
        .collect();

    // Limb sticks — filled rotated ellipses, color of the limb index.
    for (i, &(a, b)) in LIMB_SEQ.iter().enumerate() {
        if a >= pts.len() || b >= pts.len() {
            continue;
        }
        let (Some((xa, ya)), Some((xb, yb))) = (pts[a], pts[b]) else {
            continue;
        };
        let mx = (xa + xb) / 2.0;
        let my = (ya + yb) / 2.0;
        let length = (xa - xb).hypot(ya - yb); // math.hypot, f64
        let angle = (ya - yb).atan2(xa - xb).to_degrees(); // f64
        let center = (mx as i32, my as i32); // int() truncation
        let axes = ((length / 2.0) as i32, stickwidth);
        let poly = ellipse2poly(center, axes, angle as i32);
        fill_convex_poly(&mut canvas, w, h, &poly, COLORS[i]);
    }

    // Joint circles — radius = stickwidth, color of the joint index.
    for (i, p) in pts.iter().take(NUM_BODY_KEYPOINTS).enumerate() {
        if let Some((x, y)) = p {
            circle_filled(
                &mut canvas,
                w,
                h,
                (*x as i32, *y as i32),
                stickwidth,
                COLORS[i],
            );
        }
    }

    Image {
        width: canvas_w,
        height: canvas_h,
        pixels: canvas,
    }
}

/// `(cx, cy, height_frac)` for placing the InstantID face kps on a pose canvas, derived from the head
/// keypoints (nose / eyes / neck). Returns `None` when the head is not visible (a back view or an
/// occluded head), so the caller disables IdentityNet + the face-restoration pass and lets the shared
/// seed carry continuity. Mirrors `openpose_skeleton.py::face_box_from_keypoints` (normalized space).
pub fn face_box_from_keypoints(keypoints: &[BodyPoint]) -> Option<(f64, f64, f64)> {
    let get = |i: usize| keypoints.get(i).copied().flatten();
    let nose = get(0);
    let r_eye = get(14);
    let l_eye = get(15);
    let neck = get(1);
    let eyes: Vec<(f64, f64)> = [r_eye, l_eye].into_iter().flatten().collect();
    if nose.is_none() && eyes.is_empty() {
        return None; // no usable face landmarks
    }

    let cx = match nose {
        Some((x, _)) => x,
        None => eyes.iter().map(|e| e.0).sum::<f64>() / eyes.len() as f64,
    };
    let head_ys: Vec<f64> = [nose, r_eye, l_eye]
        .into_iter()
        .flatten()
        .map(|p| p.1)
        .collect();
    let top_y = head_ys.iter().copied().fold(f64::INFINITY, f64::min);
    // Estimate face height from the neck->nose span (head ≈ 1.4× that vertical run), else a default;
    // clamp to a small full-body face fraction.
    let mut face_h = match (neck, nose) {
        (Some((_, ny)), Some((_, nzy))) => (ny - nzy).abs() * 1.4,
        _ => 0.09,
    };
    face_h = face_h.clamp(0.045, 0.20);
    let cy = top_y + face_h * 0.45;
    Some((cx, cy, face_h))
}
