//! InstantID keypoint control-image renderer (sc-3111).
//!
//! A **bit-exact** Rust port of the vendored `draw_kps`
//! (`_vendor/instantid/pipeline_stable_diffusion_xl_instantid.py`), which rasterizes the 5-point
//! facial-landmark image IdentityNet consumes: on a black canvas, four limb "sticks" (rotated filled
//! ellipses, color of the limb's first kp, the whole stick layer dimmed ×0.6) then five filled
//! circles (radius 10) at the keypoints. Because IdentityNet was trained on these exact images, the
//! rasterization must match OpenCV's integer drawing. The three primitives are faithful ports of
//! OpenCV 4.13 `drawing.cpp`: `ellipse2Poly` (float `SinTable` + `cvRound` round-half-to-even),
//! `fillConvexPoly` (XY_SHIFT=16 fixed-point scanline fill + the `Line` outline), and the filled
//! `Circle` (Bresenham span fill). Validated pixel-for-pixel vs cv2 in `tests/instantid_kps.rs`.
//!
//! Also here: [`letterbox`] (resize-keep-aspect + center-pad, the sc-2009 kps-distortion rule) and the
//! canonical [`VIEW_ANGLE_KPS`] multi-view landmark sets.

use mlx_gen::image::resize_lanczos_u8;
use mlx_gen::media::Image;

const XY_SHIFT: i64 = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

/// OpenCV `cvRound`: round to nearest, ties to even (the default FP rounding mode on x86/ARM).
#[inline]
fn cv_round(x: f64) -> i32 {
    x.round_ties_even() as i32
}

/// OpenCV `drawing.cpp` `SinTable` — `sin(k°)` as **float** constants, k = 0..450. The drawing code
/// reads trig from this table (not from `libm`), so reproducing it bit-for-bit needs these *exact*
/// literals (hence the lint allows: trimming precision or substituting `FRAC_1_SQRT_2` could land on a
/// different f32 and break parity).
#[rustfmt::skip]
#[allow(clippy::excessive_precision, clippy::approx_constant)]
static SIN_TABLE: [f32; 451] = [
    0.0000000, 0.0174524, 0.0348995, 0.0523360, 0.0697565, 0.0871557,
    0.1045285, 0.1218693, 0.1391731, 0.1564345, 0.1736482, 0.1908090,
    0.2079117, 0.2249511, 0.2419219, 0.2588190, 0.2756374, 0.2923717,
    0.3090170, 0.3255682, 0.3420201, 0.3583679, 0.3746066, 0.3907311,
    0.4067366, 0.4226183, 0.4383711, 0.4539905, 0.4694716, 0.4848096,
    0.5000000, 0.5150381, 0.5299193, 0.5446390, 0.5591929, 0.5735764,
    0.5877853, 0.6018150, 0.6156615, 0.6293204, 0.6427876, 0.6560590,
    0.6691306, 0.6819984, 0.6946584, 0.7071068, 0.7193398, 0.7313537,
    0.7431448, 0.7547096, 0.7660444, 0.7771460, 0.7880108, 0.7986355,
    0.8090170, 0.8191520, 0.8290376, 0.8386706, 0.8480481, 0.8571673,
    0.8660254, 0.8746197, 0.8829476, 0.8910065, 0.8987940, 0.9063078,
    0.9135455, 0.9205049, 0.9271839, 0.9335804, 0.9396926, 0.9455186,
    0.9510565, 0.9563048, 0.9612617, 0.9659258, 0.9702957, 0.9743701,
    0.9781476, 0.9816272, 0.9848078, 0.9876883, 0.9902681, 0.9925462,
    0.9945219, 0.9961947, 0.9975641, 0.9986295, 0.9993908, 0.9998477,
    1.0000000, 0.9998477, 0.9993908, 0.9986295, 0.9975641, 0.9961947,
    0.9945219, 0.9925462, 0.9902681, 0.9876883, 0.9848078, 0.9816272,
    0.9781476, 0.9743701, 0.9702957, 0.9659258, 0.9612617, 0.9563048,
    0.9510565, 0.9455186, 0.9396926, 0.9335804, 0.9271839, 0.9205049,
    0.9135455, 0.9063078, 0.8987940, 0.8910065, 0.8829476, 0.8746197,
    0.8660254, 0.8571673, 0.8480481, 0.8386706, 0.8290376, 0.8191520,
    0.8090170, 0.7986355, 0.7880108, 0.7771460, 0.7660444, 0.7547096,
    0.7431448, 0.7313537, 0.7193398, 0.7071068, 0.6946584, 0.6819984,
    0.6691306, 0.6560590, 0.6427876, 0.6293204, 0.6156615, 0.6018150,
    0.5877853, 0.5735764, 0.5591929, 0.5446390, 0.5299193, 0.5150381,
    0.5000000, 0.4848096, 0.4694716, 0.4539905, 0.4383711, 0.4226183,
    0.4067366, 0.3907311, 0.3746066, 0.3583679, 0.3420201, 0.3255682,
    0.3090170, 0.2923717, 0.2756374, 0.2588190, 0.2419219, 0.2249511,
    0.2079117, 0.1908090, 0.1736482, 0.1564345, 0.1391731, 0.1218693,
    0.1045285, 0.0871557, 0.0697565, 0.0523360, 0.0348995, 0.0174524,
    0.0000000, -0.0174524, -0.0348995, -0.0523360, -0.0697565, -0.0871557,
    -0.1045285, -0.1218693, -0.1391731, -0.1564345, -0.1736482, -0.1908090,
    -0.2079117, -0.2249511, -0.2419219, -0.2588190, -0.2756374, -0.2923717,
    -0.3090170, -0.3255682, -0.3420201, -0.3583679, -0.3746066, -0.3907311,
    -0.4067366, -0.4226183, -0.4383711, -0.4539905, -0.4694716, -0.4848096,
    -0.5000000, -0.5150381, -0.5299193, -0.5446390, -0.5591929, -0.5735764,
    -0.5877853, -0.6018150, -0.6156615, -0.6293204, -0.6427876, -0.6560590,
    -0.6691306, -0.6819984, -0.6946584, -0.7071068, -0.7193398, -0.7313537,
    -0.7431448, -0.7547096, -0.7660444, -0.7771460, -0.7880108, -0.7986355,
    -0.8090170, -0.8191520, -0.8290376, -0.8386706, -0.8480481, -0.8571673,
    -0.8660254, -0.8746197, -0.8829476, -0.8910065, -0.8987940, -0.9063078,
    -0.9135455, -0.9205049, -0.9271839, -0.9335804, -0.9396926, -0.9455186,
    -0.9510565, -0.9563048, -0.9612617, -0.9659258, -0.9702957, -0.9743701,
    -0.9781476, -0.9816272, -0.9848078, -0.9876883, -0.9902681, -0.9925462,
    -0.9945219, -0.9961947, -0.9975641, -0.9986295, -0.9993908, -0.9998477,
    -1.0000000, -0.9998477, -0.9993908, -0.9986295, -0.9975641, -0.9961947,
    -0.9945219, -0.9925462, -0.9902681, -0.9876883, -0.9848078, -0.9816272,
    -0.9781476, -0.9743701, -0.9702957, -0.9659258, -0.9612617, -0.9563048,
    -0.9510565, -0.9455186, -0.9396926, -0.9335804, -0.9271839, -0.9205049,
    -0.9135455, -0.9063078, -0.8987940, -0.8910065, -0.8829476, -0.8746197,
    -0.8660254, -0.8571673, -0.8480481, -0.8386706, -0.8290376, -0.8191520,
    -0.8090170, -0.7986355, -0.7880108, -0.7771460, -0.7660444, -0.7547096,
    -0.7431448, -0.7313537, -0.7193398, -0.7071068, -0.6946584, -0.6819984,
    -0.6691306, -0.6560590, -0.6427876, -0.6293204, -0.6156615, -0.6018150,
    -0.5877853, -0.5735764, -0.5591929, -0.5446390, -0.5299193, -0.5150381,
    -0.5000000, -0.4848096, -0.4694716, -0.4539905, -0.4383711, -0.4226183,
    -0.4067366, -0.3907311, -0.3746066, -0.3583679, -0.3420201, -0.3255682,
    -0.3090170, -0.2923717, -0.2756374, -0.2588190, -0.2419219, -0.2249511,
    -0.2079117, -0.1908090, -0.1736482, -0.1564345, -0.1391731, -0.1218693,
    -0.1045285, -0.0871557, -0.0697565, -0.0523360, -0.0348995, -0.0174524,
    -0.0000000, 0.0174524, 0.0348995, 0.0523360, 0.0697565, 0.0871557,
    0.1045285, 0.1218693, 0.1391731, 0.1564345, 0.1736482, 0.1908090,
    0.2079117, 0.2249511, 0.2419219, 0.2588190, 0.2756374, 0.2923717,
    0.3090170, 0.3255682, 0.3420201, 0.3583679, 0.3746066, 0.3907311,
    0.4067366, 0.4226183, 0.4383711, 0.4539905, 0.4694716, 0.4848096,
    0.5000000, 0.5150381, 0.5299193, 0.5446390, 0.5591929, 0.5735764,
    0.5877853, 0.6018150, 0.6156615, 0.6293204, 0.6427876, 0.6560590,
    0.6691306, 0.6819984, 0.6946584, 0.7071068, 0.7193398, 0.7313537,
    0.7431448, 0.7547096, 0.7660444, 0.7771460, 0.7880108, 0.7986355,
    0.8090170, 0.8191520, 0.8290376, 0.8386706, 0.8480481, 0.8571673,
    0.8660254, 0.8746197, 0.8829476, 0.8910065, 0.8987940, 0.9063078,
    0.9135455, 0.9205049, 0.9271839, 0.9335804, 0.9396926, 0.9455186,
    0.9510565, 0.9563048, 0.9612617, 0.9659258, 0.9702957, 0.9743701,
    0.9781476, 0.9816272, 0.9848078, 0.9876883, 0.9902681, 0.9925462,
    0.9945219, 0.9961947, 0.9975641, 0.9986295, 0.9993908, 0.9998477,
    1.0000000,
];

/// OpenCV `drawing.cpp` `ellipse2Poly` for our fixed call shape `(arc 0..360, delta 1)`. Returns the
/// rounded, consecutive-deduped polygon vertices of a rotated ellipse. Faithful to the double-version
/// (float `SinTable` trig, `float` rotation cos/sin) + the int-version (`cvRound` + dedup).
pub(crate) fn ellipse2poly(center: (i32, i32), axes: (i32, i32), angle: i32) -> Vec<(i32, i32)> {
    // Normalize the rotation angle to [0, 360] (the double-version `while` loops), then `sincos`.
    let mut ang = angle;
    while ang < 0 {
        ang += 360;
    }
    while ang > 360 {
        ang -= 360;
    }
    let a_idx = ang + if ang < 0 { 360 } else { 0 };
    let cos_rot = SIN_TABLE[(450 - a_idx) as usize]; // `cosval` from sincos
    let sin_rot = SIN_TABLE[a_idx as usize]; // `sinval`
    let (cx, cy) = (center.0 as f64, center.1 as f64);
    let (aw, ah) = (axes.0 as f64, axes.1 as f64);

    let mut prev = (i32::MIN, i32::MIN);
    let mut pts: Vec<(i32, i32)> = Vec::with_capacity(362);
    // for( i = arc_start; i < arc_end + delta; i += delta ) with arc 0..360, delta 1.
    for i in 0..=360i32 {
        // x = axes.width * SinTable[450-angle]; y = axes.height * SinTable[angle]  (float promoted).
        let x = aw * SIN_TABLE[(450 - i) as usize] as f64;
        let y = ah * SIN_TABLE[i as usize] as f64;
        let px = cx + x * cos_rot as f64 - y * sin_rot as f64;
        let py = cy + x * sin_rot as f64 + y * cos_rot as f64;
        let pt = (cv_round(px), cv_round(py));
        if pt != prev {
            pts.push(pt);
            prev = pt;
        }
    }
    if pts.len() == 1 {
        pts = vec![center, center];
    }
    pts
}

/// OpenCV `clipLine(Size, Point&, Point&)` — clip the segment to `[0,w-1]×[0,h-1]`. Returns whether
/// any part is inside (matching OpenCV: integer arithmetic, in-place endpoint update).
fn clip_line(w: i32, h: i32, pt1: &mut (i32, i32), pt2: &mut (i32, i32)) -> bool {
    if w <= 0 || h <= 0 {
        return false;
    }
    let right = (w - 1) as i64;
    let bottom = (h - 1) as i64;
    let (mut x1, mut y1) = (pt1.0 as i64, pt1.1 as i64);
    let (mut x2, mut y2) = (pt2.0 as i64, pt2.1 as i64);
    let code = |x: i64, y: i64| -> i64 {
        (x < 0) as i64 + (x > right) as i64 * 2 + (y < 0) as i64 * 4 + (y > bottom) as i64 * 8
    };
    let mut c1 = code(x1, y1);
    let mut c2 = code(x2, y2);
    if (c1 & c2) == 0 && (c1 | c2) != 0 {
        if c1 & 12 != 0 {
            let a = if c1 < 8 { 0 } else { bottom };
            x1 += (a - y1) * (x2 - x1) / (y2 - y1);
            y1 = a;
            c1 = (x1 < 0) as i64 + (x1 > right) as i64 * 2;
        }
        if c2 & 12 != 0 {
            let a = if c2 < 8 { 0 } else { bottom };
            x2 += (a - y2) * (x2 - x1) / (y2 - y1);
            y2 = a;
            c2 = (x2 < 0) as i64 + (x2 > right) as i64 * 2;
        }
        if (c1 & c2) == 0 && (c1 | c2) != 0 {
            if c1 != 0 {
                let a = if c1 == 1 { 0 } else { right };
                y1 += (a - x1) * (y2 - y1) / (x2 - x1);
                x1 = a;
                c1 = 0;
            }
            if c2 != 0 {
                let a = if c2 == 1 { 0 } else { right };
                y2 += (a - x2) * (y2 - y1) / (x2 - x1);
                x2 = a;
                c2 = 0;
            }
        }
    }
    pt1.0 = x1 as i32;
    pt1.1 = y1 as i32;
    pt2.0 = x2 as i32;
    pt2.1 = y2 as i32;
    (c1 | c2) == 0
}

/// One channel-3 horizontal span `[xl, xr]` inclusive at row `y` (OpenCV `ICV_HLINE`, pix_size=3).
/// Caller guarantees `0 <= y < h`; clamps to the row and no-ops when `xl > xr`.
#[inline]
fn hline(img: &mut [u8], w: i32, y: i32, xl: i32, xr: i32, color: [u8; 3]) {
    if xl > xr {
        return;
    }
    let row = (y as usize) * (w as usize) * 3;
    let mut off = row + (xl as usize) * 3;
    for _ in xl..=xr {
        img[off] = color[0];
        img[off + 1] = color[1];
        img[off + 2] = color[2];
        off += 3;
    }
}

/// OpenCV `Line` (LINE_8, `leftToRight=true`) — the masked Bresenham `LineIterator`, drawing each
/// pixel's 3 channels. Clips to the canvas first.
fn line8(img: &mut [u8], w: i32, h: i32, mut pt1: (i32, i32), mut pt2: (i32, i32), color: [u8; 3]) {
    if !clip_line(w, h, &mut pt1, &mut pt2) {
        return;
    }
    let elem: i64 = 3;
    let mut bt_pix: i64 = elem;
    let mut istep: i64 = (w as i64) * 3;
    let mut dx = (pt2.0 - pt1.0) as i64;
    let mut dy = (pt2.1 - pt1.1) as i64;

    let mut s: i64 = if dx < 0 { -1 } else { 0 };
    // leftToRight = true: abs the deltas and start from the smaller-x endpoint.
    dx = (dx ^ s) - s;
    dy = (dy ^ s) - s;
    let mut p1x = pt1.0 as i64;
    let mut p1y = pt1.1 as i64;
    p1x ^= (p1x ^ pt2.0 as i64) & s;
    p1y ^= (p1y ^ pt2.1 as i64) & s;

    let mut ptr: i64 = p1y * istep + p1x * elem;

    s = if dy < 0 { -1 } else { 0 };
    dy = (dy ^ s) - s;
    istep = (istep ^ s) - s;

    s = if dy > dx { -1 } else { 0 };
    // conditional swap dx<->dy and bt_pix<->istep
    dx ^= dy & s;
    dy ^= dx & s;
    dx ^= dy & s;
    bt_pix ^= istep & s;
    istep ^= bt_pix & s;
    bt_pix ^= istep & s;

    // connectivity == 8
    let mut err = dx - (dy + dy);
    let plus_delta = dx + dx;
    let minus_delta = -(dy + dy);
    let plus_step = istep;
    let minus_step = bt_pix;
    let count = dx + 1;

    let len = img.len() as i64;
    for _ in 0..count {
        if ptr >= 0 && ptr + 2 < len {
            let i = ptr as usize;
            img[i] = color[0];
            img[i + 1] = color[1];
            img[i + 2] = color[2];
        }
        let mask: i64 = if err < 0 { -1 } else { 0 };
        err += minus_delta + (plus_delta & mask);
        ptr += minus_step + (plus_step & mask);
    }
}

/// OpenCV `FillConvexPoly` (LINE_8, shift=0): the `Line` outline pass over every edge, then the
/// XY_SHIFT=16 fixed-point two-edge scanline fill. `v` are integer polygon vertices.
// Index loops are faithful to OpenCV (the index drives `imin` and `edge[i]`); keep them as written.
#[allow(clippy::needless_range_loop)]
pub(crate) fn fill_convex_poly(img: &mut [u8], w: i32, h: i32, v: &[(i32, i32)], color: [u8; 3]) {
    let npts = v.len() as i32;
    // shift = 0 ⇒ the `delta = 1<<shift>>1` bias is 0, so `(coord + delta) >> shift == coord`.
    let delta1: i64 = XY_ONE >> 1; // line_type < LINE_AA
    let delta2: i64 = XY_ONE >> 1;

    let mut xmin = v[0].0 as i64;
    let mut xmax = v[0].0 as i64;
    let mut ymin = v[0].1 as i64;
    let mut ymax = v[0].1 as i64;
    let mut imin = 0i32;

    let last = v[(npts - 1) as usize];
    let mut p0 = ((last.0 as i64) << XY_SHIFT, (last.1 as i64) << XY_SHIFT);
    for i in 0..npts as usize {
        let p = v[i];
        if (p.1 as i64) < ymin {
            ymin = p.1 as i64;
            imin = i as i32;
        }
        ymax = ymax.max(p.1 as i64);
        xmax = xmax.max(p.0 as i64);
        xmin = xmin.min(p.0 as i64);
        let pshift = ((p.0 as i64) << XY_SHIFT, (p.1 as i64) << XY_SHIFT);
        // shift == 0: draw the edge outline with the integer-pixel Line.
        let pt0 = ((p0.0 >> XY_SHIFT) as i32, (p0.1 >> XY_SHIFT) as i32);
        let pt1 = ((pshift.0 >> XY_SHIFT) as i32, (pshift.1 >> XY_SHIFT) as i32);
        line8(img, w, h, pt0, pt1, color);
        p0 = pshift;
    }

    // (xmin/.. + delta) >> shift with shift = 0, delta = 0 — unchanged.
    if npts < 3
        || (xmax as i32) < 0
        || (ymax as i32) < 0
        || (xmin as i32) >= w
        || (ymin as i32) >= h
    {
        return;
    }
    let ymax_c = ymax.min((h - 1) as i64) as i32;

    struct Edge {
        idx: i32,
        di: i32,
        x: i64,
        dx: i64,
        ye: i32,
    }
    let mut edge = [
        Edge {
            idx: imin,
            di: 1,
            x: -XY_ONE,
            dx: 0,
            ye: ymin as i32,
        },
        Edge {
            idx: imin,
            di: npts - 1,
            x: -XY_ONE,
            dx: 0,
            ye: ymin as i32,
        },
    ];
    let mut edges = npts; // shared advance budget (OpenCV's `int edges = npts`)
    let mut y = ymin as i32;

    loop {
        // line_type < LINE_AA ⇒ always update edges.
        for i in 0..2 {
            if y >= edge[i].ye {
                let mut idx0 = edge[i].idx;
                let di = edge[i].di;
                let mut idx = idx0 + di;
                if idx >= npts {
                    idx -= npts;
                }
                // for(; edges-- > 0;) — decrement on every cond check, including the found one.
                loop {
                    if edges > 0 {
                        edges -= 1;
                        let ty = v[idx as usize].1; // (v[idx].y + delta) >> shift
                        if ty > y {
                            let xs = (v[idx0 as usize].0 as i64) << XY_SHIFT; // shift != XY_SHIFT
                            let xe = (v[idx as usize].0 as i64) << XY_SHIFT;
                            edge[i].ye = ty;
                            edge[i].dx = ((xe - xs) * 2 + (ty as i64 - y as i64))
                                / (2 * (ty as i64 - y as i64));
                            edge[i].x = xs;
                            edge[i].idx = idx;
                            break;
                        }
                        idx0 = idx;
                        idx += di;
                        if idx >= npts {
                            idx -= npts;
                        }
                    } else {
                        edges -= 1;
                        break;
                    }
                }
            }
        }

        if edges < 0 {
            break;
        }

        if y >= 0 {
            let (left, right) = if edge[0].x > edge[1].x {
                (1, 0)
            } else {
                (0, 1)
            };
            let mut xx1 = ((edge[left].x + delta1) >> XY_SHIFT) as i32;
            let mut xx2 = ((edge[right].x + delta2) >> XY_SHIFT) as i32;
            if xx2 >= 0 && xx1 < w {
                if xx1 < 0 {
                    xx1 = 0;
                }
                if xx2 >= w {
                    xx2 = w - 1;
                }
                hline(img, w, y, xx1, xx2, color);
            }
        }

        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        y += 1;
        if y > ymax_c {
            break;
        }
    }
}

/// OpenCV filled `Circle` (Bresenham symmetric span fill).
pub(crate) fn circle_filled(
    img: &mut [u8],
    w: i32,
    h: i32,
    center: (i32, i32),
    radius: i32,
    color: [u8; 3],
) {
    let (cx, cy) = (center.0 as i64, center.1 as i64);
    let (wi, hi) = (w as i64, h as i64);
    let mut err: i64 = 0;
    let mut dx: i64 = radius as i64;
    let mut dy: i64 = 0;
    let mut plus: i64 = 1;
    let mut minus: i64 = ((radius as i64) << 1) - 1;
    let inside =
        center.0 >= radius && center.0 < w - radius && center.1 >= radius && center.1 < h - radius;

    while dx >= dy {
        let y11 = cy - dy;
        let y12 = cy + dy;
        let y21 = cy - dx;
        let y22 = cy + dx;
        let x11 = cx - dx;
        let x12 = cx + dx;
        let x21 = cx - dy;
        let x22 = cx + dy;

        if inside {
            hline(img, w, y11 as i32, x11 as i32, x12 as i32, color);
            hline(img, w, y12 as i32, x11 as i32, x12 as i32, color);
            hline(img, w, y21 as i32, x21 as i32, x22 as i32, color);
            hline(img, w, y22 as i32, x21 as i32, x22 as i32, color);
        } else if x11 < wi && x12 >= 0 && y21 < hi && y22 >= 0 {
            let fx11 = x11.max(0);
            let fx12 = x12.min(wi - 1);
            if y11 >= 0 && y11 < hi {
                hline(img, w, y11 as i32, fx11 as i32, fx12 as i32, color);
            }
            if y12 >= 0 && y12 < hi {
                hline(img, w, y12 as i32, fx11 as i32, fx12 as i32, color);
            }
            if x21 < wi && x22 >= 0 {
                let gx21 = x21.max(0);
                let gx22 = x22.min(wi - 1);
                if y21 >= 0 && y21 < hi {
                    hline(img, w, y21 as i32, gx21 as i32, gx22 as i32, color);
                }
                if y22 >= 0 && y22 < hi {
                    hline(img, w, y22 as i32, gx21 as i32, gx22 as i32, color);
                }
            }
        }

        dy += 1;
        err += plus;
        plus += 2;
        let mask: i64 = if err <= 0 { 0 } else { -1 };
        err -= minus & mask;
        dx += mask;
        minus -= mask & 2;
    }
}

/// The 5-keypoint limb connections (`limbSeq`) — eyes/mouth-corners each to the nose. The color is
/// `color_list[first]`.
const LIMB_SEQ: [(usize, usize); 4] = [(0, 2), (1, 2), (3, 2), (4, 2)];
/// `color_list` (RGB): left-eye, right-eye, nose, mouth-left, mouth-right.
const COLORS: [[u8; 3]; 5] = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
];
const STICKWIDTH: i32 = 4;

/// Render the InstantID kps control image: a `width × height` RGB [`Image`] from 5 landmarks
/// `[left_eye, right_eye, nose, mouth_left, mouth_right]` (canvas-space pixel coords). Bit-exact to
/// the vendored `draw_kps`.
pub fn draw_kps(width: u32, height: u32, kps: &[(f32, f32)]) -> Image {
    assert!(
        kps.len() >= 5,
        "draw_kps needs 5 keypoints, got {}",
        kps.len()
    );
    let (w, h) = (width as i32, height as i32);
    let mut canvas = vec![0u8; (w as usize) * (h as usize) * 3];

    // Stick layer (filled rotated ellipses), accumulated at full color (0/255 per channel).
    for &(i0, i1) in &LIMB_SEQ {
        let color = COLORS[i0];
        let (x0, y0) = kps[i0];
        let (x1, y1) = kps[i1];
        // np.mean over float32 endpoints, then int() truncation.
        let mean_x = (x0 + x1) / 2.0_f32;
        let mean_y = (y0 + y1) / 2.0_f32;
        // length in float32 (numpy `**2`/`**0.5`); angle via atan2 promoted to f64.
        let ddx = x0 - x1;
        let ddy = y0 - y1;
        let length = (ddx * ddx + ddy * ddy).sqrt(); // f32
        let angle = (ddy as f64).atan2(ddx as f64).to_degrees();
        let center = (mean_x as i32, mean_y as i32);
        let axes = ((length / 2.0_f32) as i32, STICKWIDTH);
        let poly = ellipse2poly(center, axes, angle as i32);
        fill_convex_poly(&mut canvas, w, h, &poly, color);
    }
    // Dim the whole stick layer ×0.6 in float64 then truncate to u8 (255 → 152, NOT 153).
    for b in canvas.iter_mut() {
        *b = (*b as f64 * 0.6) as u8;
    }
    // Circles at full color, drawn over the dimmed sticks.
    for (idx, &(x, y)) in kps.iter().take(5).enumerate() {
        circle_filled(&mut canvas, w, h, (x as i32, y as i32), 10, COLORS[idx]);
    }

    Image {
        width,
        height,
        pixels: canvas,
    }
}

/// Resize `image` keeping aspect (PIL LANCZOS) and center-pad onto a black `width × height` canvas —
/// the sc-2009 kps-distortion rule (the control image must share the output aspect). Mirrors the
/// vendored `_letterbox`.
pub fn letterbox(image: &Image, width: u32, height: u32) -> Image {
    let (iw, ih) = (image.width, image.height);
    let ratio = (width as f64 / iw as f64).min(height as f64 / ih as f64);
    let new_w = ((iw as f64 * ratio).round() as u32).max(1);
    let new_h = ((ih as f64 * ratio).round() as u32).max(1);
    let resized = resize_lanczos_u8(
        &image.pixels,
        ih as usize,
        iw as usize,
        new_h as usize,
        new_w as usize,
    ); // f32 HWC, integer-valued [0,255]
    let mut canvas = vec![0u8; (width as usize) * (height as usize) * 3];
    let ox = ((width - new_w) / 2) as usize;
    let oy = ((height - new_h) / 2) as usize;
    for y in 0..new_h as usize {
        for x in 0..new_w as usize {
            let src = (y * new_w as usize + x) * 3;
            let dst = ((oy + y) * width as usize + (ox + x)) * 3;
            canvas[dst] = resized[src] as u8;
            canvas[dst + 1] = resized[src + 1] as u8;
            canvas[dst + 2] = resized[src + 2] as u8;
        }
    }
    Image {
        width,
        height,
        pixels: canvas,
    }
}

/// Canonical view-angle landmark sets (sc-2009), normalized `[0,1]` to a square canvas, order
/// `[left_eye, right_eye, nose, mouth_left, mouth_right]`. The pack supplies the IdentityNet pose
/// while the reference supplies identity.
// Landmark coordinates are empirical (sc-2009); some happen to sit near math constants (e.g. 0.3927 ≈
// π/8) but must stay verbatim.
#[allow(clippy::approx_constant)]
pub const VIEW_ANGLE_KPS: &[(&str, [(f32, f32); 5])] = &[
    (
        "front",
        [
            (0.4460, 0.5227),
            (0.5755, 0.5166),
            (0.5106, 0.5947),
            (0.4653, 0.6660),
            (0.5630, 0.6613),
        ],
    ),
    (
        "three_quarter_left",
        [
            (0.3679, 0.5325),
            (0.4514, 0.5354),
            (0.3553, 0.6007),
            (0.3724, 0.6718),
            (0.4349, 0.6733),
        ],
    ),
    (
        "three_quarter_right",
        [
            (0.5946, 0.4930),
            (0.6882, 0.4955),
            (0.6948, 0.5598),
            (0.6202, 0.6408),
            (0.6885, 0.6421),
        ],
    ),
    (
        "left_profile",
        [
            (0.4373, 0.3527),
            (0.4925, 0.3445),
            (0.3927, 0.4662),
            (0.4853, 0.5599),
            (0.5240, 0.5517),
        ],
    ),
    (
        "right_profile",
        [
            (0.5075, 0.3445),
            (0.5627, 0.3527),
            (0.6073, 0.4662),
            (0.4760, 0.5517),
            (0.5147, 0.5599),
        ],
    ),
    (
        "up",
        [
            (0.4535, 0.4371),
            (0.5765, 0.4332),
            (0.5077, 0.4918),
            (0.4647, 0.5704),
            (0.5646, 0.5667),
        ],
    ),
    (
        "down",
        [
            (0.4457, 0.6231),
            (0.5848, 0.6228),
            (0.5174, 0.7337),
            (0.4726, 0.7771),
            (0.5645, 0.7770),
        ],
    ),
    (
        "up_left",
        [
            (0.3757, 0.4584),
            (0.4504, 0.4681),
            (0.3490, 0.4918),
            (0.3430, 0.5857),
            (0.3936, 0.5924),
        ],
    ),
    (
        "up_right",
        [
            (0.5787, 0.4431),
            (0.6673, 0.4337),
            (0.6799, 0.4601),
            (0.6331, 0.5601),
            (0.6989, 0.5515),
        ],
    ),
    (
        "down_left",
        [
            (0.3344, 0.6464),
            (0.4363, 0.6282),
            (0.3749, 0.7418),
            (0.4090, 0.7905),
            (0.4662, 0.7762),
        ],
    ),
    (
        "down_right",
        [
            (0.5963, 0.6165),
            (0.6823, 0.6271),
            (0.6650, 0.7171),
            (0.5668, 0.7524),
            (0.6198, 0.7640),
        ],
    ),
];

/// The order the one-click "angle set" generates views in.
pub const ANGLE_SET_ORDER: [&str; 11] = [
    "front",
    "three_quarter_left",
    "three_quarter_right",
    "left_profile",
    "right_profile",
    "up",
    "down",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
];

/// Scaled `side × side` landmark array for a named view angle, or `None` if unknown — mirrors the
/// vendored `_view_angle_kps` (`np.array(points, float32) * float(side)`).
pub fn view_angle_kps(angle: &str, side: u32) -> Option<[(f32, f32); 5]> {
    VIEW_ANGLE_KPS
        .iter()
        .find(|(n, _)| *n == angle)
        .map(|(_, pts)| {
            let s = side as f32;
            let mut out = [(0.0f32, 0.0f32); 5];
            for (o, p) in out.iter_mut().zip(pts.iter()) {
                *o = (p.0 * s, p.1 * s);
            }
            out
        })
}
