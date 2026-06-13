//! SeedVR2 video-mode orchestration (sc-4814) — the **pure host logic** behind multi-frame upscaling:
//! temporal chunk planning, the overlap cross-fade that closes the causal-VAE seam, and a
//! memory-budgeted chunk sizer. The tensor passes (encode → DiT → decode) live in [`crate::pipeline`];
//! this module is deliberately model-free so the planning/blend/budget math is unit-testable.
//!
//! ## Why chunking
//! The 3-D causal VAE compresses time 4:1 (`latentT = ceil(T/4)`, `decodedT = 4·latentT`) and the DiT
//! attends over a `(T,H,W)=(4,3,3)` window, so a clip is processed in temporal **chunks**. The VAE
//! preserves the frame count only when a chunk's pixel-frame length is a multiple of 4 **and** ≥ 8
//! (spike sc-4812: `T=4`→1, `T∈{1..3}`→still; `8→8, 12→12, 16→16…`). 16 frames = one window = the
//! natural unit.
//!
//! ## Why overlap + cross-fade
//! The causal VAE re-anchors each chunk's first frame (causal pad repeats it), so butt-joined chunks
//! produce a hard seam (spike: boundary jump 20× the within-chunk change). A ≥4-frame overlap with a
//! linear cross-fade eliminates it (0.67×, matching a single-chunk reference). The blend math here is
//! a faithful port of the spike prototype `chunk_test.py`.
//!
//! ## Memory budget
//! Peak per chunk ≈ `weights + 8 GB · (out_megapixels · frames_in_chunk)` (spike anchor). The sizer
//! picks the largest valid chunk under the machine's MLX memory limit × 0.85 (matching the wan
//! `auto_tiling_budgeted` / `preflight_denoise_memory_guard` convention), falls back to per-frame
//! (`T=1`) when even 8 frames won't fit, and reports an over-budget condition catchably when a single
//! frame won't fit (extreme HD — see the spatial-tiling follow-up).

use mlx_gen::Image;
use mlx_rs::memory::get_memory_limit;

/// Default temporal chunk = 16 pixel frames (latentT=4 = exactly one `(4,3,3)` window).
pub const DEFAULT_CHUNK_FRAMES: i32 = 16;
/// Default cross-fade overlap — the spike's ≥4-frame overlap that eliminates the causal-VAE seam.
pub const DEFAULT_OVERLAP: i32 = 4;
/// A chunk's pixel-frame length must be a multiple of this (the VAE's 4:1 temporal compression).
pub const TEMPORAL_MULT: i32 = 4;
/// …and at least this many frames (below 8 the temporal compression collapses to a still / changes count).
pub const MIN_CHUNK_FRAMES: i32 = 8;
/// Cap on the auto-sized chunk: more frames/pass means fewer seams + faster per frame, but a larger
/// single allocation right against the (approximate) budget. 64 = four windows is plenty of temporal
/// context; beyond it we prefer more chunks over hugging the ceiling of an approximate cost model.
pub const MAX_CHUNK_FRAMES: i32 = 64;

/// Budget cost-model slope (spike sc-4812): peak ≈ weights + `GB_PER_MPX_FRAME · out_Mpx · frames`.
const GB_PER_MPX_FRAME: f64 = 8.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// Fraction of the MLX memory limit treated as safe (matches wan's guards).
const SAFE_FRAC: f64 = 0.85;

/// Pixel dims must be multiples of this (VAE /8 · DiT patch /2) — also the spatial-tile alignment.
pub const SPATIAL_ALIGN: i32 = 16;
/// Smallest spatial tile edge (px) the budget sizer will choose — small enough to fit any machine.
pub const MIN_TILE_PX: i32 = 256;
/// Spatial-tile overlap (px, multiple of [`SPATIAL_ALIGN`]) for the feather blend (sc-5201).
pub const SPATIAL_OVERLAP: i32 = 64;

/// The safe peak-GB budget: this machine's MLX memory limit × [`SAFE_FRAC`]. Shared by the temporal
/// chunk sizer and the spatial tile sizer.
pub fn safe_budget_gib() -> f64 {
    (get_memory_limit() as f64 / GIB) * SAFE_FRAC
}

/// Round `t` up to a valid chunk length: a multiple of [`TEMPORAL_MULT`], floored at
/// [`MIN_CHUNK_FRAMES`] — so the VAE preserves the frame count (decodedT == chunk T).
pub fn pad_to_valid_chunk(t: i32) -> i32 {
    // round up to a multiple of TEMPORAL_MULT (signed `i32::div_ceil` is still unstable).
    let r = (t.max(0) + TEMPORAL_MULT - 1) / TEMPORAL_MULT * TEMPORAL_MULT;
    r.max(MIN_CHUNK_FRAMES)
}

/// One planned temporal chunk: the pixel-frame window `[start, start+len)` fed to the model. `len` is
/// always a valid chunk length (mult of 4, ≥ 8); when the window runs past the real frame count the
/// trailing positions are last-frame padding (dropped on assembly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: i32,
    pub len: i32,
}

/// Plan the temporal chunk windows over `n` real frames. `c` is the (valid) chunk length and `ov` the
/// overlap. Windows are full size (`len == c`) placed at stride `c - ov`; the clip is conceptually
/// padded (last frame repeated) so the final window is also full size. A single window covers
/// `n <= c`. Consecutive windows overlap by exactly `ov` (no gaps), which [`assemble_overlap`] relies on.
pub fn plan_chunks(n: i32, c: i32, ov: i32) -> Vec<Chunk> {
    let c = pad_to_valid_chunk(c);
    let ov = ov.clamp(0, c - 1);
    let stride = (c - ov).max(1);
    if n <= c {
        return vec![Chunk { start: 0, len: c }];
    }
    let mut out = Vec::new();
    let mut s = 0;
    loop {
        out.push(Chunk { start: s, len: c });
        if s + c >= n {
            break;
        }
        s += stride;
    }
    out
}

/// Linearly blend two equal-size frames per byte: `(1-w)·a + w·b`, rounded to `u8`.
fn blend_frames(a: &Image, b: &Image, w: f32) -> Image {
    debug_assert_eq!(a.pixels.len(), b.pixels.len());
    let pixels = a
        .pixels
        .iter()
        .zip(b.pixels.iter())
        .map(|(&pa, &pb)| {
            let v = (1.0 - w) * pa as f32 + w * pb as f32;
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect();
    Image {
        width: a.width,
        height: a.height,
        pixels,
    }
}

/// Cross-fade-assemble per-chunk frame lists into exactly `n` output frames. `chunks[k]` holds the
/// decoded (and color-corrected) frames for `plan[k]`, covering pixel-frames
/// `[plan[k].start, plan[k].start + len)`. In each chunk's leading `ov`-frame overlap with the
/// already-assembled region the frames are linearly cross-faded (weight ramps `1/(ov+1) … ov/(ov+1)`,
/// the spike `chunk_test.py` schedule); the rest pass through. Trailing padding past frame `n` is dropped.
pub fn assemble_overlap(plan: &[Chunk], chunks: &[Vec<Image>], n: i32, ov: i32) -> Vec<Image> {
    let mut out: Vec<Image> = Vec::with_capacity(n.max(0) as usize);
    for (k, frames) in chunks.iter().enumerate() {
        let start = plan[k].start;
        for (j, f) in frames.iter().enumerate() {
            let i = start + j as i32;
            if i >= n {
                break;
            }
            if (i as usize) < out.len() {
                // overlap with the previous chunk — cross-fade toward this chunk.
                let w = (i - start + 1) as f32 / (ov + 1) as f32;
                out[i as usize] = blend_frames(&out[i as usize], f, w);
            } else {
                // new, contiguous frame.
                out.push(f.clone());
            }
        }
    }
    out
}

/// The memory plan for a video at a given output size: process in temporal chunks of N frames, fall
/// back to per-frame (`T=1`), or refuse (even one frame won't fit — extreme HD).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChunkPlan {
    /// Process `frames`-frame temporal chunks (a valid chunk length).
    Chunked(i32),
    /// Even 8 frames exceed the budget — process one frame at a time (still temporally stable).
    PerFrame,
    /// A single frame exceeds the budget at this resolution. `needed_gib`/`safe_gib` for the message.
    OverBudget { needed_gib: f64, safe_gib: f64 },
}

/// Size the temporal chunk against this machine's MLX memory limit (× 0.85). See [`plan_chunk_size_with`].
pub fn plan_chunk_size(weights_bytes: usize, out_h: i32, out_w: i32) -> ChunkPlan {
    plan_chunk_size_with(weights_bytes, out_h, out_w, safe_budget_gib())
}

/// Pure budget selector (safe ceiling injected → unit-testable without touching the global limit).
/// `peak ≈ weights + 8 GB · out_Mpx · frames`:
///   * largest valid chunk (mult-of-4, ≥8, ≤ [`MAX_CHUNK_FRAMES`]) whose peak ≤ `safe_gib` → `Chunked`,
///   * else if a single frame fits → `PerFrame`,
///   * else `OverBudget`.
pub fn plan_chunk_size_with(
    weights_bytes: usize,
    out_h: i32,
    out_w: i32,
    safe_gib: f64,
) -> ChunkPlan {
    let weights_gib = weights_bytes as f64 / GIB;
    let out_mpx = (out_h as f64 * out_w as f64) / 1.0e6;
    let avail = safe_gib - weights_gib;
    let per_frame_gib = weights_gib + GB_PER_MPX_FRAME * out_mpx; // frames = 1

    // Largest frame count whose activation term fits the remaining budget.
    let max_frames = if avail > 0.0 && out_mpx > 0.0 {
        (avail / (GB_PER_MPX_FRAME * out_mpx)).floor() as i32
    } else {
        0
    };
    if max_frames >= MIN_CHUNK_FRAMES {
        let c = (max_frames / TEMPORAL_MULT * TEMPORAL_MULT).min(MAX_CHUNK_FRAMES);
        return ChunkPlan::Chunked(c);
    }
    if per_frame_gib <= safe_gib {
        return ChunkPlan::PerFrame;
    }
    ChunkPlan::OverBudget {
        needed_gib: per_frame_gib,
        safe_gib,
    }
}

// ---------------------------------------------------------------------------
// spatial tiling (sc-5201) — for frames too large to fit the budget even at T=1
// ---------------------------------------------------------------------------

/// A spatial tile of a frame: the pixel region `[y0,y1) × [x0,x1)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpatialTile {
    pub y0: i32,
    pub y1: i32,
    pub x0: i32,
    pub x1: i32,
}

/// Tile a `n`-px axis into full-size windows of `tile` with `overlap` (stride `tile-overlap`); the
/// final window's start is clamped so it stays full size and ends at `n` (overlapping its neighbor a
/// little more). One window covers `n <= tile`. With `n`/`tile`/`overlap` all multiples of
/// [`SPATIAL_ALIGN`] every start/end is too.
fn tile_axis(n: i32, tile: i32, overlap: i32) -> Vec<(i32, i32)> {
    if n <= tile {
        return vec![(0, n)];
    }
    let stride = (tile - overlap).max(1);
    let mut out = Vec::new();
    let mut s = 0;
    loop {
        let e = (s + tile).min(n);
        let start = (e - tile).max(0); // keep the tile full size
        if out.last() != Some(&(start, e)) {
            out.push((start, e));
        }
        if e >= n {
            break;
        }
        s += stride;
    }
    out
}

/// The grid of overlapping spatial tiles covering an `h × w` frame.
pub fn plan_spatial_tiles(h: i32, w: i32, tile: i32, overlap: i32) -> Vec<SpatialTile> {
    let ys = tile_axis(h, tile, overlap);
    let xs = tile_axis(w, tile, overlap);
    let mut out = Vec::with_capacity(ys.len() * xs.len());
    for &(y0, y1) in &ys {
        for &(x0, x1) in &xs {
            out.push(SpatialTile { y0, y1, x0, x1 });
        }
    }
    out
}

/// Largest square spatial-tile edge (px, multiple of [`SPATIAL_ALIGN`], ≥ [`MIN_TILE_PX`]) whose
/// per-frame (T=1) peak `weights + 8 GB · (tile²·1e-6)` fits `safe_gib`. Floors at `MIN_TILE_PX` — the
/// smallest tile we drop to (tiling still bounds peak as far as the model allows).
pub fn plan_spatial_tile_px(weights_bytes: usize, safe_gib: f64) -> i32 {
    let weights_gib = weights_bytes as f64 / GIB;
    let avail = (safe_gib - weights_gib).max(0.0);
    let max_area_px2 = avail / (GB_PER_MPX_FRAME * 1e-6); // = tile² (px²)
    let edge = (max_area_px2.max(0.0).sqrt() as i32) / SPATIAL_ALIGN * SPATIAL_ALIGN;
    edge.max(MIN_TILE_PX)
}

/// Per-pixel feather weights `(th·tw)` for a tile, tapering linearly to ~0 over `overlap` px on each
/// edge abutting a neighbor (`fade_*`) and staying 1 at outer image edges. Separable: `w = ry·rx`.
/// Assembly divides by the accumulated weight, so exact partition-of-unity isn't required.
pub fn feather_weight(
    th: i32,
    tw: i32,
    fade_top: bool,
    fade_bottom: bool,
    fade_left: bool,
    fade_right: bool,
    overlap: i32,
) -> Vec<f32> {
    let ry = axis_ramp(th, fade_top, fade_bottom, overlap);
    let rx = axis_ramp(tw, fade_left, fade_right, overlap);
    let mut out = vec![0f32; (th * tw) as usize];
    for y in 0..th as usize {
        for x in 0..tw as usize {
            out[y * tw as usize + x] = ry[y] * rx[x];
        }
    }
    out
}

/// Linear taper along one axis: ramp up over the first `overlap` px when `fade_start`, down over the
/// last `overlap` when `fade_end`, 1 in between; floored positive so the weight-sum is never zero.
fn axis_ramp(len: i32, fade_start: bool, fade_end: bool, overlap: i32) -> Vec<f32> {
    let ov = overlap.max(1);
    (0..len)
        .map(|i| {
            let mut w = 1.0f32;
            if fade_start && i < ov {
                w = w.min((i as f32 + 1.0) / (ov as f32 + 1.0));
            }
            if fade_end && i >= len - ov {
                w = w.min((len - i) as f32 / (ov as f32 + 1.0));
            }
            w.max(1e-4)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, fill: u8) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![fill; (w * h * 3) as usize],
        }
    }

    #[test]
    fn pad_to_valid_rounds_up_and_floors() {
        assert_eq!(pad_to_valid_chunk(1), 8);
        assert_eq!(pad_to_valid_chunk(8), 8);
        assert_eq!(pad_to_valid_chunk(9), 12);
        assert_eq!(pad_to_valid_chunk(13), 16);
        assert_eq!(pad_to_valid_chunk(16), 16);
        assert_eq!(pad_to_valid_chunk(0), 8);
    }

    #[test]
    fn single_chunk_when_clip_fits() {
        assert_eq!(plan_chunks(16, 16, 4), vec![Chunk { start: 0, len: 16 }]);
        assert_eq!(plan_chunks(5, 16, 4), vec![Chunk { start: 0, len: 16 }]);
    }

    #[test]
    fn plan_matches_spike_28_frame_two_chunk() {
        // chunk_test.py: N=28, chunk 16, overlap 4 → windows [0:16] and [12:28].
        let plan = plan_chunks(28, 16, 4);
        assert_eq!(
            plan,
            vec![Chunk { start: 0, len: 16 }, Chunk { start: 12, len: 16 }]
        );
    }

    #[test]
    fn plan_covers_long_clip_uniform_overlap() {
        // stride = 12; windows at 0,12,24 cover 40 frames (last window [24,40) reaches frame 39).
        let plan = plan_chunks(40, 16, 4);
        assert_eq!(
            plan.iter().map(|c| c.start).collect::<Vec<_>>(),
            [0, 12, 24]
        );
        assert!(plan.last().unwrap().start + plan.last().unwrap().len >= 40); // full coverage
                                                                              // each consecutive pair overlaps by exactly ov=4 (no gaps).
        for w in plan.windows(2) {
            assert_eq!(w[0].start + w[0].len - w[1].start, 4);
        }
    }

    #[test]
    fn assemble_no_blend_single_chunk_truncates_to_n() {
        // one 16-frame chunk, n=5 → first 5 frames, no blending.
        let plan = plan_chunks(5, 16, 4);
        let frames: Vec<Image> = (0..16).map(|i| img(2, 2, i as u8)).collect();
        let out = assemble_overlap(&plan, &[frames], 5, 4);
        assert_eq!(out.len(), 5);
        assert_eq!(out[4].pixels[0], 4);
    }

    #[test]
    fn assemble_crossfade_matches_spike_schedule() {
        // Reproduce chunk_test.py exactly: N=28, chunk0=[0:16] all value 0, chunk1=[12:28] all 200.
        // Frames 0..11 -> 0; 12..15 -> blend (w=1/5,2/5,3/5,4/5); 16..27 -> 200.
        let plan = plan_chunks(28, 16, 4);
        let c0: Vec<Image> = (0..16).map(|_| img(1, 1, 0)).collect();
        let c1: Vec<Image> = (0..16).map(|_| img(1, 1, 200)).collect();
        let out = assemble_overlap(&plan, &[c0, c1], 28, 4);
        assert_eq!(out.len(), 28);
        for (i, f) in out.iter().enumerate() {
            let got = f.pixels[0];
            let exp = if i < 12 {
                0
            } else if i < 16 {
                let w = (i as i32 - 12 + 1) as f32 / 5.0;
                (w * 200.0).round() as u8 // (1-w)*0 + w*200
            } else {
                200
            };
            assert_eq!(got, exp, "frame {i}");
        }
    }

    #[test]
    fn budget_chunked_at_modest_res() {
        // 512² with ~7.3 GB weights and a generous 108 GiB safe budget → a large valid chunk.
        let wb = (7.3 * GIB) as usize;
        match plan_chunk_size_with(wb, 512, 512, 108.0) {
            ChunkPlan::Chunked(c) => {
                assert!((MIN_CHUNK_FRAMES..=MAX_CHUNK_FRAMES).contains(&c));
                assert_eq!(c % TEMPORAL_MULT, 0);
            }
            other => panic!("expected Chunked, got {other:?}"),
        }
    }

    #[test]
    fn budget_falls_back_to_per_frame_then_over_budget() {
        let wb = (7.3 * GIB) as usize;
        // Tight budget where 8 frames at 1024² (8·1.05·8 ≈ 67 GiB) won't fit but one frame (~16 GiB) will.
        assert_eq!(
            plan_chunk_size_with(wb, 1024, 1024, 20.0),
            ChunkPlan::PerFrame
        );
        // A single 4096² frame (8·16.8 ≈ 134 GiB) exceeds even a 108 GiB budget → OverBudget.
        assert!(matches!(
            plan_chunk_size_with(wb, 4096, 4096, 108.0),
            ChunkPlan::OverBudget { .. }
        ));
    }

    #[test]
    fn spatial_single_tile_when_frame_fits() {
        assert_eq!(
            plan_spatial_tiles(256, 256, 512, 64),
            vec![SpatialTile {
                y0: 0,
                y1: 256,
                x0: 0,
                x1: 256
            }]
        );
    }

    #[test]
    fn spatial_tiles_full_size_and_cover() {
        // 768×1024 into 512 tiles, overlap 64 → full-size tiles covering the whole frame.
        let tiles = plan_spatial_tiles(768, 1024, 512, 64);
        for t in &tiles {
            assert_eq!(t.y1 - t.y0, 512); // full-size tiles (768 > 512)
            assert_eq!(t.x1 - t.x0, 512);
        }
        assert_eq!(tiles.iter().map(|t| t.y1).max(), Some(768));
        assert_eq!(tiles.iter().map(|t| t.x1).max(), Some(1024));
        assert!(tiles.iter().any(|t| t.y0 == 0 && t.x0 == 0));
    }

    #[test]
    fn spatial_tile_px_budget_scales() {
        let wb = (7.3 * GIB) as usize;
        // Budget barely above the weights → computed edge below the floor → clamps to MIN_TILE_PX.
        assert_eq!(plan_spatial_tile_px(wb, 7.35), MIN_TILE_PX);
        // Generous budget → a large multiple-of-16 edge above the floor.
        let big = plan_spatial_tile_px(wb, 108.0);
        assert!(big > MIN_TILE_PX && big % SPATIAL_ALIGN == 0);
    }

    #[test]
    fn feather_outer_edges_unity_inner_tapers() {
        // Top-left corner tile: neighbors on the right + bottom only.
        let w = feather_weight(8, 8, false, true, false, true, 4);
        assert!((w[0] - 1.0).abs() < 1e-6); // top-left corner: both outer edges → 1
        assert!(w[(8 * 8 - 1) as usize] < 0.5); // bottom-right: fades on both inner edges
        assert!(w.iter().all(|&v| v > 0.0)); // strictly positive → weight-sum never zero
    }
}
