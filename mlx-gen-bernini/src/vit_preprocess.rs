//! sc-5136: ViT (Qwen2.5-VL) image/video preprocessing — packed patch pixels + `grid_thw` for the
//! [`crate::vision::VisionTower`].
//!
//! Port of `Qwen2VLImageProcessor` (the upstream HF processor the planner uses, not vendored):
//!   - [`smart_resize`] — target `(h_bar, w_bar)` divisible by `factor = patch·merge` (28), total
//!     pixels clamped to `[min_pixels, max_pixels]`, aspect kept, Python banker's `round`. `grid_thw`
//!     derives from this, so it is computed **exactly** (integer math + round-half-to-even).
//!   - [`pack_patches`] — the `_preprocess` packing: temporal-pad to `temporal_patch_size`, then the
//!     9-axis reshape/transpose that lays patches out in the **merge-grouped** order
//!     `(grid_t, h/m, w/m, m, m, C, T, ph, pw)` → `pixel_values [seq, C·T·patch²=1176]` + `(t,h,w)`.
//!     This is exactly the order [`crate::vision::VisionTower`] consumes.
//!   - [`preprocess_image`] — the full path: `smart_resize` → resize → rescale (1/255) → CLIP
//!     normalize → [`pack_patches`].
//!
//! **Divergence (surfaced):** the resize interpolation uses the `image` crate (Catmull-Rom), which is
//! *not* bit-identical to PIL bicubic. The target dimensions (`grid_thw`) are exact; only the
//! resampled pixels differ slightly — acceptable for a semantic vision tower (not pixel
//! reconstruction). The exactly-matchable pieces (`smart_resize` dims, the rescale+normalize+pack on a
//! fixed input) are goldened bit-for-bit.

use image::{imageops::FilterType, RgbImage};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Array;

use mlx_gen::Result;

/// CLIP image mean/std (`mllm/preprocessor_config.json`).
pub const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Patch geometry (Qwen2.5-VL vision processor).
pub const PATCH_SIZE: i64 = 14;
pub const TEMPORAL_PATCH_SIZE: i64 = 2;
pub const MERGE_SIZE: i64 = 2;
/// `patch · merge` — the dimension-divisibility factor.
pub const FACTOR: i64 = PATCH_SIZE * MERGE_SIZE;

/// Python `round` (round half to **even**) — banker's rounding, which Rust's `f64::round` does not do.
pub(crate) fn py_round(x: f64) -> i64 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f as i64
    } else if diff > 0.5 {
        f as i64 + 1
    } else {
        let fi = f as i64;
        if fi % 2 == 0 {
            fi
        } else {
            fi + 1
        }
    }
}

/// `torch.linspace(start, end, n).round()` (round half to even), as `i64`. `n == 1` → `[start]`.
fn linspace_round(start: f64, end: f64, n: i64) -> Vec<i64> {
    if n <= 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![py_round(start)];
    }
    let step = (end - start) / (n as f64 - 1.0);
    (0..n).map(|i| py_round(start + step * i as f64)).collect()
}

/// `data_utils.smart_video_nframes` — pick frame indices so the sampled clip matches the target
/// `fps` / frame count. ViT uses `fps 2 / frame_factor 2 / add_one false`; VAE uses
/// `fps 16 / frame_factor 4 / add_one true` (→ `4k+1` frames). Returns the 0-based frame indices.
#[allow(clippy::too_many_arguments)]
pub fn smart_video_nframes(
    total_frames: i64,
    video_fps: f64,
    fps: f64,
    frame_factor: Option<i64>,
    min_frames: Option<i64>,
    max_frames: Option<i64>,
    add_one: bool,
) -> Vec<i64> {
    let add = i64::from(add_one);
    let mut total = total_frames;
    let mut nframes: i64;
    let raw = total_frames as f64 / video_fps * fps;
    if let Some(ff) = frame_factor {
        nframes = (raw / ff as f64).floor() as i64 * ff + add;
        nframes = nframes.max(ff + add);
        if video_fps == fps {
            total = (total_frames as f64 / ff as f64).floor() as i64 * ff + add;
        }
    } else {
        nframes = (raw + add as f64) as i64; // int() truncates toward zero
    }

    let mut idx = linspace_round(0.0, (total - 1) as f64, nframes);

    if let Some(mut mn) = min_frames {
        if let Some(ff) = frame_factor {
            mn = (mn as f64 / ff as f64).ceil() as i64 * ff;
        }
        nframes = (mn + add).max(nframes);
    }
    while (idx.len() as i64) < nframes {
        idx.push(*idx.last().unwrap());
    }
    if let Some(mut mx) = max_frames {
        if let Some(ff) = frame_factor {
            mx = (mx as f64 / ff as f64).floor() as i64 * ff;
        }
        nframes = (mx + add).min(nframes);
    }
    if idx.len() as i64 > nframes {
        idx.truncate(nframes as usize);
    }
    idx
}

/// Qwen2.5-VL `smart_resize`: snap `(height, width)` to multiples of `factor`, keeping the aspect
/// ratio while clamping total pixels into `[min_pixels, max_pixels]`.
pub fn smart_resize(
    height: i64,
    width: i64,
    factor: i64,
    min_pixels: i64,
    max_pixels: i64,
) -> (i64, i64) {
    let (hf, wf, ff) = (height as f64, width as f64, factor as f64);
    let mut h_bar = py_round(hf / ff) * factor;
    let mut w_bar = py_round(wf / ff) * factor;
    if h_bar * w_bar > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = factor.max((hf / beta / ff).floor() as i64 * factor);
        w_bar = factor.max((wf / beta / ff).floor() as i64 * factor);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = (hf * beta / ff).ceil() as i64 * factor;
        w_bar = (wf * beta / ff).ceil() as i64 * factor;
    }
    (h_bar, w_bar)
}

/// Pack normalized frames `[F, C, H, W]` (f32, channels-first) into `pixel_values [seq, C·T·patch²]` +
/// `grid_thw (t, h, w)`. Mirrors `Qwen2VLImageProcessor._preprocess`'s temporal-pad + 9-axis
/// reshape/transpose. `H`/`W` must be multiples of `patch·merge`.
pub fn pack_patches(
    frames: &Array,
    patch: i64,
    temporal: i64,
    merge: i64,
) -> Result<(Array, [i32; 3])> {
    let s = frames.shape();
    let (f, c, h, w) = (s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64);

    // Temporal-pad to a multiple of `temporal` by repeating the last frame.
    let frames = if f % temporal != 0 {
        let pad = temporal - (f % temporal);
        let idx: Vec<i32> = vec![(f - 1) as i32; pad as usize];
        let last = frames.take_axis(Array::from_slice(&idx, &[pad as i32]), 0)?;
        concatenate_axis(&[frames, &last], 0)?
    } else {
        frames.clone()
    };

    let fp = frames.shape()[0] as i64;
    let grid_t = fp / temporal;
    let grid_h = h / patch;
    let grid_w = w / patch;
    let (gh, gw) = (grid_h / merge, grid_w / merge);
    let i = |x: i64| x as i32;

    let reshaped = frames.reshape(&[
        i(grid_t),
        i(temporal),
        i(c),
        i(gh),
        i(merge),
        i(patch),
        i(gw),
        i(merge),
        i(patch),
    ])?;
    // (grid_t, gh, gw, m, m, C, T, ph, pw)
    let perm = reshaped.transpose_axes(&[0, 3, 6, 4, 7, 2, 1, 5, 8])?;
    let seq = grid_t * grid_h * grid_w;
    let row = c * temporal * patch * patch;
    let pixel_values = perm.reshape(&[i(seq), i(row)])?;
    Ok((pixel_values, [i(grid_t), i(grid_h), i(grid_w)]))
}

/// Build `[1, 3, h, w]` f32 from channels-last RGB8 bytes, rescaled (1/255) + normalized
/// `(x - mean)/std` per channel.
fn normalized_frame(pixels_hwc: &[u8], h: i64, w: i64, mean: [f32; 3], std: [f32; 3]) -> Array {
    let (hu, wu) = (h as usize, w as usize);
    let mut data = vec![0f32; 3 * hu * wu];
    for c in 0..3usize {
        let (m, sd) = (mean[c], std[c]);
        for y in 0..hu {
            for x in 0..wu {
                let u = pixels_hwc[(y * wu + x) * 3 + c] as f32;
                data[(c * hu + y) * wu + x] = (u / 255.0 - m) / sd;
            }
        }
    }
    Array::from_slice(&data, &[1, 3, h as i32, w as i32])
}

/// Full ViT preprocessing of one RGB image → `pixel_values [seq, 1176]` + `grid_thw`. `min_pixels` /
/// `max_pixels` bound the `smart_resize` area (the request's vit pixel budget).
///
/// The resize uses the `image` crate (Catmull-Rom) — see the module note: target dims are exact, only
/// the resampled pixels differ slightly from PIL bicubic.
pub fn preprocess_image(
    img: &RgbImage,
    min_pixels: i64,
    max_pixels: i64,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<(Array, [i32; 3])> {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (rh, rw) = smart_resize(h, w, FACTOR, min_pixels, max_pixels);
    let resized = if (rh, rw) == (h, w) {
        img.clone()
    } else {
        image::imageops::resize(img, rw as u32, rh as u32, FilterType::CatmullRom)
    };
    let frame = normalized_frame(resized.as_raw(), rh, rw, mean, std);
    pack_patches(&frame, PATCH_SIZE, TEMPORAL_PATCH_SIZE, MERGE_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Banker's rounding + clamp behavior (matches Python `round` half-to-even).
    #[test]
    fn smart_resize_banker_round_and_clamp() {
        // 42/28=1.5 -> 2 (even) -> 56; 70/28=2.5 -> 2 -> 56.
        assert_eq!(smart_resize(42, 70, 28, 3136, 12845056), (56, 56));
        // 98/28=3.5 -> 4 -> 112.
        assert_eq!(smart_resize(98, 28, 28, 3136, 12845056), (112, 28));
        // identity (multiples of 28, mid-range).
        assert_eq!(smart_resize(56, 84, 28, 3136, 12845056), (56, 84));
        // 40x40 below min_pixels -> up-clamp.
        assert_eq!(smart_resize(40, 40, 28, 3136, 12845056), (56, 56));
    }

    /// pack_patches emits the merge-grouped `[seq, 1176]` + `grid_thw` for a 56x84 image.
    #[test]
    fn pack_shapes() {
        let frame = Array::zeros::<f32>(&[1, 3, 56, 84]).unwrap();
        let (pv, grid) = pack_patches(&frame, 14, 2, 2).unwrap();
        assert_eq!(grid, [1, 4, 6]);
        assert_eq!(pv.shape(), &[24, 1176]); // seq = 1*4*6, row = 3*2*14*14
    }
}
