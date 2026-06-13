//! sc-5136: VAE input preprocessing (`data_utils.VAEVideoTransform`) — the `[-1, 1]` tensor fed to
//! the Wan VAE encoder for the planner's VAE-feature conditioning of input images/videos.
//!
//! Port of `MaxLongEdgeMinShortEdgeResize` + the `ToTensor`/`Normalize(0.5, 0.5)` of
//! `VAEVideoTransform`:
//!   - [`resize_dims`] — long edge ≤ `max_size`, short edge ≥ `min_size`, snapped to `stride` (16),
//!     via `make_divisible` (Python banker's `round`). Exact integer math.
//!   - [`normalize_chw`] — `u8 → [0,1] → (x - 0.5) / 0.5 = x/127.5 - 1`, channels-first `[3, H, W]`.
//!   - [`vae_transform_image`] — full path (`resize_dims` → resize → normalize).
//!
//! Same resize-interpolation divergence as [`crate::vit_preprocess`]: the `image` crate (Catmull-Rom)
//! is not bit-identical to PIL bicubic; target dims are exact, resampled pixels differ slightly.
//!
//! The VAE *encode* of this tensor (`get_vae_features` — `DiagonalGaussianDistribution.mode()` for
//! images / `.sample()` for video, then normalize by the VAE `latents_mean/std`) wires the loaded Wan
//! VAE encoder and is assembled at the pipeline stage (sc-5140 / sc-5145).

use image::{imageops::FilterType, RgbImage};
use mlx_rs::Array;

use crate::vit_preprocess::py_round;

/// VAE-input resize defaults (`preprocess_image`/`preprocess_video`: `max 624 / min 1 / stride 16`).
pub const VAE_MAX_SIZE: i64 = 624;
pub const VAE_MIN_SIZE: i64 = 1;
pub const VAE_STRIDE: i64 = 16;

/// `data_utils.make_divisible`: round `value` to the nearest multiple of `stride` (≥ `stride`),
/// Python banker's `round`.
fn make_divisible(value: i64, stride: i64) -> i64 {
    stride.max(py_round(value as f64 / stride as f64) * stride)
}

fn apply_scale(width: i64, height: i64, scale: f64, stride: i64) -> (i64, i64) {
    (
        make_divisible(py_round(width as f64 * scale), stride),
        make_divisible(py_round(height as f64 * scale), stride),
    )
}

/// `MaxLongEdgeMinShortEdgeResize` target size `(new_width, new_height)`.
pub fn resize_dims(
    width: i64,
    height: i64,
    max_size: i64,
    min_size: i64,
    stride: i64,
) -> (i64, i64) {
    let mut scale = (max_size as f64 / width.max(height) as f64).min(1.0);
    scale = scale.max(min_size as f64 / width.min(height) as f64);
    let (mut nw, mut nh) = apply_scale(width, height, scale, stride);
    if nw.max(nh) > max_size {
        scale = max_size as f64 / nw.max(nh) as f64;
        (nw, nh) = apply_scale(nw, nh, scale, stride);
    }
    (nw, nh)
}

/// Channels-last RGB8 → `[3, H, W]` f32 in `[-1, 1]` (`ToTensor` + `Normalize(0.5, 0.5)`).
pub fn normalize_chw(pixels_hwc: &[u8], h: i64, w: i64) -> Array {
    let (hu, wu) = (h as usize, w as usize);
    let mut data = vec![0f32; 3 * hu * wu];
    for c in 0..3usize {
        for y in 0..hu {
            for x in 0..wu {
                let u = pixels_hwc[(y * wu + x) * 3 + c] as f32;
                data[(c * hu + y) * wu + x] = u / 127.5 - 1.0;
            }
        }
    }
    Array::from_slice(&data, &[3, h as i32, w as i32])
}

/// Full VAE preprocessing of one RGB image → `[3, H, W]` in `[-1, 1]` (`VAEVideoTransform.__call__`).
/// The resize uses the `image` crate (Catmull-Rom) — see the module note.
pub fn vae_transform_image(img: &RgbImage, max_size: i64, min_size: i64, stride: i64) -> Array {
    let (w, h) = (img.width() as i64, img.height() as i64);
    let (nw, nh) = resize_dims(w, h, max_size, min_size, stride);
    let resized = if (nw, nh) == (w, h) {
        img.clone()
    } else {
        image::imageops::resize(img, nw as u32, nh as u32, FilterType::CatmullRom)
    };
    normalize_chw(resized.as_raw(), nh, nw)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Banker's-round + clamp behavior of the long-edge resize.
    #[test]
    fn resize_dims_cases() {
        // 200 -> round(12.5)*16 = 12*16 = 192 (round-to-even).
        assert_eq!(resize_dims(200, 200, 624, 1, 16), (192, 192));
        // 24 -> round(1.5)*16 = 2*16 = 32.
        assert_eq!(resize_dims(24, 24, 624, 1, 16), (32, 32));
        // long edge clamp: 1920x1080 -> 624x352.
        assert_eq!(resize_dims(1920, 1080, 624, 1, 16), (624, 352));
        // already in-range multiple: identity.
        assert_eq!(resize_dims(320, 240, 624, 1, 16), (320, 240));
    }

    /// normalize maps 0->-1, 255->1, 127.5->0.
    #[test]
    fn normalize_range() {
        let px = [0u8, 0, 0, 255, 255, 255]; // 2 pixels (1x2 HWC)
        let a = normalize_chw(&px, 1, 2);
        let v: Vec<f32> = a.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
        // channel 0: [pixel0=0 -> -1, pixel1=255 -> 1]
        assert!((v[0] + 1.0).abs() < 1e-6 && (v[1] - 1.0).abs() < 1e-6);
    }
}
