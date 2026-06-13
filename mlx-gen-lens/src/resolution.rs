//! Lens resolution buckets (sc-3173) — a faithful port of `_vendor/lens/resolution.py`.
//!
//! Two base resolutions (**1024** and **1440**) crossed with **nine** aspect ratios. All
//! heights/widths are divisible by 16 so they tile cleanly into Flux.2 VAE latents (downsample
//! factor 16). Aspect ratios are `"W:H"` strings (e.g. `"16:9"` is landscape, `"9:16"` portrait);
//! the table stores `(height, width)`.

use mlx_gen::{Error, Result};

/// The supported base resolutions (`SUPPORTED_BASE_RESOLUTIONS`).
pub const SUPPORTED_BASE_RESOLUTIONS: [u32; 2] = [1024, 1440];

/// The supported aspect ratios (`SUPPORTED_ASPECT_RATIOS`), in table order.
pub const SUPPORTED_ASPECT_RATIOS: [&str; 9] = [
    "1:2", "9:16", "2:3", "3:4", "1:1", "4:3", "3:2", "16:9", "2:1",
];

/// `RESOLUTION_BUCKETS[1024]` — `(aspect, (height, width))`, byte-identical to the reference.
const BUCKETS_1024: [(&str, (u32, u32)); 9] = [
    ("1:2", (1472, 736)),
    ("9:16", (1376, 768)),
    ("2:3", (1248, 832)),
    ("3:4", (1152, 864)),
    ("1:1", (1024, 1024)),
    ("4:3", (864, 1152)),
    ("3:2", (832, 1248)),
    ("16:9", (768, 1376)),
    ("2:1", (736, 1472)),
];

/// `RESOLUTION_BUCKETS[1440]` — `round_to_16(1024_value · 1440/1024)`, byte-identical to the reference.
const BUCKETS_1440: [(&str, (u32, u32)); 9] = [
    ("1:2", (2080, 1040)),
    ("9:16", (1936, 1088)),
    ("2:3", (1760, 1168)),
    ("3:4", (1616, 1216)),
    ("1:1", (1440, 1440)),
    ("4:3", (1216, 1616)),
    ("3:2", (1168, 1760)),
    ("16:9", (1088, 1936)),
    ("2:1", (1040, 2080)),
];

/// Return `(height, width)` for the requested bucket (`resolve_resolution`).
///
/// `aspect_ratio` is interpreted as `W:H` (e.g. `"16:9"` is landscape, `"9:16"` portrait). Errors on
/// an unsupported base resolution or aspect ratio, matching the reference `ValueError`s.
pub fn resolve_resolution(base_resolution: u32, aspect_ratio: &str) -> Result<(u32, u32)> {
    let table = match base_resolution {
        1024 => &BUCKETS_1024,
        1440 => &BUCKETS_1440,
        other => {
            return Err(Error::Msg(format!(
                "Unsupported base_resolution={other}. Supported: {SUPPORTED_BASE_RESOLUTIONS:?}"
            )))
        }
    };
    table
        .iter()
        .find(|(ar, _)| *ar == aspect_ratio)
        .map(|(_, hw)| *hw)
        .ok_or_else(|| {
            Error::Msg(format!(
                "Unsupported aspect_ratio={aspect_ratio:?}. Supported: {SUPPORTED_ASPECT_RATIOS:?}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_reference() {
        // Spot-check both bases + the square / extreme buckets, and that every dim ÷16.
        assert_eq!(resolve_resolution(1024, "1:1").unwrap(), (1024, 1024));
        assert_eq!(resolve_resolution(1024, "16:9").unwrap(), (768, 1376));
        assert_eq!(resolve_resolution(1024, "1:2").unwrap(), (1472, 736));
        assert_eq!(resolve_resolution(1440, "1:1").unwrap(), (1440, 1440));
        assert_eq!(resolve_resolution(1440, "2:1").unwrap(), (1040, 2080));
        for base in SUPPORTED_BASE_RESOLUTIONS {
            for ar in SUPPORTED_ASPECT_RATIOS {
                let (h, w) = resolve_resolution(base, ar).unwrap();
                assert_eq!(h % 16, 0, "{base} {ar} h={h} not ÷16");
                assert_eq!(w % 16, 0, "{base} {ar} w={w} not ÷16");
            }
        }
    }

    #[test]
    fn rejects_unknown() {
        assert!(resolve_resolution(512, "1:1").is_err());
        assert!(resolve_resolution(1024, "5:4").is_err());
    }
}
