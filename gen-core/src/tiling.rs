//! Video-VAE decode **tiling** — the family-agnostic geometry layer shared by the LTX and Wan VAEs.
//!
//! Decoding a large/long latent in one pass is memory-bound; tiling splits it into overlapping
//! spatial/temporal tiles, decodes each independently, and trapezoidally blends the results. This
//! module is the **pure** half — tiling presets, the per-axis interval split, the 1-D blend mask,
//! and the full [`TilePlan`] for a latent. The Array blend loop (slice each tile, decode, weight,
//! pad-and-accumulate, normalize) lives in each crate's `vae.rs` so it can reach that VAE's decoder;
//! the reference allocates full-size `output`+`weights` accumulators and processes one tile at a
//! time, so the pad-and-accumulate form keeps the same bounded peak memory.
//!
//! Port of the `mlx_video` reference `models/ltx/video_vae/tiling.py` (the shared primitives) plus
//! `models/wan/tiling.py`'s `causal_temporal` generalization. The per-VAE upsample factors and the
//! causal-vs-non-causal temporal mapping are carried by [`VaeTiling`]:
//!  - **LTX** ([`VaeTiling::LTX`]): spatial ×32 (8× learned × 4× unpatchify), temporal ×8, **causal**
//!    (`out_f = 1 + (f−1)·8`).
//!  - **Wan 2.1** ([`VaeTiling::WAN`]): spatial ×8, temporal ×4, **non-causal** (`out_f = f·4`) — the
//!    temporal axis tiles exactly like a spatial axis.

/// A VAE's tiling parameters: the decoder's spatial/temporal upsample factors and whether its
/// temporal decode is causal (`out_f = 1 + (f−1)·scale`) or non-causal (`out_f = f·scale`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VaeTiling {
    pub spatial_scale: i32,
    pub temporal_scale: i32,
    pub causal_temporal: bool,
}

impl VaeTiling {
    /// LTX-2 video VAE: spatial ×32 (8× upsample × 4× unpatchify), temporal ×8, causal.
    pub const LTX: Self = Self {
        spatial_scale: 32,
        temporal_scale: 8,
        causal_temporal: true,
    };
    /// Wan 2.1 z16 VAE: spatial ×8, temporal ×4, non-causal (`T → T·4`).
    pub const WAN: Self = Self {
        spatial_scale: 8,
        temporal_scale: 4,
        causal_temporal: false,
    };
    /// Wan 2.2 z48 `vae22` VAE: spatial ×16 (8× conv upsample × 2× unpatchify), temporal ×4,
    /// **causal** (`out_f = 1 + (f−1)·4` — the decoder runs `first_chunk=True`, so the leading
    /// temporal-padding frames are trimmed). The 5B's TI2V-5B VAE (sc-2680).
    pub const WAN22: Self = Self {
        spatial_scale: 16,
        temporal_scale: 4,
        causal_temporal: true,
    };
}

/// Per-frame spatial tiling (tile + overlap in **output pixels**).
#[derive(Clone, Copy, Debug)]
pub struct SpatialTiling {
    pub tile_px: i32,
    pub overlap_px: i32,
}

/// Temporal tiling (tile + overlap in **output frames**).
#[derive(Clone, Copy, Debug)]
pub struct TemporalTiling {
    pub tile_frames: i32,
    pub overlap_frames: i32,
}

/// Which axes to tile. `None` on either axis disables tiling there. Tile/overlap sizes are in
/// **output** units (pixels / frames) and convert to latent units by the VAE's scale.
#[derive(Clone, Copy, Debug, Default)]
pub struct TilingConfig {
    pub spatial: Option<SpatialTiling>,
    pub temporal: Option<TemporalTiling>,
}

impl TilingConfig {
    /// Reference default: 512 px / 64 px spatial, 64 / 24 frame temporal.
    pub fn default_preset() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 512,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 64,
                overlap_frames: 24,
            }),
        }
    }

    /// Aggressive (smaller tiles, lowest memory): 256/64 px, 32/8 frame.
    pub fn aggressive() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 256,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 32,
                overlap_frames: 8,
            }),
        }
    }

    /// Conservative (larger tiles, faster, less saving): 768/64 px, 96/24 frame.
    pub fn conservative() -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px: 768,
                overlap_px: 64,
            }),
            temporal: Some(TemporalTiling {
                tile_frames: 96,
                overlap_frames: 24,
            }),
        }
    }

    pub fn spatial_only(tile_px: i32, overlap_px: i32) -> Self {
        Self {
            spatial: Some(SpatialTiling {
                tile_px,
                overlap_px,
            }),
            temporal: None,
        }
    }

    pub fn temporal_only(tile_frames: i32, overlap_frames: i32) -> Self {
        Self {
            spatial: None,
            temporal: Some(TemporalTiling {
                tile_frames,
                overlap_frames,
            }),
        }
    }

    /// Auto-select a config from **output** dimensions (reference `TilingConfig.auto`), or `None`
    /// when no tiling is needed. Thresholds (spatial > 512 px, temporal > 65 frames) are in output
    /// units, so this is VAE-scale-independent.
    pub fn auto(height: i32, width: i32, num_frames: i32) -> Option<Self> {
        let needs_spatial = height > 512 || width > 512;
        let needs_temporal = num_frames > 65;
        if !needs_spatial && !needs_temporal {
            return None;
        }
        let est_gb = (3.0 * num_frames as f64 * height as f64 * width as f64 * 4.0)
            / (1024.0 * 1024.0 * 1024.0);
        if est_gb > 2.0 || (height * width > 768 * 1024 && num_frames > 100) {
            return Some(Self::aggressive());
        }
        let spatial = needs_spatial.then(|| {
            let max_dim = height.max(width);
            let tile_px = if max_dim > 1024 {
                384
            } else if max_dim > 768 {
                512
            } else {
                384
            };
            SpatialTiling {
                tile_px,
                overlap_px: 64,
            }
        });
        let temporal = needs_temporal.then(|| {
            let (tile_frames, overlap_frames) = if num_frames > 200 {
                (32, 8)
            } else if num_frames > 100 {
                (48, 16)
            } else {
                (64, 24)
            };
            TemporalTiling {
                tile_frames,
                overlap_frames,
            }
        });
        Some(Self { spatial, temporal })
    }

    /// Whether tiling actually fires for a latent `[_, _, f, h, w]` under VAE `vae` (i.e. some axis
    /// exceeds its latent-space tile size).
    pub fn needs_tiling(&self, vae: VaeTiling, f: i32, h: i32, w: i32) -> bool {
        let s = self.spatial.is_some_and(|s| {
            let t = s.tile_px / vae.spatial_scale;
            h > t || w > t
        });
        let t = self
            .temporal
            .is_some_and(|tc| f > tc.tile_frames / vae.temporal_scale);
        s || t
    }

    /// Build the [`TilePlan`] for a latent of shape `[_, _, f, h, w]` under VAE `vae`.
    pub fn plan(&self, vae: VaeTiling, f: i32, h: i32, w: i32) -> TilePlan {
        let (t_tile, t_over) = match self.temporal {
            Some(tc) => (
                tc.tile_frames / vae.temporal_scale,
                tc.overlap_frames / vae.temporal_scale,
            ),
            None => (f, 0),
        };
        let (s_tile, s_over) = match self.spatial {
            Some(sc) => (
                sc.tile_px / vae.spatial_scale,
                sc.overlap_px / vae.spatial_scale,
            ),
            None => (h.max(w), 0),
        };
        TilePlan {
            t: temporal_tiles(t_tile, t_over, f, vae.temporal_scale, vae.causal_temporal),
            h: spatial_tiles(s_tile, s_over, h, vae.spatial_scale),
            w: spatial_tiles(s_tile, s_over, w, vae.spatial_scale),
            out_f: if vae.causal_temporal {
                1 + (f - 1) * vae.temporal_scale
            } else {
                f * vae.temporal_scale
            },
            out_h: h * vae.spatial_scale,
            out_w: w * vae.spatial_scale,
        }
    }
}

/// One tile along one axis: latent `[start, end)`, the output `[out_start, out_stop)` it maps to,
/// and the 1-D blend `mask` (length `out_stop − out_start`).
#[derive(Clone, Debug)]
pub struct AxisTile {
    pub start: i32,
    pub end: i32,
    pub out_start: i32,
    pub out_stop: i32,
    pub mask: Vec<f32>,
}

/// `compute_trapezoidal_mask_1d`: ones with a left fade-in (`ramp_left`) and right fade-out
/// (`ramp_right`). `left_from_0` chooses the linspace convention (temporal causal tiles fade from 0).
pub fn trapezoidal_mask(
    length: i32,
    ramp_left: i32,
    ramp_right: i32,
    left_from_0: bool,
) -> Vec<f32> {
    assert!(length > 0, "mask length must be positive");
    let length = length as usize;
    let ramp_left = ramp_left.clamp(0, length as i32) as usize;
    let ramp_right = ramp_right.clamp(0, length as i32) as usize;
    let mut mask = vec![1.0f32; length];

    if ramp_left > 0 {
        let interval = if left_from_0 {
            ramp_left + 1
        } else {
            ramp_left + 2
        };
        // linspace(0, 1, interval), drop last; if !left_from_0 also drop first.
        let full: Vec<f32> = (0..interval)
            .map(|i| i as f32 / (interval as f32 - 1.0))
            .collect();
        let fade_in: &[f32] = if left_from_0 {
            &full[..interval - 1]
        } else {
            &full[1..interval - 1]
        };
        for i in 0..ramp_left.min(fade_in.len()) {
            mask[i] *= fade_in[i];
        }
    }

    if ramp_right > 0 {
        // fade_out = linspace(1, 0, ramp_right+2)[1:-1] = (ramp_right+1-i)/(ramp_right+1), i=1..ramp_right
        for i in 0..ramp_right {
            let v = (ramp_right as f32 + 1.0 - (i as f32 + 1.0)) / (ramp_right as f32 + 1.0);
            mask[length - ramp_right + i] *= v;
        }
    }

    for v in &mut mask {
        *v = v.clamp(0.0, 1.0);
    }
    mask
}

/// Raw per-axis interval split (`split_in_spatial`): `(starts, ends, left_ramps, right_ramps)`.
fn split_spatial(size: i32, overlap: i32, dim: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    // Guard degenerate configs (F-005): a caller-supplied tile ≤ overlap (reachable via
    // `TilingConfig::spatial_only`/`temporal_only`), or a tile floored to 0 by latent downscaling,
    // would divide by zero — or wrap `amount` to a huge `usize` (capacity panic) — below. Clamp to a
    // tile ≥ 1 and an overlap in `0..size`. For every valid config (`overlap < size`) this is a no-op.
    let size = size.max(1);
    let overlap = overlap.clamp(0, size - 1);
    if dim <= size {
        return (vec![0], vec![dim], vec![0], vec![0]);
    }
    let amount = (dim + size - 2 * overlap - 1) / (size - overlap);
    let starts: Vec<i32> = (0..amount).map(|i| i * (size - overlap)).collect();
    let mut ends: Vec<i32> = starts.iter().map(|s| s + size).collect();
    *ends.last_mut().unwrap() = dim;
    let mut left = vec![overlap; amount as usize];
    left[0] = 0;
    let mut right = vec![overlap; amount as usize];
    *right.last_mut().unwrap() = 0;
    (starts, ends, left, right)
}

/// `split_in_temporal`: spatial split, then `starts[1:] -= 1`, `left_ramps[1:] += 1` (causal).
fn split_temporal(size: i32, overlap: i32, dim: i32) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) {
    let (mut starts, ends, mut left, right) = split_spatial(size, overlap, dim);
    for i in 1..starts.len() {
        starts[i] -= 1;
        left[i] += 1;
    }
    (starts, ends, left, right)
}

/// Build the spatial-axis tiles (`map_spatial_slice`: out = latent·scale, mask `left_from_0=false`).
fn spatial_tiles(tile_latent: i32, overlap_latent: i32, dim: i32, scale: i32) -> Vec<AxisTile> {
    let (starts, ends, left, right) = split_spatial(tile_latent, overlap_latent, dim);
    starts
        .iter()
        .enumerate()
        .map(|(i, &begin)| {
            let end = ends[i];
            let out_start = begin * scale;
            let out_stop = end * scale;
            let mask = trapezoidal_mask(
                out_stop - out_start,
                left[i] * scale,
                right[i] * scale,
                false,
            );
            AxisTile {
                start: begin,
                end,
                out_start,
                out_stop,
                mask,
            }
        })
        .collect()
}

/// Build the temporal-axis tiles. **Causal** (`out = 1+(latent−1)·scale`, `map_temporal_slice`,
/// `left_from_0`) for LTX; **non-causal** temporal tiles exactly like a spatial axis (`out =
/// latent·scale`) for Wan — the reference's `causal_temporal=False` path.
fn temporal_tiles(
    tile_latent: i32,
    overlap_latent: i32,
    dim: i32,
    scale: i32,
    causal: bool,
) -> Vec<AxisTile> {
    if !causal {
        return spatial_tiles(tile_latent, overlap_latent, dim, scale);
    }
    let (starts, ends, left, right) = split_temporal(tile_latent, overlap_latent, dim);
    starts
        .iter()
        .enumerate()
        .map(|(i, &begin)| {
            let end = ends[i];
            let out_start = begin * scale;
            let out_stop = 1 + (end - 1) * scale;
            let left_scaled = if left[i] > 0 {
                1 + (left[i] - 1) * scale
            } else {
                0
            };
            let mask = trapezoidal_mask(out_stop - out_start, left_scaled, right[i] * scale, true);
            AxisTile {
                start: begin,
                end,
                out_start,
                out_stop,
                mask,
            }
        })
        .collect()
}

/// The full tiling plan for a latent `[_, _, f, h, w]`: per-axis tile lists + the output dims.
pub struct TilePlan {
    pub t: Vec<AxisTile>,
    pub h: Vec<AxisTile>,
    pub w: Vec<AxisTile>,
    pub out_f: i32,
    pub out_h: i32,
    pub out_w: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapezoid_no_ramp_is_all_ones() {
        assert_eq!(trapezoidal_mask(4, 0, 0, false), vec![1.0; 4]);
    }

    #[test]
    fn trapezoid_right_fade_out() {
        // ramp_right=2: last two = (3-1)/3, (3-2)/3 = 2/3, 1/3.
        let m = trapezoidal_mask(5, 0, 2, false);
        assert_eq!(m[0], 1.0);
        assert_eq!(m[2], 1.0);
        assert!((m[3] - 2.0 / 3.0).abs() < 1e-6);
        assert!((m[4] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn trapezoid_left_from_0_fade_in() {
        // ramp_left=2, left_from_0: linspace(0,1,3)[:-1] = [0, 0.5].
        let m = trapezoidal_mask(5, 2, 0, true);
        assert!((m[0] - 0.0).abs() < 1e-6);
        assert!((m[1] - 0.5).abs() < 1e-6);
        assert_eq!(m[2], 1.0);
    }

    #[test]
    fn spatial_split_three_tiles() {
        // tile=2, overlap=1, dim=4 → amount=(4+2-2-1)/1=3.
        let (starts, ends, left, right) = split_spatial(2, 1, 4);
        assert_eq!(starts, vec![0, 1, 2]);
        assert_eq!(ends, vec![2, 3, 4]);
        assert_eq!(left, vec![0, 1, 1]);
        assert_eq!(right, vec![1, 1, 0]);
    }

    #[test]
    fn temporal_split_causal_adjust() {
        // tile=2, overlap=1, dim=3 → spatial(2,1,3): amount=(3+2-2-1)/1=2, starts=[0,1].
        // temporal: starts[1]-=1 → [0,0], left[1]+=1.
        let (starts, _ends, left, _right) = split_temporal(2, 1, 3);
        assert_eq!(starts, vec![0, 0]);
        assert_eq!(left, vec![0, 2]);
    }

    #[test]
    fn needs_tiling_thresholds_ltx() {
        // LTX spatial_scale 32: tile_px 64 → 2 latent.
        let cfg = TilingConfig::spatial_only(64, 32);
        assert!(cfg.needs_tiling(VaeTiling::LTX, 1, 4, 4)); // h=4 > 2
        assert!(!cfg.needs_tiling(VaeTiling::LTX, 10, 2, 2)); // h=w=2 not > 2
        let tc = TilingConfig::temporal_only(16, 8); // temporal_scale 8: 16 → 2 latent
        assert!(tc.needs_tiling(VaeTiling::LTX, 3, 2, 2)); // f=3 > 2
        assert!(!tc.needs_tiling(VaeTiling::LTX, 2, 99, 99)); // f=2 not > 2
    }

    #[test]
    fn needs_tiling_thresholds_wan() {
        // Wan spatial_scale 8: tile_px 64 → 8 latent; temporal_scale 4: 16 frames → 4 latent.
        let cfg = TilingConfig::spatial_only(64, 32);
        assert!(cfg.needs_tiling(VaeTiling::WAN, 1, 9, 4)); // h=9 > 8
        assert!(!cfg.needs_tiling(VaeTiling::WAN, 10, 8, 8)); // h=w=8 not > 8
        let tc = TilingConfig::temporal_only(16, 8);
        assert!(tc.needs_tiling(VaeTiling::WAN, 5, 2, 2)); // f=5 > 4
        assert!(!tc.needs_tiling(VaeTiling::WAN, 4, 99, 99)); // f=4 not > 4
    }

    /// LTX (causal) temporal mapping: `out_f = 1 + (f−1)·8`, first tile starts at 0.
    #[test]
    fn plan_ltx_causal_temporal_output_dims() {
        let cfg = TilingConfig::temporal_only(16, 8); // tile=2, overlap=1 latent
        let plan = cfg.plan(VaeTiling::LTX, 3, 2, 2);
        assert_eq!(plan.out_f, 1 + (3 - 1) * 8); // 17
        assert_eq!(plan.out_h, 2 * 32);
        assert_eq!(plan.out_w, 2 * 32);
        assert_eq!(plan.t[0].out_start, 0);
    }

    /// Wan (non-causal) temporal mapping: `out_f = f·4`, temporal tiles behave like spatial.
    #[test]
    fn plan_wan_noncausal_temporal_output_dims() {
        let cfg = TilingConfig::temporal_only(16, 8); // temporal_scale 4: tile=4, overlap=2 latent
        let plan = cfg.plan(VaeTiling::WAN, 6, 2, 2);
        assert_eq!(plan.out_f, 6 * 4); // 24, NOT 1+(6-1)*4
        assert_eq!(plan.out_h, 2 * 8);
        assert_eq!(plan.out_w, 2 * 8);
        // Non-causal: the first temporal tile starts at 0 and maps out_start = 0.
        assert_eq!(plan.t[0].out_start, 0);
        assert_eq!(plan.t.last().unwrap().out_stop, 24);
    }

    /// Coverage invariant: the summed blend weight is strictly positive at **every** output position
    /// on each axis (no zero-weight gaps → the final divide is well-defined). Checked for both VAEs.
    #[test]
    fn plan_covers_every_output_position() {
        for (vae, f, h, w) in [(VaeTiling::WAN, 9, 9, 13), (VaeTiling::LTX, 5, 5, 5)] {
            let cfg = TilingConfig {
                spatial: Some(SpatialTiling {
                    tile_px: 4 * vae.spatial_scale,
                    overlap_px: 2 * vae.spatial_scale,
                }),
                temporal: Some(TemporalTiling {
                    tile_frames: 3 * vae.temporal_scale,
                    overlap_frames: vae.temporal_scale,
                }),
            };
            let plan = cfg.plan(vae, f, h, w);
            for (axis, tiles, out) in [
                ("t", &plan.t, plan.out_f),
                ("h", &plan.h, plan.out_h),
                ("w", &plan.w, plan.out_w),
            ] {
                let mut weight = vec![0f32; out as usize];
                for tile in tiles {
                    for (i, &m) in tile.mask.iter().enumerate() {
                        weight[tile.out_start as usize + i] += m;
                    }
                }
                assert!(
                    weight.iter().all(|&v| v > 1e-6),
                    "{vae:?} axis {axis}: zero-weight output position (gap in tiling)"
                );
            }
        }
    }

    /// F-005: degenerate tile/overlap configs (tile == overlap, overlap > tile, and a tile floored to
    /// 0 by latent downscaling) must not panic — they clamp to a valid split instead of dividing by
    /// zero or wrapping `amount` to a huge length.
    #[test]
    fn split_spatial_survives_degenerate_overlap() {
        // tile == overlap (would divide by zero), overlap > tile (would wrap), tile == 0 (floored).
        for (size, overlap) in [(8, 8), (8, 16), (0, 0), (0, 4)] {
            let (starts, ends, left, right) = split_spatial(size, overlap, 64);
            assert!(
                !starts.is_empty(),
                "size={size} overlap={overlap}: no tiles"
            );
            assert_eq!(starts.len(), ends.len());
            assert_eq!(left.len(), right.len());
            assert_eq!(*ends.last().unwrap(), 64, "last tile must reach dim");
        }
    }

    /// The crash is reachable through the public `plan` via `spatial_only`/`temporal_only` with a tile
    /// ≤ overlap; it must produce a valid, gap-free plan rather than panicking.
    #[test]
    fn plan_survives_tile_equal_overlap() {
        let cfg = TilingConfig::spatial_only(64, 64); // tile_px == overlap_px
        let plan = cfg.plan(VaeTiling::WAN, 1, 16, 16);
        for (tiles, out) in [(&plan.h, plan.out_h), (&plan.w, plan.out_w)] {
            let mut weight = vec![0f32; out as usize];
            for tile in tiles {
                for (i, &m) in tile.mask.iter().enumerate() {
                    weight[tile.out_start as usize + i] += m;
                }
            }
            assert!(
                weight.iter().all(|&v| v > 1e-6),
                "tile==overlap plan left a zero-weight gap"
            );
        }
    }
}
