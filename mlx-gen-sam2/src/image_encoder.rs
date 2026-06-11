//! SAM2 image encoder — Hiera trunk + FPN neck — port of `mlx_sam/models/image_encoder.py`.
//!
//! Turns the preprocessed `pixel_values[1,3,1024,1024]` into the segmenter's image features:
//! the FPN top-down feature pyramid (`backbone_fpn`), the coarsest map (`vision_features`), and a
//! sinusoidal position encoding per level (`vision_pos_enc`). These are exactly the inputs the SAM2
//! prompt-encoder + mask-decoder consume (sc-3706).
//!
//! Outputs are returned **NCHW** (`[1, C, H, W]`) to match the reference `Sam2ImageEncoder` and the
//! `_Sam2Segmenter` contract; the trunk/neck compute in NHWC and transpose once at this boundary.

use std::f64::consts::PI;

use mlx_rs::ops::{add, broadcast_to, concatenate_axis};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::{Sam2ImageEncoderConfig, Sam2ModelSize};
use crate::hiera::Hiera;

/// Position-encoding feature count passed to `PositionEmbeddingSine` in the reference neck
/// (`num_pos_feats=256`); the encoding itself uses half of this per spatial axis.
const POS_NUM_FEATS: i32 = 256;
const POS_TEMPERATURE: f64 = 10000.0;

use crate::util::join;

/// Sinusoidal position encoding (`PositionEmbeddingSine`, `normalize=True`). `x`: NHWC `[b,h,w,c]`;
/// returns NHWC `[b, h, w, num_pos_feats]`. The values depend only on `(h, w)`, so the per-axis
/// tables are built on the host (f64) and broadcast — bit-faithful to the reference's normalized
/// `sin(even)/cos(odd)` interleave. `num_pos_feats` is the reference constructor arg (256 for the
/// FPN neck; 64 for the memory encoder, [`crate::memory`]).
pub(crate) fn position_encoding(x: &Array, num_pos_feats: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w) = (sh[0], sh[1], sh[2]);
    let feats = (num_pos_feats / 2) as usize; // self.num_pos_feats = num_pos_feats // 2
    let scale = 2.0 * PI;
    let eps = 1e-6;

    // dim_t[m] = temperature ^ (2m / feats), shared by output channels 2m and 2m+1.
    let dim_t: Vec<f64> = (0..feats / 2)
        .map(|m| POS_TEMPERATURE.powf((2 * m) as f64 / feats as f64))
        .collect();

    // Per-axis table[pos * feats + c] with c even ⇒ sin, c odd ⇒ cos (of pos_norm / dim_t[c/2]).
    let axis_table = |n: i32| -> Vec<f32> {
        let mut t = Vec::with_capacity(n as usize * feats);
        for p in 0..n {
            let pos_norm = (p as f64 + 1.0) / (n as f64 + eps) * scale;
            for c in 0..feats {
                let v = pos_norm / dim_t[c / 2];
                t.push(if c % 2 == 0 { v.sin() } else { v.cos() } as f32);
            }
        }
        t
    };

    let f = feats as i32;
    let pos_y = Array::from_slice(&axis_table(h), &[h, f]).reshape(&[1, h, 1, f])?;
    let pos_x = Array::from_slice(&axis_table(w), &[w, f]).reshape(&[1, 1, w, f])?;
    let pos_y = broadcast_to(&pos_y, &[b, h, w, f])?;
    let pos_x = broadcast_to(&pos_x, &[b, h, w, f])?;
    let out = concatenate_axis(&[&pos_y, &pos_x], 3)?;
    Ok(out.as_dtype(x.dtype())?)
}

/// FPN neck: 1×1 lateral convs + a top-down nearest-2× merge on the configured levels.
struct FpnNeck {
    /// `(weight, bias)` per 1×1 conv, indexed 0.. (coarse→fine, matching `backbone_channel_list`).
    convs: Vec<(Array, Array)>,
    fpn_top_down_levels: Vec<i32>,
    scalp: usize,
}

impl FpnNeck {
    fn from_weights(w: &Weights, prefix: &str, cfg: &Sam2ImageEncoderConfig) -> Result<Self> {
        let convs = (0..cfg.fpn.backbone_channel_list.len())
            .map(|i| -> Result<(Array, Array)> {
                Ok((
                    w.require(&join(prefix, &format!("convs.{i}.weight")))?
                        .clone(),
                    w.require(&join(prefix, &format!("convs.{i}.bias")))?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            convs,
            fpn_top_down_levels: cfg.fpn.fpn_top_down_levels.clone(),
            scalp: cfg.fpn.scalp as usize,
        })
    }

    /// `xs`: trunk outputs NHWC, fine→coarse. Returns `(features, pos)` NHWC, fine→coarse, after
    /// dropping the `scalp` coarsest levels.
    fn forward(&self, xs: &[Array]) -> Result<(Vec<Array>, Vec<Array>)> {
        let n = self.convs.len() - 1;
        let mut out: Vec<Option<Array>> = vec![None; self.convs.len()];
        let mut pos: Vec<Option<Array>> = vec![None; self.convs.len()];
        let mut prev: Option<Array> = None;

        for i in (0..=n).rev() {
            let (cw, cb) = &self.convs[n - i];
            let lateral = conv2d(&xs[i], cw, Some(cb), 1, 0)?; // 1×1, stride 1, no pad
            let cur = match &prev {
                Some(p) if self.fpn_top_down_levels.contains(&(i as i32)) => {
                    add(&lateral, &upsample_nearest(p, 2)?)?
                }
                _ => lateral,
            };
            pos[i] = Some(position_encoding(&cur, POS_NUM_FEATS)?);
            out[i] = Some(cur.clone());
            prev = Some(cur);
        }

        let mut features: Vec<Array> = out.into_iter().map(|o| o.unwrap()).collect();
        let mut positions: Vec<Array> = pos.into_iter().map(|o| o.unwrap()).collect();
        // Drop the `scalp` coarsest levels (the FPN returns levels[: len - scalp]).
        features.truncate(features.len() - self.scalp);
        positions.truncate(positions.len() - self.scalp);
        Ok((features, positions))
    }
}

/// SAM2 image encoder output (all NCHW). `backbone_fpn` is fine→coarse; `vision_features` is the
/// coarsest map (`backbone_fpn.last()`); `vision_pos_enc` is the matching per-level position
/// encoding the mask-decoder transformer adds to the image tokens.
pub struct Sam2ImageEncoderOutput {
    pub backbone_fpn: Vec<Array>,
    pub vision_features: Array,
    pub vision_pos_enc: Vec<Array>,
}

/// SAM2 image encoder: Hiera trunk + FPN neck.
pub struct Sam2ImageEncoder {
    trunk: Hiera,
    neck: FpnNeck,
}

impl Sam2ImageEncoder {
    /// Build from a converted MLX SAM2 checkpoint (`trunk.*` / `neck.*` keys) for the given config.
    pub fn from_weights(w: &Weights, cfg: &Sam2ImageEncoderConfig) -> Result<Self> {
        Ok(Self {
            trunk: Hiera::from_weights(w, "trunk", &cfg.hiera)?,
            neck: FpnNeck::from_weights(w, "neck", cfg)?,
        })
    }

    /// Convenience: build for a named model size.
    pub fn from_weights_for_size(w: &Weights, size: Sam2ModelSize) -> Result<Self> {
        Self::from_weights(w, &Sam2ImageEncoderConfig::for_size(size))
    }

    /// `pixel_values`: NCHW `[1, 3, 1024, 1024]` (SAM2 preprocessing). Returns the FPN features,
    /// the coarsest feature map, and per-level position encodings — all NCHW.
    pub fn forward(&self, pixel_values: &Array) -> Result<Sam2ImageEncoderOutput> {
        let trunk_out = self.trunk.forward(pixel_values)?; // NHWC, fine→coarse
        let (features_nhwc, pos_nhwc) = self.neck.forward(&trunk_out)?;

        // NHWC → NCHW at the boundary.
        let to_nchw = |a: &Array| -> Result<Array> { Ok(a.transpose_axes(&[0, 3, 1, 2])?) };
        let backbone_fpn = features_nhwc
            .iter()
            .map(to_nchw)
            .collect::<Result<Vec<_>>>()?;
        let vision_pos_enc = pos_nhwc.iter().map(to_nchw).collect::<Result<Vec<_>>>()?;
        let vision_features = backbone_fpn
            .last()
            .expect("FPN yields at least one feature level")
            .clone();
        Ok(Sam2ImageEncoderOutput {
            backbone_fpn,
            vision_features,
            vision_pos_enc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hiera::block_specs;
    use mlx_rs::ops::{abs, max};

    /// `position_encoding` channel 2m/2m+1 are `sin`/`cos` of the same argument, so each such pair
    /// satisfies `sin² + cos² = 1` — a layout-correctness invariant independent of weights.
    #[test]
    fn position_encoding_sin_cos_pairs_are_unit() {
        use mlx_rs::ops::{multiply, subtract};
        let x = Array::from_slice(&vec![0f32; 5 * 7 * 8], &[1, 5, 7, 8]);
        let pe = position_encoding(&x, POS_NUM_FEATS).unwrap();
        assert_eq!(pe.shape(), &[1, 5, 7, POS_NUM_FEATS]);
        // Pair the first (sin) and second (cos) channels of the pos_y block.
        let sin0 = pe.take_axis(Array::from_int(0), 3).unwrap();
        let cos0 = pe.take_axis(Array::from_int(1), 3).unwrap();
        let sumsq = add(
            multiply(&sin0, &sin0).unwrap(),
            multiply(&cos0, &cos0).unwrap(),
        )
        .unwrap();
        let one = Array::from_slice(&vec![1f32; sumsq.size()], sumsq.shape());
        let err = max(abs(subtract(&sumsq, &one).unwrap()).unwrap(), None)
            .unwrap()
            .item::<f32>();
        assert!(err < 1e-5, "sin²+cos² deviated by {err:e}");
    }

    /// Write a zero-weight synthetic checkpoint at reduced resolution (grid 64 vs the production
    /// 256) and run the full encoder forward, asserting the FPN feature/pos shapes — validates the
    /// whole graph end-to-end with no weight download.
    #[test]
    fn encoder_forward_emits_fpn_shapes() {
        // Tiny config scaled to a 256² input (grid 64) so the test stays light + fast.
        let mut cfg = Sam2ImageEncoderConfig::tiny();
        cfg.hiera.image_size = 256;
        cfg.hiera.pos_embed_hw = 64;

        let path = std::env::temp_dir().join("mlx_gen_sam2_synth_encoder.safetensors");
        write_synthetic_checkpoint(&cfg, &path);

        let w = Weights::from_file(&path).unwrap();
        let enc = Sam2ImageEncoder::from_weights(&w, &cfg).unwrap();
        let pixel_values = Array::from_slice(&vec![0f32; 3 * 256 * 256], &[1, 3, 256, 256]);
        let out = enc.forward(&pixel_values).unwrap();

        // backbone_fpn: 3 levels (4 trunk stages − scalp 1), all d_model channels, fine→coarse.
        assert_eq!(out.backbone_fpn.len(), 3);
        assert_eq!(out.vision_pos_enc.len(), 3);
        assert_eq!(out.backbone_fpn[0].shape(), &[1, 256, 64, 64]);
        assert_eq!(out.backbone_fpn[1].shape(), &[1, 256, 32, 32]);
        assert_eq!(out.backbone_fpn[2].shape(), &[1, 256, 16, 16]);
        assert_eq!(out.vision_features.shape(), &[1, 256, 16, 16]);
        assert_eq!(out.vision_pos_enc[0].shape(), &[1, 256, 64, 64]);
        let _ = std::fs::remove_file(&path);
    }

    /// Build a zero-filled checkpoint with every key the encoder requires, at the config's shapes.
    fn write_synthetic_checkpoint(cfg: &Sam2ImageEncoderConfig, path: &std::path::Path) {
        let zeros = |shape: &[i32]| {
            let n: i32 = shape.iter().product();
            Array::from_slice(&vec![0f32; n as usize], shape)
        };
        let h = &cfg.hiera;
        let mut t: Vec<(String, Array)> = vec![
            (
                "trunk.pos_embed_full".into(),
                zeros(&[1, h.pos_embed_hw, h.pos_embed_hw, h.embed_dim]),
            ),
            (
                "trunk.patch_embed.proj.weight".into(),
                zeros(&[h.embed_dim, 7, 7, 3]),
            ),
            ("trunk.patch_embed.proj.bias".into(), zeros(&[h.embed_dim])),
        ];
        for (i, s) in block_specs(h).iter().enumerate() {
            let p = format!("trunk.blocks.{i}");
            let hidden = (s.dim_out as f32 * h.mlp_ratio) as i32;
            t.push((format!("{p}.norm1.weight"), zeros(&[s.dim])));
            t.push((format!("{p}.norm1.bias"), zeros(&[s.dim])));
            t.push((
                format!("{p}.attn.qkv.weight"),
                zeros(&[s.dim_out * 3, s.dim]),
            ));
            t.push((format!("{p}.attn.qkv.bias"), zeros(&[s.dim_out * 3])));
            t.push((
                format!("{p}.attn.proj.weight"),
                zeros(&[s.dim_out, s.dim_out]),
            ));
            t.push((format!("{p}.attn.proj.bias"), zeros(&[s.dim_out])));
            t.push((format!("{p}.norm2.weight"), zeros(&[s.dim_out])));
            t.push((format!("{p}.norm2.bias"), zeros(&[s.dim_out])));
            t.push((
                format!("{p}.mlp.layers.0.weight"),
                zeros(&[hidden, s.dim_out]),
            ));
            t.push((format!("{p}.mlp.layers.0.bias"), zeros(&[hidden])));
            t.push((
                format!("{p}.mlp.layers.1.weight"),
                zeros(&[s.dim_out, hidden]),
            ));
            t.push((format!("{p}.mlp.layers.1.bias"), zeros(&[s.dim_out])));
            if s.dim != s.dim_out {
                t.push((format!("{p}.proj.weight"), zeros(&[s.dim_out, s.dim])));
                t.push((format!("{p}.proj.bias"), zeros(&[s.dim_out])));
            }
        }
        for (j, &ch) in cfg.fpn.backbone_channel_list.iter().enumerate() {
            t.push((
                format!("neck.convs.{j}.weight"),
                zeros(&[cfg.fpn.d_model, 1, 1, ch]),
            ));
            t.push((format!("neck.convs.{j}.bias"), zeros(&[cfg.fpn.d_model])));
        }
        let refs: Vec<(&str, &Array)> = t.iter().map(|(k, v)| (k.as_str(), v)).collect();
        Array::save_safetensors(refs, None, path).unwrap();
    }
}
