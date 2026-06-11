//! Hiera hierarchical-ViT image trunk — port of `mlx_sam/models/hiera.py` (`avbiswas/sam2-mlx`).
//!
//! A 4-stage windowed-attention transformer over NHWC feature maps. Each stage halves the spatial
//! resolution (a `q_stride` max-pool on the query at the first block of stages 2–4) and doubles the
//! channel width; `global_att_blocks` run full (non-windowed) attention. The trunk consumes the
//! preprocessed `pixel_values[1,3,1024,1024]` and returns one NHWC feature map per stage end (the
//! FPN neck — [`crate::image_encoder`] — turns those into the segmenter's backbone features).
//!
//! Layout note: the trunk runs entirely in **NHWC** (MLX-native, channels-last). The reference
//! transposes each stage output to NCHW only for the neck's benefit; we keep NHWC end-to-end and
//! transpose once at the encoder boundary, which is numerically identical and avoids double work.

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::module::Module;
use mlx_rs::nn::MaxPool2d;
use mlx_rs::ops::{add, pad};
use mlx_rs::Array;

use mlx_gen::nn::{conv2d, gelu_exact, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use crate::config::HieraConfig;

/// LayerNorm epsilon (`nn.LayerNorm(dim, eps=1e-6)`).
const EPS: f32 = 1e-6;

use crate::util::join;

/// Non-overlapping max-pool (`nn.MaxPool2d(kernel_size=stride, stride=stride)`) over NHWC `x`.
/// Output spatial dim is `floor((d - stride)/stride) + 1`, matching torch/MLX pooling.
fn max_pool(x: &Array, stride: i32) -> Result<Array> {
    let mut pool = MaxPool2d::new(stride, stride as i64);
    Ok(pool.forward(x)?)
}

/// Contiguous leading slice `x[:, :h, :w, :]` over an NHWC array (used to drop window padding).
fn crop_hw(x: &Array, h: i32, w: i32) -> Result<Array> {
    let rows = Array::from_slice(&(0..h).collect::<Vec<i32>>(), &[h]);
    let cols = Array::from_slice(&(0..w).collect::<Vec<i32>>(), &[w]);
    Ok(x.take_axis(&rows, 1)?.take_axis(&cols, 2)?)
}

/// Partition NHWC `x` into `window`×`window` windows along H/W, zero-padding to a multiple of
/// `window`. Returns the `[-1, window, window, c]` windows plus the padded `(hp, wp)`.
fn window_partition(x: &Array, window: i32) -> Result<(Array, (i32, i32))> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let pad_h = (window - h % window) % window;
    let pad_w = (window - w % window) % window;
    let x = if pad_h > 0 || pad_w > 0 {
        pad(x, &[(0, 0), (0, pad_h), (0, pad_w), (0, 0)][..], None, None)?
    } else {
        x.clone()
    };
    let (hp, wp) = (h + pad_h, w + pad_w);
    let windows = x
        .reshape(&[b, hp / window, window, wp / window, window, c])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[-1, window, window, c])?;
    Ok((windows, (hp, wp)))
}

/// Inverse of [`window_partition`]: stitch windows back to `[b, h, w, c]`, cropping padding.
fn window_unpartition(
    windows: &Array,
    window: i32,
    pad_hw: (i32, i32),
    hw: (i32, i32),
) -> Result<Array> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let num_per_image = (hp * wp) / window / window;
    let b = windows.shape()[0] / num_per_image;
    let x = windows
        .reshape(&[b, hp / window, wp / window, window, window, -1])?
        .transpose_axes(&[0, 1, 3, 2, 4, 5])?
        .reshape(&[b, hp, wp, -1])?;
    if hp > h || wp > w {
        crop_hw(&x, h, w)
    } else {
        Ok(x)
    }
}

/// Two-layer GELU MLP (`mlp.layers.0` → gelu → `mlp.layers.1`).
struct Mlp {
    fc1_w: Array,
    fc1_b: Array,
    fc2_w: Array,
    fc2_b: Array,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1_w: w.require(&join(prefix, "layers.0.weight"))?.clone(),
            fc1_b: w.require(&join(prefix, "layers.0.bias"))?.clone(),
            fc2_w: w.require(&join(prefix, "layers.1.weight"))?.clone(),
            fc2_b: w.require(&join(prefix, "layers.1.bias"))?.clone(),
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = gelu_exact(&linear(x, &self.fc1_w, &self.fc1_b)?)?;
        linear(&h, &self.fc2_w, &self.fc2_b)
    }
}

/// Multi-scale attention with an optional `q_stride` query pool (`MultiScaleAttention`).
struct MultiScaleAttention {
    qkv_w: Array,
    qkv_b: Array,
    proj_w: Array,
    proj_b: Array,
    num_heads: i32,
    /// `dim_out / num_heads`.
    head_dim: i32,
    /// `Some(stride)` ⇒ pool the query before attention (the stage-boundary downsample).
    q_stride: Option<i32>,
}

impl MultiScaleAttention {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        num_heads: i32,
        head_dim: i32,
        q_stride: Option<i32>,
    ) -> Result<Self> {
        Ok(Self {
            qkv_w: w.require(&join(prefix, "qkv.weight"))?.clone(),
            qkv_b: w.require(&join(prefix, "qkv.bias"))?.clone(),
            proj_w: w.require(&join(prefix, "proj.weight"))?.clone(),
            proj_b: w.require(&join(prefix, "proj.bias"))?.clone(),
            num_heads,
            head_dim,
            q_stride,
        })
    }

    /// `x`: NHWC `[b, h, w, dim]` (b = batch·num_windows). Returns `[b, h', w', dim_out]`
    /// (h'/w' halved when `q_stride` is set).
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, mut h, mut w) = (sh[0], sh[1], sh[2]);
        let (nh, hd) = (self.num_heads, self.head_dim);

        // qkv → [b, h*w, 3, nh, hd]; split into q/k/v of [b, h*w, nh, hd].
        let qkv = linear(x, &self.qkv_w, &self.qkv_b)?.reshape(&[b, h * w, 3, nh, hd])?;
        let split = |i: i32| -> Result<Array> {
            Ok(qkv
                .take_axis(Array::from_int(i), 2)?
                .reshape(&[b, h * w, nh, hd])?)
        };
        let mut q = split(0)?;
        let k = split(1)?;
        let v = split(2)?;

        if let Some(stride) = self.q_stride {
            // Pool the query spatially (kernel=stride=q_stride), shrinking the token count.
            q = max_pool(&q.reshape(&[b, h, w, nh * hd])?, stride)?;
            let qs = q.shape();
            h = qs[1];
            w = qs[2];
            q = q.reshape(&[b, h * w, nh, hd])?;
        }

        // [b, tokens, nh, hd] → [b, nh, tokens, hd] for SDPA.
        let scale = 1.0 / (hd as f32).sqrt();
        let attn = scaled_dot_product_attention(
            &q.transpose_axes(&[0, 2, 1, 3])?,
            &k.transpose_axes(&[0, 2, 1, 3])?,
            &v.transpose_axes(&[0, 2, 1, 3])?,
            scale,
            None,
            None,
        )?;
        let out = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, h, w, nh * hd])?;
        linear(&out, &self.proj_w, &self.proj_b)
    }
}

/// One Hiera block: windowed multi-scale attention + MLP, both pre-norm residual. Stage-boundary
/// blocks (`dim != dim_out`) project + pool the residual to match the attention's pooled output.
struct MultiScaleBlock {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    attn: MultiScaleAttention,
    mlp: Mlp,
    /// Residual projection + pool (present iff `dim != dim_out`).
    proj: Option<(Array, Array)>,
    window_size: i32,
    q_stride: Option<i32>,
}

impl MultiScaleBlock {
    #[allow(clippy::too_many_arguments)]
    fn from_weights(
        w: &Weights,
        prefix: &str,
        dim: i32,
        dim_out: i32,
        num_heads: i32,
        window_size: i32,
        q_stride: Option<i32>,
    ) -> Result<Self> {
        let head_dim = dim_out / num_heads;
        let proj = if dim != dim_out {
            Some((
                w.require(&join(prefix, "proj.weight"))?.clone(),
                w.require(&join(prefix, "proj.bias"))?.clone(),
            ))
        } else {
            None
        };
        Ok(Self {
            norm1_w: w.require(&join(prefix, "norm1.weight"))?.clone(),
            norm1_b: w.require(&join(prefix, "norm1.bias"))?.clone(),
            norm2_w: w.require(&join(prefix, "norm2.weight"))?.clone(),
            norm2_b: w.require(&join(prefix, "norm2.bias"))?.clone(),
            attn: MultiScaleAttention::from_weights(
                w,
                &join(prefix, "attn"),
                num_heads,
                head_dim,
                q_stride,
            )?,
            mlp: Mlp::from_weights(w, &join(prefix, "mlp"))?,
            proj,
            window_size,
            q_stride,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let normed = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), EPS)?;

        // Residual: at a stage boundary, project (Linear) + pool to the new dim/resolution.
        let shortcut = match &self.proj {
            Some((pw, pb)) => {
                let stride = self.q_stride.expect("stage-boundary block always pools");
                max_pool(&linear(&normed, pw, pb)?, stride)?
            }
            None => x.clone(),
        };

        // Windowed attention (window_size == 0 ⇒ global attention over the full map).
        let mut window = self.window_size;
        let (mut h, mut w) = (0, 0);
        let mut pad_hw = (0, 0);
        let mut attended = if window > 0 {
            h = normed.shape()[1];
            w = normed.shape()[2];
            let (windows, p) = window_partition(&normed, window)?;
            pad_hw = p;
            self.attn.forward(&windows)?
        } else {
            self.attn.forward(&normed)?
        };

        // After a query pool the effective window + spatial dims halve; recompute the unpartition
        // geometry from the (already-pooled) shortcut.
        if let Some(stride) = self.q_stride {
            window /= stride;
            let ssh = shortcut.shape();
            h = ssh[1];
            w = ssh[2];
            let pad_h = (window - h % window) % window;
            let pad_w = (window - w % window) % window;
            pad_hw = (h + pad_h, w + pad_w);
        }

        if self.window_size > 0 {
            attended = window_unpartition(&attended, window, pad_hw, (h, w))?;
        }

        let x = add(&shortcut, &attended)?;
        let mlp_in = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), EPS)?;
        Ok(add(&x, &self.mlp.forward(&mlp_in)?)?)
    }
}

/// Resolved per-block hyperparameters (the output of [`block_specs`]).
pub(crate) struct BlockSpec {
    pub dim: i32,
    pub dim_out: i32,
    pub num_heads: i32,
    pub window: i32,
    pub q_stride: Option<i32>,
}

/// `stage_ends[s] = (Σ stages[..=s]) - 1` — the block index that closes each stage.
pub(crate) fn stage_ends(cfg: &HieraConfig) -> Vec<i32> {
    let mut acc = 0;
    cfg.stages
        .iter()
        .map(|&s| {
            acc += s;
            acc - 1
        })
        .collect()
}

/// Derive every block's `(dim, dim_out, num_heads, window, q_stride)`, replicating the reference
/// loop exactly. The subtle part: `window` is read with the *pre-increment* `cur_stage`, so a
/// stage-boundary block (which projects/pools to the next width) still uses the *previous* stage's
/// window size — matching `mlx_sam.models.hiera.Hiera.__init__`.
pub(crate) fn block_specs(cfg: &HieraConfig) -> Vec<BlockSpec> {
    let depth: i32 = cfg.stages.iter().sum();
    let ends = stage_ends(cfg);
    // q-pool blocks = first `q_pool` of {stage_end + 1} (excluding the final stage end).
    let q_pool_blocks: Vec<i32> = ends[..ends.len() - 1]
        .iter()
        .map(|e| e + 1)
        .take(cfg.q_pool as usize)
        .collect();

    let mut specs = Vec::with_capacity(depth as usize);
    let mut embed_dim = cfg.embed_dim;
    let mut num_heads = cfg.num_heads;
    let mut cur_stage = 1usize;
    for i in 0..depth {
        let mut window = cfg.window_spec[cur_stage - 1];
        if cfg.global_att_blocks.contains(&i) {
            window = 0;
        }
        let mut dim_out = embed_dim;
        if i > 0 && ends.contains(&(i - 1)) {
            dim_out = (embed_dim as f32 * cfg.dim_mul) as i32;
            num_heads = (num_heads as f32 * cfg.head_mul) as i32;
            cur_stage += 1;
        }
        let q_stride = if q_pool_blocks.contains(&i) {
            Some(cfg.q_stride)
        } else {
            None
        };
        specs.push(BlockSpec {
            dim: embed_dim,
            dim_out,
            num_heads,
            window,
            q_stride,
        });
        embed_dim = dim_out;
    }
    specs
}

/// Hiera trunk: patch-embed → add learned position embedding → staged windowed-attention blocks.
pub struct Hiera {
    patch_proj_w: Array,
    patch_proj_b: Array,
    /// Learned absolute position embedding, NHWC `[1, pos_hw, pos_hw, embed_dim]` (the converter
    /// has already fused `pos_embed + tiled window pos_embed` into this single tensor).
    pos_embed_full: Array,
    blocks: Vec<MultiScaleBlock>,
    /// Block indices whose output is a stage end (collected as a trunk output).
    stage_ends: Vec<i32>,
}

impl Hiera {
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &HieraConfig) -> Result<Self> {
        let p = |leaf: &str| join(prefix, leaf);

        // Build per-block specs (window read *before* the stage increment — see `block_specs`).
        let blocks = block_specs(cfg)
            .iter()
            .enumerate()
            .map(|(i, s)| {
                MultiScaleBlock::from_weights(
                    w,
                    &p(&format!("blocks.{i}")),
                    s.dim,
                    s.dim_out,
                    s.num_heads,
                    s.window,
                    s.q_stride,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let stage_ends = stage_ends(cfg);

        Ok(Self {
            patch_proj_w: w.require(&p("patch_embed.proj.weight"))?.clone(),
            patch_proj_b: w.require(&p("patch_embed.proj.bias"))?.clone(),
            pos_embed_full: w.require(&p("pos_embed_full"))?.clone(),
            blocks,
            stage_ends,
        })
    }

    /// `pixel_values`: NCHW `[1, 3, 1024, 1024]` (the SAM2 preprocessing contract). Returns one NHWC
    /// feature map per stage end (`[1, H_s, W_s, C_s]`, fine→coarse).
    pub fn forward(&self, pixel_values: &Array) -> Result<Vec<Array>> {
        // NCHW → NHWC, then patch-embed conv (kernel 7, stride 4, pad 3).
        let x = pixel_values.transpose_axes(&[0, 2, 3, 1])?;
        let mut x = conv2d(&x, &self.patch_proj_w, Some(&self.patch_proj_b), 4, 3)?;

        // Add the learned position embedding (cropped to the patch grid; identity at 1024²).
        let sh = x.shape();
        let (h, w) = (sh[1], sh[2]);
        let pos = crop_hw(&self.pos_embed_full, h, w)?;
        x = add(&x, &pos)?;

        let mut outputs = Vec::with_capacity(self.stage_ends.len());
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if self.stage_ends.contains(&(i as i32)) {
                outputs.push(x.clone());
            }
        }
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Sam2ImageEncoderConfig;
    use mlx_rs::ops::array_eq;

    /// The derived per-stage output channels (dim_out at each stage end) must equal the FPN's
    /// `backbone_channel_list` (reversed — list is coarse→fine, stage ends are fine→coarse).
    #[test]
    fn block_specs_channels_match_fpn_config() {
        for cfg in [
            Sam2ImageEncoderConfig::tiny(),
            Sam2ImageEncoderConfig::small(),
            Sam2ImageEncoderConfig::base_plus(),
            Sam2ImageEncoderConfig::large(),
        ] {
            let specs = block_specs(&cfg.hiera);
            let ends = stage_ends(&cfg.hiera);
            let stage_channels: Vec<i32> =
                ends.iter().map(|&e| specs[e as usize].dim_out).collect();
            let mut expected = cfg.fpn.backbone_channel_list.clone();
            expected.reverse(); // coarse→fine list vs fine→coarse stage ends
            assert_eq!(
                stage_channels, expected,
                "stage channels {stage_channels:?} != reversed backbone {expected:?}"
            );
        }
    }

    /// Large-model spec spot-checks: the stage-boundary (q-pool) blocks use the *previous* stage's
    /// window (the read-before-increment subtlety), heads double per stage, and globals window-0.
    #[test]
    fn block_specs_large_known_values() {
        let cfg = Sam2ImageEncoderConfig::large().hiera;
        let s = block_specs(&cfg);
        assert_eq!(s.len(), 48);
        assert_eq!(stage_ends(&cfg), vec![1, 7, 43, 47]);

        // q-pool / stage-boundary blocks: 2, 8, 44.
        let check = |i: usize, dim, dim_out, heads, window| {
            assert_eq!(s[i].dim, dim, "block {i} dim");
            assert_eq!(s[i].dim_out, dim_out, "block {i} dim_out");
            assert_eq!(s[i].num_heads, heads, "block {i} heads");
            assert_eq!(s[i].window, window, "block {i} window");
            assert_eq!(s[i].q_stride, Some(2), "block {i} q_stride");
        };
        check(2, 144, 288, 4, 8); // window 8 = stage-1 spec, read before the increment
        check(8, 288, 576, 8, 4);
        check(44, 576, 1152, 16, 16);

        // Non-boundary + global blocks.
        assert_eq!(s[0].q_stride, None);
        assert_eq!(s[0].window, 8);
        assert_eq!(s[23].window, 0, "block 23 is a global-attention block");
        assert_eq!(s[23].q_stride, None);
    }

    /// `window_unpartition ∘ window_partition` is identity over the unpadded region (exercises the
    /// pad-then-crop path with H/W not divisible by the window).
    #[test]
    fn window_partition_round_trips() {
        let (h, w, c) = (12, 10, 4);
        let n = h * w * c;
        let vals: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let x = Array::from_slice(&vals, &[1, h, w, c]);
        let (windows, pad_hw) = window_partition(&x, 8).unwrap();
        assert_eq!(pad_hw, (16, 16));
        let back = window_unpartition(&windows, 8, pad_hw, (h, w)).unwrap();
        assert_eq!(back.shape(), &[1, h, w, c]);
        assert!(array_eq(&back, &x, false).unwrap().item::<bool>());
    }
}
