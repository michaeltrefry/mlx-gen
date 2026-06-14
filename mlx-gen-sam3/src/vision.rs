//! SAM3 PE vision encoder — port of `Sam3ViTModel` (PE backbone) + `Sam3VisionNeck` (FPN)
//! from `transformers/models/sam3/modeling_sam3.py` (epic 4910, sc-4919).
//!
//! The backbone is an isotropic windowed ViT (NOT SAM2's hierarchical Hiera): patch-embed (conv
//! stride 14, no bias) → tiled absolute position embedding → a front LayerNorm → 32 pre-norm
//! transformer layers. Most layers run **windowed** attention (window 24); layers [7,15,23,31] run
//! **global** attention. Every layer applies **2D axial RoPE** to q/k (the rotary table is fixed per
//! layer: window-sized for windowed layers, grid-sized + down-scaled for global layers). No
//! LayerScale (`layer_scale_init_value` is None in the shipped config).
//!
//! The neck runs one FPN branch per scale factor [4,2,1,0.5] over the 72² backbone grid, yielding
//! four 256-channel feature maps at 288²/144²/72²/36². Layout: backbone runs NHWC end-to-end (MLX
//! native); conv/transposed-conv weights are permuted from torch OIHW/IOHW to MLX OHWI at load.
//!
//! Scope (Phase A): the FPN **feature maps**. The neck's sine position encodings (consumed by the
//! DETR encoder) land with Phase C (sc-4921).

use std::rc::Rc;

use mlx_rs::fast::{layer_norm, scaled_dot_product_attention};
use mlx_rs::module::Module;
use mlx_rs::nn::MaxPool2d;
use mlx_rs::ops::{add, concatenate_axis, conv_transpose2d, negative, pad, stack_axis};
use mlx_rs::Array;

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::nn::{conv2d, gelu_exact};
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::config::Sam3VisionConfig;
use crate::util::{conv_transpose_w, conv_w_ohwi, join};

/// Partition NHWC `x` into `window`×`window` windows (zero-padding to a multiple of `window`).
/// Returns the `[-1, window, window, c]` windows + the padded `(hp, wp)`. (Port of SAM3
/// `window_partition`; identical to the SAM2 trunk's.)
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
        let rows = Array::from_slice(&(0..h).collect::<Vec<i32>>(), &[h]);
        let cols = Array::from_slice(&(0..w).collect::<Vec<i32>>(), &[w]);
        Ok(x.take_axis(&rows, 1)?.take_axis(&cols, 2)?)
    } else {
        Ok(x)
    }
}

/// Precomputed 2D-axial RoPE `(cos, sin)`, each `[end·end, head_dim]`, for a fixed feature grid.
/// `freqs[j] = θ^(-(4j)/head_dim)` over `j∈[0, head_dim/4)`; per position `i` the row is
/// `[x·freqs, y·freqs]` (x = i%end, y = i/end, both ·`scale`) then `repeat_interleave(2)`.
#[derive(Clone)]
struct RopeTable {
    cos: Array,
    sin: Array,
}

impl RopeTable {
    fn new(end: i32, scale: f32, theta: f32, head_dim: i32) -> Self {
        let quarter = head_dim / 4;
        let freqs: Vec<f32> = (0..quarter)
            .map(|j| 1.0 / theta.powf((4 * j) as f32 / head_dim as f32))
            .collect();
        let n = end * end;
        let mut cos = Vec::with_capacity((n * head_dim) as usize);
        let mut sin = Vec::with_capacity((n * head_dim) as usize);
        for i in 0..n {
            let x = (i % end) as f32 * scale;
            let y = (i / end) as f32 * scale;
            // 32 values [x·freqs (16), y·freqs (16)], each then duplicated (repeat_interleave 2).
            let row = freqs
                .iter()
                .map(|&f| x * f)
                .chain(freqs.iter().map(|&f| y * f));
            for v in row {
                let (c, s) = (v.cos(), v.sin());
                cos.push(c);
                cos.push(c);
                sin.push(s);
                sin.push(s);
            }
        }
        Self {
            cos: Array::from_slice(&cos, &[n, head_dim]),
            sin: Array::from_slice(&sin, &[n, head_dim]),
        }
    }
}

/// `rotate_pairwise(x)`: pairwise `(a, b) -> (-b, a)` over the last dim (the SAM3 interleaved
/// convention, paired with `repeat_interleave(2)` cos/sin).
fn rotate_pairwise(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let nd = sh.len();
    let hd = sh[nd - 1];
    let mut paired: Vec<i32> = sh[..nd - 1].to_vec();
    paired.push(hd / 2);
    paired.push(2);
    let xr = x.reshape(&paired)?;
    let last = paired.len() as i32 - 1;
    let x1 = xr.take_axis(Array::from_int(0), last)?; // even lane
    let x2 = xr.take_axis(Array::from_int(1), last)?; // odd lane
    let stacked = stack_axis(&[&negative(&x2)?, &x1], last)?;
    Ok(stacked.reshape(sh)?)
}

/// `q_embed = q·cos + rotate_pairwise(q)·sin`. `q`: `[b, nh, seq, hd]`; `cos`/`sin`: `[seq, hd]`.
fn apply_rope(q: &Array, table: &RopeTable) -> Result<Array> {
    let a = q.multiply(&table.cos)?;
    let b = rotate_pairwise(q)?.multiply(&table.sin)?;
    Ok(add(&a, &b)?)
}

/// Two-layer GELU MLP (`mlp.fc1` → exact-gelu → `mlp.fc2`).
#[derive(Clone)]
struct Mlp {
    fc1: AdaptableLinear,
    fc2: AdaptableLinear,
}

impl Mlp {
    fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            fc1: crate::load_linear(w, &join(prefix, "fc1"))?,
            fc2: crate::load_linear(w, &join(prefix, "fc2"))?,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        crate::quantize_linear(&mut self.fc1, bits)?;
        crate::quantize_linear(&mut self.fc2, bits)?;
        Ok(())
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let h = gelu_exact(&self.fc1.forward(x)?)?;
        self.fc2.forward(&h)
    }
}

/// RoPE self-attention (separate q/k/v/o projections). Operates on NHWC `[b, H, W, C]`
/// (`b = batch·num_windows` for windowed layers).
#[derive(Clone)]
struct Attention {
    q: AdaptableLinear,
    k: AdaptableLinear,
    v: AdaptableLinear,
    o: AdaptableLinear,
    num_heads: i32,
    head_dim: i32,
}

impl Attention {
    fn from_weights(w: &Weights, prefix: &str, num_heads: i32, head_dim: i32) -> Result<Self> {
        let l = |n: &str| crate::load_linear(w, &join(prefix, n));
        Ok(Self {
            q: l("q_proj")?,
            k: l("k_proj")?,
            v: l("v_proj")?,
            o: l("o_proj")?,
            num_heads,
            head_dim,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        crate::quantize_linear(&mut self.q, bits)?;
        crate::quantize_linear(&mut self.k, bits)?;
        crate::quantize_linear(&mut self.v, bits)?;
        crate::quantize_linear(&mut self.o, bits)?;
        Ok(())
    }

    fn forward(&self, x: &Array, rope: &RopeTable) -> Result<Array> {
        let sh = x.shape();
        let (b, h, w) = (sh[0], sh[1], sh[2]);
        let (nh, hd) = (self.num_heads, self.head_dim);
        let seq = h * w;
        // [b,H,W,C] → [b, seq, nh, hd] → [b, nh, seq, hd]
        let to_heads = |t: Array| -> Result<Array> {
            Ok(t.reshape(&[b, seq, nh, hd])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = apply_rope(&to_heads(self.q.forward(x)?)?, rope)?;
        let k = apply_rope(&to_heads(self.k.forward(x)?)?, rope)?;
        let v = to_heads(self.v.forward(x)?)?;

        let scale = 1.0 / (hd as f32).sqrt();
        let attn = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let out = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, h, w, nh * hd])?;
        self.o.forward(&out)
    }
}

/// One pre-norm ViT layer: (windowed) RoPE attention + GELU MLP.
#[derive(Clone)]
struct ViTLayer {
    norm1_w: Array,
    norm1_b: Array,
    norm2_w: Array,
    norm2_b: Array,
    attn: Attention,
    mlp: Mlp,
    rope: RopeTable,
    /// 0 ⇒ global attention over the full grid; else windowed with this side.
    window: i32,
    eps: f32,
}

impl ViTLayer {
    fn from_weights(
        w: &Weights,
        prefix: &str,
        cfg: &Sam3VisionConfig,
        global: bool,
    ) -> Result<Self> {
        let hd = cfg.head_dim();
        // Rotary table: windowed layers use a window grid at scale 1; global layers use the full
        // token grid scaled by `window_size / grid` (so positions span the same rotary range).
        let (end, scale) = if global {
            (cfg.grid(), cfg.window_size as f32 / cfg.grid() as f32)
        } else {
            (cfg.window_size, 1.0)
        };
        Ok(Self {
            norm1_w: w.require(&join(prefix, "layer_norm1.weight"))?.clone(),
            norm1_b: w.require(&join(prefix, "layer_norm1.bias"))?.clone(),
            norm2_w: w.require(&join(prefix, "layer_norm2.weight"))?.clone(),
            norm2_b: w.require(&join(prefix, "layer_norm2.bias"))?.clone(),
            attn: Attention::from_weights(
                w,
                &join(prefix, "attention"),
                cfg.num_attention_heads,
                hd,
            )?,
            mlp: Mlp::from_weights(w, &join(prefix, "mlp"))?,
            rope: RopeTable::new(end, scale, cfg.rope_theta, hd),
            window: if global { 0 } else { cfg.window_size },
            eps: cfg.layer_norm_eps,
        })
    }

    fn quantize(&mut self, bits: i32) -> Result<()> {
        self.attn.quantize(bits)?;
        self.mlp.quantize(bits)?;
        Ok(())
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let normed = layer_norm(x, Some(&self.norm1_w), Some(&self.norm1_b), self.eps)?;
        let attended = if self.window > 0 {
            let (h, w) = (normed.shape()[1], normed.shape()[2]);
            let (windows, pad_hw) = window_partition(&normed, self.window)?;
            let a = self.attn.forward(&windows, &self.rope)?;
            window_unpartition(&a, self.window, pad_hw, (h, w))?
        } else {
            self.attn.forward(&normed, &self.rope)?
        };
        let x = add(x, &attended)?;
        let mlp_in = layer_norm(&x, Some(&self.norm2_w), Some(&self.norm2_b), self.eps)?;
        Ok(add(&x, &self.mlp.forward(&mlp_in)?)?)
    }
}

/// PE ViT backbone: patch-embed → tiled position embedding → front LayerNorm → layers.
///
/// Cheap to `Clone` — every field is an `Array`/`AdaptableLinear` reference-counted handle (cloning
/// duplicates the handle, not the GPU buffer). The video model relies on this to quantize one shared
/// backbone and reinstall the same `Rc` into both consumers without copying weights (F-028).
#[derive(Clone)]
pub(crate) struct Backbone {
    patch_w: Array, // OHWI, no bias
    pos_embed: Array,
    front_norm_w: Array,
    front_norm_b: Array,
    layers: Vec<ViTLayer>,
    patch_size: i32,
    grid: i32,
    pretrain_grid: i32,
    eps: f32,
}

impl Backbone {
    pub(crate) fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3VisionConfig) -> Result<Self> {
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                let global = cfg.global_attn_indexes.contains(&i);
                ViTLayer::from_weights(w, &join(prefix, &format!("layers.{i}")), cfg, global)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_w: conv_w_ohwi(w.require(&join(
                prefix,
                "embeddings.patch_embeddings.projection.weight",
            ))?)?,
            pos_embed: w
                .require(&join(prefix, "embeddings.position_embeddings"))?
                .clone(),
            front_norm_w: w.require(&join(prefix, "layer_norm.weight"))?.clone(),
            front_norm_b: w.require(&join(prefix, "layer_norm.bias"))?.clone(),
            layers,
            patch_size: cfg.patch_size,
            grid: cfg.grid(),
            pretrain_grid: cfg.pretrain_grid(),
            eps: cfg.layer_norm_eps,
        })
    }

    pub(crate) fn quantize(&mut self, bits: i32) -> Result<()> {
        for layer in &mut self.layers {
            layer.quantize(bits)?;
        }
        Ok(())
    }

    /// Tile the `[1, pg², C]` position embedding to `[1, grid, grid, C]`. The shipped config has
    /// `grid` an exact multiple of `pg` (72 = 3·24), so this is exact concatenation (no crop).
    fn tiled_pos(&self) -> Result<Array> {
        let pg = self.pretrain_grid;
        let c = self.pos_embed.shape()[2];
        if self.grid % pg != 0 {
            return Err(Error::Msg(format!(
                "sam3 vision: token grid {} is not a multiple of the position-embedding grid {} \
                 (non-exact tiling not implemented)",
                self.grid, pg
            )));
        }
        let reps = (self.grid / pg) as usize;
        let p = self.pos_embed.reshape(&[1, pg, pg, c])?;
        let row: Vec<&Array> = std::iter::repeat_n(&p, reps).collect();
        let p_h = concatenate_axis(&row, 1)?; // [1, grid, pg, C]
        let row2: Vec<&Array> = std::iter::repeat_n(&p_h, reps).collect();
        Ok(concatenate_axis(&row2, 2)?) // [1, grid, grid, C]
    }

    /// `pixel_values`: NCHW `[1, 3, 1008, 1008]`. Returns the backbone feature map NHWC
    /// `[1, grid, grid, C]`.
    pub(crate) fn forward(&self, pixel_values: &Array) -> Result<Array> {
        let x = pixel_values.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let x = conv2d(&x, &self.patch_w, None, self.patch_size, 0)?; // [1,grid,grid,C]
        let mut x = add(&x, &self.tiled_pos()?)?;
        x = layer_norm(
            &x,
            Some(&self.front_norm_w),
            Some(&self.front_norm_b),
            self.eps,
        )?;
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

/// Quantize a (possibly shared) PE [`Backbone`] in place. If this is the sole owner of the `Rc`
/// (the standalone segmenter / tracker), quantize the backbone directly; otherwise clone its cheap
/// `Array` handles, quantize the clone, and swap in a fresh `Rc` (the caller is responsible for
/// re-sharing if both consumers must point at the same quantized copy — see
/// [`crate::video::Sam3VideoModel::quantize`]).
pub(crate) fn quantize_backbone_rc(backbone: &mut Rc<Backbone>, bits: i32) -> Result<()> {
    match Rc::get_mut(backbone) {
        Some(bb) => bb.quantize(bits),
        None => {
            let mut bb = (**backbone).clone();
            bb.quantize(bits)?;
            *backbone = Rc::new(bb);
            Ok(())
        }
    }
}

/// One FPN branch (`Sam3FPNLayer`): scale the backbone map, then `proj1` (1×1) → `proj2` (3×3).
pub(crate) struct FpnLayer {
    /// Transposed-conv up-scale stages (OHWI weight + bias), applied in order with exact-gelu
    /// between consecutive stages (matches `nn.GELU()` in the scale_factor==4 branch).
    up_stages: Vec<(Array, Array)>,
    /// True for scale_factor 0.5: a 2×2 max-pool downsample instead of transposed convs.
    downsample: bool,
    proj1_w: Array,
    proj1_b: Array,
    proj2_w: Array,
    proj2_b: Array,
}

impl FpnLayer {
    pub(crate) fn from_weights(w: &Weights, prefix: &str, scale: f32) -> Result<Self> {
        // Branch on an integer code (`scale·2` → 8/4/2/1) to avoid float-literal matching.
        // scale_layers indices: ConvTranspose at 0 (and 2 for scale 4), GELU at 1 (no weights),
        // MaxPool at 0 for scale 0.5 (no weights).
        let code = (scale * 2.0).round() as i32;
        let up_indices: &[i32] = match code {
            8 => &[0, 2], // scale 4.0: two transposed convs (72→144→288)
            4 => &[0],    // scale 2.0: one transposed conv (72→144)
            _ => &[],     // scale 1.0 / 0.5: no transposed conv
        };
        let up_stages = up_indices
            .iter()
            .map(|&i| -> Result<(Array, Array)> {
                Ok((
                    conv_transpose_w(
                        w.require(&join(prefix, &format!("scale_layers.{i}.weight")))?,
                    )?,
                    w.require(&join(prefix, &format!("scale_layers.{i}.bias")))?
                        .clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            up_stages,
            downsample: code == 1, // scale 0.5
            proj1_w: conv_w_ohwi(w.require(&join(prefix, "proj1.weight"))?)?,
            proj1_b: w.require(&join(prefix, "proj1.bias"))?.clone(),
            proj2_w: conv_w_ohwi(w.require(&join(prefix, "proj2.weight"))?)?,
            proj2_b: w.require(&join(prefix, "proj2.bias"))?.clone(),
        })
    }

    /// `x`: NHWC `[1, 72, 72, 1024]`. Returns NHWC `[1, Hs, Ws, fpn_dim]`.
    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for (i, (w, b)) in self.up_stages.iter().enumerate() {
            let y = conv_transpose2d(&h, w, (2, 2), None, None, None, None)?;
            h = add(&y, b)?;
            if i + 1 < self.up_stages.len() {
                h = gelu_exact(&h)?; // GELU only *between* the two transposed convs (scale 4)
            }
        }
        if self.downsample {
            let mut pool = MaxPool2d::new(2, 2);
            h = pool.forward(&h)?;
        }
        let h = conv2d(&h, &self.proj1_w, Some(&self.proj1_b), 1, 0)?; // 1×1
        conv2d(&h, &self.proj2_w, Some(&self.proj2_b), 1, 1) // 3×3 pad 1
    }
}

/// SAM3 vision encoder: PE backbone + FPN neck. Produces the multi-scale FPN feature maps the
/// detector + tracker share.
pub struct Sam3VisionEncoder {
    backbone: Rc<Backbone>,
    fpn_layers: Vec<FpnLayer>,
}

impl Sam3VisionEncoder {
    /// Load from a `facebook/sam3` weight map. `prefix` is typically
    /// `"detector_model.vision_encoder"`.
    pub fn from_weights(w: &Weights, prefix: &str, cfg: &Sam3VisionConfig) -> Result<Self> {
        let backbone = Rc::new(Backbone::from_weights(w, &join(prefix, "backbone"), cfg)?);
        Self::from_weights_with_backbone(w, prefix, cfg, backbone)
    }

    /// Load the FPN neck only, reusing an already-loaded (and possibly shared) PE [`Backbone`]. The
    /// video model uses this so the segmenter and the tracker share **one** backbone instead of each
    /// loading its own copy (F-028).
    pub(crate) fn from_weights_with_backbone(
        w: &Weights,
        prefix: &str,
        cfg: &Sam3VisionConfig,
        backbone: Rc<Backbone>,
    ) -> Result<Self> {
        let fpn_layers = cfg
            .scale_factors
            .iter()
            .enumerate()
            .map(|(i, &scale)| {
                FpnLayer::from_weights(w, &join(prefix, &format!("neck.fpn_layers.{i}")), scale)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            backbone,
            fpn_layers,
        })
    }

    /// The shared PE [`Backbone`] handle (clone of the `Rc`). Used by the F-028 sharing test to
    /// assert pointer-identity with the tracker's backbone.
    #[cfg(test)]
    pub(crate) fn backbone_rc(&self) -> Rc<Backbone> {
        self.backbone.clone()
    }

    /// Replace the PE backbone with a (typically pre-quantized, shared) one. Used by the video model
    /// after quantizing the single shared backbone.
    pub(crate) fn set_backbone(&mut self, backbone: Rc<Backbone>) {
        self.backbone = backbone;
    }

    /// Quantize the ViT backbone's attention + MLP projections (Q8/Q4). The patch-embed conv, the
    /// position embedding, and the conv-only FPN neck stay dense (sc-4925).
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        quantize_backbone_rc(&mut self.backbone, bits)
    }

    /// `pixel_values`: NCHW `[1, 3, 1008, 1008]`. Returns the FPN feature maps as **NHWC**
    /// `[1, Hs, Ws, fpn_dim]`, fine→coarse (288²/144²/72²/36²), one per `scale_factors` entry.
    pub fn forward(&self, pixel_values: &Array) -> Result<Vec<Array>> {
        let features = self.backbone_features(pixel_values)?;
        self.fpn_from_backbone(&features)
    }

    /// Run **only** the PE ViT backbone (the half shared by the detector neck and the tracker neck),
    /// returning the NHWC `[1, grid, grid, C]` feature map. The video tracker runs this once per
    /// frame and feeds both necks, avoiding a second backbone pass (sc-4924).
    pub fn backbone_features(&self, pixel_values: &Array) -> Result<Array> {
        self.backbone.forward(pixel_values)
    }

    /// Run the detector FPN neck over already-computed backbone features (see
    /// [`Self::backbone_features`]). Returns the FPN maps NHWC, fine→coarse.
    pub fn fpn_from_backbone(&self, features: &Array) -> Result<Vec<Array>> {
        self.fpn_layers
            .iter()
            .map(|l| l.forward(features))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::{abs, array_eq, max};

    /// `window_unpartition ∘ window_partition` is identity over the unpadded region (exercises the
    /// pad-then-crop path with H/W not divisible by the window).
    #[test]
    fn window_partition_round_trips() {
        let (h, w, c) = (12, 10, 4);
        let vals: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        let x = Array::from_slice(&vals, &[1, h, w, c]);
        let (windows, pad_hw) = window_partition(&x, 8).unwrap();
        assert_eq!(pad_hw, (16, 16));
        let back = window_unpartition(&windows, 8, pad_hw, (h, w)).unwrap();
        assert_eq!(back.shape(), &[1, h, w, c]);
        assert!(array_eq(&back, &x, None).unwrap().item::<bool>());
    }

    /// `rotate_pairwise` maps lanes `(a, b) -> (-b, a)`; applied twice it negates (`x -> -x`).
    #[test]
    fn rotate_pairwise_squares_to_negation() {
        let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4]);
        let twice = rotate_pairwise(&rotate_pairwise(&x).unwrap()).unwrap();
        let neg = negative(&x).unwrap();
        assert!(array_eq(&twice, &neg, None).unwrap().item::<bool>());
    }

    /// The RoPE table has the expected `[end², head_dim]` shape and unit-magnitude cos/sin pairs
    /// (row 0 is all-zero angle → cos 1, sin 0).
    #[test]
    fn rope_table_shape_and_origin() {
        let t = RopeTable::new(24, 1.0, 10000.0, 64);
        assert_eq!(t.cos.shape(), &[576, 64]);
        assert_eq!(t.sin.shape(), &[576, 64]);
        // position 0 → angle 0 → cos 1, sin 0 across the head dim.
        let cos0 = t.cos.take_axis(Array::from_int(0), 0).unwrap();
        let sin0 = abs(t.sin.take_axis(Array::from_int(0), 0).unwrap()).unwrap();
        assert!((max(&cos0, None).unwrap().item::<f32>() - 1.0).abs() < 1e-6);
        assert!(max(&sin0, None).unwrap().item::<f32>() < 1e-6);
    }
}
