//! BiSeNet face parsing — native MLX port of facexlib's `parsing_bisenet.pth` (sc-3084).
//!
//! PuLID-FLUX whitens non-face regions before the EVA-CLIP crop: `parse(x)[0]` → 19-class argmax
//! mask → background labels `[0,16,18,7,8,9,14,15]` become white, the rest grayscale
//! (`pipeline_flux.py:167-177`). This replaces facexlib's torch BiSeNet — the last torch holdout.
//!
//! ## Architecture (facexlib `parsing/{bisenet,resnet}.py`, num_class = 19)
//! - **ResNet18 backbone**: `conv1(7×7,s2)` → maxpool(3×3,s2,p1) → 4 stages of 2 `BasicBlock`s
//!   (stages 2/3/4 block-0 stride-2 + 1×1 downsample) → `feat8`/`feat16`/`feat32` (1/8, 1/16, 1/32).
//! - **ContextPath**: `ARM(256→128)` + `ARM(512→128)` (ConvBNReLU → 1×1 conv_atten → sigmoid → mul),
//!   a global-avg `conv_avg(512→128)`, nearest upsamples, and `conv_head16/32`.
//! - **FeatureFusionModule(256→256)**: channel-concat `feat_res8`+`feat_cp8`, ConvBNReLU, then an
//!   SE-style attention (`conv1→relu→conv2→sigmoid`, `feat*atten + feat`).
//! - **conv_out head**: ConvBNReLU(256→256) → 1×1 → 19 logits @ 64², then **bilinear upsample to
//!   512² with `align_corners=True`** (the flagged-risk op, hand-rolled exactly) → argmax.
//!   (`conv_out16`/`conv_out32` are training-only aux heads; PuLID uses `[0]`, so they're omitted.)
//!
//! Every conv is bias-less + BN; BNs are folded into the convs at conversion (see
//! `tools/convert_bisenet.py`). The mask is a coarse argmax → tolerant; parity is mask IoU ≈ 1.0.
//! Input: NHWC `[B,512,512,3]`, `(rgb/255 - mean) / std` with ImageNet mean/std. fp32, OHWI weights.

use mlx_gen::array::scalar;
use mlx_gen::nn::{conv2d, upsample_nearest};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::ops::indexing::{argmax_axis, IndexOp, IntoStrideBy};
use mlx_rs::ops::{add, concatenate_axis, matmul, maximum, mean_axes, multiply, pad, sigmoid};
use mlx_rs::Array;

/// PuLID background labels → whitened in the `face_features_image`.
pub const BG_LABELS: [u32; 8] = [0, 16, 18, 7, 8, 9, 14, 15];
/// ImageNet normalization (the facexlib parse-net preprocessing).
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn relu(x: &Array) -> Result<Array> {
    Ok(maximum(x, scalar(0.0))?)
}

/// Global average pool over the NHWC spatial axes → `[B,1,1,C]`.
fn global_avg(x: &Array) -> Result<Array> {
    Ok(mean_axes(x, &[1, 2][..], true)?)
}

/// `3×3` stride-2 pad-1 max pool over NHWC (overlapping windows; input is ReLU'd ≥0 so 0-pad ≡
/// PyTorch's -inf pad — every output window contains ≥1 real pixel).
fn maxpool_3x3_s2(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let (h, w) = (sh[1], sh[2]);
    let (oh, ow) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
    let p = pad(x, &[(0, 0), (1, 1), (1, 1), (0, 0)][..], None, None)?; // [B,H+2,W+2,C]
    let mut acc: Option<Array> = None;
    for dy in 0..3 {
        for dx in 0..3 {
            let win = p.index((
                ..,
                (dy..dy + 2 * oh).stride_by(2),
                (dx..dx + 2 * ow).stride_by(2),
                ..,
            ));
            acc = Some(match acc {
                None => win,
                Some(a) => maximum(&a, &win)?,
            });
        }
    }
    Ok(acc.unwrap())
}

/// Per-axis `align_corners=True` linear interpolation matrix `[out_n, in_n]`: row `i` maps output
/// pixel `i` to source `i·(in-1)/(out-1)` with bilinear weights `(1-f, f)` on `(floor, floor+1)`.
fn interp_matrix(in_n: i32, out_n: i32) -> Array {
    let mut m = vec![0f32; (out_n * in_n) as usize];
    let denom = (out_n - 1).max(1) as f32;
    for i in 0..out_n {
        let src = i as f32 * (in_n - 1) as f32 / denom;
        let x0 = src.floor() as i32;
        let x1 = (x0 + 1).min(in_n - 1);
        let f = src - x0 as f32;
        m[(i * in_n + x0) as usize] += 1.0 - f;
        m[(i * in_n + x1) as usize] += f; // += so x0==x1 (last row) sums to 1
    }
    Array::from_slice(&m, &[out_n, in_n])
}

/// Bilinear upsample of NHWC `x` to `out_h × out_w` with `align_corners=True`, applied as two
/// separable matmuls (`Wy · x` over rows, `Wx · x` over cols). Bit-faithful to `F.interpolate`.
fn upsample_bilinear_ac(x: &Array, out_h: i32, out_w: i32) -> Result<Array> {
    let sh = x.shape();
    let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);
    let wy = interp_matrix(h, out_h); // [out_h, h]
    let wx = interp_matrix(w, out_w); // [out_w, w]
                                      // rows: [out_h,h] · [b,h,w*c] -> [b,out_h,w*c]
    let mid = matmul(&wy, &x.reshape(&[b, h, w * c])?)?;
    let mid = mid.reshape(&[b * out_h, w, c])?;
    // cols: [out_w,w] · [b*out_h,w,c] -> [b*out_h,out_w,c]
    let out = matmul(&wx, &mid)?;
    Ok(out.reshape(&[b, out_h, out_w, c])?)
}

/// A biased convolution (BN folded in at conversion).
struct Conv {
    w: Array,
    b: Array,
}
impl Conv {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{p}.weight"))?.clone(),
            b: w.require(&format!("{p}.bias"))?.clone(),
        })
    }
    fn forward(&self, x: &Array, stride: i32, padding: i32) -> Result<Array> {
        conv2d(x, &self.w, Some(&self.b), stride, padding)
    }
    /// ConvBNReLU = biased conv → ReLU.
    fn forward_relu(&self, x: &Array, stride: i32, padding: i32) -> Result<Array> {
        relu(&self.forward(x, stride, padding)?)
    }
}

/// A bias-less convolution (the FFM SE 1×1s and the final 1×1 head — no BN).
struct ConvW {
    w: Array,
}
impl ConvW {
    fn load(w: &Weights, p: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{p}.weight"))?.clone(),
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        conv2d(x, &self.w, None, 1, 0)
    }
}

/// ResNet18 `BasicBlock`: `conv1(stride)→relu→conv2 (+ downsample) → relu`.
struct BasicBlock {
    conv1: Conv,
    conv2: Conv,
    downsample: Option<Conv>,
    stride: i32,
}
impl BasicBlock {
    fn forward(&self, x: &Array) -> Result<Array> {
        let r = self.conv1.forward_relu(x, self.stride, 1)?;
        let r = self.conv2.forward(&r, 1, 1)?;
        let shortcut = match &self.downsample {
            Some(ds) => ds.forward(x, self.stride, 0)?,
            None => x.clone(),
        };
        relu(&add(&r, &shortcut)?)
    }
}

/// AttentionRefinementModule: `ConvBNReLU → global-avg → 1×1 conv_atten → sigmoid → mul`.
struct Arm {
    conv: Conv,
    conv_atten: Conv, // bn_atten folded in
}
impl Arm {
    fn forward(&self, x: &Array) -> Result<Array> {
        let feat = self.conv.forward_relu(x, 1, 1)?;
        let atten = self.conv_atten.forward(&global_avg(&feat)?, 1, 0)?;
        let atten = sigmoid(&atten)?;
        Ok(multiply(&feat, &atten)?)
    }
}

/// BiSeNet (19-class face parsing).
pub struct BiSeNet {
    // ResNet18 backbone
    conv1: Conv,
    layers: Vec<Vec<BasicBlock>>,
    // ContextPath
    arm16: Arm,
    arm32: Arm,
    conv_head32: Conv,
    conv_head16: Conv,
    conv_avg: Conv,
    // FeatureFusionModule
    ffm_convblk: Conv,
    ffm_conv1: ConvW,
    ffm_conv2: ConvW,
    // output head
    conv_out_conv: Conv,
    conv_out_out: ConvW,
}

const STAGES: [(i32, i32); 4] = [(64, 1), (128, 2), (256, 2), (512, 2)]; // (out_chan, block0 stride)

impl BiSeNet {
    /// Load from the converted `bisenet_parsing.safetensors` (see `tools/convert_bisenet.py`).
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let mut layers = Vec::with_capacity(4);
        for (li, &(_oc, stride)) in STAGES.iter().enumerate() {
            let l = li + 1;
            let mut blocks = Vec::with_capacity(2);
            for b in 0..2 {
                let p = format!("resnet.layer{l}.{b}");
                let (s, ds) = if b == 0 {
                    (
                        stride,
                        if l > 1 {
                            Some(Conv::load(w, &format!("{p}.downsample"))?)
                        } else {
                            None
                        },
                    )
                } else {
                    (1, None)
                };
                blocks.push(BasicBlock {
                    conv1: Conv::load(w, &format!("{p}.conv1"))?,
                    conv2: Conv::load(w, &format!("{p}.conv2"))?,
                    downsample: ds,
                    stride: s,
                });
            }
            layers.push(blocks);
        }
        Ok(Self {
            conv1: Conv::load(w, "resnet.conv1")?,
            layers,
            arm16: Arm {
                conv: Conv::load(w, "arm16.conv")?,
                conv_atten: Conv::load(w, "arm16.conv_atten")?,
            },
            arm32: Arm {
                conv: Conv::load(w, "arm32.conv")?,
                conv_atten: Conv::load(w, "arm32.conv_atten")?,
            },
            conv_head32: Conv::load(w, "conv_head32")?,
            conv_head16: Conv::load(w, "conv_head16")?,
            conv_avg: Conv::load(w, "conv_avg")?,
            ffm_convblk: Conv::load(w, "ffm.convblk")?,
            ffm_conv1: ConvW::load(w, "ffm.conv1")?,
            ffm_conv2: ConvW::load(w, "ffm.conv2")?,
            conv_out_conv: Conv::load(w, "conv_out.conv")?,
            conv_out_out: ConvW::load(w, "conv_out.conv_out")?,
        })
    }

    /// ResNet18 backbone → `(feat8, feat16, feat32)`.
    fn resnet(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let x = self.conv1.forward_relu(x, 2, 3)?;
        let x = maxpool_3x3_s2(&x)?;
        let x = self.run_layer(0, &x)?; // layer1
        let feat8 = self.run_layer(1, &x)?; // layer2 (1/8)
        let feat16 = self.run_layer(2, &feat8)?; // layer3 (1/16)
        let feat32 = self.run_layer(3, &feat16)?; // layer4 (1/32)
        Ok((feat8, feat16, feat32))
    }

    fn run_layer(&self, idx: usize, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for blk in &self.layers[idx] {
            h = blk.forward(&h)?;
        }
        Ok(h)
    }

    /// ContextPath → `(feat_res8, feat_cp8, feat_cp16)`.
    fn context_path(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let (feat8, feat16, feat32) = self.resnet(x)?;
        let s32 = feat32.shape()[1];
        let s16 = feat16.shape()[1];
        let s8 = feat8.shape()[1];

        let avg = self.conv_avg.forward_relu(&global_avg(&feat32)?, 1, 0)?; // [B,1,1,128]
        let avg_up = upsample_nearest(&avg, s32)?; // → feat32 spatial

        let feat32_sum = add(&self.arm32.forward(&feat32)?, &avg_up)?;
        let feat32_up = upsample_nearest(&feat32_sum, s16 / s32)?;
        let feat_cp16 = self.conv_head32.forward_relu(&feat32_up, 1, 1)?;

        let feat16_sum = add(&self.arm16.forward(&feat16)?, &feat_cp16)?;
        let feat16_up = upsample_nearest(&feat16_sum, s8 / s16)?;
        let feat_cp8 = self.conv_head16.forward_relu(&feat16_up, 1, 1)?;

        Ok((feat8, feat_cp8, feat_cp16))
    }

    /// FeatureFusionModule(fsp, fcp).
    fn ffm(&self, fsp: &Array, fcp: &Array) -> Result<Array> {
        let fcat = concatenate_axis(&[fsp, fcp], 3)?;
        let feat = self.ffm_convblk.forward_relu(&fcat, 1, 0)?;
        let atten = relu(&self.ffm_conv1.forward(&global_avg(&feat)?)?)?;
        let atten = sigmoid(&self.ffm_conv2.forward(&atten)?)?;
        let feat_atten = multiply(&feat, &atten)?;
        Ok(add(&feat_atten, &feat)?)
    }

    /// Parse → 19-class logits, NHWC `[B, H, W, 19]` (bilinear-upsampled to the input size).
    pub fn parse_logits(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (h, w) = (sh[1], sh[2]);
        let (feat_res8, feat_cp8, _feat_cp16) = self.context_path(x)?;
        let feat_fuse = self.ffm(&feat_res8, &feat_cp8)?;
        let out = self.conv_out_conv.forward_relu(&feat_fuse, 1, 1)?;
        let out = self.conv_out_out.forward(&out)?; // [B,64,64,19]
        upsample_bilinear_ac(&out, h, w)
    }

    /// Parse → argmax class mask, `[B, H, W]` (`uint32`).
    pub fn parse_mask(&self, x: &Array) -> Result<Array> {
        Ok(argmax_axis(&self.parse_logits(x)?, 3, false)?)
    }
}

/// Normalize an RGB `[B,H,W,3]` `[0,1]` image to the BiSeNet parse-net input (ImageNet mean/std).
pub fn to_parse_input(rgb01: &Array) -> Result<Array> {
    let mean = Array::from_slice(&MEAN, &[1, 1, 1, 3]);
    let inv_std = Array::from_slice(&[1.0 / STD[0], 1.0 / STD[1], 1.0 / STD[2]], &[1, 1, 1, 3]);
    Ok(multiply(&mlx_rs::ops::subtract(rgb01, &mean)?, &inv_std)?)
}

/// PuLID `face_features_image`: `where(mask ∈ BG_LABELS, white(1.0), gray(rgb01))`, NHWC `[B,H,W,3]`.
/// `rgb01` is the un-normalized `[0,1]` aligned crop; `mask` is [`parse_mask`] output.
pub fn face_features_image(rgb01: &Array, mask: &Array) -> Result<Array> {
    let sh = rgb01.shape();
    let (b, h, w) = (sh[0] as usize, sh[1] as usize, sh[2] as usize);
    let rgb = rgb01
        .try_as_slice::<f32>()
        .map_err(|e| format!("face_features_image rgb readback: {e}"))?;
    let m = mask
        .try_as_slice::<u32>()
        .map_err(|e| format!("face_features_image mask readback: {e}"))?;
    let mut out = vec![0f32; b * h * w * 3];
    for i in 0..b * h * w {
        let v = if BG_LABELS.contains(&m[i]) {
            1.0
        } else {
            0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
        };
        out[i * 3] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
    Ok(Array::from_slice(&out, &[b as i32, h as i32, w as i32, 3]))
}
