//! SeedVR2 3D causal video VAE — native MLX port (sc-4813).
//!
//! Port of `mflux.models.seedvr2.model.seedvr2_vae`. The encoder maps `(B,3,T,H,W)` →
//! `(B,16,T',H',W')` (spatial /8; temporal `ceil(T/4)` via two temporal-stride down blocks); the
//! decoder inverts it. Conv weights are MLX-native `[out,kT,kH,kW,in]` (the dump/converter has
//! already transposed them from the torch `[out,in,kT,kH,kW]`).
//!
//! Parity-critical details vs the reference (`common/conv3d.py`, the *_3d blocks):
//!   * **Causal temporal padding repeats the first frame** `causal_pad` times (NOT zero-pad), where
//!     `causal_pad = 2·pad_t` when `use_padding_causal` else `kt-1`; temporal conv padding is then 0.
//!   * Spatial padding is symmetric via `conv3d`'s padding arg; the down/up samplers add their own
//!     asymmetric `(0,1)` H/W pad before a stride-2 / 1×1 conv.
//!   * GroupNorm (32 groups, eps 1e-6) runs in f32 (channels-last), pytorch-compatible.

use mlx_gen::nn::{conv3d, group_norm, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::{add, concatenate_axis, matmul, multiply, pad, split};
use mlx_rs::{Array, Dtype};

use crate::config::VaeConfig;

/// `[out,in]`-weight dense layer (`y = x·Wᵀ + b`).
struct Linear {
    w: Array,
    b: Array,
}
impl Linear {
    fn load(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        Ok(add(&matmul(x, self.w.t())?, &self.b)?)
    }
}

/// First temporal frame, keeping the axis: `x[:, :, :1]`.
fn first_t(x: &Array) -> Result<Array> {
    Ok(x.take_axis(Array::from_slice(&[0i32], &[1]), 2)?)
}

/// GroupNorm over an NCTHW tensor: transpose to channels-last, normalise in f32, transpose back.
fn gn(x_ncthw: &Array, w: &Array, b: &Array, groups: i32, eps: f32) -> Result<Array> {
    let dt = x_ncthw.dtype();
    let xl = x_ncthw
        .transpose_axes(&[0, 2, 3, 4, 1])?
        .as_dtype(Dtype::Float32)?;
    let g = group_norm(
        &xl,
        &w.as_dtype(Dtype::Float32)?,
        &b.as_dtype(Dtype::Float32)?,
        groups,
        eps,
    )?;
    Ok(g.as_dtype(dt)?.transpose_axes(&[0, 4, 1, 2, 3])?)
}

/// Causal 3-D convolution. NCTHW in/out. Generalises every conv in the VAE via `(stride, padding,
/// use_padding_causal)`; the kernel sizes come from the loaded weight `[out,kT,kH,kW,in]`.
struct CausalConv3d {
    w: Array,
    b: Array,
    st: i32,
    sh: i32,
    sw: i32,
    pt: i32,
    ph: i32,
    pw: i32,
    use_padding_causal: bool,
}
impl CausalConv3d {
    fn load(
        w: &Weights,
        prefix: &str,
        stride: (i32, i32, i32),
        padding: (i32, i32, i32),
        use_padding_causal: bool,
    ) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.clone(),
            b: w.require(&format!("{prefix}.bias"))?.clone(),
            st: stride.0,
            sh: stride.1,
            sw: stride.2,
            pt: padding.0,
            ph: padding.1,
            pw: padding.2,
            use_padding_causal,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let kt = self.w.shape()[1];
        let (x, temporal_padding) = if kt > 1 {
            let causal_pad = if self.use_padding_causal {
                2 * self.pt
            } else {
                kt - 1
            };
            if causal_pad > 0 {
                let first = first_t(x)?;
                let mut parts: Vec<Array> = (0..causal_pad).map(|_| first.clone()).collect();
                parts.push(x.clone());
                let refs: Vec<&Array> = parts.iter().collect();
                (concatenate_axis(&refs, 2)?, 0)
            } else {
                (x.clone(), 0)
            }
        } else {
            (x.clone(), self.pt)
        };
        let xl = x.transpose_axes(&[0, 2, 3, 4, 1])?; // NDHWC
        let y = conv3d(
            &xl,
            &self.w,
            Some(&self.b),
            (self.st, self.sh, self.sw),
            (temporal_padding, self.ph, self.pw),
        )?;
        Ok(y.transpose_axes(&[0, 4, 1, 2, 3])?) // NCTHW
    }
}

/// `norm1 → silu → conv1 → norm2 → silu → conv2` + (1³ conv) skip when channels differ.
struct ResnetBlock3d {
    norm1_w: Array,
    norm1_b: Array,
    conv1: CausalConv3d,
    norm2_w: Array,
    norm2_b: Array,
    conv2: CausalConv3d,
    shortcut: Option<CausalConv3d>,
    groups: i32,
    eps: f32,
}
impl ResnetBlock3d {
    fn load(w: &Weights, prefix: &str, cfg: &VaeConfig) -> Result<Self> {
        let shortcut = if w.get(&format!("{prefix}.conv_shortcut.weight")).is_some() {
            Some(CausalConv3d::load(
                w,
                &format!("{prefix}.conv_shortcut"),
                (1, 1, 1),
                (0, 0, 0),
                false,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1_w: w.require(&format!("{prefix}.norm1.weight"))?.clone(),
            norm1_b: w.require(&format!("{prefix}.norm1.bias"))?.clone(),
            conv1: CausalConv3d::load(w, &format!("{prefix}.conv1"), (1, 1, 1), (1, 1, 1), false)?,
            norm2_w: w.require(&format!("{prefix}.norm2.weight"))?.clone(),
            norm2_b: w.require(&format!("{prefix}.norm2.bias"))?.clone(),
            conv2: CausalConv3d::load(w, &format!("{prefix}.conv2"), (1, 1, 1), (1, 1, 1), false)?,
            shortcut,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let residual = match &self.shortcut {
            Some(s) => s.forward(x)?,
            None => x.clone(),
        };
        let h = gn(x, &self.norm1_w, &self.norm1_b, self.groups, self.eps)?;
        let h = self.conv1.forward(&silu(&h)?)?;
        let h = gn(&h, &self.norm2_w, &self.norm2_b, self.groups, self.eps)?;
        let h = self.conv2.forward(&silu(&h)?)?;
        Ok(add(&h, &residual)?)
    }
}

/// Per-frame single-head spatial self-attention (head_dim = C). NCTHW I/O.
struct Attention3d {
    gn_w: Array,
    gn_b: Array,
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    groups: i32,
    eps: f32,
}
impl Attention3d {
    fn load(w: &Weights, prefix: &str, cfg: &VaeConfig) -> Result<Self> {
        Ok(Self {
            gn_w: w.require(&format!("{prefix}.group_norm.weight"))?.clone(),
            gn_b: w.require(&format!("{prefix}.group_norm.bias"))?.clone(),
            to_q: Linear::load(w, &format!("{prefix}.to_q"))?,
            to_k: Linear::load(w, &format!("{prefix}.to_k"))?,
            to_v: Linear::load(w, &format!("{prefix}.to_v"))?,
            to_out: Linear::load(w, &format!("{prefix}.to_out.0"))?,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, c, t, h, wd) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let residual = x.clone();
        // (B,C,T,H,W) -> (B,T,C,H,W) -> (B*T, C, H*W) -> (B*T, H*W, C)
        let xs = x
            .transpose_axes(&[0, 2, 1, 3, 4])?
            .reshape(&[b * t, c, h * wd])?
            .transpose_axes(&[0, 2, 1])?;
        // GroupNorm (channels-last, f32)
        let dt = xs.dtype();
        let xn = group_norm(
            &xs.as_dtype(Dtype::Float32)?,
            &self.gn_w.as_dtype(Dtype::Float32)?,
            &self.gn_b.as_dtype(Dtype::Float32)?,
            self.groups,
            self.eps,
        )?
        .as_dtype(dt)?;
        let q = self.to_q.forward(&xn)?.reshape(&[b * t, 1, h * wd, c])?;
        let k = self.to_k.forward(&xn)?.reshape(&[b * t, 1, h * wd, c])?;
        let v = self.to_v.forward(&xn)?.reshape(&[b * t, 1, h * wd, c])?;
        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None, None)?;
        let o = o.reshape(&[b * t, h * wd, c])?;
        let o = self.to_out.forward(&o)?;
        // back to NCTHW
        let o = o
            .transpose_axes(&[0, 2, 1])?
            .reshape(&[b, t, c, h, wd])?
            .transpose_axes(&[0, 2, 1, 3, 4])?;
        Ok(add(&o, &residual)?)
    }
}

/// Stride-2 down sampler: asymmetric `(0,1)` H/W pad then a causal conv (spatial-only `kt=1`, or
/// temporal `kt=3, st=2`). `temporal` selects the temporal stride/kernel.
struct Downsample3d {
    conv: CausalConv3d,
}
impl Downsample3d {
    fn load(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        let (st, pt) = if temporal { (2, 1) } else { (1, 0) };
        Ok(Self {
            conv: CausalConv3d::load(w, &format!("{prefix}.conv"), (st, 2, 2), (pt, 0, 0), false)?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let xp = pad(x, &[(0, 0), (0, 0), (0, 0), (0, 1), (0, 1)][..], None, None)?;
        self.conv.forward(&xp)
    }
}

/// Pixel-shuffle upsampler: `upscale_conv` (1³, C→C·sf²·tf) → reshape/transpose → `conv` (3³).
struct Upsample3d {
    upscale_conv: CausalConv3d,
    conv: CausalConv3d,
    sf: i32,
    tf: i32,
}
impl Upsample3d {
    fn load(w: &Weights, prefix: &str, temporal: bool) -> Result<Self> {
        Ok(Self {
            upscale_conv: CausalConv3d::load(
                w,
                &format!("{prefix}.upscale_conv"),
                (1, 1, 1),
                (0, 0, 0),
                false,
            )?,
            conv: CausalConv3d::load(w, &format!("{prefix}.conv"), (1, 1, 1), (1, 1, 1), true)?,
            sf: 2,
            tf: if temporal { 2 } else { 1 },
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, c, t, h, wd) = (sh[0], sh[1], sh[2], sh[3], sh[4]);
        let x = self.upscale_conv.forward(x)?; // (B, C·sf²·tf, T, H, W)
        let (sf, tf) = (self.sf, self.tf);
        // (B, sf, sf, tf, C, T, H, W) -> (B, C, T, tf, H, sf, W, sf) -> (B, C, T·tf, H·sf, W·sf)
        let x = x
            .reshape(&[b, sf, sf, tf, c, t, h, wd])?
            .transpose_axes(&[0, 4, 5, 3, 6, 1, 7, 2])?
            .reshape(&[b, c, t * tf, h * sf, wd * sf])?;
        let x = if t == 1 && tf > 1 { first_t(&x)? } else { x };
        self.conv.forward(&x)
    }
}

/// `num_resnets` resnets then an optional sampler.
struct DownBlock3d {
    resnets: Vec<ResnetBlock3d>,
    downsampler: Option<Downsample3d>,
}
impl DownBlock3d {
    fn load(
        w: &Weights,
        prefix: &str,
        n: i32,
        temporal: bool,
        sample: bool,
        cfg: &VaeConfig,
    ) -> Result<Self> {
        let resnets = (0..n)
            .map(|i| ResnetBlock3d::load(w, &format!("{prefix}.resnets.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        let downsampler = if sample {
            Some(Downsample3d::load(
                w,
                &format!("{prefix}.downsamplers.0"),
                temporal,
            )?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsampler,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(d) = &self.downsampler {
            h = d.forward(&h)?;
        }
        Ok(h)
    }
}

struct UpBlock3d {
    resnets: Vec<ResnetBlock3d>,
    upsampler: Option<Upsample3d>,
}
impl UpBlock3d {
    fn load(
        w: &Weights,
        prefix: &str,
        n: i32,
        temporal: bool,
        sample: bool,
        cfg: &VaeConfig,
    ) -> Result<Self> {
        let resnets = (0..n)
            .map(|i| ResnetBlock3d::load(w, &format!("{prefix}.resnets.{i}"), cfg))
            .collect::<Result<Vec<_>>>()?;
        let upsampler = if sample {
            Some(Upsample3d::load(
                w,
                &format!("{prefix}.upsamplers.0"),
                temporal,
            )?)
        } else {
            None
        };
        Ok(Self { resnets, upsampler })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(u) = &self.upsampler {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

/// `resnet → attention → resnet` at constant channels.
struct MidBlock3d {
    resnet0: ResnetBlock3d,
    attn: Attention3d,
    resnet1: ResnetBlock3d,
}
impl MidBlock3d {
    fn load(w: &Weights, prefix: &str, cfg: &VaeConfig) -> Result<Self> {
        Ok(Self {
            resnet0: ResnetBlock3d::load(w, &format!("{prefix}.resnets.0"), cfg)?,
            attn: Attention3d::load(w, &format!("{prefix}.attentions.0"), cfg)?,
            resnet1: ResnetBlock3d::load(w, &format!("{prefix}.resnets.1"), cfg)?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let h = self.resnet0.forward(x)?;
        let h = self.attn.forward(&h)?;
        self.resnet1.forward(&h)
    }
}

struct Encoder3d {
    conv_in: CausalConv3d,
    down_blocks: Vec<DownBlock3d>,
    mid: MidBlock3d,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out: CausalConv3d,
    groups: i32,
    eps: f32,
}
impl Encoder3d {
    fn load(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.enc_layers_per_block;
        // down0 spatial-only; down1/down2 temporal; down3 no sampler.
        let down_blocks = vec![
            DownBlock3d::load(w, "encoder.down_blocks.0", n, false, true, cfg)?,
            DownBlock3d::load(w, "encoder.down_blocks.1", n, true, true, cfg)?,
            DownBlock3d::load(w, "encoder.down_blocks.2", n, true, true, cfg)?,
            DownBlock3d::load(w, "encoder.down_blocks.3", n, false, false, cfg)?,
        ];
        Ok(Self {
            conv_in: CausalConv3d::load(w, "encoder.conv_in", (1, 1, 1), (1, 1, 1), false)?,
            down_blocks,
            mid: MidBlock3d::load(w, "encoder.mid_block", cfg)?,
            norm_out_w: w.require("encoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("encoder.conv_norm_out.bias")?.clone(),
            conv_out: CausalConv3d::load(w, "encoder.conv_out", (1, 1, 1), (1, 1, 1), false)?,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = self.conv_in.forward(x)?;
        for d in &self.down_blocks {
            h = d.forward(&h)?;
        }
        h = self.mid.forward(&h)?;
        h = gn(
            &h,
            &self.norm_out_w,
            &self.norm_out_b,
            self.groups,
            self.eps,
        )?;
        self.conv_out.forward(&silu(&h)?)
    }
}

struct Decoder3d {
    conv_in: CausalConv3d,
    mid: MidBlock3d,
    up_blocks: Vec<UpBlock3d>,
    norm_out_w: Array,
    norm_out_b: Array,
    conv_out: CausalConv3d,
    groups: i32,
    eps: f32,
}
impl Decoder3d {
    fn load(w: &Weights, cfg: &VaeConfig) -> Result<Self> {
        let n = cfg.dec_layers_per_block;
        // up0/up1 temporal; up2 spatial-only; up3 no sampler.
        let up_blocks = vec![
            UpBlock3d::load(w, "decoder.up_blocks.0", n, true, true, cfg)?,
            UpBlock3d::load(w, "decoder.up_blocks.1", n, true, true, cfg)?,
            UpBlock3d::load(w, "decoder.up_blocks.2", n, false, true, cfg)?,
            UpBlock3d::load(w, "decoder.up_blocks.3", n, false, false, cfg)?,
        ];
        Ok(Self {
            conv_in: CausalConv3d::load(w, "decoder.conv_in", (1, 1, 1), (1, 1, 1), false)?,
            mid: MidBlock3d::load(w, "decoder.mid_block", cfg)?,
            up_blocks,
            norm_out_w: w.require("decoder.conv_norm_out.weight")?.clone(),
            norm_out_b: w.require("decoder.conv_norm_out.bias")?.clone(),
            conv_out: CausalConv3d::load(w, "decoder.conv_out", (1, 1, 1), (1, 1, 1), false)?,
            groups: cfg.group_norm_groups,
            eps: cfg.group_norm_eps,
        })
    }
    fn forward(&self, z: &Array) -> Result<Array> {
        let mut h = self.conv_in.forward(z)?;
        h = self.mid.forward(&h)?;
        for u in &self.up_blocks {
            h = u.forward(&h)?;
        }
        h = gn(
            &h,
            &self.norm_out_w,
            &self.norm_out_b,
            self.groups,
            self.eps,
        )?;
        self.conv_out.forward(&silu(&h)?)
    }
}

/// The SeedVR2 3D causal video VAE.
pub struct Seedvr2Vae {
    encoder: Encoder3d,
    decoder: Decoder3d,
    scaling_factor: f32,
    pub spatial_scale: i32,
}

impl Seedvr2Vae {
    pub fn from_weights(w: &Weights) -> Result<Self> {
        let cfg = VaeConfig::seedvr2();
        Ok(Self {
            encoder: Encoder3d::load(w, &cfg)?,
            decoder: Decoder3d::load(w, &cfg)?,
            scaling_factor: cfg.scaling_factor,
            spatial_scale: cfg.spatial_scale,
        })
    }

    /// `(B,3,T,H,W)` → scaled mean latent `(B,16,T',H',W')`. A 4-D `(B,3,H,W)` input gains `T=1`.
    pub fn encode(&self, x: &Array) -> Result<Array> {
        let x = if x.ndim() == 4 {
            x.expand_dims(2)?
        } else {
            x.clone()
        };
        let h = self.encoder.forward(&x)?; // (B,32,T',H',W')
        let mean = &split(&h, 2, 1)?[0]; // first 16 channels
        Ok(multiply(mean, Array::from_f32(self.scaling_factor))?)
    }

    /// `(B,16,T',H',W')` → `(B,3,T,H,W)`. A 4-D latent gains `T=1`.
    pub fn decode(&self, z: &Array) -> Result<Array> {
        let z = if z.ndim() == 4 {
            z.expand_dims(2)?
        } else {
            z.clone()
        };
        let z = multiply(&z, Array::from_f32(1.0 / self.scaling_factor))?;
        self.decoder.forward(&z)
    }
}

#[cfg(test)]
mod stage_tests {
    use super::*;

    fn gdir() -> std::path::PathBuf {
        std::env::var("SEEDVR2_GOLDEN_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(&std::env::var("HOME").unwrap())
                    .join(".cache/mlx-gen-seedvr2-golden")
            })
    }

    fn cmp(label: &str, got: &Array, exp: &Array) {
        assert_eq!(got.shape(), exp.shape(), "{label} shape");
        // reshape to 1-D forces a contiguous logical-order copy (conv/gn outputs are transposed views)
        let g = got
            .as_dtype(Dtype::Float32)
            .unwrap()
            .reshape(&[-1])
            .unwrap();
        let e = exp.reshape(&[-1]).unwrap();
        let (gs, es) = (g.as_slice::<f32>(), e.as_slice::<f32>());
        let mut dot = 0f64;
        let (mut na, mut nb, mut maxd, mut maxr) = (0f64, 0f64, 0f32, 0f32);
        for (a, b) in gs.iter().zip(es.iter()) {
            dot += (*a as f64) * (*b as f64);
            na += (*a as f64).powi(2);
            nb += (*b as f64).powi(2);
            maxd = maxd.max((a - b).abs());
            maxr = maxr.max(b.abs());
        }
        let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-12);
        eprintln!(
            "[{label}] {:?} cosine={cos:.6} peak_rel={:.3e}",
            got.shape(),
            maxd / maxr.max(1e-12)
        );
    }

    #[test]
    fn decoder_stage_localize() {
        let dir = gdir();
        if !dir.join("vae_io_f32.safetensors").exists() {
            eprintln!("SKIP: no goldens");
            return;
        }
        let w = Weights::from_file(dir.join("vae_f32.safetensors")).unwrap();
        let io = Weights::from_file(dir.join("vae_io_f32.safetensors")).unwrap();
        let vae = Seedvr2Vae::from_weights(&w).unwrap();
        let dec = &vae.decoder;
        let z = multiply(
            io.require("enc_img").unwrap(),
            Array::from_f32(1.0 / vae.scaling_factor),
        )
        .unwrap();
        let h = dec.conv_in.forward(&z).unwrap();
        cmp("d_conv_in", &h, io.require("d_conv_in").unwrap());
        let h = dec.mid.forward(&h).unwrap();
        cmp("d_mid", &h, io.require("d_mid").unwrap());
        let mut h = h;
        for (i, ub) in dec.up_blocks.iter().enumerate() {
            h = ub.forward(&h).unwrap();
            cmp(
                &format!("d_up{i}"),
                &h,
                io.require(&format!("d_up{i}")).unwrap(),
            );
        }
        // isolate the final tail on the GOLDEN up3 input
        let up3g = io.require("d_up3").unwrap();
        let n = gn(up3g, &dec.norm_out_w, &dec.norm_out_b, dec.groups, dec.eps).unwrap();
        cmp("d_normout", &n, io.require("d_normout").unwrap());
        let s = silu(&n).unwrap();
        cmp("d_silu", &s, io.require("d_silu").unwrap());
        let o = dec.conv_out.forward(&s).unwrap();
        cmp("d_convout", &o, io.require("d_convout").unwrap());
    }
}
