//! VAE mid-block spatial self-attention: GroupNorm → QKV → single-head SDPA over the H·W
//! tokens → out projection, with a residual. NCHW I/O.

use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::nn::{group_norm, linear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const GN_GROUPS: i32 = 32;
const GN_EPS: f32 = 1e-6;

pub struct VaeAttention {
    gn_w: Array,
    gn_b: Array,
    q_w: Array,
    q_b: Array,
    k_w: Array,
    k_b: Array,
    v_w: Array,
    v_b: Array,
    out_w: Array,
    out_b: Array,
}

impl VaeAttention {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        let g = |s: &str| w.require(&format!("{prefix}.{s}")).cloned();
        Ok(Self {
            gn_w: g("group_norm.weight")?,
            gn_b: g("group_norm.bias")?,
            q_w: g("to_q.weight")?,
            q_b: g("to_q.bias")?,
            k_w: g("to_k.weight")?,
            k_b: g("to_k.bias")?,
            v_w: g("to_v.weight")?,
            v_b: g("to_v.bias")?,
            out_w: g("to_out.0.weight")?,
            out_b: g("to_out.0.bias")?,
        })
    }

    pub fn forward(&self, x_nchw: &Array) -> Result<Array> {
        let x = x_nchw.transpose_axes(&[0, 2, 3, 1])?; // NHWC
        let sh = x.shape();
        let (b, h, w, c) = (sh[0], sh[1], sh[2], sh[3]);

        let normed = group_norm(&x, &self.gn_w, &self.gn_b, GN_GROUPS, GN_EPS)?;
        // (B,H,W,C) -> (B, H*W, 1, C) -> (B, 1, H*W, C) [single head, head_dim = C].
        let proj = |xw: &Array, xb: &Array| -> Result<Array> {
            Ok(linear(&normed, xw, xb)?
                .reshape(&[b, h * w, 1, c])?
                .transpose_axes(&[0, 2, 1, 3])?)
        };
        let q = proj(&self.q_w, &self.q_b)?;
        let k = proj(&self.k_w, &self.k_b)?;
        let v = proj(&self.v_w, &self.v_b)?;

        let scale = (c as f32).powf(-0.5);
        let o = scaled_dot_product_attention(&q, &k, &v, scale, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, h, w, c])?;
        let o = linear(&o, &self.out_w, &self.out_b)?;

        Ok(add(&x, &o)?.transpose_axes(&[0, 3, 1, 2])?) // residual, back to NCHW
    }
}
