//! Z-Image attention: QKV projections → optional QK-RMSNorm → 3D RoPE → SDPA → out
//! projection. Port of the Python fork's `models/z_image/.../attention.py`, made
//! dimension-parametric. `to_q/to_k/to_v/to_out` are adapter hosts (LoRA/LoKr targets).
//!
//! Numeric parity proven stage-by-stage in the sc-2338 spike (tolerance 1e-2 — MLX runs
//! fp32 matmul in reduced precision on Metal). Attention mask is not yet wired (the
//! turbo T2I path runs full-valid, SDPA `mask=None`); it lands with the full model.

use mlx_rs::{
    fast::{rms_norm, scaled_dot_product_attention},
    ops::{add, multiply, split, stack_axis, subtract},
    Array,
};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct ZImageAttention {
    pub to_q: AdaptableLinear,
    pub to_k: AdaptableLinear,
    pub to_v: AdaptableLinear,
    pub to_out: AdaptableLinear,
    norm_q: Option<Array>,
    norm_k: Option<Array>,
    n_heads: i32,
    head_dim: i32,
    scale: f32,
    eps: f32,
}

impl ZImageAttention {
    /// Load from `{prefix}.{to_q,to_k,to_v,to_out.0,norm_q,norm_k}.weight`. QK-norm weights
    /// are optional (present iff `qk_norm=True`, which Z-Image-turbo uses).
    pub fn from_weights(
        w: &Weights,
        prefix: &str,
        dim: i32,
        n_heads: i32,
        eps: f32,
    ) -> Result<Self> {
        let head_dim = dim / n_heads;
        Ok(Self {
            to_q: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_q.weight"))?.clone(),
                None,
            ),
            to_k: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_k.weight"))?.clone(),
                None,
            ),
            to_v: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_v.weight"))?.clone(),
                None,
            ),
            to_out: AdaptableLinear::dense(
                w.require(&format!("{prefix}.to_out.0.weight"))?.clone(),
                None,
            ),
            norm_q: w.get(&format!("{prefix}.norm_q.weight")).cloned(),
            norm_k: w.get(&format!("{prefix}.norm_k.weight")).cloned(),
            n_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            eps,
        })
    }

    pub fn forward(&self, x: &Array, freqs_cis: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s) = (sh[0], sh[1]);
        let dim = self.n_heads * self.head_dim;

        let mut q = self
            .to_q
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;
        let mut k = self
            .to_k
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;
        let v = self
            .to_v
            .forward(x)?
            .reshape(&[b, s, self.n_heads, self.head_dim])?;

        if let Some(nq) = &self.norm_q {
            q = rms_norm(&q, nq, self.eps)?;
        }
        if let Some(nk) = &self.norm_k {
            k = rms_norm(&k, nk, self.eps)?;
        }

        q = self.apply_rope(&q, freqs_cis)?;
        k = self.apply_rope(&k, freqs_cis)?;

        // (b, s, h, hd) -> (b, h, s, hd)
        let q = q.transpose_axes(&[0, 2, 1, 3])?;
        let k = k.transpose_axes(&[0, 2, 1, 3])?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let o = scaled_dot_product_attention(&q, &k, &v, self.scale, None)?;
        let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, s, dim])?;
        self.to_out.forward(&o)
    }

    /// Port of `ZImageAttention._apply_rotary_emb`. `x`:(b,s,h,hd), `fc`:(s,hd/2,2).
    fn apply_rope(&self, x: &Array, fc: &Array) -> Result<Array> {
        let sh = x.shape();
        let (b, s, h, hd) = (sh[0], sh[1], sh[2], sh[3]);
        let half = hd / 2;
        let x5 = x.reshape(&[b, s, h, half, 2])?;
        let xp = split(&x5, 2, 4)?;
        let xr = xp[0].reshape(&[b, s, h, half])?;
        let xi = xp[1].reshape(&[b, s, h, half])?;
        let fc5 = fc.reshape(&[1, s, 1, half, 2])?;
        let fp = split(&fc5, 2, 4)?;
        let cos = fp[0].reshape(&[1, s, 1, half])?;
        let sin = fp[1].reshape(&[1, s, 1, half])?;
        let out_r = subtract(&multiply(&xr, &cos)?, &multiply(&xi, &sin)?)?;
        let out_i = add(&multiply(&xr, &sin)?, &multiply(&xi, &cos)?)?;
        Ok(stack_axis(&[out_r, out_i], 4)?.reshape(&[b, s, h, hd])?)
    }
}

impl AdaptableHost for ZImageAttention {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["to_q"] => Some(&mut self.to_q),
            ["to_k"] => Some(&mut self.to_k),
            ["to_v"] => Some(&mut self.to_v),
            ["to_out", "0"] => Some(&mut self.to_out),
            _ => None,
        }
    }
}
