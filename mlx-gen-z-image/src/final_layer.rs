//! Z-Image final layer. Port of `FinalLayer`: affine-free LayerNorm scaled by
//! `1 + adaLN(silu(t_emb))`, then a Linear projection back to patch space.

use mlx_rs::fast::layer_norm;
use mlx_rs::ops::{add, multiply};
use mlx_rs::Array;

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

const LN_EPS: f32 = 1e-6;

pub struct FinalLayer {
    linear_w: Array,
    linear_b: Array,
    ada_w: Array,
    ada_b: Array,
}

impl FinalLayer {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            linear_w: w.require(&format!("{prefix}.linear.weight"))?.clone(),
            linear_b: w.require(&format!("{prefix}.linear.bias"))?.clone(),
            ada_w: w
                .require(&format!("{prefix}.adaLN_modulation.0.weight"))?
                .clone(),
            ada_b: w
                .require(&format!("{prefix}.adaLN_modulation.0.bias"))?
                .clone(),
        })
    }

    /// `x`: `(B, S, H)`, `c` (timestep emb): `(B, min(H,256))` → `(B, S, out_channels)`.
    pub fn forward(&self, x: &Array, c: &Array) -> Result<Array> {
        let scale = add(
            &linear(&silu(c)?, &self.ada_w, &self.ada_b)?,
            Array::from_slice(&[1.0f32], &[1]),
        )?;
        let scale = scale.expand_dims(1)?; // (B, 1, H)
        let normed = layer_norm(x, None, None, LN_EPS)?; // affine=False
        linear(&multiply(&normed, &scale)?, &self.linear_w, &self.linear_b)
    }
}
