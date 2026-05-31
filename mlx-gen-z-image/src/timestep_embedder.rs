//! Z-Image timestep embedding. Port of `TimestepEmbedder`: sinusoidal frequency embedding
//! (`frequency_embedding_size=256`) → Linear → SiLU → Linear, producing `min(dim, 256)` dims.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::nn::{linear, silu};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct TimestepEmbedder {
    l1_w: Array,
    l1_b: Array,
    l2_w: Array,
    l2_b: Array,
    frequency_embedding_size: i32,
}

impl TimestepEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str, frequency_embedding_size: i32) -> Result<Self> {
        Ok(Self {
            l1_w: w.require(&format!("{prefix}.linear1.weight"))?.clone(),
            l1_b: w.require(&format!("{prefix}.linear1.bias"))?.clone(),
            l2_w: w.require(&format!("{prefix}.linear2.weight"))?.clone(),
            l2_b: w.require(&format!("{prefix}.linear2.bias"))?.clone(),
            frequency_embedding_size,
        })
    }

    /// `t`: `(B,)` → `(B, out)`.
    pub fn forward(&self, t: &Array) -> Result<Array> {
        let t_freq = self.timestep_embedding(t)?;
        let h = silu(&linear(&t_freq, &self.l1_w, &self.l1_b)?)?;
        linear(&h, &self.l2_w, &self.l2_b)
    }

    /// Sinusoidal embedding: `concat(cos(t·freqs), sin(t·freqs))`, `freqs[i]=10000^(-i/half)`.
    /// `frequency_embedding_size` is even for Z-Image, so no zero-pad column is needed.
    fn timestep_embedding(&self, t: &Array) -> Result<Array> {
        let dim = self.frequency_embedding_size;
        let half = (dim / 2) as usize;
        let max_period = 10000.0_f32;
        let freqs: Vec<f32> = (0..half)
            .map(|i| (-max_period.ln() * i as f32 / half as f32).exp())
            .collect();
        let freqs = Array::from_slice(&freqs, &[1, half as i32]);

        let b = t.shape()[0];
        let t_col = t.reshape(&[b, 1])?;
        let args = multiply(&t_col, &freqs)?; // (B, half)
        let cos = mlx_rs::ops::cos(&args)?;
        let sin = mlx_rs::ops::sin(&args)?;
        Ok(mlx_rs::ops::concatenate_axis(&[&cos, &sin], 1)?)
    }
}
