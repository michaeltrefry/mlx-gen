//! Z-Image timestep embedding. Port of `TimestepEmbedder`: sinusoidal frequency embedding
//! (`frequency_embedding_size=256`) → Linear → SiLU → Linear, producing `min(dim, 256)` dims.

use mlx_rs::ops::multiply;
use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::silu;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct TimestepEmbedder {
    linear1: AdaptableLinear,
    linear2: AdaptableLinear,
    frequency_embedding_size: i32,
}

impl TimestepEmbedder {
    pub fn from_weights(w: &Weights, prefix: &str, frequency_embedding_size: i32) -> Result<Self> {
        let dense = |name: &str| -> Result<AdaptableLinear> {
            Ok(AdaptableLinear::dense(
                w.require(&format!("{prefix}.{name}.weight"))?.clone(),
                Some(w.require(&format!("{prefix}.{name}.bias"))?.clone()),
            ))
        };
        Ok(Self {
            linear1: dense("linear1")?,
            linear2: dense("linear2")?,
            frequency_embedding_size,
        })
    }

    /// Quantize both projections to Q4/Q8 (group_size 64) — both are `nn.Linear` in the fork.
    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        for lin in [&mut self.linear1, &mut self.linear2] {
            lin.quantize(bits, None)?;
        }
        Ok(())
    }

    /// `t`: `(B,)` → `(B, out)`.
    pub fn forward(&self, t: &Array) -> Result<Array> {
        let t_freq = self.timestep_embedding(t)?;
        let h = silu(&self.linear1.forward(&t_freq)?)?;
        self.linear2.forward(&h)
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

impl AdaptableHost for TimestepEmbedder {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file naming follows the fork's `t_embedder.mlp.{0,2}` (diffusers `Sequential`
        // indices) — index 0 is the first Linear (`linear1`), index 2 the second (`linear2`); the
        // SiLU at index 1 has no weights.
        match path {
            ["mlp", "0"] => Some(&mut self.linear1),
            ["mlp", "2"] => Some(&mut self.linear2),
            _ => None,
        }
    }
}
