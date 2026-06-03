//! Per-stream feed-forward: `mlp_out(gelu_approx(mlp_in(x)))` (both biased, 4× expansion).
//! Port of the fork's `QwenFeedForward`. Both Linears are [`AdaptableLinear`] so the transformer
//! can be quantized (Q8) without changing the forward.

use mlx_rs::Array;

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::nn::gelu_tanh;
use mlx_gen::weights::Weights;
use mlx_gen::Result;

use super::{join, linear_from};

pub struct FeedForward {
    mlp_in: AdaptableLinear,
    mlp_out: AdaptableLinear,
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        // Trained-file (diffusers) naming: `{img,txt}_mlp.net.0.proj` (in) / `.net.2` (out).
        match path {
            ["net", "0", "proj"] => Some(&mut self.mlp_in),
            ["net", "2"] => Some(&mut self.mlp_out),
            _ => None,
        }
    }

    fn adaptable_paths(&self) -> Vec<String> {
        ["net.0.proj", "net.2"]
            .into_iter()
            .map(String::from)
            .collect()
    }
}

impl FeedForward {
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            mlp_in: linear_from(w, &join(prefix, "mlp_in"), true)?,
            mlp_out: linear_from(w, &join(prefix, "mlp_out"), true)?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        // Dtype-preserving, golden-bit-exact tanh-GELU (sc-2779). `mlx_rs::nn::gelu_approximate`
        // uses an f32 `√(2/π)` (1 ULP off the fork's f64-host const) and promotes a bf16 input to
        // f32; `gelu_tanh` matches `nn.GELU(approx="tanh")` and preserves the input dtype.
        let h = gelu_tanh(&self.mlp_in.forward(x)?)?;
        self.mlp_out.forward(&h)
    }

    pub fn quantize(&mut self, bits: i32) -> Result<()> {
        self.mlp_in.quantize(bits, None)?;
        self.mlp_out.quantize(bits, None)?;
        Ok(())
    }
}
