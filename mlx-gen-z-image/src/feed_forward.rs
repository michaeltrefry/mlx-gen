//! Z-Image SwiGLU feed-forward: `w2(silu(w1(x)) * w3(x))`. Port of the Python fork's
//! `models/z_image/.../feed_forward.py`. `w1`/`w2`/`w3` are adapter hosts (LoRA/LoKr targets).

use mlx_rs::{
    ops::{multiply, sigmoid},
    Array,
};

use mlx_gen::adapters::{AdaptableHost, AdaptableLinear};
use mlx_gen::weights::Weights;
use mlx_gen::Result;

pub struct FeedForward {
    pub w1: AdaptableLinear,
    pub w2: AdaptableLinear,
    pub w3: AdaptableLinear,
}

impl FeedForward {
    /// Load the three projections (no bias) from `{prefix}.w{1,2,3}.weight`.
    pub fn from_weights(w: &Weights, prefix: &str) -> Result<Self> {
        Ok(Self {
            w1: AdaptableLinear::dense(w.require(&format!("{prefix}.w1.weight"))?.clone(), None),
            w2: AdaptableLinear::dense(w.require(&format!("{prefix}.w2.weight"))?.clone(), None),
            w3: AdaptableLinear::dense(w.require(&format!("{prefix}.w3.weight"))?.clone(), None),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let h1 = self.w1.forward(x)?;
        let silu = multiply(&h1, &sigmoid(&h1)?)?;
        let h3 = self.w3.forward(x)?;
        self.w2.forward(&multiply(&silu, &h3)?)
    }
}

impl AdaptableHost for FeedForward {
    fn adaptable_mut(&mut self, path: &[&str]) -> Option<&mut AdaptableLinear> {
        match path {
            ["w1"] => Some(&mut self.w1),
            ["w2"] => Some(&mut self.w2),
            ["w3"] => Some(&mut self.w3),
            _ => None,
        }
    }
}
