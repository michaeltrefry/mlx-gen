//! The Lens denoising **DiT** (sc-3168) — a 48-layer dual-stream MMDiT with joint image+text
//! attention, complex axial RoPE on both streams, and SwiGLU MLPs. A faithful port of
//! `_vendor/lens/transformer.py::LensTransformer2DModel`, architecturally a near-twin of
//! `mlx-gen-qwen-image`'s MMDiT (the RoPE, joint attention, AdaLN modulation and `AdaLayerNormContinuous`
//! all follow that seam; the Lens-specific pieces are the **multi-layer text front-end**, the **fused
//! `img_qkv`/`txt_qkv`** projections, the **`[img, txt]`** join order, and the **SwiGLU GateMLP**).
//!
//! NCS (`[batch, seq, dim]`) tensors throughout. The model consumes already-patchified image latents
//! `[B, img_len, 128]` plus the 4 captured gpt-oss text-feature layers `[B, txt_len, 2880]` and
//! predicts the patch-space velocity `[B, img_len, 128]`; patch/unpatch + the sampler are sc-3170/3173.

pub mod attention;
pub mod block;
pub mod rope;
#[allow(clippy::module_inception)]
pub mod transformer;

pub use attention::LensJointAttention;
pub use block::LensTransformerBlock;
pub use rope::LensRope3d;
pub use transformer::{LensDitConfig, LensTransformer};

use mlx_rs::ops::{add, matmul};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Result;

/// A dense Linear `y = x·Wᵀ (+ b)` over a diffusers `[out, in]` weight (+ optional `[out]` bias),
/// cast to the working `dtype` at load. `Clone` is cheap (refcounted `Array`s) and required so a
/// checkpoint segment can OWN its block copy (sc-5170, mirrors z-image).
#[derive(Clone)]
pub(crate) struct Linear {
    w: Array, // [out, in]
    b: Option<Array>,
}

impl Linear {
    pub(crate) fn load(w: &Weights, prefix: &str, bias: bool, dtype: Dtype) -> Result<Self> {
        Ok(Self {
            w: w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?,
            b: if bias {
                Some(w.require(&format!("{prefix}.bias"))?.as_dtype(dtype)?)
            } else {
                None
            },
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        let y = matmul(x, self.w.t())?;
        match &self.b {
            Some(b) => Ok(add(&y, b)?),
            None => Ok(y),
        }
    }
}

/// Load an RMSNorm/affine weight at `{prefix}.weight`, cast to `dtype`.
pub(crate) fn load_weight(w: &Weights, prefix: &str, dtype: Dtype) -> Result<Array> {
    Ok(w.require(&format!("{prefix}.weight"))?.as_dtype(dtype)?)
}

/// Join a module prefix with a leaf name.
pub(crate) fn join(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}
