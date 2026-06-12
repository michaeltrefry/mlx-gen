//! The `Transform` contract — non-prompt image→image (restore / upscale), designed around
//! SeedVR2. See `docs/MODEL_ARCHITECTURE.md` §3.3.
//!
//! Restorers/upscalers are **not** `Generator`s: there is no prompt — the input image *is* the
//! subject. SeedVR2 is a diffusion-based single-image super-resolution model (`seed` + input +
//! target size + softness → restored image, 1-step, its own VAE+transformer, fixed text
//! embedding). Scope is image→image; a video restorer would extend this later, not now.

use crate::media::Image;
use crate::runtime::{CancelFlag, Progress};
use crate::Result;

/// A non-prompt image→image transform (super-resolution / restoration).
pub trait Transform {
    fn descriptor(&self) -> &TransformDescriptor;
    fn validate(&self, req: &TransformRequest) -> Result<()>;
    fn apply(&self, req: &TransformRequest, on_progress: &mut dyn FnMut(Progress))
        -> Result<Image>;
}

/// A transform request — `Default`-able like [`GenerationRequest`](crate::generator::GenerationRequest).
#[derive(Clone, Debug, Default)]
pub struct TransformRequest {
    pub image: Image,
    pub target: TargetSize,
    /// Diffusion restorers (SeedVR2) use this; deterministic ones ignore it.
    pub seed: Option<u64>,
    /// Model-defined restoration knob (SeedVR2 "softness", 0..1).
    pub strength: Option<f32>,
    /// SeedVR2 is 1-step; override only if the model allows it.
    pub steps: Option<u32>,
    pub cancel: CancelFlag,
}

/// How big to make the output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetSize {
    /// ESRGAN-style factor × the min edge (SeedVR2 "2x"/"3x").
    Scale(f32),
    /// Target for `min(w, h)` (SeedVR2 `resolution: int`).
    MinEdge(u32),
    /// Explicit output resolution.
    Resolution { width: u32, height: u32 },
}

impl Default for TargetSize {
    fn default() -> Self {
        TargetSize::Scale(2.0)
    }
}

/// A transform's stable identity + advertised capabilities.
#[derive(Clone, Debug)]
pub struct TransformDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this transform ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub capabilities: TransformCapabilities,
}

/// What target modes / knobs a transform supports.
#[derive(Clone, Debug, Default)]
pub struct TransformCapabilities {
    pub scale: bool,
    pub min_edge: bool,
    pub resolution: bool,
    pub max_scale: f32,
    /// Uses a seed (diffusion-based, e.g. SeedVR2).
    pub is_diffusion: bool,
    pub supports_strength: bool,
    pub mac_only: bool,
}
