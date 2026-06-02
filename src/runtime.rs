//! Shared load/exec types used by both [`Generator`](crate::generator::Generator) and
//! [`Transform`](crate::transform::Transform): where weights come from, quantization +
//! precision knobs, adapter specs, cooperative cancellation, and progress events.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Where a model's weights come from. (An HF-hub fetch variant is a planned additive
/// extension — see sc-2340; today's loaders read local safetensors.)
#[derive(Clone, Debug)]
pub enum WeightsSource {
    /// A directory of (possibly sharded) `.safetensors`.
    Dir(PathBuf),
    /// A single `.safetensors` file.
    File(PathBuf),
}

/// On-the-fly quantization level — group-wise affine Q4/Q8 (see [`crate::quant`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Q4,
    Q8,
}

impl Quant {
    /// Bit-width passed to the MLX quantizer.
    pub fn bits(self) -> i32 {
        match self {
            Quant::Q4 => 4,
            Quant::Q8 => 8,
        }
    }
}

/// Compute precision for dense (non-quantized) weights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Precision {
    #[default]
    Bf16,
    Fp32,
}

/// How to load a model. `weights` is required; everything else defaults to dense bf16. The
/// device is the process-default Metal GPU — the crate runs single-device (the MLX default
/// device is not thread-safe; the worker serializes jobs per thread).
#[derive(Clone, Debug)]
pub struct LoadSpec {
    pub weights: WeightsSource,
    pub quantize: Option<Quant>,
    pub precision: Precision,
    /// Auxiliary control-branch weights overlaid onto the base model at load time — a ControlNet
    /// checkpoint applied on top of `weights` (e.g. Z-Image's Fun-Controlnet-Union safetensors).
    /// `None` for the plain base model; a control-variant loader requires it. A load-time model
    /// *component* (it alters the graph), distinct from [`adapters`](Self::adapters) below, which
    /// are forward-time residual overlays on existing linears.
    pub control: Option<WeightsSource>,
    /// LoRA/LoKr adapters baked onto the model at load time. Multiples + mixed LoRA/LoKr stack by
    /// construction (see [`crate::adapters`]). Applied during `load` on the still-mutable model —
    /// the seam, since `Generator::generate`/`Transform::apply` take `&self` and the frozen fork
    /// likewise applies adapters in its initializer. Changing the adapter set means reloading.
    pub adapters: Vec<AdapterSpec>,
}

impl LoadSpec {
    /// Dense bf16 load from the given source.
    pub fn new(weights: WeightsSource) -> Self {
        Self {
            weights,
            quantize: None,
            precision: Precision::Bf16,
            control: None,
            adapters: Vec::new(),
        }
    }

    /// Builder-style quantization override.
    pub fn with_quant(mut self, quant: Quant) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Builder-style control-branch overlay (the ControlNet checkpoint over the base `weights`).
    pub fn with_control(mut self, control: WeightsSource) -> Self {
        self.control = Some(control);
        self
    }

    /// Builder-style LoRA/LoKr adapters to bake on at load time (replaces any already set).
    pub fn with_adapters(mut self, adapters: Vec<AdapterSpec>) -> Self {
        self.adapters = adapters;
        self
    }
}

/// A single adapter to stack at load time. Multiples + mixed LoRA/LoKr are supported by
/// construction — see [`crate::adapters`]. Carried by [`LoadSpec::adapters`].
#[derive(Clone, Debug)]
pub struct AdapterSpec {
    pub path: PathBuf,
    pub scale: f32,
    pub kind: AdapterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterKind {
    Lora,
    Lokr,
}

/// Cooperative cancellation handle threaded into a request; a model checks it between steps
/// and bails early. Cloneable — the caller keeps a handle to cancel an in-flight job.
#[derive(Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation of the in-flight generation.
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl std::fmt::Debug for CancelFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("CancelFlag")
            .field(&self.is_cancelled())
            .finish()
    }
}

/// A progress event streamed to the caller during a long `generate` / `apply`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Denoising step `current` of `total` (1-based).
    Step { current: u32, total: u32 },
    /// VAE decode underway (post-denoise).
    Decoding,
}
