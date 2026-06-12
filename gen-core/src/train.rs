//! The `Trainer` contract — LoRA/LoKr fine-tuning of a registered model (epic 3039), the training
//! analog of [`Generator`](crate::generator::Generator). See `docs/MODEL_ARCHITECTURE.md`.
//!
//! SceneWorks owns the training *product* surface (datasets, plan normalization, validation, the
//! queue) in Rust; this is the *execution* surface that replaces the Python kernel for the
//! MLX-native families. The worker maps its normalized `TrainingPlan` onto a [`TrainingRequest`]
//! and calls [`Trainer::train`] — exactly as it maps `ImageRequest` → `GenerationRequest` and calls
//! `Generator::generate`. mlx-gen owns these shapes; it does not depend on the SceneWorks contract.
//!
//! The spike (sc-3042) proved the per-family training mechanism (functional autograd over an
//! external LoRA factor map, re-injected as `Adapter::Lora`, stepped with `keyed_value_and_grad` +
//! AdamW). This module is the family-agnostic glue around it: the config/progress/request shapes,
//! the [`schedule`] LR helpers, dataset bucketing, and checkpointing. Each family crate implements
//! [`Trainer`] (Z-Image in sc-3044) and self-registers via [`crate::registry::TrainerRegistration`].

// The pure LR-schedule policy lives here (gen-core); the MLX training kernels
// (checkpoint/dataset/lora/optim, incl. `TrainOptimizer`) stay in mlx-gen's `train` module.
pub mod schedule;

use std::path::PathBuf;

pub use schedule::LrSchedule;

use crate::generator::Modality;
use crate::runtime::CancelFlag;

/// Adapter network parameterization (mirrors SceneWorks `network_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NetworkType {
    /// Standard low-rank `A·B` adapter.
    #[default]
    Lora,
    /// LyCORIS Kronecker-product adapter (LoKr); `decompose_factor` is the block-split knob.
    Lokr,
}

impl NetworkType {
    /// Parse the free-form contract string; unknown / empty → `Lora`.
    pub fn parse(name: &str) -> Self {
        match name.trim().to_ascii_lowercase().as_str() {
            "lokr" => NetworkType::Lokr,
            _ => NetworkType::Lora,
        }
    }
}

/// Concrete training hyperparameters — the engine-side mirror of SceneWorks' `TrainingConfig`
/// (with the typed equivalents of the fields its Python kernel reads out of the plan's `advanced`
/// bag: LR schedule, warmup, timestep sampling, loss type, LoKr decompose factor, target modules).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingConfig {
    /// LoRA/LoKr rank (network dimension).
    pub rank: u32,
    /// LoRA alpha; the residual is scaled by `alpha/rank`.
    pub alpha: f32,
    pub learning_rate: f32,
    /// Total training micro-steps (forward/backward passes; an optimizer update fires every
    /// `gradient_accumulation` of them).
    pub steps: u32,
    pub batch_size: u32,
    pub gradient_accumulation: u32,
    /// Gradient (activation) checkpointing: recompute each transformer block's activations during
    /// the backward pass instead of retaining them, bounding the first-step working set (sc-4874 —
    /// without it a production-resolution run can exceed unified memory and the OS hard-kills the
    /// worker). This is the engine-side home of the SceneWorks "Gradient Checkpointing" toggle (which
    /// was previously a no-op on the Rust path). Strictly opt-in — never auto-enabled; a run that
    /// would exceed the memory budget with it off is refused up front by the family trainer's
    /// pre-flight guard (a catchable error recommending this flag). Numerically it changes nothing
    /// (grads are bit-identical to the dense path); the cost is recompute time.
    pub gradient_checkpointing: bool,
    /// Training compute dtype for the model forward/backward: `"bf16"` (default — halves the
    /// activation working set; the ecosystem-standard mixed precision, sc-4887) or `"f32"` (full
    /// precision, the pre-sc-4887 behavior). The trainable adapter factors, loss, gradients, and
    /// optimizer state stay f32 either way (master-weights pattern); only the frozen base weights
    /// and the activation stream are cast. Unrecognized values mean f32.
    pub train_dtype: String,
    /// Square training resolution edge in pixels; bucketed down to a multiple of 32.
    pub resolution: u32,
    /// Adapter-checkpoint cadence, in micro-steps (`0` = no intermediate checkpoints).
    pub save_every: u32,
    pub seed: u64,
    /// Optimizer name, kept a free string to stay engine-agnostic (`adamw`/`adam`/`lion`/…); the
    /// family trainer maps it to an mlx-rs optimizer. Prodigy/Rose are not in mlx-rs (sc-3048).
    pub optimizer: String,
    pub weight_decay: f32,
    pub lr_scheduler: LrSchedule,
    pub lr_warmup_steps: u32,
    pub network_type: NetworkType,
    /// LoKr block-split factor (`-1` = auto). Ignored for plain LoRA.
    pub decompose_factor: i32,
    /// LoRA target module suffixes (e.g. `["to_q","to_k","to_v","to_out.0"]`); empty = the family
    /// default set.
    pub lora_target_modules: Vec<String>,
    /// Flow-match timestep sampling distribution (`sigmoid`/`linear`/`uniform`/…) — the *noise*
    /// schedule, distinct from `lr_scheduler`.
    pub timestep_type: String,
    /// Timestep sampling bias (`balanced`/`high_noise`/`low_noise`/…).
    pub timestep_bias: String,
    /// Loss type (`mse`/`mae`/…); families default to MSE on the velocity target.
    pub loss_type: String,
    /// Trigger word baked into captions / surfaced on the output adapter.
    pub trigger_word: Option<String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 16.0,
            learning_rate: 1e-4,
            steps: 1000,
            batch_size: 1,
            gradient_accumulation: 1,
            gradient_checkpointing: false,
            train_dtype: "bf16".to_string(),
            resolution: 1024,
            save_every: 250,
            seed: 0,
            optimizer: "adamw".to_string(),
            weight_decay: 0.0,
            lr_scheduler: LrSchedule::Constant,
            lr_warmup_steps: 0,
            network_type: NetworkType::Lora,
            decompose_factor: -1,
            lora_target_modules: Vec::new(),
            timestep_type: "sigmoid".to_string(),
            timestep_bias: "balanced".to_string(),
            loss_type: "mse".to_string(),
            trigger_word: None,
        }
    }
}

/// One captioned training example. Paths are resolved by the caller (the worker resolves the
/// dataset's absolute image paths).
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingItem {
    pub image_path: PathBuf,
    pub caption: String,
}

/// A training run: the dataset, the hyperparameters, and where to write the adapter. The base model
/// is supplied at `load`-time via the [`LoadSpec`](crate::runtime::LoadSpec) (mirroring inference:
/// `load(id, spec)` then `generate(req)`), so it is not repeated here.
#[derive(Clone, Debug)]
pub struct TrainingRequest {
    pub items: Vec<TrainingItem>,
    pub config: TrainingConfig,
    /// Absolute directory the adapter (and any intermediate checkpoints) are written into.
    pub output_dir: PathBuf,
    /// Output adapter file name, e.g. `my_style.safetensors`.
    pub file_name: String,
    /// Trigger words surfaced on the produced adapter.
    pub trigger_words: Vec<String>,
    /// Cooperative cancellation, polled between steps (mirrors `GenerationRequest`).
    pub cancel: CancelFlag,
}

/// A progress event streamed during a long [`Trainer::train`] — the training analog of
/// [`Progress`](crate::runtime::Progress), with bands matching the kernel's
/// prepare→load→cache→train→checkpoint→save lifecycle.
#[derive(Clone, Debug, PartialEq)]
pub enum TrainingProgress {
    /// Resolving the dataset / building buckets.
    Preparing,
    /// Loading the (frozen) base model weights.
    LoadingModel,
    /// Encoding + caching VAE latents and prompt embeddings: item `current` of `total` (1-based).
    Caching { current: u32, total: u32 },
    /// Optimizer micro-step `step` of `total` (1-based) with the latest scalar `loss`.
    Training { step: u32, total: u32, loss: f32 },
    /// An intermediate adapter checkpoint was written at micro-step `step`.
    Checkpoint { step: u32 },
    /// Writing the final adapter.
    Saving,
}

/// What a [`Trainer::train`] produced.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingOutput {
    /// Absolute path to the final adapter safetensors.
    pub adapter_path: PathBuf,
    /// Micro-steps actually run (may be < `config.steps` if cancelled).
    pub steps: u32,
    /// The last training loss observed.
    pub final_loss: f32,
}

/// Identity + capabilities of a trainer (drives `validate` and consumer introspection). The
/// training analog of [`ModelDescriptor`](crate::generator::ModelDescriptor).
#[derive(Clone, Copy, Debug)]
pub struct TrainerDescriptor {
    /// Registry id, e.g. `"z_image_turbo"` (matches the generator id of the same base model).
    pub id: &'static str,
    pub family: &'static str,
    /// Tensor backend that registered this trainer ("mlx" | "candle"); used by the worker's
    /// per-backend capability advertisement (sc-4906, epic 3720).
    pub backend: &'static str,
    pub modality: Modality,
    pub supports_lora: bool,
    pub supports_lokr: bool,
}

/// A LoRA/LoKr trainer for one model family — the training analog of
/// [`Generator`](crate::generator::Generator). `train` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread) and takes `&mut self` because training mutates the
/// adapter parameters and optimizer state. The request carries a cancel flag and `on_progress`
/// streams the lifecycle.
pub trait Trainer {
    /// Identity + capabilities (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &TrainerDescriptor;

    /// Reject a request this trainer cannot serve (LoKr when unsupported, empty dataset,
    /// unresolvable target modules, …) before doing expensive work.
    fn validate(&self, req: &TrainingRequest) -> crate::Result<()>;

    /// Run training to completion (or until `req.cancel` trips), writing the adapter to
    /// `req.output_dir`.
    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> crate::Result<TrainingOutput>;
}
