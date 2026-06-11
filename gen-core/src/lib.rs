//! # gen-core
//!
//! The **backend-neutral contract layer** for SceneWorks generative inference. gen-core has
//! **zero tensor dependencies**: it owns the `Generator` / `Trainer` / `Captioner` / `Transform`
//! contracts, the request/output/conditioning/progress/cancel/error types, the link-time model
//! registry, and the pure host-side policy math (tokenization, PIL-compatible resize, tiling,
//! LR schedule). The tensor backends — `mlx-gen` (Apple MLX) and the forthcoming `candle-gen`
//! (Windows/CUDA) — implement these contracts and re-export this crate at their own paths.
//!
//! Numeric types here are restricted to `f32`/`f64`/`Vec<f32>`/`Vec<i32>`/`&[u8]` — never an
//! `mlx_rs::Array` or candle tensor. See epic 3720 (the unified-contract roadmap, Phase 0).

pub mod caption;
pub mod error;
pub mod generator;
pub mod imageops;
pub mod media;
pub mod registry;
pub mod runtime;
pub mod tiling;
pub mod tokenizer;
pub mod train;
pub mod transform;

pub use caption::{
    CaptionCapabilities, CaptionFinishReason, CaptionOptions, CaptionOutput, CaptionRequest,
    CaptionSampling, Captioner, CaptionerDescriptor,
};
pub use error::{Error, Result};
pub use generator::{
    default_seed, Capabilities, Conditioning, ConditioningKind, ControlClipRef, ControlKind,
    GenerationOutput, GenerationRequest, Generator, KeyframeRef, Modality, ModelDescriptor,
    ReplacementMode, VideoClipRef,
};
pub use media::{AudioTrack, Image};
pub use registry::{
    load, load_captioner, load_transform, CaptionerRegistration, ModelRegistration,
    TransformRegistration,
};
pub use registry::{load_trainer, TrainerRegistration};
pub use runtime::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, MoeExpert, Precision, Progress, Quant,
    WeightsSource,
};
pub use tiling::{TilingConfig, VaeTiling};
// NOTE: `TrainOptimizer` is intentionally NOT re-exported here — it wraps an mlx-rs optimizer and
// lives in mlx-gen (`mlx_gen::train::optim`). `LrSchedule` is pure policy and lives here.
pub use train::{
    LrSchedule, NetworkType, Trainer, TrainerDescriptor, TrainingConfig, TrainingItem,
    TrainingOutput, TrainingProgress, TrainingRequest,
};
pub use transform::{
    TargetSize, Transform, TransformCapabilities, TransformDescriptor, TransformRequest,
};

/// gen-core's package version, for the version-skew runtime guard (sc-4482).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
