//! # mlx-gen
//!
//! Rust-native inference for generative **image and video** models on Apple
//! [MLX](https://github.com/ml-explore/mlx), built on top of `mlx-rs`.
//!
//! **Status: active** — multiple merged, parity-validated provider crates spanning image,
//! video, identity, and understanding models, consumed in-process as a Rust library.
//!
//! Families: FLUX.1 / FLUX.2, Chroma, Qwen-Image (+ Edit), SDXL, Kolors, Z-Image,
//! SenseNova-U1 (image); Wan2.2, LTX-2.3, SVD (video); PuLID-FLUX, InstantID (identity);
//! JoyCaption, SAM2 (understanding). Adapters: LoRA, LoKr (with stacking), ControlNet,
//! IP-Adapter. Plus native MLX LoRA/LoKr training and group-wise Q4/Q8 quantization.
//!
//! Architecture: a *disciplined hybrid* of the frozen Python mflux fork — see
//! [`ARCHITECTURE.md`](https://github.com/michaeltrefry/mlx-gen/blob/main/ARCHITECTURE.md).

// The backend-neutral contract layer (epic 3720). gen-core owns the contracts, registry, request/
// output types, and pure policy math; mlx-gen re-exports them at the historical `mlx_gen::…` paths
// below so every downstream `use mlx_gen::…` keeps compiling. Re-exported as a module too, so
// `mlx_gen::gen_core::{Error, Result}` (the neutral contract error) is reachable by name.
pub use sceneworks_gen_core as gen_core;

// Local MLX modules (tensor ops, weights, quant, samplers' tensor application, error w/ mlx variants).
pub mod adapters;
pub mod array;
pub mod error;
pub mod nn;
pub mod quant;
pub mod sampler;
pub mod scheduler;
pub mod weights;

// Split modules: contract types in gen-core, MLX impls + lifts local (caption→joycaption,
// train→kernels, tokenizer→to_arrays, image→decoded_to_image).
pub mod caption;
pub mod image;
pub mod tokenizer;
pub mod train;

// Moved-verbatim contract modules — re-exported from gen-core at their old paths.
pub mod generator {
    pub use gen_core::generator::*;
}
pub mod media {
    pub use gen_core::media::*;
}
pub mod registry {
    pub use gen_core::registry::*;
}
pub mod runtime {
    pub use gen_core::runtime::*;
}
pub mod tiling {
    pub use gen_core::tiling::*;
}
pub mod transform {
    pub use gen_core::transform::*;
}

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
pub use sampler::{
    AlphaSchedule, DiffusionSampler, FlowMatchSampler, LcmSampler, LightningSampler, TcdSampler,
};
pub use scheduler::FlowMatchEuler;
pub use tiling::{TilingConfig, VaeTiling};
pub use train::{
    LrSchedule, NetworkType, TrainOptimizer, Trainer, TrainerDescriptor, TrainingConfig,
    TrainingItem, TrainingOutput, TrainingProgress, TrainingRequest,
};
pub use transform::{
    TargetSize, Transform, TransformCapabilities, TransformDescriptor, TransformRequest,
};
