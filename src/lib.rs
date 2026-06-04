//! # mlx-gen
//!
//! Rust-native inference for generative **image and video** models on Apple
//! [MLX](https://github.com/ml-explore/mlx), built on top of `mlx-rs`.
//!
//! **Status: name reserved / work in progress — not yet usable.**
//!
//! Planned families: FLUX / FLUX.2, Qwen-Image, Z-Image (image); Wan2.2, LTX
//! (video). Adapters: LoRA, LoKr (with stacking), ControlNet.
//!
//! Architecture: a *disciplined hybrid* of the frozen Python mflux fork — see
//! [`ARCHITECTURE.md`](https://github.com/michaeltrefry/mlx-gen/blob/main/ARCHITECTURE.md).

pub mod adapters;
pub mod array;
pub mod error;
pub mod generator;
pub mod image;
pub mod media;
pub mod nn;
pub mod quant;
pub mod registry;
pub mod runtime;
pub mod sampler;
pub mod scheduler;
pub mod tiling;
pub mod tokenizer;
pub mod transform;
pub mod weights;

pub use error::{Error, Result};
pub use generator::{
    default_seed, Capabilities, Conditioning, ConditioningKind, ControlKind, GenerationOutput,
    GenerationRequest, Generator, Modality, ModelDescriptor,
};
pub use media::{AudioTrack, Image};
pub use registry::{load, load_transform, ModelRegistration, TransformRegistration};
pub use runtime::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, Precision, Progress, Quant, WeightsSource,
};
pub use sampler::{AlphaSchedule, DiffusionSampler, LcmSampler, LightningSampler, TcdSampler};
pub use scheduler::FlowMatchEuler;
pub use tiling::{TilingConfig, VaeTiling};
pub use transform::{
    TargetSize, Transform, TransformCapabilities, TransformDescriptor, TransformRequest,
};
