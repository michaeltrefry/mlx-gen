//! The `Generator` contract — prompt-conditioned synthesis of image **or** video (or both),
//! including multi-modal models. See `docs/MODEL_ARCHITECTURE.md` §3.1.
//!
//! One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX
//! (text→video+audio). Modality is a [`ModelDescriptor`] property plus a [`GenerationOutput`]
//! variant — *not* a per-modality trait split (which breaks on multi-modal models).

use crate::media::{AudioTrack, Image};
use crate::runtime::{CancelFlag, Progress};
use crate::Result;

/// A prompt-conditioned media generator. `generate` is **synchronous** (long/blocking; the
/// worker runs each job on its own thread); the request carries a cancel flag and
/// `on_progress` streams step/decode progress.
pub trait Generator {
    /// Identity + capabilities + modality (drives `validate` and consumer UI introspection).
    fn descriptor(&self) -> &ModelDescriptor;

    /// Reject a request this model cannot serve (unsupported conditioning, guidance on a
    /// distilled model, out-of-range size/count, …) before doing expensive work.
    fn validate(&self, req: &GenerationRequest) -> Result<()>;

    /// Run generation to completion (or until `req.cancel` trips).
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput>;
}

/// What a [`Generator`] produced. The `Video` variant's `audio` is `Some` for LTX (always
/// audio) and `None` for Wan — no contract change needed across the two.
#[derive(Clone, Debug)]
pub enum GenerationOutput {
    Images(Vec<Image>),
    Video {
        frames: Vec<Image>,
        fps: u32,
        audio: Option<AudioTrack>,
    },
}

/// The request union (lifted from the SceneWorks worker's `ImageRequest`/`VideoRequest`). Most
/// fields are optional; a model reads what it supports and `validate()` rejects the rest. A
/// single `Default`-able struct (no builder): `GenerationRequest { prompt, ..Default::default() }`.
#[derive(Clone, Debug)]
pub struct GenerationRequest {
    // --- Core ---
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    /// Number of images to produce (1..=8 for image models).
    pub count: u32,

    // --- Sampling (all optional; model/descriptor supply defaults) ---
    pub seed: Option<u64>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub true_cfg: Option<f32>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,

    // --- Conditioning ---
    pub conditioning: Vec<Conditioning>,
    /// img2img strength when a single `Reference` is supplied without its own strength.
    pub strength: Option<f32>,

    // --- Video (Option; consumed by video models at the follow-on port) ---
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub duration: Option<f32>,
    pub video_mode: Option<String>,

    // --- Control ---
    pub cancel: CancelFlag,
}

impl Default for GenerationRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative_prompt: None,
            width: 1024,
            height: 1024,
            count: 1,
            seed: None,
            steps: None,
            guidance: None,
            true_cfg: None,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            conditioning: Vec::new(),
            strength: None,
            frames: None,
            fps: None,
            duration: None,
            video_mode: None,
            cancel: CancelFlag::default(),
        }
    }
}

/// Seed when a [`GenerationRequest`] omits one: nanos since the epoch (any nonzero value works —
/// this only sets which sample is drawn; a caller wanting reproducibility passes `req.seed`).
/// Shared by every generator (F-006).
pub fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// Typed conditioning inputs. Each image family uses the subset its `Capabilities` advertises.
/// (A `VideoClip { clips: Vec<(MediaRef, f32)> }` variant is a known additive extension for the
/// video port — extend/bridge/replace-person — and does not exist yet.)
#[derive(Clone, Debug)]
pub enum Conditioning {
    /// img2img / IP-Adapter / identity reference.
    Reference { image: Image, strength: Option<f32> },
    /// Multiple references with no per-image strength (Qwen-Image-Edit).
    MultiReference { images: Vec<Image> },
    /// FLUX.1-Redux references, each with its own strength.
    ReduxRefs { refs: Vec<(Image, f32)> },
    /// ControlNet / pose conditioning.
    Control {
        image: Image,
        kind: ControlKind,
        scale: f32,
    },
    /// FLUX.1-Depth.
    Depth { image: Image },
    /// FIBO-Edit / inpaint mask.
    Mask { image: Image },
}

/// The control signal carried by [`Conditioning::Control`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlKind {
    Pose,
    Canny,
    Depth,
    Other(String),
}

/// Which [`Conditioning`] variants a model accepts — for capability introspection + validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditioningKind {
    Reference,
    MultiReference,
    ReduxRefs,
    Control,
    Depth,
    Mask,
}

/// What kind of media a model emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Modality {
    Image,
    Video,
    Both,
}

/// A model's stable identity + advertised capabilities. Returned by `descriptor()` and also
/// constructible without loading weights (registry introspection).
#[derive(Clone, Debug)]
pub struct ModelDescriptor {
    pub id: &'static str,
    pub family: &'static str,
    pub modality: Modality,
    pub capabilities: Capabilities,
}

/// What a model supports — drives `validate()` and consumer UI. `Default` is "supports
/// nothing"; a model turns on what it offers (`Capabilities { supports_guidance: true,
/// ..Default::default() }`).
#[derive(Clone, Debug, Default)]
pub struct Capabilities {
    pub supports_negative_prompt: bool,
    pub supports_guidance: bool,
    pub supports_true_cfg: bool,
    pub conditioning: Vec<ConditioningKind>,
    pub supports_lora: bool,
    pub supports_lokr: bool,
    pub samplers: Vec<&'static str>,
    pub schedulers: Vec<&'static str>,
    pub min_size: u32,
    pub max_size: u32,
    pub max_count: u32,
    pub mac_only: bool,
    // Loader hints.
    pub supports_kv_cache: bool,
    pub requires_sigma_shift: bool,
}

impl Capabilities {
    /// Whether this model accepts the given conditioning kind.
    pub fn accepts(&self, kind: ConditioningKind) -> bool {
        self.conditioning.contains(&kind)
    }
}
