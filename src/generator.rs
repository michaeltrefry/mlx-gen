//! The `Generator` contract — prompt-conditioned synthesis of image **or** video (or both),
//! including multi-modal models. See `docs/MODEL_ARCHITECTURE.md` §3.1.
//!
//! One trait covers everything text→media: T2I, T2V, edit (image+text→image), LTX
//! (text→video+audio). Modality is a [`ModelDescriptor`] property plus a [`GenerationOutput`]
//! variant — *not* a per-modality trait split (which breaks on multi-modal models).

use crate::media::{AudioTrack, Image};
use crate::runtime::{CancelFlag, Progress};
use crate::{Error, Result};

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
    /// CFG-scheduling start step — the companion to [`true_cfg`](Self::true_cfg): real classifier-free
    /// guidance (and any per-branch conditioning gated with it) engages only once the denoise reaches
    /// this step, leaving earlier steps single-forward. `None` ⇒ each model's own default. Today only
    /// PuLID-FLUX honors it (default 1; its photoreal preset uses 4 to delay CFG a few steps); models
    /// without CFG scheduling ignore it.
    pub timestep_to_start_cfg: Option<u32>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub scheduler_shift: Option<f32>,

    // --- Conditioning ---
    pub conditioning: Vec<Conditioning>,
    /// img2img strength when a single `Reference` is supplied without its own strength.
    pub strength: Option<f32>,
    /// Wan-VACE control strength — the diffusers `conditioning_scale` / per-vace-layer
    /// `control_hidden_states_scale` (`hidden += proj_out(control)·scale`), broadcast to every
    /// `vace_layers` entry. `None` ⇒ the diffusers default `1.0`. Only the `wan_vace` model reads it;
    /// other models ignore it. (sc-3441)
    pub control_scale: Option<f32>,

    // --- Video (Option; consumed by video models at the follow-on port) ---
    pub frames: Option<u32>,
    pub fps: Option<u32>,
    pub duration: Option<f32>,
    pub video_mode: Option<String>,
    /// Generate this many extra leading temporal chunks (each = `vae_stride_t` latent frames) and
    /// discard them after decode, so the first *kept* frame has a full temporal receptive field of
    /// real (non-zero-padded) data — mitigates first-frame VAE/causal-conv artifacts. `None`/0 = off
    /// (the default). Consumed by Wan video models (`generate_wan.py`'s `trim_first_frames`); video
    /// models that don't support it ignore it.
    pub trim_first_frames: Option<u32>,

    // --- Prompt enhancement (LTX-2.3, sc-2845; ignored by other models) ---
    /// Rewrite `prompt` with an autoregressive Gemma-3 LLM before encoding (the reference
    /// `--enhance-prompt`). Default `false` — the diffusion path is unchanged. On any enhancer
    /// failure the model falls back to the original prompt (reference-faithful).
    pub enhance_prompt: bool,
    /// Use the separate uncensored 4-bit Gemma enhancer (`--use-uncensored-enhancer`) instead of the
    /// loaded text-encoder backbone. Only consulted when `enhance_prompt` is set.
    pub use_uncensored_enhancer: bool,
    /// Max tokens for prompt enhancement (reference default 512 when `None`).
    pub enhance_max_tokens: Option<u32>,
    /// Sampling temperature for prompt enhancement (reference default 0.7 when `None`).
    pub enhance_temperature: Option<f32>,

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
            timestep_to_start_cfg: None,
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            conditioning: Vec::new(),
            strength: None,
            control_scale: None,
            frames: None,
            fps: None,
            duration: None,
            video_mode: None,
            trim_first_frames: None,
            enhance_prompt: false,
            use_uncensored_enhancer: false,
            enhance_max_tokens: None,
            enhance_temperature: None,
            cancel: CancelFlag::default(),
        }
    }
}

/// A first_last_frame / multi-keyframe input — a borrowed, normalized view of a
/// [`Conditioning::Keyframe`]. Returned by [`GenerationRequest::keyframes`].
#[derive(Clone, Copy, Debug)]
pub struct KeyframeRef<'a> {
    pub image: &'a Image,
    pub frame_idx: i32,
    pub strength: f32,
}

/// An in-context conditioning clip — a borrowed view of a [`Conditioning::VideoClip`]. Returned by
/// [`GenerationRequest::video_clips`].
#[derive(Clone, Copy, Debug)]
pub struct VideoClipRef<'a> {
    pub frames: &'a [Image],
    pub frame_idx: i32,
    pub strength: f32,
}

/// A replace_person masked control clip — a borrowed view of a [`Conditioning::ControlClip`].
/// Returned by [`GenerationRequest::control_clip`].
#[derive(Clone, Copy, Debug)]
pub struct ControlClipRef<'a> {
    pub frames: &'a [Image],
    pub mask: &'a [Image],
    pub masking_strength: f32,
    pub start_frame: i32,
    pub mode: ReplacementMode,
}

impl GenerationRequest {
    /// All [`Conditioning::Keyframe`] inputs (first_last_frame / multi-keyframe), in request order.
    pub fn keyframes(&self) -> Vec<KeyframeRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::Keyframe {
                    image,
                    frame_idx,
                    strength,
                } => Some(KeyframeRef {
                    image,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// All [`Conditioning::VideoClip`] in-context clips (extend_clip / video_bridge), in request order.
    pub fn video_clips(&self) -> Vec<VideoClipRef<'_>> {
        self.conditioning
            .iter()
            .filter_map(|c| match c {
                Conditioning::VideoClip {
                    frames,
                    frame_idx,
                    strength,
                } => Some(VideoClipRef {
                    frames,
                    frame_idx: *frame_idx,
                    strength: *strength,
                }),
                _ => None,
            })
            .collect()
    }

    /// The replace_person masked control clip ([`Conditioning::ControlClip`]), if present. The first
    /// one wins (a request carries at most one person edit per generation).
    pub fn control_clip(&self) -> Option<ControlClipRef<'_>> {
        self.conditioning.iter().find_map(|c| match c {
            Conditioning::ControlClip {
                frames,
                mask,
                masking_strength,
                start_frame,
                mode,
            } => Some(ControlClipRef {
                frames,
                mask,
                masking_strength: *masking_strength,
                start_frame: *start_frame,
                mode: *mode,
            }),
            _ => None,
        })
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
///
/// The video families ([`Conditioning::Keyframe`] / [`Conditioning::VideoClip`] /
/// [`Conditioning::ControlClip`]) are the epic-3040 advanced-mode inputs and map onto the two LTX
/// conditioning mechanisms (see `docs/SPIKE_ADVANCED_VIDEO_3040.md`): a [`Keyframe`](Conditioning::Keyframe)
/// is **replace-latent** (overwrite the target latent at a frame index — first_last_frame); a
/// [`VideoClip`](Conditioning::VideoClip) / [`ControlClip`](Conditioning::ControlClip) is
/// **keyframe-append** (append the clip's VAE latents as extra in-context tokens — extend_clip /
/// video_bridge / replace_person, the IC-LoRA path).
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
    /// A keyframe pinned at a specific output **latent** frame index (first_last_frame / general
    /// multi-keyframe). VAE-encoded and its tokens **overwrite** the target latent at `frame_idx`
    /// with denoise mask `1 − strength` (the replace-latent mechanism — reference
    /// `VideoConditionByLatentIndex`). `strength = 1.0` fully pins the frame. first_last_frame is two
    /// of these (at `0` and the last latent frame).
    Keyframe {
        image: Image,
        frame_idx: i32,
        strength: f32,
    },
    /// An in-context conditioning **clip** (extend_clip / video_bridge — the LTX IC-LoRA path). The
    /// frames are VAE-encoded and **appended** as extra tokens at `frame_idx` (RoPE-offset on the
    /// frame axis) with denoise mask `1 − strength` (reference `VideoConditionByKeyframeIndex`).
    /// extend_clip = one clip at `frame_idx 0`; video_bridge = a left clip at `0` and a right clip at
    /// the tail.
    VideoClip {
        frames: Vec<Image>,
        frame_idx: i32,
        strength: f32,
    },
    /// A masked control clip for replace_person. `frames` is the (host-built, person-region
    /// neutralized) control clip; `mask` is the per-frame binary person mask (white = regenerate).
    /// Drives the keyframe-append in-context conditioning **plus** mask injection (force the masked
    /// region toward the re-noised source for the first `ceil(steps · masking_strength)` steps —
    /// reference `prepare_mask_injection`). Person detect/track stays in onnx and supplies these.
    ControlClip {
        frames: Vec<Image>,
        mask: Vec<Image>,
        masking_strength: f32,
        /// Output latent-frame the control clip aligns to (reference `masking_source.start_frame`).
        start_frame: i32,
        /// Replacement granularity (reference `replacement_mode`); the LTX mask path is region-driven
        /// so it is carried for the worker contract / WanVACE parity rather than changing the mask math.
        mode: ReplacementMode,
    },
}

impl Conditioning {
    /// The [`ConditioningKind`] discriminant — for capability checks / `validate()`. Centralized here
    /// so adding a [`Conditioning`] variant updates every model's validation in one place.
    pub fn kind(&self) -> ConditioningKind {
        match self {
            Conditioning::Reference { .. } => ConditioningKind::Reference,
            Conditioning::MultiReference { .. } => ConditioningKind::MultiReference,
            Conditioning::ReduxRefs { .. } => ConditioningKind::ReduxRefs,
            Conditioning::Control { .. } => ConditioningKind::Control,
            Conditioning::Depth { .. } => ConditioningKind::Depth,
            Conditioning::Mask { .. } => ConditioningKind::Mask,
            Conditioning::Keyframe { .. } => ConditioningKind::Keyframe,
            Conditioning::VideoClip { .. } => ConditioningKind::VideoClip,
            Conditioning::ControlClip { .. } => ConditioningKind::ControlClip,
        }
    }
}

/// Granularity of a replace_person edit (reference `replacement_mode`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ReplacementMode {
    /// Replace the face region only.
    #[default]
    FaceOnly,
    /// Replace the full person but keep the original outfit.
    FullPersonKeepOutfit,
    /// Replace the full person including the outfit.
    FullPersonReplaceOutfit,
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
    /// first_last_frame / multi-keyframe ([`Conditioning::Keyframe`]).
    Keyframe,
    /// extend_clip / video_bridge ([`Conditioning::VideoClip`]).
    VideoClip,
    /// replace_person ([`Conditioning::ControlClip`]).
    ControlClip,
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

    /// Reject a request that violates the **advertised** capability surface — the model-agnostic
    /// checks every `Generator::validate` shares, so a descriptor cannot promise something
    /// `validate` then silently ignores at runtime:
    ///
    /// - `count` within `1..=max_count`,
    /// - `width`/`height` within `min_size..=max_size`,
    /// - `negative_prompt` / `guidance` / `true_cfg` only when the matching `supports_*` flag is set,
    /// - `sampler` / `scheduler` (when supplied) must name an advertised entry,
    /// - every `conditioning` entry must be an [`accepts`](Self::accepts)ed kind.
    ///
    /// `id` is the model's descriptor id, used in error messages. Model-specific constraints — an
    /// empty-prompt rejection, size-alignment (multiple-of-N), frame-count divisibility,
    /// sampler→solver mapping — are layered on top by each model's own `validate`; this is the shared
    /// floor, not a replacement for them.
    pub fn validate_request(&self, id: &str, req: &GenerationRequest) -> Result<()> {
        if req.count == 0 || req.count > self.max_count {
            return Err(Error::Msg(format!(
                "{id}: count {} out of range 1..={}",
                req.count, self.max_count
            )));
        }
        if req.width < self.min_size
            || req.height < self.min_size
            || req.width > self.max_size
            || req.height > self.max_size
        {
            return Err(Error::Msg(format!(
                "{id}: size {}x{} outside supported range {}..={}",
                req.width, req.height, self.min_size, self.max_size
            )));
        }
        if req.negative_prompt.is_some() && !self.supports_negative_prompt {
            return Err(Error::Msg(format!(
                "{id}: negative prompts are not supported"
            )));
        }
        if req.guidance.is_some() && !self.supports_guidance {
            return Err(Error::Msg(format!("{id}: guidance is not supported")));
        }
        if req.true_cfg.is_some() && !self.supports_true_cfg {
            return Err(Error::Msg(format!("{id}: true_cfg is not supported")));
        }
        if let Some(s) = &req.sampler {
            if !self.samplers.contains(&s.as_str()) {
                return Err(Error::Msg(format!(
                    "{id}: unsupported sampler {s:?} (supported: {:?})",
                    self.samplers
                )));
            }
        }
        if let Some(s) = &req.scheduler {
            if !self.schedulers.contains(&s.as_str()) {
                return Err(Error::Msg(format!(
                    "{id}: unsupported scheduler {s:?} (supported: {:?})",
                    self.schedulers
                )));
            }
        }
        for c in &req.conditioning {
            let kind = c.kind();
            if !self.accepts(kind) {
                return Err(Error::Msg(format!(
                    "{id}: {kind:?} conditioning is not supported"
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 3) as usize],
        }
    }

    #[test]
    fn keyframes_accessor_collects_in_order() {
        // first_last_frame: two keyframes at 0 and the last latent frame.
        let req = GenerationRequest {
            conditioning: vec![
                Conditioning::Keyframe {
                    image: img(2, 2),
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::Reference {
                    image: img(2, 2),
                    strength: None,
                },
                Conditioning::Keyframe {
                    image: img(4, 4),
                    frame_idx: 8,
                    strength: 0.75,
                },
            ],
            ..Default::default()
        };
        let kf = req.keyframes();
        assert_eq!(kf.len(), 2);
        assert_eq!((kf[0].frame_idx, kf[0].strength), (0, 1.0));
        assert_eq!((kf[1].frame_idx, kf[1].strength), (8, 0.75));
        assert_eq!((kf[1].image.width, kf[1].image.height), (4, 4));
        // Reference is not a keyframe and is not a video clip / control clip.
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
    }

    #[test]
    fn video_clips_accessor_collects_clips() {
        // video_bridge: left clip @0, right clip @tail.
        let req = GenerationRequest {
            conditioning: vec![
                Conditioning::VideoClip {
                    frames: vec![img(2, 2), img(2, 2)],
                    frame_idx: 0,
                    strength: 1.0,
                },
                Conditioning::VideoClip {
                    frames: vec![img(2, 2)],
                    frame_idx: 24,
                    strength: 0.9,
                },
            ],
            ..Default::default()
        };
        let clips = req.video_clips();
        assert_eq!(clips.len(), 2);
        assert_eq!((clips[0].frames.len(), clips[0].frame_idx), (2, 0));
        assert_eq!((clips[1].frames.len(), clips[1].frame_idx), (1, 24));
        assert!(req.keyframes().is_empty());
    }

    #[test]
    fn control_clip_accessor_returns_first() {
        let req = GenerationRequest {
            conditioning: vec![Conditioning::ControlClip {
                frames: vec![img(2, 2), img(2, 2)],
                mask: vec![img(2, 2), img(2, 2)],
                masking_strength: 0.8,
                start_frame: 0,
                mode: ReplacementMode::FaceOnly,
            }],
            ..Default::default()
        };
        let cc = req.control_clip().expect("control clip present");
        assert_eq!((cc.frames.len(), cc.mask.len()), (2, 2));
        assert_eq!(cc.masking_strength, 0.8);
        assert_eq!(cc.mode, ReplacementMode::FaceOnly);
    }

    #[test]
    fn accessors_empty_by_default() {
        let req = GenerationRequest::default();
        assert!(req.keyframes().is_empty());
        assert!(req.video_clips().is_empty());
        assert!(req.control_clip().is_none());
    }

    /// A capability surface that turns nothing extra on: a single 256..=1024 image, no
    /// negative/guidance/true_cfg, no samplers/schedulers, only `Reference` conditioning.
    fn caps() -> Capabilities {
        Capabilities {
            conditioning: vec![ConditioningKind::Reference],
            samplers: vec!["euler"],
            min_size: 256,
            max_size: 1024,
            max_count: 1,
            ..Default::default()
        }
    }

    fn base_req() -> GenerationRequest {
        GenerationRequest {
            prompt: "x".into(),
            width: 512,
            height: 512,
            ..Default::default()
        }
    }

    #[test]
    fn validate_request_accepts_in_surface() {
        let c = caps();
        assert!(c.validate_request("m", &base_req()).is_ok());
        // An advertised sampler + an accepted conditioning kind are fine.
        assert!(c
            .validate_request(
                "m",
                &GenerationRequest {
                    sampler: Some("euler".into()),
                    conditioning: vec![Conditioning::Reference {
                        image: img(8, 8),
                        strength: None,
                    }],
                    ..base_req()
                }
            )
            .is_ok());
    }

    #[test]
    fn validate_request_enforces_advertised_surface() {
        let c = caps();
        let cases: Vec<GenerationRequest> = vec![
            // count out of range
            GenerationRequest {
                count: 0,
                ..base_req()
            },
            GenerationRequest {
                count: 2,
                ..base_req()
            },
            // size out of range (below min, above max)
            GenerationRequest {
                width: 128,
                ..base_req()
            },
            GenerationRequest {
                height: 2048,
                ..base_req()
            },
            // capability flags not advertised
            GenerationRequest {
                negative_prompt: Some("n".into()),
                ..base_req()
            },
            GenerationRequest {
                guidance: Some(3.5),
                ..base_req()
            },
            GenerationRequest {
                true_cfg: Some(4.0),
                ..base_req()
            },
            // sampler / scheduler not advertised
            GenerationRequest {
                sampler: Some("unipc".into()),
                ..base_req()
            },
            GenerationRequest {
                scheduler: Some("linear".into()),
                ..base_req()
            },
            // conditioning kind not accepted
            GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img(8, 8) }],
                ..base_req()
            },
        ];
        for (i, req) in cases.iter().enumerate() {
            assert!(
                c.validate_request("m", req).is_err(),
                "case {i} should have been rejected"
            );
        }
    }
}
