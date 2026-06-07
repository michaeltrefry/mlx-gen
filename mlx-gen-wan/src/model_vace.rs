//! `mlx-gen-wan` Wan-VACE model entry (`wan_vace`, epic 3040 / sc-3388, S3 / sc-3436) — the native
//! port of diffusers `WanVACEPipeline`: a control video + per-frame mask (+ optional reference
//! images) → controllable video, covering **replace_person** (the SceneWorks worker's current Wan
//! `replace_person` path) plus **pose / depth / sketch control** and **extend / video_bridge** (the
//! Wan answer to sc-3357 / sc-3385).
//!
//! VACE is **mode-agnostic at the engine boundary**, exactly like diffusers `WanVACEPipeline`: the
//! worker builds the per-mode control video + mask (replace_person = the person-region-neutralized
//! clip + the person mask; pose/depth control = the render + an all-active mask; extend/bridge = the
//! source frames at the kept positions + a generated-span mask) and passes them as one
//! [`Conditioning::ControlClip`]. The provider VAE-encodes the inactive/reactive split + unfolds the
//! mask into the 96-ch control latent ([`crate::vace::prepare_video_latents`] /
//! [`prepare_masks`](crate::vace::prepare_masks)) and runs the CFG VACE denoise loop
//! ([`denoise_vace`](crate::vace::denoise_vace)). Reference images (from [`Conditioning::Reference`])
//! are encoded to leading latent frames and dropped after denoise (diffusers
//! `latents[:, :, num_reference_images:]`).
//!
//! **Snapshot layout** (the cutover, sc-3055, converts the diffusers VACE repo into this): the VACE
//! transformer in **diffusers tensor layout** (read directly by [`WanVaceTransformer`]) at
//! `model.safetensors` or a `transformer/` shard dir, plus the shared native-converted UMT5
//! (`t5_encoder.safetensors` + `tokenizer.json`) and z16 Wan VAE (`vae.safetensors`) — the same
//! components the base Wan 14B uses. **e2e is checkpoint-gated** (no VACE checkpoint in the local HF
//! cache yet — `tests/wanvace_e2e.rs`, `#[ignore]`); the engine pieces are validated component-wise
//! (S1 transformer structural parity, S2 conditioning byte-parity).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Result,
    WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::tiling::TilingConfig;

use crate::config::WanVaceConfig;
use crate::pipeline::{align_dim, decode_to_frames, frames_to_images, preprocess_i2v_image};
use crate::scheduler::SolverKind;
use crate::text_encoder::{load_tokenizer, Umt5Encoder};
use crate::vace::{
    build_vace_control, denoise_vace, prepare_masks, prepare_video_latents, WanVaceTransformer,
};
use crate::vae::WanVae;

/// Public registry id: `mlx_gen::load("wan_vace", spec)`.
pub const MODEL_ID_VACE: &str = "wan_vace";

/// The Wan z16 VAE strides (the VACE checkpoints are Wan2.1-based): temporal 4, spatial 8, patch 2.
const VAE_T: usize = 4;
const VAE_S: usize = 8;

/// Stable identity + advertised capabilities for `wan_vace`.
pub fn descriptor_vace() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_VACE,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG (guide 5.0) + the Chinese anti-artifact negative prompt. The control input is a
            // masked control clip (`ControlClip`, the universal VACE form the worker builds per
            // mode); optional `Reference` images become leading conditioning frames.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::ControlClip, ConditioningKind::Reference],
            // VACE LoRA/LoKr + Q4/Q8 are separate capability layers (mirroring the base Wan slices
            // sc-2682/sc-2683) and need diffusers-name adapter routing / a VACE quantize pass —
            // tracked as follow-ons, not wired here.
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded Wan-VACE model. Holds the resolved config + snapshot dir; the heavy components (UMT5
/// TE, the z16 VAE, the VACE DiT) are **staged** inside [`WanVace::generate`] to bound peak memory.
pub struct WanVace {
    descriptor: ModelDescriptor,
    config: WanVaceConfig,
    root: PathBuf,
}

impl WanVace {
    /// The resolved VACE config (exposed for tests).
    pub fn config(&self) -> &WanVaceConfig {
        &self.config
    }
}

/// `mlx_gen::load("wan_vace", spec)` — resolve the VACE config from the snapshot dir.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "wan_vace: expected a model directory (converted snapshot), not a single file"
                    .into(),
            ))
        }
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan_vace: precision override is not wired (the DiT runs bf16 GEMMs over an f32 residual \
             stream — the parity regime)"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "wan_vace: LoRA/LoKr adapters are not yet wired for VACE (the diffusers-layout transformer \
             needs diffusers-name adapter routing — a tracked follow-on)"
                .into(),
        ));
    }
    let config = WanVaceConfig::from_model_dir(&root)?;
    Ok(Box::new(WanVace {
        descriptor: descriptor_vace(),
        config,
        root,
    }))
}

fn solver_kind(sampler: Option<&str>) -> SolverKind {
    match sampler {
        Some("euler") => SolverKind::Euler,
        Some("dpmpp2m") | Some("dpm++") => SolverKind::Dpmpp2m,
        _ => SolverKind::UniPC,
    }
}

/// Preprocess a list of frame [`Image`]s → a channels-first `[3, F, H, W]` clip in `[-1, 1]` (the
/// Wan VAE input convention), via the per-frame cover-fit lanczos resize + center-crop.
fn preprocess_clip(frames: &[Image], width: u32, height: u32) -> Result<Array> {
    if frames.is_empty() {
        return Err(Error::Msg("wan_vace: control clip has no frames".into()));
    }
    let planes: Vec<Array> = frames
        .iter()
        .map(|im| Ok(preprocess_i2v_image(im, width, height)?.expand_dims(1)?)) // [3,1,H,W]
        .collect::<Result<_>>()?;
    let refs: Vec<&Array> = planes.iter().collect();
    Ok(concatenate_axis(&refs, 1)?) // [3, F, H, W]
}

impl Generator for WanVace {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID_VACE, req)?;
        let clip = req.control_clip().ok_or_else(|| {
            Error::Msg(
                "wan_vace: needs a ControlClip (the masked control video — the worker builds it per \
                 mode: replace_person / pose-depth control / extend-bridge)"
                    .into(),
            )
        })?;
        if clip.frames.len() != clip.mask.len() {
            return Err(Error::Msg(format!(
                "wan_vace: control frames ({}) and mask frames ({}) length mismatch",
                clip.frames.len(),
                clip.mask.len()
            )));
        }
        // num_frames must be 1 + 4·k (one z16 VAE temporal chunk + 4× per chunk).
        if clip.frames.len() % VAE_T != 1 {
            return Err(Error::Msg(format!(
                "wan_vace: control clip frame count must be 1 + 4·k (got {})",
                clip.frames.len()
            )));
        }
        Ok(())
    }

    /// The VACE pipeline (port of diffusers `WanVACEPipeline.__call__`): stage the phases to bound
    /// memory — (1) UMT5 encode the prompt (+ neg unless CFG off); (2) load the z16 VAE, build the
    /// 96-ch control latent from the control clip + mask + reference images; (3) load the VACE DiT,
    /// run the CFG [`denoise_vace`] loop with per-vace-layer `control_hidden_states_scale`; (4) drop
    /// the reference latent frames and z16-VAE-decode → RGB8 frames.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let base = &self.config.base;
        let clip = req.control_clip().expect("validated present");

        // --- Resolve knobs ---
        let width = align_dim(req.width, base.patch_size.2, VAE_S);
        let height = align_dim(req.height, base.patch_size.1, VAE_S);
        let steps = req.steps.map(|s| s as usize).unwrap_or(base.sample_steps);
        let shift = req.scheduler_shift.unwrap_or(base.sample_shift);
        let kind = solver_kind(req.sampler.as_deref());
        let seed = req.seed.unwrap_or_else(default_seed);
        let guidance = req
            .guidance
            .unwrap_or_else(|| base.sample_guide_scale.effective());
        let cfg_disabled = guidance <= 1.0;
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| base.sample_neg_prompt.clone());

        // Control video [-1,1] + mask [0,1] (diffusers `clamp((m+1)/2)`), each [3, F, H, W].
        let control_video = preprocess_clip(clip.frames, width, height)?;
        let mask = preprocess_clip(clip.mask, width, height)?;
        let half = Array::from_slice(&[0.5f32], &[1]);
        let mask = multiply(&add(&mask, Array::from_slice(&[1.0f32], &[1]))?, &half)?; // (m+1)/2 ∈ [0,1]

        // Reference images (optional) → channels-first [3, H, W] each.
        let references: Vec<Array> = req
            .conditioning
            .iter()
            .filter_map(|c| match c {
                mlx_gen::Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .map(|im| preprocess_i2v_image(im, width, height))
            .collect::<Result<_>>()?;
        let num_ref = references.len();

        // --- Stage 1: UMT5 text encode ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), base.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, base)?;
            let context = enc.encode(&tokenizer, &req.prompt)?;
            let context_null = if cfg_disabled {
                None
            } else {
                Some(enc.encode(&tokenizer, &neg_prompt)?)
            };
            match &context_null {
                Some(cn) => mlx_rs::transforms::eval([&context, cn])?,
                None => mlx_rs::transforms::eval([&context])?,
            }
            (context, context_null)
        };

        // --- Stage 2: z16 VAE encode the control + mask → 96-ch control latent ---
        let control = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            let video_latents =
                prepare_video_latents(&vae, &control_video, Some(&mask), &references)?;
            let mask_latents = prepare_masks(&mask, VAE_T, VAE_S, base.patch_size.1, num_ref)?;
            let c = build_vace_control(&video_latents, &mask_latents)?;
            mlx_rs::transforms::eval([&c])?;
            c
        };
        // Control latent dims: [96, T_lat(+num_ref), h, w] → the noisy latent matches its frame/space.
        let csh = control.shape();
        let (t_total, h_lat, w_lat) = (csh[1], csh[2], csh[3]);
        let scales = vec![1.0f32; self.config.vace_layers.len()];

        // Seeded init noise [z16, T_lat(+num_ref), h, w].
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(
            &[base.vae_z_dim as i32, t_total, h_lat, w_lat],
            None,
            None,
            Some(&key),
        )?;

        // --- Stage 3: load the VACE DiT, embed contexts, CFG denoise ---
        let latents = {
            let w = load_vace_transformer_weights(&self.root)?;
            let dit = WanVaceTransformer::from_weights(&w, &self.config, Dtype::Bfloat16)?;
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_vace(
                &dit,
                &control,
                &scales,
                kind,
                base.num_train_timesteps,
                steps,
                shift,
                guidance,
                &context,
                context_null.as_ref(),
                &init_noise,
                &mut on_step,
            )?
        };

        // Drop the leading reference latent frames (diffusers `latents[:, :, num_reference_images:]`).
        let latents = if num_ref > 0 {
            let keep = Array::from_slice(
                &((num_ref as i32)..t_total).collect::<Vec<i32>>(),
                &[t_total - num_ref as i32],
            );
            latents.take_axis(&keep, 1)?
        } else {
            latents
        };

        // --- Stage 4: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        let out_frames = latents.shape()[1] * VAE_T as i32 - (VAE_T as i32 - 1);
        let tiling = TilingConfig::auto(height as i32, width as i32, out_frames);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref())?
        };
        let images = frames_to_images(&frames_u8)?;

        let fps = req.fps.unwrap_or(base.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}

/// Load the VACE transformer weights (diffusers layout) — a consolidated `model.safetensors` or a
/// sharded `transformer/` dir, whichever the snapshot provides.
fn load_vace_transformer_weights(root: &std::path::Path) -> Result<Weights> {
    let single = root.join("model.safetensors");
    if single.exists() {
        return Weights::from_file(single);
    }
    let shard_dir = root.join("transformer");
    if shard_dir.is_dir() {
        return Weights::from_dir(shard_dir);
    }
    Err(Error::Msg(format!(
        "wan_vace: no transformer weights at {} (expected model.safetensors or a transformer/ dir)",
        root.display()
    )))
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor: descriptor_vace, load }
}
