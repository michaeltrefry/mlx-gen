//! `KolorsGenerator` — the [`mlx_gen::Generator`] impl for Kolors, plus its [`descriptor`]/[`load`]
//! entry points and the `inventory` registration that wires it into `mlx_gen`'s registry under the
//! id `"kolors"` (sc-3874).
//!
//! The epic-3090 ports (sc-3091–3098) gave [`crate::Kolors`] the full capability surface but only as
//! a direct struct API (which the parity tests call). This module makes Kolors **dispatchable** —
//! `mlx_gen::load("kolors", spec).generate(req)`, the SceneWorks worker's in-process entry — by
//! mapping [`LoadSpec`]/[`GenerationRequest`] onto that API and looping `req.count` with per-image
//! seeds + cancel + streamed progress, mirroring `mlx-gen-sdxl/src/model.rs`.
//!
//! **Registration mechanism:** `inventory::submit!` here is collected by `mlx_gen`'s
//! `inventory::collect!` at *link* time — so the registration activates whenever a consumer (the
//! worker, or this crate's own test binary) links `mlx-gen-kolors`. The core `mlx-gen` crate does
//! **not** depend on the model crates (by design); there is no root-crate dependency to add.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Dtype};

use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, ControlKind, DiffusionSampler,
    Error, GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality,
    ModelDescriptor, ModelRegistration, Progress, Result, WeightsSource,
};

use mlx_gen_sdxl::{
    decode_image, denoise, denoise_control, denoise_ip, encode_init_latents, load_controlnet,
    preprocess_control_image, ControlContext, ControlNet, Denoiser, IpImageEncoder,
};

use crate::ip_adapter::load_kolors_ip_adapter;
use crate::model::{kolors_time_ids, DEFAULT_IMG2IMG_STRENGTH, SPATIAL_SCALE};
use crate::sampler::{KolorsEulerSampler, NUM_TRAIN_TIMESTEPS};
use crate::Kolors;

/// Registry id — the SceneWorks worker's `payload.model` for the Kolors family.
pub const MODEL_ID: &str = "kolors";

/// diffusers `KolorsPipeline` production defaults: 50 inference steps, CFG 5.0.
const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 5.0;
/// Default IP-Adapter scale when a request doesn't override it (carried on the `Reference` strength
/// field in IP mode, mirroring the SDXL IP-Adapter convention).
const IP_DEFAULT_SCALE: f32 = 0.6;
/// The single Kolors sampler — diffusers `EulerDiscreteScheduler` (leading), see [`KolorsEulerSampler`].
const SAMPLER: &str = "euler_discrete";

/// Kolors' identity + capabilities — constructible without loading weights (registry
/// introspection). Advertises **only** the wired + parity-proven surface (the false-capability
/// guard): T2I + img2img (`Reference`) + ControlNet-pose (`Control`) + IP-Adapter (`Reference` in
/// IP mode) + Q8/Q4. **LoRA/LoKr is NOT advertised** — epic 3090 did not port Kolors adapters
/// (sc-3874 note); wiring them is a tracked follow-on, so `load` rejects `spec.adapters` rather than
/// silently dropping them.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "kolors",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Kolors uses real classifier-free guidance over the ChatGLM3 conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // Reference = img2img init (sc-3095) OR the IP-Adapter image prompt when an IP-Adapter is
            // loaded (sc-3098); Control = the Kolors ControlNet-pose branch (sc-3097).
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Control],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec![SAMPLER],
            schedulers: vec!["discrete"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded, dispatchable Kolors generator: the [`Kolors`] pipeline plus the optionally-loaded
/// ControlNet branch and IP-Adapter image-token encoder (the decoupled-attn K/V pairs are already
/// installed into the U-Net at load).
pub struct KolorsGenerator {
    descriptor: ModelDescriptor,
    kolors: Kolors,
    control: Option<ControlNet>,
    ip_encoder: Option<IpImageEncoder>,
}

/// Build a [`KolorsGenerator`] from a [`LoadSpec`].
///
/// `spec.weights` is a `Kwai-Kolors/Kolors-diffusers` snapshot dir (the multi-component tree with
/// the materialized `tokenizer/tokenizer.json`). Dense runs **fp16** (the SDXL-family production
/// dtype; the VAE stays f32 via `load_vae`). `spec.quantize` ⇒ load-time Q8/Q4 (sc-3096);
/// `spec.control` ⇒ the Kolors ControlNet-Pose checkpoint (sc-3097); `spec.ip_adapter` ⇒ the
/// Kolors-IP-Adapter-Plus snapshot dir (sc-3098), whose K/V pairs are installed into the (pre-quant)
/// U-Net. `spec.adapters` (LoRA/LoKr) is rejected — not ported (sc-3874).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // fp16 dense path (SDXL-family production dtype). `Precision` is the registry's dense sentinel;
    // a precision override is not wired (the VAE is always f32, the rest fp16), so reject it rather
    // than silently ignore.
    if spec.precision != mlx_gen::Precision::Bf16 {
        return Err(Error::Msg(
            "kolors: precision override is not wired; the dense path runs fp16 (SDXL-family \
             production dtype) — drop the precision override"
                .into(),
        ));
    }
    let dtype = Dtype::Float16;
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "kolors expects a Kolors-diffusers snapshot directory (text_encoder/ tokenizer/ \
                 unet/ vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "kolors: LoRA/LoKr adapters are not wired (epic 3090 did not port Kolors adapters; \
             tracked as a follow-on on sc-3874) — load without `adapters`"
                .into(),
        ));
    }

    let mut kolors = match spec.quantize {
        Some(q) => Kolors::load_quantized(&root, dtype, q.bits())?,
        None => Kolors::load(&root, dtype)?,
    };

    let control = match &spec.control {
        Some(src) => Some(load_controlnet(src, dtype)?),
        None => None,
    };

    let ip_encoder =
        match &spec.ip_adapter {
            Some(WeightsSource::Dir(p)) => {
                let (enc, pairs) = load_kolors_ip_adapter(p, dtype)?;
                kolors.install_ip_adapter(pairs)?;
                Some(enc)
            }
            Some(WeightsSource::File(_)) => return Err(Error::Msg(
                "kolors ip_adapter expects a Kolors-IP-Adapter-Plus snapshot directory, not a file"
                    .into(),
            )),
            None => None,
        };

    Ok(Box::new(KolorsGenerator {
        descriptor: descriptor(),
        kolors,
        control,
        ip_encoder,
    }))
}

impl Generator for KolorsGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)?;
        // Mode-combination guards (the Kolors paths are mutually exclusive in this build).
        let has_ref = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let has_ctrl = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }));
        if has_ctrl && self.control.is_none() {
            return Err(Error::Msg(
                "kolors: a Control conditioning was passed but the model was loaded without a \
                 ControlNet (set LoadSpec::control)"
                    .into(),
            ));
        }
        if has_ctrl && has_ref {
            return Err(Error::Msg(
                "kolors: combining ControlNet (Control) with a Reference (img2img / IP) is not \
                 supported in this build"
                    .into(),
            ));
        }
        if self.ip_encoder.is_some() && has_ctrl {
            return Err(Error::Msg(
                "kolors: IP-Adapter + ControlNet in one pass is not supported in this build".into(),
            ));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let cfg = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let cfg_on = cfg > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let (h, w) = (req.height as i32, req.width as i32);
        let dtype = self.kolors.dtype();
        let ip_mode = self.ip_encoder.is_some();

        let reference = self.resolve_reference(req)?;
        let control = self.resolve_control(req)?;
        if ip_mode && reference.is_none() {
            return Err(Error::Msg(
                "kolors: an IP-Adapter is loaded but no Reference image was provided (the Reference \
                 is the image prompt in IP mode)"
                    .into(),
            ));
        }

        // Conditioning is seed-independent — encode the prompts once. CFG batch order [pos, neg]
        // (`mlx_gen_sdxl::denoise` reads row 0 as cond, row 1 as uncond).
        let pos = self.kolors.encode(&req.prompt)?;
        let neg = self.kolors.encode(negative)?;
        let conditioning = concatenate_axis(&[&pos.0, &neg.0], 0)?;
        let pooled = concatenate_axis(&[&pos.1, &neg.1], 0)?;
        let time_ids = kolors_time_ids(2, h, w);

        // Build the (seed-independent) ControlNet context once.
        let control_ctx = match control {
            Some((image, scale)) => {
                let cn = self.control.as_ref().expect("validated above");
                let img = preprocess_control_image(image, w as u32, h as u32)?;
                let img = if cfg_on {
                    concatenate_axis(&[&img, &img], 0)?
                } else {
                    img
                };
                Some(ControlContext {
                    controlnet: cn,
                    // Precompute the step-invariant conditioning embedding once per run (F-069).
                    cond_embed: cn.embed_cond(&img)?,
                    scale,
                })
            }
            None => None,
        };

        // IP-Adapter image tokens (seed-independent) — CFG-batched [tokens, zeros].
        let ip = match (ip_mode, reference) {
            (true, Some((image, strength))) => {
                let tokens = self.ip_encoder.as_ref().unwrap().tokens(image)?;
                let sh = tokens.shape();
                let zeros = mlx_rs::ops::zeros::<f32>(sh)?.as_dtype(tokens.dtype())?;
                let batched = concatenate_axis(&[&tokens, &zeros], 0)?;
                let scale = strength.unwrap_or(IP_DEFAULT_SCALE);
                Some((batched, scale))
            }
            _ => None,
        };
        // img2img only when a Reference is present AND we're not in IP mode.
        let img2img = match (ip_mode, reference) {
            (false, Some((image, strength))) => Some((
                image,
                strength
                    .or(req.strength)
                    .unwrap_or(DEFAULT_IMG2IMG_STRENGTH),
            )),
            _ => None,
        };

        let (lh, lw) = (h / SPATIAL_SCALE, w / SPATIAL_SCALE);
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            random::seed(seed)?;

            // Per-mode sampler + init latents.
            let (latents, sampler) = if let Some((image, strength)) = img2img {
                let sampler = KolorsEulerSampler::kolors_img2img(steps, strength, dtype)?;
                let x0 = encode_init_latents(self.kolors.vae(), image, w as u32, h as u32)?;
                let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
                (sampler.add_noise(&x0, &noise)?, sampler)
            } else {
                let sampler = KolorsEulerSampler::kolors(steps, dtype)?;
                let noise = random::normal::<f32>(&[1, lh, lw, 4], None, None, None)?;
                (sampler.scale_initial_noise(&noise)?, sampler)
            };

            let d = Denoiser {
                unet: self.kolors.unet(),
                sampler: &sampler,
            };
            let latents = if let Some(cc) = &control_ctx {
                denoise_control(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    cc,
                )?
            } else if let Some((tokens, scale)) = &ip {
                denoise_ip(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    tokens,
                    *scale,
                )?
            } else {
                denoise(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                )?
            };

            on_progress(Progress::Decoding);
            images.push(decode_image(self.kolors.vae(), &latents)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

impl KolorsGenerator {
    /// The single img2img / IP reference image + its strength (the per-reference strength wins). One
    /// reference only; more than one is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "kolors: multiple reference images are not supported".into(),
                    ));
                }
                reference = Some((image, *strength));
            }
        }
        Ok(reference)
    }

    /// The single ControlNet control image + `conditioning_scale`. One control branch only; the
    /// Kolors ControlNet is pose-trained, so a non-pose `ControlKind` is rejected.
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Option<(&'a Image, f32)>> {
        let mut control = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, kind, scale } = c {
                if control.is_some() {
                    return Err(Error::Msg(
                        "kolors: multiple control images are not supported".into(),
                    ));
                }
                if !matches!(kind, ControlKind::Pose) {
                    return Err(Error::Msg(format!(
                        "kolors: only Pose ControlNet is wired (got {kind:?})"
                    )));
                }
                control = Some((image, *scale));
            }
        }
        Ok(control)
    }
}

/// Capability-driven request validation (unit-testable without loaded weights).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    // Shared capability contract: count/size range, negative_prompt/guidance/true_cfg support,
    // sampler, scheduler, and conditioning kinds. Delegating to core keeps Kolors from drifting
    // out of sync with the descriptor (F-132); this was previously a hand-rolled copy that had
    // already lost the negative_prompt/guidance/true_cfg/scheduler checks.
    caps.validate_request(MODEL_ID, req)?;

    // Kolors-specific checks layered on top of the shared contract:
    if req.prompt.is_empty() {
        return Err(Error::Msg("kolors: prompt must not be empty".into()));
    }
    // `steps == 0` divides by zero in `KolorsEulerSampler::new` (`num_train_timesteps / num_steps`),
    // and `steps > 1100` (the train-timestep count) makes `step_ratio == 0` so every timestep
    // collapses to 1 — silent garbage. Reject both at the request boundary (F-124). `None` falls back
    // to DEFAULT_STEPS.
    if let Some(steps) = req.steps {
        if steps == 0 || steps as usize > NUM_TRAIN_TIMESTEPS {
            return Err(Error::Msg(format!(
                "kolors: steps must be in 1..={NUM_TRAIN_TIMESTEPS} (got {steps})"
            )));
        }
    }
    // Kolors VAE downsamples by 8; non-multiple-of-8 dims would mismatch latent shapes.
    if !req.width.is_multiple_of(8) || !req.height.is_multiple_of(8) {
        return Err(Error::Msg(format!(
            "kolors: width/height must be multiples of 8 (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_kolors() {
        let d = descriptor();
        assert_eq!(d.id, "kolors");
        assert_eq!(d.family, "kolors");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(!d.capabilities.supports_lora, "Kolors LoRA is not ported");
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(!d.capabilities.accepts(ConditioningKind::Mask));
    }

    #[test]
    fn registered_in_inventory() {
        // The `inventory::submit!` above is linked into this test binary, so `mlx_gen::load`
        // resolves "kolors" (and fails on the bogus weights dir) — proving registration without
        // needing the real snapshot. A wrong/missing registration yields the registry's
        // "no generator registered for id" error instead.
        let spec = LoadSpec {
            weights: WeightsSource::Dir("/nonexistent/kolors".into()),
            quantize: None,
            precision: mlx_gen::Precision::Bf16,
            control: None,
            ip_adapter: None,
            adapters: Vec::new(),
            extra_controls: Vec::new(),
        };
        let err = match mlx_gen::load("kolors", &spec) {
            Ok(_) => panic!("bogus weights dir must fail to load"),
            Err(e) => e.to_string(),
        };
        assert!(
            !err.contains("no generator registered"),
            "kolors should resolve in the registry; got: {err}"
        );
    }

    #[test]
    fn validate_rejects_bad_steps() {
        // F-124: `steps == 0` would divide by zero in the sampler; `steps > NUM_TRAIN_TIMESTEPS`
        // collapses every timestep to 1. Both must be rejected at the request boundary; `None` and an
        // in-range count pass.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        for bad in [Some(0), Some(NUM_TRAIN_TIMESTEPS as u32 + 1)] {
            let req = GenerationRequest {
                steps: bad,
                ..base.clone()
            };
            let err = validate_request(&caps, &req).unwrap_err().to_string();
            assert!(err.contains("steps must be in"), "steps={bad:?} got: {err}");
        }
        for ok in [None, Some(1), Some(50), Some(NUM_TRAIN_TIMESTEPS as u32)] {
            let req = GenerationRequest {
                steps: ok,
                ..base.clone()
            };
            assert!(validate_request(&caps, &req).is_ok(), "steps={ok:?}");
        }
    }

    #[test]
    fn sampler_rejects_zero_steps() {
        // The defensive guard in `KolorsEulerSampler::new` (reached via `kolors`) returns a typed error
        // rather than panicking on the divide-by-zero (F-124).
        let err = match KolorsEulerSampler::kolors(0, mlx_rs::Dtype::Float32) {
            Ok(_) => panic!("num_steps == 0 must error, not build a sampler"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("num_steps must be >= 1"), "got: {err}");
    }

    #[test]
    fn validate_delegates_to_core_capability_checks() {
        // F-132: `validate_request` now delegates the shared contract to `Capabilities::validate_request`
        // rather than re-implementing it. Assert the checks the hand-rolled copy had dropped now fire:
        // an unsupported scheduler and a `true_cfg` the descriptor doesn't advertise.
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };

        let bad_scheduler = GenerationRequest {
            scheduler: Some("ddim".into()),
            ..base.clone()
        };
        assert!(
            validate_request(&caps, &bad_scheduler).is_err(),
            "unsupported scheduler must be rejected (delegated to core)"
        );

        let bad_true_cfg = GenerationRequest {
            true_cfg: Some(4.0),
            ..base.clone()
        };
        assert!(
            validate_request(&caps, &bad_true_cfg).is_err(),
            "true_cfg must be rejected — Kolors advertises supports_true_cfg=false"
        );

        // The advertised scheduler still passes.
        let good = GenerationRequest {
            scheduler: Some("discrete".into()),
            ..base
        };
        assert!(validate_request(&caps, &good).is_ok());
    }

    #[test]
    fn rejects_lora_adapters() {
        // LoRA is intentionally unwired (sc-3874 follow-on); load must reject, not silently drop.
        // (Validated at the load layer; here we assert the descriptor doesn't advertise it.)
        assert!(!descriptor().capabilities.supports_lora);
        assert!(!descriptor().capabilities.supports_lokr);
    }
}
