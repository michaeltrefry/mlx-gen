//! `ZImageTurbo` — the Z-Image-turbo implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry.
//!
//! [`load`] assembles the full model from a `Tongyi-MAI/Z-Image-Turbo` snapshot directory (see
//! [`crate::loader`]) — tokenizer, Qwen text encoder, DiT transformer, VAE decoder — and
//! [`ZImageTurbo::generate`] runs the complete prompt→image pipeline: tokenize → encode →
//! seeded noise → flow-match Euler denoise over the DiT → VAE decode → RGB8. The chain is
//! parity-proven against the frozen Python fork on real bf16 weights (sc-2352).

use mlx_gen::array::host_i32;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, FlowMatchEuler,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::Dtype;

use crate::loader;
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, decoded_to_image, denoise_with_progress,
    encode_init_latents, init_time_step, slice_valid, unpack_latents,
};
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Z-Image-turbo is guidance-distilled to a fixed 4-step schedule; used when a request omits
/// `steps`. (`pub(crate)` so the ControlNet variant shares the same default.)
pub(crate) const DEFAULT_STEPS: u32 = 4;

/// Flow-match time-shift for Z-Image-Turbo: the model's own published schedule from
/// `scheduler/scheduler_config.json` (`FlowMatchEulerDiscreteScheduler`, `shift=3.0`,
/// `use_dynamic_shifting=false`) — static, resolution-independent.
///
/// **Deliberate choice (sc-2536; Michael, 2026-06-01) — do NOT "restore" `linear`.** The mflux
/// MLX path this port replaces (`MlxZImageAdapter` → `generate_image`'s default `linear`
/// scheduler) actually uses a *dynamic*, resolution-dependent shift (≈3.16 @1024², 1.88 @512²,
/// 25 @2048²). We use the model's static 3.0 instead: A/B renders (`tools/compare_z_image_
/// schedulers.py`) are visually identical at 1024² and only differ at lower resolutions, where
/// 3.0 reads slightly crisper — the preferred look. So 3.0 is an intentional, model-config-backed
/// deviation from the MLX path, not drift.
///
/// (The *original* port's bug — replaced by sc-2536 — was using `FlowMatchEuler::for_image`'s
/// empirical per-step `mu`, the *full* Z-Image model's scheduler, ≈shift 10. That was wrong;
/// `linear` and 3.0 are both reasonable, empirical-`mu` was not.)
///
/// `pub(crate)` so the ControlNet variant ([`crate::model_control`]) uses the identical schedule —
/// it is the same base turbo model, and the parity golden holds the schedule fixed on both sides.
pub(crate) const SCHEDULE_SHIFT: f32 = 3.0;

/// Registry id for Z-Image-turbo (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "z_image_turbo";

/// Z-Image-turbo's identity + capabilities — constructible without loading weights (registry
/// introspection). Values are conservative-but-real; sampler/scheduler lists fill in with the
/// scheduler port.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            // Turbo is guidance-distilled: no CFG, no negative prompt.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // img2img reference; ControlNet is a separate variant (sc-2349).
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded Z-Image-turbo generator: the four model components assembled from a snapshot
/// directory, plus the cached descriptor.
pub struct ZImageTurbo {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: TextEncoder,
    transformer: ZImageTransformer,
    vae: Vae,
}

/// Construct a [`ZImageTurbo`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Tongyi-MAI/Z-Image-Turbo`
/// snapshot (the diffusers multi-component tree — `tokenizer/`, `text_encoder/`, `transformer/`,
/// `vae/`). Weights load dense at their on-disk dtype (bf16); the text encoder promotes to f32
/// internally. `spec.quantize` (Q4/Q8) quantizes the **whole model** — transformer, text encoder,
/// and VAE (group_size 64) — after the dense load, matching the mflux fork's `nn.quantize` over
/// every quantizable Linear (plus the text encoder's token Embedding) so a Q4/Q8 consumer gets the
/// full memory saving and fork-matching output (sc-2532). An fp32 precision override is not wired
/// (the validated dense path is bf16) and is rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "z_image_turbo: only dense bf16 is wired in the Rust port; the text encoder already \
             runs f32 internally (drop the precision override)"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "z_image_turbo expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    // Q4/Q8 quantizes the **whole model** in place after the dense bf16 load — the fork's
    // `nn.quantize` over (transformer, text_encoder, vae), group_size 64, every quantizable Linear
    // (+ the text encoder's token Embedding). This is what "quantize to Q4/Q8" means everywhere
    // (mflux/diffusers/mlx-lm): a public consumer asking for Q4 gets the full memory saving and an
    // output that matches the fork — not a transformer-only partial quantization (sc-2532).
    let mut transformer = loader::load_transformer(root)?;
    let mut text_encoder = loader::load_text_encoder(root)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // LoRA/LoKr (sc-2602): applied after quantization, as forward-time residuals over the
    // (possibly quantized) base — fork-faithful (the fork applies adapters in its initializer over
    // the quantized model). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
    }
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder,
        transformer,
        vae,
    }))
}

impl ZImageTurbo {
    /// Prompt → `cap_feats` (f32): tokenize with the Qwen chat template, run the text encoder,
    /// and slice off the padded tail to the valid caption tokens.
    fn encode_prompt(&self, prompt: &str) -> Result<mlx_rs::Array> {
        let t = self.tokenizer.tokenize(prompt)?;
        // An empty prompt tokenizes to a `[1, 0]` array. Guard on shape before any host readback:
        // `as_slice`/`host_i32` on a size-0 array would otherwise make the intended error below
        // unreachable (Qwen T2I guards the same way). `validate_request` already rejects this, so
        // this is defense-in-depth at the encode boundary.
        if t.input_ids.shape()[1] == 0 {
            return Err(Error::Msg("z_image_turbo: empty prompt".into()));
        }
        let num_valid: i32 = host_i32(&t.attention_mask)?.iter().sum();
        if num_valid == 0 {
            return Err(Error::Msg("z_image_turbo: empty prompt".into()));
        }
        let enc = self.text_encoder.forward(&t.input_ids, &t.attention_mask)?;
        slice_valid(&enc, num_valid)
    }

    /// Extract the single img2img init image + its strength from the request's conditioning. The
    /// per-reference strength wins over `req.strength`. Z-Image img2img conditions on exactly one
    /// init image, so more than one `Reference` is an error (multi-image is `MultiReference`, which
    /// this model doesn't advertise).
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "z_image_turbo: multiple reference images are not supported (single \
                         img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }
}

impl Generator for ZImageTurbo {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        let reference = self.resolve_reference(req)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // Prompt → cap_feats (f32). txt2img runs the DiT in bf16 (the parity-proven path); img2img
        // matches the fork's f32 init latents, so keep cap f32 too (so the unified stream is one
        // dtype). The DiT promotes per-op against the bf16 weights either way.
        let cap = self.encode_prompt(&req.prompt)?;
        let cap = if is_img2img {
            cap
        } else {
            // PARITY-BF16 (sc-2609): round the text embeddings to bf16 to match the fork's golden;
            // f32 is more accurate (sharper). Flip to f32 for quality once parity is not the goal.
            cap.as_dtype(Dtype::Bfloat16)?
        };

        // Static shift=3.0 schedule (the model's scheduler_config.json), resolution- and
        // seed-independent — build it once. See SCHEDULE_SHIFT.
        let scheduler = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // Distinct seed per image in a batch (the fork's `seed + i` convention).
            let seed = base_seed.wrapping_add(i as u64);
            // Seeded noise as bf16 (the fork's `create_noise` casts to model precision).
            // PARITY-BF16 (sc-2609): bf16 to match the fork's seed→image mapping; flipping to f32 is
            // a *different* (higher-precision) noise realization, not just sharper — so it changes the
            // output, not only its crispness. Revisit alongside the other f32 flips.
            let noise = create_noise(seed, req.width, req.height)?.as_dtype(Dtype::Bfloat16)?;
            let latents = if is_img2img {
                // VAE-encode the init image to clean latents (f32), then blend with the noise at
                // `sigma = sigmas[init_time_step]` (the fork's `create_for_txt2img_or_img2img`).
                let (image, _) = reference.expect("is_img2img implies a reference");
                let clean = encode_init_latents(&self.vae, image, req.width, req.height)?;
                let sigma = scheduler.sigmas[start_step];
                add_noise_by_interpolation(&clean, &noise, sigma)?
            } else {
                noise
            };
            let latents = denoise_with_progress(
                &self.transformer,
                &scheduler,
                latents,
                &cap,
                start_step,
                &req.cancel,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            // [16,1,H,W] -> [1,16,H,W] -> [1,16,1,H,W] for VAE decode.
            let unpacked = unpack_latents(&latents)?;
            let sh = unpacked.shape();
            let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]])?;
            let decoded = self.vae.decode(&latent5)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(mlx_gen::Error::Msg(
            "z_image_turbo: prompt must not be empty".into(),
        ));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(mlx_gen::Error::Msg(format!(
            "count {} out of range 1..={}",
            req.count, caps.max_count
        )));
    }
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(mlx_gen::Error::Msg(format!(
            "{}x{} out of supported range {}..={}",
            req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(mlx_gen::Error::Msg(
            "z_image_turbo is guidance-distilled; `guidance` is not supported".into(),
        ));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(mlx_gen::Error::Msg(
            "z_image_turbo does not support a negative prompt".into(),
        ));
    }
    for c in &req.conditioning {
        let kind = match c {
            Conditioning::Reference { .. } => ConditioningKind::Reference,
            Conditioning::MultiReference { .. } => ConditioningKind::MultiReference,
            Conditioning::ReduxRefs { .. } => ConditioningKind::ReduxRefs,
            Conditioning::Control { .. } => ConditioningKind::Control,
            Conditioning::Depth { .. } => ConditioningKind::Depth,
            Conditioning::Mask { .. } => ConditioningKind::Mask,
        };
        if !caps.accepts(kind) {
            return Err(mlx_gen::Error::Msg(format!(
                "z_image_turbo does not accept {kind:?} conditioning"
            )));
        }
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
    fn descriptor_is_z_image_turbo() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo");
        assert_eq!(d.family, "z-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_lora && d.capabilities.supports_lokr);
        assert!(!d.capabilities.supports_guidance);
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        // An empty prompt must surface as a typed error (it would otherwise panic in encode via
        // `as_slice` on the size-0 token array — F-001).
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_guidance_and_bad_size() {
        let caps = descriptor().capabilities;
        // guidance on a distilled model (non-empty prompt so the empty-prompt guard doesn't mask it).
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // out-of-range size.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // a plain valid request passes.
        req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
    }

    #[test]
    fn validate_rejects_unsupported_conditioning() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest {
            prompt: "a fox".into(),
            conditioning: vec![Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn load_rejects_single_file_source() {
        // Z-Image is a multi-component snapshot, not a single safetensors file.
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()));
        // `Box<dyn Generator>` isn't Debug, so use `.err()` rather than `unwrap_err()`.
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_quantization_spec() {
        // Q4/Q8 is wired (whole model: transformer + text encoder + VAE); a quant spec must get
        // past the load entry point and fail later on the missing snapshot, not on quantization
        // being unsupported.
        for q in [mlx_gen::Quant::Q4, mlx_gen::Quant::Q8] {
            let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(q);
            let err = load(&spec).err().expect("expected an error").to_string();
            assert!(!err.contains("quantization"), "got: {err}");
        }
    }
}
