//! `QwenImage` — the Qwen-Image T2I implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry.
//!
//! [`load`] assembles the model from a `Qwen/Qwen-Image` snapshot directory (see [`crate::loader`])
//! — tokenizer, Qwen2.5-VL text encoder, 60-layer MMDiT, causal-Conv3d VAE — and
//! [`QwenImage::generate`] runs the prompt→image pipeline: tokenize (+ system template) → encode
//! (drop the 34 template tokens) → seeded packed noise → flow-match Euler denoise with classifier-
//! free guidance (two forwards/step) → unpack → VAE decode → RGB8. The component math is parity-
//! proven against the frozen Python fork (slices 1–3); the e2e bf16 path is gated by
//! `tests/e2e_real_weights.rs`.

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Quant, Result, WeightsSource,
};

use crate::loader;
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, decode_and_collect, denoise_with_progress,
    encode_init_latents, encode_prompt, init_time_step, negative_or_fallback, resolve_run_params,
    LIGHTNING_SAMPLER,
};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Registry id for Qwen-Image (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "qwen_image";

/// Qwen-Image's identity + capabilities — constructible without loading weights (registry
/// introspection). This is the **T2I** variant (`qwen_image`), which also accepts a single init
/// `Reference` image for **img2img** (sc-2530); Qwen-Image-Edit ships as a separate `qwen_image_edit`
/// model (sc-2465). LoRA/LoKr is wired (sc-2528). Few-step **Lightning** acceleration is exposed as
/// the `lightning` sampler (sc-2909); an unset sampler is the production flow-match path.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // True CFG with a negative prompt + guidance (not distilled).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // img2img: a single init `Reference` image (+ `image_strength`) seeds the latents via
            // the noise blend (sc-2530, the fork's `Img2Img` path). Reference *conditioning* for
            // editing is the separate `qwen_image_edit` variant (sc-2465).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr wired (sc-2528): the fork's `QwenLoRAMapping` targets routed onto the
            // transformer's `AdaptableHost`; stacked + mixed via the core seam.
            supports_lora: true,
            supports_lokr: true,
            // `lightning` = the few-step Lightning acceleration sampler (sc-2909); an unset
            // `req.sampler` is the production flow-match path. Any other name is rejected in
            // `validate_request` rather than silently downgraded.
            samplers: vec![LIGHTNING_SAMPLER],
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            // Flow-match schedule uses the resolution-dependent sigma shift.
            requires_sigma_shift: true,
        },
    }
}

/// A loaded Qwen-Image generator: the four model components assembled from a snapshot directory,
/// plus the cached descriptor.
pub struct QwenImage {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: QwenTextEncoder,
    transformer: QwenTransformer,
    vae: QwenVae,
}

/// Construct a [`QwenImage`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a `Qwen/Qwen-Image` snapshot (the
/// diffusers multi-component tree). Weights load dense at their on-disk dtype (bf16); the text
/// encoder promotes to f32 internally. `spec.quantize` (Q4/Q8) quantizes the transformer only
/// (group_size 64) — the fork's full `quantize=N` scope (sc-2565; see the inline note below). An
/// fp32 precision override is not wired (the validated dense path is bf16) and is rejected rather
/// than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image: only dense bf16 is wired in the Rust port (drop the precision override)"
                .into(),
        ));
    }
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "qwen_image expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    // Q4/Q8 quantizes the **transformer only** (group_size 64) after the dense bf16 load. This is
    // the fork's full `quantize=N` scope, not a descope (sc-2565): `QwenWeightDefinition` marks the
    // `text_encoder` component `skip_quantization=True` ("Quantization causes significant semantic
    // degradation"), so its Linears/Embedding are never quantized; and the VAE is all-conv
    // (`nn.Conv2d`/`Conv3d` lack `to_quantized`), so the fork's `nn.quantize(vae)` is a no-op. The
    // transformer is the only component with quantizable leaves. (Z-Image differs — its fork *does*
    // quantize the TE+VAE, hence sc-2532; do not generalize that here.)
    let mut transformer = loader::load_transformer(root)?;
    if let Some(q) = spec.quantize {
        transformer.quantize(q.bits())?;
    }
    // LoRA/LoKr (sc-2528): applied after quantization, as forward-time residuals over the
    // (possibly quantized) transformer — fork-faithful. No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    Ok(Box::new(QwenImage {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder: loader::load_text_encoder(root)?,
        transformer,
        vae: loader::load_vae(root)?,
    }))
}

impl QwenImage {
    /// Extract the single img2img init image + its strength from the request's conditioning. The
    /// per-reference strength wins over `req.strength`. Qwen-Image T2I img2img conditions on exactly
    /// one init image, so more than one `Reference` is an error (the multi-image edit path is
    /// `qwen_image_edit` + `MultiReference`, sc-2529). Returns `None` for pure txt2img.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "qwen_image: multiple reference images are not supported (single img2img \
                         init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }
}

impl Generator for QwenImage {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor.capabilities, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl QwenImage {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        // Shared step/sampler/guidance/seed resolution (F-117); `req.sampler == "lightning"` selects
        // the few-step recipe, else the production resolution-dependent schedule.
        let params = resolve_run_params(req, req.width, req.height);

        // img2img: a single `Reference` image, with a per-reference strength overriding `req.strength`.
        // `start_step = 0` for pure txt2img (the fork's `Config.init_time_step`).
        let reference = self.resolve_reference(req)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(params.steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;
        // Lightning is the CFG-distilled few-step *txt2img* recipe; an init image (img2img) is out of
        // scope (its blend seeds a different trajectory than the distillation targets).
        if params.is_lightning && is_img2img {
            return Err(Error::Msg(
                "qwen_image: the lightning sampler is txt2img-only (no img2img init image)".into(),
            ));
        }

        // Positive conditioning (bf16) always. The negative branch is only built for true CFG; the
        // Lightning LoRAs are CFG-distilled, so Lightning runs CFG-off (a single forward/step).
        let pos = encode_prompt(&self.tokenizer, &self.text_encoder, &req.prompt, MODEL_ID)?;
        let neg = if params.is_lightning {
            None
        } else {
            Some(encode_prompt(
                &self.tokenizer,
                &self.text_encoder,
                negative_or_fallback(req),
                MODEL_ID,
            )?)
        };

        // VAE-encode the init image to packed clean latents (f32) ONCE — it's seed-independent
        // (LANCZOS resize + a full VAE encode), so doing it inside the per-image loop ran the encoder
        // `count` times for identical output (F-118).
        let clean = if is_img2img {
            let (image, _) = reference.expect("is_img2img implies a reference");
            Some(encode_init_latents(
                &self.vae, image, req.width, req.height,
            )?)
        } else {
            None
        };

        let images = decode_and_collect(
            &self.vae,
            req.count,
            params.base_seed,
            req.width,
            req.height,
            on_progress,
            |seed, progress| {
                // Latents stay f32 through the loop: the fork keeps txt2img/img2img noise f32, and MLX
                // promotes the bf16 transformer weights to f32 per-op (only `prompt_embeds` is bf16).
                let noise = create_noise(seed, req.width, req.height)?;
                let latents = match &clean {
                    // Blend the (hoisted) clean latents with this image's noise at
                    // `sigma = sigmas[init_time_step]` (fork `create_for_txt2img_or_img2img`).
                    Some(clean) => {
                        let sigma = params.sampler.sigma(start_step);
                        add_noise_by_interpolation(clean, &noise, sigma)?
                    }
                    None => noise,
                };
                denoise_with_progress(
                    &self.transformer,
                    &params.sampler,
                    latents,
                    &pos,
                    neg.as_ref(),
                    params.guidance,
                    req.width,
                    req.height,
                    start_step,
                    &req.cancel,
                    progress,
                )
            },
        )?;
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation, factored out for unit testing without loaded weights.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "count {} out of range 1..={}",
            req.count, caps.max_count
        )));
    }
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}x{} out of supported range {}..={}",
            req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    // Qwen-Image latents pack 2×2; sizes must be a multiple of 16 per side (VAE/8 then patch/2).
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}x{} must be a multiple of 16 per side",
            req.width, req.height
        )));
    }
    for c in &req.conditioning {
        let kind = c.kind();
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "qwen_image (T2I) does not accept {kind:?} conditioning"
            )));
        }
    }
    // Reject an unsupported sampler rather than silently downgrading it. An unset sampler is the
    // production flow-match path; only the advertised names (`lightning`) are accepted.
    if let Some(s) = &req.sampler {
        if !caps.samplers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "qwen_image: unsupported sampler {s:?} (supported: {:?}, or unset for the \
                 production flow-match path)",
                caps.samplers
            )));
        }
    }
    // The production flow-match schedule needs >= 2 steps: at 1 step `qwen_sigmas`' terminal-sigma
    // rescale divides by zero (`scale == 0`) and produces a `[NaN, 0.0]` schedule that silently
    // renders garbage (F-113). Lightning's distilled few-step recipe is unaffected, so only guard the
    // production path; an unset `steps` uses the safe `DEFAULT_STEPS`.
    let is_lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
    if !is_lightning {
        if let Some(steps) = req.steps {
            if steps < 2 {
                return Err(Error::Msg(format!(
                    "qwen_image: steps must be >= 2 for the production sampler (got {steps})"
                )));
            }
        }
    }
    Ok(())
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load`] into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_qwen_image() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image");
        assert_eq!(d.family, "qwen-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.requires_sigma_shift);
        // Lightning acceleration is advertised (sc-2909).
        assert!(d.capabilities.samplers.contains(&LIGHTNING_SAMPLER));
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        // Unset sampler → the production flow-match path.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // The advertised `lightning` sampler is accepted.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some(LIGHTNING_SAMPLER.into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // An unknown sampler is rejected, not silently downgraded.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            sampler: Some("lcm".into()),
            ..Default::default()
        };
        let err = validate_request(&caps, &req)
            .expect_err("expected an error")
            .to_string();
        assert!(err.contains("unsupported sampler"), "got: {err}");
    }

    #[test]
    fn validate_rejects_production_steps_below_two() {
        // F-113: 0/1 production steps make qwen_sigmas' terminal rescale divide by zero → NaN
        // schedule. Reject them; Lightning few-step and the default (unset) path stay valid.
        let caps = descriptor().capabilities;
        let prod = |steps| GenerationRequest {
            prompt: "a fox".into(),
            steps,
            ..Default::default()
        };
        for s in [Some(0), Some(1)] {
            let err = validate_request(&caps, &prod(s))
                .expect_err("expected an error")
                .to_string();
            assert!(err.contains("steps must be >= 2"), "steps {s:?} got: {err}");
        }
        assert!(validate_request(&caps, &prod(Some(2))).is_ok());
        assert!(validate_request(&caps, &prod(None)).is_ok());
        // Lightning at 1 step is fine (distilled few-step recipe).
        let lightning = GenerationRequest {
            prompt: "a fox".into(),
            steps: Some(1),
            sampler: Some(LIGHTNING_SAMPLER.into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &lightning).is_ok());
    }

    #[test]
    fn validate_rejects_bad_size_and_conditioning() {
        let caps = descriptor().capabilities;
        // out-of-range size.
        let req = GenerationRequest {
            width: 64,
            height: 64,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // non-multiple-of-16 size.
        let req = GenerationRequest {
            width: 1000,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // T2I accepts no conditioning.
        let req = GenerationRequest {
            conditioning: vec![Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // guidance + negative prompt + valid size passes.
        let req = GenerationRequest {
            prompt: "a fox".into(),
            negative_prompt: Some("blurry".into()),
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
    }

    #[test]
    fn load_rejects_single_file() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }

    #[test]
    fn load_accepts_q8_spec() {
        // Q8 is wired (transformer-only); a Q8 spec must get past the quant gate and fail later on
        // the missing snapshot, not on quantization being unsupported.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(mlx_gen::Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(!err.contains("quantization"), "got: {err}");
    }
}
