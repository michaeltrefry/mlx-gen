//! FLUX.1 provider registration and txt2img generation path.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Conditioning, DiffusionSampler, Error, FlowMatchSampler, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, ModelDescriptor, ModelRegistration, Precision,
    Progress, Result, WeightsSource,
};
use mlx_gen_z_image::vae::Vae;
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::Dtype;

use crate::config::{FluxVariant, DEFAULT_SAMPLER, HYPER_SAMPLER};
use crate::image_encoder::FluxIpImageEncoder;
use crate::ip_adapter::{FluxIpAdapter, FluxIpInjector};
use crate::loader;
use crate::pipeline::{build_linear_sigmas, create_noise, unpack_latents};
use crate::text_encoder::FluxTextEncoders;
use crate::transformer::FluxTransformer;

/// Default `ip_adapter_scale` when a `Conditioning::Reference` omits its `strength` (epic 3621).
/// SceneWorks maps `ipAdapterScale` (default 0.7) → `strength`; this is the engine-side fallback.
const DEFAULT_IP_SCALE: f32 = 0.7;
/// `true_cfg` clamp range for the IP-Adapter dev path (SceneWorks default 4.0).
const TRUE_CFG_MIN: f32 = 1.0;
const TRUE_CFG_MAX: f32 = 10.0;

pub fn descriptor_schnell() -> ModelDescriptor {
    descriptor_for(FluxVariant::Schnell)
}

pub fn descriptor_dev() -> ModelDescriptor {
    descriptor_for(FluxVariant::Dev)
}

pub fn descriptor_for(variant: FluxVariant) -> ModelDescriptor {
    variant.descriptor()
}

pub fn load_schnell(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(FluxVariant::Schnell, spec)
}

pub fn load_dev(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(FluxVariant::Dev, spec)
}

fn load_variant(variant: FluxVariant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(load_flux1(variant, spec)?))
}

/// Load a fully-weighted [`Flux1`] (the concrete type, not boxed) — the entry point the PuLID-FLUX
/// provider (`mlx-gen-pulid`, sc-3074) wraps so it can drive the denoise through
/// [`Flux1::generate_with_injector`].
pub fn load_flux1(variant: FluxVariant, spec: &LoadSpec) -> Result<Flux1> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{}: only dense bf16 is wired for the FLUX.1 port plan",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a FLUX.1 snapshot directory (tokenizer/ tokenizer_2/ text_encoder/ \
                 text_encoder_2/ transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };

    let t5_tokenizer = loader::load_t5_tokenizer(root, variant)?;
    let clip_tokenizer = loader::load_clip_tokenizer(root)?;
    let mut text_encoders = FluxTextEncoders {
        t5: loader::load_t5_encoder(root)?,
        clip: loader::load_clip_encoder(root)?,
    };
    let mut transformer = loader::load_transformer(root, variant)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        text_encoders.quantize(bits)?;
        transformer.quantize(bits)?;
        vae.quantize(bits)?;
    }
    // Install LoRA/LoKr adapters AFTER quantization (the fork merges/applies post-quantize too; a
    // forward-time residual over the now-quantized base, never a fused merge). No-op when empty; a
    // non-empty spec list that matches nothing — or any unmatched target — errors loudly (sc-2534).
    crate::adapters::apply_flux_adapters(&mut transformer, &spec.adapters)?;

    // Optional XLabs IP-Adapter (epic 3621), loaded from `LoadSpec::ip_adapter`. The plain txt2img
    // path is unaffected when absent.
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => Some(loader::load_flux_ip_adapter(p)?),
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(format!(
                "{}: ip_adapter expects a directory (ip_adapter.safetensors + image_encoder/), not a single file",
                variant.id()
            )))
        }
        None => None,
    };

    Ok(Flux1 {
        descriptor: descriptor_for(variant),
        variant,
        t5_tokenizer: Some(t5_tokenizer),
        clip_tokenizer: Some(clip_tokenizer),
        text_encoders: Some(text_encoders),
        transformer: Some(transformer),
        vae: Some(vae),
        ip_adapter,
    })
}

pub struct Flux1 {
    descriptor: ModelDescriptor,
    variant: FluxVariant,
    t5_tokenizer: Option<TextTokenizer>,
    clip_tokenizer: Option<TextTokenizer>,
    text_encoders: Option<FluxTextEncoders>,
    transformer: Option<FluxTransformer>,
    vae: Option<Vae>,
    /// XLabs IP-Adapter (epic 3621): the CLIP-ViT-L/14 image encoder (sc-3622) + the adapter modules
    /// (sc-3623). `Some` only when a `LoadSpec::ip_adapter` was supplied. A `Conditioning::Reference`
    /// request errors loudly when this is `None`.
    ip_adapter: Option<(FluxIpImageEncoder, FluxIpAdapter)>,
}

impl Flux1 {
    pub fn new_for_tests(variant: FluxVariant) -> Self {
        Self {
            descriptor: descriptor_for(variant),
            variant,
            t5_tokenizer: None,
            clip_tokenizer: None,
            text_encoders: None,
            transformer: None,
            vae: None,
            ip_adapter: None,
        }
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        let t5_tokenizer = self.t5_tokenizer.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: T5 tokenizer is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })?;
        let clip_tokenizer = self.clip_tokenizer.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: CLIP tokenizer is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })?;
        let text_encoders = self.text_encoders.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: text encoders are not loaded in this test-only instance",
                self.descriptor.id
            ))
        })?;
        let t5 = t5_tokenizer.tokenize(prompt)?;
        let clip = clip_tokenizer.tokenize(prompt)?;
        text_encoders.encode(&t5.input_ids, &clip.input_ids)
    }

    fn transformer(&self) -> Result<&FluxTransformer> {
        self.transformer.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: transformer is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })
    }

    fn vae(&self) -> Result<&Vae> {
        self.vae.as_ref().ok_or_else(|| {
            Error::Msg(format!(
                "{}: VAE is not loaded in this test-only instance",
                self.descriptor.id
            ))
        })
    }
}

impl Generator for Flux1 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reference-image (XLabs IP-Adapter) path, epic 3621. `validate` (run inside the injector
        // generate methods) has already confirmed at most one `Reference`; extract it here.
        if let Some((image, strength)) = single_reference(req)? {
            let (encoder, adapter) = self.ip_adapter.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "{}: a reference image needs an IP-adapter — load it via LoadSpec::with_ip_adapter",
                    self.descriptor.id
                ))
            })?;
            let embeds = encoder.encode(image)?;
            let scale = strength.unwrap_or(DEFAULT_IP_SCALE).clamp(0.0, 1.0);
            let pos = FluxIpInjector::new(adapter, &embeds, scale)?;
            // dev + an explicit `true_cfg` → real CFG against `negative_prompt`, image prompt only on
            // the positive branch (diffusers' Flux IP `true_cfg_scale`). Otherwise the single distilled
            // forward (schnell always; dev without true_cfg).
            return match req.true_cfg {
                Some(cfg) if self.variant.supports_guidance() => {
                    let neg = FluxIpInjector::disabled(adapter, &embeds)?;
                    let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
                    self.generate_with_injector_cfg(
                        req,
                        &pos,
                        &neg,
                        neg_prompt,
                        cfg.clamp(TRUE_CFG_MIN, TRUE_CFG_MAX),
                        0,
                        on_progress,
                    )
                }
                _ => self.generate_with_injector(req, Some(&pos), on_progress),
            };
        }
        self.generate_with_injector(req, None, on_progress)
    }
}

/// Extract the single reference image + its optional `strength` (`ip_adapter_scale`) from a request,
/// or `None` for plain txt2img. Errors on `MultiReference` or more than one reference (FLUX.1's
/// XLabs IP-Adapter conditions on exactly one image).
fn single_reference(req: &GenerationRequest) -> Result<Option<(&Image, Option<f32>)>> {
    match req.conditioning.as_slice() {
        [] => Ok(None),
        [Conditioning::Reference { image, strength }] => Ok(Some((image, *strength))),
        _ => Err(Error::Msg(
            "flux: reference conditioning supports exactly one Reference image".into(),
        )),
    }
}

impl Flux1 {
    /// As [`Generator::generate`], but threading an optional per-block image-stream residual
    /// injector (the PuLID id cross-attn, sc-3072/3074) into every denoise step's transformer
    /// forward. `injector = None` is byte-identical to [`Generator::generate`] (it IS that path).
    pub fn generate_with_injector(
        &self,
        req: &GenerationRequest,
        injector: Option<&dyn crate::transformer::DitImageInjector>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let transformer = self.transformer()?;
        let vae = self.vae()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        // Sampler selection (sc-2908). FLUX is flow-match: the base render and the few-step `hyper`
        // profile share the SAME flow-match schedule (mflux's `LinearScheduler`) — `hyper` only
        // changes the default step count + guidance (the acceleration is a distilled LoRA the caller
        // loads at `scale≈0.125` via `spec.adapters`, not a different scheduler). An unset sampler is
        // the base flow-match path; `validate_request` rejects any name not in the descriptor.
        let sampler_name = req.sampler.as_deref().unwrap_or(DEFAULT_SAMPLER);
        let (def_steps, def_guidance) = profile_defaults(self.variant, sampler_name);
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let guidance = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(def_guidance)
        } else {
            0.0
        };
        // The FLUX diffusion path is MIXED precision, matching the mflux reference (sc-2787, verified
        // against the bf16 golden's per-tensor dtypes): the latents (`create_noise` → f32) and the
        // main residual stream stay f32 — the fork's scheduler casts the noise prediction to
        // `latents.dtype` (f32) and its T5 `prompt_embeds` is f32 (T5LayerNorm upcast). Only the CLIP
        // pooled embedding and the time/text/guidance conditioning run bf16 (handled in the encoders
        // and `TimeTextEmbed`). So latents are NOT cast to bf16 here — that would diverge from the
        // fork. (The old "f32 everywhere to dodge the x_embedder bf16 GEMM bug" is obsolete: that bug
        // is fixed by sc-2772, and the fork runs the x_embedder in f32 anyway because latents are f32.)
        let (prompt_embeds, pooled_prompt_embeds) = self.encode_prompt(&req.prompt)?;
        let sigmas = build_linear_sigmas(
            steps,
            req.width,
            req.height,
            self.variant.requires_sigma_shift(),
        )?;
        // Drive the denoise through the swappable `DiffusionSampler` seam (sc-2769). FLUX's impl is the
        // flow-match Euler sampler over these sigmas: `scale_model_input` is identity, `timestep(t)` is
        // `sigmas[t]` (fed straight to the transformer), and `step` is `x + v·(σ_{t+1}−σ_t)` — exactly
        // the proven inline loop, so the base render stays bit-exact (guarded by the e2e parity test).
        let sampler = FlowMatchSampler::new(sigmas);
        let n_steps = sampler.num_steps();

        // sc-2963 (rollout of sc-2957): run the MMDiT's fusable elementwise glue (adaLN affine,
        // gated residual, tanh-GELU FFN, RoPE rotation) through `mx.compile` — bit-exact (`max|Δ|=0`,
        // compile_parity.rs) and a per-step win at production geometry. Process-global, idempotent.
        crate::transformer::set_compile_glue(true);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let mut latents = create_noise(seed, req.width, req.height)?;
            for t in 0..n_steps {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                let x_in = sampler.scale_model_input(&latents, t)?;
                let velocity = transformer.forward_injected(
                    &x_in,
                    &prompt_embeds,
                    &pooled_prompt_embeds,
                    sampler.timestep(t),
                    guidance,
                    req.width,
                    req.height,
                    injector,
                )?;
                latents = sampler.step(&velocity, &latents, t)?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: n_steps as u32,
                });
            }

            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&latents, req.width, req.height)?;
            let decoded = vae.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }

    /// Dual-branch real-CFG denoise — the seam PuLID-FLUX `true_cfg > 1.0` (sc-3075) uses. Each step
    /// runs a positive forward (`req.prompt` + `pos_injector`) and, once `t >= timestep_to_start_cfg`,
    /// a negative forward (`neg_prompt` + `neg_injector`), combined as `neg + true_cfg·(pos − neg)`.
    /// The distilled `guidance` is still applied on both branches (the upstream PuLID recipe). The
    /// injectors are generic, so this carries no PuLID-specific code; with two no-op injectors it
    /// reduces to classifier-free guidance between two prompts.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_with_injector_cfg(
        &self,
        req: &GenerationRequest,
        pos_injector: &dyn crate::transformer::DitImageInjector,
        neg_injector: &dyn crate::transformer::DitImageInjector,
        neg_prompt: &str,
        true_cfg: f32,
        timestep_to_start_cfg: usize,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let transformer = self.transformer()?;
        let vae = self.vae()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let sampler_name = req.sampler.as_deref().unwrap_or(DEFAULT_SAMPLER);
        let (def_steps, def_guidance) = profile_defaults(self.variant, sampler_name);
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let guidance = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(def_guidance)
        } else {
            0.0
        };
        let (pos_embeds, pos_pooled) = self.encode_prompt(&req.prompt)?;
        let (neg_embeds, neg_pooled) = self.encode_prompt(neg_prompt)?;
        let sigmas = build_linear_sigmas(
            steps,
            req.width,
            req.height,
            self.variant.requires_sigma_shift(),
        )?;
        let sampler = FlowMatchSampler::new(sigmas);
        let n_steps = sampler.num_steps();
        crate::transformer::set_compile_glue(true);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let mut latents = create_noise(seed, req.width, req.height)?;
            for t in 0..n_steps {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                let x_in = sampler.scale_model_input(&latents, t)?;
                let pos = transformer.forward_injected(
                    &x_in,
                    &pos_embeds,
                    &pos_pooled,
                    sampler.timestep(t),
                    guidance,
                    req.width,
                    req.height,
                    Some(pos_injector),
                )?;
                let velocity = if t >= timestep_to_start_cfg {
                    let neg = transformer.forward_injected(
                        &x_in,
                        &neg_embeds,
                        &neg_pooled,
                        sampler.timestep(t),
                        guidance,
                        req.width,
                        req.height,
                        Some(neg_injector),
                    )?;
                    // neg + true_cfg · (pos − neg)
                    add(&neg, &multiply(&subtract(&pos, &neg)?, scalar(true_cfg))?)?
                } else {
                    pos
                };
                latents = sampler.step(&velocity, &latents, t)?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: n_steps as u32,
                });
            }
            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&latents, req.width, req.height)?;
            let decoded = vae.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Few-step profile defaults `(steps, guidance)` applied when the request omits them (sc-2908). The
/// base flow-match path uses the variant's own defaults; the `hyper` profile (Hyper-FLUX.1-dev) is 8
/// steps at guidance 3.5 — paired with the ByteDance Hyper-FLUX 8-step LoRA loaded at `scale≈0.125`
/// (the documented `lora_scale`) via `spec.adapters`. `hyper` is dev-only (it is a FLUX.1-dev LoRA)
/// and schnell never advertises it, so it never reaches here for schnell.
fn profile_defaults(variant: FluxVariant, sampler: &str) -> (u32, f32) {
    match sampler {
        HYPER_SAMPLER => (8, crate::config::DEFAULT_GUIDANCE),
        _ => (variant.default_steps(), crate::config::DEFAULT_GUIDANCE),
    }
}

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    // Reject a sampler the variant does not advertise (e.g. `hyper` on schnell, or any typo) rather
    // than silently falling back to the base flow-match path.
    if let Some(s) = &req.sampler {
        if !desc.capabilities.samplers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "{}: unsupported sampler {s:?} (supported: {:?})",
                desc.id, desc.capabilities.samplers
            )));
        }
    }
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of 16, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    let caps = &desc.capabilities;
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}: size {}x{} outside supported range {}..={}",
            desc.id, req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "{}: count must be 1..={}",
            desc.id, caps.max_count
        )));
    }
    // Reference conditioning (XLabs IP-Adapter, epic 3621): exactly one `Reference`. The negative
    // prompt + true_cfg knobs ride the IP-Adapter dev path (real CFG), so they are permitted
    // alongside a reference on a guidance-capable (dev) variant even though base txt2img advertises
    // neither — but never on the plain (no-reference) path, which stays bit-identical.
    let has_reference = match req.conditioning.as_slice() {
        [] => false,
        [Conditioning::Reference { .. }] => true,
        _ => {
            return Err(Error::Msg(format!(
                "{}: only a single Reference image is supported (no MultiReference / multiple references)",
                desc.id
            )))
        }
    };
    let ref_cfg_ok = has_reference && caps.supports_guidance;
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt && !ref_cfg_ok {
        return Err(Error::Msg(format!(
            "{}: negative prompts are not supported",
            desc.id
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(Error::Msg(format!(
            "{}: guidance is not supported by this distilled variant",
            desc.id
        )));
    }
    if req.true_cfg.is_some() && !caps.supports_true_cfg && !ref_cfg_ok {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    Ok(())
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_schnell, load: load_schnell }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_dev, load: load_dev }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX1_DEV_ID, FLUX1_SCHNELL_ID};

    #[test]
    fn validates_base_txt2img_request() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn schnell_rejects_guidance() {
        let model = Flux1::new_for_tests(FluxVariant::Schnell);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("guidance is not supported"));
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            width: 1025,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
    }

    #[test]
    fn constants_match_expected_ids() {
        assert_eq!(FluxVariant::Schnell.id(), FLUX1_SCHNELL_ID);
        assert_eq!(FluxVariant::Dev.id(), FLUX1_DEV_ID);
    }

    // ---- epic 3621: XLabs IP-Adapter reference conditioning -----------------------------------

    fn tiny_image() -> Image {
        Image {
            width: 8,
            height: 8,
            pixels: vec![128u8; 8 * 8 * 3],
        }
    }

    fn reference(strength: Option<f32>) -> Conditioning {
        Conditioning::Reference {
            image: tiny_image(),
            strength,
        }
    }

    #[test]
    fn both_variants_advertise_reference_conditioning() {
        for v in [FluxVariant::Dev, FluxVariant::Schnell] {
            assert!(
                descriptor_for(v)
                    .capabilities
                    .conditioning
                    .contains(&mlx_gen::ConditioningKind::Reference),
                "{} must advertise Reference conditioning",
                v.id()
            );
        }
    }

    #[test]
    fn validate_accepts_single_reference_with_ip_knobs_on_dev() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            guidance: Some(3.5),
            true_cfg: Some(4.0),
            negative_prompt: Some("blurry".into()),
            conditioning: vec![reference(Some(0.7))],
            ..Default::default()
        };
        // true_cfg + negative_prompt are permitted alongside a reference on dev even though base
        // txt2img advertises neither.
        model.validate(&req).unwrap();
    }

    #[test]
    fn validate_rejects_true_cfg_without_reference() {
        // The IP knobs are scoped to the reference path; the plain dev path stays unchanged.
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            true_cfg: Some(4.0),
            ..Default::default()
        };
        assert!(model
            .validate(&req)
            .unwrap_err()
            .to_string()
            .contains("true_cfg is not supported"));
    }

    #[test]
    fn validate_rejects_true_cfg_on_schnell_even_with_reference() {
        // schnell is not guidance/CFG-capable, so true_cfg is rejected regardless of a reference.
        let model = Flux1::new_for_tests(FluxVariant::Schnell);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            true_cfg: Some(4.0),
            conditioning: vec![reference(None)],
            ..Default::default()
        };
        assert!(model
            .validate(&req)
            .unwrap_err()
            .to_string()
            .contains("true_cfg is not supported"));
    }

    #[test]
    fn validate_rejects_multiple_references() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            conditioning: vec![reference(None), reference(None)],
            ..Default::default()
        };
        assert!(model
            .validate(&req)
            .unwrap_err()
            .to_string()
            .contains("single Reference"));
    }

    #[test]
    fn generate_reference_without_ip_adapter_errors() {
        // No `LoadSpec::ip_adapter` → a reference request errors loudly before touching the (absent)
        // transformer, so the descriptor never advertises a path that silently no-ops.
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        let req = GenerationRequest {
            prompt: "a portrait".into(),
            conditioning: vec![reference(Some(0.7))],
            ..Default::default()
        };
        let err = model.generate(&req, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("needs an IP-adapter"), "got: {err}");
    }

    // ---- sc-2908: sampler capability surface + few-step profile -----------------------------

    #[test]
    fn dev_advertises_hyper_schnell_does_not() {
        // Hyper-FLUX is a FLUX.1-dev LoRA: dev exposes the base + `hyper` samplers; schnell (already
        // a distilled 4-step checkpoint) exposes only the base flow-match sampler.
        let dev = descriptor_for(FluxVariant::Dev).capabilities.samplers;
        assert_eq!(dev, vec![DEFAULT_SAMPLER, HYPER_SAMPLER]);
        let schnell = descriptor_for(FluxVariant::Schnell).capabilities.samplers;
        assert_eq!(schnell, vec![DEFAULT_SAMPLER]);
    }

    #[test]
    fn validate_accepts_base_and_hyper_on_dev() {
        let model = Flux1::new_for_tests(FluxVariant::Dev);
        for s in [DEFAULT_SAMPLER, HYPER_SAMPLER] {
            let req = GenerationRequest {
                prompt: "a red fox".into(),
                guidance: Some(3.5),
                sampler: Some(s.into()),
                ..Default::default()
            };
            assert!(
                model.validate(&req).is_ok(),
                "sampler {s:?} should be accepted on dev"
            );
        }
        // An unset sampler is the base flow-match path.
        let req = GenerationRequest {
            prompt: "a red fox".into(),
            guidance: Some(3.5),
            ..Default::default()
        };
        assert!(model.validate(&req).is_ok());
    }

    #[test]
    fn validate_rejects_hyper_on_schnell_and_unknown_samplers() {
        // `hyper` is dev-only — schnell does not advertise it, so it is rejected, not downgraded.
        let schnell = Flux1::new_for_tests(FluxVariant::Schnell);
        let err = schnell
            .validate(&GenerationRequest {
                prompt: "a red fox".into(),
                sampler: Some(HYPER_SAMPLER.into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported sampler"), "got: {err}");
        // Any unknown sampler name is rejected on dev too.
        let dev = Flux1::new_for_tests(FluxVariant::Dev);
        for bad in ["lcm", "lightning", "euler", "nonsense"] {
            let err = dev
                .validate(&GenerationRequest {
                    prompt: "a red fox".into(),
                    guidance: Some(3.5),
                    sampler: Some(bad.into()),
                    ..Default::default()
                })
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("unsupported sampler"),
                "sampler {bad:?}: {err}"
            );
        }
    }

    #[test]
    fn hyper_profile_defaults_are_eight_steps_guidance_3_5() {
        // The few-step profile: 8 steps at guidance 3.5 (the Hyper-FLUX.1-dev recommendation).
        assert_eq!(profile_defaults(FluxVariant::Dev, HYPER_SAMPLER), (8, 3.5));
        // The base path keeps the variant's own defaults (dev 25, schnell 4).
        assert_eq!(
            profile_defaults(FluxVariant::Dev, DEFAULT_SAMPLER),
            (
                FluxVariant::Dev.default_steps(),
                crate::config::DEFAULT_GUIDANCE
            )
        );
        assert_eq!(
            profile_defaults(FluxVariant::Schnell, DEFAULT_SAMPLER),
            (
                FluxVariant::Schnell.default_steps(),
                crate::config::DEFAULT_GUIDANCE
            )
        );
    }
}
