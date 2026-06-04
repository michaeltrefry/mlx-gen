//! FLUX.1 provider registration and txt2img generation path.

use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_gen_z_image::vae::Vae;
use mlx_rs::ops::{add, multiply};
use mlx_rs::Dtype;

use crate::config::FluxVariant;
use crate::loader;
use crate::pipeline::{build_linear_sigmas, create_noise, unpack_latents};
use crate::text_encoder::FluxTextEncoders;
use crate::transformer::FluxTransformer;

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

    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{}: FLUX.1 adapter installation awaits the transformer adaptable-path map",
            variant.id()
        )));
    }

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

    Ok(Box::new(Flux1 {
        descriptor: descriptor_for(variant),
        variant,
        t5_tokenizer: Some(t5_tokenizer),
        clip_tokenizer: Some(clip_tokenizer),
        text_encoders: Some(text_encoders),
        transformer: Some(transformer),
        vae: Some(vae),
    }))
}

pub struct Flux1 {
    descriptor: ModelDescriptor,
    variant: FluxVariant,
    t5_tokenizer: Option<TextTokenizer>,
    clip_tokenizer: Option<TextTokenizer>,
    text_encoders: Option<FluxTextEncoders>,
    transformer: Option<FluxTransformer>,
    vae: Option<Vae>,
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
        self.validate(req)?;
        let transformer = self.transformer()?;
        let vae = self.vae()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or_else(|| self.variant.default_steps()) as usize;
        let guidance = if self.variant.supports_guidance() {
            req.guidance.unwrap_or(crate::config::DEFAULT_GUIDANCE)
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

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let mut latents = create_noise(seed, req.width, req.height)?;
            for t in 0..steps {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                let velocity = transformer.forward(
                    &latents,
                    &prompt_embeds,
                    &pooled_prompt_embeds,
                    sigmas[t],
                    guidance,
                    req.width,
                    req.height,
                )?;
                let dt = sigmas[t + 1] - sigmas[t];
                latents = add(
                    &latents,
                    &multiply(&velocity, mlx_rs::Array::from_slice(&[dt], &[1]))?,
                )?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: steps as u32,
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

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
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
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
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
    if req.true_cfg.is_some() && !caps.supports_true_cfg {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    if !req.conditioning.is_empty() {
        return Err(Error::Msg(format!(
            "{}: conditioning variants are not implemented in the base txt2img port yet",
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
}
