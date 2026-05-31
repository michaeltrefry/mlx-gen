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
    Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, ModelRegistration, Precision, Progress, Result,
    WeightsSource,
};
use mlx_rs::{Array, Dtype};

use crate::loader;
use crate::pipeline::{
    create_noise, decoded_to_image, denoise_with_progress, qwen_scheduler, unpack_latents,
};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Qwen-Image default inference steps (the fork's `num_inference_steps`).
const DEFAULT_STEPS: u32 = 4;
/// Qwen-Image default CFG guidance (the fork's `guidance=4.0`).
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Empty/whitespace negative prompts fall back to a single space (the fork's `QwenPromptEncoder`).
const NEGATIVE_FALLBACK: &str = " ";

/// Registry id for Qwen-Image (matches the SceneWorks worker's `payload.model`).
pub const MODEL_ID: &str = "qwen_image";

/// Qwen-Image's identity + capabilities — constructible without loading weights (registry
/// introspection). T2I only for now (Edit reference conditioning is sc-2465); LoRA wiring is a
/// later slice, so it is advertised off rather than silently ignored.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            // True CFG with a negative prompt + guidance (not distilled).
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Edit (reference conditioning) is a separate variant (sc-2465).
            conditioning: Vec::new(),
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
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
/// encoder promotes to f32 internally. Q8 quantization and an fp32 override are not yet wired (the
/// validated path is dense bf16) — both are rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "qwen_image: Q8 quantization is not yet wired in the Rust port; the validated path is \
             dense bf16 (drop `quantize` to load)"
                .into(),
        ));
    }
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
    Ok(Box::new(QwenImage {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder: loader::load_text_encoder(root)?,
        transformer: loader::load_transformer(root)?,
        vae: loader::load_vae(root)?,
    }))
}

impl QwenImage {
    /// Prompt → conditioning embeds (bf16): apply the system template, tokenize, run the text
    /// encoder, drop the 34 template tokens. An empty prompt is allowed only for the negative
    /// branch (the caller substitutes a space).
    fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        let t = self.tokenizer.tokenize(prompt)?;
        if t.input_ids.shape()[1] == 0 {
            return Err(Error::Msg("qwen_image: empty prompt".into()));
        }
        let embeds = self.text_encoder.encode(&t.input_ids, &t.attention_mask)?;
        Ok(embeds.as_dtype(Dtype::Bfloat16)?)
    }
}

impl Generator for QwenImage {
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
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // Positive + negative conditioning (bf16). Empty negative → a single space (fork fallback).
        let pos = self.encode_prompt(&req.prompt)?;
        let neg_prompt = match req.negative_prompt.as_deref() {
            Some(s) if !s.trim().is_empty() => s,
            _ => NEGATIVE_FALLBACK,
        };
        let neg = self.encode_prompt(neg_prompt)?;

        // The schedule is resolution-dependent but seed-independent — build it once.
        let scheduler = qwen_scheduler(steps, req.width, req.height);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            // Latents stay f32 through the loop: the fork keeps txt2img noise f32, and MLX
            // promotes the bf16 transformer weights to f32 per-op (only `prompt_embeds` is bf16).
            let noise = create_noise(seed, req.width, req.height)?;
            let latents = denoise_with_progress(
                &self.transformer,
                &scheduler,
                noise,
                &pos,
                &neg,
                guidance,
                req.width,
                req.height,
                &req.cancel,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&latents, req.width, req.height)?;
            let decoded = self.vae.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Seed when a request omits one: nanos since the epoch (only sets which sample is drawn).
fn default_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
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
        let kind = match c {
            Conditioning::Reference { .. } => ConditioningKind::Reference,
            Conditioning::MultiReference { .. } => ConditioningKind::MultiReference,
            Conditioning::ReduxRefs { .. } => ConditioningKind::ReduxRefs,
            Conditioning::Control { .. } => ConditioningKind::Control,
            Conditioning::Depth { .. } => ConditioningKind::Depth,
            Conditioning::Mask { .. } => ConditioningKind::Mask,
        };
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "qwen_image (T2I) does not accept {kind:?} conditioning"
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
    fn descriptor_is_qwen_image() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image");
        assert_eq!(d.family, "qwen-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
        assert!(d.capabilities.supports_true_cfg);
        assert!(d.capabilities.requires_sigma_shift);
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
    fn load_rejects_single_file_and_quant() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");

        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(mlx_gen::Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("quantization"), "got: {err}");
    }
}
