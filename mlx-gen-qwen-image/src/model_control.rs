//! `QwenImageControl` — the Qwen-Image **ControlNet (strict pose)** variant (epic 3401), registered
//! as its own `Generator` (`qwen_image_control`) via the InstantX `Qwen-Image-ControlNet-Union`
//! checkpoint (DWPose-trained, Apache-2.0).
//!
//! Identical to [`crate::model::QwenImage`] (T2I) except it also loads a [`QwenControlNet`] control
//! branch and `generate` threads a VAE-encoded pose skeleton through it: each denoise step the
//! control branch emits 5 per-block residuals that are injected into the frozen base 60-layer MMDiT
//! (`interval = 12`, scaled by the request's control scale). [`load`] needs the base snapshot
//! (`spec.weights`) **and** the control checkpoint (`spec.control`); it applies both dense, then
//! quantizes base + control together (Q4/Q8, transformer-only — the fork's overlay-then-quantize
//! ordering). Identity comes from a character LoRA on the **base** (`spec.adapters`); the control
//! branch is never an adapter target.
//!
//! v1 is **pose-only** (the InstantX Union also supports canny/depth, rejected here) and **base
//! pose-from-prompt** (composing with the edit model is a later reach). Parity vs the diffusers
//! `QwenImageControlNetPipeline` is gated by `tests/control_real_weights.rs` (`#[ignore]`, M-series).

use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, ControlKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::{Array, Dtype};

use crate::control_transformer::QwenControlNet;
use crate::loader;
use crate::model::{validate_request, LIGHTNING_SAMPLER};
use crate::pipeline::{
    create_noise, decoded_to_image, denoise_control_with_progress, encode_init_latents,
    qwen_scheduler, unpack_latents,
};
use crate::sampler::{lightning, FlowMatchSampler};
use crate::text_encoder::QwenTextEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;

/// Registry id for the Qwen-Image ControlNet (strict pose) variant.
pub const MODEL_ID: &str = "qwen_image_control";

/// Default inference steps (the base T2I flow-match default).
const DEFAULT_STEPS: u32 = 4;
/// Default CFG guidance (the base T2I default).
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Empty/whitespace negative prompts fall back to a single space (the base `QwenPromptEncoder`).
const NEGATIVE_FALLBACK: &str = " ";
/// Lightning default steps — must match the loaded distillation LoRA variant (4- or 8-step).
const LIGHTNING_DEFAULT_STEPS: u32 = 8;

/// The control variant's identity + capabilities — the base Qwen-Image T2I surface (true CFG /
/// negative prompt / guidance / Lightning) plus the **required** `Control` (pose skeleton)
/// conditioning. LoRA/LoKr (character identity) is on the base transformer.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            // Control (required, pose) only in v1 — no img2img Reference / edit compose yet.
            conditioning: vec![ConditioningKind::Control],
            supports_lora: true,
            supports_lokr: true,
            samplers: vec![LIGHTNING_SAMPLER],
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: true,
        },
    }
}

/// A loaded control generator: the base components + the control branch.
pub struct QwenImageControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: QwenTextEncoder,
    transformer: QwenTransformer,
    controlnet: QwenControlNet,
    vae: QwenVae,
}

/// Construct a [`QwenImageControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a base `Qwen/Qwen-Image` snapshot directory and `spec.control` (required)
/// the InstantX `Qwen-Image-ControlNet-Union` checkpoint (a single `.safetensors` `File`, or a
/// `Dir`). Base + control load dense (bf16); `spec.quantize` (Q4/Q8) then quantizes both transformers
/// (group_size 64). The text encoder + VAE stay dense (the fork's transformer-only quant scope —
/// see [`crate::model::load`]).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image_control: only dense bf16 is wired in the Rust port (drop the precision \
             override)"
                .into(),
        ));
    }
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "qwen_image_control expects a base snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/) as `weights`, not a single .safetensors file"
                    .into(),
            )),
        };
    let control = spec.control.as_ref().ok_or_else(|| {
        Error::Msg(
            "qwen_image_control requires the InstantX Qwen-Image-ControlNet-Union weights — set \
             LoadSpec::control (e.g. with_control(WeightsSource::File(...)))"
                .into(),
        )
    })?;

    // Base + control applied dense first, THEN quantize together (the overlay-then-quantize ordering,
    // matching the Z-Image control port): quantizing before loading the control branch would not let
    // the dense control Linears compose. The text encoder + VAE stay dense (fork's quant scope).
    let mut transformer = loader::load_transformer(root)?;
    let mut controlnet = loader::load_controlnet(control)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        controlnet.quantize(bits)?;
    }
    // Character-identity LoRA/LoKr targets the base transformer only (the control branch is never an
    // adapter target). No-op when `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
    }
    Ok(Box::new(QwenImageControl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder: loader::load_text_encoder(root)?,
        transformer,
        controlnet,
        vae: loader::load_vae(root)?,
    }))
}

impl QwenImageControl {
    /// Prompt → conditioning embeds (bf16), identical to the base T2I `encode_prompt`.
    fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        let t = self.tokenizer.tokenize(prompt)?;
        if t.input_ids.shape()[1] == 0 {
            return Err(Error::Msg("qwen_image_control: empty prompt".into()));
        }
        let embeds = self.text_encoder.encode(&t.input_ids, &t.attention_mask)?;
        Ok(embeds.as_dtype(Dtype::Bfloat16)?)
    }

    /// Extract the required pose control image + its scale. v1 is **pose-only**: a non-`Pose`
    /// `ControlKind` (canny/depth/other) is rejected rather than silently treated as pose, even
    /// though the Union weights support it.
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        let mut found = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, kind, scale } = c {
                if *kind != ControlKind::Pose {
                    return Err(Error::Msg(format!(
                        "qwen_image_control v1 supports pose control only, got {kind:?}"
                    )));
                }
                if found.is_some() {
                    return Err(Error::Msg(
                        "qwen_image_control: a single control image is supported".into(),
                    ));
                }
                found = Some((image, *scale));
            }
        }
        found.ok_or_else(|| {
            Error::Msg("qwen_image_control requires a Control (pose skeleton) conditioning".into())
        })
    }
}

impl Generator for QwenImageControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)?;
        if !req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }))
        {
            return Err(Error::Msg(
                "qwen_image_control requires a Control (pose skeleton) conditioning".into(),
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

        let is_lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
        let default_steps = if is_lightning {
            LIGHTNING_DEFAULT_STEPS
        } else {
            DEFAULT_STEPS
        };
        let steps = req.steps.unwrap_or(default_steps) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let base_seed = req.seed.unwrap_or_else(default_seed);

        let (control_image, control_scale) = self.resolve_control(req)?;

        // Positive conditioning always; negative only for true CFG (Lightning is CFG-distilled).
        let pos = self.encode_prompt(&req.prompt)?;
        let neg = if is_lightning {
            None
        } else {
            let neg_prompt = match req.negative_prompt.as_deref() {
                Some(s) if !s.trim().is_empty() => s,
                _ => NEGATIVE_FALLBACK,
            };
            Some(self.encode_prompt(neg_prompt)?)
        };

        let sampler = if is_lightning {
            lightning(steps)
        } else {
            FlowMatchSampler::new(qwen_scheduler(steps, req.width, req.height).sigmas)
        };

        // VAE-encode + pack the pose skeleton to the control latent `[1, seq, 64]` (constant across
        // steps + the batch). Same encode/pack as an init image (the diffusers control path encodes
        // the control image with the VAE and `_pack_latents` 2×2, identical to the noise packing).
        let control_cond = encode_init_latents(&self.vae, control_image, req.width, req.height)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, req.width, req.height)?;
            let latents = denoise_control_with_progress(
                &self.transformer,
                &self.controlnet,
                &sampler,
                noise,
                &control_cond,
                &pos,
                neg.as_ref(),
                guidance,
                control_scale,
                req.width,
                req.height,
                0,
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

inventory::submit! {
    ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_qwen_image_control() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_control");
        assert_eq!(d.family, "qwen-image");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.supports_lora);
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights, proving the overlay
        // is a hard requirement (it fails here before touching the missing base snapshot).
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("Qwen-Image-ControlNet-Union"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/q.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
