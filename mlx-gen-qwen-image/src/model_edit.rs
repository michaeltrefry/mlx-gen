//! `QwenImageEdit` — the Qwen-Image-**Edit** implementation of [`mlx_gen::Generator`] (id
//! `qwen_image_edit`), plus its [`descriptor`]/[`load`] entry points and `inventory` registration.
//!
//! [`load`] assembles the model from a `Qwen/Qwen-Image-Edit-2509` snapshot — tokenizer + Qwen2-VL
//! image processor, the Qwen2.5-VL vision-language encoder (LM + vision transformer), the 60-layer
//! MMDiT, and the causal-Conv3d VAE. [`QwenImageEdit::generate`] runs the reference-conditioned
//! pipeline: tokenize the edit template with the reference image → VL-encode (vision embeds spliced
//! into the prompt) → **dual-latent** conditioning (VAE-encode the reference, pack, concat with the
//! noise over the sequence axis) → flow-match Euler denoise with the reference `cond_grid` in the
//! RoPE (two forwards/step, CFG) → slice the noise prefix → VAE decode → RGB8. The dual-latent
//! denoise core is parity-proven against the fork (`tests/edit_real_weights.rs`).

use mlx_gen::array::host_i32;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Result, WeightsSource,
};
use mlx_rs::{Array, Dtype};

use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::loader;
use crate::model::validate_request;
use crate::pipeline::{
    create_noise, decoded_to_image, denoise_edit_with_progress, qwen_scheduler, unpack_latents,
};
use crate::text_encoder::vision::grid::Grid;
use crate::text_encoder::QwenVisionLanguageEncoder;
use crate::transformer::QwenTransformer;
use crate::vae::QwenVae;
use crate::vl_tokenizer::{
    condition_resize_dims, encode_reference_latents, preprocess_edit_image, tokenize_edit_text,
};

/// Qwen-Image-Edit default inference steps (the fork's `num_inference_steps`).
const DEFAULT_STEPS: u32 = 4;
/// Qwen-Image-Edit default CFG guidance (the fork's `guidance=4.0`).
const DEFAULT_GUIDANCE: f32 = 4.0;

/// Registry id for Qwen-Image-Edit.
pub const MODEL_ID: &str = "qwen_image_edit";

/// Qwen-Image-Edit's identity + capabilities. Accepts a single `Reference` conditioning image (the
/// fork's `use_picture_prefix=False` edit path); multi-reference is a tracked follow-on (sc-2529).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
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

/// A loaded Qwen-Image-Edit generator.
pub struct QwenImageEdit {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    processor: QwenImageProcessor,
    vl_encoder: QwenVisionLanguageEncoder,
    transformer: QwenTransformer,
    vae: QwenVae,
}

/// Construct a [`QwenImageEdit`] from a [`LoadSpec`] (a `Qwen/Qwen-Image-Edit-2509` snapshot dir).
/// `spec.quantize` (Q4/Q8) quantizes the **transformer only** (group_size 64) after the dense bf16
/// load — same as T2I ([`crate::model::load`]). This is the fork's full `quantize=N` scope, not a
/// descope: the Edit variant uses the same `QwenWeightDefinition`, whose `text_encoder` component
/// (the VL model — **LM + vision tower**, all under `text_encoder/`) is `skip_quantization=True`,
/// and whose VAE is all-conv (no `to_quantized` leaves). So the VL encoder and VAE stay bf16,
/// matching the fork (sc-2565).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "qwen_image_edit: only dense bf16 is wired in the Rust port (drop the precision override)"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "qwen_image_edit: LoRA/LoKr adapter application is not yet wired into load() — the core \
             seam (LoadSpec.adapters → adapters::loader::apply_adapter_specs) exists, but the \
             Qwen key→module map lands in sc-2528"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "qwen_image_edit expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let mut transformer = loader::load_transformer(root)?;
    if let Some(q) = spec.quantize {
        transformer.quantize(q.bits())?;
    }
    Ok(Box::new(QwenImageEdit {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        processor: QwenImageProcessor::default(),
        vl_encoder: loader::load_vision_language_encoder(root)?,
        transformer,
        vae: loader::load_vae(root)?,
    }))
}

impl QwenImageEdit {
    /// Edit conditioning embeds (f16, matching the fork) for one prompt: tokenize the edit template
    /// (the `<|image_pad|>` run length is `n_image_tokens`, from the shared image preprocess), then
    /// run the LM over the spliced sequence reusing the already-computed `vision` embeds — so the
    /// vision tower is **not** re-run for the positive vs negative prompt (F-004).
    fn encode_edit(&self, prompt: &str, n_image_tokens: usize, vision: &Array) -> Result<Array> {
        let tok = tokenize_edit_text(&self.tokenizer, prompt, n_image_tokens)?;
        let embeds =
            self.vl_encoder
                .encode_with_vision(&tok.input_ids, &tok.attention_mask, vision)?;
        Ok(embeds.as_dtype(Dtype::Float16)?)
    }
}

impl Generator for QwenImageEdit {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)?;
        if reference_image(req).is_none() {
            return Err(Error::Msg(
                "qwen_image_edit requires a Reference conditioning image".into(),
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
        let reference = reference_image(req).expect("validated present");
        let img = || ImageInput {
            data: &reference.pixels,
            height: reference.height as usize,
            width: reference.width as usize,
        };

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let (out_w, out_h) = (req.width, req.height);

        // VL condition / dual-latent reference resolution (~384² area, /32), from the ref aspect.
        let (vl_w, vl_h) =
            condition_resize_dims(reference.width as usize, reference.height as usize);

        // Preprocess the reference once (condition-resize + patchify) and run the 32-block vision
        // tower once — both depend only on the image, so the positive + negative encodes reuse the
        // vision embeds rather than re-running the tower per prompt (F-004).
        let pre = preprocess_edit_image(&self.processor, img())?;
        let grids: Vec<Grid> = host_i32(&pre.grid_thw)?
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let vision = self.vl_encoder.encode_vision(&pre.pixel_values, &grids)?;

        // Positive + negative conditioning embeds (f16): only the LM forward runs per prompt.
        let pos = self.encode_edit(&req.prompt, pre.n_image_tokens, &vision)?;
        let neg = self.encode_edit(
            req.negative_prompt.as_deref().unwrap_or(""),
            pre.n_image_tokens,
            &vision,
        )?;

        // Dual-latent reference (static across steps + samples): VAE-encode → pack, + its cond grid.
        let (static_latents, cond_grid) =
            encode_reference_latents(&self.vae, img(), vl_w as u32, vl_h as u32)?;
        let cond_grids = [cond_grid];

        let scheduler = qwen_scheduler(steps, out_w, out_h);
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, out_w, out_h)?;
            let latents = denoise_edit_with_progress(
                &self.transformer,
                &scheduler,
                noise,
                &static_latents,
                &cond_grids,
                &pos,
                &neg,
                guidance,
                out_w,
                out_h,
                &req.cancel,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&latents, out_w, out_h)?;
            let decoded = self.vae.decode(&unpacked)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// The first `Reference` conditioning image, if any.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_qwen_image_edit() {
        let d = descriptor();
        assert_eq!(d.id, "qwen_image_edit");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.accepts(ConditioningKind::Depth));
    }

    #[test]
    fn load_accepts_q8_spec() {
        // Q8 is wired (transformer-only, slice 7b): a Q8 spec must get past the quant gate and fail
        // later on the missing snapshot, not on quantization being unsupported.
        let spec =
            LoadSpec::new(WeightsSource::Dir("/nonexistent".into())).with_quant(mlx_gen::Quant::Q8);
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(!err.contains("not wired"), "got: {err}");
    }

    #[test]
    fn generate_requires_a_reference_image() {
        let caps = descriptor().capabilities;
        // A valid-size request with no Reference conditioning fails validation.
        let req = GenerationRequest {
            prompt: "make it autumn".into(),
            ..Default::default()
        };
        // validate_request (size/conditioning) passes, but the edit generator needs a reference.
        assert!(validate_request(&caps, &req).is_ok());
        assert!(reference_image(&req).is_none());
    }
}
