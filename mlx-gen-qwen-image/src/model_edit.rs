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
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use crate::image_processor::{ImageInput, QwenImageProcessor};
use crate::loader;
use crate::model::{validate_request, LIGHTNING_SAMPLER};
use crate::pipeline::{
    create_noise, decoded_to_image, denoise_edit_with_progress, qwen_scheduler, unpack_latents,
};
use crate::sampler::FlowMatchSampler;
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
/// Lightning default steps — must match the loaded LoRA variant (4-step / 8-step); 8 is the
/// higher-quality default (e.g. `lightx2v/Qwen-Image-Edit-2511-Lightning` 8-step). sc-2909.
const LIGHTNING_DEFAULT_STEPS: u32 = 8;

/// Registry id for Qwen-Image-Edit.
pub const MODEL_ID: &str = "qwen_image_edit";

/// Qwen-Image-Edit's identity + capabilities. Accepts one `Reference` or N `MultiReference`
/// conditioning images — the fork's `use_picture_prefix=False` edit path, where every reference is
/// VAE-encoded and folded into the transformer's dual-latent sequence (sc-2529).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "qwen-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: true,
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            // LoRA/LoKr wired (sc-2528): shared `QwenTransformer` host; stacked + mixed.
            supports_lora: true,
            supports_lokr: true,
            // `lightning` = the few-step Lightning sampler (sc-2909), e.g.
            // `lightx2v/Qwen-Image-Edit-2511-Lightning`; an unset sampler is the production path.
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
    // LoRA/LoKr (sc-2528): same load-time, post-quantize, residual-over-base path as T2I.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_qwen_adapters(&mut transformer, &spec.adapters)?;
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
        if reference_images(req).is_empty() {
            return Err(Error::Msg(
                "qwen_image_edit requires a Reference or MultiReference conditioning image".into(),
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
        let references = reference_images(req);
        let first = references[0];
        let last = *references.last().expect("validated non-empty");

        // `req.sampler == "lightning"` selects the few-step Lightning recipe (sc-2909): static-shift
        // schedule + CFG-off single forward + its own step default. Unset = production. The matching
        // Edit Lightning LoRA must be supplied via `spec.adapters`.
        let is_lightning = req.sampler.as_deref() == Some(LIGHTNING_SAMPLER);
        let default_steps = if is_lightning {
            LIGHTNING_DEFAULT_STEPS
        } else {
            DEFAULT_STEPS
        };
        let steps = req.steps.unwrap_or(default_steps) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let (out_w, out_h) = (req.width, req.height);

        // VL condition / dual-latent reference resolution (~384² area, /32). The fork's
        // `_compute_dimensions` derives all dims from `image_paths[-1]`, so the dual-latent
        // resolution comes from the **last** reference's aspect (identical to the first when the
        // references share an aspect ratio, the common case).
        let (vl_w, vl_h) = condition_resize_dims(last.width as usize, last.height as usize);

        // Text/VL conditioning: the fork's `use_picture_prefix=False` edit template carries a
        // single `<|image_pad|>`, so only the **first** reference enters the prompt embeds (verified:
        // multi-image `input_ids` is byte-identical to single-image, and the vision splice consumes
        // only image-0's rows). Run the existing single-image text path on `references[0]` — its
        // block-diagonal vision output is identical whether computed alone or alongside the others.
        // The tower runs once (image-only), so the positive + negative encodes reuse it (F-004).
        let pre = preprocess_edit_image(&self.processor, image_input(first))?;
        let grids: Vec<Grid> = host_i32(&pre.grid_thw)?
            .chunks(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        let vision = self.vl_encoder.encode_vision(&pre.pixel_values, &grids)?;

        // Positive conditioning embeds (f16): only the LM forward runs per prompt. The negative
        // branch is built only for true CFG — the Lightning LoRAs are CFG-distilled, so Lightning
        // runs CFG-off (a single forward/step).
        let pos = self.encode_edit(&req.prompt, pre.n_image_tokens, &vision)?;
        let neg = if is_lightning {
            None
        } else {
            Some(self.encode_edit(
                req.negative_prompt.as_deref().unwrap_or(""),
                pre.n_image_tokens,
                &vision,
            )?)
        };

        // Dual-latent references (static across steps + samples): VAE-encode **each** reference at
        // the VL resolution, pack, and concatenate over the sequence axis — one `cond_grid` per
        // reference so the MMDiT RoPE spans `[noise] + references` (fork
        // `QwenEditUtil.create_image_conditioning_latents` + `forward_multi`).
        let mut packed = Vec::with_capacity(references.len());
        let mut cond_grids = Vec::with_capacity(references.len());
        for im in &references {
            let (latents, grid) =
                encode_reference_latents(&self.vae, image_input(im), vl_w as u32, vl_h as u32)?;
            packed.push(latents);
            cond_grids.push(grid);
        }
        let static_latents = if packed.len() == 1 {
            packed.pop().expect("len checked")
        } else {
            concatenate_axis(&packed.iter().collect::<Vec<_>>(), 1)?
        };

        // Build the sampler once (seed-independent): the static-shift Lightning schedule, or the
        // production `qwen_scheduler` (resolution-dependent).
        let sampler = if is_lightning {
            FlowMatchSampler::lightning(steps)
        } else {
            FlowMatchSampler::new(qwen_scheduler(steps, out_w, out_h))
        };
        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, out_w, out_h)?;
            let latents = denoise_edit_with_progress(
                &self.transformer,
                &sampler,
                noise,
                &static_latents,
                &cond_grids,
                &pos,
                neg.as_ref(),
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

/// Borrow an [`Image`] as an [`ImageInput`] (RGB uint8 HWC) for the preprocess/VAE-encode paths.
fn image_input(im: &Image) -> ImageInput<'_> {
    ImageInput {
        data: &im.pixels,
        height: im.height as usize,
        width: im.width as usize,
    }
}

/// The conditioning reference images, in order — a single `Reference` or every `MultiReference`
/// image. The first drives the text/VL prompt embeds (fork `use_picture_prefix=False`); all of them
/// are VAE-encoded into the dual-latent sequence.
fn reference_images(req: &GenerationRequest) -> Vec<&Image> {
    let mut out = Vec::new();
    for c in &req.conditioning {
        match c {
            Conditioning::Reference { image, .. } => out.push(image),
            Conditioning::MultiReference { images } => out.extend(images.iter()),
            _ => {}
        }
    }
    out
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
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
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
        assert!(reference_images(&req).is_empty());
    }

    #[test]
    fn reference_images_collects_single_and_multi() {
        use mlx_gen::Conditioning;
        let img = |w| Image {
            width: w,
            height: 8,
            pixels: vec![0u8; (w * 8 * 3) as usize],
        };
        // A single `Reference` yields one image.
        let single = GenerationRequest {
            conditioning: vec![Conditioning::Reference {
                image: img(8),
                strength: None,
            }],
            ..Default::default()
        };
        assert_eq!(reference_images(&single).len(), 1);
        // `MultiReference` yields every image, in order (first drives the text path, last the dims).
        let multi = GenerationRequest {
            conditioning: vec![Conditioning::MultiReference {
                images: vec![img(8), img(16), img(24)],
            }],
            ..Default::default()
        };
        let got = reference_images(&multi);
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].width, 8);
        assert_eq!(got.last().unwrap().width, 24);
    }
}
