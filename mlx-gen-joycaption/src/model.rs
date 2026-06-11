//! JoyCaption provider registration and image-to-text caption path.

use mlx_gen::caption::joycaption::language::{
    expand_joycaption_image_tokens, input_arrays_from_ids, splice_image_features, LlamaConfig,
    LlamaDecoder, LlavaProjector,
};
use mlx_gen::caption::joycaption::vision::{
    joycaption_vision_features, SiglipImageProcessor, SiglipVisionConfig, SiglipVisionTower,
};
use mlx_gen::caption::joycaption::{
    self, decode_generated, encode_chat_prompt, load_tokenizer_from_spec, IMAGE_TOKEN_ID,
    JOY_CAPTION_FAMILY, JOY_CAPTION_MODEL_ID,
};
use mlx_gen::runtime::Precision;
use mlx_gen::weights::Weights;
use mlx_gen::{
    gen_core, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor, CaptionerRegistration,
    Error, LoadSpec, Progress, Result, WeightsSource,
};
use mlx_rs::Array;

pub fn descriptor() -> CaptionerDescriptor {
    CaptionerDescriptor {
        id: JOY_CAPTION_MODEL_ID,
        family: JOY_CAPTION_FAMILY,
        capabilities: joycaption::capabilities(),
    }
}

pub fn load(spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
    Ok(Box::new(load_joycaption(spec)?))
}

pub fn load_joycaption(spec: &LoadSpec) -> Result<JoyCaption> {
    validate_load_spec(spec)?;

    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "joycaption expects a Hugging Face snapshot directory with tokenizer.json and \
                 sharded .safetensors, not a single .safetensors file"
                    .to_owned(),
            ))
        }
    };

    let tokenizer = load_tokenizer_from_spec(spec)?;
    let weights = Weights::from_dir(root)?;
    Ok(JoyCaption {
        descriptor: descriptor(),
        tokenizer,
        image_processor: SiglipImageProcessor::default(),
        vision: SiglipVisionTower::from_weights(
            &weights,
            "vision_tower.vision_model",
            SiglipVisionConfig::default(),
        )?,
        projector: LlavaProjector::from_weights(&weights, "multi_modal_projector")?,
        llama: LlamaDecoder::from_weights(&weights, "language_model", LlamaConfig::default())?,
    })
}

fn validate_load_spec(spec: &LoadSpec) -> Result<()> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "joycaption: only dense bf16 loading is validated".to_owned(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "joycaption: quantized loading is not validated".to_owned(),
        ));
    }
    if spec.control.is_some() {
        return Err(Error::Msg(
            "joycaption: control weights are not supported".to_owned(),
        ));
    }
    if spec.ip_adapter.is_some() {
        return Err(Error::Msg(
            "joycaption: IP-Adapter weights are not supported".to_owned(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "joycaption: LoRA/LoKr adapters are not supported".to_owned(),
        ));
    }
    Ok(())
}

pub struct JoyCaption {
    descriptor: CaptionerDescriptor,
    tokenizer: mlx_gen::tokenizer::TextTokenizer,
    image_processor: SiglipImageProcessor,
    vision: SiglipVisionTower,
    projector: LlavaProjector,
    llama: LlamaDecoder,
}

impl JoyCaption {
    fn prompt_embeds(&self, req: &CaptionRequest) -> Result<(Vec<i32>, Array)> {
        if req.cancel.is_cancelled() {
            return Err(Error::Msg("caption generation cancelled".to_owned()));
        }

        let pixels = self.image_processor.preprocess(&req.image)?;
        let vision_output = self.vision.forward(&pixels)?;
        let vision_features = joycaption_vision_features(&vision_output)?;
        let projected = self.projector.forward(&vision_features)?;

        if req.cancel.is_cancelled() {
            return Err(Error::Msg("caption generation cancelled".to_owned()));
        }

        let ids = encode_chat_prompt(&self.tokenizer, &req.prompt)?;
        let ids = expand_joycaption_image_tokens(&ids);
        let (input_ids, _) = input_arrays_from_ids(&ids);
        let embeds = self.llama.embed(&input_ids)?;
        let spliced = splice_image_features(&embeds, &input_ids, &projected, IMAGE_TOKEN_ID)?;
        Ok((ids, spliced))
    }
}

fn normalized_request(req: &CaptionRequest) -> CaptionRequest {
    let mut out = req.clone();
    if out.prompt.trim().is_empty() {
        out.prompt = joycaption::build_prompt(&out.options);
    }
    out
}

impl Captioner for JoyCaption {
    fn descriptor(&self) -> &CaptionerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &CaptionRequest) -> gen_core::Result<()> {
        let req = normalized_request(req);
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, &req)
    }

    fn caption(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<CaptionOutput> {
        self.caption_impl(req, on_progress).map_err(Into::into)
    }
}

impl JoyCaption {
    /// The rich-`Result` body behind [`Captioner::caption`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn caption_impl(
        &self,
        req: &CaptionRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<CaptionOutput> {
        let req = normalized_request(req);
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, &req)?;

        let (ids, embeds) = self.prompt_embeds(&req)?;
        on_progress(Progress::Step {
            current: 1,
            total: 2,
        });

        let generation =
            self.llama
                .generate_from_embeds(&ids, &embeds, req.sampling, &req.cancel)?;
        on_progress(Progress::Step {
            current: 2,
            total: 2,
        });

        let token_ids = generation
            .token_ids
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    Error::Msg(format!("joycaption: generated negative token id {id}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let text = decode_generated(&self.tokenizer, &token_ids)?;
        Ok(CaptionOutput {
            text,
            generated_tokens: Some(token_ids.len() as u32),
            finish_reason: Some(generation.finish_reason),
        })
    }
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load`] into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Captioner>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    CaptionerRegistration { descriptor, load: load_registered }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use mlx_gen::caption::CaptionOptions;
    use mlx_gen::media::Image;
    use mlx_gen::runtime::{AdapterKind, AdapterSpec, Quant};

    fn image() -> Image {
        Image {
            width: 384,
            height: 384,
            pixels: vec![127; 384 * 384 * 3],
        }
    }

    fn request() -> CaptionRequest {
        CaptionRequest {
            image: image(),
            prompt: "Write a short caption.".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn descriptor_advertises_joycaption_limits() {
        let d = descriptor();
        assert_eq!(d.id, JOY_CAPTION_MODEL_ID);
        assert_eq!(d.family, JOY_CAPTION_FAMILY);
        assert!(d.capabilities.supports_custom_prompt);
        assert!(d.capabilities.supports_low_vram);
        assert!(d.capabilities.mac_only);
        assert_eq!(d.capabilities.max_new_tokens, 1024);
        assert!(d.capabilities.caption_types.contains(&"Straightforward"));
        assert!(d.capabilities.caption_lengths.contains(&"medium-length"));
    }

    #[test]
    fn validation_accepts_empty_prompt_when_options_can_render_it() {
        let req = CaptionRequest {
            prompt: String::new(),
            options: CaptionOptions {
                caption_type: "Straightforward".to_owned(),
                caption_length: "short".to_owned(),
                ..Default::default()
            },
            ..request()
        };
        let req = normalized_request(&req);
        assert_eq!(
            req.prompt,
            joycaption::build_prompt(&CaptionOptions {
                caption_type: "Straightforward".to_owned(),
                caption_length: "short".to_owned(),
                ..Default::default()
            })
        );
        assert!(descriptor()
            .capabilities
            .validate_request(JOY_CAPTION_MODEL_ID, &req)
            .is_ok());
    }

    #[test]
    fn load_rejects_unsupported_specs_before_touching_disk() {
        let root = PathBuf::from("/nonexistent/joycaption");

        let mut fp32 = LoadSpec::new(WeightsSource::Dir(root.clone()));
        fp32.precision = Precision::Fp32;
        assert!(load_joycaption(&fp32)
            .err()
            .expect("fp32 specs are rejected before disk access")
            .to_string()
            .contains("dense bf16"));

        let q4 = LoadSpec::new(WeightsSource::Dir(root.clone())).with_quant(Quant::Q4);
        assert!(load_joycaption(&q4)
            .err()
            .expect("quantized specs are rejected before disk access")
            .to_string()
            .contains("quantized"));

        let adapters =
            LoadSpec::new(WeightsSource::Dir(root)).with_adapters(vec![AdapterSpec::new(
                PathBuf::from("adapter.safetensors"),
                1.0,
                AdapterKind::Lora,
            )]);
        assert!(load_joycaption(&adapters)
            .err()
            .expect("adapter specs are rejected before disk access")
            .to_string()
            .contains("adapters"));
    }

    #[test]
    fn load_rejects_single_file_snapshot() {
        let spec = LoadSpec::new(WeightsSource::File("/unused.safetensors".into()));
        let err = load_joycaption(&spec)
            .err()
            .expect("file spec rejected")
            .to_string();
        assert!(err.contains("snapshot directory"), "{err}");
    }
}
