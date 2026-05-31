//! `ZImageTurbo` — the Z-Image-turbo implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry.
//!
//! The DiT transformer ([`crate::transformer`]) and VAE decoder ([`crate::vae`]) are ported and
//! parity-tested; the prompt→`cap_feats` text encoder and the flow-match Euler scheduler are
//! not yet ported, so [`ZImageTurbo::generate`] validates the request and then reports the
//! pipeline as pending (those are the follow-on stories to the sc-2403 restructure).

use mlx_gen::{
    Capabilities, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, ModelRegistration, Progress, Result,
};

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

/// A loaded Z-Image-turbo generator.
pub struct ZImageTurbo {
    descriptor: ModelDescriptor,
}

/// Construct a [`ZImageTurbo`] from a [`LoadSpec`].
///
/// The DiT transformer + VAE assembly from the weight tree (and the text encoder / scheduler)
/// land with the pipeline-completion stories; today this establishes the registered model so
/// the contract + registry are exercised end-to-end.
pub fn load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    Ok(Box::new(ZImageTurbo {
        descriptor: descriptor(),
    }))
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
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        Err(mlx_gen::Error::Msg(
            "z_image_turbo: end-to-end generation is not yet wired — the Qwen text encoder \
             (prompt → cap_feats) and the flow-match Euler scheduler are pending ports \
             (follow-on to the sc-2403 restructure). The DiT transformer and VAE decoder are \
             ported and parity-tested."
                .into(),
        ))
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
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
    fn validate_rejects_guidance_and_bad_size() {
        let caps = descriptor().capabilities;
        // guidance on a distilled model.
        let mut req = GenerationRequest {
            guidance: Some(4.0),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // out-of-range size.
        req = GenerationRequest {
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
            conditioning: vec![Conditioning::Depth {
                image: mlx_gen::Image::default(),
            }],
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }
}
