//! Model + transform discovery — the link-time registry, i.e. the Rust stand-in for a DI
//! container's resolve-by-id. See `docs/MODEL_ARCHITECTURE.md` §4.
//!
//! A provider crate self-registers just by being linked (`inventory::submit!`); `mlx-gen` has
//! no central match statement to edit, so adding a model is purely additive. A consumer that
//! links one provider sees exactly one registration. Mirrors the worker's `payload.model` →
//! `MODEL_TARGETS` → load.

use crate::caption::{Captioner, CaptionerDescriptor};
use crate::generator::{Generator, ModelDescriptor};
use crate::runtime::LoadSpec;
use crate::train::{Trainer, TrainerDescriptor};
use crate::transform::{Transform, TransformDescriptor};
use crate::{Error, Result};

/// A generator provider's registration — `descriptor` for introspection (no weights loaded),
/// `load` to construct the model. ≈ `services.AddKeyedSingleton<IGenerator>("id", factory)`.
pub struct ModelRegistration {
    pub descriptor: fn() -> ModelDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Generator>>,
}

inventory::collect!(ModelRegistration);

/// A transform provider's registration (parallel to [`ModelRegistration`]).
pub struct TransformRegistration {
    pub descriptor: fn() -> TransformDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Transform>>,
}

inventory::collect!(TransformRegistration);

/// A trainer provider's registration (parallel to [`ModelRegistration`]) — `descriptor` for
/// introspection, `load` to construct the trainer with its (frozen) base model from a [`LoadSpec`].
pub struct TrainerRegistration {
    pub descriptor: fn() -> TrainerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Trainer>>,
}

inventory::collect!(TrainerRegistration);

/// A captioner provider's registration (parallel to [`ModelRegistration`]).
pub struct CaptionerRegistration {
    pub descriptor: fn() -> CaptionerDescriptor,
    pub load: fn(&LoadSpec) -> Result<Box<dyn Captioner>>,
}

inventory::collect!(CaptionerRegistration);

/// All registered generators (one per linked provider crate).
pub fn generators() -> impl Iterator<Item = &'static ModelRegistration> {
    inventory::iter::<ModelRegistration>.into_iter()
}

/// All registered transforms.
pub fn transforms() -> impl Iterator<Item = &'static TransformRegistration> {
    inventory::iter::<TransformRegistration>.into_iter()
}

/// All registered trainers (one per linked provider crate that supports training).
pub fn trainers() -> impl Iterator<Item = &'static TrainerRegistration> {
    inventory::iter::<TrainerRegistration>.into_iter()
}

/// All registered captioners (one per linked provider crate that supports image-to-text captioning).
pub fn captioners() -> impl Iterator<Item = &'static CaptionerRegistration> {
    inventory::iter::<CaptionerRegistration>.into_iter()
}

/// Load a generator by model id (e.g. `"z_image_turbo"`).
pub fn load(id: &str, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let reg = generators()
        .find(|r| (r.descriptor)().id == id)
        .ok_or_else(|| Error::Msg(format!("no generator registered for id '{id}'")))?;
    (reg.load)(spec)
}

/// Load a transform by id.
pub fn load_transform(id: &str, spec: &LoadSpec) -> Result<Box<dyn Transform>> {
    let reg = transforms()
        .find(|r| (r.descriptor)().id == id)
        .ok_or_else(|| Error::Msg(format!("no transform registered for id '{id}'")))?;
    (reg.load)(spec)
}

/// Load a trainer by model id (e.g. `"z_image_turbo"`) with its (frozen) base model.
pub fn load_trainer(id: &str, spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let reg = trainers()
        .find(|r| (r.descriptor)().id == id)
        .ok_or_else(|| Error::Msg(format!("no trainer registered for id '{id}'")))?;
    (reg.load)(spec)
}

/// Load a captioner by model id (e.g. `"joy_caption"`).
pub fn load_captioner(id: &str, spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
    let reg = captioners()
        .find(|r| (r.descriptor)().id == id)
        .ok_or_else(|| Error::Msg(format!("no captioner registered for id '{id}'")))?;
    (reg.load)(spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caption::{
        CaptionCapabilities, CaptionOutput, CaptionRequest, Captioner, CaptionerDescriptor,
    };
    use crate::generator::{
        Capabilities, GenerationOutput, GenerationRequest, Modality, ModelDescriptor,
    };
    use crate::media::Image;
    use crate::runtime::{Progress, WeightsSource};

    struct DummyGen {
        desc: ModelDescriptor,
    }

    impl Generator for DummyGen {
        fn descriptor(&self) -> &ModelDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &GenerationRequest) -> Result<()> {
            Ok(())
        }
        fn generate(
            &self,
            _req: &GenerationRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<GenerationOutput> {
            Ok(GenerationOutput::Images(vec![Image::default()]))
        }
    }

    fn dummy_descriptor() -> ModelDescriptor {
        ModelDescriptor {
            id: "dummy_test_model",
            family: "test",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities::default(),
        }
    }

    fn dummy_load(_spec: &LoadSpec) -> Result<Box<dyn Generator>> {
        Ok(Box::new(DummyGen {
            desc: dummy_descriptor(),
        }))
    }

    inventory::submit! {
        ModelRegistration { descriptor: dummy_descriptor, load: dummy_load }
    }

    struct DummyCaptioner {
        desc: CaptionerDescriptor,
    }

    impl Captioner for DummyCaptioner {
        fn descriptor(&self) -> &CaptionerDescriptor {
            &self.desc
        }
        fn validate(&self, _req: &CaptionRequest) -> Result<()> {
            Ok(())
        }
        fn caption(
            &self,
            _req: &CaptionRequest,
            _on_progress: &mut dyn FnMut(Progress),
        ) -> Result<CaptionOutput> {
            Ok(CaptionOutput {
                text: "caption".to_owned(),
                generated_tokens: Some(1),
                finish_reason: None,
            })
        }
    }

    fn dummy_captioner_descriptor() -> CaptionerDescriptor {
        CaptionerDescriptor {
            id: "dummy_test_captioner",
            family: "test",
            capabilities: CaptionCapabilities {
                min_image_size: 1,
                max_image_size: 4096,
                max_prompt_chars: 4000,
                max_name_chars: 120,
                max_extra_options: 16,
                max_extra_option_chars: 500,
                max_trigger_words: 32,
                max_trigger_word_chars: 120,
                max_new_tokens: 1024,
                ..Default::default()
            },
        }
    }

    fn dummy_captioner_load(_spec: &LoadSpec) -> Result<Box<dyn Captioner>> {
        Ok(Box::new(DummyCaptioner {
            desc: dummy_captioner_descriptor(),
        }))
    }

    inventory::submit! {
        CaptionerRegistration {
            descriptor: dummy_captioner_descriptor,
            load: dummy_captioner_load,
        }
    }

    #[test]
    fn registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let g = load("dummy_test_model", &spec).expect("dummy is registered");
        assert_eq!(g.descriptor().id, "dummy_test_model");
        assert_eq!(g.descriptor().modality, Modality::Image);
    }

    #[test]
    fn unknown_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load("no_such_model", &spec).is_err());
    }

    #[test]
    fn dummy_appears_in_iteration() {
        assert!(generators().any(|r| (r.descriptor)().id == "dummy_test_model"));
    }

    #[test]
    fn captioner_registry_resolves_by_id() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let c =
            load_captioner("dummy_test_captioner", &spec).expect("dummy captioner is registered");
        assert_eq!(c.descriptor().id, "dummy_test_captioner");
    }

    #[test]
    fn unknown_captioner_id_errors() {
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        assert!(load_captioner("no_such_captioner", &spec).is_err());
    }

    #[test]
    fn dummy_captioner_appears_in_iteration() {
        assert!(captioners().any(|r| (r.descriptor)().id == "dummy_test_captioner"));
    }
}
