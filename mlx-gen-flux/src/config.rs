//! FLUX.1 family configuration, lifted from the frozen Python mflux fork's
//! `ModelConfig.{schnell,dev}` and `FluxWeightDefinition.get_tokenizers`.

use mlx_gen::{Capabilities, ConditioningKind, Modality, ModelDescriptor};

pub const FLUX1_SCHNELL_ID: &str = "flux1_schnell";
pub const FLUX1_DEV_ID: &str = "flux1_dev";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
pub const DEFAULT_GUIDANCE: f32 = 3.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxVariant {
    Schnell,
    Dev,
}

impl FluxVariant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Schnell => FLUX1_SCHNELL_ID,
            Self::Dev => FLUX1_DEV_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            Self::Schnell => "black-forest-labs/FLUX.1-schnell",
            Self::Dev => "black-forest-labs/FLUX.1-dev",
        }
    }

    pub fn default_steps(self) -> u32 {
        match self {
            Self::Schnell => 4,
            Self::Dev => 25,
        }
    }

    pub fn max_sequence_length(self) -> usize {
        match self {
            Self::Schnell => 256,
            Self::Dev => 512,
        }
    }

    pub fn supports_guidance(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn requires_sigma_shift(self) -> bool {
        matches!(self, Self::Dev)
    }

    pub fn descriptor(self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id(),
            family: "flux",
            modality: Modality::Image,
            capabilities: Capabilities {
                supports_negative_prompt: false,
                supports_guidance: self.supports_guidance(),
                supports_true_cfg: false,
                // Base FLUX.1 txt2img only for this story; Redux/Depth/Fill/Control are later variants.
                conditioning: vec![ConditioningKind::Reference],
                supports_lora: true,
                supports_lokr: false,
                samplers: Vec::new(),
                schedulers: vec!["linear"],
                min_size: 256,
                max_size: 2048,
                max_count: 8,
                mac_only: true,
                supports_kv_cache: false,
                requires_sigma_shift: self.requires_sigma_shift(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FluxTokenizerKind {
    Clip,
    T5,
}

impl FluxTokenizerKind {
    pub fn subdir(self) -> &'static str {
        match self {
            Self::Clip => "tokenizer",
            Self::T5 => "tokenizer_2",
        }
    }

    pub fn max_length(self, variant: FluxVariant) -> usize {
        match self {
            Self::Clip => 77,
            Self::T5 => variant.max_sequence_length(),
        }
    }

    pub fn pad_token_id(self) -> i32 {
        match self {
            // CLIP's `<|endoftext|>` id in the FLUX.1 tokenizer.
            Self::Clip => 49407,
            // T5's `<pad>` id.
            Self::T5 => 0,
        }
    }
}
