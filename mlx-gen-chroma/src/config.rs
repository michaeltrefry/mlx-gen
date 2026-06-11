//! Chroma family configuration (epic 3531), lifted from the diffusers `ChromaTransformer2DModel` /
//! `ChromaPipeline` reference (the parity source — mflux has no Chroma port).
//!
//! Chroma is a FLUX.1-schnell-derived DiT: same MMDiT skeleton (19 dual + 38 single blocks,
//! `inner_dim = 3072`, 24 heads × 128, FluxPosEmbed RoPE), but the FLUX `time_text_embed`
//! (timestep + guidance + pooled-CLIP) is replaced by a `distilled_guidance_layer` (Approximator)
//! that generates *all* per-block modulation, conditioning is **T5-XXL only** (no CLIP / no pooled),
//! and attention is **masked** by the T5 padding mask. Generation uses **true CFG** (real negative
//! prompt), not FLUX's distilled guidance.

use mlx_gen::{Capabilities, Modality, ModelDescriptor, Quant};

pub const CHROMA1_HD_ID: &str = "chroma1_hd";
pub const CHROMA1_BASE_ID: &str = "chroma1_base";
pub const CHROMA1_FLASH_ID: &str = "chroma1_flash";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;

/// The base flow-match sampler name. An unset `req.sampler` resolves to this.
pub const DEFAULT_SAMPLER: &str = "flow_match";

/// T5 sequence length Chroma conditions on (diffusers `_get_t5_prompt_embeds(max_sequence_length=512)`).
pub const MAX_SEQUENCE_LENGTH: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChromaVariant {
    /// `lodestones/Chroma1-HD` — the high-detail full-CFG model.
    Hd,
    /// `lodestones/Chroma1-Base` — the base full-CFG model.
    Base,
    /// `lodestones/Chroma1-Flash` — the few-step distilled model.
    Flash,
}

impl ChromaVariant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Hd => CHROMA1_HD_ID,
            Self::Base => CHROMA1_BASE_ID,
            Self::Flash => CHROMA1_FLASH_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        match self {
            Self::Hd => "lodestones/Chroma1-HD",
            Self::Base => "lodestones/Chroma1-Base",
            Self::Flash => "lodestones/Chroma1-Flash",
        }
    }

    /// Default denoise steps. HD/Base run full CFG; Flash is a few-step distilled checkpoint. Exact
    /// values are reconciled against each repo's scheduler/README in sc-3840.
    pub fn default_steps(self) -> u32 {
        match self {
            Self::Hd | Self::Base => 28,
            Self::Flash => 8,
        }
    }

    /// Default true-CFG scale. Flash is distilled toward CFG≈1 (single forward); refined in sc-3840.
    pub fn default_true_cfg(self) -> f32 {
        match self {
            Self::Hd | Self::Base => 4.0,
            Self::Flash => 1.0,
        }
    }

    /// Static flow-match time `shift` (diffusers `FlowMatchEulerDiscreteScheduler`,
    /// `use_dynamic_shifting=false`): `σ' = shift·σ / (1 + (shift-1)·σ)`. HD's `scheduler_config.json`
    /// pins `shift=3.0`; Flash pins `1.0`. **Base** ships `use_beta_sigmas=true` (a beta-spaced
    /// schedule, not linspace) — handled in sc-3840; this `1.0` is a placeholder for Base.
    pub fn sigma_shift(self) -> f32 {
        match self {
            Self::Hd => 3.0,
            Self::Base | Self::Flash => 1.0,
        }
    }

    /// Base ships `use_beta_sigmas=true` — a beta-distribution sigma spacing (see [`crate::beta`])
    /// instead of the shifted linspace HD/Flash use.
    pub fn use_beta_sigmas(self) -> bool {
        matches!(self, Self::Base)
    }

    pub fn descriptor(self) -> ModelDescriptor {
        ModelDescriptor {
            id: self.id(),
            family: "chroma",
            backend: "mlx",
            modality: Modality::Image,
            capabilities: Capabilities {
                // Chroma uses real classifier-free guidance with a true negative prompt.
                supports_negative_prompt: true,
                // No distilled guidance-scalar embedding (`guidance_embeds=false`); CFG is `true_cfg`.
                supports_guidance: false,
                supports_true_cfg: true,
                // v1 = T2I only. ControlNet / IP-Adapter / img2img are later ports.
                conditioning: vec![],
                supported_quants: &[Quant::Q4, Quant::Q8],
                // LoRA/LoKr via the shared core adapter seam (sc-3842), over the diffusers/peft
                // (and kohya) `transformer_blocks.*`/`single_transformer_blocks.*` paths.
                supports_lora: true,
                supports_lokr: true,
                samplers: vec![DEFAULT_SAMPLER],
                schedulers: vec!["linear"],
                min_size: 256,
                max_size: 2048,
                max_count: 8,
                mac_only: true,
                supports_kv_cache: false,
                // FLUX-style flow-match sigma shift (calculate_shift) is applied in the generate path.
                requires_sigma_shift: true,
            },
        }
    }
}

/// Static dims of `ChromaTransformer2DModel` (diffusers defaults — identical across the three
/// variants; only the weights and sampling profile differ).
#[derive(Clone, Copy, Debug)]
pub struct ChromaTransformerConfig {
    pub in_channels: usize,
    pub num_layers: usize,
    pub num_single_layers: usize,
    pub num_attention_heads: usize,
    pub attention_head_dim: usize,
    pub joint_attention_dim: usize,
    pub axes_dims_rope: [usize; 3],
    pub approximator_num_channels: usize,
    pub approximator_hidden_dim: usize,
    pub approximator_layers: usize,
}

impl Default for ChromaTransformerConfig {
    fn default() -> Self {
        Self {
            in_channels: 64,
            num_layers: 19,
            num_single_layers: 38,
            num_attention_heads: 24,
            attention_head_dim: 128,
            joint_attention_dim: 4096,
            axes_dims_rope: [16, 56, 56],
            approximator_num_channels: 64,
            approximator_hidden_dim: 5120,
            approximator_layers: 5,
        }
    }
}

impl ChromaTransformerConfig {
    pub fn inner_dim(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// The modulation index length `out_dim = 3·N_single + 2·6·N_double + 2` — the number of
    /// `[inner_dim]` modulation rows the Approximator emits (`pooled_temb` has shape `[B, out_dim, inner]`).
    pub fn mod_index_len(&self) -> usize {
        3 * self.num_single_layers + 2 * 6 * self.num_layers + 2
    }
}
