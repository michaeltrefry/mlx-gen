//! FLUX.2-klein configuration, lifted from the frozen Python mflux fork's
//! `ModelConfig.flux2_klein_*` (`models/common/config/model_config.py`) and the FLUX.2
//! transformer / Qwen3 text-encoder / VAE constructors.
//!
//! The config is **dimension-parametric**: the same Rust code runs the real 9b model, the 4b
//! variant, and tiny parity fixtures. The fork distinguishes the variants only by a handful of
//! `transformer_overrides` / `text_encoder_overrides` (block/head counts, the Qwen3 hidden /
//! intermediate sizes); everything else is shared.

use mlx_gen::{Capabilities, ConditioningKind, Modality, ModelDescriptor};

pub const FLUX2_KLEIN_9B_ID: &str = "flux2_klein_9b";
pub const FLUX2_KLEIN_9B_EDIT_ID: &str = "flux2_klein_9b_edit";

pub const DEFAULT_WIDTH: u32 = 1024;
pub const DEFAULT_HEIGHT: u32 = 1024;
/// Distilled klein default; the fork generates in 4 steps at guidance 1.0.
pub const DEFAULT_STEPS: u32 = 4;
/// Distilled klein runs at guidance 1.0 (no CFG). Base (non-distilled) variants allow >1.0.
pub const DEFAULT_GUIDANCE: f32 = 1.0;

/// The FLUX.2-klein variants this crate targets. 9b is the story target; the enum keeps the
/// door open for 4b (a near-free addition — only the dims in [`Flux2Config`] change).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flux2Variant {
    /// FLUX.2-klein-9b, distilled, txt2img.
    Klein9b,
    /// FLUX.2-klein-9b, distilled, image-conditioned edit (single + multi reference).
    Klein9bEdit,
}

impl Flux2Variant {
    pub fn id(self) -> &'static str {
        match self {
            Self::Klein9b => FLUX2_KLEIN_9B_ID,
            Self::Klein9bEdit => FLUX2_KLEIN_9B_EDIT_ID,
        }
    }

    pub fn hf_model(self) -> &'static str {
        // Both txt2img and edit load the same 9b snapshot; the edit path differs only in how
        // reference images are tokenized into extra sequence tokens.
        "black-forest-labs/FLUX.2-klein-9B"
    }

    pub fn is_edit(self) -> bool {
        matches!(self, Self::Klein9bEdit)
    }

    /// The dimension-parametric model config for this variant.
    pub fn config(self) -> Flux2Config {
        Flux2Config::klein_9b()
    }

    pub fn descriptor(self) -> ModelDescriptor {
        // Both variants accept a single `Reference`, but the semantics differ by variant: for the
        // edit variant it is the reference image concatenated as extra sequence tokens (sc-2346);
        // for txt2img it is an **img2img** init image seeding the latents via the noise blend
        // (sc-2644). Multi-reference edit (`MultiReference`) is sc-2645. Advertise what this port
        // delivers, no more, no less.
        let conditioning = vec![ConditioningKind::Reference];
        ModelDescriptor {
            id: self.id(),
            family: "flux2",
            modality: Modality::Image,
            capabilities: Capabilities {
                supports_negative_prompt: false,
                // klein is distilled (guidance 1.0); base variants would flip this on. The
                // fork's `supports_guidance` is True, but the distilled klein the story targets
                // runs CFG-free, so we expose guidance but default it to 1.0.
                supports_guidance: true,
                supports_true_cfg: false,
                conditioning,
                // Transformer-only LoRA/LoKr (sc-2646): both variants share the `Flux2Transformer`,
                // which hosts the adapters; the VAE + Qwen3 TE are not adapter targets.
                supports_lora: true,
                supports_lokr: true,
                samplers: Vec::new(),
                schedulers: vec!["flow_match_euler"],
                min_size: 256,
                max_size: 2048,
                max_count: 8,
                mac_only: true,
                supports_kv_cache: false,
                // FLUX.2 uses the empirical-mu shifted flow-match schedule.
                requires_sigma_shift: true,
            },
        }
    }
}

/// Dimension-parametric FLUX.2 model dimensions. Field values come from the fork's
/// `ModelConfig` + the FLUX.2 module constructors; the 9b values are the story target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flux2Config {
    // --- MMDiT transformer ---
    /// Double (joint img+txt) blocks. 9b: 8, 4b: 5.
    pub num_double_layers: usize,
    /// Single (fused parallel attention+SwiGLU) blocks. 9b: 24, 4b: 20.
    pub num_single_layers: usize,
    /// Attention heads. 9b: 32, 4b: 24.
    pub num_heads: usize,
    /// Per-head dim (constant across variants). `inner_dim = num_heads * head_dim`.
    pub head_dim: usize,
    /// Latent channels entering/leaving the transformer = `num_latent_channels * 4` (2×2 patch).
    pub in_channels: usize,
    pub out_channels: usize,
    /// Text-embedding width entering the joint blocks = `3 * te_hidden_size` (the concat of the
    /// three Qwen3 hidden-state layers). 9b: 12288, 4b: 7680.
    pub joint_attention_dim: usize,
    /// Single-block SwiGLU expansion ratio (mlp_hidden = mlp_ratio * inner_dim).
    pub mlp_ratio: f32,
    /// Sinusoidal timestep-embedding width feeding `time_guidance_embed.linear_1` (klein: 256).
    pub timestep_channels: usize,

    // --- 4-axis RoPE over ids (t, h, w, layer) ---
    pub axes_dim: [usize; 4],
    pub rope_theta: f32,

    // --- Qwen3 text encoder (consumed in S1; carried here for variant identity) ---
    pub te_hidden_size: usize,
    pub te_intermediate_size: usize,
    /// Concatenated hidden-state layers forming `prompt_embeds` (`joint_attention_dim` wide).
    pub te_out_layers: [usize; 3],
    pub max_sequence_length: usize,

    // --- VAE / latent geometry ---
    pub num_latent_channels: usize,
    pub vae_scale_factor: usize,
}

impl Flux2Config {
    /// FLUX.2-klein-9b (the story target).
    pub fn klein_9b() -> Self {
        Self {
            num_double_layers: 8,
            num_single_layers: 24,
            num_heads: 32,
            head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            joint_attention_dim: 12288,
            mlp_ratio: 3.0,
            timestep_channels: 256,
            axes_dim: [32, 32, 32, 32],
            rope_theta: 2000.0,
            te_hidden_size: 4096,
            te_intermediate_size: 12288,
            te_out_layers: [9, 18, 27],
            max_sequence_length: 512,
            num_latent_channels: 32,
            vae_scale_factor: 8,
        }
    }

    /// `num_heads * head_dim` — the transformer inner width (9b: 4096).
    pub fn inner_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klein_9b_dims_match_fork() {
        let c = Flux2Config::klein_9b();
        assert_eq!(c.num_double_layers, 8);
        assert_eq!(c.num_single_layers, 24);
        assert_eq!(c.num_heads, 32);
        assert_eq!(c.inner_dim(), 4096);
        assert_eq!(c.joint_attention_dim, 3 * c.te_hidden_size);
        assert_eq!(c.in_channels, c.num_latent_channels * 4);
        // RoPE axes sum to the head dim; each axis emits dim/2 freqs → cos/sin width head_dim/2.
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim);
    }

    #[test]
    fn descriptors_have_expected_ids() {
        assert_eq!(Flux2Variant::Klein9b.id(), FLUX2_KLEIN_9B_ID);
        assert_eq!(Flux2Variant::Klein9bEdit.id(), FLUX2_KLEIN_9B_EDIT_ID);
        assert!(Flux2Variant::Klein9bEdit.is_edit());
        assert!(!Flux2Variant::Klein9b.is_edit());
    }
}
