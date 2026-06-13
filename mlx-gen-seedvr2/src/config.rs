//! SeedVR2 model configuration (the 3B default + the 7B override set).
//!
//! Mirrors the mflux reference `SeedVR2Transformer` constructor defaults
//! (`model/seedvr2_transformer/transformer.py`) and the `ModelConfig.seedvr2_3b/7b`
//! `transformer_overrides`. The VAE config is shared across both variants.

/// Diffusion-transformer hyper-parameters.
#[derive(Clone, Copy, Debug)]
pub struct DitConfig {
    pub vid_in_channels: i32,      // 33 = noise(16) + cond latent(16) + mask(1)
    pub vid_out_channels: i32,     // 16
    pub vid_dim: i32,              // 3B 2560 / 7B 3072
    pub txt_in_dim: i32,           // 5120 (precomputed neg-prompt embedding width)
    pub heads: i32,                // 3B 20 / 7B 24
    pub head_dim: i32,             // 128
    pub expand_ratio: i32,         // 4
    pub num_layers: i32,           // 3B 32 / 7B 36
    pub mm_layers: i32,            // dual-stream layers; >= this index uses shared (`.all`) weights
    pub patch_t: i32,              // 1
    pub patch_h: i32,              // 2
    pub patch_w: i32,              // 2
    pub rope_dim: i32,             // 3B 128 / 7B 64
    pub rope_on_text: bool,        // 3B true / 7B false
    pub rope_pixel: bool,          // freqs_for: 3B "lang"(false) / 7B "pixel"(true)
    pub swiglu_mlp: bool,          // 3B swiglu(true) / 7B "normal" gelu(false)
    pub use_output_ada: bool,      // 3B true / 7B false
    pub last_layer_vid_only: bool, // 3B true / 7B false
    pub norm_eps: f32,             // 1e-5
    pub window: (i32, i32, i32),   // (4,3,3)
}

impl DitConfig {
    /// SeedVR2-3B (the primary variant).
    pub fn seedvr2_3b() -> Self {
        Self {
            vid_in_channels: 33,
            vid_out_channels: 16,
            vid_dim: 2560,
            txt_in_dim: 5120,
            heads: 20,
            head_dim: 128,
            expand_ratio: 4,
            num_layers: 32,
            mm_layers: 10,
            patch_t: 1,
            patch_h: 2,
            patch_w: 2,
            rope_dim: 128,
            rope_on_text: true,
            rope_pixel: false,
            swiglu_mlp: true,
            use_output_ada: true,
            last_layer_vid_only: true,
            norm_eps: 1e-5,
            window: (4, 3, 3),
        }
    }

    /// SeedVR2-7B override set (sc-5197). Confirmed against the reference `SeedVR2Transformer`
    /// constructor + the 7B checkpoint: dim 3072 / 24 heads / 36 layers, `mm_layers=36` (every layer
    /// dual-stream — no shared `.all`), `rope_dim=64` **pixel-mode** RoPE with `rope_on_text=false`
    /// (the non-mm rope path, no temporal offset), `mlp_type="normal"` (GELU, `gelu_approx`), and no
    /// output AdaLN / no last-layer-vid-only. The `freqs` buffer is loaded from the checkpoint (pixel
    /// `linspace·π`, length `rope_dim/3/2 = 10`).
    pub fn seedvr2_7b() -> Self {
        Self {
            vid_dim: 3072,
            heads: 24,
            num_layers: 36,
            mm_layers: 36,
            rope_dim: 64,
            rope_on_text: false,
            rope_pixel: true,
            swiglu_mlp: false,
            use_output_ada: false,
            last_layer_vid_only: false,
            ..Self::seedvr2_3b()
        }
    }

    pub fn emb_dim(&self) -> i32 {
        6 * self.vid_dim
    }
}

/// 3D causal video VAE config (shared by 3B and 7B).
#[derive(Clone, Copy, Debug)]
pub struct VaeConfig {
    pub in_channels: i32,             // 3
    pub out_channels: i32,            // 3
    pub latent_channels: i32,         // 16
    pub block_out_channels: [i32; 4], // (128,256,512,512)
    pub enc_layers_per_block: i32,    // 2
    pub dec_layers_per_block: i32,    // 3
    pub temporal_down_blocks: i32,    // 2
    pub temporal_up_blocks: i32,      // 2
    pub scaling_factor: f32,          // 0.9152
    pub spatial_scale: i32,           // 8
    pub group_norm_groups: i32,       // 32
    pub group_norm_eps: f32,          // 1e-6
}

impl VaeConfig {
    pub fn seedvr2() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: [128, 256, 512, 512],
            enc_layers_per_block: 2,
            dec_layers_per_block: 3,
            temporal_down_blocks: 2,
            temporal_up_blocks: 2,
            scaling_factor: 0.9152,
            spatial_scale: 8,
            group_norm_groups: 32,
            group_norm_eps: 1e-6,
        }
    }
}
