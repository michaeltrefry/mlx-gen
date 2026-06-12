//! SAM3 vision-encoder configuration — mirrors `Sam3ViTConfig` + `Sam3VisionConfig`
//! (`transformers/models/sam3`) for the `facebook/sam3` checkpoint (epic 4910, sc-4919).
//!
//! Only the vision tower (PE ViT backbone + FPN neck) is modelled here — Phase A. The text
//! encoder, DETR detector, and mask head configs land in later phases.

/// PE ViT backbone + FPN neck hyperparameters. Defaults are the shipped `facebook/sam3` values.
#[derive(Clone, Debug)]
pub struct Sam3VisionConfig {
    // --- backbone (Sam3ViTConfig) ---
    /// Backbone embedding dim (1024).
    pub hidden_size: i32,
    /// FFN intermediate dim (4736).
    pub intermediate_size: i32,
    /// Number of transformer layers (32).
    pub num_hidden_layers: i32,
    /// Attention heads (16); `head_dim = hidden_size / num_attention_heads` (64).
    pub num_attention_heads: i32,
    /// Input channels (3).
    pub num_channels: i32,
    /// Inference image size (1008) → `image_size / patch_size` token grid (72).
    pub image_size: i32,
    /// Patch / conv-stem stride (14).
    pub patch_size: i32,
    /// Pretrain image size (336) → the position-embedding grid (`336/14 = 24`), tiled to 72.
    pub pretrain_image_size: i32,
    /// Windowed-attention window (24, in tokens).
    pub window_size: i32,
    /// Layer indices that run full (global) attention instead of windowed ([7, 15, 23, 31]).
    pub global_attn_indexes: Vec<i32>,
    /// RoPE base frequency (10000).
    pub rope_theta: f32,
    /// LayerNorm epsilon (1e-6).
    pub layer_norm_eps: f32,

    // --- neck (Sam3VisionConfig) ---
    /// FPN channel dim (256).
    pub fpn_hidden_size: i32,
    /// Per-level FPN scale factors ([4.0, 2.0, 1.0, 0.5]) over the 72² backbone grid → 288/144/72/36.
    pub scale_factors: Vec<f32>,
}

impl Default for Sam3VisionConfig {
    fn default() -> Self {
        Self::sam3()
    }
}

impl Sam3VisionConfig {
    /// The shipped `facebook/sam3` vision configuration.
    pub fn sam3() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4736,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 1008,
            patch_size: 14,
            pretrain_image_size: 336,
            window_size: 24,
            global_attn_indexes: vec![7, 15, 23, 31],
            rope_theta: 10000.0,
            layer_norm_eps: 1e-6,
            fpn_hidden_size: 256,
            scale_factors: vec![4.0, 2.0, 1.0, 0.5],
        }
    }

    /// `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Token grid side at inference (`image_size / patch_size`).
    pub fn grid(&self) -> i32 {
        self.image_size / self.patch_size
    }

    /// Position-embedding grid side (`pretrain_image_size / patch_size`).
    pub fn pretrain_grid(&self) -> i32 {
        self.pretrain_image_size / self.patch_size
    }
}

/// DETR detector configuration (encoder + decoder + presence + scoring). The shared working width
/// is `hidden_size` (256); the vision/text features are already projected to it upstream.
#[derive(Clone, Debug)]
pub struct Sam3DetrConfig {
    /// Working width of the DETR stack (256).
    pub hidden_size: i32,
    /// FFN intermediate dim (2048).
    pub intermediate_size: i32,
    /// Attention heads (8); `head_dim = hidden_size / num_attention_heads` (32).
    pub num_attention_heads: i32,
    /// Encoder layers (6).
    pub num_encoder_layers: i32,
    /// Decoder layers (6).
    pub num_decoder_layers: i32,
    /// Object queries (200).
    pub num_queries: i32,
    /// LayerNorm epsilon (1e-5).
    pub layer_norm_eps: f32,
    /// Presence-logit clamp magnitude (10.0).
    pub presence_clamp: f32,
    /// Dot-product scoring clamp magnitude (12.0).
    pub score_clamp: f32,
}

impl Default for Sam3DetrConfig {
    fn default() -> Self {
        Self::sam3()
    }
}

impl Sam3DetrConfig {
    /// The shipped `facebook/sam3` DETR configuration.
    pub fn sam3() -> Self {
        Self {
            hidden_size: 256,
            intermediate_size: 2048,
            num_attention_heads: 8,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            num_queries: 200,
            layer_norm_eps: 1e-5,
            presence_clamp: 10.0,
            score_clamp: 12.0,
        }
    }

    /// `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

/// Geometry/exemplar prompt-encoder configuration — mirrors `Sam3GeometryEncoderConfig`
/// (`transformers/models/sam3`). The box/point **PVS** prompt path (sc-4923). Numerically the same
/// working width as the DETR stack (256/8 heads/2048 FFN), with a distinct `roi_size`.
#[derive(Clone, Debug)]
pub struct Sam3GeometryConfig {
    /// Working width (256).
    pub hidden_size: i32,
    /// FFN intermediate dim (2048).
    pub intermediate_size: i32,
    /// Transformer layers (3).
    pub num_layers: i32,
    /// Attention heads (8); `head_dim = hidden_size / num_attention_heads` (32).
    pub num_attention_heads: i32,
    /// ROI-align output side for box pooling (7).
    pub roi_size: i32,
    /// LayerNorm epsilon (1e-6).
    pub layer_norm_eps: f32,
}

impl Default for Sam3GeometryConfig {
    fn default() -> Self {
        Self::sam3()
    }
}

impl Sam3GeometryConfig {
    /// The shipped `facebook/sam3` geometry-encoder configuration.
    pub fn sam3() -> Self {
        Self {
            hidden_size: 256,
            intermediate_size: 2048,
            num_layers: 3,
            num_attention_heads: 8,
            roi_size: 7,
            layer_norm_eps: 1e-6,
        }
    }

    /// `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}

/// CLIP text-encoder configuration (the `Sam3Config.text_config`, a CLIP-H text tower) + the SAM3
/// `text_projection` output dim. Defaults are the shipped `facebook/sam3` values.
#[derive(Clone, Debug)]
pub struct Sam3TextConfig {
    /// CLIP hidden dim (1024).
    pub hidden_size: i32,
    /// FFN intermediate dim (4096).
    pub intermediate_size: i32,
    /// Transformer layers (24).
    pub num_hidden_layers: i32,
    /// Attention heads (16).
    pub num_attention_heads: i32,
    /// BPE vocab size (49408).
    pub vocab_size: i32,
    /// Max sequence length / position-embedding capacity (32).
    pub max_position_embeddings: i32,
    /// LayerNorm epsilon (1e-5 — note: distinct from the vision encoder's 1e-6).
    pub layer_norm_eps: f32,
    /// SAM3 `text_projection` output dim — the prompt/conditioning width fed to the DETR stack (256).
    pub projection_dim: i32,
    /// Padding / EOS token id (49407); the CLIP tokenizer pads to `max_position_embeddings` with it.
    pub pad_token_id: i32,
}

impl Default for Sam3TextConfig {
    fn default() -> Self {
        Self::sam3()
    }
}

impl Sam3TextConfig {
    /// The shipped `facebook/sam3` text configuration.
    pub fn sam3() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            vocab_size: 49408,
            max_position_embeddings: 32,
            layer_norm_eps: 1e-5,
            projection_dim: 256,
            pad_token_id: 49407,
        }
    }

    /// `head_dim = hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }
}
