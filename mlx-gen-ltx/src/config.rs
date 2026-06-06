//! LTX-2.3 model configuration — **config-driven** from the model's `embedded_config.json`,
//! mirroring the reference `generate_av.py` build logic (lines 1464–1529 of the
//! `mlx-video-with-audio` package).
//!
//! The shipped LTX-2.3 models are `AVTransformer3DModel`s: gated attention, adaLN coefficient 9,
//! **no** PixArt caption-projection linears (so `caption_channels` is the connector output
//! `connector_heads × connector_head_dim = 4096`, not the 2.0 default 3840), and an 8-layer
//! learnable-register connector. The 2.0 `generate.py` path hardcodes a different (non-gated,
//! coeff-6, caption-proj-true, 3840) config and cannot run against these checkpoints — hence "read
//! `embedded_config.json`, don't hardcode 2.0 values" (sc-2679 S0).
//!
//! This core is **VideoOnly**: only the video-stack transformer fields are consumed by the
//! denoise path. The audio + connector fields are read here (so the reader is complete and the
//! sibling slices reuse it) but are inert for T2V.

use std::path::Path;

use serde_json::Value;

use mlx_gen::{Error, Result};

/// Rotary-embedding layout. LTX-2.3 uses [`RopeType::Split`] (the 2.0 default is interleaved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeType {
    Interleaved,
    Split,
}

impl RopeType {
    fn from_str(s: &str) -> RopeType {
        match s {
            "split" => RopeType::Split,
            _ => RopeType::Interleaved,
        }
    }
}

/// The full LTX transformer config. Dimension-parametric: every field is read from
/// `embedded_config.json` where present, falling back to the reference's hardcoded defaults.
#[derive(Clone, Debug)]
pub struct LtxConfig {
    // --- Video transformer ---
    pub num_attention_heads: i32,
    pub attention_head_dim: i32,
    pub in_channels: i32,
    pub out_channels: i32,
    pub num_layers: i32,
    pub cross_attention_dim: i32,
    /// Input dim of the text features entering cross-attention. When both caption-projection
    /// linears are absent (LTX-2.3) this equals the connector output `conn_heads × conn_head_dim`.
    pub caption_channels: i32,
    pub caption_projection_first_linear: bool,
    pub caption_projection_second_linear: bool,
    /// adaLN-single scale_shift_table row count: **9** for the gated family, **6** otherwise.
    pub adaln_embedding_coefficient: i32,
    pub apply_gated_attention: bool,
    /// `cross_attention_adaln=true` for 2.3 (the per-block `scale_shift_table` carries the extra
    /// rows 6..9 used by the text cross-attention; see transformer.py `v_has_ca_ada`).
    pub cross_attention_adaln: bool,
    pub norm_eps: f64,

    // --- Positional / RoPE ---
    pub positional_embedding_theta: f64,
    pub positional_embedding_max_pos: [i32; 3],
    pub use_middle_indices_grid: bool,
    pub rope_type: RopeType,
    pub double_precision_rope: bool,
    pub timestep_scale_multiplier: i32,

    // --- Connector (S1: Embeddings1DConnector) — read here, consumed by the TE slice ---
    pub use_embeddings_connector: bool,
    pub connector_num_layers: i32,
    pub connector_num_attention_heads: i32,
    pub connector_attention_head_dim: i32,
    pub connector_num_learnable_registers: i32,
    pub connector_positional_embedding_max_pos: i32,
    pub connector_apply_gated_attention: bool,

    // --- Audio stack (sc-2684 AudioVideo) ---
    pub audio_num_attention_heads: i32,
    pub audio_attention_head_dim: i32,
    pub audio_cross_attention_dim: i32,
    pub audio_caption_channels: i32,
    /// Audio latent flattened channels = `latent_channels(8) × mel_bins(16)` = 128.
    pub audio_in_channels: i32,
    pub audio_out_channels: i32,
    /// Audio text-feature connector dims (its own `Embeddings1DConnector`; 32 × 64 = 2048).
    pub audio_connector_num_attention_heads: i32,
    pub audio_connector_attention_head_dim: i32,
    /// 1-D audio RoPE max position (`audio_positional_embedding_max_pos = [20]` → the single `[0]`).
    pub audio_positional_embedding_max_pos: i32,
    /// Cross-modal gate timestep multiplier (`av_ca_timestep_scale_multiplier`, 1000 in 2.3).
    pub av_ca_timestep_scale_multiplier: i32,
}

impl LtxConfig {
    /// Video inner dimension `heads × head_dim` (4096 for LTX-2.3).
    pub fn inner_dim(&self) -> i32 {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Audio inner dimension `audio_heads × audio_head_dim` (2048 for LTX-2.3).
    pub fn audio_inner_dim(&self) -> i32 {
        self.audio_num_attention_heads * self.audio_attention_head_dim
    }

    /// Cross-modal 1-D RoPE max position = `max(video_max_pos[0], audio_max_pos[0])`
    /// (`LTXModel.__init__`'s `cross_pe_max_pos`). Both are 20 for LTX-2.3.
    pub fn cross_pe_max_pos(&self) -> i32 {
        self.positional_embedding_max_pos[0].max(self.audio_positional_embedding_max_pos)
    }

    /// The reference 2.0 `generate.py` defaults (non-gated, coeff-6, caption-proj present,
    /// caption_channels 3840, interleaved-default overridden to split by `generate.py`). Used as
    /// the fallback when no `embedded_config.json` is present; LTX-2.3 overrides most of these.
    pub fn video_only_defaults() -> Self {
        LtxConfig {
            num_attention_heads: 32,
            attention_head_dim: 128,
            in_channels: 128,
            out_channels: 128,
            num_layers: 48,
            cross_attention_dim: 4096,
            caption_channels: 3840,
            caption_projection_first_linear: true,
            caption_projection_second_linear: true,
            adaln_embedding_coefficient: 6,
            apply_gated_attention: false,
            cross_attention_adaln: false,
            norm_eps: 1e-6,
            positional_embedding_theta: 10000.0,
            positional_embedding_max_pos: [20, 2048, 2048],
            use_middle_indices_grid: true,
            rope_type: RopeType::Split,
            double_precision_rope: true,
            timestep_scale_multiplier: 1000,
            use_embeddings_connector: false,
            connector_num_layers: 8,
            connector_num_attention_heads: 32,
            connector_attention_head_dim: 128,
            connector_num_learnable_registers: 128,
            connector_positional_embedding_max_pos: 4096,
            connector_apply_gated_attention: false,
            audio_num_attention_heads: 32,
            audio_attention_head_dim: 64,
            audio_cross_attention_dim: 2048,
            audio_caption_channels: 3840,
            audio_in_channels: 128,
            audio_out_channels: 128,
            audio_connector_num_attention_heads: 32,
            audio_connector_attention_head_dim: 64,
            audio_positional_embedding_max_pos: 20,
            av_ca_timestep_scale_multiplier: 1000,
        }
    }

    /// Build the config from the `transformer` block of a parsed `embedded_config.json`,
    /// reproducing `generate_av.py`'s field resolution exactly.
    pub fn from_embedded_transformer(t: &Value) -> Self {
        let mut cfg = Self::video_only_defaults();

        // Plain dimension-parametric reads (default = the reference's hardcoded value).
        cfg.num_attention_heads = get_i32(t, "num_attention_heads", cfg.num_attention_heads);
        cfg.attention_head_dim = get_i32(t, "attention_head_dim", cfg.attention_head_dim);
        cfg.in_channels = get_i32(t, "in_channels", cfg.in_channels);
        cfg.out_channels = get_i32(t, "out_channels", cfg.out_channels);
        cfg.num_layers = get_i32(t, "num_layers", cfg.num_layers);
        cfg.cross_attention_dim = get_i32(t, "cross_attention_dim", cfg.cross_attention_dim);
        cfg.norm_eps = get_f64(t, "norm_eps", cfg.norm_eps);
        cfg.positional_embedding_theta = get_f64(
            t,
            "positional_embedding_theta",
            cfg.positional_embedding_theta,
        );
        cfg.positional_embedding_max_pos = get_i32_3(
            t,
            "positional_embedding_max_pos",
            cfg.positional_embedding_max_pos,
        );
        cfg.use_middle_indices_grid =
            get_bool(t, "use_middle_indices_grid", cfg.use_middle_indices_grid);
        if let Some(s) = t.get("rope_type").and_then(Value::as_str) {
            cfg.rope_type = RopeType::from_str(s);
        }
        // `frequencies_precision: "float64"` ⇒ double-precision RoPE (generate_av hardcodes true).
        cfg.double_precision_rope = t
            .get("frequencies_precision")
            .and_then(Value::as_str)
            .map(|s| s == "float64")
            .unwrap_or(true);
        cfg.timestep_scale_multiplier = get_i32(
            t,
            "timestep_scale_multiplier",
            cfg.timestep_scale_multiplier,
        );

        // Caption-projection / gated-attention resolution (generate_av.py lines 1480–1498).
        cfg.caption_projection_first_linear = get_bool(t, "caption_projection_first_linear", true);
        cfg.caption_projection_second_linear =
            get_bool(t, "caption_projection_second_linear", true);
        cfg.apply_gated_attention = get_bool(t, "apply_gated_attention", false);
        cfg.adaln_embedding_coefficient = if cfg.apply_gated_attention { 9 } else { 6 };
        cfg.cross_attention_adaln = get_bool(t, "cross_attention_adaln", cfg.apply_gated_attention);

        // Connector dims (used to derive caption_channels when caption-proj is absent).
        cfg.use_embeddings_connector = get_bool(t, "use_embeddings_connector", false);
        cfg.connector_num_layers = get_i32(t, "connector_num_layers", cfg.connector_num_layers);
        cfg.connector_num_attention_heads = get_i32(
            t,
            "connector_num_attention_heads",
            cfg.connector_num_attention_heads,
        );
        cfg.connector_attention_head_dim = get_i32(
            t,
            "connector_attention_head_dim",
            cfg.connector_attention_head_dim,
        );
        cfg.connector_num_learnable_registers = get_i32(
            t,
            "connector_num_learnable_registers",
            cfg.connector_num_learnable_registers,
        );
        cfg.connector_positional_embedding_max_pos = t
            .get("connector_positional_embedding_max_pos")
            .and_then(|v| v.as_array())
            .and_then(|a| a.first())
            .and_then(Value::as_i64)
            .map(|n| n as i32)
            .unwrap_or(cfg.connector_positional_embedding_max_pos);
        cfg.connector_apply_gated_attention = get_bool(
            t,
            "connector_apply_gated_attention",
            cfg.apply_gated_attention,
        );

        // Audio connector dims (for the audio-caption derivation, mirrored below).
        cfg.audio_num_attention_heads = get_i32(
            t,
            "audio_num_attention_heads",
            cfg.audio_num_attention_heads,
        );
        cfg.audio_attention_head_dim =
            get_i32(t, "audio_attention_head_dim", cfg.audio_attention_head_dim);
        cfg.audio_cross_attention_dim = get_i32(
            t,
            "audio_cross_attention_dim",
            cfg.audio_cross_attention_dim,
        );
        cfg.audio_connector_num_attention_heads = get_i32(
            t,
            "audio_connector_num_attention_heads",
            cfg.audio_connector_num_attention_heads,
        );
        cfg.audio_connector_attention_head_dim = get_i32(
            t,
            "audio_connector_attention_head_dim",
            cfg.audio_connector_attention_head_dim,
        );

        // caption_channels derivation (generate_av.py lines 1484–1498).
        let no_caption_proj =
            !cfg.caption_projection_first_linear && !cfg.caption_projection_second_linear;
        if no_caption_proj {
            cfg.caption_channels =
                cfg.connector_num_attention_heads * cfg.connector_attention_head_dim;
            cfg.audio_caption_channels =
                cfg.audio_connector_num_attention_heads * cfg.audio_connector_attention_head_dim;
        } else {
            cfg.caption_channels = get_i32(t, "caption_channels", cfg.caption_channels);
            cfg.audio_caption_channels =
                get_i32(t, "audio_caption_channels", cfg.audio_caption_channels);
        }

        cfg
    }

    /// Load the config from a model directory's `embedded_config.json` (the `transformer` block).
    /// Falls back to [`video_only_defaults`](Self::video_only_defaults) if the file is absent.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("embedded_config.json");
        if !path.exists() {
            return Ok(Self::video_only_defaults());
        }
        let text = std::fs::read_to_string(&path)?;
        let root_cfg: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse embedded_config.json: {e}")))?;
        let t = root_cfg
            .get("transformer")
            .ok_or_else(|| Error::Msg("ltx: embedded_config.json missing `transformer`".into()))?;
        Ok(Self::from_embedded_transformer(t))
    }
}

/// One entry of the VAE's `decoder_blocks` / `encoder_blocks` list (`embedded_config.json` →
/// `vae.{decoder,encoder}_blocks`). `kind` is the block tag (`res_x`, `compress_space[_res]`,
/// `compress_time[_res]`, `compress_all[_res]`); `num_layers` is the res-block count for `res_x`
/// groups; `multiplier` is the channel expand/reduce factor for the compress blocks. Channel sizes
/// themselves ride on the weights — these fields drive the *structure* (block order, res counts,
/// and the per-compress stride derived from `kind`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaeBlock {
    pub kind: String,
    pub num_layers: i32,
    pub multiplier: i32,
}

impl VaeBlock {
    /// The per-axis `(temporal, height, width)` stride implied by a compress block's `kind`. The
    /// `_res` suffix (encoder `SpaceToDepth`) shares the base name's stride; `res_x` has none.
    pub fn stride(&self) -> (i32, i32, i32) {
        match self.kind.trim_end_matches("_res") {
            "compress_space" => (1, 2, 2),
            "compress_time" => (2, 1, 1),
            "compress_all" => (2, 2, 2),
            _ => (1, 1, 1),
        }
    }

    pub fn is_compress(&self) -> bool {
        self.kind.starts_with("compress_")
    }
}

/// The LTX video VAE config (`embedded_config.json` → `vae`), driving the encoder + decoder
/// structure. `decoder_blocks` is listed in **encoder** order (the decoder reverses it at build
/// time, matching the reference `_build_up_blocks_from_config`).
#[derive(Clone, Debug)]
pub struct LtxVaeConfig {
    pub latent_channels: i32,
    pub patch_size: i32,
    /// `false` for the shipped 2.3 checkpoint → the decoder runs without decode-time noise /
    /// timestep modulation (no scale-shift tables in the weights). Kept config-gated so a future
    /// ts-conditioned checkpoint is supported, not silently dropped.
    pub timestep_conditioning: bool,
    /// `"zeros"` for 2.3 (the 2.0 default was `"reflect"`). Spatial conv padding mode.
    pub spatial_padding_mode: String,
    pub decoder_blocks: Vec<VaeBlock>,
    pub encoder_blocks: Vec<VaeBlock>,
}

impl LtxVaeConfig {
    /// The LTX-2.3 VAE structure (used as the fallback when no `embedded_config.json` is present, and
    /// as the unit-test reference). Channel sizes are not encoded here — they ride on the weights —
    /// only the block order, res counts, and compress kinds.
    pub fn defaults() -> Self {
        let b = |kind: &str, n: i32, m: i32| VaeBlock {
            kind: kind.to_string(),
            num_layers: n,
            multiplier: m,
        };
        LtxVaeConfig {
            latent_channels: 128,
            patch_size: 4,
            timestep_conditioning: false,
            spatial_padding_mode: "zeros".into(),
            decoder_blocks: vec![
                b("res_x", 4, 1),
                b("compress_space", 0, 2),
                b("res_x", 6, 1),
                b("compress_time", 0, 2),
                b("res_x", 4, 1),
                b("compress_all", 0, 1),
                b("res_x", 2, 1),
                b("compress_all", 0, 2),
                b("res_x", 2, 1),
            ],
            encoder_blocks: vec![
                b("res_x", 4, 1),
                b("compress_space_res", 0, 2),
                b("res_x", 6, 1),
                b("compress_time_res", 0, 2),
                b("res_x", 4, 1),
                b("compress_all_res", 0, 2),
                b("res_x", 2, 1),
                b("compress_all_res", 0, 1),
                b("res_x", 2, 1),
            ],
        }
    }

    /// Parse the `vae` block of a parsed `embedded_config.json`.
    pub fn from_embedded_vae(v: &Value) -> Self {
        let mut cfg = Self::defaults();
        cfg.latent_channels = get_i32(v, "latent_channels", cfg.latent_channels);
        cfg.patch_size = get_i32(v, "patch_size", cfg.patch_size);
        cfg.timestep_conditioning = get_bool(v, "timestep_conditioning", cfg.timestep_conditioning);
        if let Some(s) = v.get("spatial_padding_mode").and_then(Value::as_str) {
            cfg.spatial_padding_mode = s.to_string();
        }
        if let Some(blocks) = parse_vae_blocks(v.get("decoder_blocks")) {
            cfg.decoder_blocks = blocks;
        }
        if let Some(blocks) = parse_vae_blocks(v.get("encoder_blocks")) {
            cfg.encoder_blocks = blocks;
        }
        cfg
    }

    /// Load from a model directory's `embedded_config.json` (`vae` block). Falls back to
    /// [`defaults`](Self::defaults) when the file or block is absent.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("embedded_config.json");
        if !path.exists() {
            return Ok(Self::defaults());
        }
        let text = std::fs::read_to_string(&path)?;
        let root_cfg: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse embedded_config.json: {e}")))?;
        match root_cfg.get("vae") {
            Some(v) => Ok(Self::from_embedded_vae(v)),
            None => Ok(Self::defaults()),
        }
    }
}

/// The LTX-2.3 **audio** VAE decoder config (`embedded_config.json` → `audio_vae.model.params.
/// ddconfig`). Drives the decoder structure (2-D causal-conv autoencoder, PIXEL norm, causal-on-
/// **height/time**); channels ride on `audio_vae.safetensors`.
///
/// **`mid_block_add_attention` is `false`** for the shipped 2.3 checkpoint (no `mid.attn_1` weights).
/// The reference `load_audio_decoder` *hardcodes* the `AudioDecoder` constructor and so defaults this
/// to `true`, building a **randomly-initialized** mid attention (the checkpoint ships none) — a
/// reference bug that makes its audio decode non-deterministic across processes (~4% of the signal).
/// We honor the **config** (`false` → skip the mid attention), the model's intended decode. The
/// `AttnBlock` is still implemented + config-gated, so a future `true`-with-weights checkpoint works.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AudioVaeConfig {
    pub ch: i32,
    pub out_ch: i32,
    pub ch_mult: Vec<i32>,
    pub num_res_blocks: i32,
    pub z_channels: i32,
    pub mel_bins: i32,
    pub mid_block_add_attention: bool,
}

impl AudioVaeConfig {
    /// The shipped 2.3 audio-VAE structure (fallback when no `embedded_config.json` is present).
    pub fn defaults() -> Self {
        AudioVaeConfig {
            ch: 128,
            out_ch: 2,
            ch_mult: vec![1, 2, 4],
            num_res_blocks: 2,
            z_channels: 8,
            mel_bins: 64,
            mid_block_add_attention: false,
        }
    }

    /// Number of resolution levels (`len(ch_mult)`); the decoder upsamples on levels `1..num`.
    pub fn num_resolutions(&self) -> usize {
        self.ch_mult.len()
    }

    /// Parse the `ddconfig` block (`audio_vae.model.params.ddconfig`).
    pub fn from_ddconfig(v: &Value) -> Self {
        let mut cfg = Self::defaults();
        cfg.ch = get_i32(v, "ch", cfg.ch);
        cfg.out_ch = get_i32(v, "out_ch", cfg.out_ch);
        if let Some(arr) = v.get("ch_mult").and_then(Value::as_array) {
            let parsed: Vec<i32> = arr
                .iter()
                .filter_map(|n| n.as_i64().map(|x| x as i32))
                .collect();
            if !parsed.is_empty() {
                cfg.ch_mult = parsed;
            }
        }
        cfg.num_res_blocks = get_i32(v, "num_res_blocks", cfg.num_res_blocks);
        cfg.z_channels = get_i32(v, "z_channels", cfg.z_channels);
        cfg.mel_bins = get_i32(v, "mel_bins", cfg.mel_bins);
        cfg.mid_block_add_attention =
            get_bool(v, "mid_block_add_attention", cfg.mid_block_add_attention);
        cfg
    }

    /// Load from a model directory's `embedded_config.json` (`audio_vae.model.params.ddconfig`).
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("embedded_config.json");
        if !path.exists() {
            return Ok(Self::defaults());
        }
        let text = std::fs::read_to_string(&path)?;
        let root_cfg: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse embedded_config.json: {e}")))?;
        let dd = root_cfg
            .get("audio_vae")
            .and_then(|v| v.get("model"))
            .and_then(|v| v.get("params"))
            .and_then(|v| v.get("ddconfig"));
        Ok(match dd {
            Some(v) => Self::from_ddconfig(v),
            None => Self::defaults(),
        })
    }
}

/// One vocoder generator's config (`embedded_config.json` → `vocoder.{vocoder,bwe}`). Drives the
/// HiFi-GAN / BigVGAN structure: the ConvTranspose1d upsample strides + the dilated ResBlock/AMPBlock
/// kernel sizes/dilations (channel counts ride on `vocoder.safetensors`). `is_bigvgan()` selects
/// SnakeBeta+AMPBlock1 vs leaky-ReLU+ResBlock (`load_vocoder`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocoderGenConfig {
    pub upsample_rates: Vec<i32>,
    pub upsample_kernel_sizes: Vec<i32>,
    pub resblock_kernel_sizes: Vec<i32>,
    pub resblock_dilation_sizes: Vec<Vec<i32>>,
    pub upsample_initial_channel: i32,
    pub resblock: String,
    pub activation: String,
    pub use_tanh_at_final: bool,
    pub use_bias_at_final: bool,
    pub apply_final_activation: bool,
    pub stereo: bool,
}

impl VocoderGenConfig {
    /// The HiFi-GAN-shaped `load_vocoder` defaults (overridden by the embedded config).
    pub fn defaults() -> Self {
        VocoderGenConfig {
            upsample_rates: vec![6, 5, 2, 2, 2],
            upsample_kernel_sizes: vec![16, 15, 8, 4, 4],
            resblock_kernel_sizes: vec![3, 7, 11],
            resblock_dilation_sizes: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            upsample_initial_channel: 1024,
            resblock: "1".into(),
            activation: "leaky_relu".into(),
            use_tanh_at_final: true,
            use_bias_at_final: true,
            apply_final_activation: true,
            stereo: true,
        }
    }

    /// SnakeBeta + AMPBlock1 (BigVGAN) vs leaky-ReLU + ResBlock (HiFi-GAN).
    pub fn is_bigvgan(&self) -> bool {
        self.activation.to_lowercase() == "snakebeta" || self.resblock.to_uppercase() == "AMP1"
    }

    fn read(v: &Value, base: &VocoderGenConfig) -> Self {
        let mut cfg = base.clone();
        cfg.upsample_initial_channel =
            get_i32(v, "upsample_initial_channel", cfg.upsample_initial_channel);
        if let Some(a) = get_i32_vec(v, "upsample_rates") {
            cfg.upsample_rates = a;
        }
        if let Some(a) = get_i32_vec(v, "upsample_kernel_sizes") {
            cfg.upsample_kernel_sizes = a;
        }
        if let Some(a) = get_i32_vec(v, "resblock_kernel_sizes") {
            cfg.resblock_kernel_sizes = a;
        }
        if let Some(a) = v.get("resblock_dilation_sizes").and_then(Value::as_array) {
            let parsed: Vec<Vec<i32>> = a
                .iter()
                .filter_map(|row| {
                    row.as_array().map(|r| {
                        r.iter()
                            .filter_map(|n| n.as_i64().map(|x| x as i32))
                            .collect()
                    })
                })
                .collect();
            if !parsed.is_empty() {
                cfg.resblock_dilation_sizes = parsed;
            }
        }
        if let Some(s) = v.get("resblock").and_then(Value::as_str) {
            cfg.resblock = s.to_string();
        }
        if let Some(s) = v.get("activation").and_then(Value::as_str) {
            cfg.activation = s.to_lowercase();
        }
        cfg.use_tanh_at_final = get_bool(v, "use_tanh_at_final", cfg.use_tanh_at_final);
        cfg.use_bias_at_final = get_bool(v, "use_bias_at_final", cfg.use_bias_at_final);
        cfg.apply_final_activation =
            get_bool(v, "apply_final_activation", cfg.apply_final_activation);
        cfg.stereo = get_bool(v, "stereo", cfg.stereo);
        cfg
    }
}

/// The full vocoder config (`embedded_config.json` → `vocoder`): the core generator + an optional
/// bandwidth-extension (BWE) stage (`VocoderWithBWE`, the shipped 2.3 path → 48 kHz).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VocoderConfig {
    pub core: VocoderGenConfig,
    pub bwe: Option<VocoderGenConfig>,
    /// Core generator output sample rate (`AUDIO_SAMPLE_RATE`, 24 kHz).
    pub output_sample_rate: i32,
    // BWE STFT params + sample rates (`VocoderWithBWE`); meaningful only when `bwe` is `Some`.
    pub bwe_input_sample_rate: i32,
    pub bwe_output_sample_rate: i32,
    pub bwe_hop_length: i32,
    pub bwe_win_length: i32,
}

impl VocoderConfig {
    pub fn defaults() -> Self {
        VocoderConfig {
            core: VocoderGenConfig::defaults(),
            bwe: None,
            output_sample_rate: 24000,
            bwe_input_sample_rate: 24000,
            bwe_output_sample_rate: 24000,
            bwe_hop_length: 80,
            bwe_win_length: 512,
        }
    }

    /// The audio-track sample rate: the BWE output when present, else the core output.
    pub fn final_sample_rate(&self) -> i32 {
        if self.bwe.is_some() {
            self.bwe_output_sample_rate
        } else {
            self.output_sample_rate
        }
    }

    /// Parse the `vocoder` block.
    pub fn from_embedded_vocoder(v: &Value) -> Self {
        let mut cfg = Self::defaults();
        if let Some(core) = v.get("vocoder") {
            cfg.core = VocoderGenConfig::read(core, &VocoderGenConfig::defaults());
        }
        if let Some(bwe) = v.get("bwe").filter(|b| b.is_object()) {
            // The BWE generator's `apply_final_activation` defaults to false in `load_vocoder`.
            let mut bwe_base = VocoderGenConfig::defaults();
            bwe_base.use_tanh_at_final = false;
            bwe_base.use_bias_at_final = false;
            bwe_base.apply_final_activation = false;
            cfg.bwe = Some(VocoderGenConfig::read(bwe, &bwe_base));
            cfg.bwe_input_sample_rate = get_i32(bwe, "input_sampling_rate", cfg.output_sample_rate);
            cfg.bwe_output_sample_rate =
                get_i32(bwe, "output_sampling_rate", cfg.bwe_output_sample_rate);
            cfg.bwe_hop_length = get_i32(bwe, "hop_length", cfg.bwe_hop_length);
            cfg.bwe_win_length = get_i32(bwe, "win_size", cfg.bwe_win_length);
        }
        cfg
    }

    /// Load from a model directory's `embedded_config.json` (`vocoder` block).
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("embedded_config.json");
        if !path.exists() {
            return Ok(Self::defaults());
        }
        let text = std::fs::read_to_string(&path)?;
        let root_cfg: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse embedded_config.json: {e}")))?;
        Ok(match root_cfg.get("vocoder") {
            Some(v) => Self::from_embedded_vocoder(v),
            None => Self::defaults(),
        })
    }
}

/// The `split_model.json` manifest's quantization fields. The reference `generate_av.py` drives
/// **selective transformer quant** from these (and `convert.py` writes them): `quantized` gates it,
/// `quantization_bits` (reference default **4**) and `quantization_group_size` (default **64**) set
/// the packing geometry. The shipped checkpoints are bits 4 (`base_q4`) or bits 8 (`base_q8`) — so
/// the geometry is **read from the manifest, never hardcoded** (sc-2686). The per-Linear
/// `_should_quantize` predicate (a layer is quantized iff its weights carry `.scales`) is applied at
/// load in [`crate::transformer`], matching `generate_av.py`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitModel {
    /// `quantized` — whether the transformer carries selectively-quantized Linears.
    pub quantized: bool,
    /// `quantization_bits` (default 4): 4 → Q4, 8 → Q8.
    pub bits: i32,
    /// `quantization_group_size` (default 64): the affine-quant group width.
    pub group: i32,
}

impl SplitModel {
    /// A non-quantized manifest (the file is absent or `quantized:false`). Bits/group hold the
    /// reference defaults so a downstream quant geometry is always well-defined.
    pub fn dense() -> Self {
        SplitModel {
            quantized: false,
            bits: 4,
            group: 64,
        }
    }

    /// Parse a model directory's `split_model.json`. An absent file → [`dense`](Self::dense) (no
    /// quant), mirroring `generate_av.py` (which only quantizes when the manifest exists and sets
    /// `quantized:true`).
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("split_model.json");
        if !path.exists() {
            return Ok(Self::dense());
        }
        let text = std::fs::read_to_string(&path)?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("ltx: parse split_model.json: {e}")))?;
        Ok(Self::from_value(&v))
    }

    /// Parse the manifest fields from a parsed `split_model.json` value.
    pub fn from_value(v: &Value) -> Self {
        SplitModel {
            quantized: get_bool(v, "quantized", false),
            bits: get_i32(v, "quantization_bits", 4),
            group: get_i32(v, "quantization_group_size", 64),
        }
    }
}

/// Parse a `[["res_x", {"num_layers": 4}], ["compress_space_res", {"multiplier": 2}], …]` list.
fn parse_vae_blocks(v: Option<&Value>) -> Option<Vec<VaeBlock>> {
    let arr = v?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for entry in arr {
        let pair = entry.as_array()?;
        if pair.len() != 2 {
            return None;
        }
        let kind = pair[0].as_str()?.to_string();
        let params = &pair[1];
        out.push(VaeBlock {
            num_layers: get_i32(params, "num_layers", 0),
            multiplier: get_i32(params, "multiplier", 1),
            kind,
        });
    }
    Some(out)
}

fn get_i32(v: &Value, key: &str, default: i32) -> i32 {
    v.get(key)
        .and_then(Value::as_i64)
        .map(|n| n as i32)
        .unwrap_or(default)
}

fn get_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn get_bool(v: &Value, key: &str, default: bool) -> bool {
    v.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Read a JSON int array `key` → `Vec<i32>` (None if absent/empty).
fn get_i32_vec(v: &Value, key: &str) -> Option<Vec<i32>> {
    let arr = v.get(key)?.as_array()?;
    let out: Vec<i32> = arr
        .iter()
        .filter_map(|n| n.as_i64().map(|x| x as i32))
        .collect();
    (!out.is_empty()).then_some(out)
}

fn get_i32_3(v: &Value, key: &str, default: [i32; 3]) -> [i32; 3] {
    match v.get(key).and_then(Value::as_array) {
        Some(a) if a.len() == 3 => {
            let mut out = default;
            for (i, slot) in out.iter_mut().enumerate() {
                if let Some(n) = a[i].as_i64() {
                    *slot = n as i32;
                }
            }
            out
        }
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An LTX-2.3 `transformer` block of `embedded_config.json`.
    fn ltx23_transformer() -> Value {
        serde_json::json!({
            "_class_name": "AVTransformer3DModel",
            "attention_head_dim": 128,
            "caption_channels": 3840,
            "cross_attention_dim": 4096,
            "in_channels": 128,
            "norm_eps": 1e-06,
            "num_attention_heads": 32,
            "num_layers": 48,
            "out_channels": 128,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_cross_attention_dim": 2048,
            "use_embeddings_connector": true,
            "connector_attention_head_dim": 128,
            "connector_num_attention_heads": 32,
            "connector_num_layers": 8,
            "connector_positional_embedding_max_pos": [4096],
            "connector_num_learnable_registers": 128,
            "use_middle_indices_grid": true,
            "apply_gated_attention": true,
            "connector_apply_gated_attention": true,
            "caption_projection_first_linear": false,
            "caption_projection_second_linear": false,
            "audio_connector_attention_head_dim": 64,
            "audio_connector_num_attention_heads": 32,
            "cross_attention_adaln": true,
            "text_encoder_norm_type": "per_token_rms",
            "rope_type": "split",
            "frequencies_precision": "float64",
            "positional_embedding_theta": 10000.0,
            "positional_embedding_max_pos": [20, 2048, 2048],
            "timestep_scale_multiplier": 1000,
            "av_ca_timestep_scale_multiplier": 1000.0
        })
    }

    #[test]
    fn ltx23_config_matches_reference_build_logic() {
        let cfg = LtxConfig::from_embedded_transformer(&ltx23_transformer());
        // Gated family → adaLN coeff 9.
        assert!(cfg.apply_gated_attention);
        assert_eq!(cfg.adaln_embedding_coefficient, 9);
        assert!(cfg.cross_attention_adaln);
        // No caption projection → caption_channels = connector_heads × connector_head_dim = 4096.
        assert!(!cfg.caption_projection_first_linear);
        assert!(!cfg.caption_projection_second_linear);
        assert_eq!(cfg.caption_channels, 4096);
        assert_eq!(cfg.audio_caption_channels, 32 * 64);
        // Audio stack dims (sc-2684).
        assert_eq!(cfg.audio_inner_dim(), 2048);
        assert_eq!(cfg.audio_connector_num_attention_heads, 32);
        assert_eq!(cfg.audio_connector_attention_head_dim, 64);
        assert_eq!(cfg.audio_in_channels, 128);
        assert_eq!(cfg.audio_out_channels, 128);
        assert_eq!(cfg.audio_positional_embedding_max_pos, 20);
        assert_eq!(cfg.cross_pe_max_pos(), 20);
        assert_eq!(cfg.av_ca_timestep_scale_multiplier, 1000);
        // Core dims.
        assert_eq!(cfg.inner_dim(), 4096);
        assert_eq!(cfg.num_layers, 48);
        assert_eq!(cfg.cross_attention_dim, 4096);
        assert_eq!(cfg.rope_type, RopeType::Split);
        assert!(cfg.double_precision_rope);
        assert_eq!(cfg.positional_embedding_max_pos, [20, 2048, 2048]);
        assert!(cfg.use_middle_indices_grid);
        assert_eq!(cfg.timestep_scale_multiplier, 1000);
        // Connector.
        assert!(cfg.use_embeddings_connector);
        assert_eq!(cfg.connector_num_layers, 8);
        assert_eq!(cfg.connector_num_attention_heads, 32);
        assert_eq!(cfg.connector_attention_head_dim, 128);
        assert_eq!(cfg.connector_num_learnable_registers, 128);
        assert_eq!(cfg.connector_positional_embedding_max_pos, 4096);
        assert!(cfg.connector_apply_gated_attention);
    }

    #[test]
    fn defaults_are_the_2_0_values() {
        let cfg = LtxConfig::video_only_defaults();
        assert_eq!(cfg.adaln_embedding_coefficient, 6);
        assert!(!cfg.apply_gated_attention);
        assert_eq!(cfg.caption_channels, 3840);
        assert!(cfg.caption_projection_first_linear);
    }

    #[test]
    fn split_model_reads_quant_geometry() {
        // The actual `ltx_2_3_base_q4` manifest → Q4, group 64.
        let q4 = serde_json::json!({
            "format": "split", "model_version": "2.3.0", "variant": "distilled",
            "quantized": true, "quantization_bits": 4, "quantization_group_size": 64
        });
        let m = SplitModel::from_value(&q4);
        assert!(m.quantized);
        assert_eq!(m.bits, 4);
        assert_eq!(m.group, 64);
        // `base_q8` → Q8.
        let q8 = serde_json::json!({"quantized": true, "quantization_bits": 8, "quantization_group_size": 64});
        assert_eq!(SplitModel::from_value(&q8).bits, 8);
        // Missing keys → reference defaults (bits 4, group 64); `quantized:false` → dense.
        let dense = serde_json::json!({"quantized": false});
        let m = SplitModel::from_value(&dense);
        assert!(!m.quantized);
        assert_eq!((m.bits, m.group), (4, 64));
        assert_eq!(
            SplitModel::dense(),
            SplitModel::from_value(&serde_json::json!({}))
        );
    }
}
