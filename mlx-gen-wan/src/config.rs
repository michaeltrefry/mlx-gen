//! Wan2.2 model configuration — port of `models/wan/config.py`'s `WanModelConfig`.
//!
//! The DiT transformer is **dimension-parametric**: the same block structure (self-attn +
//! qk-RMSNorm + cross-attn + adaLN-6vec + plain GELU-tanh FFN) serves every Wan variant, and only
//! the dimensions/inference knobs change between them. This module carries the full set of presets
//! the reference exposes, but the crate's first-class target is **`wan22_ti2v_5b`** (the dense 5B
//! with its own z48 VAE — sc-2680). The 14B presets (`wan22_t2v_14b` dual-expert MoE, `wan22_i2v_14b`
//! channel-concat) and the Wan2.1 variants are carried for completeness and wired by the core's
//! remaining slices (sc-2678 S2/S5).
//!
//! `from_config_json` reads the field layout the reference's `convert_wan.py` serializes via
//! `WanModelConfig.to_dict()` (the same field names the on-disk `config.json` uses).

use std::path::Path;

use serde_json::Value;

use mlx_gen::{Error, Result};

/// The Wan2.2 anti-artifact negative prompt (the reference's `sample_neg_prompt` default; Chinese).
/// Used when a T2V/TI2V request omits its own negative prompt and CFG is active.
pub const SAMPLE_NEG_PROMPT: &str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走";

/// CFG guidance scale. Dense models (5B, Wan2.1) use a single scalar; the Wan2.2 dual-expert MoE
/// models select `low` below the timestep boundary and `high` at/above it (`generate_wan.py`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GuideScale {
    Single(f32),
    Dual { low: f32, high: f32 },
}

impl GuideScale {
    /// The effective scale for a single-model run (or the value to report). For dual models a
    /// caller picks `low`/`high` per step; `effective` returns `low` as a representative default.
    pub fn effective(self) -> f32 {
        match self {
            GuideScale::Single(s) => s,
            GuideScale::Dual { low, .. } => low,
        }
    }

    /// Whether CFG is disabled (all relevant scales ≤ 1.0 → the B=1 fast path).
    pub fn cfg_disabled(self) -> bool {
        match self {
            GuideScale::Single(s) => s <= 1.0,
            GuideScale::Dual { low, high } => low <= 1.0 && high <= 1.0,
        }
    }
}

/// A pre-quantized snapshot's quantization manifest (`config.json`'s `quantization` block, written by
/// `convert_wan.py`). When present, the transformer's `_quantize_predicate` Linears ship **packed**
/// on disk (`.scales`/`.biases`/u32 `.weight`) at this bit-width + group, and the loader builds them
/// quantized directly (no load-time re-quantize) — the `loading.py` consume path. `None` ⇒ a dense
/// bf16 snapshot (the loader may still quantize at load via `spec.quantize`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WanQuant {
    pub bits: i32,
    pub group_size: i32,
}

/// Configuration for a Wan T2V / I2V / TI2V model (2.1 and 2.2). Mirrors `WanModelConfig`.
#[derive(Clone, Debug, PartialEq)]
pub struct WanModelConfig {
    pub model_type: String,
    pub model_version: String,
    pub patch_size: (usize, usize, usize),
    pub text_len: usize,
    pub in_dim: usize,
    pub dim: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub text_dim: usize,
    pub out_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub window_size: (i64, i64),
    pub qk_norm: bool,
    pub cross_attn_norm: bool,
    pub eps: f64,

    // VAE
    pub vae_stride: (usize, usize, usize),
    pub vae_z_dim: usize,

    // Inference
    pub dual_model: bool,
    pub boundary: f32,
    pub sample_shift: f32,
    pub sample_steps: usize,
    pub sample_guide_scale: GuideScale,
    pub num_train_timesteps: usize,
    pub sample_fps: u32,
    pub frame_num: usize,
    pub sample_neg_prompt: String,

    // Resolution constraints (0 = no limit; e.g. 704*1280 for TI2V-5B).
    pub max_area: usize,

    // UMT5-XXL text encoder.
    pub t5_vocab_size: usize,
    pub t5_dim: usize,
    pub t5_dim_attn: usize,
    pub t5_dim_ffn: usize,
    pub t5_num_heads: usize,
    pub t5_num_layers: usize,
    pub t5_num_buckets: usize,

    /// Pre-quantized snapshot manifest (`config.json`'s `quantization` block), or `None` for a dense
    /// bf16 snapshot. Drives the [`crate::transformer`] consume path (sc-2682).
    pub quantization: Option<WanQuant>,
}

impl WanModelConfig {
    /// Per-head dimension `dim / num_heads` (128 for both the 5B and the 14B).
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    /// Whether this is the image-conditioned 5B mask-blend variant (`model_type == "ti2v"`).
    pub fn is_ti2v(&self) -> bool {
        self.model_type == "ti2v"
    }

    /// Whether this is the channel-concat I2V variant (`model_type == "i2v"`, in_dim 36).
    pub fn is_i2v_concat(&self) -> bool {
        self.model_type == "i2v"
    }

    /// Whether this VAE is the Wan2.2 z48 VAE (vae22) vs the 2.1 WanVAE (z16).
    pub fn is_wan22_vae(&self) -> bool {
        self.vae_z_dim == 48
    }

    /// The base config = `wan22_t2v_14b` (the reference's dataclass defaults).
    fn base() -> Self {
        Self {
            model_type: "t2v".into(),
            model_version: "2.2".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 16,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 40,
            num_layers: 40,
            window_size: (-1, -1),
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
            vae_stride: (4, 8, 8),
            vae_z_dim: 16,
            dual_model: true,
            boundary: 0.875,
            sample_shift: 12.0,
            sample_steps: 40,
            sample_guide_scale: GuideScale::Dual {
                low: 3.0,
                high: 4.0,
            },
            num_train_timesteps: 1000,
            sample_fps: 16,
            frame_num: 81,
            sample_neg_prompt: SAMPLE_NEG_PROMPT.into(),
            max_area: 0,
            t5_vocab_size: 256384,
            t5_dim: 4096,
            t5_dim_attn: 4096,
            t5_dim_ffn: 10240,
            t5_num_heads: 64,
            t5_num_layers: 24,
            t5_num_buckets: 32,
            quantization: None,
        }
    }

    /// Wan2.2 T2V 14B: dual-expert MoE, 40 layers, dim 5120 (the reference default).
    pub fn wan22_t2v_14b() -> Self {
        Self::base()
    }

    /// Wan2.1 T2V 14B: single (dense) model, 40 layers, dim 5120.
    pub fn wan21_t2v_14b() -> Self {
        Self {
            model_version: "2.1".into(),
            dual_model: false,
            boundary: 0.0,
            sample_shift: 5.0,
            sample_steps: 50,
            sample_guide_scale: GuideScale::Single(5.0),
            ..Self::base()
        }
    }

    /// Wan2.1 T2V 1.3B: single (dense) model, 30 layers, dim 1536.
    pub fn wan21_t2v_1_3b() -> Self {
        Self {
            model_version: "2.1".into(),
            dim: 1536,
            ffn_dim: 8960,
            num_heads: 12,
            num_layers: 30,
            dual_model: false,
            boundary: 0.0,
            sample_shift: 5.0,
            sample_steps: 50,
            sample_guide_scale: GuideScale::Single(5.0),
            ..Self::base()
        }
    }

    /// Wan2.2 I2V 14B: dual-expert MoE, channel-concat conditioning (in_dim 36), 40 layers.
    pub fn wan22_i2v_14b() -> Self {
        Self {
            model_type: "i2v".into(),
            in_dim: 36,
            out_dim: 16,
            dual_model: true,
            boundary: 0.900,
            sample_shift: 5.0,
            sample_guide_scale: GuideScale::Dual {
                low: 3.5,
                high: 3.5,
            },
            max_area: 704 * 1280,
            ..Self::base()
        }
    }

    /// Wan2.2 TI2V 5B: text+image to video, **dense**, 30 layers, dim 3072, z48 VAE. The crate's
    /// first-class target (sc-2680).
    pub fn wan22_ti2v_5b() -> Self {
        Self {
            model_type: "ti2v".into(),
            dim: 3072,
            ffn_dim: 14336,
            in_dim: 48,
            out_dim: 48,
            num_heads: 24,
            num_layers: 30,
            vae_z_dim: 48,
            vae_stride: (4, 16, 16),
            dual_model: false,
            boundary: 0.0,
            sample_shift: 5.0,
            sample_steps: 40,
            sample_guide_scale: GuideScale::Single(5.0),
            sample_fps: 24,
            max_area: 704 * 1280,
            ..Self::base()
        }
    }

    /// Resolve the preset that best matches a parsed `config.json`, then overlay any explicit
    /// fields present in the JSON. Mirrors `convert_wan_checkpoint`'s preset selection: a
    /// `model_type == "ti2v"` with `dim == 3072` is the 5B; otherwise dual/dense + version pick.
    pub fn from_config_json(v: &Value) -> Self {
        let model_type = v.get("model_type").and_then(Value::as_str).unwrap_or("t2v");
        let dim = v.get("dim").and_then(Value::as_u64).unwrap_or(5120) as usize;
        let version = v
            .get("model_version")
            .and_then(Value::as_str)
            .unwrap_or("2.2");
        let dual = v.get("dual_model").and_then(Value::as_bool);

        // Preset selection (the reference's auto-detection).
        let mut cfg = if model_type == "ti2v" && dim == 3072 {
            Self::wan22_ti2v_5b()
        } else if model_type == "i2v" {
            Self::wan22_i2v_14b()
        } else if version == "2.1" || dual == Some(false) {
            if dim == 1536 {
                Self::wan21_t2v_1_3b()
            } else {
                Self::wan21_t2v_14b()
            }
        } else {
            Self::wan22_t2v_14b()
        };

        cfg.overlay_json(v);
        cfg
    }

    /// Load from a model directory's `config.json` (the layout `convert_wan.py` writes). Falls back
    /// to the 5B preset when the file is absent (the crate's default target).
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let path = root.join("config.json");
        if !path.exists() {
            return Ok(Self::wan22_ti2v_5b());
        }
        let text = std::fs::read_to_string(&path)?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| Error::Msg(format!("wan: parse config.json: {e}")))?;
        Ok(Self::from_config_json(&v))
    }

    /// Overlay every field explicitly present in the JSON on top of the selected preset.
    fn overlay_json(&mut self, v: &Value) {
        if let Some(s) = v.get("model_type").and_then(Value::as_str) {
            self.model_type = s.to_string();
        }
        if let Some(s) = v.get("model_version").and_then(Value::as_str) {
            self.model_version = s.to_string();
        }
        set_usize3(v, "patch_size", &mut self.patch_size);
        set_usize(v, "text_len", &mut self.text_len);
        set_usize(v, "in_dim", &mut self.in_dim);
        set_usize(v, "dim", &mut self.dim);
        set_usize(v, "ffn_dim", &mut self.ffn_dim);
        set_usize(v, "freq_dim", &mut self.freq_dim);
        set_usize(v, "text_dim", &mut self.text_dim);
        set_usize(v, "out_dim", &mut self.out_dim);
        set_usize(v, "num_heads", &mut self.num_heads);
        set_usize(v, "num_layers", &mut self.num_layers);
        if let Some(arr) = v.get("window_size").and_then(Value::as_array) {
            if arr.len() == 2 {
                if let (Some(a), Some(b)) = (arr[0].as_i64(), arr[1].as_i64()) {
                    self.window_size = (a, b);
                }
            }
        }
        set_bool(v, "qk_norm", &mut self.qk_norm);
        set_bool(v, "cross_attn_norm", &mut self.cross_attn_norm);
        set_f64(v, "eps", &mut self.eps);
        set_usize3(v, "vae_stride", &mut self.vae_stride);
        set_usize(v, "vae_z_dim", &mut self.vae_z_dim);
        set_bool(v, "dual_model", &mut self.dual_model);
        set_f32(v, "boundary", &mut self.boundary);
        set_f32(v, "sample_shift", &mut self.sample_shift);
        set_usize(v, "sample_steps", &mut self.sample_steps);
        if let Some(gs) = parse_guide_scale(v.get("sample_guide_scale")) {
            self.sample_guide_scale = gs;
        }
        set_usize(v, "num_train_timesteps", &mut self.num_train_timesteps);
        if let Some(n) = v.get("sample_fps").and_then(Value::as_u64) {
            self.sample_fps = n as u32;
        }
        set_usize(v, "frame_num", &mut self.frame_num);
        if let Some(s) = v.get("sample_neg_prompt").and_then(Value::as_str) {
            self.sample_neg_prompt = s.to_string();
        }
        set_usize(v, "max_area", &mut self.max_area);
        set_usize(v, "t5_vocab_size", &mut self.t5_vocab_size);
        set_usize(v, "t5_dim", &mut self.t5_dim);
        set_usize(v, "t5_dim_attn", &mut self.t5_dim_attn);
        set_usize(v, "t5_dim_ffn", &mut self.t5_dim_ffn);
        set_usize(v, "t5_num_heads", &mut self.t5_num_heads);
        set_usize(v, "t5_num_layers", &mut self.t5_num_layers);
        set_usize(v, "t5_num_buckets", &mut self.t5_num_buckets);
        // Pre-quantized snapshot manifest: `"quantization": {"bits": 4, "group_size": 64}` (written
        // by convert_wan.py). Defaults match mflux/the reference (group 64) if a key is omitted.
        if let Some(q) = v.get("quantization").filter(|q| q.is_object()) {
            self.quantization = Some(WanQuant {
                bits: q.get("bits").and_then(Value::as_i64).unwrap_or(4) as i32,
                group_size: q.get("group_size").and_then(Value::as_i64).unwrap_or(64) as i32,
            });
        }
    }
}

/// Configuration for a **Wan-VACE** model (sc-3388 / epic 3040) — the base Wan DiT plus the two
/// VACE-only fields. VACE (Video All-in-one Creation and Editing) is purely additive on the base
/// `WanModelConfig`: the same dimension-parametric DiT, plus `vace_layers` (which main layers receive
/// a control hint, must include 0) and `vace_in_channels` (the control-latent channel count, 96 =
/// 32 video latent + 64 mask unfold). Mirrors diffusers `WanVACETransformer3DModel`'s two extra
/// config fields over `WanTransformer3DModel`.
///
/// The VACE checkpoint ships in **diffusers layout** (the SceneWorks worker loads it via
/// `WanVACEPipeline.from_pretrained`), so [`crate::vace`] reads diffusers tensor names directly — no
/// native conversion. The base dims here are still the native [`WanModelConfig`] (reused for the VAE,
/// scheduler, and resolution math); only the transformer-shaping fields + the two VACE fields drive
/// the DiT.
#[derive(Clone, Debug, PartialEq)]
pub struct WanVaceConfig {
    /// The base Wan DiT / VAE / inference config (dims, VAE, scheduler knobs).
    pub base: WanModelConfig,
    /// Which main-block indices receive a VACE control hint (diffusers default
    /// `[0, 5, 10, 15, 20, 25, 30, 35]`; must include 0 so block 0 carries `proj_in`).
    pub vace_layers: Vec<usize>,
    /// The control-latent channel count (diffusers default 96 = 32 video + 64 mask-unfold).
    pub vace_in_channels: usize,
}

impl WanVaceConfig {
    /// Build from a parsed config (native `WanModelConfig` field names + `vace_layers` /
    /// `vace_in_channels`). The base config is resolved by [`WanModelConfig::from_config_json`]; the
    /// two VACE fields default to the diffusers 14B defaults when absent.
    pub fn from_config_json(v: &Value) -> Self {
        let base = WanModelConfig::from_config_json(v);
        let vace_layers = v
            .get("vace_layers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![0, 5, 10, 15, 20, 25, 30, 35]);
        let vace_in_channels = v
            .get("vace_in_channels")
            .and_then(Value::as_u64)
            .unwrap_or(96) as usize;
        Self {
            base,
            vace_layers,
            vace_in_channels,
        }
    }

    /// Build from a **diffusers** `transformer/config.json` (the layout the real VACE checkpoint
    /// ships). Maps the diffusers field names (`num_attention_heads`, `attention_head_dim`,
    /// `in_channels`, `out_channels`, `text_dim`, `freq_dim`, `ffn_dim`, `num_layers`,
    /// `cross_attn_norm`, `eps`, `patch_size`, `vace_layers`, `vace_in_channels`) onto the native
    /// base config (started from the Wan2.1 dense preset — VACE is Wan2.1-based: z16 VAE, stride
    /// 4×8×8). VAE/scheduler knobs not present in the transformer config keep the Wan2.1 defaults.
    pub fn from_diffusers_json(v: &Value) -> Self {
        let heads = v
            .get("num_attention_heads")
            .and_then(Value::as_u64)
            .unwrap_or(40) as usize;
        let head_dim = v
            .get("attention_head_dim")
            .and_then(Value::as_u64)
            .unwrap_or(128) as usize;
        let mut base = if heads * head_dim <= 1536 {
            WanModelConfig::wan21_t2v_1_3b()
        } else {
            WanModelConfig::wan21_t2v_14b()
        };
        base.dim = heads * head_dim;
        base.num_heads = heads;
        set_usize(v, "in_channels", &mut base.in_dim);
        set_usize(v, "out_channels", &mut base.out_dim);
        set_usize(v, "text_dim", &mut base.text_dim);
        set_usize(v, "freq_dim", &mut base.freq_dim);
        set_usize(v, "ffn_dim", &mut base.ffn_dim);
        set_usize(v, "num_layers", &mut base.num_layers);
        set_bool(v, "cross_attn_norm", &mut base.cross_attn_norm);
        set_f64(v, "eps", &mut base.eps);
        set_usize3(v, "patch_size", &mut base.patch_size);
        let vace_layers = v
            .get("vace_layers")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![0, 5, 10, 15, 20, 25, 30, 35]);
        let vace_in_channels = v
            .get("vace_in_channels")
            .and_then(Value::as_u64)
            .unwrap_or(96) as usize;
        Self {
            base,
            vace_layers,
            vace_in_channels,
        }
    }

    /// Load a VACE config from a model directory: a diffusers `transformer/config.json` if present
    /// (the real checkpoint layout), else a native `config.json`, else the diffusers 14B defaults.
    pub fn from_model_dir(root: &Path) -> Result<Self> {
        let diffusers = root.join("transformer").join("config.json");
        if diffusers.exists() {
            let text = std::fs::read_to_string(&diffusers)?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| Error::Msg(format!("wan-vace: parse transformer/config.json: {e}")))?;
            return Ok(Self::from_diffusers_json(&v));
        }
        let native = root.join("config.json");
        if native.exists() {
            let text = std::fs::read_to_string(&native)?;
            let v: Value = serde_json::from_str(&text)
                .map_err(|e| Error::Msg(format!("wan-vace: parse config.json: {e}")))?;
            return Ok(Self::from_config_json(&v));
        }
        Ok(Self {
            base: WanModelConfig::wan21_t2v_14b(),
            vace_layers: vec![0, 5, 10, 15, 20, 25, 30, 35],
            vace_in_channels: 96,
        })
    }

    /// Per-head dimension (`dim / num_heads`).
    pub fn head_dim(&self) -> usize {
        self.base.head_dim()
    }
}

fn parse_guide_scale(v: Option<&Value>) -> Option<GuideScale> {
    match v {
        Some(Value::Number(n)) => n.as_f64().map(|x| GuideScale::Single(x as f32)),
        Some(Value::Array(a)) if a.len() == 2 => {
            let low = a[0].as_f64()? as f32;
            let high = a[1].as_f64()? as f32;
            Some(GuideScale::Dual { low, high })
        }
        _ => None,
    }
}

fn set_usize(v: &Value, key: &str, slot: &mut usize) {
    if let Some(n) = v.get(key).and_then(Value::as_u64) {
        *slot = n as usize;
    }
}

fn set_f64(v: &Value, key: &str, slot: &mut f64) {
    if let Some(n) = v.get(key).and_then(Value::as_f64) {
        *slot = n;
    }
}

fn set_f32(v: &Value, key: &str, slot: &mut f32) {
    if let Some(n) = v.get(key).and_then(Value::as_f64) {
        *slot = n as f32;
    }
}

fn set_bool(v: &Value, key: &str, slot: &mut bool) {
    if let Some(b) = v.get(key).and_then(Value::as_bool) {
        *slot = b;
    }
}

fn set_usize3(v: &Value, key: &str, slot: &mut (usize, usize, usize)) {
    if let Some(arr) = v.get(key).and_then(Value::as_array) {
        if arr.len() == 3 {
            if let (Some(a), Some(b), Some(c)) = (arr[0].as_u64(), arr[1].as_u64(), arr[2].as_u64())
            {
                *slot = (a as usize, b as usize, c as usize);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ti2v_5b_preset_matches_reference() {
        let c = WanModelConfig::wan22_ti2v_5b();
        assert_eq!(c.model_type, "ti2v");
        assert_eq!(c.model_version, "2.2");
        assert_eq!(c.dim, 3072);
        assert_eq!(c.ffn_dim, 14336);
        assert_eq!(c.in_dim, 48);
        assert_eq!(c.out_dim, 48);
        assert_eq!(c.num_heads, 24);
        assert_eq!(c.num_layers, 30);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.vae_z_dim, 48);
        assert_eq!(c.vae_stride, (4, 16, 16));
        assert!(!c.dual_model);
        assert_eq!(c.boundary, 0.0);
        assert_eq!(c.sample_shift, 5.0);
        assert_eq!(c.sample_steps, 40);
        assert_eq!(c.sample_guide_scale, GuideScale::Single(5.0));
        assert_eq!(c.sample_fps, 24);
        assert_eq!(c.max_area, 704 * 1280);
        assert!(c.is_ti2v());
        assert!(c.is_wan22_vae());
        assert_eq!(c.patch_size, (1, 2, 2));
        assert!(c.qk_norm && c.cross_attn_norm);
    }

    #[test]
    fn t2v_14b_default_is_dual_moe() {
        let c = WanModelConfig::wan22_t2v_14b();
        assert!(c.dual_model);
        assert_eq!(c.boundary, 0.875);
        assert_eq!(c.sample_shift, 12.0);
        assert_eq!(c.dim, 5120);
        assert_eq!(c.num_layers, 40);
        assert_eq!(c.in_dim, 16);
        assert_eq!(c.vae_z_dim, 16);
        assert_eq!(
            c.sample_guide_scale,
            GuideScale::Dual {
                low: 3.0,
                high: 4.0
            }
        );
        assert!(!c.is_wan22_vae());
    }

    #[test]
    fn i2v_14b_is_channel_concat() {
        let c = WanModelConfig::wan22_i2v_14b();
        assert_eq!(c.model_type, "i2v");
        assert_eq!(c.in_dim, 36);
        assert!(c.is_i2v_concat());
        assert_eq!(c.boundary, 0.900);
    }

    #[test]
    fn config_json_autodetects_5b() {
        // The 5B's serialized config (model_type ti2v + dim 3072 → 5B preset).
        let v = serde_json::json!({
            "model_type": "ti2v",
            "model_version": "2.2",
            "dim": 3072,
            "in_dim": 48,
            "vae_z_dim": 48,
            "vae_stride": [4, 16, 16],
            "sample_guide_scale": 5.0,
            "sample_fps": 24
        });
        let c = WanModelConfig::from_config_json(&v);
        assert_eq!(c.num_layers, 30);
        assert_eq!(c.num_heads, 24);
        assert_eq!(c.ffn_dim, 14336);
        assert_eq!(c.sample_guide_scale, GuideScale::Single(5.0));
    }

    #[test]
    fn config_json_parses_14b_dual_guide_array() {
        // The on-disk 14B config carries a [low, high] guide array.
        let v = serde_json::json!({
            "model_type": "t2v",
            "model_version": "2.2",
            "dim": 5120,
            "dual_model": true,
            "sample_guide_scale": [3.0, 4.0],
            "boundary": 0.875
        });
        let c = WanModelConfig::from_config_json(&v);
        assert_eq!(
            c.sample_guide_scale,
            GuideScale::Dual {
                low: 3.0,
                high: 4.0
            }
        );
        assert!(c.dual_model);
    }

    #[test]
    fn config_json_parses_quantization_manifest() {
        // A dense bf16 snapshot carries no `quantization` block.
        let dense = serde_json::json!({"model_type": "t2v", "dim": 5120, "dual_model": true});
        assert_eq!(WanModelConfig::from_config_json(&dense).quantization, None);

        // A pre-quantized snapshot (convert_wan.py) carries `{bits, group_size}`.
        let q4 = serde_json::json!({
            "model_type": "t2v", "dim": 5120, "dual_model": true,
            "quantization": {"group_size": 64, "bits": 4}
        });
        assert_eq!(
            WanModelConfig::from_config_json(&q4).quantization,
            Some(WanQuant {
                bits: 4,
                group_size: 64
            })
        );

        // Missing keys fall back to the reference/mflux defaults (group 64, 4-bit).
        let bare = serde_json::json!({"model_type": "t2v", "quantization": {}});
        assert_eq!(
            WanModelConfig::from_config_json(&bare).quantization,
            Some(WanQuant {
                bits: 4,
                group_size: 64
            })
        );
    }
}
