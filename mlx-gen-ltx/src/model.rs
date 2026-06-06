//! `mlx-gen-ltx` model entry: the LTX-2.3 **AudioVideo** descriptor, the config-driven `load`, the
//! public `generate`, and registry self-registration.
//!
//! **Scope (sc-2684):** the production path is the full **synchronized audio+video** generation
//! (`generate_av.py`) — prompt → Gemma-3 tokenizer → [`LtxTextEncoder::encode_av`] (video 4096 +
//! audio 2048 embeddings) → seeded noise → the joint 2-stage distilled denoise ([`generate_av_latents`]:
//! both streams through the dual-modality [`AvDiT`] with cross-modal attention every step; the video is
//! 2× upsampled between stages, the audio is not) → [`LtxVideoVae`] decode → uint8 RGB frames **plus**
//! [`AudioDecoder`] → [`LtxVocoder`] → an [`mlx_gen::media::AudioTrack`]. The audio is always denoised
//! (it conditions the video via cross-modal attention), so the video differs from the video-only
//! sc-2679 building block (`LtxDiT`, audio disabled). `--no-audio` (`req.video_mode == "no_audio"`)
//! runs the full A/V denoise but skips the audio decode (`audio: None`).
//!
//! 16-bit-WAV write + peak-normalize + the `ffmpeg -c:v copy -c:a aac -shortest` mux are **host-side**
//! (the `AudioTrack` is the raw vocoder waveform — `generate_av.py`'s `audio_np` before `save_audio`),
//! matching how MP4 video muxing already lives outside the crate (the Wan sibling).
//!
//! The Gemma text-encoder weights are a **separate** snapshot (the base model dir holds only the
//! `connector`/transformer/vae); [`resolve_gemma_dir`] locates them via `$LTX_GEMMA_DIR` or the HF
//! cache (`mlx-community/gemma-3-12b-it-bf16`).
//!
//! **Quantization (sc-2686).** The transformer ships **selectively quantized** (attn/ff Linears
//! packed U32 + `scales`); the **bits/group ride on the checkpoint's `split_model.json`** —
//! `ltx_2_3_base_q4` is **Q4**, `ltx_2_3_base_q8` is **Q8**, group 64 — read into the DiT
//! [`Precision`], never hardcoded. `LoadSpec::quantize`, when set, only *asserts* the expected level
//! (LTX can't re-quantize a dense checkpoint — there is no dense LTX transformer; it ships pre-packed),
//! so a mismatch with the manifest is a load error. Connector / VAE / upsampler are dense bf16 (the
//! reference quantizes the transformer only); the Gemma text encoder is dense bf16 by default
//! (reference TE quant rides on the *Gemma* snapshot's `config.json`).
//!
//! **Precision.** Selected by `LoadSpec::precision`: `Bf16` (the default) → the reference's **native**
//! bf16 activations × quantized weights ([`Precision::quant_bf16`]) — the production-speed path;
//! `Fp32` → [`Precision::quant_f32`] (f32 activations × quantized weights) — the quality target. Both
//! are bit-exact to their reference golden (sc-2842). The latent statistics follow the path dtype (so
//! the upsampler + denoise run in that precision); the VAE decode stays f32 (a post-sampling quality
//! island, pixel-parity either way), and the Gemma backbone runs bf16 as the reference does. Distilled
//! 2-stage → **no CFG** (guidance baked in).
//!
//! **I2V (sc-2685):** a single conditioning [`Conditioning::Reference`] image is VAE-encoded at both
//! stage resolutions and injected into the **video** stream as a clean latent at frame 0 (per-frame
//! denoise mask, `image_strength` → `1 − strength`), threaded through the joint A/V denoise via
//! `generate_av_latents`' `video_cond` — the audio stays pure-noise, matching `generate_av.py`'s
//! I2V+Audio. The VAE **encoder** is loaded for this. LoRA/LoKr are sibling slices.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Precision as LoadPrecision, Progress, Result, WeightsSource,
};

use crate::audio_vae::AudioDecoder;
use crate::config::{AudioVaeConfig, LtxConfig, LtxVaeConfig, SplitModel, VocoderConfig};
use crate::enhance::{self, EnhanceConfig, SampleParams};
use crate::gemma::{GemmaConfig, GemmaModel, GemmaQuant};
use crate::pipeline::{
    decode_audio_track, decode_to_frames, generate_av_latents, preprocess_conditioning_image,
};
use crate::positions::{compute_audio_frames, create_audio_position_grid, create_position_grid};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::LtxTokenizer;
use crate::transformer::{AvDiT, Precision};
use crate::upsampler::LatentUpsampler;
use crate::vae::LtxVideoVae;
use crate::vocoder::LtxVocoder;

/// Public registry id: `mlx_gen::load("ltx_2_3", spec)`.
pub const MODEL_ID: &str = "ltx_2_3";

/// Reference text-encoder token budget (`LTX2TextEncoder.encode` default `max_length=1024`).
const MAX_PROMPT_TOKENS: usize = 1024;
/// LTX-2 latent channels.
const LATENT_CHANNELS: i32 = 128;
/// Audio latent channels (pre-patchify) and mel bins — the audio latent is `(1, 8, T, 16)`.
const AUDIO_LATENT_CHANNELS: i32 = 8;
const AUDIO_MEL_BINS: i32 = 16;
/// VAE temporal compression (8×): `latent_frames = 1 + (frames − 1) / 8`.
const TEMPORAL_SCALE: u32 = 8;
/// VAE spatial compression (32×); stage-1 additionally halves resolution.
const SPATIAL_SCALE: u32 = 32;
/// I2V conditioning strength when neither the `Reference` nor `req.strength` supplies one (reference
/// CLI `--image-strength` default): `1.0` = full denoise, fully pinning the conditioned frame.
const DEFAULT_IMAGE_STRENGTH: f32 = 1.0;
/// I2V conditioned frame index (reference CLI `--image-frame-idx` default). Single-image I2V pins the
/// **first** latent frame; multi-keyframe / first-last-frame at other indices is parity-plus (the
/// [`crate::conditioning`] primitive supports any index, but the reference CLI only wires one).
const IMAGE_FRAME_IDX: i32 = 0;

/// Stable identity + advertised capabilities for the LTX-2.3 AudioVideo model (produces video frames
/// + a synchronized audio track).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ltx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Distilled 2-stage path: CFG is forced to 1.0, so no guidance / negative prompt.
            // I2V single-image conditioning (sc-2685) is wired via `Reference`; audio is always
            // produced (sc-2684). Q4/Q8-of-everything is a sibling slice.
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference],
            // LoRA (sc-2687) + LoKr (sc-2393) in generate: forward-time residuals + per-pass
            // strength over the full video+audio+cross-modal surface.
            supports_lora: true,
            supports_lokr: true,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            // height/width must be divisible by 64 (stage-1 runs at //2//32).
            min_size: 64,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded LTX-2.3 model: the assembled **AudioVideo** components + the cached descriptor. The
/// production path is the joint A/V denoise (`generate_av.py`) — the audio latents are always
/// denoised (the cross-modal attention couples them to the video every step), so the video stream
/// differs from the video-only sc-2679 building block. Audio is decoded into the output unless
/// `--no-audio` (`req.video_mode == "no_audio"`).
pub struct Ltx {
    descriptor: ModelDescriptor,
    tokenizer: LtxTokenizer,
    text_encoder: LtxTextEncoder,
    transformer: AvDiT,
    upsampler: LatentUpsampler,
    vae: LtxVideoVae,
    audio_decoder: AudioDecoder,
    vocoder: LtxVocoder,
    latent_mean: Array,
    latent_std: Array,
    audio_sample_rate: u32,
    stat_dt: Dtype,
}

/// Locate the Gemma-3-12B text-encoder snapshot. `$LTX_GEMMA_DIR` wins; otherwise the newest
/// `mlx-community/gemma-3-12b-it-bf16` snapshot in the HF cache.
fn resolve_gemma_dir() -> Result<std::path::PathBuf> {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return Ok(d.into());
    }
    let home = std::env::var("HOME").map_err(|_| Error::Msg("ltx_2_3: HOME unset".into()))?;
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots");
    let newest = std::fs::read_dir(&base)
        .map_err(|_| {
            Error::Msg(format!(
                "ltx_2_3: gemma snapshot not found at {} (set $LTX_GEMMA_DIR)",
                base.display()
            ))
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    newest.ok_or_else(|| Error::Msg("ltx_2_3: no gemma snapshot in the HF cache".into()))
}

/// Read the Gemma snapshot's `config.json` top-level `quantization` block — the reference TE-quant
/// trigger (`utils.apply_quantization`). `None` for the default `…-bf16` snapshot (no block). Only the
/// `affine` mode is consumed (the one `quantized_matmul`/`dequantize` implement); a non-affine mode is
/// a hard error rather than a silent mis-decode.
fn resolve_gemma_quant(gemma_dir: &std::path::Path) -> Result<Option<GemmaQuant>> {
    let path = gemma_dir.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("ltx_2_3: parse gemma config.json: {e}")))?;
    let Some(q) = v.get("quantization") else {
        return Ok(None);
    };
    if let Some(mode) = q.get("mode").and_then(|m| m.as_str()) {
        if mode != "affine" {
            return Err(Error::Msg(format!(
                "ltx_2_3: gemma quantization mode {mode:?} is not supported (only affine)"
            )));
        }
    }
    match (
        q.get("group_size").and_then(|x| x.as_i64()),
        q.get("bits").and_then(|x| x.as_i64()),
    ) {
        (Some(g), Some(b)) => Ok(Some(GemmaQuant {
            group: g as i32,
            bits: b as i32,
        })),
        _ => Ok(None),
    }
}

/// Locate the **uncensored** 4-bit Gemma enhancer snapshot (sc-2845 `--use-uncensored-enhancer`,
/// reference `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`). `$LTX_UNCENSORED_GEMMA_DIR` wins; otherwise
/// the newest matching snapshot in the HF cache. A standalone mlx_lm checkpoint (`model.` key prefix).
fn resolve_uncensored_dir() -> Result<std::path::PathBuf> {
    if let Ok(d) = std::env::var("LTX_UNCENSORED_GEMMA_DIR") {
        return Ok(d.into());
    }
    let home = std::env::var("HOME").map_err(|_| Error::Msg("ltx_2_3: HOME unset".into()))?;
    let base = std::path::PathBuf::from(home).join(
        ".cache/huggingface/hub/models--TheCluster--amoral-gemma-3-12B-v2-mlx-4bit/snapshots",
    );
    let newest = std::fs::read_dir(&base)
        .map_err(|_| {
            Error::Msg(format!(
                "ltx_2_3: uncensored enhancer snapshot not found at {} — set \
                 $LTX_UNCENSORED_GEMMA_DIR or download TheCluster/amoral-gemma-3-12B-v2-mlx-4bit",
                base.display()
            ))
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    newest.ok_or_else(|| {
        Error::Msg("ltx_2_3: no uncensored enhancer snapshot in the HF cache".into())
    })
}

/// Load the model from a split-weight snapshot directory (the `ltx_2_3_base*` tree). Reads
/// `embedded_config.json`, locates the Gemma TE separately, and assembles every component.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ltx_2_3: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    // Quantization geometry rides on the checkpoint's `split_model.json` (sc-2686): the transformer is
    // shipped selectively quantized (Q4 for `base_q4`, Q8 for `base_q8`), bits/group from the
    // manifest — never hardcoded. The per-Linear `.scales` predicate (in `transformer.rs`) then picks
    // which Linears are quantized, matching `generate_av.py`'s `_should_quantize`.
    let split = SplitModel::from_model_dir(root)?;
    // `spec.quantize`, when set, only *asserts* the expected level. LTX can't re-quantize a dense
    // checkpoint (there is no dense LTX transformer — it ships pre-packed from the reference
    // `convert.py`, which casts f32→bf16 before quantizing), so a mismatch is a hard load error
    // rather than a silent re-quant.
    if let Some(q) = spec.quantize {
        if !split.quantized {
            return Err(Error::Msg(format!(
                "ltx_2_3: spec.quantize={q:?} but {} carries no split_model.json quant manifest — \
                 LTX quant is checkpoint-driven; point at a quantized checkpoint (e.g. ltx_2_3_base_q4)",
                root.display()
            )));
        }
        if q.bits() != split.bits {
            return Err(Error::Msg(format!(
                "ltx_2_3: spec.quantize={q:?} (bits {}) disagrees with the checkpoint's \
                 split_model.json (bits {})",
                q.bits(),
                split.bits
            )));
        }
    }
    // Precision selection. `Bf16` (the [`LoadSpec`] default) → the reference's **native** bf16
    // activations × quantized weights — the production-speed path; `Fp32` → f32 activations ×
    // quantized weights — the quality target. Both are bit-exact to their reference golden (sc-2842;
    // the distilled stage-1 sampler is chaos-sensitive, so each per-forward is bit-exact). The latent
    // statistics (the upsampler's un-/re-normalize) follow the path dtype so the whole denoise stays
    // in that precision; the VAE decode stays f32 in both — a post-sampling quality island.
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (
            Precision::quant_bf16(split.bits, split.group),
            Dtype::Bfloat16,
        ),
        LoadPrecision::Fp32 => (
            Precision::quant_f32(split.bits, split.group),
            Dtype::Float32,
        ),
    };

    let config = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;
    let audio_vae_config = AudioVaeConfig::from_model_dir(root)?;
    let vocoder_config = VocoderConfig::from_model_dir(root)?;

    let gemma_dir = resolve_gemma_dir()?;
    let gemma_w = Weights::from_dir(&gemma_dir)?;
    // Selectively quantize the Gemma backbone iff the snapshot's `config.json` says so (the reference
    // `apply_quantization` path; sc-2686). The default `…-bf16` snapshot ⇒ `None` ⇒ dense bf16 TE.
    let gemma_quant = resolve_gemma_quant(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let upsampler_w = Weights::from_file(root.join("upsampler.safetensors"))?;
    let vae_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let audio_vae_w = Weights::from_file(root.join("audio_vae.safetensors"))?;
    let vocoder_w = Weights::from_file(root.join("vocoder.safetensors"))?;
    // The VAE **encoder** is loaded so the model can serve I2V (sc-2685): it VAE-encodes the
    // conditioning image at both stage resolutions (pure-T2V+A requests never touch it). The reference
    // `generate_av.py` supports I2V+Audio — the video is image-conditioned, the audio stays pure-noise.
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // The AudioVideo text encoder runs **bf16** activations (the reference TE dtype; S1-validated) —
    // dense for the default `…-bf16` Gemma or selectively quantized per the snapshot — producing both
    // the video (4096) and audio (2048) embeddings. Its bf16 embeddings enter the DiT, which upcasts
    // the cross-attn context as the reference transformer does.
    let text_encoder = LtxTextEncoder::from_weights_av(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        gemma_quant,
        &config,
        Dtype::Bfloat16,
    )?;
    let mut transformer = AvDiT::from_weights(&transformer_w, &config, dit_prec)?;
    // LoRA (sc-2687) + LoKr (sc-2393) in generate: forward-time residuals over the (quantized/dense)
    // base, applied on the still-mutable transformer — the load-time seam. Routes the full
    // video+audio+cross-modal surface. `pass_scales` (per-adapter) carries one strength per distilled
    // denoise pass; the pipeline selects the active pass per stage. No-op when `spec.adapters` empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_ltx_adapters(
            &mut transformer,
            &spec.adapters,
            crate::pipeline::NUM_DENOISE_PASSES,
        )?;
    }

    let upsampler = LatentUpsampler::from_weights(&upsampler_w)?;
    // The VAE carries its encoder (Some) so the model can serve I2V conditioning.
    let vae = LtxVideoVae::from_weights(&vae_w, Some(&vae_enc_w), &vae_config)?;
    // The audio VAE decoder + vocoder run f32 (post-sampling quality islands, gated bit-exact).
    let audio_decoder = AudioDecoder::from_weights(&audio_vae_w, &audio_vae_config)?;
    let vocoder = LtxVocoder::from_weights(&vocoder_w, &vocoder_config)?;
    let audio_sample_rate = vocoder_config.final_sample_rate() as u32;
    // The VAE `per_channel_statistics` double as the upsampler's latent norm, at the path dtype.
    let latent_mean = to_dtype(vae_w.require("per_channel_statistics.mean")?, stat_dt)?;
    let latent_std = to_dtype(vae_w.require("per_channel_statistics.std")?, stat_dt)?;

    Ok(Box::new(Ltx {
        descriptor: descriptor(),
        tokenizer: LtxTokenizer::from_dir(&gemma_dir)?,
        text_encoder,
        transformer,
        upsampler,
        vae,
        audio_decoder,
        vocoder,
        latent_mean,
        latent_std,
        audio_sample_rate,
        stat_dt,
    }))
}

impl Ltx {
    /// Latent dims `(frames, stage1_h, stage1_w, stage2_h, stage2_w)` for a request.
    pub(crate) fn latent_dims(req: &GenerationRequest) -> (usize, usize, usize, usize, usize) {
        let frames = req.frames.unwrap_or(1).max(1);
        let latent_frames = 1 + (frames as usize - 1) / TEMPORAL_SCALE as usize;
        let (h, w) = (req.height, req.width);
        (
            latent_frames,
            (h / 2 / SPATIAL_SCALE) as usize,
            (w / 2 / SPATIAL_SCALE) as usize,
            (h / SPATIAL_SCALE) as usize,
            (w / SPATIAL_SCALE) as usize,
        )
    }

    /// Audio latent-frame count for the request (`compute_audio_frames(num_frames, fps)`).
    pub(crate) fn audio_frames(req: &GenerationRequest) -> usize {
        compute_audio_frames(
            req.frames.unwrap_or(1).max(1) as usize,
            req.fps.unwrap_or(24) as f64,
        )
    }

    /// `--no-audio` toggle: `req.video_mode == "no_audio"` runs the full A/V denoise but skips the
    /// audio decode + returns `audio: None` (the reference `--no-audio`).
    fn no_audio(req: &GenerationRequest) -> bool {
        matches!(
            req.video_mode.as_deref(),
            Some("no_audio") | Some("video_only")
        )
    }

    /// The full A/V path with **injected** stage noise (the deterministic seam `generate` calls with
    /// RNG-drawn noise and the e2e parity test calls with the reference samples). Encodes the prompt
    /// to both video + audio embeddings, resolves an optional I2V conditioning image (VAE-encoded at
    /// both stage resolutions — the video is image-conditioned, the audio stays pure-noise, matching
    /// `generate_av.py`'s I2V+Audio), then defers to
    /// [`generate_av_from_embeddings`](Self::generate_av_from_embeddings).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_with_noise(
        &self,
        req: &GenerationRequest,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (ids, mask) = self.tokenizer.encode(&req.prompt, MAX_PROMPT_TOKENS)?;
        let (video_ctx, audio_ctx) = self.text_encoder.encode_av(&ids, &mask)?;
        // I2V: VAE-encode the conditioning image at each stage's pixel resolution (half-res + full-res).
        let cond = match self.resolve_reference(req)? {
            Some((image, strength)) => Some((
                self.encode_conditioning(image, req.height / 2, req.width / 2)?,
                self.encode_conditioning(image, req.height, req.width)?,
                strength,
            )),
            None => None,
        };
        let video_cond = cond
            .as_ref()
            .map(|(img1, img2, strength)| (img1, img2, IMAGE_FRAME_IDX, *strength));
        self.generate_av_from_embeddings(
            req,
            &video_ctx,
            &audio_ctx,
            video_s1,
            video_s2,
            audio_s1,
            audio_s2,
            video_cond,
            on_progress,
        )
    }

    /// The A/V path from **injected** text embeddings + noise — the pipeline-only seam (no Gemma), so
    /// the parity test can gate the joint 2-stage pipeline + video/audio decode against the reference
    /// conditioning. `video_ctx` `(1, ctx, 4096)`, `audio_ctx` `(1, ctx, 2048)`; video noise
    /// `(1,128,F,h,w)` per stage; audio noise `(1,8,T,16)` per stage (`T = audio_frames`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn generate_av_from_embeddings(
        &self,
        req: &GenerationRequest,
        video_ctx: &Array,
        audio_ctx: &Array,
        video_s1: &Array,
        video_s2: &Array,
        audio_s1: &Array,
        audio_s2: &Array,
        video_cond: Option<(&Array, &Array, i32, f32)>,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let pos1 = create_position_grid(1, lf, h1, w1);
        let pos2 = create_position_grid(1, lf, h2, w2);
        let audio_pos = create_audio_position_grid(1, Self::audio_frames(req));

        let mut step = 0usize;
        let (video_latents, audio_latents) = generate_av_latents(
            &self.transformer,
            &self.upsampler,
            video_s1,
            &pos1,
            video_s2,
            &pos2,
            audio_s1,
            audio_s2,
            &audio_pos,
            video_ctx,
            audio_ctx,
            &self.latent_mean,
            &self.latent_std,
            video_cond,
            &mut |_| {
                step += 1;
                on_progress(Progress::Step {
                    current: step as u32,
                    total: 11,
                });
            },
        )?;

        on_progress(Progress::Decoding);
        let frames = decode_to_frames(&self.vae, &video_latents)?;
        let images = frames_to_images(&frames)?;
        // Audio always denoised (it conditions the video); decode it unless `--no-audio`.
        let audio = if Self::no_audio(req) {
            None
        } else {
            Some(decode_audio_track(
                &self.audio_decoder,
                &self.vocoder,
                &audio_latents,
                self.audio_sample_rate,
            )?)
        };
        Ok(GenerationOutput::Video {
            frames: images,
            fps: req.fps.unwrap_or(24),
            audio,
        })
    }

    /// Extract the single I2V conditioning image + its strength from the request. The per-reference
    /// strength wins over `req.strength`, falling back to [`DEFAULT_IMAGE_STRENGTH`]. LTX I2V
    /// conditions on exactly one image (multi-keyframe / first-last-frame is parity-plus), so more
    /// than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, f32)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "ltx_2_3: multiple reference images are not supported (single-image I2V \
                         only; multi-keyframe / first-last-frame is parity-plus, sc-2685)"
                            .into(),
                    ));
                }
                reference = Some((
                    image,
                    strength.or(req.strength).unwrap_or(DEFAULT_IMAGE_STRENGTH),
                ));
            }
        }
        Ok(reference)
    }

    /// VAE-encode the conditioning image at a stage's pixel resolution `(px_h, px_w)` → the f32 clean
    /// latent `(1, 128, 1, px_h/32, px_w/32)`. The encoder is an f32 quality island (like the VAE
    /// decode); the caller casts the latent to the path dtype.
    fn encode_conditioning(&self, image: &Image, px_h: u32, px_w: u32) -> Result<Array> {
        let video = preprocess_conditioning_image(image, px_w, px_h)?; // f32 (1,3,1,px_h,px_w)
        self.vae.encode(&video)
    }

    /// Prompt enhancement (sc-2845). Returns the rewritten prompt when `req.enhance_prompt` is set and
    /// the enhancer produces non-empty output; `None` (use the original prompt) when off, or — matching
    /// `generate_av.py`'s try/except — on **any** enhancer failure or empty output. Failures are logged
    /// to stderr with the reference's `ENHANCER_FALLBACK:` token; success with `ENHANCED_PROMPT:`.
    fn maybe_enhance(&self, req: &GenerationRequest) -> Option<String> {
        if !req.enhance_prompt {
            return None;
        }
        match self.run_enhance(req) {
            Ok(p) if !p.trim().is_empty() => {
                eprintln!("ENHANCED_PROMPT:{p}");
                Some(p)
            }
            Ok(_) => {
                eprintln!("ENHANCER_FALLBACK:EmptyOutput:prompt enhancer returned empty output");
                None
            }
            Err(e) => {
                eprintln!("ENHANCER_FALLBACK:{e}");
                None
            }
        }
    }

    /// Run the configured enhancer: the uncensored 4-bit Gemma (`use_uncensored_enhancer`) or the
    /// already-loaded text-encoder backbone. I2V (a `Reference` image present) selects the I2V system
    /// prompt **only on the uncensored path** — the reference's censored `enhance_t2v` always uses the
    /// T2V system prompt (`generate_av.py` never calls `enhance_i2v` there), which we match.
    fn run_enhance(&self, req: &GenerationRequest) -> Result<String> {
        let is_i2v = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let temperature = req
            .enhance_temperature
            .unwrap_or(enhance::DEFAULT_TEMPERATURE);
        let cfg = EnhanceConfig {
            max_tokens: req
                .enhance_max_tokens
                .map(|m| m as usize)
                .unwrap_or(enhance::DEFAULT_MAX_TOKENS),
            seed: req.seed.unwrap_or(enhance::DEFAULT_SEED),
        };
        if req.use_uncensored_enhancer {
            let (model, tokenizer) = Self::load_uncensored_enhancer()?;
            let system = if is_i2v {
                enhance::I2V_SYSTEM_PROMPT
            } else {
                enhance::T2V_SYSTEM_PROMPT
            };
            enhance::enhance(
                &model,
                &tokenizer,
                system,
                &req.prompt,
                &cfg,
                &SampleParams::uncensored(temperature),
            )
        } else {
            enhance::enhance(
                self.text_encoder.gemma(),
                &self.tokenizer,
                enhance::T2V_SYSTEM_PROMPT,
                &req.prompt,
                &cfg,
                &SampleParams::censored(temperature),
            )
        }
    }

    /// Load the separate uncensored 4-bit Gemma enhancer + its tokenizer on demand (the reference
    /// `enhance_with_model` loads it per call). A standalone mlx_lm checkpoint → `model.` key prefix;
    /// its `config.json` `quantization` block drives the 4-bit dequant.
    fn load_uncensored_enhancer() -> Result<(GemmaModel, LtxTokenizer)> {
        let dir = resolve_uncensored_dir()?;
        let w = Weights::from_dir(&dir)?;
        let quant = resolve_gemma_quant(&dir)?;
        let model =
            GemmaModel::from_weights_with_prefix(&w, GemmaConfig::gemma_3_12b(), quant, "model.")?;
        let tokenizer = LtxTokenizer::from_dir(&dir)?;
        Ok((model, tokenizer))
    }
}

/// Capability-driven request validation (weight-free, so it's unit-testable without a load):
/// non-empty prompt, 64-aligned width/height (stage-1 runs at //2//32), `num_frames = 1 + 8·k`, and
/// only advertised conditioning kinds (I2V `Reference`; everything else is rejected).
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("ltx_2_3: prompt must not be empty".into()));
    }
    if !req.width.is_multiple_of(64) || !req.height.is_multiple_of(64) {
        return Err(Error::Msg(format!(
            "ltx_2_3: width/height must be divisible by 64 (got {}x{})",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames % 8 != 1 {
            return Err(Error::Msg(format!(
                "ltx_2_3: num_frames must be 1 + 8·k (got {frames})"
            )));
        }
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
            return Err(Error::Msg(format!(
                "ltx_2_3 does not accept {kind:?} conditioning (single-image I2V via Reference only)"
            )));
        }
    }
    Ok(())
}

/// `(F, H, W, 3)` uint8 → one [`Image`] per frame.
pub(crate) fn frames_to_images(frames: &Array) -> Result<Vec<Image>> {
    let sh = frames.shape(); // (F, H, W, 3)
    let (f, h, w) = (sh[0] as usize, sh[1] as u32, sh[2] as u32);
    let data = frames.as_slice::<u8>();
    let per = (h as usize) * (w as usize) * 3;
    Ok((0..f)
        .map(|i| Image {
            width: w,
            height: h,
            pixels: data[i * per..(i + 1) * per].to_vec(),
        })
        .collect())
}

impl Generator for Ltx {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor.capabilities, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        // Prompt enhancement (sc-2845): rewrite `req.prompt` before any encoding. Default off; on any
        // enhancer failure / empty output it falls back to the original prompt (reference-faithful), so
        // the e2e parity seams (which build requests with `enhance_prompt = false`) are untouched.
        let enhanced = self.maybe_enhance(req);
        let owned;
        let req = match enhanced {
            Some(prompt) => {
                owned = GenerationRequest {
                    prompt,
                    ..req.clone()
                };
                &owned
            }
            None => req,
        };
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let af = Self::audio_frames(req) as i32;
        let seed = req.seed.unwrap_or_else(default_seed);
        // Seeded noise at the path dtype (the reference seeds `normal(...).astype(model_dtype)`). RNG
        // is not portable to mlx-python, so the pixel/waveform parity gate injects the reference
        // samples via `generate_with_noise`. Distinct keys per stage/modality. I2V conditioning (when
        // a `Reference` is supplied) + the audio decode are handled inside `generate_with_noise`.
        let normal = |key: u64, shape: &[i32]| -> Result<Array> {
            let k = random::key(key)?;
            Ok(random::normal::<f32>(shape, None, None, Some(&k))?.as_dtype(self.stat_dt)?)
        };
        let video_s1 = normal(seed, &[1, LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32])?;
        let video_s2 = normal(
            seed.wrapping_add(1),
            &[1, LATENT_CHANNELS, lf as i32, h2 as i32, w2 as i32],
        )?;
        let audio_s1 = normal(
            seed.wrapping_add(2),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        let audio_s2 = normal(
            seed.wrapping_add(3),
            &[1, AUDIO_LATENT_CHANNELS, af, AUDIO_MEL_BINS],
        )?;
        self.generate_with_noise(req, &video_s1, &video_s2, &audio_s1, &audio_s2, on_progress)
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latent_dims_matches_reference_formula() {
        // 256×256, 9 frames: latent_frames = 1+(9-1)/8 = 2; stage1 = H/2/32 = 4; stage2 = H/32 = 8.
        let req = GenerationRequest {
            width: 256,
            height: 256,
            frames: Some(9),
            ..Default::default()
        };
        assert_eq!(Ltx::latent_dims(&req), (2, 4, 4, 8, 8));
        // 512×768, 1 frame: latent_frames = 1; stage1 = 8×12; stage2 = 16×24.
        let req = GenerationRequest {
            width: 768,
            height: 512,
            frames: Some(1),
            ..Default::default()
        };
        assert_eq!(Ltx::latent_dims(&req), (1, 8, 12, 16, 24));
    }

    #[test]
    fn validate_request_enforces_constraints() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(33),
            ..Default::default()
        };
        assert!(validate_request(&caps, &base).is_ok());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                prompt: String::new(),
                ..base.clone()
            }
        )
        .is_err());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                width: 500,
                ..base.clone()
            }
        )
        .is_err());
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                frames: Some(32),
                ..base.clone()
            }
        )
        .is_err());
    }

    #[test]
    fn validate_request_conditioning() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a".into(),
            width: 512,
            height: 512,
            frames: Some(9),
            ..Default::default()
        };
        let img = Image {
            width: 4,
            height: 4,
            pixels: vec![0u8; 4 * 4 * 3],
        };
        // A single I2V `Reference` is accepted.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                conditioning: vec![Conditioning::Reference {
                    image: img.clone(),
                    strength: Some(0.8),
                }],
                ..base.clone()
            }
        )
        .is_ok());
        // Unsupported conditioning (e.g. Depth) is rejected.
        assert!(validate_request(
            &caps,
            &GenerationRequest {
                conditioning: vec![Conditioning::Depth { image: img.clone() }],
                ..base.clone()
            }
        )
        .is_err());
        // More than one `Reference` is rejected at resolve time (single-image I2V only).
        let two = GenerationRequest {
            conditioning: vec![
                Conditioning::Reference {
                    image: img.clone(),
                    strength: None,
                },
                Conditioning::Reference {
                    image: img,
                    strength: None,
                },
            ],
            ..base
        };
        // resolve_reference needs an `Ltx`; assert the capability check passes but resolve errors is
        // covered by the integration path — here just confirm the kinds are individually accepted.
        assert!(validate_request(&caps, &two).is_ok());
    }

    #[test]
    fn frames_to_images_splits_per_frame() {
        // (F=2, H=1, W=2, 3): each frame = 6 bytes.
        let data: Vec<u8> = (0..12).collect();
        let frames = Array::from_slice(&data, &[2, 1, 2, 3]);
        let imgs = frames_to_images(&frames).unwrap();
        assert_eq!(imgs.len(), 2);
        assert_eq!((imgs[0].width, imgs[0].height), (2, 1));
        assert_eq!(imgs[0].pixels, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(imgs[1].pixels, vec![6, 7, 8, 9, 10, 11]);
    }
}
