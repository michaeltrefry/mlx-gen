//! `mlx-gen-ltx` model entry: the LTX-2.3 **video-only T2V** descriptor, the config-driven `load`,
//! the public `generate`, and registry self-registration.
//!
//! **Scope (sc-2679 S6):** the full end-to-end T2V pipeline is now wired — prompt → Gemma-3 tokenizer
//! → [`LtxTextEncoder`] (Gemma backbone + feature extractor + connector) → seeded noise → the
//! 2-stage distilled denoise ([`crate::pipeline`]: stage-1 half-res → 2× [`LatentUpsampler`] →
//! re-noise → stage-2 full-res) → [`LtxVideoVae`] decode → uint8 RGB frames.
//!
//! The Gemma text-encoder weights are a **separate** snapshot (the base model dir holds only the
//! `connector`/transformer/vae); [`resolve_gemma_dir`] locates them via `$LTX_GEMMA_DIR` or the HF
//! cache (`mlx-community/gemma-3-12b-it-bf16`).
//!
//! **Precision.** Selected by `LoadSpec::precision`: `Bf16` (the default) → the reference's **native**
//! bf16 activations × Q8 ([`Precision::Bf16Q8`]) — the production-speed path; `Fp32` →
//! [`Precision::F32Q8`] (f32 activations × Q8) — the quality target. Both are bit-exact to their
//! reference golden (sc-2842). The latent statistics follow the path dtype (so the upsampler + denoise
//! run in that precision); the VAE decode stays f32 (a post-sampling quality island, pixel-parity
//! either way), and the Gemma backbone runs bf16 as the reference does. Distilled 2-stage → **no CFG**
//! (guidance baked in).
//!
//! **I2V (sc-2685):** a single conditioning [`Conditioning::Reference`] image is VAE-encoded at both
//! stage resolutions and injected as a clean latent at frame 0 (per-frame denoise mask, `image_strength`
//! → `1 − strength`), driving the conditioned 2-stage denoise ([`generate_i2v_latents`]). The VAE
//! **encoder** is loaded for this. Q4/Q8-of-everything, LoRA/LoKr, and the audio half are sibling slices.

use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    Precision as LoadPrecision, Progress, Result, WeightsSource,
};

use crate::config::{LtxConfig, LtxVaeConfig};
use crate::gemma::GemmaConfig;
use crate::pipeline::decode_to_frames;
use crate::pipeline::{generate_i2v_latents, generate_t2v_latents, preprocess_conditioning_image};
use crate::positions::create_position_grid;
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::LtxTokenizer;
use crate::transformer::{LtxDiT, Precision};
use crate::upsampler::LatentUpsampler;
use crate::vae::LtxVideoVae;

/// Public registry id: `mlx_gen::load("ltx_2_3", spec)`.
pub const MODEL_ID: &str = "ltx_2_3";

/// Reference text-encoder token budget (`LTX2TextEncoder.encode` default `max_length=1024`).
const MAX_PROMPT_TOKENS: usize = 1024;
/// LTX-2 latent channels.
const LATENT_CHANNELS: i32 = 128;
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

/// Stable identity + advertised capabilities for the LTX-2.3 video-only core.
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "ltx",
        modality: Modality::Video,
        capabilities: Capabilities {
            // Distilled 2-stage path: CFG is forced to 1.0, so no guidance / negative prompt in
            // the core. I2V single-image conditioning (sc-2685) is wired via `Reference`. (LoRA,
            // LoKr, Q4/Q8, and the audio half are sibling slices.)
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
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

/// The loaded LTX-2.3 model: the assembled T2V components + the cached descriptor.
pub struct Ltx {
    descriptor: ModelDescriptor,
    tokenizer: LtxTokenizer,
    text_encoder: LtxTextEncoder,
    transformer: LtxDiT,
    upsampler: LatentUpsampler,
    vae: LtxVideoVae,
    latent_mean: Array,
    latent_std: Array,
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
    // Precision selection. `Bf16` (the [`LoadSpec`] default) → the reference's **native** bf16
    // activations × Q8 — the production-speed path; `Fp32` → f32 activations × Q8 — the quality
    // target. Both are bit-exact to their reference golden (sc-2842; the distilled stage-1 sampler is
    // chaos-sensitive, so each per-forward is bit-exact). The latent statistics (the upsampler's
    // un-/re-normalize) follow the path dtype so the whole denoise stays in that precision; the VAE
    // decode stays f32 in both — a post-sampling quality island (pixel-parity either way).
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (Precision::Bf16Q8, Dtype::Bfloat16),
        LoadPrecision::Fp32 => (Precision::F32Q8, Dtype::Float32),
    };
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "ltx_2_3: Q4/Q8-of-everything is a sibling slice (sc-2686); the transformer is already \
             shipped Q8"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "ltx_2_3: LoRA/LoKr adapters are sibling slices (sc-2687 / sc-2393), not yet wired"
                .into(),
        ));
    }

    let config = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;

    let gemma_dir = resolve_gemma_dir()?;
    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let upsampler_w = Weights::from_file(root.join("upsampler.safetensors"))?;
    let vae_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    // The VAE **encoder** is loaded so the model can serve I2V (sc-2685): it VAE-encodes the
    // conditioning image at both stage resolutions. ~640 MB — negligible beside the 20 GB
    // transformer, and it makes I2V available without a reload (pure-T2V requests never touch it).
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // The text encoder runs **bf16** end-to-end (the reference TE dtype; S1-validated). Its bf16
    // `video_embeddings` enter the F32Q8 DiT, which upcasts the cross-attn context to f32 — exactly
    // as the f32-upcast reference transformer does.
    let text_encoder = LtxTextEncoder::from_weights(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        &config,
        Dtype::Bfloat16,
    )?;
    let transformer = LtxDiT::from_weights(&transformer_w, &config, dit_prec)?;
    let upsampler = LatentUpsampler::from_weights(&upsampler_w)?;
    let vae = LtxVideoVae::from_weights(&vae_w, Some(&vae_enc_w), &vae_config)?;
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
        latent_mean,
        latent_std,
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

    /// Prompt → `video_embeddings` (f32) via the Gemma tokenizer + text encoder.
    fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        let (ids, mask) = self.tokenizer.encode(prompt, MAX_PROMPT_TOKENS)?;
        self.text_encoder.encode(&ids, &mask)
    }

    /// The full T2V path with **injected** stage noise (the deterministic seam `generate` calls with
    /// RNG-drawn noise and the e2e parity test calls with the reference samples). Encodes the prompt,
    /// then defers to [`generate_from_embeddings`](Self::generate_from_embeddings).
    pub(crate) fn generate_with_noise(
        &self,
        req: &GenerationRequest,
        stage1_noise: &Array,
        stage2_noise: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let context = self.encode_prompt(&req.prompt)?;
        self.generate_from_embeddings(req, &context, stage1_noise, stage2_noise, on_progress)
    }

    /// The T2V path from **injected** text embeddings + noise — the pipeline-only seam (no text
    /// encoder), so the parity test can gate the 2-stage pipeline + decode against the reference's
    /// conditioning without loading the Gemma backbone. `context` is `(1, ctx, 4096)`; `stage1_noise`
    /// `(1, 128, F, h1, w1)` and `stage2_noise` `(1, 128, F, h2, w2)`.
    pub(crate) fn generate_from_embeddings(
        &self,
        req: &GenerationRequest,
        context: &Array,
        stage1_noise: &Array,
        stage2_noise: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let pos1 = create_position_grid(1, lf, h1, w1);
        let pos2 = create_position_grid(1, lf, h2, w2);

        let mut step = 0usize;
        let latents = generate_t2v_latents(
            &self.transformer,
            &self.upsampler,
            stage1_noise,
            &pos1,
            stage2_noise,
            &pos2,
            context,
            &self.latent_mean,
            &self.latent_std,
            &mut |_| {
                step += 1;
                on_progress(Progress::Step {
                    current: step as u32,
                    total: 11,
                });
            },
        )?;

        on_progress(Progress::Decoding);
        let frames = decode_to_frames(&self.vae, &latents)?;
        let images = frames_to_images(&frames)?;
        Ok(GenerationOutput::Video {
            frames: images,
            fps: req.fps.unwrap_or(24),
            audio: None,
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

    /// The full I2V path with **injected** stage noise (the deterministic seam `generate` calls with
    /// RNG-drawn noise). Encodes the prompt + the conditioning image at both stage resolutions, then
    /// runs the 2-stage conditioned pipeline ([`generate_i2v_latents`]).
    fn generate_i2v_with_noise(
        &self,
        req: &GenerationRequest,
        image: &Image,
        strength: f32,
        stage1_noise: &Array,
        stage2_noise: &Array,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let context = self.encode_prompt(&req.prompt)?;
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let pos1 = create_position_grid(1, lf, h1, w1);
        let pos2 = create_position_grid(1, lf, h2, w2);

        // Encode the image at each stage's pixel resolution: stage 1 = half-res (height/2 × width/2),
        // stage 2 = full-res. The reference resizes the original image directly to each.
        let img1 = self.encode_conditioning(image, req.height / 2, req.width / 2)?;
        let img2 = self.encode_conditioning(image, req.height, req.width)?;

        let mut step = 0usize;
        let latents = generate_i2v_latents(
            &self.transformer,
            &self.upsampler,
            &img1,
            stage1_noise,
            &pos1,
            &img2,
            stage2_noise,
            &pos2,
            &context,
            &self.latent_mean,
            &self.latent_std,
            IMAGE_FRAME_IDX,
            strength,
            &mut |_| {
                step += 1;
                on_progress(Progress::Step {
                    current: step as u32,
                    total: 11,
                });
            },
        )?;

        on_progress(Progress::Decoding);
        let frames = decode_to_frames(&self.vae, &latents)?;
        let images = frames_to_images(&frames)?;
        Ok(GenerationOutput::Video {
            frames: images,
            fps: req.fps.unwrap_or(24),
            audio: None,
        })
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
        let (lf, h1, w1, h2, w2) = Self::latent_dims(req);
        let seed = req.seed.unwrap_or_else(default_seed);
        // Seeded f32 noise (the f32-activation regime). RNG is not portable to mlx-python, so the
        // pixel-level parity gate injects the reference samples via `generate_{with,i2v_with}_noise`.
        let k1 = random::key(seed)?;
        let k2 = random::key(seed.wrapping_add(1))?;
        let stage1_noise = random::normal::<f32>(
            &[1, LATENT_CHANNELS, lf as i32, h1 as i32, w1 as i32],
            None,
            None,
            Some(&k1),
        )?;
        let stage2_noise = random::normal::<f32>(
            &[1, LATENT_CHANNELS, lf as i32, h2 as i32, w2 as i32],
            None,
            None,
            Some(&k2),
        )?;
        // I2V when a single conditioning `Reference` is supplied; otherwise pure-noise T2V.
        match self.resolve_reference(req)? {
            Some((image, strength)) => self.generate_i2v_with_noise(
                req,
                image,
                strength,
                &stage1_noise,
                &stage2_noise,
                on_progress,
            ),
            None => self.generate_with_noise(req, &stage1_noise, &stage2_noise, on_progress),
        }
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
