//! `mlx-gen-wan` model entries: the Wan2.2 **TI2V-5B** (`wan2_2_ti2v_5b`, dense, z48 VAE — S0
//! scaffold, denoise pending in sc-2680), the Wan2.2 **T2V-A14B** (`wan2_2_t2v_14b`, dual-expert MoE,
//! z16 VAE — fully wired here on the S1–S5 core), and the Wan2.2 **I2V-A14B** (`wan2_2_i2v_14b`,
//! dual-expert MoE, channel-concat image conditioning, in_dim 36 — sc-2681), plus their registry
//! self-registration.
//!
//! The 5B `load` resolves `config.json` and stubs `generate` (its z48 VAE + dense denoise are
//! sc-2680). The shared [`Wan14b`] struct serves both A14B variants — [`Wan14b::generate`] runs the
//! complete pipeline: UMT5-XXL encode → (I2V only) build the channel-concat conditioning `y` →
//! per-step dual-expert MoE denoise (boundary-switched high/low experts, [`denoise_moe`]) → z16 VAE
//! decode → RGB8 frames, **staging** each heavy component (T5, the two 27 GB experts, the VAE) in and
//! out to bound peak memory (mirrors `generate_wan.py`). The I2V variant differs only by the `y`
//! conditioning (the image's first-frame VAE latent + temporal mask, channel-concatenated to in_dim
//! 36) and the max-area resolution cap.

use std::path::PathBuf;

use mlx_gen::tiling::TilingConfig;
use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, AdapterSpec, Capabilities, Conditioning, ConditioningKind, Error,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    MoeExpert, Precision, Progress, Quant, Result, WeightsSource,
};
use mlx_rs::random;
use mlx_rs::Array;

use crate::adapters::merge_wan_adapters;
use crate::config::{GuideScale, WanModelConfig};
use crate::pipeline::{
    align_dim, best_output_size, build_i2v_y, build_ti2v_keyframe_z, build_ti2v_mask,
    build_ti2v_multi_mask, decode_to_frames, decode_to_frames_22, denoise, denoise_moe,
    denoise_ti2v, frames_to_images, latent_shape, preprocess_ti2v_image, ti2v_blend_init, Expert,
};
use crate::scheduler::SolverKind;
use crate::text_encoder::{load_tokenizer, Umt5Encoder};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

/// Public registry id: `mlx_gen::load("wan2_2_ti2v_5b", spec)`.
pub const MODEL_ID: &str = "wan2_2_ti2v_5b";

/// Stable identity + advertised capabilities for the Wan2.2 TI2V-5B (dense text+image→video).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            // 5B uses real CFG (guide 5.0) with the Chinese anti-artifact negative prompt, and
            // accepts a single image as the TI2V mask-blend conditioning reference. Keyframe =
            // Wan-native first_last_frame / multi-keyframe (epic 3040, sc-3357) via the same
            // mask-blend, pinning the listed latent frames instead of only frame 0.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::Keyframe],
            // Q4/Q8 (sc-2682) loads via `spec.quantize` (transformer-only); LoRA/LoKr merge onto the
            // single dense model at generate time (the reference `_loras_single` path — shared
            // untagged specs only, reusing the sc-2683/sc-2393 `merge_wan_adapters` seam).
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 32; cap the long edge at 1280 (max_area 704×1280).
            min_size: 32,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Cross-attention text K/V is cached across denoise steps.
            supports_kv_cache: true,
            // Wan pins a static `sample_shift` from config (not the empirical per-resolution mu).
            requires_sigma_shift: false,
        },
    }
}

/// The loaded Wan2.2 TI2V-5B (dense). Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the single 5B DiT, the z48 vae22) are **staged** inside [`Wan::generate`] —
/// loaded, used, then dropped in turn — to bound peak memory (mirrors `generate_wan.py`, which never
/// holds the T5 encoder + the 10 GB transformer resident at once).
pub struct Wan {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    root: PathBuf,
    /// LoRA/LoKr adapters merged onto the single dense model at generate time (the reference
    /// `_loras_single` path). Empty for a plain load. `moe_expert`-tagged specs are rejected (dense).
    adapters: Vec<AdapterSpec>,
    /// Optional Q4/Q8 quantization for the transformer (sc-2682). `None` = dense bf16 (or a
    /// pre-quantized snapshot, which `from_weights` builds packed from its `config.json` manifest).
    quant: Option<Quant>,
}

impl Wan {
    /// The resolved model config (exposed for tests).
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }

    /// Merge the load-time LoRA/LoKr adapters onto the single dense model weight map in place,
    /// before the [`WanTransformer`] is built. No-op without adapters. The dense 5B has no MoE
    /// experts, so it takes only **shared** (untagged) specs — the reference's `_loras_single`
    /// (`--lora`, not `--lora-high/low`); a `moe_expert`-tagged spec is a misconfiguration here.
    /// Reuses the sc-2683/sc-2393 [`merge_wan_adapters`] seam (`MoeExpert::High` ⇒ only the
    /// `moe_expert == None` pass fires, since all specs are untagged).
    fn merge_adapters(&self, w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        if self.config.quantization.is_some() {
            return Err(Error::Msg(format!(
                "{}: LoRA adapters on a pre-quantized snapshot need dequantize-then-merge (the \
                 reference loading.py path), not yet wired — load a dense bf16 snapshot (LoRA \
                 merges, then `spec.quantize` quantizes the merged weights), or drop the adapters",
                self.descriptor.id
            )));
        }
        if self.adapters.iter().any(|s| s.moe_expert.is_some()) {
            return Err(Error::Msg(format!(
                "{}: `moe_expert` (high/low) tagging is only for the dual-expert A14B — the dense \
                 5B takes shared (untagged) adapters",
                self.descriptor.id
            )));
        }
        let report = merge_wan_adapters(w, &self.adapters, MoeExpert::High)?;
        if report.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} adapter file(s) matched no module — check the format (PEFT `lora_A/B` or \
                 kohya `lora_down/up`, `diffusion_model.`-prefixed Wan module names)",
                self.descriptor.id,
                self.adapters.len()
            )));
        }
        if !report.skipped.is_empty() {
            eprintln!(
                "{}: {} adapter target(s) not present in this checkpoint, skipped: {:?}",
                self.descriptor.id,
                report.skipped.len(),
                report.skipped
            );
        }
        Ok(())
    }
}

/// Load the Wan2.2 TI2V-5B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `model.safetensors` + `t5_encoder.safetensors` + `vae.safetensors` + `tokenizer.json` +
/// `config.json`). The DiT runs bf16 GEMMs over an f32 residual (the S3 parity regime). Q4/Q8
/// (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot; LoRA/LoKr (sc-2683 / sc-2393)
/// merge onto the single dense model at generate time.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_ti2v_5b: expected a model directory (converted MLX snapshot), not a single file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: precision override is not wired (the DiT runs bf16 GEMMs over an f32 \
             residual stream — the parity regime)"
                .into(),
        ));
    }
    let config = WanModelConfig::from_model_dir(&root)?;
    if config.dual_model || !config.is_ti2v() {
        return Err(Error::Msg(format!(
            "wan2_2_ti2v_5b: config.json is not the dense TI2V-5B (model_type={}, dual_model={}); \
             expected the converted Wan2.2 TI2V-5B checkpoint (model_type=ti2v, dual_model=false)",
            config.model_type, config.dual_model
        )));
    }
    let quant = resolve_load_time_quant(MODEL_ID, &config, spec.quantize)?;
    Ok(Box::new(Wan {
        descriptor: descriptor(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
    }))
}

impl Generator for Wan {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability floor: size range (the advertised `min_size` = patch×vae_stride = 32 is
        // the sub-tile lower bound; `max_size` caps the long edge), count, guidance/negative/true_cfg,
        // sampler (`unipc`/`euler`/`dpmpp2m`), scheduler, and conditioning (`Reference`/`Keyframe`).
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "wan2_2_ti2v_5b: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        Ok(())
    }

    /// The dense 5B pipeline (port of `generate_wan.py`'s single-model path, sc-2680) — **T2V** when
    /// no image is given, **TI2V** mask-blend when a `Reference` image is. Resolves request knobs,
    /// then **stages** the phases to bound memory: (1) UMT5 encode the prompt (+ neg, unless CFG is
    /// off); (1b, TI2V) load the z48 vae22, encode the conditioning image → `z_img`, build the
    /// first-frame mask + per-token mask, blend the noise init; (2) load the 5B DiT (merge adapters,
    /// quantize), embed the contexts, run the dense [`denoise`] (T2V) or [`denoise_ti2v`] mask-blend
    /// loop; (3) load the vae22 decoder → RGB8 frames. CFG runs with the single guidance scale.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reject anything outside the advertised surface before doing expensive work — in particular
        // an unknown `sampler`, which `solver_kind` would otherwise silently map to UniPC.
        self.validate(req)?;
        let cfg = &self.config;

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let trim_out = trim * cfg.vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim_out;
        let mut width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
        let mut height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
        // Enforce the model's max-area cap (704×1280) with an aspect-preserving, grid-aligned fit.
        if cfg.max_area > 0 && (width as usize) * (height as usize) > cfg.max_area {
            let dw = (cfg.patch_size.2 * cfg.vae_stride.2) as u32;
            let dh = (cfg.patch_size.1 * cfg.vae_stride.1) as u32;
            (width, height) = best_output_size(width, height, dw, dh, cfg.max_area);
        }
        let steps = req.steps.map(|s| s as usize).unwrap_or(cfg.sample_steps);
        let shift = req.scheduler_shift.unwrap_or(cfg.sample_shift);
        // Unset → UniPC (the reference default); `validate` has already rejected any unadvertised name.
        let kind = SolverKind::from_name(req.sampler.as_deref().unwrap_or("unipc"));
        let seed = req.seed.unwrap_or_else(default_seed);
        // The 5B is dense → a single guidance scale (config Single(5.0), overridable per request).
        let guidance = match (cfg.sample_guide_scale, req.guidance) {
            (_, Some(g)) => g,
            (GuideScale::Single(s), None) => s,
            (GuideScale::Dual { low, .. }, None) => low, // unreachable for the dense 5B
        };
        let cfg_disabled = guidance <= 1.0;
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, cfg.vae_stride);

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            let context = enc.encode(&tokenizer, &req.prompt)?;
            let context_null = if cfg_disabled {
                None
            } else {
                Some(enc.encode(&tokenizer, &neg_prompt)?)
            };
            match &context_null {
                Some(cn) => mlx_rs::transforms::eval([&context, cn])?,
                None => mlx_rs::transforms::eval([&context])?,
            }
            (context, context_null)
        };

        // Seeded init noise (f32) — shape matches the reference; exact RNG values differ across the
        // mlx-python/mlx-rs split (expected).
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (TI2V only): encode the conditioning image + build the mask-blend init ---
        // A `Reference` image → z48-VAE-encode to `z_img [z,1,h,w]`, build the first-frame mask
        // (`[z,T,h,w]`, 0 at frame 0) + per-token mask (`[1,L]`), and blend `(1−mask)·z_img +
        // mask·noise`. Without an image this is pure-noise T2V.
        // Channels-first `[z,1,h,w]` latent for one preprocessed TI2V image (z48-VAE-encode → reshape).
        let encode_kf = |vae: &Wan22Vae, image: &Image| -> Result<Array> {
            let img_thwc = preprocess_ti2v_image(image, width, height)?; // [1,1,H,W,3]
            let z = vae.encode(&img_thwc)?; // [1,1,h,w,z]
            Ok(z.reshape(&z.shape()[1..])?.transpose_axes(&[3, 0, 1, 2])?) // [z,1,h,w]
        };
        let (t_lat, h_lat, w_lat) = (lat[1] as usize, lat[2] as usize, lat[3] as usize);
        let keyframes = req.keyframes();
        let (latents_init, ti2v) = if !keyframes.is_empty() {
            // Wan-native first_last_frame / multi-keyframe (sc-3357): pin each Keyframe's latent frame
            // via the mask-blend (frame_idx is a latent index, negative-from-end → `-1` = last frame).
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = Wan22Vae::from_weights(&w)?;
            let mut frames: Vec<(Array, usize)> = Vec::with_capacity(keyframes.len());
            let mut indices: Vec<usize> = Vec::with_capacity(keyframes.len());
            for kf in &keyframes {
                let idx = if kf.frame_idx < 0 {
                    t_lat as i32 + kf.frame_idx
                } else {
                    kf.frame_idx
                };
                if idx < 0 || idx as usize >= t_lat {
                    return Err(Error::Msg(format!(
                        "wan2_2_ti2v_5b: keyframe latent frame index {} out of bounds for {t_lat} \
                         latent frames",
                        kf.frame_idx
                    )));
                }
                frames.push((encode_kf(&vae, kf.image)?, idx as usize));
                indices.push(idx as usize);
            }
            let z = build_ti2v_keyframe_z(&frames, cfg.vae_z_dim, t_lat, h_lat, w_lat)?;
            let (mask, mask_tokens) =
                build_ti2v_multi_mask(&indices, cfg.vae_z_dim, t_lat, h_lat, w_lat, cfg.patch_size);
            let latents = ti2v_blend_init(&z, &mask, &init_noise)?;
            mlx_rs::transforms::eval([&latents, &z])?;
            (latents, Some((z, mask, mask_tokens)))
        } else {
            match i2v_reference(req) {
                Some(image) => {
                    let z_img = {
                        let w = Weights::from_file(self.root.join("vae.safetensors"))?;
                        let vae = Wan22Vae::from_weights(&w)?;
                        encode_kf(&vae, image)?
                    };
                    let (mask, mask_tokens) =
                        build_ti2v_mask(cfg.vae_z_dim, t_lat, h_lat, w_lat, cfg.patch_size);
                    let latents = ti2v_blend_init(&z_img, &mask, &init_noise)?;
                    mlx_rs::transforms::eval([&latents, &z_img])?;
                    (latents, Some((z_img, mask, mask_tokens)))
                }
                None => (init_noise.clone(), None),
            }
        };

        // --- Stage 2: load the DiT, merge adapters + quantize, embed contexts, denoise (→ freed) ---
        let latents = {
            let mut w = Weights::from_file(self.root.join("model.safetensors"))?;
            // Merge LoRA/LoKr on the dense bf16 weights (no-op without adapters). Runs BEFORE
            // quantization (the fork order: a LoRA folds into the dense weight, then it is quantized;
            // load rejects LoRA on a pre-quantized snapshot).
            self.merge_adapters(&mut w)?;
            let mut dit = WanTransformer::from_weights(&w, cfg)?;
            if let Some(q) = self.quant {
                dit.quantize(q.bits(), None)?;
            }
            let ctx_cond = dit.embed_text(&context)?;
            let ctx_uncond = match &context_null {
                Some(cn) => Some(dit.embed_text(cn)?),
                None => None,
            };
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            match &ti2v {
                Some((z_img, mask, mask_tokens)) => denoise_ti2v(
                    &dit,
                    kind,
                    cfg.num_train_timesteps,
                    steps,
                    shift,
                    guidance,
                    &ctx_cond,
                    ctx_uncond.as_ref(),
                    &latents_init,
                    z_img,
                    mask,
                    mask_tokens,
                    &mut on_step,
                )?,
                None => denoise(
                    &dit,
                    kind,
                    cfg.num_train_timesteps,
                    steps,
                    shift,
                    guidance,
                    &ctx_cond,
                    ctx_uncond.as_ref(),
                    &latents_init,
                    &mut on_step,
                )?,
            }
        };

        // --- Stage 3: z48 vae22 decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        // Causal temporal decode: t_lat → 1 + (t_lat−1)·4 output frames (= gen_frames).
        let tiling = TilingConfig::auto(height as i32, width as i32, gen_frames as i32);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = Wan22Vae::from_weights(&w)?;
            decode_to_frames_22(&vae, &latents, tiling.as_ref())?
        };
        let mut images = frames_to_images(&frames_u8)?;
        // Discard the extra leading frames generated for `trim_first_frames`.
        if trim_out > 0 {
            images.drain(0..trim_out.min(images.len()));
        }

        let fps = req.fps.unwrap_or(cfg.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load }
}

// ===========================================================================================
// Wan2.2 T2V-A14B — dual-expert MoE text→video (the S1–S5 core, fully wired)
// ===========================================================================================

/// Public registry id for the dual-expert MoE T2V model: `mlx_gen::load("wan2_2_t2v_14b", spec)`.
pub const MODEL_ID_T2V_14B: &str = "wan2_2_t2v_14b";

/// Stable identity + advertised capabilities for the Wan2.2 T2V-A14B (dual-expert MoE text→video).
pub fn descriptor_t2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_T2V_14B,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            // CFG with the per-expert (low, high) guidance pair + the Chinese anti-artifact negative
            // prompt. Pure text→video: no image conditioning.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: Vec::new(),
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            // Cross-attention text K/V is cached across denoise steps (per expert).
            supports_kv_cache: true,
            requires_sigma_shift: false,
        },
    }
}

/// The loaded Wan2.2 T2V-A14B. Holds the resolved config + the snapshot directory; the heavy
/// components (UMT5 TE, the two 14B experts, the z16 VAE) are **staged** inside
/// [`Wan14b::generate`] — loaded, used, then dropped in turn — to bound peak memory (mirrors
/// `generate_wan.py`, which never holds the T5 encoder and both 27 GB experts resident at once).
pub struct Wan14b {
    descriptor: ModelDescriptor,
    config: WanModelConfig,
    root: PathBuf,
    /// LoRA adapters merged onto the experts at generate time (sc-2683). Empty for a plain load;
    /// `moe_expert`-tagged specs route to the high/low expert (shared = both).
    adapters: Vec<AdapterSpec>,
    /// Optional Q4/Q8 quantization for the transformer experts (sc-2682). `None` = dense bf16 (or a
    /// pre-quantized snapshot, which `from_weights` builds packed from its `config.json` manifest —
    /// see [`resolve_load_time_quant`]). When `Some`, [`Wan14b::generate`] quantizes **each** expert
    /// independently after load (transformer-only: attn `q/k/v/o` + `ffn.fc1/fc2`; T5 + VAE stay f32).
    quant: Option<Quant>,
}

impl Wan14b {
    /// The resolved model config.
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }

    /// Merge the load-time LoRA adapters onto the two expert weight maps in place (sc-2683), before
    /// the [`WanTransformer`]s are built. No-op when no adapters were supplied (the no-adapter path
    /// is byte-identical). Shared specs merge onto both experts, `moe_expert`-tagged specs onto their
    /// own (the reference `(loras)+(loras_high/low)` split). Errors if a non-empty adapter set matched
    /// no module across *either* expert (a format/prefix misconfiguration); per-key skips (a target
    /// absent from this checkpoint) are surfaced as a warning, not fatal, mirroring the reference.
    fn merge_adapters(&self, low_w: &mut Weights, high_w: &mut Weights) -> Result<()> {
        if self.adapters.is_empty() {
            return Ok(());
        }
        // A LoRA delta folds into the *dense* bf16 weight (then that may be quantized at load). On a
        // pre-quantized snapshot the experts ship packed (u32 codes + scales), so there is no dense
        // weight to merge into — the reference's dequantize-then-merge path (sc-2682's `loading.py`
        // LoRA branch) is a follow-on, not wired. Surface it rather than corrupt the packed weights.
        if self.config.quantization.is_some() {
            return Err(Error::Msg(format!(
                "{}: LoRA adapters on a pre-quantized snapshot need dequantize-then-merge (the \
                 reference loading.py path), not yet wired — load a dense bf16 snapshot (LoRA merges, \
                 then `spec.quantize` quantizes the merged weights), or drop the adapters",
                self.descriptor.id
            )));
        }
        let low = merge_wan_adapters(low_w, &self.adapters, MoeExpert::Low)?;
        let high = merge_wan_adapters(high_w, &self.adapters, MoeExpert::High)?;
        if low.applied + high.applied == 0 {
            return Err(Error::Msg(format!(
                "{}: {} LoRA file(s) matched no module across either expert — check the format \
                 (expected PEFT `lora_A/B` or kohya `lora_down/up`, `diffusion_model.`-prefixed Wan \
                 module names)",
                self.descriptor.id,
                self.adapters.len()
            )));
        }
        let mut skipped = low.skipped;
        skipped.extend(high.skipped);
        skipped.sort();
        skipped.dedup();
        if !skipped.is_empty() {
            eprintln!(
                "{}: {} LoRA target(s) not present in this checkpoint, skipped: {skipped:?}",
                self.descriptor.id,
                skipped.len()
            );
        }
        Ok(())
    }
}

/// Resolve the **load-time** quantization to apply in [`Wan14b::generate`], reconciling the requested
/// `spec.quantize` against a pre-quantized snapshot's `config.json` manifest (`cfg.quantization`).
///
/// A *pre-quantized* snapshot (manifest present) ships packed weights on disk → [`WanTransformer::
/// from_weights`] builds the experts quantized directly (the `loading.py` consume path), so **no**
/// load-time re-quantization is applied (returns `None`). A *dense bf16* snapshot honors
/// `spec.quantize` (quantized in-memory after load). A bits conflict is a hard error: the on-disk
/// manifest is authoritative, so we don't silently ignore (or re-quantize at) a different width.
/// (This is a deliberately *loud* "stored wins" — a pre-quantized snapshot at a different width is a
/// hard error here, not a silent downgrade.)
fn resolve_load_time_quant(
    id: &str,
    cfg: &WanModelConfig,
    requested: Option<Quant>,
) -> Result<Option<Quant>> {
    match (cfg.quantization, requested) {
        (Some(stored), Some(req)) if stored.bits != req.bits() => Err(Error::Msg(format!(
            "{id}: snapshot is pre-quantized {}-bit (config.json quantization block), but \
             spec.quantize requested {}-bit — the on-disk manifest is authoritative; drop the \
             precision override or convert a snapshot at the requested width",
            stored.bits,
            req.bits()
        ))),
        // Pre-quantized snapshot: `from_weights` builds it quantized; no load-time requant.
        (Some(_), _) => Ok(None),
        // Dense bf16 snapshot: quantize at load if requested.
        (None, req) => Ok(req),
    }
}

/// Load the Wan2.2 T2V-A14B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder.safetensors` +
/// `vae.safetensors` + `tokenizer.json` + `config.json`). LoRA adapters merge per-expert at generate
/// time (sc-2683); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot. LoKr is the
/// sibling sc-2393.
pub fn load_t2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_t2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.dual_model {
        return Err(Error::Msg(format!(
            "wan2_2_t2v_14b: config.json is not a dual-expert model (dual_model=false, \
             model_type={}); expected the converted Wan2.2 A14B MoE checkpoint",
            config.model_type
        )));
    }
    let quant = resolve_load_time_quant(MODEL_ID_T2V_14B, &config, spec.quantize)?;
    Ok(Box::new(Wan14b {
        descriptor: descriptor_t2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
    }))
}

impl Generator for Wan14b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        let id = self.descriptor.id;
        // Shared capability floor: size range (the advertised `min_size` = patch×vae_stride = 16 is
        // the sub-tile lower bound; `max_size` caps the long edge), count, guidance/negative/true_cfg,
        // sampler (`unipc`/`euler`/`dpmpp2m`), scheduler, and conditioning (none for T2V, `Reference`
        // for I2V).
        self.descriptor.capabilities.validate_request(id, req)?;
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "{id}: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        // I2V channel-concat requires a single reference image (the first conditioning frame), and
        // does not support `trim_first_frames` (the reference builds `y` from `num_frames`, so an
        // extended noise length would mismatch the conditioning's temporal dim).
        if self.config.is_i2v_concat() {
            if i2v_reference(req).is_none() {
                return Err(Error::Msg(format!(
                    "{id}: image-to-video requires a Reference conditioning image"
                )));
            }
            if req.trim_first_frames.unwrap_or(0) > 0 {
                return Err(Error::Msg(format!(
                    "{id}: trim_first_frames is not supported for I2V (the conditioning `y` is built \
                     from num_frames)"
                )));
            }
        }
        Ok(())
    }

    /// The full dual-expert MoE pipeline (port of `generate_wan.py`'s dual-model path) — serves both
    /// **T2V-A14B** and **I2V-A14B** (the struct's config selects). Resolves request knobs against the
    /// config defaults, then **stages** the phases to bound memory: (1) load UMT5, encode the prompt +
    /// negative prompt, drop the encoder; (1b, I2V only) load the z16 VAE encoder, build the
    /// channel-concat conditioning `y` from the reference image, drop it; (2) load both 14B experts,
    /// embed the contexts per expert, run the boundary-switched [`denoise_moe`] loop (with `y` for
    /// I2V), drop the experts; (3) load the z16 VAE, decode to RGB8 frames. CFG runs with the
    /// per-expert (low, high) guidance.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        // Reject anything outside the advertised surface before doing expensive work — in particular
        // an unknown `sampler`, which `solver_kind` would otherwise silently map to UniPC.
        self.validate(req)?;
        let cfg = &self.config;

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        // trim_first_frames: generate `trim` extra leading temporal chunks (each = vae_stride_t = 4
        // latent frames → 4 output frames after the non-causal T→4T decode) and discard them after
        // decode, so the first kept frame sees a full temporal receptive field (port of
        // generate_wan.py). gen_frames stays 1+4k since frames is and we add a multiple of 4.
        let trim = req.trim_first_frames.unwrap_or(0) as usize;
        let trim_out = trim * cfg.vae_stride.0; // discarded output frames = trim · 4
        let gen_frames = frames + trim * cfg.vae_stride.0;
        // validate() already rejected sub-tile + bad frame counts; round H/W down to the grid.
        let mut width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
        let mut height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
        // Enforce the model's max-area cap (I2V-14B / TI2V-5B: 704×1280) with an aspect-preserving,
        // grid-aligned fit (no-op for T2V, whose `max_area` is 0). Mirrors `generate_wan.py`.
        if cfg.max_area > 0 && (width as usize) * (height as usize) > cfg.max_area {
            let dw = (cfg.patch_size.2 * cfg.vae_stride.2) as u32;
            let dh = (cfg.patch_size.1 * cfg.vae_stride.1) as u32;
            (width, height) = best_output_size(width, height, dw, dh, cfg.max_area);
        }
        let steps = req.steps.map(|s| s as usize).unwrap_or(cfg.sample_steps);
        let shift = req.scheduler_shift.unwrap_or(cfg.sample_shift);
        // Unset → UniPC (the reference default); `validate` has already rejected any unadvertised name.
        let kind = SolverKind::from_name(req.sampler.as_deref().unwrap_or("unipc"));
        let seed = req.seed.unwrap_or_else(default_seed);
        // A scalar request `guidance` overrides both experts; otherwise use the config (low, high).
        let (low_gs, high_gs) = match (cfg.sample_guide_scale, req.guidance) {
            (_, Some(g)) => (g, g),
            (GuideScale::Dual { low, high }, None) => (low, high),
            (GuideScale::Single(s), None) => (s, s),
        };
        let neg_prompt = req
            .negative_prompt
            .clone()
            .unwrap_or_else(|| cfg.sample_neg_prompt.clone());

        // Init-noise latent geometry: [z_dim, t_lat, h_lat, w_lat] for the (possibly trim-extended)
        // generation length.
        let lat = latent_shape(gen_frames, height, width, cfg.vae_z_dim, cfg.vae_stride);

        // --- Stage 1: UMT5 text encode (loaded → used → freed) ---
        let tokenizer = load_tokenizer(self.root.join("tokenizer.json"), cfg.text_len)?;
        let (context, context_null) = {
            let w = Weights::from_file(self.root.join("t5_encoder.safetensors"))?;
            let enc = Umt5Encoder::from_weights(&w, cfg)?;
            let context = enc.encode(&tokenizer, &req.prompt)?;
            let context_null = enc.encode(&tokenizer, &neg_prompt)?;
            mlx_rs::transforms::eval([&context, &context_null])?;
            (context, context_null)
        };

        // Seeded init noise (f32, no batch dim) — matches the reference's `mx.random.normal(shape)`
        // shape; exact seeded-RNG values differ across the mlx-python/mlx-rs split (expected). I2V
        // (like the reference) starts from pure noise — the image enters via the `y` channel-concat.
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 1b (I2V only): build the channel-concat conditioning `y` (→ VAE encoder freed) ---
        // First frame = the reference image, the rest zero, VAE-encoded under a temporal mask →
        // `[20, T_lat, h_lat, w_lat]` (f32), concatenated onto each forward's noise latent in
        // `denoise_moe`. `frames` (not `gen_frames`) — validate() rejected `trim` for I2V.
        let y = if cfg.is_i2v_concat() {
            let image = i2v_reference(req).ok_or_else(|| {
                Error::Msg(format!(
                    "{}: image-to-video requires a Reference conditioning image",
                    self.descriptor.id
                ))
            })?;
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            let y = build_i2v_y(&vae, image, frames, height, width, cfg.vae_stride)?;
            mlx_rs::transforms::eval([&y])?;
            Some(y)
        } else {
            None
        };

        // --- Stage 2: load both experts, embed per-expert, dual-expert MoE denoise (→ freed) ---
        let latents = {
            let mut low_w = Weights::from_file(self.root.join("low_noise_model.safetensors"))?;
            let mut high_w = Weights::from_file(self.root.join("high_noise_model.safetensors"))?;
            // Merge LoRA adapters per expert (sc-2683) on the dense bf16 weights — no-op without
            // adapters. This runs BEFORE quantization (the fork order: a LoRA folds into the dense
            // weight, then that is quantized; load rejects LoRA on an already-pre-quantized snapshot).
            self.merge_adapters(&mut low_w, &mut high_w)?;
            // Q4/Q8 (sc-2682), two routes, both transformer-only (attn q/k/v/o + ffn.fc1/fc2; T5
            // above + VAE below stay f32 — the reference's quant scope):
            //   • pre-quantized snapshot (config.json `quantization` block) → `from_weights` already
            //     built the experts quantized from the on-disk packed weights (`self.quant` is None,
            //     resolved in load), so they load at the reduced ~Q4/Q8 size — the low-peak path;
            //   • dense bf16 snapshot + `spec.quantize` → quantize each expert in-memory after the
            //     (optional) LoRA merge.
            // Either way both experts are quantized independently.
            let mut low_dit = WanTransformer::from_weights(&low_w, cfg)?;
            let mut high_dit = WanTransformer::from_weights(&high_w, cfg)?;
            if let Some(q) = self.quant {
                low_dit.quantize(q.bits(), None)?;
                high_dit.quantize(q.bits(), None)?;
            }

            // Each expert has its own text_embedding weights, so contexts are embedded per expert.
            let low = Expert {
                transformer: &low_dit,
                ctx_cond: low_dit.embed_text(&context)?,
                ctx_uncond: Some(low_dit.embed_text(&context_null)?),
                guidance: low_gs,
            };
            let high = Expert {
                transformer: &high_dit,
                ctx_cond: high_dit.embed_text(&context)?,
                ctx_uncond: Some(high_dit.embed_text(&context_null)?),
                guidance: high_gs,
            };
            let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;
            let total = steps as u32;
            let mut on_step = |i: usize| {
                on_progress(Progress::Step {
                    current: i as u32,
                    total,
                })
            };
            denoise_moe(
                &low,
                &high,
                boundary_timestep,
                kind,
                cfg.num_train_timesteps,
                steps,
                shift,
                &init_noise,
                y.as_ref(),
                &mut on_step,
            )?
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        // Auto-select VAE decode tiling from the actual decoded output dims (t_lat·4 frames after the
        // non-causal decode); `None` for small outputs → single-pass. decode_to_frames re-checks
        // `needs_tiling`.
        let out_frames = lat[1] * cfg.vae_stride.0 as i32;
        let tiling = TilingConfig::auto(height as i32, width as i32, out_frames);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents, tiling.as_ref())?
        };
        let mut images = frames_to_images(&frames_u8)?;
        // Discard the extra leading frames generated for `trim_first_frames`.
        if trim_out > 0 {
            images.drain(0..trim_out.min(images.len()));
        }

        let fps = req.fps.unwrap_or(cfg.sample_fps);
        Ok(GenerationOutput::Video {
            frames: images,
            fps,
            audio: None,
        })
    }
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor: descriptor_t2v_14b, load: load_t2v_14b }
}

// ===========================================================================================
// Wan2.2 I2V-A14B — dual-expert MoE image→video (channel-concat conditioning, in_dim 36)
// ===========================================================================================

/// Public registry id for the channel-concat I2V model: `mlx_gen::load("wan2_2_i2v_14b", spec)`.
pub const MODEL_ID_I2V_14B: &str = "wan2_2_i2v_14b";

/// The single conditioning reference image for I2V (the first video frame), if present.
fn i2v_reference(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

/// Stable identity + advertised capabilities for the Wan2.2 I2V-A14B (dual-expert MoE image→video).
/// Identical to the T2V-A14B but advertises a single `Reference` conditioning image (the channel-
/// concat first frame) and the (3.5, 3.5) per-expert guidance.
pub fn descriptor_i2v_14b() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID_I2V_14B,
        family: "wan",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // A single image is channel-concatenated as the first-frame conditioning (in_dim 36).
            conditioning: vec![ConditioningKind::Reference],
            // LoRA + LoKr merge per-expert at generate time (sc-2683 / sc-2393, PEFT/kohya + LoKr,
            // MoE high/low); Q4/Q8 (sc-2682) loads via `spec.quantize` or a pre-quantized snapshot.
            supports_lora: true,
            supports_lokr: true,
            samplers: vec!["unipc", "euler", "dpmpp2m"],
            schedulers: Vec::new(),
            // H/W align to patch×vae_stride = 16 (z16 VAE, spatial stride 8); long edge cap 1280.
            min_size: 16,
            max_size: 1280,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: true,
            requires_sigma_shift: false,
        },
    }
}

/// Load the Wan2.2 I2V-A14B from a converted MLX snapshot directory (same layout as the T2V-A14B:
/// `low_noise_model` + `high_noise_model` + `t5_encoder` + `vae` (with encoder) + `tokenizer.json` +
/// `config.json`). Requires `model_type == "i2v"` (in_dim 36) and a dual-expert checkpoint. LoRA
/// adapters merge per-expert at generate time (sc-2683); Q4/Q8 (sc-2682) loads via `spec.quantize`
/// or a pre-quantized snapshot. LoKr is the sibling sc-2393.
pub fn load_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => return Err(Error::Msg(
            "wan2_2_i2v_14b: expected a model directory (converted MLX snapshot), not a single \
                 file"
                .into(),
        )),
    };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: precision override is not wired (the experts run bf16 GEMMs over an \
             f32 residual stream — the parity regime)"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(&root)?;
    if !config.is_i2v_concat() {
        return Err(Error::Msg(format!(
            "wan2_2_i2v_14b: config.json is not a channel-concat I2V model (model_type={}, \
             in_dim={}); expected the converted Wan2.2 I2V-A14B checkpoint (model_type=i2v, \
             in_dim=36)",
            config.model_type, config.in_dim
        )));
    }
    if !config.dual_model {
        return Err(Error::Msg(
            "wan2_2_i2v_14b: config.json is not a dual-expert model (dual_model=false); expected \
             the converted Wan2.2 I2V-A14B MoE checkpoint"
                .into(),
        ));
    }
    let quant = resolve_load_time_quant(MODEL_ID_I2V_14B, &config, spec.quantize)?;
    Ok(Box::new(Wan14b {
        descriptor: descriptor_i2v_14b(),
        config,
        root,
        adapters: spec.adapters.clone(),
        quant,
    }))
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor: descriptor_i2v_14b, load: load_i2v_14b }
}
