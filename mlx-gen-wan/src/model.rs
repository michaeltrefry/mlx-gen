//! `mlx-gen-wan` model entries: the Wan2.2 **TI2V-5B** (`wan2_2_ti2v_5b`, dense, z48 VAE — S0
//! scaffold, denoise pending in sc-2680) and the Wan2.2 **T2V-A14B** (`wan2_2_t2v_14b`, dual-expert
//! MoE, z16 VAE — fully wired here on the S1–S5 core), plus their registry self-registration.
//!
//! The 5B `load` resolves `config.json` and stubs `generate` (its z48 VAE + dense denoise are
//! sc-2680). The 14B [`Wan14b::generate`] runs the complete T2V pipeline: UMT5-XXL encode → per-step
//! dual-expert MoE denoise (boundary-switched high/low experts, [`denoise_moe`]) → z16 VAE decode →
//! RGB8 frames, **staging** each heavy component (T5, the two 27 GB experts, the VAE) in and out to
//! bound peak memory (mirrors `generate_wan.py`).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, Capabilities, ConditioningKind, Error, GenerationOutput, GenerationRequest,
    Generator, LoadSpec, Modality, ModelDescriptor, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::random;

use crate::config::{GuideScale, WanModelConfig};
use crate::pipeline::{
    align_dim, decode_to_frames, denoise_moe, frames_to_images, latent_shape, Expert,
};
use crate::scheduler::SolverKind;
use crate::text_encoder::{load_tokenizer, Umt5Encoder};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;

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
            // accepts a single image as the TI2V mask-blend conditioning reference.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference],
            // LoRA/LoKr (sc-2683 / sc-2393) and Q4/Q8 (sc-2682) are sibling slices.
            supports_lora: false,
            supports_lokr: false,
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

/// The loaded Wan model. S0 holds the resolved config; the network components (UMT5 TE, DiT, z48
/// VAE) attach across S1–S5.
pub struct Wan {
    descriptor: ModelDescriptor,
    #[allow(dead_code)] // consumed by the S1–S5 pipeline.
    config: WanModelConfig,
}

impl Wan {
    /// The resolved model config (exposed for the S1–S5 pipeline slices + tests).
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }
}

/// Load the model from a snapshot directory. Reads + resolves `config.json` (the config seam). The
/// 5B path runs f32 activations (quality + dodging the pmetal bf16 GEMM bug); quantization and
/// adapters are sibling slices, rejected here for now.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "wan2_2_ti2v_5b: expected a model directory (split-weight snapshot), not a single \
                 file"
                    .into(),
            )),
        };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: precision override is not wired (the dense path runs f32 activations)"
                .into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: Q4/Q8 quantization is a sibling slice (sc-2682), not yet wired".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "wan2_2_ti2v_5b: LoRA/LoKr adapters are sibling slices (sc-2683 / sc-2393), not yet \
             wired"
                .into(),
        ));
    }

    let config = WanModelConfig::from_model_dir(root)?;
    Ok(Box::new(Wan {
        descriptor: descriptor(),
        config,
    }))
}

impl Generator for Wan {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        // H/W align to patch×vae_stride (32 for the 5B); the pipeline rounds down, but reject
        // sub-tile sizes outright.
        let align = (self.config.patch_size.1 * self.config.vae_stride.1) as u32;
        if req.width < align || req.height < align {
            return Err(Error::Msg(format!(
                "wan2_2_ti2v_5b: width/height must be ≥ {align} (got {}x{})",
                req.width, req.height
            )));
        }
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

    fn generate(
        &self,
        _req: &GenerationRequest,
        _on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        Err(Error::Msg(
            "wan2_2_ti2v_5b: the T2V/TI2V denoise pipeline is not yet wired — S0 ships the \
             scaffold, config, 3 flow-match solvers, 3-axis RoPE, and 3-D patchify; the UMT5 TE / \
             z48 VAE / DiT / pipeline land in S1–S5 (sc-2678 / sc-2680)"
                .into(),
        ))
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
            // LoRA/LoKr (sc-2683 / sc-2393) and Q4/Q8 (sc-2682) are sibling slices.
            supports_lora: false,
            supports_lokr: false,
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
}

impl Wan14b {
    /// The resolved model config.
    pub fn config(&self) -> &WanModelConfig {
        &self.config
    }
}

/// Map a request `sampler` string to a [`SolverKind`] (default UniPC, the reference's default).
fn solver_kind(sampler: Option<&str>) -> SolverKind {
    match sampler {
        Some("euler") => SolverKind::Euler,
        Some("dpmpp2m") | Some("dpm++") => SolverKind::Dpmpp2m,
        _ => SolverKind::UniPC,
    }
}

/// Load the Wan2.2 T2V-A14B from a converted MLX snapshot directory (`convert_wan.py` output:
/// `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder.safetensors` +
/// `vae.safetensors` + `tokenizer.json` + `config.json`). Quantization + adapters are sibling slices.
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
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: Q4/Q8 quantization is a sibling slice (sc-2682), not yet wired".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "wan2_2_t2v_14b: LoRA/LoKr adapters are sibling slices (sc-2683 / sc-2393), not yet \
             wired"
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
    Ok(Box::new(Wan14b {
        descriptor: descriptor_t2v_14b(),
        config,
        root,
    }))
}

impl Generator for Wan14b {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        // H/W align to patch×vae_stride (16 for the z16 VAE); the pipeline rounds down, but reject
        // sub-tile sizes outright.
        let align = (self.config.patch_size.1 * self.config.vae_stride.1) as u32;
        if req.width < align || req.height < align {
            return Err(Error::Msg(format!(
                "wan2_2_t2v_14b: width/height must be ≥ {align} (got {}x{})",
                req.width, req.height
            )));
        }
        if let Some(frames) = req.frames {
            // num_frames must be 1 + 4·k (one VAE temporal chunk + 4× per chunk).
            if frames % 4 != 1 {
                return Err(Error::Msg(format!(
                    "wan2_2_t2v_14b: num_frames must be 1 + 4·k (got {frames})"
                )));
            }
        }
        Ok(())
    }

    /// The full T2V pipeline (port of `generate_wan.py`'s dual-model path). Resolves request knobs
    /// against the config defaults, then **stages** three phases to bound memory: (1) load UMT5,
    /// encode the prompt + negative prompt, drop the encoder; (2) load both 14B experts, embed the
    /// contexts per expert, run the boundary-switched [`denoise_moe`] loop, drop the experts; (3)
    /// load the z16 VAE, decode to RGB8 frames. CFG runs with the per-expert (low, high) guidance.
    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let cfg = &self.config;

        // --- Resolve request knobs against config defaults ---
        let frames = req.frames.map(|f| f as usize).unwrap_or(cfg.frame_num);
        // validate() already rejected sub-tile + bad frame counts; round H/W down to the grid.
        let width = align_dim(req.width, cfg.patch_size.2, cfg.vae_stride.2);
        let height = align_dim(req.height, cfg.patch_size.1, cfg.vae_stride.1);
        let steps = req.steps.map(|s| s as usize).unwrap_or(cfg.sample_steps);
        let shift = req.scheduler_shift.unwrap_or(cfg.sample_shift);
        let kind = solver_kind(req.sampler.as_deref());
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

        // Init-noise latent geometry: [z_dim, t_lat, h_lat, w_lat].
        let lat = latent_shape(frames, height, width, cfg.vae_z_dim, cfg.vae_stride);

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
        // shape; exact seeded-RNG values differ across the mlx-python/mlx-rs split (expected).
        let key = random::key(seed)?;
        let init_noise = random::normal::<f32>(&lat[..], None, None, Some(&key))?;

        // --- Stage 2: load both experts, embed per-expert, dual-expert MoE denoise (→ freed) ---
        let latents = {
            let low_w = Weights::from_file(self.root.join("low_noise_model.safetensors"))?;
            let high_w = Weights::from_file(self.root.join("high_noise_model.safetensors"))?;
            let low_dit = WanTransformer::from_weights(&low_w, cfg)?;
            let high_dit = WanTransformer::from_weights(&high_w, cfg)?;

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
                &mut on_step,
            )?
        };

        // --- Stage 3: z16 VAE decode → RGB8 frames ---
        on_progress(Progress::Decoding);
        let frames_u8 = {
            let w = Weights::from_file(self.root.join("vae.safetensors"))?;
            let vae = WanVae::from_weights(&w)?;
            decode_to_frames(&vae, &latents)?
        };
        let images = frames_to_images(&frames_u8)?;

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
