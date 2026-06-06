//! sc-3045 — LoRA/LoKr **training** on the SDXL U-Net, in pure Rust on mlx-rs. The SDXL realization
//! of the core [`Trainer`] contract (epic 3039), built on the same functional-autograd mechanism the
//! Z-Image trainer proved (sc-3042/3044) and the host-generic factor machinery hoisted to core
//! ([`mlx_gen::train::lora`], sc-3045). Parity target = the SceneWorks torch `SdxlLoraTrainer` /
//! `_SdxlLoraBackend`.
//!
//! **What is SDXL-specific here** (everything else is the shared core machinery):
//!   * **Noise / objective — discrete DDPM in the vendored sigma-space.** SDXL inference runs the
//!     vendored k-diffusion Euler-Ancestral sampler ([`EulerSampler`]): latents are stored
//!     *renormalized*, `scale_model_input` is the identity, and the per-step time `t` is the float
//!     sigma-table index in `[0, 1000]` that the U-Net's sinusoidal embedding consumes. Crucially the
//!     renormalized model input `(x0 + σ·noise)·rsqrt(σ²+1)` is **algebraically identical** to the
//!     diffusers DDPM `noisy = √(ᾱ)·x0 + √(1−ᾱ)·noise` (since `rsqrt(σ²+1) = √(ᾱ)`,
//!     `σ·rsqrt(σ²+1) = √(1−ᾱ)`), and the **epsilon** target is the unit `noise`. So training reuses
//!     the crate's own [`EulerSampler::add_noise_with`] at a sampled integer table-index `t` — making
//!     train/inference consistent **by construction** — and regresses the U-Net's `eps` toward
//!     `noise`. (SDXL-base is epsilon-prediction; the v-prediction the torch reference's
//!     `prediction_type` branch supports is never taken for SDXL-base, and the crate's eps-only
//!     sampler could not consume a v-pred adapter — so eps is the correct and only objective here.)
//!     `t` is sampled **uniform over the integer table indices `[1, 1000]`**, which maps 1:1 onto the
//!     diffusers `randint(0, 1000)` the torch trainer uses (the table is `concat([0], σ_1..σ_1000)`).
//!   * **`added_cond_kwargs`.** The U-Net forward takes the pooled `text_embeds` (CLIP-bigG pooled)
//!     and the 6-element `time_ids`. The crate's inference path hardcodes
//!     `time_ids = [512,512,0,0,512,512]` (the vendored `generate_latents` quirk — it ignores the
//!     real size); training feeds the **same** [`text_time_ids`] so the conditioning the LoRA learns
//!     under matches what inference applies it under. (This deliberately diverges from the torch
//!     trainer's real-resolution time_ids — that would mismatch this engine's inference.)
//!   * **Dual-CLIP conditioning.** `encoder_hidden_states = concat(CLIP-L.hidden[-2], bigG.hidden[-2])`
//!     and pooled `text_embeds = bigG.pooled`, via [`encode_conditioning`]. Single forward, no CFG
//!     (the torch ref encodes with `do_classifier_free_guidance=False`).
//!   * **f32 base.** The U-Net + both text encoders + VAE load at f32 for clean autograd (the
//!     inference path runs fp16; the trained f32 factors merge into the fp16 base at load, casts
//!     handled by the loader). The VAE encodes the f32 init image to the scaled latent `x0`.
//!   * **Adapter surface, matched to inference consumption.** LoRA targets the **complete** UNet
//!     attention surface (down/mid/up `to_q/k/v/to_out.0`) — what `LoraCoverage::Complete`
//!     (`model::load`'s default) merges, and what the torch PEFT suffix-match selects. LoKr targets
//!     the **vendored** surface (down/up attention only): the SDXL LoKr loader keeps `mid_block` out
//!     (sc-2640), so training mid_block LoKr would produce factors no inference path reads. LoRA
//!     saves PEFT keys under `base_model.model.unet.` (what `_SdxlLoraBackend` emits); LoKr saves the
//!     bare `<path>.lokr_*` keys; both reconstruct at **f32** (the SDXL merge dtype).

use std::path::Path;

use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainerRegistration, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::subtract;
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::DiffusionConfig;
use crate::model::MODEL_ID;
use crate::pipeline::{encode_conditioning, encode_init_latents, text_time_ids};
use crate::sampler::EulerSampler;
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::UNet2DConditionModel;
use crate::vae::Autoencoder;

/// SDXL reconstructs its LoKr delta at **f32** (the f32-everywhere merge path); training must match
/// so the adapter round-trips through the inference loader.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// PEFT save-key prefix for the LoRA adapter — what `peft.save_pretrained()` / the SceneWorks
/// `_SdxlLoraBackend` emit, and what the SDXL loader's PEFT key classifier
/// (`adapters::classify_key`) expects.
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// The default SDXL attention LoRA targets — the suffixes `to_q`/`to_k`/`to_v`/`to_out.0` the torch
/// trainer uses (`DEFAULT_LORA_TARGET_MODULES`, `training_adapters.py:72`), suffix-matched across the
/// UNet attention modules exactly as PEFT's `LoraConfig(target_modules=…)` does.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// LoRA/LoKr trainer for Stable Diffusion XL, implementing the core [`Trainer`] surface: a frozen
/// f32 base (U-Net + dual CLIP + VAE + tokenizer) that caches a captioned image dataset to
/// VAE-latents + dual-CLIP conditioning/pooled embeds, then runs the functional-autograd loop with
/// the sc-3043 runtime glue (LR schedule, gradient accumulation, checkpoint cadence, cancel,
/// progress bands), and writes an adapter that round-trips through the SDXL inference loader.
pub struct SdxlTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: ClipBpeTokenizer,
    te1: ClipTextEncoder,
    te2: ClipTextEncoder,
    vae: Autoencoder,
    unet: UNet2DConditionModel,
    /// The SDXL noise schedule (the same sigma table the inference Euler-Ancestral sampler uses);
    /// training reuses its [`EulerSampler::add_noise_with`] for the renormalized DDPM noising.
    sampler: EulerSampler,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from an SDXL snapshot directory (the diffusers multi-component tree:
/// `tokenizer/ text_encoder/ text_encoder_2/ unet/ vae/`). Loads the base at **f32** (training needs
/// the dense, high-precision base for clean autograd; inference runs fp16). Registered via
/// [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(
                "sdxl trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 text_encoder_2/ unet/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(SdxlTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: crate::loader::load_tokenizer(root)?,
        te1: crate::loader::load_text_encoder_1(root)?,
        te2: crate::loader::load_text_encoder_2(root)?,
        vae: crate::loader::load_vae(root)?,
        unet: crate::loader::load_unet(root)?,
        sampler: EulerSampler::new(&DiffusionConfig::sdxl_base(), true),
    }))
}

inventory::submit! {
    TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer }
}

impl SdxlTrainer {
    /// Caption → `(conditioning [1, N, 2048], pooled [1, 1280])`: tokenize (no negative — training is
    /// CFG-off), run both CLIP encoders, and assemble the SDXL dual-CLIP conditioning + pooled embed
    /// exactly as the inference [`encode_conditioning`] path.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)> {
        let tokens = self.tokenizer.tokenize_batch(caption, None)?;
        encode_conditioning(&self.te1, &self.te2, &tokens)
    }
}

impl Trainer for SdxlTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> Result<()> {
        if req.items.is_empty() {
            return Err("sdxl trainer: dataset is empty".into());
        }
        if req.config.rank == 0 {
            return Err("sdxl trainer: rank must be > 0".into());
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(format!(
                "sdxl trainer: optimizer '{}' is not available on MLX training (supported: \
                 adamw, adam, rose, prodigy)",
                req.config.optimizer
            )
            .into());
        }
        Ok(())
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // --- prepare → load → cache: VAE-latents + dual-CLIP (conditioning, pooled) into memory ---
        on_progress(TrainingProgress::LoadingModel); // base already resident from load_trainer
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Array, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_init_latents(&self.vae, &img, edge, edge)?; // scaled latent [1,h,w,4]
            let (cond, pooled) = self.encode_prompt(&item.caption)?;
            eval([&x0, &cond, &pooled])?;
            cache.push((x0, cond, pooled));
        }
        if cache.is_empty() {
            return Err("sdxl trainer: no usable dataset items (all cancelled?)".into());
        }

        // SDXL micro-conditioning `time_ids`, built once and shared (B=1). Matches the inference
        // path's hardcoded `[512,512,0,0,512,512]` so the LoRA trains under the conditioning it is
        // applied under.
        let time_ids = text_time_ids(1);

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(&self.unet, cfg);
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) =
                    build_lora_targets(&mut self.unet, &target_paths, cfg.rank as i32, cfg.seed)?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut self.unet,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.decompose_factor,
                    cfg.seed,
                )?;
                (TrainAdapter::Lokr { targets }, params)
            }
        };
        let alpha = cfg.alpha;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };
        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;

        let accum = cfg.gradient_accumulation.max(1);
        let (total_updates, warmup_updates) =
            schedule_updates(cfg.steps, accum, cfg.lr_warmup_steps);
        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();

        // --- train loop ---
        let mut accumulated: Option<LoraParams> = None;
        let mut update_idx: u32 = 0;
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            let (x0, cond, pooled) = &cache[((step - 1) as usize) % cache.len()];
            // Uniform integer DDPM timestep over the sigma-table indices [1, max_time].
            let t = sample_timestep(
                &self.sampler,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            )?;
            let noise = random::normal::<f32>(
                x0.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut self.unet,
                &self.sampler,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                cond,
                pooled,
                &time_ids,
                t,
                &noise,
                mae,
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let mult =
                    lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.set_lr_scaled(mult);
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    accum,
                )?;
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                let clipped: LoraParams = clipped
                    .into_iter()
                    .map(|(k, v)| (k, v.into_owned()))
                    .collect();
                opt.step(&mut params, &clipped)?;
                eval(params.values())?;
                update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                std::fs::create_dir_all(&req.output_dir)?;
                let ckpt = req.output_dir.join(checkpoint_filename(&stem, step));
                adapter.save(
                    &params,
                    alpha,
                    rank,
                    cfg.decompose_factor,
                    PEFT_PREFIX,
                    &ckpt,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save(
            &params,
            alpha,
            rank,
            cfg.decompose_factor,
            PEFT_PREFIX,
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted UNet paths by suffix-matching them against the routable Linear surface — the same match
/// PEFT's `LoraConfig(target_modules=…)` does over the UNet attention modules.
///
/// The surface is chosen to match each adapter kind's **inference consumption** (so nothing trains
/// that no inference path reads, and the adapter round-trips cleanly):
///   * **LoRA** → the **complete** surface ([`UNet2DConditionModel::lora_target_paths_complete`]),
///     which `LoraCoverage::Complete` (`model::load`'s default) merges — down / **mid** / up
///     attention. Matches the torch PEFT suffix-match (which hits mid_block too).
///   * **LoKr** → the **vendored** surface ([`UNet2DConditionModel::lora_target_paths`]), down / up
///     attention only: the SDXL LoKr loader keeps `mid_block` out (sc-2640), so a mid_block LoKr
///     factor would be skipped at load. Training to the vendored surface keeps train/inference in
///     lock-step. (Extending the LoKr inference surface to mid_block is a separate engine change.)
fn resolve_target_paths(unet: &UNet2DConditionModel, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        DEFAULT_TARGET_SUFFIXES
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    let surface = match cfg.network_type {
        NetworkType::Lora => unet.lora_target_paths_complete(),
        NetworkType::Lokr => unet.lora_target_paths(),
    };
    surface
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// One forward+backward over the trainable adapter factors: build the renormalized DDPM input at
/// integer table-index `t`, inject `params` (LoRA or LoKr), run the U-Net, regress the predicted
/// `eps` toward the unit `noise`, return `(loss, grads)`.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    unet: &mut UNet2DConditionModel,
    sampler: &EulerSampler,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Array,
    pooled: &Array,
    time_ids: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
) -> Result<(f32, LoraParams)> {
    // Renormalized model input = `(x0 + σ(t)·noise)·rsqrt(σ(t)²+1)` — algebraically the diffusers
    // DDPM `noisy`; the epsilon target is the unit `noise`. Reusing the sampler's own `add_noise_with`
    // makes the training input bit-consistent with the inference convention.
    let noisy = sampler.add_noise_with(x0, noise, t)?;
    let target = noise.clone();
    let (cond, pooled, time_ids) = (cond.clone(), pooled.clone(), time_ids.clone());
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        adapter.install(unet, &p, alpha, rank, LOKR_DTYPE)?;
        let eps = unet
            .forward(&noisy, t, &cond, &pooled, &time_ids)
            .map_err(|e| Exception::custom(e.to_string()))?;
        let diff = subtract(&eps, &target)?;
        // MSE / MAE — `mean(None)` reduces to a 0-d scalar (grad requires a scalar cotangent).
        let loss = if mae {
            diff.abs()?.mean(None)?
        } else {
            diff.square()?.mean(None)?
        };
        Ok(vec![loss])
    };
    let mut vg = keyed_value_and_grad(loss_fn);
    let (val, grads) = vg(params.clone(), 0)?;
    Ok((val[0].item::<f32>(), grads))
}

/// Sample a **uniform integer** DDPM timestep over the sigma-table indices `[1, max_time]` (the
/// vendored table is `concat([0], σ_1..σ_1000)`, so index `t` maps to diffusers `ᾱ[t-1]` — i.e. a
/// uniform draw here equals the torch trainer's `randint(0, num_train_timesteps)`). Deterministic in
/// `seed`. At an integer `t` the sampler's sigma interpolation is exact (`σ = σ_t`).
fn sample_timestep(sampler: &EulerSampler, seed: u64) -> Result<f32> {
    let k = random::key(seed)?;
    let max_t = sampler.max_time(); // 1000.0
    let u = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k))?.item::<f32>();
    // floor(1 + u·max_t) ∈ [1, max_t] (u ∈ [0,1)); clamp the u→1 edge defensively.
    let t = (1.0 + u * max_t).floor().clamp(1.0, max_t);
    Ok(t)
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`](mlx_gen::media::Image).
fn decode_image(path: &Path) -> Result<mlx_gen::media::Image> {
    let dynimg = image::open(path)
        .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(mlx_gen::media::Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}
