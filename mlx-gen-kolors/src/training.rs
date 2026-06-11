//! sc-4568 — LoRA/LoKr **training** on the Kolors U-Net, in pure Rust on mlx-rs. The Kolors
//! realization of the core [`Trainer`] contract (epic 3039), built on the same functional-autograd
//! mechanism the Z-Image / SDXL trainers proved (sc-3042/3044/3045) and the host-generic factor
//! machinery in core ([`mlx_gen::train::lora`]). Parity target = the SceneWorks torch Kolors LoRA
//! trainer (the legacy `KolorsDiffusersAdapter` training path, epic 1929) this replaces.
//!
//! Kolors **is an SDXL-base U-Net under a ChatGLM3-6B text encoder**, so this is the SDXL trainer
//! ([`mlx_gen_sdxl::training`]) with three Kolors deltas; everything else is the shared core
//! machinery (autograd loop, LR schedule, gradient accumulation, checkpoint cadence, cancel):
//!
//!   * **Text encoder — ChatGLM3-6B, not dual-CLIP.** Conditioning is the ChatGLM3 penultimate hidden
//!     state `context` `[1, 256, 4096]` and the last-token last-layer `pooled` `[1, 4096]` — exactly
//!     the inference [`Kolors::encode`](crate::Kolors::encode) path (tokenize with the left-padded
//!     `position_ids`, then `ChatGlmModel::encode_prompt`). Single forward, no CFG (training is
//!     CFG-off, like every diffusers LoRA script). The SDXL U-Net auto-detects the `encoder_hid_proj`
//!     (4096→2048) and the 5632-wide add-embedding from the Kolors checkpoint (sc-3093), so its
//!     `forward` consumes the ChatGLM `(context, pooled)` directly.
//!   * **Micro-conditioning `time_ids` = `(H, W, 0, 0, H, W)`.** Kolors inference feeds the real
//!     resolution ([`crate::model::kolors_time_ids`], the diffusers `_get_add_time_ids`), unlike the
//!     SDXL engine which hardcodes `[512,512,0,0,512,512]`. Training feeds the **same** real-resolution
//!     ids at the bucketed training edge so the LoRA learns under the conditioning inference applies it
//!     under.
//!   * **Noise / objective — discrete DDPM over the Kolors `scaled_linear` schedule with
//!     `num_train_timesteps = 1100`.** The diffusers Kolors LoRA script noises with a `DDPMScheduler`:
//!     `noisy = √ᾱ_t·x0 + √(1−ᾱ_t)·noise` at a uniform integer `t ∈ [0, 1100)`, regressing the U-Net's
//!     **epsilon** toward `noise` (SDXL-base lineage; Kolors is epsilon-prediction — its inference
//!     [`KolorsEulerSampler`](crate::sampler) is epsilon Euler). This is train/inference-consistent
//!     **by construction**: the Kolors inference sampler's per-train-step sigma is
//!     `σ_t = √((1−ᾱ_t)/ᾱ_t)`, and the renormalized k-diffusion input `(x0+σ_t·noise)·rsqrt(σ_t²+1)`
//!     is algebraically identical to the DDPM `noisy` (`rsqrt(σ²+1)=√ᾱ`, `σ·rsqrt(σ²+1)=√(1−ᾱ)`), with
//!     the U-Net consuming the integer `t` as its sinusoidal time exactly as inference consumes the
//!     leading timesteps off the **same** `√((1−ᾱ)/ᾱ)` table. Unlike the SDXL engine's vendored sigma
//!     table — which is `concat([0], σ_1..σ_1000)` and so trains/infers at table-index `t↔ᾱ[t−1]` (a
//!     deliberate +1 offset) — Kolors inference indexes `ᾱ[T]` directly, so training uses the **direct**
//!     `ᾱ_t` (no offset) to stay in lock-step.
//!   * **f32 base.** The U-Net + ChatGLM3 + VAE load at f32 for clean autograd (inference runs fp16);
//!     the trained f32 factors merge into the fp16 base at load (casts handled by the loader).
//!   * **Adapter surface + save keys, matched to inference consumption.** The Kolors U-Net is the SDXL
//!     `UNet2DConditionModel`, so the trained adapter round-trips through the SDXL adapter merge
//!     ([`mlx_gen_sdxl::apply_sdxl_adapters`]): LoRA targets the **complete** attention surface
//!     (down/mid/up `to_q/k/v/to_out.0`) under the PEFT prefix `base_model.model.unet.`; LoKr targets
//!     the **vendored** surface (down/up attention only — the SDXL LoKr loader keeps `mid_block` out,
//!     sc-2640) and reconstructs at **f32** (the SDXL/Kolors merge dtype). (Wiring this LoRA into the
//!     Kolors *inference* registry — which today rejects `spec.adapters` — is a separate follow-on, the
//!     sc-3874 note; the produced adapter already reloads through the SDXL inference path, validated by
//!     `tests/trainer_e2e.rs`.)

use std::path::Path;

use mlx_gen::sampler::AlphaSchedule;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    gen_core, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainerRegistration, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_sdxl::UNet2DConditionModel;
use mlx_gen_sdxl::{encode_init_latents, load_unet_kolors_dtype, load_vae, Autoencoder};

use crate::chatglm3::{ChatGlmConfig, ChatGlmModel};
use crate::model::kolors_time_ids;
use crate::registry::MODEL_ID;
use crate::sampler::NUM_TRAIN_TIMESTEPS;
use crate::tokenizer::KolorsTokenizer;

/// Kolors `scaled_linear` betas — `β₀ = 0.00085`, `β₁ = 0.014` (the [`KolorsEulerSampler`] config).
const BETA_START: f32 = 0.00085;
const BETA_END: f32 = 0.014;

/// Kolors reconstructs its LoKr delta at **f32** (the SDXL-family f32-everywhere merge path the Kolors
/// U-Net inherits); training must match so the adapter round-trips through the inference loader.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// PEFT save-key prefix for the LoRA adapter. The Kolors U-Net is a diffusers `UNet2DConditionModel`
/// (the SDXL U-Net), so this is the SDXL prefix `peft.save_pretrained()` / the SceneWorks Kolors
/// trainer emit, and what the SDXL loader's PEFT key classifier expects on reload.
const PEFT_PREFIX: &str = "base_model.model.unet.";

/// The default attention LoRA targets — the suffixes `to_q`/`to_k`/`to_v`/`to_out.0` the torch trainer
/// uses, suffix-matched across the U-Net attention modules exactly as PEFT's `LoraConfig` does.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// LoRA/LoKr trainer for Kolors, implementing the core [`Trainer`] surface: a frozen f32 base
/// (ChatGLM3-6B encoder + tokenizer + SDXL-family U-Net with the ChatGLM context projection + SDXL
/// VAE) that caches a captioned image dataset to VAE-latents + ChatGLM `(context, pooled)`, then runs
/// the functional-autograd loop and writes an adapter that round-trips through the SDXL inference
/// loader (the Kolors U-Net == SDXL U-Net).
pub struct KolorsTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: KolorsTokenizer,
    chatglm: ChatGlmModel,
    vae: Autoencoder,
    unet: UNet2DConditionModel,
    /// Discrete DDPM `alphas_cumprod` over the Kolors `scaled_linear` schedule
    /// (`num_train_timesteps = 1100`); training noises `x0` with `√ᾱ_t·x0 + √(1−ᾱ_t)·noise`.
    schedule: AlphaSchedule,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "kolors",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from a `Kwai-Kolors/Kolors-diffusers` snapshot directory (the multi-component
/// tree: `tokenizer/ text_encoder/ unet/ vae/`, with the materialized `tokenizer/tokenizer.json`).
/// Loads the base at **f32** (training needs the dense, high-precision base for clean autograd;
/// inference runs fp16). Registered via [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(
                "kolors trainer expects a Kolors-diffusers snapshot directory (tokenizer/ \
                 text_encoder/ unet/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let dtype = Dtype::Float32;
    let te_w = mlx_gen::weights::Weights::from_dir(root.join("text_encoder"))?;
    Ok(Box::new(KolorsTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: KolorsTokenizer::from_dir(root.join("tokenizer"))?,
        chatglm: ChatGlmModel::from_weights(&te_w, ChatGlmConfig::chatglm3_6b(), None, dtype)?,
        vae: load_vae(root)?, // SDXL VAE (sdxl-vae-fp16-fix), f32
        unet: load_unet_kolors_dtype(root, dtype)?,
        schedule: AlphaSchedule::scaled_linear(NUM_TRAIN_TIMESTEPS, BETA_START, BETA_END)?,
    }))
}

/// Registry adapter: the trainer registry's `load` slot is typed on [`gen_core::Result`] (epic
/// 3720); bridge the crate's rich-`Result` [`load_trainer`] into it.
fn load_trainer_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    load_trainer(spec).map_err(Into::into)
}

inventory::submit! {
    TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer_registered }
}

impl KolorsTrainer {
    /// Caption → `(context [1, 256, 4096], pooled [1, 4096])`: tokenize (left-padded, with the
    /// ChatGLM `position_ids`) and run the ChatGLM3 encoder exactly as the inference
    /// [`Kolors::encode`](crate::Kolors::encode) path.
    fn encode_prompt(&self, caption: &str) -> Result<(Array, Array)> {
        let t = self.tokenizer.encode(caption)?;
        self.chatglm
            .encode_prompt(&t.input_ids, &t.attention_mask, Some(&t.position_ids))
    }
}

impl Trainer for KolorsTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        if req.items.is_empty() {
            return Err("kolors trainer: dataset is empty".into());
        }
        if req.config.rank == 0 {
            return Err("kolors trainer: rank must be > 0".into());
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(format!(
                "kolors trainer: optimizer '{}' is not available on MLX training (supported: \
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
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl KolorsTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        self.validate(req)?;
        let cfg = &req.config;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // --- prepare → load → cache: VAE-latents + ChatGLM (context, pooled) into memory ---
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
            return Err("kolors trainer: no usable dataset items (all cancelled?)".into());
        }

        // Kolors micro-conditioning `time_ids = (H, W, 0, 0, H, W)`, built once and shared (B=1).
        // Matches the inference path's real-resolution ids so the LoRA trains under the conditioning
        // it is applied under.
        let time_ids = kolors_time_ids(1, edge as i32, edge as i32);

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
            // Uniform integer DDPM timestep over `[0, num_train_timesteps)`.
            let t = sample_timestep(
                &self.schedule,
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
                &self.schedule,
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
/// dotted U-Net paths by suffix-matching them against the routable Linear surface — the same match
/// PEFT's `LoraConfig(target_modules=…)` does over the U-Net attention modules.
///
/// The surface is chosen to match each adapter kind's **inference consumption** on the SDXL U-Net the
/// Kolors model reuses (so nothing trains that no inference path reads, and the adapter round-trips):
///   * **LoRA** → the **complete** surface ([`UNet2DConditionModel::lora_target_paths_complete`]),
///     which `LoraCoverage::Complete` (the SDXL `model::load` default) merges — down / **mid** / up
///     attention. Matches the torch PEFT suffix-match (which hits `mid_block` too).
///   * **LoKr** → the **vendored** surface ([`UNet2DConditionModel::lora_target_paths`]), down / up
///     attention only: the SDXL LoKr loader keeps `mid_block` out (sc-2640), so a `mid_block` LoKr
///     factor would be skipped at load. Training to the vendored surface keeps train/inference in
///     lock-step.
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

/// One forward+backward over the trainable adapter factors: build the DDPM noisy input at integer
/// timestep `t`, inject `params` (LoRA or LoKr), run the U-Net, regress the predicted `eps` toward the
/// unit `noise`, return `(loss, grads)`.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    unet: &mut UNet2DConditionModel,
    schedule: &AlphaSchedule,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cond: &Array,
    pooled: &Array,
    time_ids: &Array,
    t: usize,
    noise: &Array,
    mae: bool,
) -> Result<(f32, LoraParams)> {
    // DDPM noisy = `√ᾱ_t·x0 + √(1−ᾱ_t)·noise`; the epsilon target is the unit `noise`. The U-Net
    // consumes the integer `t` as its sinusoidal time — train/inference consistent with the Kolors
    // EulerDiscrete inference sampler (whose σ_t = √((1−ᾱ_t)/ᾱ_t) makes its renormalized input equal
    // this DDPM noisy).
    let noisy = add_ddpm_noise(schedule, x0, noise, t)?;
    let t_f = t as f32;
    let target = noise.clone();
    let (cond, pooled, time_ids) = (cond.clone(), pooled.clone(), time_ids.clone());
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        adapter.install(unet, &p, alpha, rank, LOKR_DTYPE)?;
        let eps = unet
            .forward(&noisy, t_f, &cond, &pooled, &time_ids)
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

/// Discrete DDPM `add_noise` at integer timestep `t`: `√ᾱ_t·x0 + √(1−ᾱ_t)·noise` (diffusers
/// `DDPMScheduler.add_noise`, the noising the torch Kolors LoRA script uses). The `√ᾱ_t` / `√(1−ᾱ_t)`
/// coefficients are host f32 off the MLX-built `alphas_cumprod`, matching the reference's
/// `alphas_cumprod[t]**0.5`.
fn add_ddpm_noise(schedule: &AlphaSchedule, x0: &Array, noise: &Array, t: usize) -> Result<Array> {
    use mlx_gen::array::scalar;
    let acp = schedule.alphas_cumprod[t];
    let sqrt_acp = acp.sqrt();
    let sqrt_one_minus = (1.0 - acp).sqrt();
    let x0 = x0.as_dtype(Dtype::Float32)?;
    let noise = noise.as_dtype(Dtype::Float32)?;
    Ok(add(
        &multiply(&x0, scalar(sqrt_acp))?,
        &multiply(&noise, scalar(sqrt_one_minus))?,
    )?)
}

/// Sample a **uniform integer** DDPM timestep over `[0, num_train_timesteps)` — diffusers'
/// `randint(0, num_train_timesteps)` the torch trainer uses. Deterministic in `seed`.
fn sample_timestep(schedule: &AlphaSchedule, seed: u64) -> Result<usize> {
    let n = schedule.alphas_cumprod.len();
    let k = random::key(seed)?;
    let u = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k))?.item::<f32>();
    // floor(u·n) ∈ [0, n-1] (u ∈ [0,1)); clamp the u→1 edge defensively.
    Ok(((u * n as f32) as usize).min(n - 1))
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
