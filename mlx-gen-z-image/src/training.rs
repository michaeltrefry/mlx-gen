//! LoRA/LoKr *training* on the Z-Image DiT, in pure Rust on mlx-rs (epic 3039).
//!
//! The production [`ZImageTurboTrainer`] (sc-3044) realizes the core [`Trainer`] contract on the
//! REAL 30-block Z-Image transformer. The mechanism (the sc-3042 spike that first proved it has been
//! retired — F-043):
//!
//!   * **Trainable LoRA injection** — the model crates do NOT use mlx-rs's `Module`/`ModuleParameters`
//!     system (hand-rolled `&self` forwards over raw `Array`s, `src/adapters.rs:6`), so training uses
//!     the *functional* autograd: the trainable factors live OUTSIDE the model in a [`LoraParams`],
//!     and each step they are re-injected into the target [`AdaptableLinear`]s as a single
//!     `Adapter::Lora` via [`AdaptableLinear::set_adapters`]. The injection mirrors the inference
//!     reload (`adapters::loader::install_lora_groups`) op-for-op, so the trained adapter round-trips
//!     through the normal inference path bit-for-bit. The host-generic factor machinery lives in core
//!     [`mlx_gen::train::lora`] (hoisted in sc-3045 so every family trainer shares it); this module
//!     keeps only the Z-Image-specific forward, flow-match noising, and dual-encoder caching.
//!   * **Autograd + optimizer** — `keyed_value_and_grad` over the factor map + `AdamW::update_single`
//!     per parameter + `clip_grad_norm` (proven in `tests/lora_train_probe.rs`).
//!   * **Flow-match velocity target** — the Z-Image `forward()` already negates its raw output
//!     (`transformer.rs:246`) and the denoise loop integrates it as `latents += dσ·v` with
//!     `timestep = 1-σ`, so the regression target for `forward()` is `noise - latents` (the *raw*
//!     diffusers output trains toward `latents - noise`; the negation flips the sign — see
//!     `SceneWorks training_adapters.py:485` `flow_matching_velocity_target`).
//!   * **safetensors out** — PEFT keys `{path}.lora_A.weight` `[r,in]`, `{path}.lora_B.weight`
//!     `[out,r]`, `{path}.alpha`, reloadable by `apply_z_image_adapters`.
//!
//! sc-3043 generalizes this into a reusable `Trainer` surface (dataset/VAE-cache/bucket, checkpoint,
//! LR schedule, the `lora_train` job); sc-3044 hardens it for Z-Image + adds LoKr.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core;
use mlx_gen::media::Image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
// Re-export the `LoraTarget` that `build_lora_targets` returns so the crate's public surface is
// unchanged (the host-generic factor machinery moved to `mlx_gen::train::lora` in sc-3045).
pub use mlx_gen::train::lora::LoraTarget;
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainerRegistration, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::model::MODEL_ID;
use crate::pipeline::encode_init_latents;
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// Z-Image reconstructs its LoKr delta at **bf16** (the bf16-residual inference path); training must
/// match so the adapter round-trips bit-for-bit.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma`:
/// `x_t = (1-σ)·x0 + σ·noise`, `target = noise - x0`, `timestep = 1-σ`.
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?; // velocity for the already-negated forward output
    Ok((x_t, target, 1.0 - sigma))
}

// ===========================================================================================
// sc-3044: the production `Trainer` impl — realizes the sc-3043 contract on Z-Image, end to end.
// ===========================================================================================

/// LoRA trainer for Z-Image-Turbo, implementing the core [`Trainer`] surface: a frozen base model
/// (transformer + VAE + text encoder + tokenizer) that caches a captioned image dataset to
/// VAE-latents/prompt-embeds, then runs the functional-autograd LoRA loop with the sc-3043
/// runtime glue (LR schedule, gradient accumulation, checkpoint cadence, cancel, progress bands),
/// and writes a PEFT adapter that round-trips through the inference loader.
pub struct ZImageTurboTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: TextEncoder,
    vae: Vae,
    transformer: ZImageTransformer,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "z_image",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from a snapshot directory (the diffusers multi-component tree). No
/// quantization — training needs the dense base. Registered via [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(
                "z_image_turbo trainer expects a snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file"
                    .into(),
            ))
        }
    };
    Ok(Box::new(ZImageTurboTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: crate::loader::load_tokenizer(root)?,
        text_encoder: crate::loader::load_text_encoder(root)?,
        vae: crate::loader::load_vae(root)?,
        transformer: crate::loader::load_transformer(root)?,
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

/// Recognized `timestep_type` values — the noise-schedule samplers [`sample_sigma`] branches on
/// (`linear`/`uniform`/`weighted`) plus the `sigmoid` default it falls back to. Any other string
/// would silently sample sigmoid (F-041).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values — the high/low-noise tilts [`sample_sigma`] branches on plus
/// the neutral default (`balanced`/`none`/`neutral`) it falls back to.
const TIMESTEP_BIASES: [&str; 9] = [
    "balanced",
    "none",
    "neutral",
    "high",
    "high_noise",
    "favor_high_noise",
    "low",
    "low_noise",
    "favor_low_noise",
];
/// Recognized `loss_type` values — `mae`/`l1` select MAE, `mse`/`l2` the MSE default; any other
/// string would silently train MSE (F-041).
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer (mirrors the inference-side `validate_request`). Rejects an
/// empty dataset, zero rank, **zero steps** (F-040 — a 0-step run would otherwise fall straight
/// through to the save and write a no-op `B = 0` identity adapter), an unsupported optimizer, and —
/// rather than letting a typo silently fall back to a default sampler/loss (F-041) — an unrecognized
/// `timestep_type` / `timestep_bias` / `loss_type`. The non-empty target-module resolution (also
/// F-041) is checked in [`Trainer::validate`], which has the loaded DiT to match suffixes against.
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("z_image_turbo trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("z_image_turbo trainer: rank must be > 0".into());
    }
    if req.config.steps == 0 {
        return Err("z_image_turbo trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "z_image_turbo trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&req.config.timestep_type).as_str()) {
        return Err(format!(
            "z_image_turbo trainer: timestep_type '{}' is not recognized (supported: {})",
            req.config.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&req.config.timestep_bias).as_str()) {
        return Err(format!(
            "z_image_turbo trainer: timestep_bias '{}' is not recognized (supported: {})",
            req.config.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&req.config.loss_type).as_str()) {
        return Err(format!(
            "z_image_turbo trainer: loss_type '{}' is not recognized (supported: {})",
            req.config.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for ZImageTurboTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_request(req)?;
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would resolve
        // to an empty target set — a full-length run that trains zero parameters yet "succeeds"
        // (F-041). Catch it here, where the loaded DiT is available to match suffixes against.
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "z_image_turbo trainer: lora_target_modules {:?} matched no adaptable module on the \
                 DiT",
                req.config.lora_target_modules
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

impl ZImageTurboTrainer {
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

        // sc-4887 — training compute dtype. bf16 halves the activation working set (and the resident
        // base) and is the ecosystem-standard mixed precision; the trainable factors / loss / grads /
        // optimizer stay f32 (master-weights). The cast is destructive (f32→bf16), so a trainer that
        // was already cast cannot honor a later f32 request — reload instead of silently training at
        // the wrong precision.
        let use_bf16 = cfg.train_dtype.trim().eq_ignore_ascii_case("bf16")
            || cfg.train_dtype.trim().eq_ignore_ascii_case("bfloat16");
        let compute_dtype = if use_bf16 {
            Dtype::Bfloat16
        } else {
            Dtype::Float32
        };
        if !use_bf16 && self.transformer.compute_dtype() == Some(Dtype::Bfloat16) {
            return Err(
                "z_image_turbo trainer: this trainer instance was already cast to bf16 by a \
                 previous run; reload the trainer for f32 training"
                    .into(),
            );
        }

        // sc-4874 — fail-fast pre-flight memory guard. The dense (non-block-checkpointed) first step
        // materializes the whole forward graph in one MLX `eval`; at high resolution that working set
        // can exceed unified memory and the OS hard-kills the worker with an UNCATCHABLE SIGKILL (no
        // in-process error, the run just appears to hang at the last cached latent). We cannot catch
        // that kill, so we predict it and refuse up front with an actionable, catchable error —
        // BEFORE the (~minutes-long) latent caching — when gradient checkpointing is not enabled.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        if !will_checkpoint {
            preflight_memory_guard(edge, use_bf16)?;
        }

        if use_bf16 {
            self.transformer.cast_weights(Dtype::Bfloat16)?;
        }

        // --- prepare → load → cache: VAE-latents + prompt-embeds into memory before the loop ---
        on_progress(TrainingProgress::LoadingModel); // base model is already resident from load_trainer
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_init_latents(&self.vae, &img, edge, edge)?; // clean latent [16,1,h,w]
            let cap = crate::pipeline::encode_prompt(
                &self.tokenizer,
                &self.text_encoder,
                &item.caption,
                "z_image_turbo trainer",
            )?;
            eval([&x0, &cap])?;
            cache.push((x0, cap));
        }
        if cache.is_empty() {
            return Err("z_image_turbo trainer: no usable dataset items (all cancelled?)".into());
        }

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
        let target_paths = resolve_target_paths(&self.transformer, cfg);
        let rank = cfg.rank as f32;
        let (adapter, mut params) = match cfg.network_type {
            NetworkType::Lora => {
                let (targets, params) = build_lora_targets(
                    &mut self.transformer,
                    &target_paths,
                    cfg.rank as i32,
                    cfg.seed,
                )?;
                (TrainAdapter::Lora { targets }, params)
            }
            NetworkType::Lokr => {
                let (targets, params) = build_lokr_targets(
                    &mut self.transformer,
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

        // sc-4874 — gradient checkpointing. Collect, per MAIN block, the adapter-routable LOCAL paths
        // trained on it (e.g. `"attention.to_q"`); the long unified main stack is where the
        // first-step activation memory concentrates, so that is what we checkpoint. Refiner/embedder
        // targets are not collected here — they train through ordinary autograd in the
        // (non-checkpointed) pre-main forward.
        let n_layers = self.transformer.cfg.n_layers;
        let mut main_block_local_targets: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("layers.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        main_block_local_targets[i].push(local.to_string());
                    }
                }
            }
        }
        // Gradient checkpointing is an OPT-IN OPTION (the SceneWorks "Gradient Checkpointing"
        // toggle), never auto-forced — a run that would OOM is caught instead by the fail-fast
        // pre-flight guard below, which surfaces a catchable error and *recommends* this flag rather
        // than silently changing the user's training dynamics. Only the LoRA path is checkpointed
        // today — LoKr (a distinct Kronecker reconstruction) falls back to the dense path (follow-up).
        let is_lora = matches!(adapter, TrainAdapter::Lora { .. });
        let use_checkpoint = is_lora && cfg.gradient_checkpointing;
        let checkpoint_main: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&main_block_local_targets)
        } else {
            None
        };
        // sc-4886 — attention-segment checkpointing is ALWAYS on in training (LoRA and LoKr): it is
        // numerically identical to the retained backward (same decomposed attention math, recomputed)
        // and removes the dominant seq² per-block retention — the flash-backward surrogate every
        // torch trainer gets from its fused SDPA kernel. When whole-block checkpointing is on, the
        // main stack's flag goes OFF (the block recompute already covers attention; nesting would
        // recompute it twice for no memory win) — the refiners are never block-checkpointed, so
        // theirs stays on.
        self.transformer.set_sdpa_checkpoint(!use_checkpoint, true);
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
            let (x0, cap) = &cache[((step - 1) as usize) % cache.len()];
            let sigma = sample_sigma(
                &cfg.timestep_type,
                &cfg.timestep_bias,
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
                &mut self.transformer,
                &params,
                &adapter,
                alpha,
                rank,
                x0,
                cap,
                sigma,
                &noise,
                mae,
                checkpoint_main,
                compute_dtype,
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
                adapter.save(&params, alpha, rank, cfg.decompose_factor, "", &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // Cancelled before completing a single step (`steps == 0` is rejected upstream by
        // `validate`): the LoRA factors are still freshly initialized with `B = 0`, a mathematically
        // no-op adapter. Surface the cancellation as an error (as the inference denoise loop does)
        // rather than writing a valid-looking `.safetensors` and returning `Ok` — downstream tooling
        // would otherwise ship an identity LoRA as a trained artifact (F-040).
        if steps_run == 0 {
            return Err("z_image_turbo trainer: cancelled before any training step".into());
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
            "",
            &adapter_path,
        )?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the
/// unified token count `s` — an empirical fit to peaks measured on the 128 GB target.
///
/// The structure follows the sc-4874 root-cause decomposition: `weights + linear·s + quad·s²`,
/// where the constant is the resident base, the linear term is the per-token retained hidden-state
/// activations across the 30 blocks, and the quadratic term is seq² attention. Since sc-4886 the
/// training dense path always runs attention-segment checkpointing, which demotes the seq² term
/// from "one retained `[30-heads,s,s]` probability matrix per block" (3.92e-6·s² ≈ 66 GB at 1024,
/// the original sc-4874 explosion) to a single layer's backward transient (~1/30 of that).
/// sc-4887 bf16 halves the weights + activation terms.
///
/// Measured (`first_step_attn_ckpt_sweep`, 128 GB Mac17,6, rank 16 / 136 targets / batch 1):
/// f32 edge 512/768/1024 → 42.1/56.9/79.2 GB (exact 3-point fit); bf16 → 32.2/43.3 GB at 768/1024
/// (fit with the quadratic constrained to half the f32 coefficient; the 512 bf16 sample was
/// contaminated by not-yet-freed f32 buffers right after the cast). Assumes micro-batch 1 — true
/// of the trainer's loop (one cached item per step; `batch_size` shapes nothing here) — and f32
/// retained activations; refit if either changes.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        19.1 + 0.00522 * s + 1.55e-7 * s * s
    } else {
        30.8 + 0.01045 * s + 3.09e-7 * s * s
    }
}

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. `edge` is the bucketed training
/// edge; the unified token count is ≈ `(edge/16)²` (latent /8, patch 2) plus the small padded
/// caption block. The budget is MLX's own reported memory limit (≈ the device's recommended working
/// set), scaled by 0.85 to leave headroom for the worker/host — exceeding it is the regime where the
/// dense run was observed to die. Only consulted when gradient checkpointing is OFF.
fn preflight_memory_guard(edge: u32, bf16: bool) -> Result<()> {
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + 32.0; // + one padded caption block
    let projected = projected_dense_peak_gb(s, bf16);
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "z_image_turbo trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's ~{safe:.0} GB \
             safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS would hard-kill the \
             worker (SIGKILL) at the first step with no recoverable error (sc-4874). Enable Gradient \
             Checkpointing (recomputes block activations in the backward) or reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default `to_q`/`to_k`/`to_v`/`to_out.0`) to full
/// dotted paths by matching them against every adapter-routable module on the DiT — the same
/// suffix-match PEFT's `LoraConfig(target_modules=…)` does. This trains the attention projections in
/// the main `layers` AND the noise/context refiner stacks (matching the torch trainer), and handles
/// non-attention suffixes (FFN `w1`/`w2`/`w3`, `adaLN_modulation.0`) when configured — not just a
/// hardcoded `layers.{i}.attention.{suffix}`.
fn resolve_target_paths(transformer: &ZImageTransformer, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        ["to_q", "to_k", "to_v", "to_out.0"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    } else {
        cfg.lora_target_modules.clone()
    };
    AdaptableHost::adaptable_paths(transformer)
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Decode an image file (PNG/JPEG) into the core RGB8 [`Image`].
fn decode_image(path: &Path) -> Result<Image> {
    let dynimg = image::open(path)
        .map_err(|e| mlx_gen::Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Sample a normalised flow-match timestep (interpolation coefficient) `σ ∈ [1e-3, 1-1e-3]` — a
/// faithful port of the SceneWorks `sample_training_timestep`: `sigmoid(randn)` by default,
/// `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted; bias `high` → `√σ`,
/// `low` → `σ²`. Deterministic in `seed`.
fn sample_sigma(timestep_type: &str, timestep_bias: &str, seed: u64) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t = match ttype.as_str() {
        "linear" | "uniform" => {
            random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>()
        }
        "weighted" => {
            let k2 = random::key(seed ^ 0x9E37_79B9)?;
            let base = random::uniform::<_, f32>(0.0f32, 1.0f32, &[1], Some(&k1))?.item::<f32>();
            let center = sigmoid(random::normal::<f32>(&[1], None, None, Some(&k2))?.item::<f32>());
            (base + center) / 2.0
        }
        _ => sigmoid(random::normal::<f32>(&[1], None, None, Some(&k1))?.item::<f32>()),
    };
    let bias = timestep_bias
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '-'], "_");
    let t = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t * t,
        _ => t,
    };
    Ok(t.clamp(1e-3, 1.0 - 1e-3))
}

/// One forward+backward over the trainable adapter factors: inject `params` (LoRA or LoKr), run the
/// DiT, regress the (already-negated) `forward()` output toward the velocity `noise - x0`, return
/// `(loss, grads)`.
/// `checkpoint_main`, when `Some`, lists per-main-block LOCAL LoRA target paths and switches the
/// forward to the gradient-checkpointed path (sc-4874) — each main block recomputes its activations
/// in the backward instead of retaining them. `None` runs the dense (activation-retaining) forward.
/// `dtype` is the training compute dtype (sc-4887): for bf16 the latent / caption / RoPE inputs are
/// cast at entry (the weights were cast once in `train_impl`) and the LoRA factors are cast inside
/// the traced install, so the whole DiT graph runs bf16; the noising math, loss, and grads stay f32.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut ZImageTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    cap: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
    checkpoint_main: Option<&[Vec<String>]>,
    dtype: Dtype,
) -> Result<(f32, LoraParams)> {
    let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
    let x_t = x_t.as_dtype(dtype)?; // no-op in f32 mode
    let capf = cap.clone();
    let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets: refiners/embedders train through this on the (non-checkpointed)
        // pre-main forward; the main-block adapters installed here are simply replaced inside each
        // checkpoint segment by the explicit-input factors, so they cost nothing on the ckpt path.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let sh = x_t.shape();
        let prep = transformer
            .prepare((sh[0], sh[1], sh[2], sh[3]), &capf)
            .and_then(|prep| {
                if dtype == Dtype::Float32 {
                    Ok(prep)
                } else {
                    prep.cast_floats(dtype)
                }
            })
            .map_err(|e| Exception::custom(e.to_string()))?;
        let v = match checkpoint_main {
            Some(locals) => transformer
                .forward_with_main_checkpointed(&prep, &x_t, timestep, &p, locals, alpha)
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward_with(&prep, &x_t, timestep)
                .map_err(|e| Exception::custom(e.to_string()))?,
        };
        let diff = subtract(&v, &target)?;
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

// ===========================================================================================
// sc-4874 — first-step signal-9 repro/instrumentation (weight-gated, run as its own process).
//
// The production run dies with SIGKILL the instant the first real training step begins, at the
// default resolution 1024 (the e2e `trainer_e2e.rs` runs at 64 and never exercises this regime).
// This harness drives the exact inner step (`compute_loss_grads` + the step-1 grad `eval` the real
// loop forces at training.rs' optimizer step) directly, sweeping resolution with MLX peak-memory
// probes around it, to pinpoint the working set at the death point and test whether a memory
// ceiling converts the silent kill into a catchable error.
//
//   cargo test -p mlx-gen-z-image --release --lib first_step -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::media::Image;
    use mlx_gen::train::lora::build_lora_targets;
    use mlx_rs::memory::{
        clear_cache, get_active_memory, get_peak_memory, reset_peak_memory, set_memory_limit,
    };
    use std::path::PathBuf;

    fn snapshot() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var("HOME").ok()?;
        let snaps = PathBuf::from(home)
            .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
        std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir())
    }

    /// A solid-colour `edge`×`edge` RGB source image (the latent magnitude is irrelevant; the graph
    /// size — driven by resolution — is the variable under test).
    fn swatch(edge: u32) -> Image {
        let mut img = image::RgbImage::new(edge, edge);
        for px in img.pixels_mut() {
            *px = image::Rgb([180u8, 60, 90]);
        }
        Image {
            width: edge,
            height: edge,
            pixels: img.into_raw(),
        }
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Run a single first training step at `edge` and report peak GPU memory across the
    /// forward+backward. Forces the backward (grad eval) — the real step-1 kill point.
    /// `dtype` is the compute dtype handed to `compute_loss_grads` (the caller is responsible for
    /// having `cast_weights` the transformer to match); the SDPA-checkpoint flag is likewise set by
    /// the caller on `trainer.transformer` (sc-4886 arms A/B against the retained backward).
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        trainer: &mut ZImageTurboTrainer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        cap: &Array,
        edge: u32,
        checkpoint_main: Option<&[Vec<String>]>,
        dtype: Dtype,
        tag: &str,
    ) -> Result<(f32, f64, [i32; 4])> {
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge)?;
        eval([&x0])?;
        let shape = {
            let s = x0.shape();
            [s[0], s[1], s[2], s[3]]
        };
        let noise = random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1)?))?;
        eval([&noise])?;

        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            &mut trainer.transformer,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            cap,
            0.5,
            &noise,
            false,
            checkpoint_main,
            dtype,
        )?;
        // `compute_loss_grads` only forces the loss (forward). The real trainer forces the backward
        // at the step-1 optimizer `eval`; do the same here so the peak reflects the true working set.
        eval(grads.values())?;
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "  [edge {edge:>4} {tag}] latent {shape:?}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            gb(before),
            gb(peak)
        );
        Ok((loss, gb(peak), shape))
    }

    /// Per-main-block LOCAL LoRA target paths (mirrors `train_impl`), for driving the checkpointed
    /// path from the harness.
    fn main_block_local_targets(trainer: &ZImageTurboTrainer) -> Vec<Vec<String>> {
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.transformer, &cfg);
        let n_layers = trainer.transformer.cfg.n_layers;
        let mut out: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("layers.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        out[i].push(local.to_string());
                    }
                }
            }
        }
        out
    }

    fn build_trainer_and_adapter() -> (ZImageTurboTrainer, TrainAdapter, LoraParams, Array) {
        let root = snapshot().expect("Z-Image-Turbo snapshot (HF cache or ZIMAGE_SNAPSHOT)");
        let mut trainer = ZImageTurboTrainer {
            descriptor: trainer_descriptor(),
            tokenizer: crate::loader::load_tokenizer(&root).unwrap(),
            text_encoder: crate::loader::load_text_encoder(&root).unwrap(),
            vae: crate::loader::load_vae(&root).unwrap(),
            transformer: crate::loader::load_transformer(&root).unwrap(),
        };
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(&trainer.transformer, &cfg);
        let (targets, params) =
            build_lora_targets(&mut trainer.transformer, &target_paths, 16, 7).unwrap();
        let cap = crate::pipeline::encode_prompt(
            &trainer.tokenizer,
            &trainer.text_encoder,
            "a solid colour swatch",
            "sc-4874 repro",
        )
        .unwrap();
        eval([&cap]).unwrap();
        eprintln!(
            "[sc-4874] loaded trainer; {} LoRA targets; cap {:?}",
            targets.len(),
            cap.shape()
        );
        (trainer, TrainAdapter::Lora { targets }, params, cap)
    }

    /// Attribute the first-step peak to the FORWARD eval vs the BACKWARD eval (sc-4874 root-cause:
    /// is the "blip" the forward materializing, or the autograd backward retaining/recomputing?).
    /// `compute_loss_grads` forces the forward (via `loss.item()`) and returns lazy grads; we then
    /// force the backward. Run at SAFE resolutions (512/768) so it characterizes the split without
    /// risking the OOM kill, and extrapolate.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_memory_attribution() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        for edge in [512u32, 768] {
            let img = center_crop_square(&swatch(edge));
            let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
            let noise =
                random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap()))
                    .unwrap();
            eval([&x0, &noise]).unwrap();

            clear_cache();
            reset_peak_memory();
            let build_active = get_active_memory();
            let (_loss, grads) = compute_loss_grads(
                &mut trainer.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                Dtype::Float32,
            )
            .unwrap();
            // `loss.item()` inside `compute_loss_grads` already forced the FORWARD graph.
            let fwd_peak = get_peak_memory();
            eval(grads.values()).unwrap(); // force the BACKWARD
            let full_peak = get_peak_memory();
            eprintln!(
                "[sc-4874] edge {edge}: lazy-build active {:.2} GB | forward-eval peak {:.2} GB | +backward peak {:.2} GB (backward added {:.2} GB)",
                gb(build_active),
                gb(fwd_peak),
                gb(full_peak),
                gb(full_peak.saturating_sub(fwd_peak)),
            );
        }
    }

    /// Sweep resolution from tiny → production, printing the peak-memory curve. If the process dies
    /// (SIGKILL) at some edge, the prints from the survived edges are already flushed (eprintln +
    /// per-step completion), so the curve shows exactly where it falls over.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process (may SIGKILL)"]
    fn first_step_memory_sweep() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        eprintln!("[sc-4874] sweeping first-step peak memory by resolution:");
        for edge in [64u32, 256, 512, 768, 1024] {
            eprintln!("[sc-4874] === edge {edge} ===");
            match one_step(
                &mut trainer,
                &adapter,
                &params,
                &cap,
                edge,
                None,
                Dtype::Float32,
                "dense",
            ) {
                Ok((_, peak, shape)) => {
                    eprintln!("[sc-4874] edge {edge} SURVIVED  latent {shape:?}  peak {peak:.2} GB")
                }
                Err(e) => eprintln!("[sc-4874] edge {edge} returned CATCHABLE error: {e}"),
            }
        }
        eprintln!("[sc-4874] sweep complete (reached the end without SIGKILL)");
    }

    /// Production resolution (1024) with a low MLX memory ceiling: tests whether forcing MLX to
    /// stay under a limit converts the silent SIGKILL into a catchable Rust error (story step 2).
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_1024_with_memory_limit() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let limit_gb = std::env::var("SC4874_LIMIT_GB")
            .ok()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(24.0);
        let prev = set_memory_limit((limit_gb * 1024.0 * 1024.0 * 1024.0) as usize);
        eprintln!(
            "[sc-4874] set memory limit to {limit_gb:.1} GB (prev {:.2} GB); running edge 1024…",
            gb(prev)
        );
        match one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            None,
            Dtype::Float32,
            "dense",
        ) {
            Ok((loss, peak, shape)) => eprintln!(
                "[sc-4874] edge 1024 SURVIVED under {limit_gb:.1} GB limit  latent {shape:?}  loss {loss:.5}  peak {peak:.2} GB"
            ),
            Err(e) => eprintln!("[sc-4874] edge 1024 returned CATCHABLE error under limit: {e}"),
        }
    }

    /// The fix: at production resolution 1024, gradient checkpointing must drop the first-step peak
    /// well under the dense path's ~135 GB (which exceeds the 128 GB unified memory). Runs the dense
    /// step first (baseline), then the checkpointed step, and asserts a substantial reduction.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_1024_checkpointed_vs_dense() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let locals = main_block_local_targets(&trainer);
        let n_main: usize = locals.iter().map(|v| v.len()).sum();
        eprintln!("[sc-4874] checkpointing {n_main} LoRA targets across the main stack");

        let (_, dense_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            None,
            Dtype::Float32,
            "dense",
        )
        .expect("dense step");
        let (_, ckpt_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            Some(&locals),
            Dtype::Float32,
            "blk-ckpt",
        )
        .expect("checkpointed step");
        eprintln!(
            "[sc-4874] edge 1024  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_peak / dense_peak)
        );
        assert!(
            ckpt_peak < dense_peak,
            "checkpointing must reduce the first-step peak: dense {dense_peak:.2} GB vs ckpt {ckpt_peak:.2} GB"
        );
        // The whole point is fitting under 128 GB with production headroom — expect a large drop.
        assert!(
            ckpt_peak < 128.0,
            "checkpointed peak must fit unified memory: {ckpt_peak:.2} GB"
        );
    }

    /// Gradient checkpointing must not change the math: the checkpointed forward+grads must match the
    /// dense path within fp tolerance (it reuses the same install + block forward, recompute-only).
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let locals = main_block_local_targets(&trainer);
        let edge = 256u32; // small enough that the dense path is cheap; math is resolution-agnostic
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        eval([&x0, &noise]).unwrap();

        let grads_of = |t: &mut ZImageTurboTrainer, ck: Option<&[Vec<String>]>| -> LoraParams {
            let (_l, g) = compute_loss_grads(
                &mut t.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                ck,
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_dense = grads_of(&mut trainer, None);
        let g_ckpt = grads_of(&mut trainer, Some(&locals));

        let mut max_rel = 0f32;
        for (k, gd) in &g_dense {
            let gc = g_ckpt.get(k).expect("same keys");
            let num = gd.subtract(gc).unwrap().abs().unwrap().max(None).unwrap();
            let den = gd.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        eprintln!("[sc-4874] checkpointed-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense within tolerance: max rel {max_rel:.2e}"
        );
    }

    /// Max relative grad diff between two param maps (shared by the sc-4886/4887 parity tests).
    fn max_rel_diff(ga: &LoraParams, gb_: &LoraParams) -> f32 {
        let mut max_rel = 0f32;
        for (k, a) in ga {
            let b = gb_.get(k).expect("same keys");
            let num = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
            let den = a.abs().unwrap().max(None).unwrap().item::<f32>().max(1e-6);
            max_rel = max_rel.max(num.item::<f32>() / den);
        }
        max_rel
    }

    /// sc-4886 — the always-on attention-segment checkpointing must not change the math: grads with
    /// the SDPA checkpoint on must match the retained backward (flag off). Same decomposed
    /// attention, recomputed instead of retained → expect (near-)bit-identical.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        eval([&x0, &noise]).unwrap();

        let grads_of = |t: &mut ZImageTurboTrainer, on: bool| -> LoraParams {
            t.transformer.set_sdpa_checkpoint(on, on);
            let (_l, g) = compute_loss_grads(
                &mut t.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                Dtype::Float32,
            )
            .unwrap();
            eval(g.values()).unwrap();
            g
        };
        let g_retained = grads_of(&mut trainer, false);
        let g_ckpt = grads_of(&mut trainer, true);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-4886] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "attention-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// sc-4886/4887 — first-step peak sweep on the NEW training dense path (attention-segment
    /// checkpointing always on), f32 then bf16. These measured points are the basis of the
    /// `projected_dense_peak_gb` guard fit — refit the constants if this prints materially
    /// different numbers.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn first_step_attn_ckpt_sweep() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        trainer.transformer.set_sdpa_checkpoint(true, true);
        eprintln!("[sc-4886] attn-ckpt dense sweep, f32:");
        for edge in [512u32, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cap,
                edge,
                None,
                Dtype::Float32,
                "attn-ckpt f32",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4887] casting weights to bf16…");
        trainer
            .transformer
            .cast_weights(Dtype::Bfloat16)
            .expect("cast");
        clear_cache();
        eprintln!("[sc-4887] attn-ckpt dense sweep, bf16:");
        for edge in [512u32, 768, 1024] {
            let _ = one_step(
                &mut trainer,
                &adapter,
                &params,
                &cap,
                edge,
                None,
                Dtype::Bfloat16,
                "attn-ckpt bf16",
            )
            .map_err(|e| eprintln!("  edge {edge} CATCHABLE error: {e}"));
        }
        eprintln!("[sc-4887] block-ckpt + bf16 at 1024:");
        let locals = main_block_local_targets(&trainer);
        trainer.transformer.set_sdpa_checkpoint(false, true);
        let _ = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            1024,
            Some(&locals),
            Dtype::Bfloat16,
            "blk-ckpt bf16",
        )
        .map_err(|e| eprintln!("  blk-ckpt bf16 CATCHABLE error: {e}"));
    }

    /// sc-4887 — bf16 is mixed precision, NOT bit parity: assert the grads point the same way as
    /// the f32 path (per-param cosine) and the loss is finite. Runs f32 first (the weight cast is
    /// destructive), then casts the same trainer to bf16. Also asserts the bf16 working set is
    /// genuinely smaller — a silent f32 re-promotion anywhere in the forward would pass the cosine
    /// check while saving nothing, so the memory ratio IS the dtype assertion.
    #[test]
    #[ignore = "needs real Z-Image weights; run as its own process"]
    fn bf16_grads_direction_and_memory_vs_f32() {
        let (mut trainer, adapter, params, cap) = build_trainer_and_adapter();
        trainer.transformer.set_sdpa_checkpoint(true, true);

        // Memory A/B at 768 (big enough that activations dominate the peak).
        let (_, f32_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            768,
            None,
            Dtype::Float32,
            "attn-ckpt f32",
        )
        .expect("f32 step");

        // Grad reference at 256 in f32.
        let edge = 256u32;
        let img = center_crop_square(&swatch(edge));
        let x0 = encode_init_latents(&trainer.vae, &img, edge, edge).unwrap();
        let noise =
            random::normal::<f32>(x0.shape(), None, None, Some(&random::key(1).unwrap())).unwrap();
        eval([&x0, &noise]).unwrap();
        let grads_of = |t: &mut ZImageTurboTrainer, dt: Dtype| -> (f32, LoraParams) {
            let (l, g) = compute_loss_grads(
                &mut t.transformer,
                &params,
                &adapter,
                16.0,
                16.0,
                &x0,
                &cap,
                0.5,
                &noise,
                false,
                None,
                dt,
            )
            .unwrap();
            eval(g.values()).unwrap();
            (l, g)
        };
        let (f32_loss, g_f32) = grads_of(&mut trainer, Dtype::Float32);

        trainer
            .transformer
            .cast_weights(Dtype::Bfloat16)
            .expect("cast");
        clear_cache();
        let (bf16_loss, g_bf16) = grads_of(&mut trainer, Dtype::Bfloat16);
        assert!(
            bf16_loss.is_finite(),
            "bf16 loss must be finite: {bf16_loss}"
        );
        eprintln!("[sc-4887] loss f32 {f32_loss:.5} vs bf16 {bf16_loss:.5}");

        // Cosine between bf16 and f32 grads (both arrive f32 through the astype VJP). Gate on the
        // GLOBAL cosine (the concatenated gradient — what the optimizer step actually follows) and
        // the norm-weighted view; per-param minima are dominated by tiny-norm params whose direction
        // bf16 rounding legitimately scrambles while contributing nothing to the update. Print the
        // worst offenders with their norms so a real systematic bug (large-norm divergence) is
        // distinguishable from precision noise on negligible grads.
        let mut per: Vec<(String, f32, f32, f32)> = Vec::new(); // (key, cos, na, nb)
        let (mut gdot, mut gna2, mut gnb2) = (0f64, 0f64, 0f64);
        for (k, a) in &g_f32 {
            let b = g_bf16.get(k).expect("same keys");
            let dot = a.multiply(b).unwrap().sum(None).unwrap().item::<f32>();
            let na2 = a.square().unwrap().sum(None).unwrap().item::<f32>();
            let nb2 = b.square().unwrap().sum(None).unwrap().item::<f32>();
            gdot += dot as f64;
            gna2 += na2 as f64;
            gnb2 += nb2 as f64;
            let (na, nb) = (na2.sqrt(), nb2.sqrt());
            if na > 1e-12 && nb > 1e-12 {
                per.push((k.to_string(), dot / (na * nb), na, nb));
            }
        }
        let global_cos = (gdot / (gna2.sqrt() * gnb2.sqrt())) as f32;
        per.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
        let max_norm = per.iter().map(|p| p.2).fold(0f32, f32::max);
        eprintln!(
            "[sc-4887] bf16-vs-f32 grads: global cosine {global_cos:.5}; worst per-param (cos, |f32|, |bf16|, |f32|/max):"
        );
        for (k, c, na, nb) in per.iter().take(5) {
            eprintln!(
                "    {k}: cos {c:.4}  |g| {na:.3e} vs {nb:.3e}  rel-norm {:.2e}",
                na / max_norm
            );
        }
        // Any LARGE-norm param (≥1% of the biggest grad) must also agree directionally — that is
        // the systematic-bug detector; small-norm direction noise is expected mixed-precision.
        let min_large = per
            .iter()
            .filter(|p| p.2 >= 0.01 * max_norm)
            .map(|p| p.1)
            .fold(1f32, f32::min);
        eprintln!("[sc-4887] min cosine among params with |g| >= 1% of max: {min_large:.4}");
        assert!(
            global_cos > 0.995,
            "bf16 global grad must point the same way as f32: {global_cos:.5}"
        );
        // Measured on real weights: 0.966 (worst = main-stack to_k.lora_b — k-grads flow through
        // the bf16 softmax backward, the most precision-sensitive chain; no norm shrink). The
        // structural failure this gate exists for looked very different: a CLUSTER at cos 0.43-0.81
        // with systematically smaller bf16 norms (the caption-entry sensitivity, fixed by keeping
        // it f32 — see `train_unify_dtype`).
        assert!(
            min_large > 0.95,
            "a large-norm param's bf16 grad diverged from f32 (systematic bug, not precision): {min_large:.4}"
        );

        let (_, bf16_peak, _) = one_step(
            &mut trainer,
            &adapter,
            &params,
            &cap,
            768,
            None,
            Dtype::Bfloat16,
            "attn-ckpt bf16",
        )
        .expect("bf16 step");
        eprintln!(
            "[sc-4887] 768 peak f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB ({:.0}%)",
            100.0 * bf16_peak / f32_peak
        );
        assert!(
            bf16_peak < 0.70 * f32_peak,
            "bf16 must materially shrink the working set (silent f32 re-promotion?): \
             f32 {f32_peak:.2} GB vs bf16 {bf16_peak:.2} GB"
        );
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::projected_dense_peak_gb;

    // The empirical fit must reproduce the measured first-step peaks within a few GB and stay
    // monotonic — it is the basis of the pre-flight OOM guard, so a regression here silently
    // mis-sizes the guard. s ≈ (edge/16)²: edge 512→1024, 768→2304, 1024→4096 tokens.
    // The measured points come from `first_step_attn_ckpt_sweep` (the training dense path always
    // runs attention-segment checkpointing since sc-4886; the original retained-attention 135 GB
    // curve from sc-4874 no longer describes any reachable training path).
    #[test]
    fn projection_matches_measured_curve() {
        // f32 attn-ckpt dense (measured by first_step_attn_ckpt_sweep, 128 GB Mac17,6):
        // s = (edge/16)² + 32 → edge 512/768/1024.
        for (s, measured) in [(1056.0, 42.1), (2336.0, 56.9), (4128.0, 79.2)] {
            let p = projected_dense_peak_gb(s, false);
            assert!(
                (p - measured).abs() < 3.0,
                "f32 projection at s={s} = {p:.1} GB, expected ≈{measured} GB"
            );
        }
        // bf16 attn-ckpt dense (measured at 768/1024; the 512 sample was cast-transient-polluted).
        for (s, measured) in [(2336.0, 32.2), (4128.0, 43.3)] {
            let p = projected_dense_peak_gb(s, true);
            assert!(
                (p - measured).abs() < 3.0,
                "bf16 projection at s={s} = {p:.1} GB, expected ≈{measured} GB"
            );
        }
        for bf16 in [false, true] {
            // Monotonic increasing in token count; bf16 strictly below f32.
            assert!(projected_dense_peak_gb(1056.0, bf16) < projected_dense_peak_gb(2336.0, bf16));
            assert!(projected_dense_peak_gb(2336.0, bf16) < projected_dense_peak_gb(4128.0, bf16));
            assert!(projected_dense_peak_gb(2336.0, true) < projected_dense_peak_gb(2336.0, false));
        }
        // With attention-segment checkpointing always on, the 1024 dense step fits the 128 GB
        // target in BOTH dtypes (budget ≈ 121.6 × 0.85 ≈ 103 GB) — the sc-4874 kill regime is gone
        // without the Gradient Checkpointing checkbox.
        assert!(projected_dense_peak_gb(4128.0, false) < 103.0);
        assert!(projected_dense_peak_gb(4128.0, true) < 103.0);
    }
}

#[cfg(test)]
mod validate_request_tests {
    use super::validate_request;
    use mlx_gen::{TrainingConfig, TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn request(items: usize, steps: u32, rank: u32) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "a cat".into(),
                })
                .collect(),
            config: TrainingConfig {
                steps,
                rank,
                ..Default::default()
            },
            output_dir: PathBuf::from("/tmp/z-image-trainer-test"),
            file_name: "adapter.safetensors".into(),
            trigger_words: Vec::new(),
            cancel: Default::default(),
        }
    }

    #[test]
    fn rejects_zero_steps() {
        // F-040: a 0-step run must fail validation — otherwise the train loop runs no iterations and
        // falls straight through to the save, writing a no-op `B = 0` identity adapter.
        let err = validate_request(&request(1, 0, 16)).unwrap_err();
        assert!(format!("{err}").contains("steps must be > 0"), "got: {err}");
    }

    #[test]
    fn accepts_valid_request_and_keeps_existing_guards() {
        assert!(validate_request(&request(1, 100, 16)).is_ok());
        assert!(validate_request(&request(0, 100, 16)).is_err()); // empty dataset
        assert!(validate_request(&request(1, 100, 0)).is_err()); // zero rank
    }

    #[test]
    fn rejects_unrecognized_schedule_and_loss_strings() {
        // F-041: a typo in these strings used to silently fall back to a default sampler/loss.
        let with = |f: fn(&mut TrainingConfig)| {
            let mut r = request(1, 100, 16);
            f(&mut r.config);
            validate_request(&r)
        };
        assert!(with(|c| c.timestep_type = "sgmoid".into()).is_err());
        assert!(with(|c| c.timestep_bias = "hihg_noise".into()).is_err());
        assert!(with(|c| c.loss_type = "huber".into()).is_err());

        // The defaults and documented spellings still pass, case- and separator-insensitively.
        assert!(with(|c| c.timestep_type = "Linear".into()).is_ok());
        assert!(with(|c| c.timestep_bias = "High-Noise".into()).is_ok());
        assert!(with(|c| c.loss_type = "L1".into()).is_ok());
        // A default request (sigmoid / balanced / mse) is accepted.
        assert!(validate_request(&request(1, 100, 16)).is_ok());
    }
}
