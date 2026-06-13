//! LoRA/LoKr **training** on the Lens DiT, in pure Rust on mlx-rs (sc-5148, epic 3164) — the
//! native-MLX replacement for the Python `lens_train_runner.py` (torch in `/opt/lens-venv`), the last
//! Python holdout for Lens (zero-Python north star, epic 3482).
//!
//! [`LensTrainer`] realizes the core [`Trainer`](mlx_gen::Trainer) contract on the real 48-block Lens
//! MMDiT, mirroring [`ZImageTurboTrainer`](mlx_gen_z_image) — the model crates don't use mlx-rs's
//! `Module` system (hand-rolled `&self` forwards over raw `Array`s), so training uses the **functional
//! autograd**: the trainable factors live OUTSIDE the model in a [`LoraParams`] map, re-injected each
//! step into the target [`AdaptableLinear`](mlx_gen::adapters::AdaptableLinear)s via the shared core
//! seam ([`mlx_gen::train::lora`]), stepped with `keyed_value_and_grad` + the core [`TrainOptimizer`] +
//! `clip_grad_norm`. The injection mirrors the inference reload op-for-op, so the trained adapter
//! round-trips through [`apply_lens_adapters`](crate::adapters::apply_lens_adapters) (sc-3174)
//! bit-for-bit.
//!
//! ## What is Lens-specific (everything else reuses the family-agnostic core unchanged)
//!
//! Ported from `lens_train_runner.py`:
//!   * **Flow-match velocity target = `noise − x0`** with the transformer **timestep = `t`** (the noise
//!     fraction) fed directly. The Lens DiT [`forward`](crate::dit::LensTransformer::forward) returns
//!     the **raw** patch-space velocity (no negation — the pipeline feeds it to `FlowMatchEuler::step`
//!     un-negated), so the regression target is the velocity itself. This is the **opposite sign** of
//!     the Z-Image trainer, whose Rust `forward()` negates → target `noise − x0` *with* `timestep =
//!     1 − σ`. `x_t = (1 − t)·x0 + t·noise`.
//!   * **Latents by inverting the Lens `_decode`.** The Lens latent space *is* the Flux.2 one, so a
//!     pixel → `[1, seq, 128]` training latent is exactly the Flux.2 `encode_init_latents` chain
//!     (`preprocess_ref_image → Flux2Vae::encode_mean → patchify → bn-normalize → pack`). Uses the
//!     deterministic latent **mean** (the only public encode path + the established mlx-gen img2img
//!     convention); the Python's `latent_dist.sample()` reparam-noise is a minor regularizer dropped
//!     deliberately.
//!   * **Caption features.** The pipeline's positive-only `encode_one`: tokenize → the gpt-oss
//!     [`encode`](crate::text_encoder::encoder::LensTextEncoder::encode) (4 captured layers) → slice
//!     at [`TXT_OFFSET`] → a ones mask. Single-conditional (no CFG), matching the Python.
//!   * **Targets** default to `img_qkv`/`txt_qkv`/`to_out.0`/`to_add_out` (the `AdaptableHost for
//!     LensTransformer` paths, sc-3174); LoKr reconstructs at [`LOKR_DTYPE`] (what the lens adapter
//!     loader uses, so the trained LoKr round-trips). The gpt-oss encoder loads **Q8** (~12 GB vs
//!     ~40 GB dense bf16) — frozen, used only to cache caption features, then dropped before the train
//!     loop (the 32 GB-Mac free pattern); Q8 also matches the Q8 inference default (sc-3172/sc-5105).
//!
//! Registered under the **`lens`** id (the base, non-distilled `microsoft/Lens` — sc-1583; arch-
//! identical to `lens_turbo`, so the adapter applies to both, sc-3174).
//!
//! ## Memory hardening (sc-5170, the z-image sc-4874/4886/4887 analog)
//! Production-resolution (1024) Lens LoRA training is memory-hardened with the z-image pattern:
//!   * **SDPA-segment checkpointing** is always on in training (LoRA and LoKr) — the joint SDPA runs
//!     inside an `mlx::checkpoint` so its backward recomputes the attention rather than retaining the
//!     `[heads, joint, joint]` probability matrix (the dominant seq² term; MLX has no fused SDPA
//!     backward). Numerically identical, and the flash-backward surrogate every torch trainer gets.
//!   * **`gradient_checkpointing`** (the SceneWorks toggle) is an opt-in OPTION (LoRA only): each of
//!     the 48 dual-stream blocks recomputes its activations in the backward via
//!     [`LensTransformer::forward_with_main_checkpointed`](crate::dit::LensTransformer::forward_with_main_checkpointed),
//!     threading the per-block LoRA factors as explicit checkpoint inputs so the adapter graph
//!     survives the recompute. LoKr keeps the dense path (caught by the guard) — mirroring z-image.
//!   * **Fail-fast OOM preflight guard** — [`preflight_memory_guard`] projects the dense first-step
//!     peak from resolution (a fitted curve) and, when checkpointing is off and the run would exceed
//!     this machine's memory budget, returns a catchable, actionable error BEFORE the (minutes-long)
//!     latent caching — converting the otherwise-uncatchable SIGKILL into a recommendation to enable
//!     the toggle. The default-off functional path is unaffected.

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::gen_core;
use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::Weights;
use mlx_gen::{
    Error, LoadSpec, Modality, NetworkType, Precision, Quant, Result, TrainOptimizer, Trainer,
    TrainerDescriptor, TrainerRegistration, TrainingConfig, TrainingOutput, TrainingProgress,
    TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::memory::get_memory_limit;
use mlx_rs::ops::{add, multiply, ones, split_sections, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use mlx_gen_flux2::{load_vae, pack_latents, patchify_latents, preprocess_ref_image, Flux2Vae};

use crate::config::GptOssConfig;
use crate::dit::{LensDitConfig, LensTransformer};
use crate::pipeline::{DEFAULT_DATE, VAE_SCALE_FACTOR};
use crate::registry::MODEL_ID_BASE;
use crate::text::{LensTokenizer, TXT_OFFSET};
use crate::text_encoder::encoder::LensTextEncoder;

/// The lens adapter loader reconstructs LoKr deltas at bf16 (`src/adapters/loader.rs`); training must
/// reconstruct at the same dtype so the trained LoKr round-trips through `apply_lens_adapters`.
const LOKR_DTYPE: Dtype = Dtype::Bfloat16;

/// The gpt-oss encoder is loaded Q8 for the trainer (~12 GB vs ~40 GB dense bf16): it is frozen and
/// used only to cache caption features once, then dropped. Q8 is the Lens inference default (sc-3172),
/// so the cached features match the deployed encode path.
const TRAINER_ENCODER_QUANT: Option<Quant> = Some(Quant::Q8);

/// The Lens trainer default target modules (`lens_train_runner.DEFAULT_LORA_TARGET_MODULES`): the
/// dual-stream joint-attention projections. `to_out` is an `nn.ModuleList([Linear, Identity])`, so the
/// trainable Linear is `to_out.0` (sc-2218); `img_qkv`/`txt_qkv` are the fused per-stream QKV.
const DEFAULT_TARGET_MODULES: [&str; 4] = ["img_qkv", "txt_qkv", "to_out.0", "to_add_out"];

/// Recognized `timestep_type` values [`sample_sigma`] branches on (plus the `sigmoid` default).
const TIMESTEP_TYPES: [&str; 4] = ["sigmoid", "linear", "uniform", "weighted"];
/// Recognized `timestep_bias` values [`sample_sigma`] branches on (plus the neutral default).
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
/// Recognized `loss_type` values — `mae`/`l1` → MAE, `mse`/`l2` → the MSE default.
const LOSS_TYPES: [&str; 4] = ["mse", "l2", "mae", "l1"];

/// `(x_t, target)` for a single sample at flow-match `t`: `x_t = (1−t)·x0 + t·noise`,
/// `target = noise − x0` (the velocity the **raw** Lens DiT output is regressed onto; the transformer
/// timestep is `t` itself, fed by the caller — see the module docs for the sign vs Z-Image).
fn build_batch(x0: &Array, noise: &Array, t: f32) -> Result<(Array, Array)> {
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let s = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?;
    Ok((x_t, target))
}

/// The production [`Trainer`] for the base `microsoft/Lens` DiT: a frozen base (gpt-oss encoder + Lens
/// MMDiT + Flux.2 VAE + tokenizer) that caches a captioned dataset to VAE-latents/caption-features,
/// then runs the functional-autograd LoRA/LoKr loop with the core runtime glue (LR schedule, gradient
/// accumulation, checkpoint cadence, cancel, progress bands), writing a PEFT adapter that reloads
/// through the inference path.
pub struct LensTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: LensTokenizer,
    /// The 20 B-param gpt-oss encoder, in an `Option` so it can be **dropped after caching** — it is
    /// idle during training (every caption is already cached), yet a multi-GB resident.
    encoder: Option<LensTextEncoder>,
    transformer: LensTransformer,
    vae: Flux2Vae,
    /// The compute dtype (bf16 production / f32 tight-gate), fixed at load from `spec.precision`.
    dtype: Dtype,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID_BASE,
        family: "lens",
        backend: "mlx",
        modality: Modality::Image,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Construct the trainer from a `microsoft/Lens` snapshot directory (the diffusers multi-component
/// tree: `tokenizer/ text_encoder/ transformer/ vae/`). The DiT is loaded **dense** (the adapter host);
/// the encoder is Q8. `spec.precision` selects the compute dtype (bf16 default / f32 tight-gate).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p.clone(),
            WeightsSource::File(_) => return Err(Error::Msg(
                "lens trainer expects a snapshot directory (tokenizer/ text_encoder/ transformer/ \
                 vae/), not a single .safetensors file"
                    .into(),
            )),
        };
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let tokenizer = LensTokenizer::from_file(root.join("tokenizer").join("tokenizer.json"))?;
    let enc_cfg = GptOssConfig::lens();
    let enc_w = Weights::from_dir(root.join("text_encoder"))?;
    let encoder =
        LensTextEncoder::from_weights_quant(&enc_w, &enc_cfg, dtype, TRAINER_ENCODER_QUANT)?;
    let dit_cfg = LensDitConfig::lens();
    let dit_w = Weights::from_dir(root.join("transformer"))?;
    let transformer = LensTransformer::from_weights(&dit_w, &dit_cfg, dtype)?;
    let vae = load_vae(&root)?;
    Ok(Box::new(LensTrainer {
        descriptor: trainer_descriptor(),
        tokenizer,
        encoder: Some(encoder),
        transformer,
        vae,
        dtype,
    }))
}

/// Registry adapter: the trainer registry's `load` slot is typed on [`gen_core::Result`] (epic 3720);
/// bridge the crate's rich-`Result` [`load_trainer`] into it.
fn load_trainer_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Trainer>> {
    load_trainer(spec).map_err(Into::into)
}

inventory::submit! {
    TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer_registered }
}

/// Normalize a free-form config string the way the trainer's own parsers do (trim, lowercase,
/// `-`/space → `_`) so validation accepts exactly the spellings the run would.
fn normalize_cfg(s: &str) -> String {
    s.trim().to_ascii_lowercase().replace([' ', '-'], "_")
}

/// Capability-free training-request validation, factored out so it can be unit-tested without a loaded
/// trainer. Rejects an empty dataset, zero rank, **zero steps** (a 0-step run would write a no-op
/// `B = 0` identity adapter), an unsupported optimizer, and an unrecognized
/// `timestep_type`/`timestep_bias`/`loss_type` (rather than silently falling back to a default).
/// `gradient_checkpointing` is now a supported toggle (sc-5170) — the checkpointed DiT forward + the
/// fail-fast OOM preflight guard are wired in [`LensTrainer::train_impl`].
fn validate_request(req: &TrainingRequest) -> Result<()> {
    let cfg = &req.config;
    if req.items.is_empty() {
        return Err("lens trainer: dataset is empty".into());
    }
    if cfg.rank == 0 {
        return Err("lens trainer: rank must be > 0".into());
    }
    if cfg.steps == 0 {
        return Err("lens trainer: steps must be > 0".into());
    }
    if !TrainOptimizer::is_supported(&cfg.optimizer) {
        return Err(format!(
            "lens trainer: optimizer '{}' is not available on MLX training (supported: adamw, adam, \
             rose, prodigy)",
            cfg.optimizer
        )
        .into());
    }
    if !TIMESTEP_TYPES.contains(&normalize_cfg(&cfg.timestep_type).as_str()) {
        return Err(format!(
            "lens trainer: timestep_type '{}' is not recognized (supported: {})",
            cfg.timestep_type,
            TIMESTEP_TYPES.join(", ")
        )
        .into());
    }
    if !TIMESTEP_BIASES.contains(&normalize_cfg(&cfg.timestep_bias).as_str()) {
        return Err(format!(
            "lens trainer: timestep_bias '{}' is not recognized (supported: {})",
            cfg.timestep_bias,
            TIMESTEP_BIASES.join(", ")
        )
        .into());
    }
    if !LOSS_TYPES.contains(&normalize_cfg(&cfg.loss_type).as_str()) {
        return Err(format!(
            "lens trainer: loss_type '{}' is not recognized (supported: {})",
            cfg.loss_type,
            LOSS_TYPES.join(", ")
        )
        .into());
    }
    Ok(())
}

impl Trainer for LensTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        validate_request(req)?;
        // Non-default `lora_target_modules` that match no adaptable module on the DiT would train zero
        // parameters yet "succeed". Catch it here, where the loaded DiT is available to match against.
        if resolve_target_paths(&self.transformer, &req.config).is_empty() {
            return Err(format!(
                "lens trainer: lora_target_modules {:?} matched no adaptable module on the Lens DiT \
                 (targets are img_qkv/txt_qkv/to_out.0/to_add_out)",
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

impl LensTrainer {
    /// The rich-`Result` body behind [`Trainer::train`]; the trait wrapper bridges its tail into
    /// [`gen_core::Error`] (epic 3720), keeping `?` on `mlx_rs`/family helpers transparent here.
    fn train_impl(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> Result<TrainingOutput> {
        validate_request(req)?;
        let cfg = &req.config;

        let target_paths = resolve_target_paths(&self.transformer, cfg);
        if target_paths.is_empty() {
            return Err(format!(
                "lens trainer: lora_target_modules {:?} matched no adaptable module on the Lens DiT",
                cfg.lora_target_modules
            )
            .into());
        }

        // The DiT compute dtype is fixed at load (`spec.precision`); the Lens DiT has no
        // cast-after-load, so `train_dtype` is *enforced* against it (never a silent no-op). The
        // common case (TrainingConfig default bf16 + LoadSpec default Bf16) matches, so this only fires
        // on an explicit f32-vs-bf16 mismatch — telling the caller to load at the matching precision.
        let want_bf16 = {
            let t = cfg.train_dtype.trim();
            t.eq_ignore_ascii_case("bf16") || t.eq_ignore_ascii_case("bfloat16")
        };
        let loaded_bf16 = self.dtype == Dtype::Bfloat16;
        if want_bf16 != loaded_bf16 {
            return Err(format!(
                "lens trainer: train_dtype '{}' does not match the loaded precision ({}). Load the \
                 trainer with {} to train at that dtype.",
                cfg.train_dtype,
                if loaded_bf16 { "bf16" } else { "f32" },
                if want_bf16 {
                    "Precision::Bf16"
                } else {
                    "Precision::Fp32"
                }
            )
            .into());
        }
        let compute_dtype = self.dtype;
        // bf16 mixed precision: cast the folded LoRA residual to the activation dtype so the adapted
        // Linear stays bf16 (else it silently re-promotes the chain to f32). f32 → no cast.
        let lora_dtype = (compute_dtype != Dtype::Float32).then_some(compute_dtype);

        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);
        // Lens latent grid: a cell maps to a 16×16 pixel tile (Flux.2 8× VAE ∘ 2× DiT patchify). The
        // ÷32 bucket guarantees the VAE-encoded `edge/8` is even, so the 2×2 patchify divides cleanly.
        let latent = (edge / VAE_SCALE_FACTOR) as usize; // latent_h == latent_w (square)

        // sc-5170 — fail-fast pre-flight memory guard. The dense (non-block-checkpointed) first step
        // materializes the whole forward graph in one MLX `eval`; at high resolution that working set
        // can exceed unified memory and the OS hard-kills the worker with an UNCATCHABLE SIGKILL (no
        // in-process error — the run just appears to hang at the last cached latent). We cannot catch
        // that kill, so we predict it and refuse up front with a catchable, actionable error BEFORE
        // the (minutes-long) latent caching, UNLESS the run will block-checkpoint (LoRA + the toggle).
        // LoKr always takes the dense path (no clean thread-as-input form), so it is guarded
        // regardless of the toggle.
        let will_checkpoint =
            matches!(cfg.network_type, NetworkType::Lora) && cfg.gradient_checkpointing;
        if !will_checkpoint {
            preflight_memory_guard(edge, want_bf16)?;
        }

        // --- prepare → load → cache: VAE-latents + 4-layer caption features into memory ---
        on_progress(TrainingProgress::LoadingModel); // base model is already resident from load_trainer
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Vec<Array>, Array)> = Vec::with_capacity(req.items.len());
        for (i, item) in req.items.iter().enumerate() {
            if req.cancel.is_cancelled() {
                break;
            }
            on_progress(TrainingProgress::Caching {
                current: i as u32 + 1,
                total,
            });
            let img = center_crop_square(&decode_image(&item.image_path)?);
            let x0 = encode_latents(&self.vae, &img, edge)?; // [1, seq, 128]
            let encoder = self.encoder.as_ref().ok_or_else(|| {
                Error::Msg(
                    "lens trainer: text encoder already freed (caching after train loop)".into(),
                )
            })?;
            let (features, mask) =
                encode_caption(&self.tokenizer, encoder, &item.caption, compute_dtype)?;
            let mut to_eval: Vec<&Array> = Vec::with_capacity(features.len() + 2);
            to_eval.push(&x0);
            to_eval.push(&mask);
            to_eval.extend(features.iter());
            eval(to_eval)?;
            cache.push((x0, features, mask));
        }
        if cache.is_empty() {
            // A cancel mid-cache is a genuine cancellation → typed `Error::Canceled`; an empty cache
            // with no cancel is a real "no usable dataset items" error.
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            return Err("lens trainer: no usable dataset items".into());
        }

        // Every caption is cached now — free the 20 B-param encoder and evict its buffers before the
        // train loop, reclaiming that resident for the DiT working set.
        self.encoder = None;
        mlx_rs::memory::clear_cache();

        // --- adapter targets + params (LoRA or LoKr) + optimizer ---
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

        // sc-5170 — gradient checkpointing. Collect, per block, the adapter-routable LOCAL paths
        // trained on it (e.g. `"attn.img_qkv"`), in trained-file order — the factors a checkpoint
        // segment threads as explicit inputs. Every Lens adapter target lives in a block, so this
        // covers the whole trainable surface.
        let n_layers = self.transformer.cfg.num_layers;
        let mut block_local_targets: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
                .and_then(|rest| rest.split_once('.'))
            {
                if let Ok(i) = idx.parse::<usize>() {
                    if i < n_layers {
                        block_local_targets[i].push(local.to_string());
                    }
                }
            }
        }
        // Opt-in OPTION (the SceneWorks "Gradient Checkpointing" toggle), never auto-forced — a run
        // that would OOM is caught instead by the pre-flight guard above, which recommends this flag
        // rather than silently changing the user's training dynamics. LoRA only — LoKr (a captured-
        // param Kronecker reconstruction) falls back to the dense path.
        let use_checkpoint =
            matches!(adapter, TrainAdapter::Lora { .. }) && cfg.gradient_checkpointing;
        let checkpoint_blocks: Option<&[Vec<String>]> = if use_checkpoint {
            Some(&block_local_targets)
        } else {
            None
        };
        // SDPA-segment checkpointing is ALWAYS on in training (LoRA and LoKr): numerically identical
        // to the retained backward (same decomposed attention, recomputed) and removes the dominant
        // seq² per-block retention. When whole-block checkpointing is on, the per-block SDPA flag goes
        // OFF (the block recompute already covers attention; nesting would recompute it twice).
        self.transformer.set_sdpa_checkpoint(!use_checkpoint);

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
            let (x0, features, mask) = &cache[((step - 1) as usize) % cache.len()];
            let t = sample_sigma(
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
                features,
                mask,
                t,
                &noise,
                mae,
                compute_dtype,
                lora_dtype,
                latent,
                checkpoint_blocks,
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

        // Cancelled before a single step completed (`steps == 0` is rejected by `validate`): the
        // factors are still the `B = 0` no-op init. Surface the cancellation rather than writing a
        // valid-looking identity adapter as a trained artifact.
        if steps_run == 0 {
            return Err(Error::Canceled);
        }

        // --- save final adapter (the diffusers/PEFT format `apply_lens_adapters` loads, sc-3174) ---
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

/// Number of caption tokens assumed by the pre-flight projection. The unified attention sequence is
/// `img_len + txt_len`; `img_len = (edge/16)²` dominates (4096 at edge 1024), so the exact caption
/// length barely shifts the projection — this is the length the guard's fit was measured at, kept as a
/// representative constant rather than threading the (per-item, variable) real caption length.
const PREFLIGHT_TXT_TOKENS: f64 = 64.0;

/// Projected DENSE (non-block-checkpointed) first-step peak memory, in GB, as a function of the
/// unified token count `s = img_len + txt_len` — an empirical fit to peaks measured on the 128 GB
/// target. The structure follows the sc-4874 decomposition `weights + linear·s + quad·s²`: the
/// constant is the resident DiT base (the gpt-oss encoder is freed before the train loop), the linear
/// term is the per-token hidden-state activations across the 48 dual-stream blocks, and the quadratic
/// term is the seq² attention transient — demoted from "one retained `[24-head, joint, joint]`
/// probability matrix per block" to a single block's backward transient by the always-on SDPA-segment
/// checkpointing (sc-5170). bf16 roughly halves the weights + activation terms.
///
/// MEASURED (`first_step_ckpt_sweep`, 128 GB Mac17,6, rank 16 / 192 targets / batch 1, caption
/// `txt_len = 64`) with SDPA-segment checkpointing on and only the DiT resident (the gpt-oss encoder
/// is freed before the train loop, the VAE is idle):
///   f32  edge 512/768/1024 (s = 1088/2368/4160) → 22.78 / 31.83 / 45.22 GB
///   bf16 edge 512/768/1024                       → 25.14 / 34.18 / 47.57 GB
/// The exact 3-point fit reproduces all three per dtype. NOTE the Lens bf16 dense path is ~2-2.4 GB
/// ABOVE f32 (a near-constant ~10 GB activation offset) — the OPPOSITE of z-image, where bf16 halves
/// the working set; on the Lens DiT bf16 is the load-time precision/ecosystem default (sc-5148), not a
/// memory win. The DiT is smaller per-layer than z-image's, so this curve is measured fresh (not
/// reused). Assumes micro-batch 1; refit (via `first_step_ckpt_sweep`) if the activation shape or
/// resident set changes. `projection_matches_measured_curve` pins it to the measured points.
fn projected_dense_peak_gb(s: f64, bf16: bool) -> f64 {
    if bf16 {
        PREFLIGHT_BF16.0 + PREFLIGHT_BF16.1 * s + PREFLIGHT_BF16.2 * s * s
    } else {
        PREFLIGHT_F32.0 + PREFLIGHT_F32.1 * s + PREFLIGHT_F32.2 * s * s
    }
}

/// `(weights, linear, quad)` fit constants for [`projected_dense_peak_gb`] — the exact 3-point fits to
/// the measured `first_step_ckpt_sweep` peaks (see its docs). `projection_matches_measured_curve`
/// enforces the measured anchors; refit both tuples if the sweep prints materially different numbers.
const PREFLIGHT_F32: (f64, f64, f64) = (15.43, 6.618e-3, 1.308e-7);
const PREFLIGHT_BF16: (f64, f64, f64) = (17.80, 6.602e-3, 1.333e-7);

/// Refuse a run whose dense first step would exceed this machine's memory budget (and thus get
/// SIGKILLed), returning a catchable, actionable error instead. The budget is MLX's own reported
/// memory limit (≈ the device's recommended working set); the rest is [`check_preflight_budget`].
/// Only consulted when gradient checkpointing is OFF (LoKr, or LoRA with the toggle off).
fn preflight_memory_guard(edge: u32, bf16: bool) -> Result<()> {
    let budget_gb = get_memory_limit() as f64 / (1024.0 * 1024.0 * 1024.0);
    check_preflight_budget(edge, bf16, budget_gb)
}

/// The pure guard logic (no MLX global state, so it is unit-testable): refuse if the projected dense
/// first-step peak exceeds `budget_gb × 0.85`. `edge` is the bucketed training edge; the unified token
/// count is `(edge/16)²` (latent /8, patch 2) plus a representative caption block. The 0.85 leaves
/// headroom for the worker/host — exceeding it is the regime where the dense run was observed to die.
fn check_preflight_budget(edge: u32, bf16: bool, budget_gb: f64) -> Result<()> {
    let tokens_per_side = (edge as f64 / 16.0).ceil();
    let s = tokens_per_side * tokens_per_side + PREFLIGHT_TXT_TOKENS;
    let projected = projected_dense_peak_gb(s, bf16);
    let safe = budget_gb * 0.85;
    if projected > safe {
        return Err(format!(
            "lens trainer: a dense first training step at resolution {edge} needs ~{projected:.0} GB \
             (the forward working set materializes in one allocation), exceeding this machine's \
             ~{safe:.0} GB safe budget ({budget_gb:.0} GB MLX limit × 0.85). Without mitigation the OS \
             would hard-kill the worker (SIGKILL) at the first step with no recoverable error \
             (sc-4874/sc-5170). Enable Gradient Checkpointing (recomputes block activations in the \
             backward) or reduce the training resolution."
        )
        .into());
    }
    Ok(())
}

/// Resolve the config's target-module *suffixes* (default [`DEFAULT_TARGET_MODULES`]) to full dotted
/// paths by matching them against every adapter-routable module on the DiT — the same suffix-match
/// PEFT's `LoraConfig(target_modules=…)` does (`transformer_blocks.{i}.attn.{suffix}`).
fn resolve_target_paths(transformer: &LensTransformer, cfg: &TrainingConfig) -> Vec<String> {
    let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
        DEFAULT_TARGET_MODULES
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
        .map_err(|e| Error::Msg(format!("decode image {}: {e}", path.display())))?;
    let rgb = dynimg.to_rgb8();
    let (width, height) = (rgb.width(), rgb.height());
    Ok(Image {
        width,
        height,
        pixels: rgb.into_raw(),
    })
}

/// Encode a center-cropped square image into a Lens training latent `[1, latent·latent, 128]` — the
/// inverse of the Lens `_decode`. The Lens latent space *is* the Flux.2 one, so this is the Flux.2
/// `encode_init_latents` chain built from public helpers: `preprocess_ref_image` (resize + `[−1,1]`
/// NHWC) → `Flux2Vae::encode_mean` (latent mean) → NCHW → 2×2 `patchify_latents` →
/// `Flux2Vae::bn_normalize_nchw` → `pack_latents`. `pack_latents`/`patchify_latents` are plain
/// row-major reshapes consistent with the lens `vae::decode` plain-reshape path, so the latent lives in
/// exactly the space the DiT predicts in. `crop_to_even`/`match_latent_spatial_size` (in the fork's
/// `encode_init_latents`) are no-ops at the ÷32-bucketed square edge, so they are elided.
fn encode_latents(vae: &Flux2Vae, image: &Image, edge: u32) -> Result<Array> {
    let pre = preprocess_ref_image(image, edge, edge)?; // NHWC [1, edge, edge, 3]
    let enc = vae.encode_mean(&pre)?; // NHWC [1, edge/8, edge/8, 32]
    let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the packing helpers
    let patchified = patchify_latents(&enc)?; // [1, 128, edge/16, edge/16]
    let normed = vae.bn_normalize_nchw(&patchified)?; // (x − mean)/std on the packed 128-ch
    pack_latents(&normed) // [1, latent·latent, 128]
}

/// Encode a caption into its per-layer DiT text features (sliced at [`TXT_OFFSET`]) + the valid mask —
/// the pipeline's positive-only `encode_one` (single-conditional training; the Python keeps the
/// positives of `encode_prompt(neg="")`). Returns `(features, mask)`: `features` is 4 × `[1, S, 2880]`,
/// `mask` is `[1, S]` (all-1; a single prompt is unpadded).
fn encode_caption(
    tokenizer: &LensTokenizer,
    encoder: &LensTextEncoder,
    caption: &str,
    dtype: Dtype,
) -> Result<(Vec<Array>, Array)> {
    let out = tokenizer.encode(caption, DEFAULT_DATE)?;
    let l = out.ids.len() as i32;
    let offset = TXT_OFFSET as i32;
    if l <= offset {
        return Err(format!(
            "lens trainer: caption tokenized to {l} tokens (≤ the {offset}-token harmony preamble), \
             leaving no conditioning tokens"
        )
        .into());
    }
    let input_ids = Array::from_slice(&out.ids, &[1, l]);
    let layers = encoder.encode(&input_ids)?; // num_text_layers × [1, L, 2880]
                                              // `[:, offset:, :]` — split at the offset along the sequence axis, keep the tail.
    let features = layers
        .iter()
        .map(|f| Ok(split_sections(f, &[offset], 1)?[1].as_dtype(dtype)?))
        .collect::<Result<Vec<_>>>()?;
    let mask = ones::<f32>(&[1, l - offset])?;
    Ok((features, mask))
}

/// Sample a normalized flow-match timestep (interpolation coefficient) `t ∈ [1e-3, 1−1e-3]` — a
/// faithful port of the SceneWorks `sample_training_timestep` (identical to the Z-Image trainer):
/// `sigmoid(randn)` by default, `uniform` for linear, `(uniform + sigmoid(randn))/2` for weighted;
/// bias `high` → `√t`, `low` → `t²`. Deterministic in `seed`.
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
/// Lens DiT, regress the **raw** `forward()` velocity onto `noise − x0`, return `(loss, grads)`. The
/// transformer timestep is `t` (the noise fraction) directly. `dtype` is the training compute dtype:
/// `x_t`/features are cast at entry (the weights were loaded at this dtype), the LoRA factors are cast
/// inside the traced install (`lora_dtype`), so the DiT graph runs at `dtype`; the noising math, loss,
/// and grads stay f32.
///
/// `checkpoint_blocks`, when `Some`, lists per-block LOCAL LoRA target paths and switches the forward
/// to the gradient-checkpointed path (sc-5170) — each block recomputes its activations in the backward
/// instead of retaining them. `None` runs the dense (activation-retaining) forward. Either way the
/// per-block SDPA-segment checkpointing flag is the caller's responsibility (set on `transformer`).
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    transformer: &mut LensTransformer,
    params: &LoraParams,
    adapter: &TrainAdapter,
    alpha: f32,
    rank: f32,
    x0: &Array,
    features: &[Array],
    mask: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    dtype: Dtype,
    lora_dtype: Option<Dtype>,
    latent: usize,
    checkpoint_blocks: Option<&[Vec<String>]>,
) -> Result<(f32, LoraParams)> {
    let (x_t, target) = build_batch(x0, noise, t)?;
    let x_t = x_t.as_dtype(dtype)?; // no-op in f32 mode
    let timestep = Array::from_slice(&[t], &[1]);
    let feats: Vec<Array> = features
        .iter()
        .map(|f| f.as_dtype(dtype))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let mask = mask.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        // Install ALL targets so the dense path (and any non-checkpointed targets) train through
        // ordinary autograd; on the checkpointed path the block adapters installed here are simply
        // replaced inside each checkpoint segment by the explicit-input factors, so they cost nothing.
        adapter.install_as(transformer, &p, alpha, rank, lora_dtype, LOKR_DTYPE)?;
        let v = match checkpoint_blocks {
            Some(locals) => transformer
                .forward_with_main_checkpointed(
                    &x_t,
                    &feats,
                    Some(&mask),
                    &timestep,
                    1,
                    latent,
                    latent,
                    &p,
                    locals,
                    alpha,
                )
                .map_err(|e| Exception::custom(e.to_string()))?,
            None => transformer
                .forward(&x_t, &feats, Some(&mask), &timestep, 1, latent, latent)
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
// sc-5170 — first-step memory/grad-parity harness (weight-gated, run as its own process).
//
// Drives the exact inner training step (`compute_loss_grads` + the step-1 grad `eval` the real loop
// forces at the optimizer step) directly on the REAL Lens DiT — synthesizing the latent + caption
// features (the encoder/VAE are irrelevant to the DiT working set, which is what is under test) —
// sweeping resolution with MLX peak-memory probes around it. The sweep produces the points the
// `projected_dense_peak_gb` guard is fit to; the parity tests prove the checkpointed forward and the
// SDPA-segment checkpointing do not change the gradients.
//
//   cargo test -p mlx-gen-lens --release --lib first_step -- --ignored --nocapture
//   cargo test -p mlx-gen-lens --release --lib grads_match -- --ignored --nocapture
// ===========================================================================================
#[cfg(test)]
mod first_step_repro {
    use super::*;
    use mlx_gen::train::lora::build_lora_targets;
    use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};
    use std::path::PathBuf;

    /// The base `microsoft/Lens` snapshot (the `LENS_SNAPSHOT` override, else the newest HF-cache
    /// snapshot with a `transformer/` tree).
    fn snapshot() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("LENS_SNAPSHOT") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var("HOME").ok()?;
        let snaps =
            PathBuf::from(home).join(".cache/huggingface/hub/models--microsoft--Lens/snapshots");
        std::fs::read_dir(&snaps)
            .ok()?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir() && p.join("transformer").is_dir())
    }

    fn gb(bytes: usize) -> f64 {
        bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Load just the Lens DiT at `dtype` (the only component whose activations drive the first-step
    /// working set; the encoder is freed before the train loop and the VAE is idle).
    fn build_dit(dtype: Dtype) -> LensTransformer {
        let root = snapshot().expect("microsoft/Lens snapshot (HF cache or LENS_SNAPSHOT)");
        let w = Weights::from_dir(root.join("transformer")).unwrap();
        LensTransformer::from_weights(&w, &LensDitConfig::lens(), dtype).unwrap()
    }

    /// Default-target LoRA factors (rank 16) on a freshly-loaded DiT — the production target surface
    /// (192 = 48 blocks × 4 joint-attention projections).
    fn build_targets(dit: &mut LensTransformer) -> (TrainAdapter, LoraParams) {
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(dit, &cfg);
        let (targets, params) = build_lora_targets(dit, &target_paths, 16, 7).unwrap();
        (TrainAdapter::Lora { targets }, params)
    }

    /// Per-block LOCAL LoRA target paths (mirrors `train_impl`), for driving the checkpointed path.
    fn block_local_targets(dit: &LensTransformer) -> Vec<Vec<String>> {
        let cfg = TrainingConfig {
            rank: 16,
            ..Default::default()
        };
        let target_paths = resolve_target_paths(dit, &cfg);
        let n_layers = dit.cfg.num_layers;
        let mut out: Vec<Vec<String>> = vec![Vec::new(); n_layers];
        for path in &target_paths {
            if let Some((idx, local)) = path
                .strip_prefix("transformer_blocks.")
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

    /// Synthesize one training batch at `edge`: a clean latent `[1, (edge/16)², 128]`, matching noise,
    /// 4 caption-feature layers `[1, txt_len, 2880]`, and an all-valid mask `[1, txt_len]`. The latent
    /// magnitude is irrelevant — the graph SIZE (driven by resolution) is the variable under test.
    fn synth(edge: u32) -> (Array, Array, Vec<Array>, Array, usize) {
        let latent = (edge / VAE_SCALE_FACTOR) as usize;
        let seq = (latent * latent) as i32;
        let txt = 64i32; // PREFLIGHT_TXT_TOKENS — a representative caption length
        let x0 = random::normal::<f32>(&[1, seq, 128], None, None, Some(&random::key(1).unwrap()))
            .unwrap();
        let noise =
            random::normal::<f32>(&[1, seq, 128], None, None, Some(&random::key(2).unwrap()))
                .unwrap();
        let feats: Vec<Array> = (0..4)
            .map(|k| {
                random::normal::<f32>(
                    &[1, txt, 2880],
                    None,
                    None,
                    Some(&random::key(10 + k).unwrap()),
                )
                .unwrap()
            })
            .collect();
        let mask = ones::<f32>(&[1, txt]).unwrap();
        mlx_rs::transforms::eval(
            std::iter::once(&x0)
                .chain(std::iter::once(&noise))
                .chain(std::iter::once(&mask))
                .chain(feats.iter())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        (x0, noise, feats, mask, latent)
    }

    /// Run a single first training step at `edge` and report peak GPU memory across the
    /// forward+backward (forces the backward grad eval — the real step-1 kill point). `sdpa_ckpt` arms
    /// the always-on SDPA-segment checkpointing; `checkpoint_blocks` switches on whole-block
    /// checkpointing.
    #[allow(clippy::too_many_arguments)]
    fn one_step(
        dit: &mut LensTransformer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        edge: u32,
        dtype: Dtype,
        checkpoint_blocks: Option<&[Vec<String>]>,
        sdpa_ckpt: bool,
        tag: &str,
    ) -> (f32, f64) {
        dit.set_sdpa_checkpoint(sdpa_ckpt);
        let (x0, noise, feats, mask, latent) = synth(edge);
        let lora_dtype = (dtype != Dtype::Float32).then_some(dtype);
        clear_cache();
        reset_peak_memory();
        let before = get_active_memory();
        let t0 = std::time::Instant::now();
        let (loss, grads) = compute_loss_grads(
            dit,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            &feats,
            &mask,
            0.5,
            &noise,
            false,
            dtype,
            lora_dtype,
            latent,
            checkpoint_blocks,
        )
        .unwrap();
        // `compute_loss_grads` only forces the loss (forward). The real trainer forces the backward at
        // the step-1 optimizer `eval`; do the same here so the peak reflects the true working set.
        eval(grads.values()).unwrap();
        let secs = t0.elapsed().as_secs_f64();
        let peak = get_peak_memory();
        eprintln!(
            "  [edge {edge:>4} {tag}] seq {}  loss {loss:.5}  active-before {:.2} GB  peak {:.2} GB  step {secs:.2}s",
            latent * latent,
            gb(before),
            gb(peak)
        );
        (loss, gb(peak))
    }

    /// Max relative grad diff between two param maps.
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

    /// Grads for one configuration at a small (resolution-agnostic for the math) edge.
    fn grads_at(
        dit: &mut LensTransformer,
        adapter: &TrainAdapter,
        params: &LoraParams,
        edge: u32,
        checkpoint_blocks: Option<&[Vec<String>]>,
        sdpa_ckpt: bool,
    ) -> LoraParams {
        dit.set_sdpa_checkpoint(sdpa_ckpt);
        let (x0, noise, feats, mask, latent) = synth(edge);
        let (_l, g) = compute_loss_grads(
            dit,
            params,
            adapter,
            16.0,
            16.0,
            &x0,
            &feats,
            &mask,
            0.5,
            &noise,
            false,
            Dtype::Float32,
            None,
            latent,
            checkpoint_blocks,
        )
        .unwrap();
        eval(g.values()).unwrap();
        g
    }

    /// Whole-block gradient checkpointing must not change the math: the checkpointed forward+grads
    /// must match the dense path within fp tolerance (it reuses the same install + block forward,
    /// recompute-only). Compares the production paths — dense (SDPA-ckpt on) vs block-checkpointed
    /// (SDPA-ckpt off, block recompute covers attention).
    #[test]
    #[ignore = "needs real microsoft/Lens weights; run as its own process"]
    fn checkpointed_grads_match_dense() {
        let mut dit = build_dit(Dtype::Float32);
        let (adapter, params) = build_targets(&mut dit);
        let locals = block_local_targets(&dit);
        let edge = 256u32; // small → dense is cheap; the math is resolution-agnostic
        let g_dense = grads_at(&mut dit, &adapter, &params, edge, None, true);
        let g_ckpt = grads_at(&mut dit, &adapter, &params, edge, Some(&locals), false);
        let max_rel = max_rel_diff(&g_dense, &g_ckpt);
        eprintln!("[sc-5170] checkpointed-vs-dense grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-3,
            "checkpointed grads must match dense within tolerance: max rel {max_rel:.2e}"
        );
    }

    /// The always-on SDPA-segment checkpointing must not change the math: grads with the SDPA
    /// checkpoint on must match the retained backward (flag off). Same decomposed attention,
    /// recomputed instead of retained → expect (near-)bit-identical.
    #[test]
    #[ignore = "needs real microsoft/Lens weights; run as its own process"]
    fn attn_ckpt_grads_match_retained() {
        let mut dit = build_dit(Dtype::Float32);
        let (adapter, params) = build_targets(&mut dit);
        let edge = 256u32;
        let g_retained = grads_at(&mut dit, &adapter, &params, edge, None, false);
        let g_ckpt = grads_at(&mut dit, &adapter, &params, edge, None, true);
        let max_rel = max_rel_diff(&g_retained, &g_ckpt);
        eprintln!("[sc-5170] attn-ckpt-vs-retained grad max relative diff: {max_rel:.2e}");
        assert!(
            max_rel < 1e-5,
            "SDPA-segment checkpointing must not change grads: max rel {max_rel:.2e}"
        );
    }

    /// The fit basis: first-step peak by resolution on the production DENSE path (SDPA-segment
    /// checkpointing always on), f32 then bf16. These printed points are what `projected_dense_peak_gb`
    /// is fit to — refit the `PREFLIGHT_F32`/`PREFLIGHT_BF16` constants if this prints materially
    /// different numbers. Plus a block-checkpointed 1024 point (the OOM mitigation).
    #[test]
    #[ignore = "needs real microsoft/Lens weights; run as its own process (may SIGKILL at 1024 dense)"]
    fn first_step_ckpt_sweep() {
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let tag = if dtype == Dtype::Float32 {
                "f32"
            } else {
                "bf16"
            };
            eprintln!("[sc-5170] dense first-step sweep ({tag}), SDPA-ckpt on:");
            let mut dit = build_dit(dtype);
            let (adapter, params) = build_targets(&mut dit);
            for edge in [512u32, 768, 1024] {
                let _ = one_step(
                    &mut dit,
                    &adapter,
                    &params,
                    edge,
                    dtype,
                    None,
                    true,
                    &format!("dense {tag}"),
                );
            }
            // The OOM mitigation: block-checkpointed 1024 (SDPA-ckpt off, block recompute covers it).
            let locals = block_local_targets(&dit);
            let _ = one_step(
                &mut dit,
                &adapter,
                &params,
                1024,
                dtype,
                Some(&locals),
                false,
                &format!("blk-ckpt {tag}"),
            );
            drop(dit);
            clear_cache();
        }
    }

    /// The fix demonstration: at production resolution 1024, gradient checkpointing must drop the
    /// first-step peak below the dense path's. Runs the dense step first (baseline), then the
    /// checkpointed step, and asserts a reduction that fits unified memory.
    #[test]
    #[ignore = "needs real microsoft/Lens weights; run as its own process"]
    fn first_step_1024_checkpointed_vs_dense() {
        let mut dit = build_dit(Dtype::Bfloat16);
        let (adapter, params) = build_targets(&mut dit);
        let locals = block_local_targets(&dit);
        let n: usize = locals.iter().map(|v| v.len()).sum();
        eprintln!("[sc-5170] checkpointing {n} LoRA targets across the 48-block stack");
        let (_, dense_peak) = one_step(
            &mut dit,
            &adapter,
            &params,
            1024,
            Dtype::Bfloat16,
            None,
            true,
            "dense bf16",
        );
        let (_, ckpt_peak) = one_step(
            &mut dit,
            &adapter,
            &params,
            1024,
            Dtype::Bfloat16,
            Some(&locals),
            false,
            "blk-ckpt bf16",
        );
        eprintln!(
            "[sc-5170] edge 1024 bf16  dense {dense_peak:.2} GB  ckpt {ckpt_peak:.2} GB  ({:.0}% reduction)",
            100.0 * (1.0 - ckpt_peak / dense_peak)
        );
        assert!(
            ckpt_peak < dense_peak,
            "checkpointing must reduce the first-step peak: dense {dense_peak:.2} vs ckpt {ckpt_peak:.2} GB"
        );
        assert!(
            ckpt_peak < 128.0,
            "checkpointed peak must fit unified memory: {ckpt_peak:.2} GB"
        );
    }
}

#[cfg(test)]
mod preflight_tests {
    use super::{check_preflight_budget, projected_dense_peak_gb};

    /// The empirical fit must reproduce the measured first-step peaks and stay monotonic — it is the
    /// basis of the pre-flight OOM guard, so a regression here silently mis-sizes the guard. The
    /// points come from `first_step_ckpt_sweep` (the training dense path always runs SDPA-segment
    /// checkpointing since sc-5170). s = (edge/16)² + 64: edge 512/768/1024 → 1088/2368/4160.
    #[test]
    fn projection_matches_measured_curve() {
        for (s, measured) in [(1088.0, 22.78), (2368.0, 31.83), (4160.0, 45.22)] {
            let p = projected_dense_peak_gb(s, false);
            assert!(
                (p - measured).abs() < 1.0,
                "f32 projection at s={s} = {p:.2} GB, expected ≈{measured} GB"
            );
        }
        for (s, measured) in [(1088.0, 25.14), (2368.0, 34.18), (4160.0, 47.57)] {
            let p = projected_dense_peak_gb(s, true);
            assert!(
                (p - measured).abs() < 1.0,
                "bf16 projection at s={s} = {p:.2} GB, expected ≈{measured} GB"
            );
        }
        // Monotonic increasing in token count, in both dtypes.
        for bf16 in [false, true] {
            assert!(projected_dense_peak_gb(1088.0, bf16) < projected_dense_peak_gb(2368.0, bf16));
            assert!(projected_dense_peak_gb(2368.0, bf16) < projected_dense_peak_gb(4160.0, bf16));
        }
        // Lens bf16 sits ABOVE f32 (the ~10 GB activation offset) — the opposite of z-image; this
        // pins that the curves were not accidentally swapped.
        assert!(projected_dense_peak_gb(4160.0, true) > projected_dense_peak_gb(4160.0, false));
    }

    /// The guard must FIRE (catchable error, not SIGKILL) when the dense first-step peak exceeds the
    /// machine's safe budget, and PASS when it fits — the sc-5170 acceptance for the dense-OOM case.
    #[test]
    fn guard_fires_over_budget_and_passes_under() {
        // A 24 GB-class budget (safe ≈ 20.4 GB): dense 1024 (~47 GB) must be refused with an
        // actionable error that recommends Gradient Checkpointing.
        let err = check_preflight_budget(1024, true, 24.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Gradient Checkpointing"), "got: {err}");
        assert!(
            err.contains("1024"),
            "error should name the resolution: {err}"
        );
        // A 128 GB-class budget (safe ≈ 108 GB) comfortably fits dense 1024 in both dtypes.
        assert!(check_preflight_budget(1024, true, 128.0).is_ok());
        assert!(check_preflight_budget(1024, false, 128.0).is_ok());
        // A 64 GB-class budget (safe ≈ 54 GB) still fits 1024 dense (~45-48 GB) but not a much larger
        // resolution — the guard is machine-aware, not a fixed threshold.
        assert!(check_preflight_budget(1024, true, 64.0).is_ok());
        assert!(check_preflight_budget(1440, true, 64.0).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::CancelFlag;
    use std::path::PathBuf;

    fn base_config() -> TrainingConfig {
        TrainingConfig {
            rank: 8,
            steps: 10,
            ..Default::default()
        }
    }

    fn req_with(config: TrainingConfig) -> TrainingRequest {
        TrainingRequest {
            items: vec![mlx_gen::TrainingItem {
                image_path: PathBuf::from("/tmp/x.png"),
                caption: "a swatch".into(),
            }],
            config,
            output_dir: PathBuf::from("/tmp/lens_unused"),
            file_name: "lora.safetensors".into(),
            trigger_words: vec![],
            cancel: CancelFlag::new(),
        }
    }

    #[test]
    fn descriptor_is_the_base_lens_id() {
        let d = trainer_descriptor();
        assert_eq!(d.id, "lens");
        assert_eq!(d.family, "lens");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.supports_lora && d.supports_lokr);
    }

    #[test]
    fn validate_rejects_empty_dataset_and_zero_rank_steps() {
        let mut r = req_with(base_config());
        r.items.clear();
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("dataset is empty"));

        let r = req_with(TrainingConfig {
            rank: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("rank"));

        let r = req_with(TrainingConfig {
            steps: 0,
            ..base_config()
        });
        assert!(validate_request(&r)
            .unwrap_err()
            .to_string()
            .contains("steps"));
    }

    #[test]
    fn validate_accepts_gradient_checkpointing() {
        // sc-5170 — gradient_checkpointing is now a supported toggle (checkpointed DiT forward + OOM
        // preflight guard), no longer a hard rejection. It must pass capability-free validation; the
        // checkpointed path is exercised in `train_impl` (and the real-weight harness).
        let r = req_with(TrainingConfig {
            gradient_checkpointing: true,
            ..base_config()
        });
        assert!(validate_request(&r).is_ok());
    }

    #[test]
    fn validate_rejects_unrecognized_optimizer_timestep_loss() {
        for cfg in [
            TrainingConfig {
                optimizer: "nope".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_type: "bogus".into(),
                ..base_config()
            },
            TrainingConfig {
                timestep_bias: "sideways".into(),
                ..base_config()
            },
            TrainingConfig {
                loss_type: "huber".into(),
                ..base_config()
            },
        ] {
            assert!(validate_request(&req_with(cfg)).is_err());
        }
        // The recognized spellings (incl. alias normalization) pass.
        assert!(validate_request(&req_with(TrainingConfig {
            timestep_type: "Weighted".into(),
            timestep_bias: "high-noise".into(),
            loss_type: "L1".into(),
            optimizer: "adamw".into(),
            ..base_config()
        }))
        .is_ok());
    }

    #[test]
    fn build_batch_is_lens_velocity_with_no_sign_flip() {
        // target = noise − x0 (the RAW Lens DiT velocity; the OPPOSITE sign of z-image's negated
        // forward), and x_t = (1−t)·x0 + t·noise. Timestep `t` is passed straight to the DiT (the
        // caller), unlike z-image's `1 − σ` — covered by checking the interpolation here.
        let x0 = Array::from_slice(&[2.0f32, 4.0, 6.0], &[1, 3, 1]);
        let noise = Array::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3, 1]);
        let t = 0.25f32;
        let (x_t, target) = build_batch(&x0, &noise, t).unwrap();
        // target = noise − x0 = [-1, -3, -5]
        assert_eq!(target.as_slice::<f32>(), &[-1.0, -3.0, -5.0]);
        // x_t = 0.75·x0 + 0.25·noise = [1.75, 3.25, 4.75]
        let xt = x_t.as_slice::<f32>();
        for (got, want) in xt.iter().zip([1.75f32, 3.25, 4.75].iter()) {
            assert!((got - want).abs() < 1e-6, "x_t {got} != {want}");
        }
    }

    #[test]
    fn sample_sigma_is_deterministic_and_in_range() {
        for kind in ["sigmoid", "linear", "weighted"] {
            for bias in ["balanced", "high", "low"] {
                let a = sample_sigma(kind, bias, 42).unwrap();
                let b = sample_sigma(kind, bias, 42).unwrap();
                assert_eq!(a, b, "{kind}/{bias} must be deterministic in seed");
                assert!(
                    (1e-3..=1.0 - 1e-3).contains(&a),
                    "{kind}/{bias} t={a} out of range"
                );
            }
        }
        // high-noise bias (√t) lifts the value vs low-noise bias (t²) for the same draw.
        assert!(
            sample_sigma("sigmoid", "high", 7).unwrap()
                > sample_sigma("sigmoid", "low", 7).unwrap()
        );
    }
}
