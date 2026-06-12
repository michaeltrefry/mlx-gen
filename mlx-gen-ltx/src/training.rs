//! sc-3047 — LTX-2.3 LoRA **training** on the video DiT, in pure Rust on mlx-rs. The Rust port of
//! SceneWorks' pure-MLX `_LtxMlxLoraBackend` / `LtxMlxLoraTrainer` (`training_adapters.py:3249-3628`),
//! realizing the core [`Trainer`] contract (epic 3039). Retiring the Python version removes the last
//! Python-MLX trainer (blocks sc-3049 cutover → sc-3242 `mlx-video` drop).
//!
//! Built on the same functional-autograd mechanism the Z-Image spike proved (sc-3042) and the
//! sc-3043 runtime glue, but LTX has its **own** adapter seam: its [`crate::transformer::Linear`]
//! carries a per-pass [`LoraStack`](crate::transformer) (not the core `AdaptableLinear`), so this
//! module uses the LTX-local [`Linear::set_train_lora`] training seam and its own target
//! enumeration / save, while reusing the core [`LoraParams`] + grad-accumulation helpers and the
//! runtime (schedule / dataset / checkpoint).
//!
//! **What is LTX-specific:**
//!   * **Video-only forward over `LtxDiT`.** The reference loads the AV model and trains with
//!     `audio=None`; [`LtxDiT`] is exactly that video-only reduction (the AV checkpoint embeds the
//!     same `transformer_blocks.{i}` video blocks), and the trained video-attention adapter reloads
//!     onto the AvDiT inference path unchanged.
//!   * **Rectified-flow target = `noise - clean`.** LTX denoises with `x_t - σ·v` over
//!     `x_t = (1-σ)·x0 + σ·noise` and feeds the **raw** transformer output straight to `to_denoised`
//!     (no negation, unlike Z-Image), so the velocity that recovers `x0` is `v = noise - x0`. The
//!     **timestep fed to the DiT is the raw σ** (broadcast over tokens), σ ~ U(1e-3, 1-1e-3). MSE.
//!   * **Latent layout.** A still image VAE-encodes (single frame T=1) to a normalized latent
//!     `(1,128,1,h,w)`, flattened to the patchified `(1, S, 128)` the DiT consumes; the position
//!     grid is built once for the fixed latent resolution. The 24 GB Gemma text encoder is freed
//!     after the one-time prompt-embed cache (mirroring the reference), before the train loop.
//!   * **Adapter surface.** `attn1`/`attn2` (self + text cross-attention) `to_q/k/v/to_out.0`, the
//!     reference `inject_video_attention_lora` default. Residual LoRA over the (Q4) base — the base
//!     is frozen, gradients flow only through the trainable factors (functional autograd handles the
//!     `quantized_matmul` base as a constant). Saved as `{module}.lora_A/B.weight` + `.alpha` (the
//!     `to_out.0` diffusers spelling the inference loader normalizes), so it round-trips through
//!     [`crate::apply_ltx_adapters`].
//!   * **LoRA-only.** The reference LTX MLX trainer has no LoKr (LTX *inference* supports LoKr via
//!     sc-2393, but no LoKr trainer exists); LoKr requests are rejected with that explanation.

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::media::Image;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{accumulate_grads, average_grads, LoraParams};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::{to_dtype, Weights};
use mlx_gen::{
    gen_core, LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainerRegistration, TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::{LtxConfig, LtxVaeConfig, SplitModel};
use crate::gemma::GemmaConfig;
use crate::model::MODEL_ID;
use crate::pipeline::preprocess_conditioning_image;
use crate::positions::{create_position_grid, SPATIAL_SCALE};
use crate::text_encoder::LtxTextEncoder;
use crate::tokenizer::LtxTokenizer;
use crate::transformer::{LtxDiT, Precision};
use crate::vae::LtxVideoVae;

/// Gemma prompt token budget for caption encoding (the captions are short; padding tokens are
/// attended with `mask=None`, matching the reference `Modality(context_mask=None)`).
const MAX_PROMPT_TOKENS: usize = 128;

/// The reference `inject_video_attention_lora` default targets (`DEFAULT_LORA_TARGET_MODULES`,
/// `training_adapters.py:72`), restricted to `attn1`/`attn2`. `to_out.0` is the diffusers spelling
/// the inference loader normalizes to the checkpoint's `to_out`.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["to_q", "to_k", "to_v", "to_out.0"];

/// One LoRA-trained attention `Linear`: its diffusers save spelling (e.g. `…attn1.to_out.0`), the
/// resolution segments after the `to_out.0`→`to_out` normalization, and the factor-map keys.
struct LtxLoraTarget {
    save_path: String,
    segs: Vec<String>,
    a_key: Rc<str>,
    b_key: Rc<str>,
}

/// LoRA trainer for LTX-2.3, implementing the core [`Trainer`] surface: a frozen LtxDiT (f32
/// activations × Q4/Q8 weights) + VAE + Gemma text encoder + tokenizer that caches a captioned
/// image dataset to (normalized latent, prompt-embed) pairs, then runs the functional-autograd
/// rectified-flow loop with the sc-3043 runtime glue, and writes a LoRA that round-trips through
/// [`crate::apply_ltx_adapters`].
///
/// **Single-use** (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB) after the
/// embed cache, so the instance cannot run a second job — `validate` (hence `train`) rejects a reuse
/// up front. Construct a fresh trainer (via [`load_trainer`]) per job.
pub struct LtxTrainer {
    descriptor: TrainerDescriptor,
    /// Freed after the one-time prompt-embed cache (the 24 GB Gemma backbone), before the loop.
    tokenizer: Option<LtxTokenizer>,
    text_encoder: Option<LtxTextEncoder>,
    vae: LtxVideoVae,
    transformer: LtxDiT,
    cfg: LtxConfig,
}

fn trainer_descriptor() -> TrainerDescriptor {
    TrainerDescriptor {
        id: MODEL_ID,
        family: "ltx",
        backend: "mlx",
        modality: Modality::Video,
        supports_lora: true,
        // The reference LTX MLX trainer is LoRA-only; LoKr training is unsupported (see `validate`).
        supports_lokr: false,
    }
}

/// Construct the trainer from an LTX-2.3 split-weight snapshot directory (transformer / VAE /
/// connector + the Gemma-3-12B text-encoder snapshot resolved like inference). The transformer loads
/// at **f32 activations × quantized weights** (`quant_f32`) for clean autograd — the base is frozen,
/// gradients flow only through the trainable LoRA factors. Registered via [`TrainerRegistration`].
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => return Err(mlx_gen::Error::Msg(
            "ltx_2_3 trainer expects a split-weight snapshot directory (transformer.safetensors \
                 / vae_*.safetensors / connector.safetensors), not a single file"
                .into(),
        )),
    };
    let split = SplitModel::from_model_dir(root)?;
    let cfg = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;

    let gemma_dir = crate::model::resolve_gemma_dir()?;
    let gemma_w = Weights::from_dir(&gemma_dir)?;
    let gemma_quant = crate::model::resolve_gemma_quant(&gemma_dir)?;
    let connector_w = Weights::from_file(root.join("connector.safetensors"))?;
    let transformer_w = Weights::from_file(root.join("transformer.safetensors"))?;
    let vae_dec_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let vae_enc_w = Weights::from_file(root.join("vae_encoder.safetensors"))?;

    // Video-only text encoder (bf16, the reference TE dtype); we cast its embeds to f32 per-item for
    // the f32 training forward.
    let text_encoder = LtxTextEncoder::from_weights(
        &gemma_w,
        &connector_w,
        GemmaConfig::gemma_3_12b(),
        gemma_quant,
        &cfg,
        Dtype::Bfloat16,
    )?;
    let transformer = LtxDiT::from_weights(
        &transformer_w,
        &cfg,
        Precision::quant_f32(split.bits, split.group),
    )?;
    let vae = LtxVideoVae::from_weights(&vae_dec_w, Some(&vae_enc_w), &vae_config)?;
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(Box::new(LtxTrainer {
        descriptor: trainer_descriptor(),
        tokenizer: Some(tokenizer),
        text_encoder: Some(text_encoder),
        vae,
        transformer,
        cfg,
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

/// Capability-free request validation, factored out of [`Trainer::validate`] so it can be
/// unit-tested without a loaded trainer. Rejects an empty dataset, zero rank, LoKr (LoRA-only
/// trainer), and unsupported optimizers. The single-use / text-encoder-present check stays in
/// [`Trainer::validate`], which has the trainer state to inspect (F-055).
fn validate_request(req: &TrainingRequest) -> Result<()> {
    if req.items.is_empty() {
        return Err("ltx_2_3 trainer: dataset is empty".into());
    }
    if req.config.rank == 0 {
        return Err("ltx_2_3 trainer: rank must be > 0".into());
    }
    if req.config.network_type == NetworkType::Lokr {
        return Err(
            "ltx_2_3 trainer: LoKr training is not supported — the reference LTX MLX \
                    trainer is LoRA-only. (LTX *inference* supports LoKr via sc-2393, but no \
                    LoKr trainer exists yet; that would be a separate extension.)"
                .into(),
        );
    }
    if !TrainOptimizer::is_supported(&req.config.optimizer) {
        return Err(format!(
            "ltx_2_3 trainer: optimizer '{}' is not available on MLX training (supported: \
             adamw, adam, rose, prodigy)",
            req.config.optimizer
        )
        .into());
    }
    Ok(())
}

impl Trainer for LtxTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> gen_core::Result<()> {
        // Single-use enforcement (F-055): `train` frees the Gemma text encoder + tokenizer (~24 GB)
        // after the embed cache, so a second `train` on the same instance can't re-encode. Fail here,
        // up front (validate runs before any progress is emitted), instead of with a late, confusing
        // "text encoder missing" mid-run. Construct a fresh trainer (via `load_trainer`) per job.
        if self.text_encoder.is_none() || self.tokenizer.is_none() {
            return Err(
                "ltx_2_3 trainer: single-use — the Gemma text encoder was freed after the \
                        first train() to reclaim ~24 GB; construct a fresh trainer for each job"
                    .into(),
            );
        }
        validate_request(req).map_err(Into::into)
    }

    fn train(
        &mut self,
        req: &TrainingRequest,
        on_progress: &mut dyn FnMut(TrainingProgress),
    ) -> gen_core::Result<TrainingOutput> {
        self.train_impl(req, on_progress).map_err(Into::into)
    }
}

impl LtxTrainer {
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
        let edge = bucket_resolution(cfg.resolution); // pixel edge, multiple of 32
        let latent_edge = (edge / SPATIAL_SCALE as u32).max(1) as usize; // latent tokens per side

        // --- prepare → load → cache: normalized latents + prompt embeds (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Array)> = Vec::with_capacity(req.items.len());
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg("ltx_2_3 trainer: text encoder missing".into())
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg("ltx_2_3 trainer: tokenizer missing".into()))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                let prep = preprocess_conditioning_image(&img, edge, edge)?; // (1,3,1,edge,edge)
                let latent = self.vae.encode(&prep)?; // (1,128,1,le,le), normalized, f32
                let clean = flatten_latent(&latent)?; // (1, S, 128)
                let (ids, mask) = tok.encode(&item.caption, MAX_PROMPT_TOKENS)?;
                let ctx = to_dtype(&te.encode(&ids, &mask)?, Dtype::Float32)?; // (1, L, 4096)
                eval([&clean, &ctx])?;
                cache.push((clean, ctx));
            }
        }
        if cache.is_empty() {
            return Err("ltx_2_3 trainer: no usable dataset items (all cancelled?)".into());
        }
        // Free the Gemma text encoder + tokenizer (~24 GB) before training — they are only needed for
        // the one-time embed cache (mirrors the reference `prepare_dataset` release).
        self.text_encoder = None;
        self.tokenizer = None;

        // The RoPE position grid is identical across items at a fixed latent resolution (single
        // frame) — build it once.
        let positions = create_position_grid(1, 1, latent_edge, latent_edge);

        // --- adapter targets + trainable factors ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let (targets, mut params) = build_targets(
            &mut self.transformer,
            self.cfg.num_layers,
            &suffixes,
            cfg.rank as i32,
            cfg.seed,
        )?;
        if targets.is_empty() {
            return Err(
                "ltx_2_3 trainer: no LoRA targets resolved (check lora_target_modules)".into(),
            );
        }
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
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
            let (clean, ctx) = &cache[((step - 1) as usize) % cache.len()];
            // σ ~ U(1e-3, 1-1e-3), deterministic in seed (the reference's uniform timestep).
            let sigma = {
                let k = random::key(cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64))?;
                random::uniform::<_, f32>(1e-3f32, 1.0 - 1e-3, &[1], Some(&k))?.item::<f32>()
            };
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut self.transformer,
                &params,
                &targets,
                alpha,
                rank,
                clean,
                ctx,
                &positions,
                sigma,
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
                save_lora(&params, &targets, alpha, cfg.rank, &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        save_lora(&params, &targets, alpha, cfg.rank, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Flatten a single-frame VAE latent `(1, 128, 1, le, le)` to the patchified `(1, S, 128)` the DiT
/// consumes (`S = le·le`) — the reference's `transpose(reshape(latent, (B, C, -1)), (0, 2, 1))`.
fn flatten_latent(latent: &Array) -> Result<Array> {
    let sh = latent.shape(); // [1, 128, 1, le, le]
    let (b, c) = (sh[0], sh[1]);
    let s = sh[2] * sh[3] * sh[4];
    let flat = latent.reshape(&[b, c, s])?; // (1, 128, S)
    Ok(flat.transpose_axes(&[0, 2, 1])?) // (1, S, 128)
}

/// `to_out.0` → `to_out`, the only diffusers→checkpoint rename in the attention LoRA surface (the
/// inference loader does the same in `adapters::normalize`); other suffixes pass through.
fn resolve_segments(save_path: &str) -> Vec<String> {
    save_path
        .replace(".to_out.0", ".to_out")
        .split('.')
        .map(String::from)
        .collect()
}

/// Enumerate the `attn1`/`attn2` × `suffixes` targets across the DiT's `num_layers` blocks, resolve
/// each on the (mutable) DiT, read its `[out,in]` base shape, and initialise the trainable factors
/// the reference `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`, `B = 0` `[out,rank]` — keyed
/// `{save_path}.lora_a` / `.lora_b`. Targets that do not resolve (a missing gated branch, a typo'd
/// suffix) are skipped.
fn build_targets(
    dit: &mut LtxDiT,
    num_layers: i32,
    suffixes: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LtxLoraTarget>, LoraParams)> {
    let mut targets = Vec::new();
    let mut params = LoraParams::new();
    let small = Array::from_slice(&[0.02f32], &[1]);
    let mut idx: u64 = 0;
    for i in 0..num_layers {
        for attn in ["attn1", "attn2"] {
            for suf in suffixes {
                let save_path = format!("transformer_blocks.{i}.{attn}.{suf}");
                let segs = resolve_segments(&save_path);
                let seg_refs: Vec<&str> = segs.iter().map(String::as_str).collect();
                let Some(lin) = dit.adaptable_mut(&seg_refs) else {
                    continue;
                };
                let shape = lin.base_shape(); // [out, in]
                let (out_f, in_f) = (shape[0], shape[1]);
                let a_key: Rc<str> = Rc::from(format!("{save_path}.lora_a"));
                let b_key: Rc<str> = Rc::from(format!("{save_path}.lora_b"));
                let ka = random::key(seed.wrapping_add(2 * idx + 1))?;
                let a = multiply(
                    &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
                    &small,
                )?;
                let b = Array::zeros::<f32>(&[out_f, rank])?;
                eval([&a, &b])?;
                params.insert(a_key.clone(), a);
                params.insert(b_key.clone(), b);
                targets.push(LtxLoraTarget {
                    save_path,
                    segs,
                    a_key,
                    b_key,
                });
                idx += 1;
            }
        }
    }
    Ok((targets, params))
}

/// Inject the current trainable factors as one LoRA residual per target via the LTX training seam —
/// transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold `alpha/rank` into `b` — so the residual
/// is `(x·Aᵀ·Bᵀ)·(alpha/rank)`, matching the reference `_MlxLoRALinear`. Differentiable.
fn install_train_lora(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b = params[&t.b_key]
            .t()
            .multiply(Array::from_slice(&[alpha / rank], &[1]))?; // [out,r] -> [r,out] · (α/r)
        let seg_refs: Vec<&str> = t.segs.iter().map(String::as_str).collect();
        let lin = dit
            .adaptable_mut(&seg_refs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.save_path)))?;
        lin.set_train_lora(a, b);
    }
    Ok(())
}

/// One forward+backward over the trainable factors: build the rectified-flow input `x_t`, inject the
/// factors, run the video DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    dit: &mut LtxDiT,
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    positions: &Array,
    sigma: f32,
    noise: &Array,
    mae: bool,
) -> Result<(f32, LoraParams)> {
    // x_t = (1-σ)·clean + σ·noise; target = noise - clean (the raw-output velocity); timestep = σ.
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, clean)?;
    let timestep = Array::from_slice(&[sigma], &[1, 1]); // (B, 1), broadcast over tokens
    let ctx = context.clone();
    let pos = positions.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        install_train_lora(dit, &p, targets, alpha, rank)?;
        let v = dit
            .forward(&x_t, &timestep, &ctx, None, &pos)
            .map_err(|e| Exception::custom(e.to_string()))?;
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

/// Write the trained LoRA as safetensors keyed by the LTX module paths — `{module}.lora_A.weight`
/// `[rank,in]`, `{module}.lora_B.weight` `[out,rank]`, scalar `{module}.alpha` (= `alpha`) — the
/// reference `_save_lora` format, reloadable by [`crate::apply_ltx_adapters`] (which folds
/// `scale = alpha/rank`). `networkType`/`rank`/`alpha` metadata mirrors the other family trainers.
fn save_lora(
    params: &LoraParams,
    targets: &[LtxLoraTarget],
    alpha: f32,
    rank: u32,
    path: &Path,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{}.alpha", t.save_path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        entries.push((format!("{}.lora_A.weight", t.save_path), &params[&t.a_key]));
        entries.push((format!("{}.lora_B.weight", t.save_path), &params[&t.b_key]));
    }
    for (k, v) in &alphas {
        entries.push((k.clone(), v));
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lora".to_string());
    meta.insert("rank".to_string(), rank.to_string());
    meta.insert("alpha".to_string(), alpha.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
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

#[cfg(test)]
mod validate_request_tests {
    use super::validate_request;
    use mlx_gen::{NetworkType, TrainingConfig, TrainingItem, TrainingRequest};
    use std::path::PathBuf;

    fn request(items: usize) -> TrainingRequest {
        TrainingRequest {
            items: (0..items)
                .map(|i| TrainingItem {
                    image_path: PathBuf::from(format!("img{i}.png")),
                    caption: "a cat".into(),
                })
                .collect(),
            config: TrainingConfig::default(),
            output_dir: PathBuf::from("/tmp/ltx-trainer-test"),
            file_name: "adapter.safetensors".into(),
            trigger_words: vec![],
            cancel: Default::default(),
        }
    }

    #[test]
    fn accepts_valid_and_rejects_bad_requests() {
        assert!(validate_request(&request(1)).is_ok());
        assert!(validate_request(&request(0)).is_err()); // empty dataset

        let mut r = request(1);
        r.config.rank = 0;
        assert!(validate_request(&r).is_err()); // zero rank

        let mut r = request(1);
        r.config.network_type = NetworkType::Lokr;
        assert!(validate_request(&r).is_err()); // LoKr is LoRA-only here

        let mut r = request(1);
        r.config.optimizer = "sgd".into();
        assert!(validate_request(&r).is_err()); // unsupported optimizer
    }
}
