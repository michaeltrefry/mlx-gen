//! sc-3046 — Wan2.2 LoRA/LoKr **training** on the A14B dual-expert MoE DiT, in pure Rust on mlx-rs.
//! The Rust port of SceneWorks' `WanMoeLoraTrainer` (the dual-expert path) realizing the core
//! [`Trainer`] contract (epic 3039), on the functional-autograd mechanism the spike proved.
//!
//! **Dual-expert MoE.** Wan2.2-A14B is two full transformers — a **high-noise** expert (denoise
//! timesteps `[boundary, 1]`) and a **low-noise** expert (`[0, boundary]`), `boundary = 0.875`. Each
//! gets its **own** LoRA (separate factor map + optimizer + LR schedule). Training **alternates** per
//! micro-step (the reference's `step % 2`): odd steps train the high expert on a timestep sampled in
//! its band, even steps the low expert in its band. Two adapters are saved —
//! `{stem}.high_noise.safetensors` + `{stem}.low_noise.safetensors` — which the inference loader
//! consumes per expert (`AdapterSpec.moe_expert{High,Low}`, sc-2683).
//!
//! **Wan-specific pieces** (the rest is the shared core `train::lora` machinery + sc-3043 runtime):
//!   * **Engine seam.** Wan inference *merges* LoRA into the weight map at load (no forward-time
//!     residual), so `WanTransformer` gained an [`AdaptableHost`] impl (sc-3046) for training: the
//!     trainer injects a fresh `Adapter::Lora` per target each step via
//!     [`AdaptableLinear::set_adapters`](mlx_gen::adapters::AdaptableLinear::set_adapters) and the
//!     block forward applies it.
//!   * **Flow-match target = `noise - clean`** (the raw transformer output is the velocity, no
//!     negation — opposite to Z-Image), **timestep = `t·1000`** (`t∈[0,1]` scaled, NOT `1-σ`), `t`
//!     sampled per-expert *within its noise band*.
//!   * **3D-VAE latents** — a still image VAE-encodes (single frame T=1) to the z16 **normalized**
//!     latent `[16, 1, h, w]` (the VAE applies the per-channel mean/std), the space the DiT operates
//!     in. The UMT5 context is embedded **per expert** (each expert has its own `text_embedding`); the
//!     ~11 GB UMT5 encoder is freed after the one-time cache.
//!   * **Native target naming** — `blocks.{i}.self_attn.{q,k,v,o}` / `cross_attn.{q,k,v,o}` (the
//!     reference `to_q/k/v/to_out.0` suffix surface in Wan's native checkpoint naming). Saved bare,
//!     so `apply_wan_adapters`' `normalize_wan_key` resolves them on reload. LoRA + LoKr (LoKr
//!     reconstructs f32, matching the Wan merge path).
//!
//! **sc-3279 — the two Wan siblings, folded into this one trainer.** The mechanism is identical
//! (still-image T=1 flow-match velocity regression, per the reference `_WanLoraBackend`/
//! `_WanMoeLoraBackend`); only the latent space + input channels differ, so the same struct serves
//! all three Wan trainers via three registrations:
//!   * **Dense TI2V-5B** (`wan2_2_ti2v_5b`, single expert) — the z48 [`Wan22Vae`] (channels-last
//!     `[1,1,H,W,3]` encode, transposed to channels-first `[48,1,h,w]`), in_dim 48.
//!   * **I2V-A14B** (`wan2_2_i2v_14b`, dual expert, boundary 0.900) — the same z16 VAE, but the
//!     forward needs in_dim 36, so the 16-channel noisy latent is concatenated with a **zero** 20-
//!     channel `y` (the reference trains the no-conditioning T2V velocity objective; the attention
//!     LoRA is the trained surface and is blind to the padded conditioning channels).

use std::path::Path;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::media::Image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::lora::{
    accumulate_grads, average_grads, build_lokr_targets, build_lora_targets, LoraParams,
    TrainAdapter,
};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::weights::Weights;
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Result, TrainOptimizer, Trainer, TrainerDescriptor,
    TrainerRegistration, TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest,
    WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::optimizers::clip_grad_norm;
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::config::WanModelConfig;
use crate::model::{MODEL_ID as MODEL_ID_TI2V_5B, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};
use crate::pipeline::{preprocess_i2v_image, preprocess_ti2v_image};
use crate::text_encoder::{load_tokenizer, Umt5Encoder};
use crate::transformer::WanTransformer;
use crate::vae::WanVae;
use crate::vae22::Wan22Vae;

/// Wan reconstructs its LoKr delta at **f32** (the f32 merge path, `merge_one_lokr`); training matches
/// so the adapter round-trips.
const LOKR_DTYPE: Dtype = Dtype::Float32;

/// The reference attention LoRA targets `to_q/to_k/to_v/to_out.0` in Wan's **native** naming: the
/// self/cross-attention `q/k/v/o`. Suffix-matched against the per-block adaptable surface.
const DEFAULT_TARGET_SUFFIXES: [&str; 4] = ["q", "k", "v", "o"];

/// The VAE the trainer encodes its dataset through — the family-specific latent space the DiT
/// regresses velocity in. **z16** 2.1 [`WanVae`] for the 14B T2V/I2V experts; **z48** [`Wan22Vae`]
/// for the dense TI2V-5B (sc-3279). Both produce a normalized **channels-first** clean latent
/// `[z, 1, h, w]` (a still image is one latent frame, `T = 1`).
enum WanTrainVae {
    Z16(WanVae),
    Z48(Wan22Vae),
}

impl WanTrainVae {
    /// Encode a center-cropped square still image → the normalized channels-first clean latent
    /// `[z, 1, h, w]`, exactly as each model's inference encode: the z16 video encode squeezed to
    /// one frame; the z48 channels-last image encode transposed channels-first (`model.rs`'s TI2V
    /// conditioning, `transpose([3,0,1,2])`).
    fn encode_clean(&self, img: &Image, edge: u32) -> Result<Array> {
        match self {
            WanTrainVae::Z16(vae) => {
                let chw = preprocess_i2v_image(img, edge, edge)?; // [3, H, W] in [-1,1]
                let nct_hw = chw.reshape(&[1, 3, 1, edge as i32, edge as i32])?; // [1,3,1,H,W]
                let latent = vae.encode(&nct_hw)?; // [1,16,1,h,w] normalized
                let s = latent.shape();
                Ok(latent.reshape(&[s[1], s[2], s[3], s[4]])?) // [16,1,h,w]
            }
            WanTrainVae::Z48(vae) => {
                let thwc = preprocess_ti2v_image(img, edge, edge)?; // [1,1,H,W,3] channels-last
                let latent = vae.encode(&thwc)?; // [1,1,h,w,48] channels-last normalized
                                                 // [1,1,h,w,48] → [1,h,w,48] → [48,1,h,w] (channels-first, the latent convention).
                Ok(latent
                    .reshape(&latent.shape()[1..])?
                    .transpose_axes(&[3, 0, 1, 2])?)
            }
        }
    }
}

/// Per-expert training state: its save suffix + noise band, plus its own adapter / factor map /
/// optimizer / grad accumulator / LR-schedule bookkeeping (each expert trains independently on the
/// micro-steps routed to it).
struct ExpertState {
    suffix: &'static str, // "" for dense, "high_noise"/"low_noise" for the MoE files
    band: (f32, f32),     // the timestep band this expert is sampled in
    adapter: TrainAdapter,
    params: LoraParams,
    opt: TrainOptimizer,
    accumulated: Option<LoraParams>,
    micro: u32,      // micro-steps routed to this expert so far (drives accumulation)
    update_idx: u32, // optimizer updates applied (drives the LR schedule)
    total_updates: u32,
    warmup_updates: u32,
}

/// Dual-expert (A14B MoE) Wan2.2 LoRA/LoKr trainer. Loads both experts + the z16 VAE + UMT5 + the
/// tokenizer, caches a captioned image dataset to (normalized latent, per-expert context) pairs, then
/// runs the alternating per-expert functional-autograd flow-match loop with the sc-3043 runtime glue,
/// and writes one adapter per expert that round-trips through `apply_wan_adapters`.
pub struct WanMoeTrainer {
    descriptor: TrainerDescriptor,
    tokenizer: Option<TextTokenizer>,
    text_encoder: Option<Umt5Encoder>,
    vae: WanTrainVae,
    /// `[low, high]` for the dual-expert MoE; `[single]` for a dense checkpoint.
    experts: Vec<WanTransformer>,
    cfg: WanModelConfig,
}

fn trainer_descriptor(id: &'static str) -> TrainerDescriptor {
    TrainerDescriptor {
        id,
        family: "wan",
        modality: Modality::Video,
        supports_lora: true,
        supports_lokr: true,
    }
}

/// Dual-expert MoE T2V-A14B trainer (z16 VAE, in_dim 16) — sc-3046.
fn descriptor_t2v_14b() -> TrainerDescriptor {
    trainer_descriptor(MODEL_ID_T2V_14B)
}
/// Dual-expert MoE I2V-A14B trainer (z16 VAE, channel-concat in_dim 36, boundary 0.900) — sc-3279.
fn descriptor_i2v_14b() -> TrainerDescriptor {
    trainer_descriptor(MODEL_ID_I2V_14B)
}
/// Dense TI2V-5B trainer (z48 vae22, in_dim 48, single expert) — sc-3279.
fn descriptor_ti2v_5b() -> TrainerDescriptor {
    trainer_descriptor(MODEL_ID_TI2V_5B)
}

/// Construct the Wan trainer from a converted MLX snapshot directory, picking the VAE (z16 vs z48)
/// and expert layout (dual `low_noise_model`/`high_noise_model` vs dense `model.safetensors`) from
/// the checkpoint's `config.json`. The transformers load bf16 (Wan's native dtype; the trainable f32
/// factors promote against the bf16 base — clean autograd, the base frozen). `descriptor.id` selects
/// which Wan variant this registration serves, and is checked against the config.
fn build_trainer(spec: &LoadSpec, descriptor: TrainerDescriptor) -> Result<Box<dyn Trainer>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(mlx_gen::Error::Msg(format!(
                "{} trainer expects a converted snapshot directory (model/expert safetensors + \
                 t5_encoder + vae + tokenizer.json), not a single file",
                descriptor.id
            )))
        }
    };
    let cfg = WanModelConfig::from_model_dir(root)?;
    check_config_matches(descriptor.id, &cfg)?;
    let tokenizer = load_tokenizer(root.join("tokenizer.json"), cfg.text_len)?;
    let t5_w = Weights::from_file(root.join("t5_encoder.safetensors"))?;
    let text_encoder = Umt5Encoder::from_weights(&t5_w, &cfg)?;
    let vae_w = Weights::from_file(root.join("vae.safetensors"))?;
    // The 5B operates in the z48 vae22 latent space; the 14B T2V/I2V experts in the z16 2.1 VAE.
    let vae = if cfg.vae_z_dim == 48 {
        WanTrainVae::Z48(Wan22Vae::from_weights(&vae_w)?)
    } else {
        WanTrainVae::Z16(WanVae::from_weights(&vae_w)?)
    };

    let experts = if cfg.dual_model {
        let low_w = Weights::from_file(root.join("low_noise_model.safetensors"))?;
        let high_w = Weights::from_file(root.join("high_noise_model.safetensors"))?;
        vec![
            WanTransformer::from_weights(&low_w, &cfg)?,
            WanTransformer::from_weights(&high_w, &cfg)?,
        ]
    } else {
        let w = Weights::from_file(root.join("model.safetensors"))?;
        vec![WanTransformer::from_weights(&w, &cfg)?]
    };

    Ok(Box::new(WanMoeTrainer {
        descriptor,
        tokenizer: Some(tokenizer),
        text_encoder: Some(text_encoder),
        vae,
        experts,
        cfg,
    }))
}

/// Reject a snapshot whose `config.json` doesn't match the registration id (a dense 5B routed to the
/// MoE id, or vice-versa) before the expensive weight load — the same shape checks the `model.rs`
/// generators apply.
fn check_config_matches(id: &str, cfg: &WanModelConfig) -> Result<()> {
    let ok = match id {
        _ if id == MODEL_ID_TI2V_5B => !cfg.dual_model && cfg.vae_z_dim == 48,
        _ if id == MODEL_ID_I2V_14B => cfg.dual_model && cfg.is_i2v_concat(),
        _ if id == MODEL_ID_T2V_14B => cfg.dual_model && !cfg.is_i2v_concat(),
        _ => true,
    };
    if !ok {
        return Err(mlx_gen::Error::Msg(format!(
            "{id} trainer: config.json does not match (model_type={}, in_dim={}, dual_model={}, \
             vae_z_dim={})",
            cfg.model_type, cfg.in_dim, cfg.dual_model, cfg.vae_z_dim
        )));
    }
    Ok(())
}

/// Construct the dual-expert MoE T2V-A14B trainer (sc-3046).
pub fn load_trainer(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    build_trainer(spec, descriptor_t2v_14b())
}
/// Construct the dual-expert MoE I2V-A14B trainer (sc-3279) — same MoE machinery, channel-concat
/// in_dim 36 (the 16-channel latent is zero-`y`-padded to 36 each step).
pub fn load_trainer_i2v_14b(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    build_trainer(spec, descriptor_i2v_14b())
}
/// Construct the dense TI2V-5B trainer (sc-3279) — z48 vae22, single expert.
pub fn load_trainer_ti2v_5b(spec: &LoadSpec) -> Result<Box<dyn Trainer>> {
    build_trainer(spec, descriptor_ti2v_5b())
}

inventory::submit! {
    TrainerRegistration { descriptor: descriptor_t2v_14b, load: load_trainer }
}
inventory::submit! {
    TrainerRegistration { descriptor: descriptor_i2v_14b, load: load_trainer_i2v_14b }
}
inventory::submit! {
    TrainerRegistration { descriptor: descriptor_ti2v_5b, load: load_trainer_ti2v_5b }
}

impl Trainer for WanMoeTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> Result<()> {
        let id = self.descriptor.id;
        if req.items.is_empty() {
            return Err(format!("{id} trainer: dataset is empty").into());
        }
        if req.config.rank == 0 {
            return Err(format!("{id} trainer: rank must be > 0").into());
        }
        if !TrainOptimizer::is_supported(&req.config.optimizer) {
            return Err(format!(
                "{id} trainer: optimizer '{}' is not available on MLX training (supported: \
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
        let id = self.descriptor.id;
        let cfg = &req.config;
        let n_experts = self.experts.len();
        let dual = n_experts == 2;
        let boundary = self.cfg.boundary;
        // Channel-concat conditioning width (I2V-A14B: in_dim 36 = latent 16 + y 20; 0 for T2V/5B).
        // The forward gets a zero `y` of this width appended to the noisy latent (sc-3279).
        let y_channels = self.cfg.in_dim as i32 - self.cfg.vae_z_dim as i32;
        on_progress(TrainingProgress::Preparing);
        let edge = bucket_resolution(cfg.resolution);

        // --- prepare → load → cache: normalized latents + per-expert UMT5 context (then free the TE) ---
        on_progress(TrainingProgress::LoadingModel);
        let total = req.items.len() as u32;
        let mut cache: Vec<(Array, Vec<Array>)> = Vec::with_capacity(req.items.len());
        {
            let te = self.text_encoder.as_ref().ok_or_else(|| {
                mlx_gen::Error::Msg(format!("{id} trainer: text encoder missing"))
            })?;
            let tok = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| mlx_gen::Error::Msg(format!("{id} trainer: tokenizer missing")))?;
            for (i, item) in req.items.iter().enumerate() {
                if req.cancel.is_cancelled() {
                    break;
                }
                on_progress(TrainingProgress::Caching {
                    current: i as u32 + 1,
                    total,
                });
                let img = center_crop_square(&decode_image(&item.image_path)?);
                // [z,1,h,w] normalized channels-first (z16 14B / z48 5B — dispatched by the VAE kind).
                let clean = self.vae.encode_clean(&img, edge)?;
                let t5_embed = te.encode(tok, &item.caption)?; // [L, text_dim]
                                                               // Each expert has its own text_embedding, so embed the context per expert.
                let mut ctxs = Vec::with_capacity(n_experts);
                for e in &self.experts {
                    ctxs.push(e.embed_text(&t5_embed)?); // [1, text_len, dim]
                }
                let mut to_eval: Vec<&Array> = vec![&clean];
                to_eval.extend(ctxs.iter());
                eval(to_eval)?;
                cache.push((clean, ctxs));
            }
        }
        if cache.is_empty() {
            return Err(format!("{id} trainer: no usable dataset items (all cancelled?)").into());
        }
        // Free the UMT5 encoder + tokenizer (~11 GB) before training (the reference frees it post-cache).
        self.text_encoder = None;
        self.tokenizer = None;

        // --- per-expert adapter targets + factors + optimizer + schedule ---
        let suffixes: Vec<String> = if cfg.lora_target_modules.is_empty() {
            DEFAULT_TARGET_SUFFIXES
                .iter()
                .map(|s| s.to_string())
                .collect()
        } else {
            cfg.lora_target_modules.clone()
        };
        let alpha = cfg.alpha;
        let rank = cfg.rank as f32;
        let mae = {
            let lt = cfg.loss_type.to_ascii_lowercase();
            lt == "mae" || lt == "l1"
        };
        let accum = cfg.gradient_accumulation.max(1);
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };

        let mut states: Vec<ExpertState> = Vec::with_capacity(n_experts);
        for (idx, expert) in self.experts.iter_mut().enumerate() {
            let target_paths = resolve_target_paths(expert, &suffixes);
            if target_paths.is_empty() {
                return Err(format!(
                    "{id} trainer: no LoRA targets resolved (check lora_target_modules)"
                )
                .into());
            }
            // Distinct seed per expert so the two experts' gaussian init differs.
            let seed = cfg.seed.wrapping_add(idx as u64 * 0x9E37_79B9);
            let (adapter, params) = build_adapter(expert, &target_paths, cfg, seed)?;
            let (band, suffix) = if dual {
                if idx == 0 {
                    ((0.0, boundary), "low_noise")
                } else {
                    ((boundary, 1.0), "high_noise")
                }
            } else {
                ((0.0, 1.0), "")
            };
            // This expert sees ~steps/n_experts micro-steps; size its LR schedule accordingly.
            let expert_micro = (cfg.steps / n_experts as u32).max(1);
            let (total_updates, warmup_updates) =
                schedule_updates(expert_micro, accum, cfg.lr_warmup_steps);
            let opt = TrainOptimizer::from_config(&cfg.optimizer, cfg.learning_rate, weight_decay)?;
            states.push(ExpertState {
                suffix,
                band,
                adapter,
                params,
                opt,
                accumulated: None,
                micro: 0,
                update_idx: 0,
                total_updates,
                warmup_updates,
            });
        }

        let stem = Path::new(&req.file_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("lora")
            .to_string();
        let ext = Path::new(&req.file_name)
            .extension()
            .and_then(|s| s.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_else(|| ".safetensors".to_string());

        // --- train loop: alternate experts (high on odd steps, low on even — the reference's step%2) ---
        let mut last_loss = 0.0f32;
        let mut steps_run = 0u32;
        for step in 1..=cfg.steps {
            if req.cancel.is_cancelled() {
                break;
            }
            // Dual: high (index 1) on odd steps, low (index 0) on even. Dense: the single expert.
            let ei = if dual && step % 2 == 1 { 1 } else { 0 };
            let (clean, ctxs) = &cache[((step - 1) as usize) % cache.len()];
            let ctx = &ctxs[ei];
            let band = states[ei].band;
            let t = sample_band_timestep(
                &cfg.timestep_type,
                &cfg.timestep_bias,
                band,
                cfg.seed.wrapping_mul(0x9E37_79B9).wrapping_add(step as u64),
            )?;
            let noise = random::normal::<f32>(
                clean.shape(),
                None,
                None,
                Some(&random::key(
                    cfg.seed.wrapping_add(step as u64).wrapping_mul(2) + 1,
                )?),
            )?;
            let (loss, grads) = compute_loss_grads(
                &mut self.experts[ei],
                &states[ei].adapter,
                &states[ei].params,
                alpha,
                rank,
                clean,
                ctx,
                t,
                &noise,
                mae,
                y_channels,
            )?;
            last_loss = loss;
            steps_run = step;

            let st = &mut states[ei];
            st.micro += 1;
            accumulate_grads(&mut st.accumulated, grads)?;
            // Fire an optimizer update every `accum` micro-steps for THIS expert (or on the final step).
            if st.micro.is_multiple_of(accum) || step == cfg.steps {
                let mult = lr_multiplier(
                    cfg.lr_scheduler,
                    st.update_idx,
                    st.total_updates,
                    st.warmup_updates,
                );
                st.opt.set_lr_scaled(mult);
                let avg = average_grads(
                    st.accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    accum,
                )?;
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                let clipped: LoraParams = clipped
                    .into_iter()
                    .map(|(k, v)| (k, v.into_owned()))
                    .collect();
                st.opt.step(&mut st.params, &clipped)?;
                eval(st.params.values())?;
                st.update_idx += 1;
            }

            on_progress(TrainingProgress::Training {
                step,
                total: cfg.steps,
                loss: last_loss,
            });

            if cfg.save_every > 0 && step % cfg.save_every == 0 && step != cfg.steps {
                save_experts(
                    &states,
                    &req.output_dir,
                    &stem,
                    &ext,
                    Some(step),
                    alpha,
                    rank,
                    cfg,
                )?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // --- save one adapter per expert (the MoE high/low pair, or a single dense file) ---
        on_progress(TrainingProgress::Saving);
        let paths = save_experts(
            &states,
            &req.output_dir,
            &stem,
            &ext,
            None,
            alpha,
            rank,
            cfg,
        )?;
        // The high-noise file is the primary returned path (the worker discovers the low pair).
        let adapter_path = paths
            .iter()
            .zip(&states)
            .find(|(_, st)| st.suffix == "high_noise" || st.suffix.is_empty())
            .map(|(p, _)| p.clone())
            .unwrap_or_else(|| req.output_dir.join(&req.file_name));
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
}

/// Per-expert adapter filename: `{stem}.{suffix}{ext}` (final) or `{stem}-step{step:06}.{suffix}{ext}`
/// (checkpoint, matching the [`checkpoint_filename`](mlx_gen::train::checkpoint::checkpoint_filename)
/// convention), with the `.{suffix}` dropped for the single dense file.
fn expert_filename(stem: &str, ext: &str, suffix: &str, step: Option<u32>) -> String {
    let base = match step {
        Some(s) => format!("{stem}-step{s:06}"),
        None => stem.to_string(),
    };
    if suffix.is_empty() {
        format!("{base}{ext}")
    } else {
        format!("{base}.{suffix}{ext}")
    }
}

/// Write each expert's adapter (the MoE high/low pair, or a single dense file). `step = Some` writes
/// intermediate checkpoints; `None` the final adapter. Returns the written paths (expert order).
#[allow(clippy::too_many_arguments)]
fn save_experts(
    states: &[ExpertState],
    dir: &Path,
    stem: &str,
    ext: &str,
    step: Option<u32>,
    alpha: f32,
    rank: f32,
    cfg: &TrainingConfig,
) -> Result<Vec<std::path::PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let mut paths = Vec::with_capacity(states.len());
    for st in states {
        let path = dir.join(expert_filename(stem, ext, st.suffix, step));
        st.adapter
            .save(&st.params, alpha, rank, cfg.decompose_factor, "", &path)?;
        paths.push(path);
    }
    Ok(paths)
}

/// Resolve the config's target-module suffixes (default native `q/k/v/o` — the reference
/// `to_q/k/v/to_out.0` attention surface) against the expert's per-block adaptable Linears.
fn resolve_target_paths(expert: &WanTransformer, suffixes: &[String]) -> Vec<String> {
    AdaptableHost::adaptable_paths(expert)
        .into_iter()
        .filter(|path| {
            suffixes
                .iter()
                .any(|s| path == s || path.ends_with(&format!(".{s}")))
        })
        .collect()
}

/// Build one expert's trainable adapter (LoRA or LoKr per `cfg.network_type`) over `target_paths`.
fn build_adapter(
    expert: &mut WanTransformer,
    target_paths: &[String],
    cfg: &TrainingConfig,
    seed: u64,
) -> Result<(TrainAdapter, LoraParams)> {
    match cfg.network_type {
        NetworkType::Lora => {
            let (targets, params) =
                build_lora_targets(expert, target_paths, cfg.rank as i32, seed)?;
            Ok((TrainAdapter::Lora { targets }, params))
        }
        NetworkType::Lokr => {
            let (targets, params) = build_lokr_targets(
                expert,
                target_paths,
                cfg.rank as i32,
                cfg.decompose_factor,
                seed,
            )?;
            Ok((TrainAdapter::Lokr { targets }, params))
        }
    }
}

/// One forward+backward over an expert's trainable factors: build the flow-match input `x_t`, inject
/// the factors, run the DiT, regress the raw velocity toward `noise - clean`, return `(loss, grads)`.
/// `y_channels > 0` (I2V-A14B, = 20) appends a **zero** channel-concat conditioning block to the
/// noisy latent so the in_dim-36 forward runs — the reference's no-conditioning T2V objective; the
/// padded channels are a constant input (not differentiated), and the trained surface is attention.
#[allow(clippy::too_many_arguments)]
fn compute_loss_grads(
    expert: &mut WanTransformer,
    adapter: &TrainAdapter,
    params: &LoraParams,
    alpha: f32,
    rank: f32,
    clean: &Array,
    context: &Array,
    t: f32,
    noise: &Array,
    mae: bool,
    y_channels: i32,
) -> Result<(f32, LoraParams)> {
    // x_t = (1-t)·clean + t·noise; target = noise - clean (raw velocity); transformer timestep = t·1000.
    let one_minus = Array::from_slice(&[1.0 - t], &[1]);
    let tt = Array::from_slice(&[t], &[1]);
    let x_t = add(&multiply(clean, &one_minus)?, &multiply(noise, &tt)?)?;
    let target = subtract(noise, clean)?;
    // I2V channel-concat: append a zero `y` `[y_channels, F, h, w]` → in_dim 36 (mirrors `predict`'s
    // `[latents, y]` order). A constant input, built once outside the autograd closure.
    let x_in = if y_channels > 0 {
        let s = x_t.shape(); // [z, F, h, w]
        let zero_y = Array::zeros::<f32>(&[y_channels, s[1], s[2], s[3]])?;
        concatenate_axis(&[&x_t, &zero_y], 0)?
    } else {
        x_t
    };
    let timestep = t * 1000.0;
    let ctx = context.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        adapter.install(expert, &p, alpha, rank, LOKR_DTYPE)?;
        let v = expert
            .forward(&x_in, timestep, &ctx)
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

/// Sample a flow-match timestep within `band = (lo, hi)`: a faithful port of the reference
/// `sample_training_timestep` (`sigmoid(randn)` default, `uniform` for linear, `(uniform+sigmoid)/2`
/// for weighted; bias `high` → `√t`, `low` → `t²`), then scaled into the expert's noise band
/// `lo + t_unit·(hi-lo)`. Deterministic in `seed`.
fn sample_band_timestep(
    timestep_type: &str,
    timestep_bias: &str,
    band: (f32, f32),
    seed: u64,
) -> Result<f32> {
    let k1 = random::key(seed)?;
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
    let ttype = timestep_type.trim().to_ascii_lowercase().replace('-', "_");
    let t_unit = match ttype.as_str() {
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
    let t_unit = match bias.as_str() {
        "high" | "high_noise" | "favor_high_noise" => t_unit.sqrt(),
        "low" | "low_noise" | "favor_low_noise" => t_unit * t_unit,
        _ => t_unit,
    };
    let t_unit = t_unit.clamp(1e-3, 1.0 - 1e-3);
    Ok(band.0 + t_unit * (band.1 - band.0))
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
