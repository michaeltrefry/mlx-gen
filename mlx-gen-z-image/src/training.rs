//! sc-3042 SPIKE — LoRA *training* on the Z-Image DiT, in pure Rust on mlx-rs.
//!
//! This is the gate for epic 3039 (LoRA/LoKr training in Rust). It proves the mechanism the whole
//! epic rests on, on the REAL 30-block Z-Image transformer:
//!
//!   * **Trainable LoRA injection** — the model crates do NOT use mlx-rs's `Module`/`ModuleParameters`
//!     system (hand-rolled `&self` forwards over raw `Array`s, `src/adapters.rs:6`), so training uses
//!     the *functional* autograd: the trainable factors live OUTSIDE the model in a
//!     `LoraParams`, and each step they are re-injected into the target [`AdaptableLinear`]s
//!     as a single `Adapter::Lora` via [`AdaptableLinear::set_adapters`]. The injection mirrors the
//!     inference reload (`adapters::loader::install_lora_groups`) op-for-op — transpose the
//!     `[r,in]`/`[out,r]` factors, fold `alpha/rank` into `b`, `scale = 1` — so the trained adapter
//!     round-trips through the normal inference path bit-for-bit.
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

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use mlx_gen::adapters::{reconstruct_lokr_delta, AdaptableHost, Adapter};
use mlx_gen::array::host_i32;
use mlx_gen::media::Image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::train::checkpoint::checkpoint_filename;
use mlx_gen::train::dataset::{bucket_resolution, center_crop_square};
use mlx_gen::train::schedule::{lr_multiplier, schedule_updates};
use mlx_gen::{
    LoadSpec, Modality, NetworkType, Result, Trainer, TrainerDescriptor, TrainerRegistration,
    TrainingConfig, TrainingOutput, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_rs::error::{Exception, Result as MlxResult};
use mlx_rs::ops::{add, multiply, subtract};
use mlx_rs::optimizers::{clip_grad_norm, AdamW, Optimizer};
use mlx_rs::transforms::{eval, keyed_value_and_grad};
use mlx_rs::{random, Array, Dtype};

use crate::model::MODEL_ID;
use crate::pipeline::{encode_init_latents, slice_valid};
use crate::text_encoder::TextEncoder;
use crate::transformer::ZImageTransformer;
use crate::vae::Vae;

/// The trainable LoRA factor map — keyed `{path}.lora_a` `[r,in]` / `{path}.lora_b` `[out,r]`. The
/// autograd arguments (`keyed_value_and_grad`) and the optimizer-stepped state.
type LoraParams = HashMap<Rc<str>, Array>;

/// One LoRA-trained Linear: its dotted module path (e.g. `layers.0.attention.to_q`) plus the
/// pre-built parameter-map keys and the `[out, in]` dims read off the base weight.
pub struct LoraTarget {
    pub path: String,
    a_key: Rc<str>,
    b_key: Rc<str>,
    pub in_f: i32,
    pub out_f: i32,
}

/// The default Z-Image attention LoRA targets across the main `layers` stack — the suffixes
/// `to_q`/`to_k`/`to_v`/`to_out.0` the SceneWorks torch trainer uses
/// (`DEFAULT_LORA_TARGET_MODULES`, `training_adapters.py:72`).
pub fn attention_targets(n_layers: usize) -> Vec<String> {
    let mut out = Vec::with_capacity(n_layers * 4);
    for i in 0..n_layers {
        for proj in [
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
        ] {
            out.push(format!("layers.{i}.{proj}"));
        }
    }
    out
}

/// A minimal Z-Image LoRA trainer: a frozen base transformer + an external trainable factor map,
/// stepped with `keyed_value_and_grad` + AdamW. Spike-scoped (single in-memory sample at a time);
/// the dataset/scheduling glue is sc-3043.
pub struct ZImageLoraTrainer {
    transformer: ZImageTransformer,
    targets: Vec<LoraTarget>,
    params: LoraParams,
    alpha: f32,
    opt: AdamW,
}

impl ZImageLoraTrainer {
    /// Build a trainer over `target_paths` (dotted module paths into the DiT). LoRA factors are
    /// initialised the Python `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank, in]`, `B = 0`
    /// `[out, rank]` — so the adapter starts as an exact no-op and only learns from the gradient.
    pub fn new(
        transformer: ZImageTransformer,
        target_paths: &[String],
        rank: i32,
        alpha: f32,
        lr: f32,
        seed: u64,
    ) -> Result<Self> {
        let mut transformer = transformer;
        let (targets, params) = build_lora_targets(&mut transformer, target_paths, rank, seed)?;
        Ok(Self {
            transformer,
            targets,
            params,
            alpha,
            opt: AdamW::new(lr),
        })
    }

    pub fn num_targets(&self) -> usize {
        self.targets.len()
    }

    /// Overwrite the optimizer learning rate (LR schedules mutate this between steps — mlx-rs has no
    /// built-in scheduler).
    pub fn set_lr(&mut self, lr: f32) {
        self.opt.lr = Array::from_slice(&[lr], &[1]);
    }

    /// One optimizer step on a single `(clean_latent, cap_feats)` sample at flow-match `sigma`.
    /// `x_t = (1-σ)·x0 + σ·noise`, target `= noise - x0`, `timestep = 1-σ`. Returns the scalar loss.
    pub fn train_step(
        &mut self,
        x0: &Array,
        cap_feats: &Array,
        sigma: f32,
        noise: &Array,
    ) -> Result<f32> {
        let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
        let params_now = self.params.clone();

        // Disjoint field borrows: the loss closure needs `&mut transformer` AND `&targets` at once.
        let transformer = &mut self.transformer;
        let targets: &[LoraTarget] = &self.targets;
        let alpha = self.alpha;
        let capf = cap_feats.clone();

        let (grads, loss) = {
            let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
                install_training_lora(transformer, &p, targets, alpha)?;
                let v = transformer
                    .forward(&x_t, timestep, &capf)
                    .map_err(|e| Exception::custom(e.to_string()))?;
                // MSE — `mean(None)` reduces to a 0-d scalar (grad requires a scalar cotangent).
                Ok(vec![subtract(&v, &target)?.square()?.mean(None)?])
            };
            let mut vg = keyed_value_and_grad(loss_fn);
            let (val, grads) = vg(params_now, 0)?;
            (grads, val[0].item::<f32>())
        };

        // Global-norm clip then AdamW per parameter.
        let (clipped, _norm) = clip_grad_norm(&grads, 1.0)?;
        for (k, g) in clipped.iter() {
            let mut param = self.params[k].clone();
            self.opt.update_single(k, g.as_ref(), &mut param)?;
            self.params.insert(k.clone(), param);
        }
        eval(self.params.values())?;
        Ok(loss)
    }

    /// The flow-match loss at `sigma` for the CURRENT adapter state, with no gradient — the
    /// verification probe (base-vs-trained, and round-trip). `with_adapter=false` evaluates the bare
    /// frozen base (adapters cleared) to measure the LoRA's effect.
    pub fn eval_loss(
        &mut self,
        x0: &Array,
        cap_feats: &Array,
        sigma: f32,
        noise: &Array,
        with_adapter: bool,
    ) -> Result<f32> {
        let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
        if with_adapter {
            let params = self.params.clone();
            install_training_lora(&mut self.transformer, &params, &self.targets, self.alpha)?;
        } else {
            clear_lora(&mut self.transformer, &self.targets);
        }
        let v = self.transformer.forward(&x_t, timestep, cap_feats)?;
        let loss = subtract(&v, &target)?.square()?.mean(None)?;
        Ok(loss.item::<f32>())
    }

    /// Round-trip proof: clear the trainable injection, reload `adapter_path` through the REAL
    /// inference path ([`crate::apply_z_image_adapters`]) onto this same frozen base, and re-measure
    /// the flow-match loss at `sigma`. Should reproduce [`eval_loss`](Self::eval_loss)`(…, true)`
    /// bit-for-bit — the trainer injects the SAME `(transpose, alpha/rank fold, scale=1)` the loader
    /// applies. Restores the cleared state afterwards (a bare base). Uses one transformer (no second
    /// multi-GB load).
    pub fn roundtrip_eval(
        &mut self,
        adapter_path: impl AsRef<Path>,
        x0: &Array,
        cap_feats: &Array,
        sigma: f32,
        noise: &Array,
    ) -> Result<f32> {
        clear_lora(&mut self.transformer, &self.targets);
        let spec = mlx_gen::AdapterSpec {
            path: adapter_path.as_ref().to_path_buf(),
            scale: 1.0,
            kind: mlx_gen::AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        };
        crate::apply_z_image_adapters(&mut self.transformer, std::slice::from_ref(&spec))?;
        let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
        let v = self.transformer.forward(&x_t, timestep, cap_feats)?;
        let loss = subtract(&v, &target)?.square()?.mean(None)?;
        clear_lora(&mut self.transformer, &self.targets);
        Ok(loss.item::<f32>())
    }

    /// Write the trained adapter as PEFT-format safetensors — `{path}.lora_A.weight` `[r,in]`,
    /// `{path}.lora_B.weight` `[out,r]`, scalar `{path}.alpha` — reloadable by
    /// [`crate::apply_z_image_adapters`]. Metadata records the network type/rank/alpha (the epic-2193
    /// reload contract; LoKr will also need `decomposeFactor`).
    pub fn save_peft(&self, path: impl AsRef<Path>, rank: i32) -> Result<()> {
        save_lora_peft(&self.params, &self.targets, self.alpha, rank as u32, path)
    }
}

/// `(x_t, target, timestep)` for a single sample at flow-match `sigma`:
/// `x_t = (1-σ)·x0 + σ·noise`, `target = noise - x0`, `timestep = 1-σ`.
fn build_batch(x0: &Array, noise: &Array, sigma: f32) -> Result<(Array, Array, f32)> {
    let one_minus = Array::from_slice(&[1.0 - sigma], &[1]);
    let s = Array::from_slice(&[sigma], &[1]);
    let x_t = mlx_rs::ops::add(&multiply(x0, &one_minus)?, &multiply(noise, &s)?)?;
    let target = subtract(noise, x0)?; // velocity for the already-negated forward output
    Ok((x_t, target, 1.0 - sigma))
}

/// Inject the current trainable factors as one `Adapter::Lora` per target — EXACTLY as the inference
/// reload (`install_lora_groups`): transpose `[r,in]`→`[in,r]` and `[out,r]`→`[r,out]`, fold
/// `alpha/rank` into `b`, `scale = 1`. Differentiable (the transposes/fold are traced).
fn install_training_lora(
    transformer: &mut ZImageTransformer,
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
) -> MlxResult<()> {
    for t in targets {
        let a = params[&t.a_key].t(); // [r,in] -> [in,r]
        let b_t = params[&t.b_key].t(); // [out,r] -> [r,out]
        let rank = a.shape()[1] as f32;
        let b = b_t.multiply(Array::from_slice(&[alpha / rank], &[1]))?;
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = AdaptableHost::adaptable_mut(transformer, &segs)
            .ok_or_else(|| Exception::custom(format!("LoRA target not found: {}", t.path)))?;
        lin.set_adapters(vec![Adapter::Lora { a, b, scale: 1.0 }]);
    }
    Ok(())
}

/// Clear every target's adapter stack (back to the bare frozen base).
fn clear_lora(transformer: &mut ZImageTransformer, targets: &[LoraTarget]) {
    for t in targets {
        let segs: Vec<&str> = t.path.split('.').collect();
        if let Some(lin) = AdaptableHost::adaptable_mut(transformer, &segs) {
            lin.set_adapters(Vec::new());
        }
    }
}

/// Resolve each `target_paths` entry on the DiT, read its `[out,in]` dims, and initialise the
/// trainable LoRA factors the Python `_MlxLoRALinear` way — `A ~ N(0, 0.02)` `[rank,in]`,
/// `B = 0` `[out,rank]` — keyed `{path}.lora_a` / `{path}.lora_b`. Shared by the spike trainer and
/// the production [`ZImageTurboTrainer`].
fn build_lora_targets(
    transformer: &mut ZImageTransformer,
    target_paths: &[String],
    rank: i32,
    seed: u64,
) -> Result<(Vec<LoraTarget>, LoraParams)> {
    let mut targets = Vec::with_capacity(target_paths.len());
    let mut params: LoraParams = HashMap::new();
    for (i, path) in target_paths.iter().enumerate() {
        let segs: Vec<&str> = path.split('.').collect();
        let lin =
            AdaptableHost::adaptable_mut(transformer, &segs).ok_or_else(|| -> mlx_gen::Error {
                format!("LoRA target does not resolve on the Z-Image DiT: {path}").into()
            })?;
        let shape = lin.base_shape(); // [out, in]
        let (out_f, in_f) = (shape[0], shape[1]);

        let a_key: Rc<str> = Rc::from(format!("{path}.lora_a"));
        let b_key: Rc<str> = Rc::from(format!("{path}.lora_b"));
        // Distinct subkey per target so the RNG init differs per layer.
        let ka = random::key(seed.wrapping_add(2 * i as u64 + 1))?;
        let a = multiply(
            &random::normal::<f32>(&[rank, in_f], None, None, Some(&ka))?,
            Array::from_slice(&[0.02f32], &[1]),
        )?;
        let b = Array::zeros::<f32>(&[out_f, rank])?;
        eval([&a, &b])?;
        params.insert(a_key.clone(), a);
        params.insert(b_key.clone(), b);
        targets.push(LoraTarget {
            path: path.clone(),
            a_key,
            b_key,
            in_f,
            out_f,
        });
    }
    Ok((targets, params))
}

/// Write the trainable factors as PEFT-format safetensors — `{path}.lora_A.weight` `[r,in]`,
/// `{path}.lora_B.weight` `[out,r]`, scalar `{path}.alpha` — reloadable by
/// [`crate::apply_z_image_adapters`]. Metadata records the network type/rank/alpha (the epic-2193
/// reload contract). Shared by the spike trainer and [`ZImageTurboTrainer`].
fn save_lora_peft(
    params: &LoraParams,
    targets: &[LoraTarget],
    alpha: f32,
    rank: u32,
    path: impl AsRef<Path>,
) -> Result<()> {
    let alphas: Vec<(String, Array)> = targets
        .iter()
        .map(|t| {
            (
                format!("{}.alpha", t.path),
                Array::from_slice(&[alpha], &[1]),
            )
        })
        .collect();
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        entries.push((format!("{}.lora_A.weight", t.path), &params[&t.a_key]));
        entries.push((format!("{}.lora_B.weight", t.path), &params[&t.b_key]));
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

// ===========================================================================================
// sc-3044: the production `Trainer` impl — realizes the sc-3043 contract on Z-Image, end to end.
// ===========================================================================================

/// LoRA trainer for Z-Image-Turbo, implementing the core [`Trainer`] surface: a frozen base model
/// (transformer + VAE + text encoder + tokenizer) that caches a captioned image dataset to
/// VAE-latents/prompt-embeds, then runs the spike's functional-autograd LoRA loop with the sc-3043
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

inventory::submit! {
    TrainerRegistration { descriptor: trainer_descriptor, load: load_trainer }
}

impl ZImageTurboTrainer {
    /// Prompt → `cap_feats` (f32): the inference `encode_prompt` path (tokenize with the Qwen chat
    /// template, run the text encoder, slice off the padded tail).
    fn encode_prompt(&self, prompt: &str) -> Result<Array> {
        let t = self.tokenizer.tokenize(prompt)?;
        if t.input_ids.shape()[1] == 0 {
            return Err(mlx_gen::Error::Msg(
                "z_image_turbo trainer: empty caption".into(),
            ));
        }
        let num_valid: i32 = host_i32(&t.attention_mask)?.iter().sum();
        if num_valid == 0 {
            return Err(mlx_gen::Error::Msg(
                "z_image_turbo trainer: empty caption".into(),
            ));
        }
        let enc = self.text_encoder.forward(&t.input_ids, &t.attention_mask)?;
        slice_valid(&enc, num_valid)
    }
}

impl Trainer for ZImageTurboTrainer {
    fn descriptor(&self) -> &TrainerDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &TrainingRequest) -> Result<()> {
        if req.items.is_empty() {
            return Err("z_image_turbo trainer: dataset is empty".into());
        }
        if req.config.rank == 0 {
            return Err("z_image_turbo trainer: rank must be > 0".into());
        }
        let opt = req.config.optimizer.to_ascii_lowercase();
        if opt != "adamw" && opt != "adam" {
            return Err(format!(
                "z_image_turbo trainer: optimizer '{}' is not available on MLX (only adamw/adam; \
                 Prodigy/Rose tracked as sc-3048)",
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
            let cap = self.encode_prompt(&item.caption)?;
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
        // AdamW with wd=0 is identical to Adam, so the one optimizer covers both choices.
        let weight_decay = if cfg.optimizer.eq_ignore_ascii_case("adam") {
            0.0
        } else {
            cfg.weight_decay
        };
        let mut opt = AdamW::new(cfg.learning_rate);
        opt.weight_decay = Array::from_slice(&[weight_decay], &[1]);

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
            )?;
            last_loss = loss;
            steps_run = step;
            accumulate_grads(&mut accumulated, grads)?;

            if step % accum == 0 || step == cfg.steps {
                let lr = cfg.learning_rate
                    * lr_multiplier(cfg.lr_scheduler, update_idx, total_updates, warmup_updates);
                opt.lr = Array::from_slice(&[lr], &[1]);
                let avg = average_grads(
                    accumulated
                        .take()
                        .expect("an update fires only after accumulation"),
                    accum,
                )?;
                let (clipped, _norm) = clip_grad_norm(&avg, 1.0)?;
                for (k, g) in clipped.iter() {
                    let mut p = params[k].clone();
                    opt.update_single(k, g.as_ref(), &mut p)?;
                    params.insert(k.clone(), p);
                }
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
                adapter.save(&params, alpha, rank, cfg.decompose_factor, &ckpt)?;
                on_progress(TrainingProgress::Checkpoint { step });
            }
        }

        // --- save final adapter ---
        on_progress(TrainingProgress::Saving);
        std::fs::create_dir_all(&req.output_dir)?;
        let adapter_path = req.output_dir.join(&req.file_name);
        adapter.save(&params, alpha, rank, cfg.decompose_factor, &adapter_path)?;
        Ok(TrainingOutput {
            adapter_path,
            steps: steps_run,
            final_loss: last_loss,
        })
    }
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
) -> Result<(f32, LoraParams)> {
    let (x_t, target, timestep) = build_batch(x0, noise, sigma)?;
    let capf = cap.clone();
    let loss_fn = move |p: LoraParams, _: i32| -> MlxResult<Vec<Array>> {
        adapter.install(transformer, &p, alpha, rank)?;
        let v = transformer
            .forward(&x_t, timestep, &capf)
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

/// Sum `grads` into the running accumulator (gradient accumulation across micro-steps).
fn accumulate_grads(acc: &mut Option<LoraParams>, grads: LoraParams) -> Result<()> {
    match acc {
        None => *acc = Some(grads),
        Some(a) => {
            for (k, g) in grads {
                let entry = a
                    .get(&k)
                    .ok_or_else(|| mlx_gen::Error::Msg(format!("grad key {k} vanished")))?;
                let summed = add(entry, &g)?;
                a.insert(k, summed);
            }
        }
    }
    Ok(())
}

/// Divide accumulated gradients by `accum` (the mean over the accumulation window).
fn average_grads(grads: LoraParams, accum: u32) -> Result<LoraParams> {
    if accum <= 1 {
        return Ok(grads);
    }
    let inv = Array::from_slice(&[1.0 / accum as f32], &[1]);
    let mut out = HashMap::with_capacity(grads.len());
    for (k, g) in grads {
        out.insert(k, multiply(&g, &inv)?);
    }
    Ok(out)
}

// ===========================================================================================
// LoKr (LyCORIS Kronecker-product adapter) training.
// ===========================================================================================

/// One LoKr-trained Linear: its path, the base `[out,in]` shape, and the factor-map keys —
/// `lokr_w1` always, then either full `lokr_w2` or low-rank `lokr_w2_a`/`lokr_w2_b`.
struct LokrTarget {
    path: String,
    base_shape: Vec<i32>,
    w1_key: Rc<str>,
    w2_key: Option<Rc<str>>,
    w2a_key: Option<Rc<str>>,
    w2b_key: Option<Rc<str>>,
}

/// LyCORIS dimension factorization: split `dimension` into `(a, b)`, `a*b = dimension`, `a <= b`,
/// with `a` as close to `factor` (or balanced/√dimension when `factor < 0`) as a divisor allows.
/// The LoKr weight `[out,in]` then factors as `kron(w1, w2)` with `w1 = [fac(out).0, fac(in).0]`,
/// `w2 = [fac(out).1, fac(in).1]`. Port of LyCORIS `factorization`.
fn factorization(dimension: i32, factor: i32) -> (i32, i32) {
    if factor > 0 && dimension % factor == 0 {
        let n = dimension / factor;
        return if factor > n { (n, factor) } else { (factor, n) };
    }
    let factor = if factor < 0 { dimension } else { factor };
    let (mut m, mut n) = (1i32, dimension);
    let mut length = m + n;
    while m < n {
        let mut new_m = m + 1;
        while dimension % new_m != 0 {
            new_m += 1;
        }
        let new_n = dimension / new_m;
        if new_m + new_n > length || new_m > factor {
            break;
        }
        m = new_m;
        n = new_n;
        length = m + n;
    }
    if m > n {
        (n, m)
    } else {
        (m, n)
    }
}

/// Initialise trainable LoKr factors per target. The weight `[out,in]` factors as
/// `kron(w1[out_a,in_a], w2[out_b,in_b])`; `w2` is low-ranked to `rank` when `rank < min(out_b,in_b)`.
/// `w1 ~ N(0,0.02)`; the SECOND factor is zero-initialised (`w2` full, or `w2_b` low-rank) so the
/// initial delta is exactly 0 (the LoKr analog of LoRA's `B = 0`). `factor` is the decompose knob
/// (`-1` = balanced/auto).
fn build_lokr_targets(
    transformer: &mut ZImageTransformer,
    target_paths: &[String],
    rank: i32,
    factor: i32,
    seed: u64,
) -> Result<(Vec<LokrTarget>, LoraParams)> {
    let mut targets = Vec::with_capacity(target_paths.len());
    let mut params = LoraParams::new();
    let small = Array::from_slice(&[0.02f32], &[1]);
    for (i, path) in target_paths.iter().enumerate() {
        let segs: Vec<&str> = path.split('.').collect();
        let lin =
            AdaptableHost::adaptable_mut(transformer, &segs).ok_or_else(|| -> mlx_gen::Error {
                format!("LoKr target does not resolve on the Z-Image DiT: {path}").into()
            })?;
        let shape = lin.base_shape(); // [out, in]
        let (out_a, out_b) = factorization(shape[0], factor);
        let (in_a, in_b) = factorization(shape[1], factor);

        let k1 = random::key(seed.wrapping_add(7 * i as u64 + 1))?;
        let w1 = multiply(
            &random::normal::<f32>(&[out_a, in_a], None, None, Some(&k1))?,
            &small,
        )?;
        let w1_key: Rc<str> = Rc::from(format!("{path}.lokr_w1"));
        eval([&w1])?;
        params.insert(w1_key.clone(), w1);

        let (mut w2_key, mut w2a_key, mut w2b_key) = (None, None, None);
        if rank > 0 && rank < out_b.min(in_b) {
            // Low-rank w2 = w2_a @ w2_b; w2_b zero-init → factor2 starts at 0.
            let k2 = random::key(seed.wrapping_add(7 * i as u64 + 3))?;
            let w2a = multiply(
                &random::normal::<f32>(&[out_b, rank], None, None, Some(&k2))?,
                &small,
            )?;
            let w2b = Array::zeros::<f32>(&[rank, in_b])?;
            let ak: Rc<str> = Rc::from(format!("{path}.lokr_w2_a"));
            let bk: Rc<str> = Rc::from(format!("{path}.lokr_w2_b"));
            eval([&w2a, &w2b])?;
            params.insert(ak.clone(), w2a);
            params.insert(bk.clone(), w2b);
            w2a_key = Some(ak);
            w2b_key = Some(bk);
        } else {
            // Full w2, zero-init.
            let w2 = Array::zeros::<f32>(&[out_b, in_b])?;
            let wk: Rc<str> = Rc::from(format!("{path}.lokr_w2"));
            eval([&w2])?;
            params.insert(wk.clone(), w2);
            w2_key = Some(wk);
        }
        targets.push(LokrTarget {
            path: path.clone(),
            base_shape: shape,
            w1_key,
            w2_key,
            w2a_key,
            w2b_key,
        });
    }
    Ok((targets, params))
}

/// Inject each target's LoKr delta — reconstructed from the trainable factors EXACTLY as the
/// inference loader (`reconstruct_lokr_delta` at bf16; `Adapter::Lokr` residual `x·ΔWᵀ`) — so the
/// trained adapter round-trips bit-for-bit. Differentiable (kron/matmul/cast are traced).
fn install_training_lokr(
    transformer: &mut ZImageTransformer,
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
) -> MlxResult<()> {
    for t in targets {
        let w1 = params.get(t.w1_key.as_ref());
        let w2 = t.w2_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let w2a = t.w2a_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let w2b = t.w2b_key.as_ref().and_then(|k| params.get(k.as_ref()));
        let delta = reconstruct_lokr_delta(
            alpha,
            rank,
            &t.base_shape,
            w1,
            None,
            None,
            w2,
            w2a,
            w2b,
            Dtype::Bfloat16,
        )
        .map_err(|e| Exception::custom(e.to_string()))?;
        let segs: Vec<&str> = t.path.split('.').collect();
        let lin = AdaptableHost::adaptable_mut(transformer, &segs)
            .ok_or_else(|| Exception::custom(format!("LoKr target not found: {}", t.path)))?;
        lin.set_adapters(vec![Adapter::Lokr { delta, scale: 1.0 }]);
    }
    Ok(())
}

/// Write the trainable LoKr factors as safetensors — `{path}.lokr_w1` + (`lokr_w2` | `lokr_w2_a` +
/// `lokr_w2_b`) — with `networkType=lokr` + `rank`/`alpha`/`decomposeFactor` metadata (the epic-2193
/// reload contract). `parse_lokr` reconstructs the delta from these shapes.
fn save_lokr(
    params: &LoraParams,
    targets: &[LokrTarget],
    alpha: f32,
    rank: f32,
    decompose_factor: i32,
    path: &Path,
) -> Result<()> {
    let mut entries: Vec<(String, &Array)> = Vec::with_capacity(targets.len() * 3);
    for t in targets {
        let keys = [
            Some(&t.w1_key),
            t.w2_key.as_ref(),
            t.w2a_key.as_ref(),
            t.w2b_key.as_ref(),
        ];
        for key in keys.into_iter().flatten() {
            entries.push((key.to_string(), &params[key.as_ref()]));
        }
    }
    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("networkType".to_string(), "lokr".to_string());
    meta.insert("rank".to_string(), (rank as i64).to_string());
    meta.insert("alpha".to_string(), (alpha as i64).to_string());
    meta.insert("decomposeFactor".to_string(), decompose_factor.to_string());
    Array::save_safetensors(entries, Some(&meta), path)?;
    Ok(())
}

/// The trainable adapter kind — dispatches the per-step inject and the save the train loop calls,
/// so one loop drives both LoRA and LoKr.
enum TrainAdapter {
    Lora { targets: Vec<LoraTarget> },
    Lokr { targets: Vec<LokrTarget> },
}

impl TrainAdapter {
    fn install(
        &self,
        transformer: &mut ZImageTransformer,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
    ) -> MlxResult<()> {
        match self {
            TrainAdapter::Lora { targets } => {
                install_training_lora(transformer, params, targets, alpha)
            }
            TrainAdapter::Lokr { targets } => {
                install_training_lokr(transformer, params, targets, alpha, rank)
            }
        }
    }

    fn save(
        &self,
        params: &LoraParams,
        alpha: f32,
        rank: f32,
        decompose_factor: i32,
        path: &Path,
    ) -> Result<()> {
        match self {
            TrainAdapter::Lora { targets } => {
                save_lora_peft(params, targets, alpha, rank as u32, path)
            }
            TrainAdapter::Lokr { targets } => {
                save_lokr(params, targets, alpha, rank, decompose_factor, path)
            }
        }
    }
}
