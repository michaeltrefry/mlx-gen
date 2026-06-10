//! `Sdxl` — the Stable Diffusion XL implementation of [`mlx_gen::Generator`], plus its
//! [`descriptor`]/[`load`] entry points and the `inventory` registration that wires it into
//! `mlx_gen`'s registry under the id `"sdxl"` (the SceneWorks worker's `payload.model`).
//!
//! SDXL is the in-process Apple `mlx-examples/stable_diffusion` path (vendored at
//! `_vendor/mlx_sd/`) brought into Rust — a **U-Net** generator (conv ResBlocks + spatial/cross
//! attention + time/`text_time` micro-conditioning), dual CLIP text encoders, an SDXL VAE, and a
//! discrete Euler-Ancestral sampler with real classifier-free guidance. Parity target = the
//! vendored fp16 reference path (`StableDiffusionXL.generate_latents`), validated stage-by-stage.
//!
//! Slices land incrementally (sc-2400): this module starts as the contract + capability surface;
//! [`load`] assembles components as each slice (tokenizer → text encoders → U-Net → VAE → sampler)
//! is wired and parity-proven.

use mlx_gen::{
    default_seed, AlphaSchedule, Capabilities, Conditioning, ConditioningKind, DiffusionSampler,
    Error, GenerationOutput, GenerationRequest, Generator, Image, LcmSampler, LightningSampler,
    LoadSpec, Modality, ModelDescriptor, Precision, Progress, Result, TcdSampler, WeightsSource,
};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::Dtype;

use crate::config::DiffusionConfig;
use crate::inpaint::{preprocess_mask, InpaintBlend};
use crate::ip_adapter::IpImageEncoder;
use crate::loader;
use crate::pipeline::{
    decode_image, denoise, denoise_inpaint, denoise_ip, denoise_multi_control, encode_conditioning,
    encode_init_latents, preprocess_control_image, text_time_ids, ControlContext, Denoiser,
};
use crate::sampler::{AncestralEuler, EulerSampler};
use crate::text_encoder::ClipTextEncoder;
use crate::tokenizer::ClipBpeTokenizer;
use crate::unet::{ControlNet, UNet2DConditionModel};
use crate::vae::Autoencoder;

/// img2img default strength (the vendored `generate_latents_from_image` default).
const DEFAULT_STRENGTH: f32 = 0.8;
/// Masked-inpaint / outpaint default strength — the worker's `SdxlDiffusersAdapter` uses 0.85 for
/// `use_inpaint`/`outpaint` (vs 0.6 for a plain edit). An explicit request strength still wins.
const INPAINT_DEFAULT_STRENGTH: f32 = 0.85;
/// Default `ip_adapter_scale` (sc-3059) when a request doesn't override it (the worker's plus-face
/// default ≈ 0.6). In IP mode the `Reference` strength field carries the IP scale.
const IP_DEFAULT_SCALE: f32 = 0.6;

/// SDXL-base-1.0 production defaults (the SceneWorks `MlxSdxlAdapter`): 30 inference steps,
/// CFG 7.0, native 1024². Used when a request omits the corresponding field (consumed by the
/// `generate` pipeline slice, sc-2400 S5).
pub(crate) const DEFAULT_STEPS: u32 = 30;
pub(crate) const DEFAULT_GUIDANCE: f32 = 7.0;

/// The few-step acceleration samplers (sc-2769). Selected per request via `req.sampler`; each is
/// paired with its acceleration LoRA at load (`spec.adapters`) by the caller (the SceneWorks
/// variant manifest, epic 2755) — selecting one without its LoRA loaded yields undertrained noise.
pub(crate) const ACCEL_SAMPLERS: [&str; 3] = ["lcm", "lightning", "hyper"];

/// `original_inference_steps` for the LCM/TCD timestep selection (diffusers' default).
const LCM_ORIGINAL_STEPS: usize = 50;

/// Per-variant few-step defaults `(steps, CFG, TCD eta)`, applied when the request omits `steps`/
/// `guidance`. **Locked by the sc-2758 SDXL acceleration A/B characterization** (re-tuned here per
/// sc-2907; `sdxl` and `realvisxl` came out identical, so the table keys on the sampler only). CFG is
/// 1.0 (off) for all three — Lightning/Hyper are trained CFG-free and LCM-LoRA runs at low/no CFG —
/// which also halves the per-step UNet work. Lightning's step count must match the loaded LoRA
/// (2/4/8); LCM uses a single LoRA at any step count.
fn accel_defaults(sampler: &str) -> (u32, f32, f32) {
    match sampler {
        // LCM is the weakest method and 4 steps is too soft as a default; sc-2758 locks 8 as the
        // quality floor (the LCM-LoRA is step-free, so this is a plain default, not LoRA-bound).
        "lcm" => (8, 1.0, 0.0),
        "lightning" => (4, 1.0, 0.0),
        // Hyper-SD: TCD, deterministic (eta=0) — sc-2758 locked eta=0 for the step-graded
        // (1/2/4/8-step) LoRAs, which is the default LoRA path here.
        "hyper" => (4, 1.0, 0.0),
        _ => (DEFAULT_STEPS, DEFAULT_GUIDANCE, 0.0),
    }
}

/// Registry id — matches the SceneWorks worker's `payload.model` (`MODEL_TARGETS["sdxl"]`).
pub const MODEL_ID: &str = "sdxl";

/// SDXL's identity + capabilities — constructible without loading weights (registry
/// introspection). Capability flags are turned on as each slice lands and is parity-proven, so the
/// descriptor never advertises a path that isn't wired (avoids the false-capability trap —
/// [[false-green-gates-mask-descope]]).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sdxl",
        modality: Modality::Image,
        capabilities: Capabilities {
            // SDXL uses real classifier-free guidance: honors the negative prompt + a CFG scale.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            // img2img Reference (sc-2638) + masked inpaint/outpaint (Mask, sc-3057) + tile-ControlNet
            // detail (Control, sc-3058 — requires a control checkpoint via LoadSpec::control). LoRA
            // (kohya `lora_unet_` + PEFT, sc-2639) and LoKr (sc-2640 — Rust is more capable than the
            // vendored path, which rejects LoKr) are wired.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::Mask,
                ConditioningKind::Control,
            ],
            supports_lora: true,
            supports_lokr: true,
            // `euler_ancestral` is the production default (full-CFG, 30-step); `lcm`/`lightning`/
            // `hyper` are the few-step acceleration samplers (sc-2769), each driven by its diffusers-
            // faithful schedule and paired with an acceleration LoRA at load. A request naming any
            // other sampler is rejected in `validate_request` rather than silently downgraded.
            samplers: vec!["euler_ancestral", "lcm", "lightning", "hyper"],
            schedulers: vec!["discrete"],
            min_size: 512,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded SDXL generator: the dual CLIP encoders + tokenizer, the U-Net, the VAE, the
/// Euler-Ancestral sampler (production default), and the `alphas_cumprod` schedule the few-step
/// acceleration samplers (LCM/Lightning/Hyper) build on — assembled from a snapshot directory.
pub struct Sdxl {
    descriptor: ModelDescriptor,
    tokenizer: ClipBpeTokenizer,
    te1: ClipTextEncoder,
    te2: ClipTextEncoder,
    unet: UNet2DConditionModel,
    /// ControlNet branches (sc-3058; MultiControlNet sc-3378), loaded from `LoadSpec::control` +
    /// `LoadSpec::extra_controls`. Empty when no control checkpoint was supplied. `generate` requires
    /// exactly one `Control` conditioning per loaded branch (paired by order); their residuals are
    /// summed (the diffusers `MultiControlNetModel` rule).
    controls: Vec<ControlNet>,
    /// Optional IP-Adapter image-token source (sc-3059), loaded from `LoadSpec::ip_adapter`. When
    /// present, the model is in "IP mode": a `Reference` conditioning is the image prompt (txt2img +
    /// IP), not an img2img init. The decoupled-attn K/V projections are installed into `unet`.
    ip_adapter: Option<IpImageEncoder>,
    vae: Autoencoder,
    sampler: EulerSampler,
    /// DDPM `alphas_cumprod` from the SDXL `scaled_linear` betas — shared by the acceleration
    /// samplers (sc-2769). Built once at load (the ancestral `sampler` keeps its own σ table).
    alpha_schedule: AlphaSchedule,
}

/// Construct an [`Sdxl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] pointing at a
/// `stabilityai/stable-diffusion-xl-base-1.0` snapshot (the diffusers multi-component tree —
/// `tokenizer/`, `tokenizer_2/`, `text_encoder/`, `text_encoder_2/`, `unet/`, `vae/`).
///
/// **Dtype:** the U-Net + both CLIP text encoders run **fp16**, matching the production reference
/// (`StableDiffusionXL(float16=True)`); the **VAE stays f32** (the vendored always loads the
/// autoencoder f32 — the SDXL VAE is fp16-unstable). The whole fp16 path is byte-identical to the
/// reference on MLX 0.31.2 (sc-2721; needs sc-2772's NAX 16-bit fix + the compiled `gelu_exact`).
/// The lower-level `load_unet`/`load_text_encoder_*` keep an f32 path for the tight stage gates.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        // `Precision::Bf16` is the registry's dense sentinel; the dense path runs fp16 (the
        // production dtype). A non-default precision flag is rejected rather than silently ignored.
        return Err(Error::Msg(
            "sdxl: precision override is not wired; the dense path runs fp16 (the production \
             reference dtype) — drop the precision override"
                .into(),
        ));
    }
    let dtype = Dtype::Float16;
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "sdxl expects a snapshot directory (tokenizer/ text_encoder/ unet/ vae/ …), not a \
                 single .safetensors file"
                    .into(),
            )),
        };
    let mut unet = loader::load_unet_dtype(root, dtype)?;
    if !spec.adapters.is_empty() {
        // Merge LoRA (kohya `lora_unet_` / PEFT, sc-2639) and LoKr (sc-2640) into the dense fp16
        // U-Net weights at load — the production reference merges into the `float16=True` U-Net too,
        // and merging (not a
        // forward-time residual) keeps the chaos-sensitive ancestral sampler bit-exact. Out-of-surface
        // keys (mid_block/ff/conv) are surfaced in the report, not dropped.
        //
        // Coverage (sc-2671): default to the strictly-more-correct COMPLETE surface — mid_block +
        // the GEGLU FF the vendored `lora.py` silently drops — so SDXL LoRAs apply in full, matching
        // diffusers (Michael's correctness-over-parity call, 2026-06-03). `SDXL_LORA_VENDORED` is the
        // escape hatch back to the legacy 515-module surface for byte-parity with the retired Python
        // path.
        let coverage = if std::env::var_os("SDXL_LORA_VENDORED").is_some() {
            eprintln!(
                "sdxl: SDXL_LORA_VENDORED set — restricting LoRA to the legacy vendored 515-module \
                 surface (mid_block + ff dropped; byte-parity with the retired Python path)"
            );
            crate::adapters::LoraCoverage::Vendored
        } else {
            crate::adapters::LoraCoverage::Complete
        };
        crate::adapters::apply_sdxl_adapters_with(&mut unet, &spec.adapters, coverage)?;
    }
    let mut te1 = loader::load_text_encoder_1_dtype(root, dtype)?;
    let mut te2 = loader::load_text_encoder_2_dtype(root, dtype)?;
    let vae = loader::load_vae(root)?; // VAE always f32 (vendored loads the autoencoder float16=False)

    // ControlNet branches (sc-3058; MultiControlNet sc-3378) — `spec.control` first, then each
    // `spec.extra_controls`, all at the U-Net dtype (fp16). Quantized with the U-Net below when
    // `spec.quantize` is set (the encoder-copy Linears; conv stem / cond-embedding / zero-convs stay
    // dense, matching the U-Net scope).
    let mut controls: Vec<ControlNet> = Vec::new();
    if let Some(src) = &spec.control {
        controls.push(loader::load_controlnet(src, dtype)?);
    }
    for src in &spec.extra_controls {
        controls.push(loader::load_controlnet(src, dtype)?);
    }

    // Optional IP-Adapter (sc-3059) — install the decoupled-attn K/V pairs into the still-mutable,
    // pre-quant U-Net (so they quantize with it) and keep the image-token encoder.
    let ip_adapter = match &spec.ip_adapter {
        Some(WeightsSource::Dir(p)) => {
            let (enc, pairs) = loader::load_ip_adapter(p, dtype)?;
            unet.install_ip_adapter(pairs)?;
            Some(enc)
        }
        Some(WeightsSource::File(_)) => {
            return Err(Error::Msg(
                "sdxl ip_adapter expects an h94/IP-Adapter snapshot directory, not a single file"
                    .into(),
            ));
        }
        None => None,
    };

    if let Some(q) = spec.quantize {
        // Q4/Q8 (group_size 64) over every quantizable Linear of the U-Net + both CLIP encoders —
        // applied AFTER the adapter merge (the merge needs the dense weight; `merge_dense_delta`
        // errors on a quantized base, matching the fork's "LoRA merged pre-quantization"). The core
        // `AdaptableLinear::quantize` casts each weight to bf16 before packing (sc-2604): SDXL ships
        // fp16/fp32 on disk, and quantizing the as-loaded dtype would give drifted group scales — the
        // sc-1975 "Q8 broken on base-1.0". Convs / norms / token & position embeddings stay dense
        // (gather lookups, not matmuls). The **VAE stays f32** — its only Linears are the tiny
        // quant/post-quant projections (negligible memory), and a dense decode preserves output
        // quality. Scope verified empirically by the full `load(Q).generate()` gate (sc-2641).
        let bits = q.bits();
        unet.quantize(bits)?;
        te1.quantize(bits)?;
        te2.quantize(bits)?;
        for cn in &mut controls {
            cn.quantize(bits)?;
        }
    }

    let cfg = DiffusionConfig::sdxl_base();
    let alpha_schedule =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end)?;
    Ok(Box::new(Sdxl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        te1,
        te2,
        unet,
        controls,
        ip_adapter,
        vae,
        sampler: EulerSampler::new_with_dtype(&cfg, true, dtype)?,
        alpha_schedule,
    }))
}

impl Generator for Sdxl {
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

        let sampler_name = req.sampler.as_deref().unwrap_or("euler_ancestral");
        let is_accel = ACCEL_SAMPLERS.contains(&sampler_name);
        // Per-variant defaults for the few-step samplers; the production defaults otherwise.
        let (def_steps, def_cfg, eta) = if is_accel {
            accel_defaults(sampler_name)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE, 0.0)
        };
        let steps = req.steps.unwrap_or(def_steps) as usize;
        let cfg = req.guidance.unwrap_or(def_cfg);
        let cfg_on = cfg > 1.0;
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let reference = self.resolve_reference(req)?;
        let mask_img = self.resolve_mask(req)?;
        let max_time = self.sampler.max_time();

        // Acceleration variants are txt2img-only in v1 (epic 2755 "Image-only v1"); reject an init
        // image rather than silently ignoring it.
        if is_accel && reference.is_some() {
            return Err(Error::Msg(format!(
                "sdxl: the {sampler_name:?} acceleration sampler is txt2img-only (no img2img \
                 reference) in this build"
            )));
        }
        // Inpaint (Mask) rides the ancestral img2img path and needs an init image to blend against.
        if mask_img.is_some() {
            if is_accel {
                return Err(Error::Msg(
                    "sdxl: inpaint masks are not supported with the acceleration samplers".into(),
                ));
            }
            if reference.is_none() {
                return Err(Error::Msg(
                    "sdxl: inpaint requires an init image (a Reference) alongside the Mask".into(),
                ));
            }
        }
        // ControlNet (sc-3058; MultiControlNet sc-3378): each `Control` conditioning pairs, in order,
        // with a loaded control branch (`spec.control` + `spec.extra_controls`); their residuals are
        // summed. Needs the ancestral path; not combined with an inpaint mask in this build.
        let control_reqs = self.resolve_control(req)?;
        if !control_reqs.is_empty() {
            if is_accel {
                return Err(Error::Msg(
                    "sdxl: ControlNet is not supported with the acceleration samplers".into(),
                ));
            }
            if control_reqs.len() != self.controls.len() {
                return Err(Error::Msg(format!(
                    "sdxl: {} Control conditioning(s) passed but the model was loaded with {} control \
                     checkpoint(s) (set LoadSpec::control + extra_controls, one per Control, in order)",
                    control_reqs.len(),
                    self.controls.len()
                )));
            }
            if mask_img.is_some() {
                return Err(Error::Msg(
                    "sdxl: combining a ControlNet (Control) with an inpaint Mask is not supported"
                        .into(),
                ));
            }
        }

        // Build the ControlNet contexts once (seed-independent): preprocess each control image to
        // [0,1] NHWC and CFG-batch it to match the U-Net input, paired by order with a loaded branch.
        let mut control_ctxs: Vec<ControlContext> = Vec::with_capacity(control_reqs.len());
        for ((image, scale), cn) in control_reqs.iter().zip(&self.controls) {
            let img = preprocess_control_image(image, req.width, req.height)?;
            let img = if cfg_on {
                concatenate_axis(&[&img, &img], 0)?
            } else {
                img
            };
            control_ctxs.push(ControlContext {
                controlnet: cn,
                // Precompute the step-invariant conditioning embedding once per run (F-069).
                cond_embed: cn.embed_cond(&img)?,
                scale: *scale,
            });
        }

        // IP-Adapter (sc-3059): when the model carries IP weights and a Reference is present (no
        // mask/control/accel), the Reference is the image prompt (txt2img + IP), NOT an img2img init.
        // The IP scale rides the Reference `strength` field (default 0.6). Tokens are seed-independent
        // → built once, CFG-batched with a zeros uncond row so the negative pass gets no IP signal.
        let ip_mode = self.ip_adapter.is_some()
            && reference.is_some()
            && mask_img.is_none()
            && control_reqs.is_empty()
            && !is_accel;
        let ip_scale = reference.and_then(|(_, s)| s).unwrap_or(IP_DEFAULT_SCALE);
        let ip_tokens = if ip_mode {
            let enc = self
                .ip_adapter
                .as_ref()
                .expect("ip_adapter present in ip_mode");
            let (image, _) = reference.expect("reference present in ip_mode");
            let tokens = enc.tokens(image)?;
            Some(if cfg_on {
                let zeros = enc.zeros_like_tokens(tokens.dtype())?;
                concatenate_axis(&[&tokens, &zeros], 0)?
            } else {
                tokens
            })
        } else {
            None
        };

        // Seed-independent conditioning + init encodes, hoisted above the per-image count loop
        // (F-068): the dual-CLIP forwards and the f32 VAE encode draw no RNG, so re-running them per
        // image was pure waste — the IP tokens above already follow this pattern. `random::seed` stays
        // per-iteration so each image's init-noise stream still matches the reference.
        let tokens = self
            .tokenizer
            .tokenize_batch(&req.prompt, if cfg_on { Some(negative) } else { None })?;
        let (conditioning, pooled) = encode_conditioning(&self.te1, &self.te2, &tokens)?;
        let time_ids = text_time_ids(pooled.shape()[0]);
        let latent_shape = [1, (req.height / 8) as i32, (req.width / 8) as i32, 4];
        // img2img/inpaint init latents (the f32 VAE encode) and the inpaint mask are seed-independent
        // too. `init_latents` is Some exactly for the ancestral img2img/inpaint paths — a Reference
        // that is neither an accel run nor an IP image prompt; `mask_latent` adds the inpaint mask.
        let init_latents = match reference {
            Some((image, _)) if !is_accel && !ip_mode => Some(encode_init_latents(
                &self.vae, image, req.width, req.height,
            )?),
            _ => None,
        };
        let mask_latent = match mask_img {
            Some(mask) if init_latents.is_some() => {
                Some(preprocess_mask(mask, req.width, req.height)?)
            }
            _ => None,
        };

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // One image per iteration (the vendored `_run_one`, n_images=1), each with its own seed.
            let seed = base_seed.wrapping_add(i as u64);
            // Seed the global RNG up front; the hoisted conditioning/VAE encodes drew no RNG, so the
            // first draw here is the init noise (the prior / img2img add_noise) — matching the
            // reference stream.
            mlx_rs::random::seed(seed)?;

            // Build the run's sampler + its seeded init latents. The denoise loop is driven entirely
            // by the sampler's own schedule (`sampler.num_steps()`), so the trait owns the per-step
            // timestep, the input scaling, and the step math.
            let (latents, sampler, blend): (
                mlx_rs::Array,
                Box<dyn DiffusionSampler + '_>,
                Option<InpaintBlend>,
            ) = if is_accel {
                // Few-step acceleration (txt2img): unit-noise prior scaled into the sampler's space.
                let s = self.build_accel_sampler(sampler_name, steps, eta);
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                let lat = s.scale_initial_noise(&noise)?;
                (lat, s, None)
            } else if let (Some(x_0), Some(mask_latent)) = (&init_latents, &mask_latent) {
                // Masked inpaint (sc-3057): same ancestral img2img start, but keep the FIXED prior
                // noise so the per-step blend can pin the black (keep) region to the init noised to
                // each step's σ. Default strength 0.85 (the worker's inpaint default).
                let strength = reference
                    .and_then(|(_, s)| s)
                    .unwrap_or(INPAINT_DEFAULT_STRENGTH)
                    .clamp(0.0, 1.0);
                let start_step = max_time * strength;
                let noise = mlx_rs::random::normal::<f32>(&latent_shape, None, None, None)?;
                let x_t = self.sampler.add_noise_with(x_0, &noise, start_step)?;
                let eff = (steps as f32 * strength) as usize;
                // The kept region is noised to each step's "next" time `t_prev` (schedule[i].1).
                let t_prev: Vec<f32> = self
                    .sampler
                    .timesteps(eff, start_step)?
                    .into_iter()
                    .map(|(_, tp)| tp)
                    .collect();
                let blend = InpaintBlend::new(
                    &self.sampler,
                    mask_latent.clone(),
                    x_0.clone(),
                    noise,
                    t_prev,
                );
                (
                    x_t,
                    Box::new(AncestralEuler::new(&self.sampler, eff, start_step)?),
                    Some(blend),
                )
            } else if let Some(x_0) = &init_latents {
                // img2img (ancestral; the vendored `generate_latents_from_image`): start at
                // `max_time·strength`, run `int(steps·strength)` steps — NO min-1 floor (strength ≤
                // 1/steps ⇒ 0 steps ⇒ init returned unchanged, dodging the σ=0 ancestral `σ_up` 0/0
                // → NaN).
                let strength = reference
                    .and_then(|(_, s)| s)
                    .unwrap_or(DEFAULT_STRENGTH)
                    .clamp(0.0, 1.0);
                let start_step = max_time * strength;
                let x_t = self.sampler.add_noise(x_0, start_step)?;
                let eff = (steps as f32 * strength) as usize;
                (
                    x_t,
                    Box::new(AncestralEuler::new(&self.sampler, eff, start_step)?),
                    None,
                )
            } else {
                // txt2img (ancestral): seeded prior.
                let prior = self.sampler.sample_prior(&latent_shape)?;
                (
                    prior,
                    Box::new(AncestralEuler::new(&self.sampler, steps, max_time)?),
                    None,
                )
            };

            let d = Denoiser {
                unet: &self.unet,
                sampler: sampler.as_ref(),
            };
            let latents = if let Some(tokens) = &ip_tokens {
                denoise_ip(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    tokens,
                    ip_scale,
                )?
            } else if !control_ctxs.is_empty() {
                denoise_multi_control(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    &control_ctxs,
                )?
            } else if let Some(b) = &blend {
                denoise_inpaint(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                    b,
                )?
            } else {
                denoise(
                    &d,
                    latents,
                    &conditioning,
                    &pooled,
                    &time_ids,
                    cfg,
                    &req.cancel,
                    on_progress,
                )?
            };

            on_progress(Progress::Decoding);
            images.push(decode_image(&self.vae, &latents)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

impl Sdxl {
    /// Build the per-run few-step acceleration sampler (sc-2769). `name` is one of
    /// [`ACCEL_SAMPLERS`]; `steps` is the inference step count (Lightning must match the loaded
    /// LoRA's 2/4/8); `eta` is the TCD stochasticity (Hyper-SD). The samplers cast the U-Net input to
    /// fp16 (the loaded compute dtype) and run their step math in f32.
    fn build_accel_sampler(&self, name: &str, steps: usize, eta: f32) -> Box<dyn DiffusionSampler> {
        let n_train = self.alpha_schedule.alphas_cumprod.len();
        let sched = self.alpha_schedule.clone();
        match name {
            "lcm" => Box::new(LcmSampler::new(
                sched,
                n_train,
                LCM_ORIGINAL_STEPS,
                steps,
                Dtype::Float16,
            )),
            "lightning" => Box::new(LightningSampler::new(
                &sched,
                n_train,
                steps,
                Dtype::Float16,
            )),
            "hyper" => Box::new(TcdSampler::new(
                sched,
                n_train,
                LCM_ORIGINAL_STEPS,
                steps,
                eta,
                Dtype::Float16,
            )),
            // `generate` only calls this for `name ∈ ACCEL_SAMPLERS`.
            _ => unreachable!("build_accel_sampler: {name:?} is not an acceleration sampler"),
        }
    }

    /// Extract the single img2img init image + its strength from the request's conditioning (the
    /// per-reference strength wins over `req.strength`). SDXL img2img conditions on exactly one init
    /// image, so more than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple reference images are not supported (single img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// Extract the single inpaint mask from the request's conditioning (sc-3057). White = repaint,
    /// black = keep. SDXL supports one mask; more than one is an error.
    fn resolve_mask<'a>(&self, req: &'a GenerationRequest) -> Result<Option<&'a Image>> {
        let mut mask = None;
        for c in &req.conditioning {
            if let Conditioning::Mask { image } = c {
                if mask.is_some() {
                    return Err(Error::Msg(
                        "sdxl: multiple inpaint masks are not supported".into(),
                    ));
                }
                mask = Some(image);
            }
        }
        Ok(mask)
    }

    /// Collect the ControlNet control images + `conditioning_scale`s (sc-3058; MultiControlNet
    /// sc-3378), in request order. Each pairs with a loaded control branch (`spec.control` +
    /// `spec.extra_controls`); the count must match (validated in `generate`). A single `Control` is
    /// the common case; more than one runs as MultiControlNet (residuals summed).
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<Vec<(&'a Image, f32)>> {
        let mut controls = Vec::new();
        for c in &req.conditioning {
            if let Conditioning::Control { image, scale, .. } = c {
                controls.push((image, *scale));
            }
        }
        Ok(controls)
    }
}

/// Capability-driven request validation, factored out so it can be unit-tested without loaded
/// weights. Rejects unsupported guidance / negative prompt / conditioning / size / count.
pub(crate) fn validate_request(caps: &Capabilities, req: &GenerationRequest) -> Result<()> {
    if req.prompt.is_empty() {
        return Err(Error::Msg("sdxl: prompt must not be empty".into()));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "count {} out of range 1..={}",
            req.count, caps.max_count
        )));
    }
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}x{} out of supported range {}..={}",
            req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    // SDXL works in latent space at /8; both dims must be multiples of 8.
    if !req.width.is_multiple_of(8) || !req.height.is_multiple_of(8) {
        return Err(Error::Msg(format!(
            "sdxl: width/height must be multiples of 8 (got {}x{})",
            req.width, req.height
        )));
    }
    if req.guidance.is_some() && !caps.supports_guidance {
        return Err(Error::Msg(
            "sdxl: `guidance` is not supported by this build".into(),
        ));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(
            "sdxl: negative prompt is not supported by this build".into(),
        ));
    }
    // Reject an unsupported sampler instead of silently downgrading it to the ancestral default.
    if let Some(s) = &req.sampler {
        if !caps.samplers.contains(&s.as_str()) {
            return Err(Error::Msg(format!(
                "sdxl: unsupported sampler {s:?} (supported: {:?})",
                caps.samplers
            )));
        }
    }
    for c in &req.conditioning {
        let kind = c.kind();
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "sdxl does not accept {kind:?} conditioning"
            )));
        }
    }
    Ok(())
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

use mlx_gen::ModelRegistration;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_sdxl() {
        let d = descriptor();
        assert_eq!(d.id, "sdxl");
        assert_eq!(d.family, "sdxl");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_negative_prompt);
    }

    #[test]
    fn registered_in_core_registry() {
        // Linking this crate must self-register the model (inventory link-time collection).
        assert!(
            mlx_gen::registry::generators().any(|r| (r.descriptor)().id == "sdxl"),
            "sdxl is not registered in mlx_gen's generator registry"
        );
    }

    #[test]
    fn validate_rejects_empty_prompt() {
        let caps = descriptor().capabilities;
        let req = GenerationRequest::default(); // default prompt is empty
        let err = validate_request(&caps, &req).unwrap_err().to_string();
        assert!(err.contains("empty"), "got: {err}");
    }

    #[test]
    fn validate_accepts_cfg_and_negative_prompt_rejects_bad_size() {
        let caps = descriptor().capabilities;
        // Real CFG + negative prompt are supported.
        let mut req = GenerationRequest {
            prompt: "a fox".into(),
            guidance: Some(7.0),
            negative_prompt: Some("blurry".into()),
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_ok());
        // Non-multiple-of-8 size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 1020,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
        // Out-of-range size is rejected.
        req = GenerationRequest {
            prompt: "a fox".into(),
            width: 256,
            height: 256,
            ..Default::default()
        };
        assert!(validate_request(&caps, &req).is_err());
    }

    #[test]
    fn validate_sampler_selection() {
        let caps = descriptor().capabilities;
        let base = GenerationRequest {
            prompt: "a fox".into(),
            ..Default::default()
        };
        // The default + every wired sampler is accepted (an unset sampler defaults to ancestral).
        assert!(validate_request(&caps, &base).is_ok());
        for ok in ["euler_ancestral", "lcm", "lightning", "hyper"] {
            assert!(
                validate_request(
                    &caps,
                    &GenerationRequest {
                        sampler: Some(ok.into()),
                        ..base.clone()
                    }
                )
                .is_ok(),
                "sampler {ok:?} should be accepted"
            );
        }
        // `euler` (and any unknown sampler) is rejected, not silently downgraded.
        for bad in ["euler", "ddim", "nonsense"] {
            let err = validate_request(
                &caps,
                &GenerationRequest {
                    sampler: Some(bad.into()),
                    ..base.clone()
                },
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains("unsupported sampler"), "got: {err}");
        }
    }

    #[test]
    fn load_rejects_single_file_source() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/sdxl.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("snapshot directory"), "got: {err}");
    }
}
