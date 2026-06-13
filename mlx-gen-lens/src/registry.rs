//! `LensGenerator` — the [`mlx_gen::Generator`] impl wiring the Lens pipeline ([`crate::pipeline`])
//! into `mlx_gen`'s registry under **two** ids (sc-3173):
//!
//! - **`lens_turbo`** — the distilled turbo variant: **4 steps, guidance 1.0** (≈ no CFG).
//! - **`lens`** — the base variant: **20 steps, CFG 5.0**.
//!
//! Both ids share the identical crate/architecture/weights tree and differ **only** in their default
//! `num_steps` / `guidance_scale` (the reference ships them as separate model cards with the same
//! arch). A request's explicit `steps` / `guidance` still override the per-id default.
//!
//! **Surface.** This is a pure **T2I** generator: no img2img / ControlNet / IP conditioning (none
//! exists in the Lens port), no LoRA/LoKr inference merge (training stays Python). The dense path is
//! bf16; the `Fp32` precision override is honored. **Q4/Q8** quantize the gpt-oss encoder's MoE
//! experts at load (sc-3172 — the ~38 GB / 20 B-param bulk → ~12 GB); the DiT stays dense (sc-3175).
//!
//! **Registration mechanism:** the two `inventory::submit!`s below are collected by `mlx_gen`'s
//! `inventory::collect!` at *link* time, so they activate whenever a consumer (the worker, or this
//! crate's own test binary) links `mlx-gen-lens`. The core `mlx-gen` crate does **not** depend on the
//! model crates (by design); there is no root-crate dependency to add.

use mlx_rs::Dtype;

use mlx_gen::{
    default_seed, gen_core, Capabilities, Error, GenerationOutput, GenerationRequest, Generator,
    LoadSpec, Modality, ModelDescriptor, ModelRegistration, Precision, Progress, Quant, Result,
    WeightsSource,
};

use crate::pipeline::{GenerateOptions, LensPipeline, DEFAULT_DATE, VAE_SCALE_FACTOR};

/// Registry id — the distilled turbo variant.
pub const MODEL_ID_TURBO: &str = "lens_turbo";
/// Registry id — the base variant.
pub const MODEL_ID_BASE: &str = "lens";

/// Per-variant sampling defaults (`num_steps`, `guidance_scale`) baked into the loaded generator.
#[derive(Clone, Copy)]
struct Defaults {
    id: &'static str,
    steps: u32,
    guidance: f32,
}

const TURBO_DEFAULTS: Defaults = Defaults {
    id: MODEL_ID_TURBO,
    steps: 4,
    guidance: 1.0,
};
const BASE_DEFAULTS: Defaults = Defaults {
    id: MODEL_ID_BASE,
    steps: 20,
    guidance: 5.0,
};

/// Lens' identity + capabilities for `id` — constructible without loading weights (registry
/// introspection). Advertises **only** the wired + parity-proven surface: T2I with negative-prompt /
/// guidance CFG, no conditioning, no quant (yet), no LoRA.
fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "lens",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            // The norm-rescaled CFG path is always present; turbo simply defaults guidance to 1.0.
            supports_negative_prompt: true,
            supports_guidance: true,
            supports_true_cfg: false,
            conditioning: vec![], // pure T2I — no img2img / control / IP in the Lens port
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["flow_match_euler"],
            schedulers: vec!["flow_match"],
            // Buckets span 736..2080 (all ÷16); allow any ÷16 size in a sane range.
            min_size: 256,
            max_size: 2080,
            max_count: 8,
            mac_only: true,
            // sc-3172: Q4/Q8 quantize the gpt-oss encoder's MoE experts (the ~38 GB / 20 B-param bulk)
            // at load → ~12 GB. The DiT stays dense (its quant is sc-3175).
            supported_quants: &[Quant::Q4, Quant::Q8],
            supports_kv_cache: false,
            // The Lens schedule computes its own empirical-μ shift internally (not a loader hint).
            requires_sigma_shift: false,
        },
    }
}

/// Public descriptor accessors (used by the registry submits + tests).
pub fn descriptor_turbo() -> ModelDescriptor {
    descriptor_for(MODEL_ID_TURBO)
}
pub fn descriptor_base() -> ModelDescriptor {
    descriptor_for(MODEL_ID_BASE)
}

/// A loaded, dispatchable Lens generator: the pipeline + the variant's descriptor & sampling defaults.
pub struct LensGenerator {
    descriptor: ModelDescriptor,
    defaults: Defaults,
    pipe: LensPipeline,
}

/// Build a [`LensGenerator`] from a [`LoadSpec`] with the given per-variant defaults.
///
/// `spec.weights` is a `microsoft/Lens-Turbo` (or `microsoft/Lens`) snapshot dir (the diffusers
/// multi-component tree). Dense runs **bf16**; `Precision::Fp32` loads the tight-gate f32 path.
/// `spec.quantize` (Q4/Q8) quantizes the encoder's MoE experts at load (sc-3172). `control` /
/// `ip_adapter` / `adapters` are not part of the Lens port and are rejected.
fn load_with(spec: &LoadSpec, defaults: Defaults) -> Result<Box<dyn Generator>> {
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Msg(format!(
            "{}: ControlNet / IP-Adapter conditioning is not part of the Lens port",
            defaults.id
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{}: LoRA/LoKr adapters are not wired for inference (training stays Python)",
            defaults.id
        )));
    }
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
            "{}: expects a Lens snapshot directory (tokenizer/ text_encoder/ transformer/ vae/), \
                 not a single .safetensors file",
            defaults.id
        )))
        }
    };
    let pipe = LensPipeline::load_quant(&root, dtype, spec.quantize)?;
    Ok(Box::new(LensGenerator {
        descriptor: descriptor_for(defaults.id),
        defaults,
        pipe,
    }))
}

impl Generator for LensGenerator {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        self.validate_impl(req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl LensGenerator {
    /// The rich-`Result` body behind [`Generator::validate`].
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(self.defaults.id, &self.descriptor.capabilities, req)?;
        Ok(())
    }

    /// The rich-`Result` body behind [`Generator::generate`]: map the request onto the pipeline,
    /// looping `count` with per-image seeds and streaming step/decode progress.
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate_impl(req)?;

        let steps = req.steps.unwrap_or(self.defaults.steps) as usize;
        let guidance = req.guidance.unwrap_or(self.defaults.guidance);
        let negative = req.negative_prompt.as_deref().unwrap_or("");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let total = steps as u32;

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let opts = GenerateOptions {
                prompt: &req.prompt,
                negative_prompt: negative,
                height: req.height,
                width: req.width,
                num_steps: steps,
                guidance_scale: guidance,
                seed,
                date: DEFAULT_DATE,
                // The local reasoner (sc-3176) is a standalone opt-in; the registry path leaves it off
                // (matching the vendor default), so no reasoner is attached here.
                enable_reasoner: false,
            };
            // Re-encode per image is cheap relative to denoise and keeps the RNG order matching the
            // struct API (one noise draw per image, no shared state). Progress is streamed via the
            // pipeline's per-step callback; cancellation is honored inside `denoise`.
            let image = self
                .pipe
                .generate_with_progress(&opts, &req.cancel, &mut |cur| {
                    on_progress(Progress::Step {
                        current: cur as u32,
                        total,
                    });
                })?;
            on_progress(Progress::Decoding);
            let _ = i;
            images.push(image);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Capability-driven request validation (unit-testable without loaded weights).
pub(crate) fn validate_request(
    id: &str,
    caps: &Capabilities,
    req: &GenerationRequest,
) -> Result<()> {
    // Shared capability contract: count/size range, negative_prompt/guidance/true_cfg, sampler,
    // scheduler, conditioning kinds.
    caps.validate_request(id, req)?;

    if req.prompt.is_empty() {
        return Err(Error::Msg(format!("{id}: prompt must not be empty")));
    }
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    // The Flux.2 VAE + DiT patchify downsample by 16; non-multiple-of-16 dims mismatch latent shapes.
    if !req.width.is_multiple_of(VAE_SCALE_FACTOR) || !req.height.is_multiple_of(VAE_SCALE_FACTOR) {
        return Err(Error::Msg(format!(
            "{id}: width/height must be multiples of {VAE_SCALE_FACTOR} (got {}x{})",
            req.width, req.height
        )));
    }
    Ok(())
}

/// Registry adapters: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` loads into it.
fn load_turbo_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, TURBO_DEFAULTS).map_err(Into::into)
}
fn load_base_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, BASE_DEFAULTS).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_turbo, load: load_turbo_registered }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_base, load: load_base_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptors_are_lens() {
        for (d, id, steps, g) in [
            (descriptor_turbo(), MODEL_ID_TURBO, 4u32, 1.0f32),
            (descriptor_base(), MODEL_ID_BASE, 20, 5.0),
        ] {
            assert_eq!(d.id, id);
            assert_eq!(d.family, "lens");
            assert_eq!(d.modality, Modality::Image);
            assert!(d.capabilities.supports_guidance);
            assert!(d.capabilities.supports_negative_prompt);
            assert!(!d.capabilities.supports_true_cfg);
            assert!(d.capabilities.conditioning.is_empty());
            assert!(!d.capabilities.supports_lora);
            // sc-3172: encoder MoE experts quantize to Q4/Q8 at load.
            assert_eq!(d.capabilities.supported_quants, &[Quant::Q4, Quant::Q8]);
            // The defaults are exercised end-to-end in the e2e test; assert the constants here.
            let def = if id == MODEL_ID_TURBO {
                TURBO_DEFAULTS
            } else {
                BASE_DEFAULTS
            };
            assert_eq!((def.steps, def.guidance), (steps, g));
        }
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        // The `inventory::submit!`s are linked into this test binary, so `mlx_gen::load` resolves
        // both ids (and fails on the bogus weights dir) — proving registration without the snapshot.
        for id in [MODEL_ID_TURBO, MODEL_ID_BASE] {
            let spec = LoadSpec {
                weights: WeightsSource::Dir("/nonexistent/lens".into()),
                quantize: None,
                precision: Precision::Bf16,
                control: None,
                ip_adapter: None,
                adapters: Vec::new(),
                extra_controls: Vec::new(),
            };
            let err = match mlx_gen::load(id, &spec) {
                Ok(_) => panic!("bogus weights dir must fail to load"),
                Err(e) => e.to_string(),
            };
            assert!(
                !err.contains("no generator registered"),
                "{id} should resolve in the registry; got: {err}"
            );
        }
    }

    #[test]
    fn load_rejects_unsupported_overlays_not_quant() {
        let base = LoadSpec {
            weights: WeightsSource::Dir("/nonexistent/lens".into()),
            quantize: None,
            precision: Precision::Bf16,
            control: None,
            ip_adapter: None,
            adapters: Vec::new(),
            extra_controls: Vec::new(),
        };
        // A ControlNet overlay is rejected (not part of the Lens port) — the message names it, before
        // any weights load.
        let with_control = LoadSpec {
            control: Some(WeightsSource::Dir("/nonexistent/cn".into())),
            ..base.clone()
        };
        let err = match load_with(&with_control, TURBO_DEFAULTS) {
            Ok(_) => panic!("control must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("not part of the Lens port"), "got: {err}");

        // Quantize is NOT rejected (sc-3172) — it proceeds to the load and fails only on the bogus
        // weights dir, never with an "unsupported" message.
        let quant = LoadSpec {
            quantize: Some(Quant::Q8),
            ..base.clone()
        };
        let err = match load_with(&quant, TURBO_DEFAULTS) {
            Ok(_) => panic!("bogus weights dir must fail to load"),
            Err(e) => e.to_string(),
        };
        assert!(
            !err.contains("quantization") && !err.contains("not part of"),
            "quantize must be accepted (sc-3172); got: {err}"
        );
    }

    #[test]
    fn validate_rejects_bad_inputs() {
        let caps = descriptor_turbo().capabilities;
        let ok = GenerationRequest {
            prompt: "a fox".into(),
            width: 1024,
            height: 1024,
            ..Default::default()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &ok).is_ok());

        let empty = GenerationRequest {
            prompt: "".into(),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &empty).is_err());

        let zero_steps = GenerationRequest {
            steps: Some(0),
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &zero_steps).is_err());

        let bad_dims = GenerationRequest {
            width: 1000, // not ÷16
            ..ok.clone()
        };
        assert!(validate_request(MODEL_ID_TURBO, &caps, &bad_dims).is_err());
    }
}
