//! sc-3194: the `Generator` registration that wires SenseNova-U1's image modes into the mlx-gen
//! provider registry as **`sensenova_u1_8b`**.
//!
//! The `Generator` contract emits images, so the image-producing modes route through one
//! [`Generator::generate`] dispatch: **T2I** (no conditioning → [`T2iModel::generate`]) and
//! **image-edit / Character Studio** (a [`Conditioning::Reference`]/[`MultiReference`] →
//! [`T2iModel::it2i_generate`]). The generic request maps as: `guidance` → text `cfg_scale`,
//! `true_cfg` → image `img_cfg_scale` (edit ≈ 1.0, character ≈ 1.5), `scheduler_shift` →
//! `timestep_shift`, `steps`/`seed`/`width`/`height` as given.
//!
//! VQA ([`T2iModel::vqa`], text out) and interleave / Document Studio
//! ([`T2iModel::interleave_gen`], text + images) cannot be expressed by [`GenerationOutput`]
//! (`Images`/`Video` only), so they are consumed by the SceneWorks worker through those public
//! [`T2iModel`] methods directly — the registry path here covers exactly the image-generation
//! surface. `spec.quantize` (Q4/Q8) quantizes the backbone decoder stack (sc-3193).
//!
//! A **second** id, `sensenova_u1_8b_fast` (sc-3192), shares this loader: its [`load_fast`] merges
//! the 8-step distill LoRA into the dense generation path before any quantization, and its
//! generator applies the distilled defaults (`cfg_scale=1.0`, `num_steps=8`). Registering it under
//! a distinct id makes the merge part of the model cache key — the worker caches by id, so the
//! merged variant can never be served for the base id (and vice versa) even though they share the
//! same on-disk base weights. User-supplied LoRAs stay rejected for both ids (`supports_lora=false`,
//! matching the torch adapter); the distill LoRA is a curated property of the fast variant, not a
//! user adapter.

use mlx_rs::ops::divide;
use mlx_rs::Array;

use mlx_gen::image::{decoded_to_image, resize_bicubic_u8};
use mlx_gen::{
    default_seed, gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Quant, Result, WeightsSource,
};

use crate::config::NeoChatConfig;
use crate::distill::resolve_distill_lora;
use crate::loader::{check_coverage, load_raw};
use crate::t2i::{smart_resize, StepReporter, T2iModel, T2iOptions};
use crate::text::load_tokenizer;
use mlx_gen::weights::Weights;

pub const MODEL_ID: &str = "sensenova_u1_8b";
/// The 8-step distilled variant (sc-3192): same base weights with the distill LoRA merged in.
pub const MODEL_ID_FAST: &str = "sensenova_u1_8b_fast";

const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
/// Distilled defaults (`docs/base_vs_distill.md`): 8 NFE at CFG 1.0 (guidance off).
const DEFAULT_STEPS_FAST: u32 = 8;
const DEFAULT_GUIDANCE_FAST: f32 = 1.0;
const DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
/// Cell = patch·merge: every side must be a multiple of this (the patchify grid).
const CELL: u32 = 32;
/// Source-image preprocessing bounds (the reference `it2i_generate` `load_image_native`).
const REF_MIN_PIXELS: i64 = 512 * 512;
const REF_MAX_PIXELS: i64 = 2048 * 2048;

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}

/// The descriptor for the 8-step distilled variant. Identical capabilities to the base — only the
/// id and the generation defaults (applied in [`SenseNova::options`]) differ.
pub fn descriptor_fast() -> ModelDescriptor {
    descriptor_for(MODEL_ID_FAST)
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "sensenova-u1",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // `guidance` → text cfg_scale; `true_cfg` → image cfg (it2i edit≈1.0 / character≈1.5).
            supports_guidance: true,
            supports_true_cfg: true,
            // Reference image(s) → it2i edit / Character Studio reference. No control/depth/mask.
            conditioning: vec![
                ConditioningKind::Reference,
                ConditioningKind::MultiReference,
            ],
            supports_lora: false,
            supports_lokr: false,
            supported_quants: &[Quant::Q4, Quant::Q8],
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            // The backbone uses a KV cache for the AR prefix + denoise.
            supports_kv_cache: true,
            // Flow-match schedule uses a timestep shift (mapped from scheduler_shift).
            requires_sigma_shift: true,
        },
    }
}

/// A loaded SenseNova-U1 generator: the unified [`T2iModel`] + tokenizer + cached descriptor.
pub struct SenseNova {
    descriptor: ModelDescriptor,
    tokenizer: mlx_gen::tokenizer::TextTokenizer,
    model: T2iModel,
    /// The 8-step distilled variant — selects the distilled generation defaults (8 NFE, CFG 1.0).
    fast: bool,
}

impl SenseNova {
    /// The unified model, for the worker paths the `Generator` contract can't express (VQA text,
    /// interleave text+images): call [`T2iModel::vqa`] / [`T2iModel::interleave_gen`] directly.
    pub fn model(&self) -> &T2iModel {
        &self.model
    }

    /// The tokenizer (shared by every mode).
    pub fn tokenizer(&self) -> &mlx_gen::tokenizer::TextTokenizer {
        &self.tokenizer
    }
}

/// Construct the base [`SenseNova`] (`sensenova_u1_8b`) from a [`LoadSpec`]. `spec.weights` must be a
/// [`WeightsSource::Dir`] pointing at a `sensenova/SenseNova-U1-8B-MoT` snapshot. Weights load dense
/// at their on-disk dtype (bf16); `spec.quantize` (Q4/Q8) then quantizes the backbone decoder stack
/// (sc-3193).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_inner(spec, false)
}

/// Construct the 8-step distilled [`SenseNova`] (`sensenova_u1_8b_fast`, sc-3192): the same base
/// snapshot with the distill LoRA merged into the dense generation path **before** any
/// quantization, plus the distilled generation defaults. The LoRA is resolved by
/// [`resolve_distill_lora`] (env override / co-located / HF cache).
pub fn load_fast(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_inner(spec, true)
}

fn load_inner(spec: &LoadSpec, fast: bool) -> Result<Box<dyn Generator>> {
    let id = if fast { MODEL_ID_FAST } else { MODEL_ID };
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(format!(
            "{id}: only dense bf16 is wired (drop the precision override)"
        )));
    }
    // User-supplied LoRAs are unsupported on both ids (the distill LoRA is merged internally by the
    // fast loader, not stacked via `spec.adapters`).
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: user-supplied adapters are not supported (supports_lora=false)"
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id} expects a snapshot directory, not a single .safetensors file"
            )))
        }
    };
    let cfg = NeoChatConfig::from_dir(root)?;
    let weights = load_raw(root)?;
    // F-137: diff the checkpoint against the canonical key set before building modules (the loader
    // module doc promised this validation). Missing keys still fail via `require` with the exact
    // name during `from_weights`; this additionally rejects extra/renamed tensors that would
    // otherwise load silently with whatever subset matches.
    check_coverage(weights.keys(), &cfg).require_no_unexpected(id)?;
    let mut model = T2iModel::from_weights(&weights, &cfg)?;
    // The fast variant merges the 8-step distill LoRA into the dense generation path. This MUST
    // precede quantization (the merge seam errors on a quantized base). Assert full coverage —
    // `7 · layers` gen-path projections + the 2 FM-head Linears — so a stale/mismatched LoRA fails
    // loudly rather than silently merging a subset.
    if fast {
        let lora_path = resolve_distill_lora(root)?;
        let lora = Weights::from_file(&lora_path)?;
        let applied = model.merge_distill_lora(&lora)?;
        let expected = cfg.llm.num_hidden_layers * 7 + 2;
        if applied != expected {
            return Err(Error::Msg(format!(
                "{id}: distill LoRA merged {applied} targets, expected {expected} \
                 (7·{} gen-path linears + 2 fm_head) — wrong LoRA file?",
                cfg.llm.num_hidden_layers
            )));
        }
    }
    // Q4/Q8 quantize the backbone decoder stack after the dense load (sc-3193). For the fast variant
    // the distill LoRA is already merged, so quantization sees the distilled weights.
    if let Some(q) = spec.quantize {
        model.quantize(q.bits())?;
    }
    let tokenizer = load_tokenizer(root)?;
    Ok(Box::new(SenseNova {
        descriptor: descriptor_for(id),
        tokenizer,
        model,
        fast,
    }))
}

impl SenseNova {
    /// Collect the reference images (`Reference` + `MultiReference`) as preprocessed
    /// `[3,H,W]`-in-`[0,1]` arrays for [`T2iModel::it2i_generate`]. Empty ⇒ T2I.
    fn references(&self, req: &GenerationRequest) -> Result<Vec<Array>> {
        let mut out = Vec::new();
        for c in &req.conditioning {
            match c {
                Conditioning::Reference { image, .. } => out.push(image_to_chw01(image)?),
                Conditioning::MultiReference { images } => {
                    for image in images {
                        out.push(image_to_chw01(image)?);
                    }
                }
                _ => {}
            }
        }
        Ok(out)
    }

    fn options(&self, req: &GenerationRequest, seed: u64) -> T2iOptions {
        // The distilled variant defaults to 8 NFE at CFG 1.0; the base to 50 NFE at CFG 4.0. An
        // explicit request value always wins.
        let (def_steps, def_guidance) = if self.fast {
            (DEFAULT_STEPS_FAST, DEFAULT_GUIDANCE_FAST)
        } else {
            (DEFAULT_STEPS, DEFAULT_GUIDANCE)
        };
        T2iOptions {
            cfg_scale: req.guidance.unwrap_or(def_guidance),
            img_cfg_scale: req.true_cfg.unwrap_or(1.0),
            num_steps: req.steps.unwrap_or(def_steps) as usize,
            timestep_shift: req.scheduler_shift.unwrap_or(DEFAULT_TIMESTEP_SHIFT),
            seed,
            ..Default::default()
        }
    }
}

impl Generator for SenseNova {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Use the descriptor's own id so the base and `_fast` variants attribute rejections to the
        // right model (F-143).
        let id = self.descriptor.id;
        self.descriptor.capabilities.validate_request(id, req)?;
        validate_dims_and_steps(id, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl SenseNova {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the
    /// family helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`]
    /// (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let references = self.references(req)?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let (w, h) = (req.width as i32, req.height as i32);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            // Check the worker's cancel flag between images too (a 50-step 8B run is multi-minute;
            // the per-step check lives in the denoise loop via the StepReporter). F-128.
            if req.cancel.is_cancelled() {
                return Err(Error::Msg("sensenova: generation cancelled".into()));
            }
            let opts = self.options(req, base_seed.wrapping_add(i as u64));
            // Thread cancellation + per-step progress into the denoise loop. Progress now reports the
            // denoise step (Kolors/SDXL semantics), not the image index as the old single tick did.
            let reporter = StepReporter::new(&req.cancel, on_progress);
            let out = if references.is_empty() {
                self.model.generate(
                    &self.tokenizer,
                    &req.prompt,
                    w,
                    h,
                    &opts,
                    None,
                    Some(reporter),
                )?
            } else {
                self.model.it2i_generate(
                    &self.tokenizer,
                    &req.prompt,
                    &references,
                    w,
                    h,
                    &opts,
                    None,
                    Some(reporter),
                )?
            };
            images.push(decoded_to_image(&out.image)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

/// Request-boundary checks beyond the capability surface: 32-pixel alignment per side and a positive
/// step count. Factored out so it can be unit-tested without loaded weights. `id` is the rejecting
/// model's descriptor id (base or `_fast`) so the error attributes to the right variant (F-143).
fn validate_dims_and_steps(id: &str, req: &GenerationRequest) -> Result<()> {
    if !req.width.is_multiple_of(CELL) || !req.height.is_multiple_of(CELL) {
        return Err(Error::Msg(format!(
            "{id}: {}x{} must be a multiple of {CELL} per side",
            req.width, req.height
        )));
    }
    // `steps == 0` builds an empty denoise trajectory, so `generate`/`it2i_generate`/`interleave_gen`
    // panic on `.last().expect("at least one step")` (F-125). Reject it at the boundary; `None` falls
    // back to the variant default.
    if req.steps == Some(0) {
        return Err(Error::Msg(format!("{id}: steps must be >= 1")));
    }
    Ok(())
}

/// Decode an [`Image`] (RGB8 HWC) to a `[3,H,W]` f32 tensor in `[0,1]`, smart-resized to a
/// 32-aligned bucket within `[512², 2048²]` pixels (the reference `load_image_native`).
fn image_to_chw01(img: &Image) -> Result<Array> {
    let (in_w, in_h) = (img.width as i32, img.height as i32);
    let (out_h, out_w) = smart_resize(in_h, in_w, CELL as i32, REF_MIN_PIXELS, REF_MAX_PIXELS);
    // Resize (bicubic, PIL-faithful) → f32 HWC in [0,255].
    let hwc = resize_bicubic_u8(
        &img.pixels,
        in_h as usize,
        in_w as usize,
        out_h as usize,
        out_w as usize,
    );
    let hwc = Array::from_slice(&hwc, &[out_h, out_w, 3]);
    let chw = hwc.transpose_axes(&[2, 0, 1])?; // HWC → CHW
    divide(&chw, Array::from_f32(255.0)).map_err(Error::from)
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load`] into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

/// Registry adapter for the 8-step distilled variant (sc-3192).
fn load_fast_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_fast(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_fast, load: load_fast_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_sensenova() {
        let d = descriptor();
        assert_eq!(d.id, "sensenova_u1_8b");
        assert_eq!(d.modality, Modality::Image);
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(d.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(d.capabilities.supports_guidance);
        assert!(d.capabilities.supports_true_cfg);
    }

    #[test]
    fn descriptor_fast_differs_only_in_id() {
        let base = descriptor();
        let fast = descriptor_fast();
        assert_eq!(fast.id, "sensenova_u1_8b_fast");
        assert_ne!(fast.id, base.id);
        // Same capability surface as the base — only the id (and the generation defaults) differ.
        assert_eq!(fast.family, base.family);
        assert_eq!(fast.modality, base.modality);
        assert_eq!(
            fast.capabilities.supports_guidance,
            base.capabilities.supports_guidance
        );
        assert_eq!(
            fast.capabilities.supports_true_cfg,
            base.capabilities.supports_true_cfg
        );
        assert!(fast.capabilities.accepts(ConditioningKind::Reference));
        assert!(fast.capabilities.accepts(ConditioningKind::MultiReference));
        assert!(!fast.capabilities.supports_lora);
        assert_eq!(fast.capabilities.max_size, base.capabilities.max_size);
    }

    #[test]
    fn registered_in_registry() {
        // The `inventory::submit!`s are linked into the test binary, so the registry finds both ids.
        let ids: Vec<&str> = mlx_gen::registry::generators()
            .map(|r| (r.descriptor)().id)
            .collect();
        assert!(ids.contains(&MODEL_ID), "{MODEL_ID} not registered");
        assert!(
            ids.contains(&MODEL_ID_FAST),
            "{MODEL_ID_FAST} not registered"
        );
    }

    #[test]
    fn load_rejects_single_file() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load(&spec).is_err());
        // The fast loader rejects a single file the same way (before touching the LoRA).
        assert!(load_fast(&spec).is_err());
    }

    #[test]
    fn both_loaders_reject_user_adapters() {
        // `supports_lora=false` on both ids; the distill LoRA is merged internally by `load_fast`,
        // never supplied via `spec.adapters`.
        let mut spec = LoadSpec::new(WeightsSource::Dir("/tmp/does-not-exist".into()));
        spec.adapters = vec![mlx_gen::AdapterSpec::new(
            "/tmp/some.safetensors".into(),
            1.0,
            mlx_gen::AdapterKind::Lora,
        )];
        let msg = |r: Result<Box<dyn Generator>>| match r {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected an error rejecting adapters"),
        };
        assert!(msg(load(&spec)).contains("adapters"));
        assert!(msg(load_fast(&spec)).contains("adapters"));
    }

    #[test]
    fn validate_rejects_unaligned_size() {
        let d = descriptor();
        let req = GenerationRequest {
            width: 300,
            height: 256,
            ..Default::default()
        };
        // Capability floor passes (in range) but the 32-alignment check rejects 300.
        assert!(d.capabilities.validate_request(MODEL_ID, &req).is_ok());
        assert!(!300u32.is_multiple_of(CELL));
        let err = validate_dims_and_steps(MODEL_ID, &req)
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple of"), "got: {err}");
        // F-143: the rejecting model's own id is in the message, so the fast variant attributes the
        // error to `sensenova_u1_8b_fast`, not the hardcoded base id.
        let fast_err = validate_dims_and_steps(MODEL_ID_FAST, &req)
            .unwrap_err()
            .to_string();
        assert!(
            fast_err.contains(MODEL_ID_FAST),
            "fast id should appear: {fast_err}"
        );
    }

    #[test]
    fn validate_rejects_zero_steps() {
        // F-125: `steps == 0` builds an empty denoise trajectory → `.expect("at least one step")`
        // panic. Reject at the boundary; `None` and any positive count pass.
        let bad = GenerationRequest {
            width: 512,
            height: 512,
            steps: Some(0),
            ..Default::default()
        };
        let err = validate_dims_and_steps(MODEL_ID, &bad)
            .unwrap_err()
            .to_string();
        assert!(err.contains("steps must be >= 1"), "got: {err}");

        for steps in [None, Some(1), Some(50)] {
            let ok = GenerationRequest {
                width: 512,
                height: 512,
                steps,
                ..Default::default()
            };
            assert!(
                validate_dims_and_steps(MODEL_ID, &ok).is_ok(),
                "steps={steps:?}"
            );
        }
    }
}
