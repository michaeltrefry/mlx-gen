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
//! surface. The 8-step `sensenova_u1_8b_fast` distill variant (sc-3192) and Q4/Q8 quant (sc-3193)
//! register/extend this loader in their own slices.

use mlx_rs::ops::divide;
use mlx_rs::Array;

use mlx_gen::image::{decoded_to_image, resize_bicubic_u8};
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Result, WeightsSource,
};

use crate::config::NeoChatConfig;
use crate::loader::load_raw;
use crate::t2i::{smart_resize, T2iModel, T2iOptions};
use crate::text::load_tokenizer;

pub const MODEL_ID: &str = "sensenova_u1_8b";

const DEFAULT_STEPS: u32 = 50;
const DEFAULT_GUIDANCE: f32 = 4.0;
const DEFAULT_TIMESTEP_SHIFT: f32 = 3.0;
/// Cell = patch·merge: every side must be a multiple of this (the patchify grid).
const CELL: u32 = 32;
/// Source-image preprocessing bounds (the reference `it2i_generate` `load_image_native`).
const REF_MIN_PIXELS: i64 = 512 * 512;
const REF_MAX_PIXELS: i64 = 2048 * 2048;

pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "sensenova-u1",
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

/// Construct a [`SenseNova`] from a [`LoadSpec`]. `spec.weights` must be a [`WeightsSource::Dir`]
/// pointing at a `sensenova/SenseNova-U1-8B-MoT` snapshot. Weights load dense at their on-disk dtype
/// (bf16). Quantization (sc-3193) and LoRA (the 8-step distill, sc-3192) are not yet wired and are
/// rejected rather than silently ignored.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "sensenova_u1_8b: only dense bf16 is wired (drop the precision override)".into(),
        ));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(
            "sensenova_u1_8b: Q4/Q8 quantization is not wired yet (sc-3193)".into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "sensenova_u1_8b: adapters (the 8-step distill LoRA) are not wired yet (sc-3192)"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "sensenova_u1_8b expects a snapshot directory, not a single .safetensors file"
                    .into(),
            ))
        }
    };
    let cfg = NeoChatConfig::from_dir(root)?;
    let weights = load_raw(root)?;
    let model = T2iModel::from_weights(&weights, &cfg)?;
    let tokenizer = load_tokenizer(root)?;
    Ok(Box::new(SenseNova {
        descriptor: descriptor(),
        tokenizer,
        model,
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
        T2iOptions {
            cfg_scale: req.guidance.unwrap_or(DEFAULT_GUIDANCE),
            img_cfg_scale: req.true_cfg.unwrap_or(1.0),
            num_steps: req.steps.unwrap_or(DEFAULT_STEPS) as usize,
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

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        if !req.width.is_multiple_of(CELL) || !req.height.is_multiple_of(CELL) {
            return Err(Error::Msg(format!(
                "sensenova_u1_8b: {}x{} must be a multiple of {CELL} per side",
                req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate(
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
            on_progress(Progress::Step {
                current: i + 1,
                total: req.count,
            });
            let opts = self.options(req, base_seed.wrapping_add(i as u64));
            let out = if references.is_empty() {
                self.model
                    .generate(&self.tokenizer, &req.prompt, w, h, &opts, None)?
            } else {
                self.model.it2i_generate(
                    &self.tokenizer,
                    &req.prompt,
                    &references,
                    w,
                    h,
                    &opts,
                    None,
                )?
            };
            images.push(decoded_to_image(&out.image)?);
        }
        Ok(GenerationOutput::Images(images))
    }
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

inventory::submit! {
    ModelRegistration { descriptor, load }
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
    fn registered_in_registry() {
        // The `inventory::submit!` is linked into the test binary, so the registry finds the id.
        let found = mlx_gen::registry::generators().any(|r| (r.descriptor)().id == MODEL_ID);
        assert!(
            found,
            "sensenova_u1_8b not registered in the generator registry"
        );
    }

    #[test]
    fn load_rejects_single_file() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/x.safetensors".into()));
        assert!(load(&spec).is_err());
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
    }
}
