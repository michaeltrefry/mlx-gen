//! `Seedvr2Generator` — the [`mlx_gen::Generator`] wiring the SeedVR2 pipeline into `mlx_gen`'s
//! registry (sc-4813). Registered under `seedvr2` (alias) + `seedvr2_3b`.
//!
//! **Surface.** A one-step image **upscaler**: the input LR image arrives as a
//! [`Conditioning::Reference`]; `width`/`height` are the target output size (both ÷16). No prompt,
//! no guidance/CFG (1-step), no LoRA. `spec.weights` is the raw `numz/SeedVR2_comfyUI` checkpoint dir
//! (converted in-memory at load — no Python). Dense bf16 default; `Fp32` honored (the parity path).
//!
//! 7B (needs pixel-mode RoPE) and int8 (Linear-only quant) are tracked follow-ups; only 3B is wired.

use mlx_rs::Dtype;

use mlx_gen::{
    default_seed, gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Result, WeightsSource,
};

use crate::config::DitConfig;
use crate::pipeline::Seedvr2Pipeline;

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
const VAE_SCALE: u32 = 16; // VAE /8 · patch /2
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "seedvr2",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false, // precomputed neg-embed; no prompt surface
            supports_guidance: false,        // one-step, guidance fixed at 1.0
            supports_true_cfg: false,
            conditioning: vec![ConditioningKind::Reference], // the LR input image
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["seedvr2_euler"],
            schedulers: vec!["seedvr2_euler"],
            min_size: VAE_SCALE,
            max_size: 4096,
            max_count: 8,
            mac_only: true,
            supported_quants: &[], // int8 (Linear-only) is a follow-up
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

pub fn descriptor() -> ModelDescriptor {
    descriptor_for(MODEL_ID)
}
pub fn descriptor_3b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_3B)
}

pub struct Seedvr2Generator {
    descriptor: ModelDescriptor,
    pipe: Seedvr2Pipeline,
}

fn load_with(spec: &LoadSpec, id: &'static str) -> Result<Box<dyn Generator>> {
    if spec.control.is_some() || !spec.extra_controls.is_empty() || spec.ip_adapter.is_some() {
        return Err(Error::Msg(format!(
            "{id}: ControlNet / IP-Adapter conditioning is not part of SeedVR2"
        )));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(format!(
            "{id}: LoRA/LoKr adapters are not supported"
        )));
    }
    if spec.quantize.is_some() {
        return Err(Error::Msg(format!(
            "{id}: int8/int4 quantization is not yet wired (follow-up)"
        )));
    }
    let dtype = match spec.precision {
        Precision::Bf16 => Dtype::Bfloat16,
        Precision::Fp32 => Dtype::Float32,
    };
    let dir = match &spec.weights {
        WeightsSource::Dir(p) => p.clone(),
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{id}: expects a numz/SeedVR2_comfyUI checkpoint directory, not a single file"
            )))
        }
    };
    let pipe = Seedvr2Pipeline::load(&dir, DIT_FILE_3B, &DitConfig::seedvr2_3b(), dtype)?;
    Ok(Box::new(Seedvr2Generator {
        descriptor: descriptor_for(id),
        pipe,
    }))
}

impl Generator for Seedvr2Generator {
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

/// The LR input image carried by the request's `Reference` conditioning.
fn reference_image(req: &GenerationRequest) -> Option<&Image> {
    req.conditioning.iter().find_map(|c| match c {
        Conditioning::Reference { image, .. } => Some(image),
        _ => None,
    })
}

impl Seedvr2Generator {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        self.descriptor
            .capabilities
            .validate_request(self.descriptor.id, req)?;
        if reference_image(req).is_none() {
            return Err(Error::Msg(format!(
                "{}: requires a Reference (input) image to upscale",
                self.descriptor.id
            )));
        }
        if !req.width.is_multiple_of(VAE_SCALE) || !req.height.is_multiple_of(VAE_SCALE) {
            return Err(Error::Msg(format!(
                "{}: width/height must be multiples of {VAE_SCALE} (got {}x{})",
                self.descriptor.id, req.width, req.height
            )));
        }
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate_impl(req)?;
        let image = reference_image(req).expect("validated");
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let softness = 0.0; // no request field; the reference default

        let mut out = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            if req.cancel.is_cancelled() {
                return Err(Error::Canceled);
            }
            on_progress(Progress::Step {
                current: 1,
                total: 1,
            });
            let seed = base_seed.wrapping_add(i as u64);
            let img =
                self.pipe
                    .generate(image, req.width as i32, req.height as i32, seed, softness)?;
            on_progress(Progress::Decoding);
            out.push(img);
        }
        Ok(GenerationOutput::Images(out))
    }
}

fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID).map_err(Into::into)
}
fn load_registered_3b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_3B).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_3b, load: load_registered_3b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_seedvr2() {
        let d = descriptor();
        assert_eq!(d.id, MODEL_ID);
        assert_eq!(d.family, "seedvr2");
        assert_eq!(d.backend, "mlx");
        assert_eq!(d.modality, Modality::Image);
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B] {
            let spec = LoadSpec {
                weights: WeightsSource::Dir("/nonexistent/seedvr2".into()),
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
                "{id} should resolve; got: {err}"
            );
        }
    }
}
