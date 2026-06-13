//! `Seedvr2Generator` — the [`mlx_gen::Generator`] wiring the SeedVR2 pipeline into `mlx_gen`'s
//! registry (sc-4813 image, sc-4814 video). Registered under `seedvr2` (alias) + `seedvr2_3b`.
//!
//! **Surface.** A one-step super-resolution **upscaler** over image **and** video (`Modality::Both`),
//! dispatched on the request's conditioning:
//!   * [`Conditioning::Reference`] — the LR input image → [`GenerationOutput::Images`];
//!   * [`Conditioning::VideoClip`] — the LR input frame sequence → [`GenerationOutput::Video`]
//!     (temporal chunking + overlap cross-fade + a memory-budgeted chunk sizer; sc-4814).
//!
//! `width`/`height` are the target output size (both ÷16). No prompt, no guidance/CFG (1-step), no
//! LoRA. `spec.weights` is the raw `numz/SeedVR2_comfyUI` checkpoint dir (converted in-memory at
//! load — no Python). Dense bf16 default; `Fp32` honored (the parity path). Video `fps` passes
//! through `req.fps` (the worker supplies the source cadence; audio mux is the worker's job).
//!
//! 3B (default) + 7B (pixel-mode RoPE — sc-5197) are wired; `spec.quantize` Q4/Q8 quantizes the DiT
//! Linears at load (sc-5198).

use mlx_rs::Dtype;

use mlx_gen::{
    default_seed, gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, ModelRegistration,
    Precision, Progress, Quant, Result, WeightsSource,
};

use crate::config::DitConfig;
use crate::pipeline::Seedvr2Pipeline;

pub const MODEL_ID: &str = "seedvr2";
pub const MODEL_ID_3B: &str = "seedvr2_3b";
pub const MODEL_ID_7B: &str = "seedvr2_7b";
const VAE_SCALE: u32 = 16; // VAE /8 · patch /2
const DIT_FILE_3B: &str = "seedvr2_ema_3b_fp16.safetensors";
const DIT_FILE_7B: &str = "seedvr2_ema_7b_fp16.safetensors";
/// Output fps when the request omits one (the worker normally supplies the source cadence).
const DEFAULT_FPS: u32 = 24;

/// The DiT checkpoint file + transformer config for a registered id (3B default; 7B is the
/// pixel-mode-RoPE variant — sc-5197). The VAE is shared across both.
fn variant(id: &str) -> (&'static str, DitConfig) {
    if id == MODEL_ID_7B {
        (DIT_FILE_7B, DitConfig::seedvr2_7b())
    } else {
        (DIT_FILE_3B, DitConfig::seedvr2_3b())
    }
}

fn descriptor_for(id: &'static str) -> ModelDescriptor {
    ModelDescriptor {
        id,
        family: "seedvr2",
        backend: "mlx",
        modality: Modality::Both, // image (Reference) + video (VideoClip) upscaling
        capabilities: Capabilities {
            supports_negative_prompt: false, // precomputed neg-embed; no prompt surface
            supports_guidance: false,        // one-step, guidance fixed at 1.0
            supports_true_cfg: false,
            // the LR input image (image upscale) or LR frame sequence (video upscale)
            conditioning: vec![ConditioningKind::Reference, ConditioningKind::VideoClip],
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["seedvr2_euler"],
            schedulers: vec!["seedvr2_euler"],
            min_size: VAE_SCALE,
            max_size: 4096,
            max_count: 8,
            mac_only: true,
            supported_quants: &[Quant::Q4, Quant::Q8], // Linear-only DiT quant (sc-5198)
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
pub fn descriptor_7b() -> ModelDescriptor {
    descriptor_for(MODEL_ID_7B)
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
    let (dit_file, cfg) = variant(id);
    let mut pipe = Seedvr2Pipeline::load(&dir, dit_file, &cfg, dtype)?;
    // sc-5198: Q4/Q8 quantize the DiT Linears at load (the VAE stays dense).
    if let Some(q) = spec.quantize {
        pipe.quantize(q.bits())?;
    }
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
        let has_video = req.video_clips().iter().any(|c| !c.frames.is_empty());
        if !has_video && reference_image(req).is_none() {
            return Err(Error::Msg(format!(
                "{}: requires a Reference image (image upscale) or a non-empty VideoClip frame \
                 sequence (video upscale)",
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
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let softness = 0.0; // no request field; the reference default

        // Video upscale: a VideoClip carries the LR source frame sequence → one upscaled clip.
        if let Some(clip) = req.video_clips().into_iter().next() {
            on_progress(Progress::Step {
                current: 1,
                total: 1,
            });
            let frames = self.pipe.generate_video(
                clip.frames,
                req.width as i32,
                req.height as i32,
                base_seed,
                softness,
                None,
            )?;
            on_progress(Progress::Decoding);
            return Ok(GenerationOutput::Video {
                frames,
                fps: req.fps.unwrap_or(DEFAULT_FPS),
                audio: None,
            });
        }

        let image = reference_image(req).expect("validated");
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
fn load_registered_7b(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_with(spec, MODEL_ID_7B).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_registered }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_3b, load: load_registered_3b }
}
inventory::submit! {
    ModelRegistration { descriptor: descriptor_7b, load: load_registered_7b }
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
        assert_eq!(d.modality, Modality::Both); // image (Reference) + video (VideoClip)
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::Reference));
        assert!(d
            .capabilities
            .conditioning
            .contains(&ConditioningKind::VideoClip));
        assert!(!d.capabilities.supports_guidance);
        assert!(d.capabilities.mac_only);
    }

    #[test]
    fn both_ids_resolve_in_registry() {
        for id in [MODEL_ID, MODEL_ID_3B, MODEL_ID_7B] {
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
