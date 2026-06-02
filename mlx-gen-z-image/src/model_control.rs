//! `ZImageTurboControl` — the Z-Image-turbo **Fun-Controlnet-Union** variant (sc-2349 / sc-2257):
//! strict pose (VACE-style) conditioning via `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1`,
//! registered as its own `Generator` (`z_image_turbo_control`).
//!
//! Identical to [`crate::model::ZImageTurbo`] except the transformer is a
//! [`ZImageControlTransformer`] (base DiT + control branch) and `generate` threads a VAE-encoded
//! control context through it. [`load`] needs the base snapshot (`spec.weights`) **and** the
//! control checkpoint (`spec.control`); it applies both dense, then quantizes the whole transformer
//! together (the fork's `d32454c` ordering — quantizing before the overlay would leave the control
//! Linears unable to accept their real weights). The control patch embedder stays dense (its
//! in-features is not divisible by the quant group size).
//!
//! Parity-proven against the frozen Python fork (sc-2257): the control branch is bit-identical to
//! the base transformer at `control_context_scale = 0`, and the full control render matches the
//! fork's control golden — see `tests/z_control_transformer.rs` and `tests/control_real_weights.rs`.

use mlx_gen::array::host_i32;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Capabilities, Conditioning, ConditioningKind, Error, FlowMatchEuler,
    GenerationOutput, GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::Dtype;

use crate::control_transformer::ZImageControlTransformer;
use crate::loader;
use crate::model::{validate_request, DEFAULT_STEPS, SCHEDULE_SHIFT};
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, decoded_to_image, denoise_control_with_progress,
    encode_control_context, encode_init_latents, init_time_step, slice_valid, unpack_latents,
};
use crate::text_encoder::TextEncoder;
use crate::vae::Vae;

/// Registry id for the Z-Image-turbo Fun-Controlnet-Union variant.
pub const MODEL_ID: &str = "z_image_turbo_control";

/// The control variant's identity + capabilities. Same distilled turbo base (no CFG / negative
/// prompt) as `z_image_turbo`, plus `Control` conditioning (the required pose/union skeleton) and
/// `Reference` (an optional img2img init — the fork's `generate_image` accepts both).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "z-image",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            supports_guidance: false,
            supports_true_cfg: false,
            // Control (required) + an optional img2img Reference init.
            conditioning: vec![ConditioningKind::Control, ConditioningKind::Reference],
            supports_lora: true,
            supports_lokr: true,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
        },
    }
}

/// A loaded control generator: base components + the control transformer assembled from the base
/// snapshot and the Fun-Controlnet-Union overlay.
pub struct ZImageTurboControl {
    descriptor: ModelDescriptor,
    tokenizer: TextTokenizer,
    text_encoder: TextEncoder,
    transformer: ZImageControlTransformer,
    vae: Vae,
}

/// Construct a [`ZImageTurboControl`] from a [`LoadSpec`].
///
/// `spec.weights` must be a [`WeightsSource::Dir`] base `Tongyi-MAI/Z-Image-Turbo` snapshot, and
/// `spec.control` (required) the Fun-Controlnet-Union checkpoint (a single `.safetensors` `File`,
/// or a `Dir` of them). Weights load dense (bf16); `spec.quantize` (Q4/Q8) then quantizes the whole
/// transformer (base + control, group_size 64) plus the text encoder + VAE — the fork's whole-model
/// quant, with the control patch embedder left dense (its in-features is not a multiple of 64).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        return Err(Error::Msg(
            "z_image_turbo_control: only dense bf16 is wired (the text encoder runs f32 \
             internally); drop the precision override"
                .into(),
        ));
    }
    if !spec.adapters.is_empty() {
        return Err(Error::Msg(
            "z_image_turbo_control: LoRA/LoKr adapter application is not yet wired into load() — \
             the core seam (LoadSpec.adapters → adapters::loader::apply_adapter_specs) exists, but \
             the Z-Image key→module map lands in sc-2602"
                .into(),
        ));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => return Err(Error::Msg(
            "z_image_turbo_control expects a base snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/) as `weights`, not a single .safetensors file"
                .into(),
        )),
    };
    let control = spec.control.as_ref().ok_or_else(|| {
        Error::Msg(
            "z_image_turbo_control requires the Fun-Controlnet-Union weights — set \
             LoadSpec::control (e.g. with_control(WeightsSource::File(...)))"
                .into(),
        )
    })?;

    // Base + control applied dense first, THEN quantize together (the fork's ordering): quantizing
    // before the overlay would replace the control Linears with QuantizedLinear that can't accept
    // the raw bf16 control weights.
    let mut transformer = loader::load_control_transformer(root, control)?;
    let mut text_encoder = loader::load_text_encoder(root)?;
    let mut vae = loader::load_vae(root)?;
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }
    Ok(Box::new(ZImageTurboControl {
        descriptor: descriptor(),
        tokenizer: loader::load_tokenizer(root)?,
        text_encoder,
        transformer,
        vae,
    }))
}

impl ZImageTurboControl {
    /// Prompt → `cap_feats` (f32): tokenize with the Qwen chat template, run the text encoder, slice
    /// off the padded tail. (Identical to the base model's `encode_prompt`.)
    fn encode_prompt(&self, prompt: &str) -> Result<mlx_rs::Array> {
        let t = self.tokenizer.tokenize(prompt)?;
        // Guard on shape before any host readback (an empty prompt tokenizes to `[1, 0]`) — the
        // base model's panic-safe encode boundary (F-001/2/7). `validate_request` already rejects
        // an empty prompt; this is defense-in-depth.
        if t.input_ids.shape()[1] == 0 {
            return Err(Error::Msg("z_image_turbo_control: empty prompt".into()));
        }
        let num_valid: i32 = host_i32(&t.attention_mask)?.iter().sum();
        if num_valid == 0 {
            return Err(Error::Msg("z_image_turbo_control: empty prompt".into()));
        }
        let enc = self.text_encoder.forward(&t.input_ids, &t.attention_mask)?;
        slice_valid(&enc, num_valid)
    }

    /// Extract the (required) control image + its `control_context_scale` from the request. The
    /// Fun-Controlnet-Union is a *union* ControlNet (pose/canny/depth share one VAE-encoded control
    /// path), so any [`mlx_gen::ControlKind`] is accepted — the pose skeleton is the validated use.
    fn resolve_control<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        let mut found = None;
        for c in &req.conditioning {
            if let Conditioning::Control { image, scale, .. } = c {
                if found.is_some() {
                    return Err(Error::Msg(
                        "z_image_turbo_control: a single control image is supported".into(),
                    ));
                }
                found = Some((image, *scale));
            }
        }
        found.ok_or_else(|| {
            Error::Msg(
                "z_image_turbo_control requires a Control conditioning (the pose/union skeleton)"
                    .into(),
            )
        })
    }

    /// Extract the optional img2img init image + strength (the fork's `generate_image(image_path,
    /// image_strength)` — control can run on top of an init image). Per-reference strength wins over
    /// `req.strength`; more than one `Reference` is an error.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(
                        "z_image_turbo_control: multiple reference images are not supported (single \
                         img2img init only)"
                            .into(),
                    ));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }
}

impl Generator for ZImageTurboControl {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability checks (size/count/guidance/negative/accepted conditioning), then the
        // control-specific requirement that a Control conditioning is present.
        validate_request(&self.descriptor.capabilities, req)?;
        if !req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Control { .. }))
        {
            return Err(Error::Msg(
                "z_image_turbo_control requires a Control conditioning (the pose/union skeleton)"
                    .into(),
            ));
        }
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;

        let steps = req.steps.unwrap_or(DEFAULT_STEPS) as usize;
        let base_seed = req.seed.unwrap_or_else(default_seed);

        // Required pose/union control + optional img2img init.
        let (control_image, control_scale) = self.resolve_control(req)?;
        let reference = self.resolve_reference(req)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // Prompt → cap_feats (f32). The control path runs **f32** throughout (like img2img): the
        // control branch's patch embedder is a K≤512 dense GEMM that the pmetal NAX build mis-runs
        // in bf16, and f32 also sidesteps the bf16-kernel toolchain residual (source-built MLX vs
        // the fork's wheel) that otherwise compounds over the 8-step loop — f32 lands materially
        // closer to the fork's golden (latent peak-rel 0.14 vs 0.27 for bf16). Seeded noise is still
        // bf16-rounded to reproduce the fork's exact seeded sample, then promoted to f32.
        let cap = self.encode_prompt(&req.prompt)?;

        // Static shift=3.0 schedule (shared with the base turbo, sc-2536) — build once.
        let scheduler = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);

        // The 33ch control context is constant across steps + the batch — build once (f32).
        let control_context =
            encode_control_context(&self.vae, control_image, req.width, req.height)?;

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            // bf16-round to match the fork's seeded sample exactly, then promote to f32 for the loop.
            // PARITY-BF16 (sc-2609): bf16 matches the fork's seed→image mapping; f32 is a different
            // (higher-precision) realization, not just sharper. Revisit with the other f32 flips.
            let noise = create_noise(seed, req.width, req.height)?.as_dtype(Dtype::Bfloat16)?;
            let latents = if is_img2img {
                let (image, _) = reference.expect("is_img2img implies a reference");
                let clean = encode_init_latents(&self.vae, image, req.width, req.height)?;
                let sigma = scheduler.sigmas[start_step];
                add_noise_by_interpolation(&clean, &noise, sigma)?
            } else {
                noise.as_dtype(Dtype::Float32)?
            };
            let latents = denoise_control_with_progress(
                &self.transformer,
                &scheduler,
                latents,
                &cap,
                &control_context,
                control_scale,
                start_step,
                &req.cancel,
                on_progress,
            )?;

            on_progress(Progress::Decoding);
            let unpacked = unpack_latents(&latents)?;
            let sh = unpacked.shape();
            let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]])?;
            let decoded = self.vae.decode(&latent5)?.as_dtype(Dtype::Float32)?;
            images.push(decoded_to_image(&decoded)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

inventory::submit! {
    ModelRegistration { descriptor, load }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_is_z_image_turbo_control() {
        let d = descriptor();
        assert_eq!(d.id, "z_image_turbo_control");
        assert_eq!(d.family, "z-image");
        assert!(d.capabilities.accepts(ConditioningKind::Control));
        assert!(d.capabilities.accepts(ConditioningKind::Reference));
        assert!(!d.capabilities.supports_guidance);
    }

    #[test]
    fn load_rejects_missing_control_weights() {
        // Without `spec.control`, load must fail on the missing control weights (not on the
        // missing snapshot) — proving the control overlay is wired as a hard requirement.
        let spec = LoadSpec::new(WeightsSource::Dir("/nonexistent".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("Fun-Controlnet-Union"), "got: {err}");
    }

    #[test]
    fn load_rejects_single_file_base() {
        let spec = LoadSpec::new(WeightsSource::File("/tmp/z.safetensors".into()))
            .with_control(WeightsSource::File("/tmp/control.safetensors".into()));
        let err = load(&spec).err().expect("expected an error").to_string();
        assert!(err.contains("base snapshot directory"), "got: {err}");
    }
}
