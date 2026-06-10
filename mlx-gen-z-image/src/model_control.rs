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
    self, denoise_control_with_progress, encode_control_context, encode_init_latents,
    init_time_step,
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
    // LoRA/LoKr (sc-2602): install onto the composed base DiT (the control branch is not an adapter
    // target). Same load-time, post-quantize, residual-over-base path as the plain turbo. No-op when
    // `spec.adapters` is empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_z_image_adapters(&mut transformer, &spec.adapters)?;
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
        let reference = pipeline::resolve_reference(req, MODEL_ID)?;
        let start_step = match reference {
            Some((_, strength)) => init_time_step(steps, strength),
            None => 0,
        };
        let is_img2img = start_step > 0;

        // Prompt → cap_feats. The fork's control path is **mixed precision**, NOT pure bf16: it feeds
        // the latents (`x`) and `cap_feats` as bf16 but `control_context` as **f32** (sc-2720, verified
        // against the fork). The f32 control branch then promotes the bf16 image/caption stream to f32
        // when its hints are added, and `latents += dt·velocity` makes the latents f32 after step 0 —
        // so most of the loop runs f32. We match that exactly: bf16 cap (txt2img) + f32 control_context
        // below. (img2img keeps f32 cap, mirroring the base img2img; the DiT promotes per-op either way.)
        let cap =
            pipeline::encode_prompt(&self.tokenizer, &self.text_encoder, &req.prompt, MODEL_ID)?;
        let cap = if is_img2img {
            cap
        } else {
            // PARITY-BF16 (sc-2609): round the text embeddings to bf16 to match the fork's cap_feats.
            cap.as_dtype(Dtype::Bfloat16)?
        };

        // Static shift=3.0 schedule (shared with the base turbo, sc-2536) — build once.
        let scheduler = FlowMatchEuler::for_static_shift(steps, SCHEDULE_SHIFT);

        // The 33ch control context is constant across steps + the batch — build once. It stays **f32**
        // (the fork feeds it f32, which promotes the whole control branch to f32 — see the forward).
        let control_context =
            encode_control_context(&self.vae, control_image, req.width, req.height)?;

        // VAE-encode the init image once too: like control_context, the clean img2img latents depend
        // only on the init image + dims, not the per-image seed, so they're constant across the batch
        // (F-034). Only the noise (and its blend) vary per image.
        let clean = if is_img2img {
            let (image, _) = reference.expect("is_img2img implies a reference");
            Some(encode_init_latents(
                &self.vae, image, req.width, req.height,
            )?)
        } else {
            None
        };

        // Per-image batch render shared with the base variant (F-035); the control branch's only
        // difference is the `denoise_control_with_progress` step threading the f32 control context +
        // scale (the mixed-precision dtype flow, sc-2720, is preserved inside the closure).
        let images = pipeline::render_batch(
            &self.vae,
            &scheduler,
            clean.as_ref(),
            start_step,
            base_seed,
            req,
            on_progress,
            |latents, op| {
                denoise_control_with_progress(
                    &self.transformer,
                    &scheduler,
                    latents,
                    &cap,
                    &control_context,
                    control_scale,
                    start_step,
                    &req.cancel,
                    op,
                )
            },
        )?;
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
