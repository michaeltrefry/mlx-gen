//! FLUX.2-klein provider registration + the txt2img generation path.
//!
//! `load()` assembles the tokenizer, Qwen3 text encoder, MMDiT transformer, and 32-ch VAE from a
//! snapshot directory; `spec.quantize` (Q4/Q8, sc-2643) then quantizes the whole model in place.
//! `generate()` runs the flow-match denoise loop (CFG dual-forward when `guidance > 1`; distilled
//! klein defaults to 1.0 = single forward), then BN-denormalizes + 2×2-unpatchifies + VAE-decodes.
//! Both the txt2img (`flux2_klein_9b`) and single-reference edit (`flux2_klein_9b_edit`) variants
//! share this path.
//!
//! Activations run f32 (matmul(f32, bf16)→f32): dodges the dense 16-bit Metal GEMM bug and is the
//! quality target. Pixel-parity with the fork's bf16 render is therefore not the gate (see the
//! e2e test) — component f32 parity + visual correctness is.

use mlx_gen::array::scalar;
use mlx_gen::image::decoded_to_image;
use mlx_gen::tokenizer::TextTokenizer;
use mlx_gen::{
    default_seed, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec, ModelDescriptor,
    ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply, pad, subtract};
use mlx_rs::Array;

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE};
use crate::pipeline::{
    add_noise_by_interpolation, create_noise, init_time_step, pack_latents, patchify_latents,
    prepare_grid_ids, prepare_text_ids, preprocess_ref_image, schedule, timesteps_x1000,
};
use crate::text_encoder::Qwen3TextEncoder;
use crate::transformer::Flux2Transformer;
use crate::vae::Flux2Vae;
use crate::{loader, Flux2Config};

pub fn descriptor_klein_9b() -> ModelDescriptor {
    Flux2Variant::Klein9b.descriptor()
}

pub fn descriptor_klein_9b_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bEdit.descriptor()
}

pub fn load_klein_9b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

pub fn load_klein_9b_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bEdit, spec)
}

fn load_variant(variant: Flux2Variant, spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    if spec.precision != Precision::Bf16 {
        // The dense path loads at the on-disk dtype and runs f32 activations; an explicit fp32
        // precision override isn't a separate wired mode. Q4/Q8 (sc-2643) go through `spec.quantize`.
        return Err(Error::Msg(format!(
            "{}: only the default precision is wired; drop the precision override (Q4/Q8 = spec.quantize)",
            variant.id()
        )));
    }
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(format!(
                "{} expects a FLUX.2-klein snapshot directory (tokenizer/ text_encoder/ \
                 transformer/ vae/), not a single .safetensors file",
                variant.id()
            )))
        }
    };

    let mut text_encoder = loader::load_text_encoder(root)?;
    let mut transformer = loader::load_transformer(root)?;
    let mut vae = loader::load_vae(root)?;
    // Q4/Q8 quantizes the **whole model** in place after the dense load — the fork's `nn.quantize`
    // over (transformer, text_encoder, vae), group_size 64, every quantizable Linear (+ the text
    // encoder's token Embedding). Full-model scope like Z-Image (sc-2532), unlike Qwen's
    // transformer-only quant (sc-2565) — quant scope is per-fork. The VAE's quantized surface is
    // just its two mid-block attentions (everything else there is Conv/GroupNorm). The dense load
    // runs f32, but `quantize` casts weights to bf16 before packing so the scales byte-match the
    // fork's bf16 `nn.quantize` (sc-2604).
    if let Some(q) = spec.quantize {
        let bits = q.bits();
        transformer.quantize(bits)?;
        text_encoder.quantize(bits)?;
        vae.quantize(bits)?;
    }

    // LoRA/LoKr (sc-2646): applied AFTER quantization, as forward-time residuals over the
    // (possibly quantized) transformer — fork-faithful, transformer-only. No-op when empty.
    if !spec.adapters.is_empty() {
        crate::adapters::apply_flux2_adapters(&mut transformer, &spec.adapters)?;
    }

    Ok(Box::new(Flux2 {
        descriptor: variant.descriptor(),
        variant,
        config: variant.config(),
        tokenizer: Some(loader::load_tokenizer(root)?),
        text_encoder: Some(text_encoder),
        transformer: Some(transformer),
        vae: Some(vae),
    }))
}

/// The FLUX.2-klein generator.
pub struct Flux2 {
    descriptor: ModelDescriptor,
    variant: Flux2Variant,
    config: Flux2Config,
    tokenizer: Option<TextTokenizer>,
    text_encoder: Option<Qwen3TextEncoder>,
    transformer: Option<Flux2Transformer>,
    vae: Option<Flux2Vae>,
}

impl Flux2 {
    /// Construct a weightless instance for validation tests.
    pub fn new_for_tests(variant: Flux2Variant) -> Self {
        Self {
            descriptor: variant.descriptor(),
            variant,
            config: variant.config(),
            tokenizer: None,
            text_encoder: None,
            transformer: None,
            vae: None,
        }
    }

    fn parts(
        &self,
    ) -> Result<(
        &TextTokenizer,
        &Qwen3TextEncoder,
        &Flux2Transformer,
        &Flux2Vae,
    )> {
        let err = |what: &str| Error::Msg(format!("{}: {what} is not loaded", self.descriptor.id));
        Ok((
            self.tokenizer.as_ref().ok_or_else(|| err("tokenizer"))?,
            self.text_encoder
                .as_ref()
                .ok_or_else(|| err("text encoder"))?,
            self.transformer
                .as_ref()
                .ok_or_else(|| err("transformer"))?,
            self.vae.as_ref().ok_or_else(|| err("VAE"))?,
        ))
    }

    /// Encode a prompt → `(prompt_embeds [1,512,joint], text_ids [1,512,4])`.
    fn encode(
        &self,
        tokenizer: &TextTokenizer,
        te: &Qwen3TextEncoder,
        prompt: &str,
    ) -> Result<(Array, Array)> {
        let tok = tokenizer.tokenize(prompt)?;
        let embeds = te.prompt_embeds(&tok.input_ids, &tok.attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }

    /// Edit reference conditioning: resize → VAE-encode → patchify → BN-normalize → pack, plus the
    /// grid ids at `t = 10` (the per-reference time offset). Returns `(image_latents [1,seq_ref,128],
    /// image_latent_ids [1,seq_ref,4])`. Mirrors the fork's `prepare_reference_image_conditioning`.
    fn encode_reference(
        &self,
        vae: &Flux2Vae,
        image: &mlx_gen::media::Image,
        width: u32,
        height: u32,
    ) -> Result<(Array, Array)> {
        let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
        let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
        let enc = crop_to_even(&enc)?;
        let patchified = patchify_latents(&enc)?; // [1,128,h,w]
        let normed = vae.bn_normalize_nchw(&patchified)?;
        let packed = pack_latents(&normed)?; // [1, seq_ref, 128]
        let sh = patchified.shape();
        let ids = prepare_grid_ids(sh[2] as usize, sh[3] as usize, 10);
        Ok((packed, ids))
    }

    /// Extract the single img2img init image + its strength from the txt2img request. The
    /// per-reference strength wins over `req.strength`. txt2img img2img conditions on exactly one
    /// init image, so more than one `Reference` is an error (multi-reference is the edit variant +
    /// `MultiReference`, sc-2645). Returns `None` for pure txt2img.
    fn resolve_reference<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Option<(&'a mlx_gen::media::Image, Option<f32>)>> {
        let mut reference = None;
        for c in &req.conditioning {
            if let mlx_gen::Conditioning::Reference { image, strength } = c {
                if reference.is_some() {
                    return Err(Error::Msg(format!(
                        "{}: multiple reference images are not supported (single img2img init only)",
                        self.descriptor.id
                    )));
                }
                reference = Some((image, strength.or(req.strength)));
            }
        }
        Ok(reference)
    }

    /// img2img init conditioning: resize → VAE-encode → NCHW → crop-to-even → center-crop/pad to the
    /// target latent grid → 2×2 patchify → BN-normalize → pack. Returns the **clean** packed latents
    /// `[1, lat_h·lat_w, 128]` (seed-independent — blended with the per-seed noise in `generate`).
    /// Mirrors the fork's `_prepare_img2img_latents` (minus the noise blend); same encode chain as
    /// `encode_reference`, plus the `_match_latent_spatial_size` step and the txt2img grid ids.
    fn encode_init_latents(
        &self,
        vae: &Flux2Vae,
        image: &mlx_gen::media::Image,
        width: u32,
        height: u32,
    ) -> Result<Array> {
        let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
        let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
        let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
        let enc = crop_to_even(&enc)?;
        // Target the denoise latent grid: `latent_h·2 × latent_w·2 = H/8 × W/8`. A no-op at the
        // standard multiple-of-16 sizes (encoded H/8 already equals the target).
        let enc = match_latent_spatial_size(&enc, (height / 8) as i32, (width / 8) as i32)?;
        let patchified = patchify_latents(&enc)?; // [1,128,h,w]
        let normed = vae.bn_normalize_nchw(&patchified)?;
        pack_latents(&normed) // [1, lat_h·lat_w, 128]
    }
}

/// Crop a NCHW latent's spatial dims down to even (the fork's `crop_to_even_spatial`), so the 2×2
/// patchify divides cleanly. A no-op at the standard multiple-of-16 sizes.
fn crop_to_even(x: &Array) -> Result<Array> {
    let sh = x.shape();
    let mut x = x.clone();
    if sh[2] % 2 != 0 {
        let idx = Array::from_slice(&(0..sh[2] - 1).collect::<Vec<i32>>(), &[sh[2] - 1]);
        x = x.take_axis(&idx, 2)?;
    }
    if sh[3] % 2 != 0 {
        let idx = Array::from_slice(&(0..sh[3] - 1).collect::<Vec<i32>>(), &[sh[3] - 1]);
        x = x.take_axis(&idx, 3)?;
    }
    Ok(x)
}

/// Center-crop or symmetric-pad a NCHW latent's spatial dims to `(target_h, target_w)` — the fork's
/// `_match_latent_spatial_size`. A no-op at the standard multiple-of-16 sizes (the VAE-encoded H/8
/// already equals the `latent_h·2` target); guards odd / mismatched user images.
fn match_latent_spatial_size(x: &Array, target_h: i32, target_w: i32) -> Result<Array> {
    let mut x = x.clone();
    let (h, w) = (x.shape()[2], x.shape()[3]);
    if h != target_h {
        if h > target_h {
            let off = (h - target_h) / 2;
            let idx = Array::from_slice(&(off..off + target_h).collect::<Vec<i32>>(), &[target_h]);
            x = x.take_axis(&idx, 2)?;
        } else {
            let before = (target_h - h) / 2;
            let after = (target_h - h) - before;
            x = pad(
                &x,
                &[(0, 0), (0, 0), (before, after), (0, 0)][..],
                None,
                None,
            )?;
        }
    }
    if w != target_w {
        if w > target_w {
            let off = (w - target_w) / 2;
            let idx = Array::from_slice(&(off..off + target_w).collect::<Vec<i32>>(), &[target_w]);
            x = x.take_axis(&idx, 3)?;
        } else {
            let before = (target_w - w) / 2;
            let after = (target_w - w) - before;
            x = pad(
                &x,
                &[(0, 0), (0, 0), (0, 0), (before, after)][..],
                None,
                None,
            )?;
        }
    }
    Ok(x)
}

impl Generator for Flux2 {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> Result<()> {
        validate_request(&self.descriptor, req)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let (tokenizer, te, transformer, vae) = self.parts()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(crate::config::DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);

        // Edit: build the reference-image conditioning (single reference for this story; multi =
        // sc-2645). The transformer sees the joint sequence `[txt, target, ref]`.
        let reference = if self.variant.is_edit() {
            let image = req
                .conditioning
                .iter()
                .find_map(|c| match c {
                    mlx_gen::Conditioning::Reference { image, .. } => Some(image),
                    _ => None,
                })
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "{}: edit requires a reference image",
                        self.descriptor.id
                    ))
                })?;
            Some(self.encode_reference(vae, image, req.width, req.height)?)
        } else {
            None
        };

        // img2img (txt2img variant): a single `Reference` init image seeds the latents via the
        // noise blend at `sigmas[start_step]`, with the denoise loop starting at `start_step`
        // (= the fork's `_prepare_img2img_latents` + `Config.init_time_step`). The edit variant
        // consumes its `Reference` above (token concat), so img2img is txt2img-only.
        let img2img = if self.variant.is_edit() {
            None
        } else {
            self.resolve_reference(req)?
        };
        let start_step = match &img2img {
            Some((_, strength)) => init_time_step(steps, *strength),
            None => 0,
        };

        let (prompt_embeds, text_ids) = self.encode(tokenizer, te, &req.prompt)?;
        // klein is distilled (guidance 1.0); CFG dual-forward only kicks in for base variants.
        let negative = if guidance > 1.0 {
            Some(self.encode(tokenizer, te, " ")?)
        } else {
            None
        };

        let sched = schedule(steps, req.width, req.height);
        let timesteps = timesteps_x1000(&sched);
        let lat_h = (req.height / 16) as usize;
        let lat_w = (req.width / 16) as usize;
        let latent_ids = prepare_grid_ids(lat_h, lat_w, 0);
        let in_channels = self.config.in_channels as i32;

        // The img2img clean init latents are seed-independent — encode once, blend with per-seed
        // noise below. `None` for pure txt2img (or strength ≤ 0, where `start_step == 0`).
        let clean_init = match &img2img {
            Some((image, _)) if start_step > 0 => {
                Some(self.encode_init_latents(vae, image, req.width, req.height)?)
            }
            _ => None,
        };

        // For an edit, the transformer's image input/ids are `[target, ref]`; its output keeps the
        // image stream, of which we take the leading `target_seq` tokens. txt2img has no ref, so the
        // concat + slice are no-ops.
        let forward = |latents: &Array, embeds: &Array, ids: &Array, ts: f32| -> Result<Array> {
            let target_seq = latents.shape()[1];
            let (hidden, img_ids) = match &reference {
                Some((ref_lat, ref_ids)) => (
                    concatenate_axis(&[latents, ref_lat], 1)?,
                    concatenate_axis(&[&latent_ids, ref_ids], 1)?,
                ),
                None => (latents.clone(), latent_ids.clone()),
            };
            let out = transformer.forward(&hidden, embeds, &img_ids, ids, ts)?;
            let idx = Array::from_slice(&(0..target_seq).collect::<Vec<i32>>(), &[target_seq]);
            Ok(out.take_axis(&idx, 1)?)
        };

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, req.width, req.height, self.config.in_channels)?;
            // img2img: `(1-σ)·clean + σ·noise` at `σ = sigmas[start_step]`; txt2img: pure noise.
            let mut latents = match &clean_init {
                Some(clean) => add_noise_by_interpolation(clean, &noise, sched.sigmas[start_step])?,
                None => noise,
            };
            // img2img skips the first `start_step` steps (the fork loops `range(init_time_step, n)`).
            for (t, &ts) in timesteps.iter().enumerate().skip(start_step) {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                let v = forward(&latents, &prompt_embeds, &text_ids, ts)?;
                let v = match &negative {
                    Some((neg_embeds, neg_ids)) => {
                        let vn = forward(&latents, neg_embeds, neg_ids, ts)?;
                        // noise = neg + guidance·(pos − neg)
                        add(&vn, &multiply(&subtract(&v, &vn)?, scalar(guidance))?)?
                    }
                    None => v,
                };
                latents = sched.step(&latents, &v, t)?;
                on_progress(Progress::Step {
                    current: t as u32 + 1,
                    total: steps as u32,
                });
            }
            on_progress(Progress::Decoding);
            let packed = latents.reshape(&[1, lat_h as i32, lat_w as i32, in_channels])?;
            let decoded = vae.decode_packed_latents(&packed)?; // NHWC [1,H,W,3]
            let nchw = decoded.transpose_axes(&[0, 3, 1, 2])?;
            images.push(decoded_to_image(&nchw)?);
        }
        Ok(GenerationOutput::Images(images))
    }
}

fn validate_request(desc: &ModelDescriptor, req: &GenerationRequest) -> Result<()> {
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of 16, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    let caps = &desc.capabilities;
    if req.width < caps.min_size
        || req.height < caps.min_size
        || req.width > caps.max_size
        || req.height > caps.max_size
    {
        return Err(Error::Msg(format!(
            "{}: size {}x{} outside supported range {}..={}",
            desc.id, req.width, req.height, caps.min_size, caps.max_size
        )));
    }
    if req.count == 0 || req.count > caps.max_count {
        return Err(Error::Msg(format!(
            "{}: count must be 1..={}",
            desc.id, caps.max_count
        )));
    }
    if req.negative_prompt.is_some() && !caps.supports_negative_prompt {
        return Err(Error::Msg(format!(
            "{}: negative prompts are not supported by FLUX.2",
            desc.id
        )));
    }
    if req.true_cfg.is_some() && !caps.supports_true_cfg {
        return Err(Error::Msg(format!(
            "{}: true_cfg is not supported",
            desc.id
        )));
    }
    for c in &req.conditioning {
        let kind = conditioning_kind(c);
        if !caps.accepts(kind) {
            return Err(Error::Msg(format!(
                "{}: conditioning {kind:?} is not supported by this variant",
                desc.id
            )));
        }
    }
    Ok(())
}

fn conditioning_kind(c: &mlx_gen::Conditioning) -> mlx_gen::ConditioningKind {
    use mlx_gen::{Conditioning as C, ConditioningKind as K};
    match c {
        C::Reference { .. } => K::Reference,
        C::MultiReference { .. } => K::MultiReference,
        C::ReduxRefs { .. } => K::ReduxRefs,
        C::Control { .. } => K::Control,
        C::Depth { .. } => K::Depth,
        C::Mask { .. } => K::Mask,
    }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b, load: load_klein_9b }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b_edit, load: load_klein_9b_edit }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{FLUX2_KLEIN_9B_EDIT_ID, FLUX2_KLEIN_9B_ID};
    use mlx_gen::media::Image;
    use mlx_gen::Conditioning;

    #[test]
    fn validates_basic_txt2img_request() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "a hummingbird".into(),
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn rejects_empty_prompt() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest::default();
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("prompt is required"));
    }

    #[test]
    fn rejects_non_multiple_of_16() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            width: 1023,
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("multiples of 16"));
    }

    #[test]
    fn txt2img_accepts_reference_conditioning() {
        // A `Reference` on the txt2img variant is an img2img init image (sc-2644).
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn txt2img_rejects_multiple_references() {
        // img2img conditions on exactly one init image; the resolver rejects more than one.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![
                Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.6),
                },
                Conditioning::Reference {
                    image: Image::default(),
                    strength: Some(0.6),
                },
            ],
            ..Default::default()
        };
        let err = model.resolve_reference(&req).unwrap_err().to_string();
        assert!(err.contains("multiple reference images"));
    }

    #[test]
    fn edit_accepts_single_reference() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            conditioning: vec![Conditioning::Reference {
                image: Image::default(),
                strength: None,
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
    }

    #[test]
    fn generate_without_weights_errors_not_loaded() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            ..Default::default()
        };
        let mut progress = |_p: Progress| {};
        let err = model.generate(&req, &mut progress).unwrap_err().to_string();
        assert!(err.contains("not loaded"));
    }

    #[test]
    fn ids_match_expected() {
        assert_eq!(descriptor_klein_9b().id, FLUX2_KLEIN_9B_ID);
        assert_eq!(descriptor_klein_9b_edit().id, FLUX2_KLEIN_9B_EDIT_ID);
    }
}
