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
    default_seed, gen_core, Error, GenerationOutput, GenerationRequest, Generator, LoadSpec,
    ModelDescriptor, ModelRegistration, Precision, Progress, Result, WeightsSource,
};
use mlx_rs::ops::{add, concatenate_axis, multiply, pad, subtract};
use mlx_rs::Array;

use crate::config::{Flux2Variant, DEFAULT_GUIDANCE};
use crate::kv_cache::{CacheMode, Flux2KvCache};
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

pub fn descriptor_klein_9b_kv_edit() -> ModelDescriptor {
    Flux2Variant::Klein9bKvEdit.descriptor()
}

pub fn load_klein_9b(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9b, spec)
}

pub fn load_klein_9b_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bEdit, spec)
}

pub fn load_klein_9b_kv_edit(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    load_variant(Flux2Variant::Klein9bKvEdit, spec)
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
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tok);
        let embeds = te.prompt_embeds(&input_ids, &attention_mask)?;
        let ids = prepare_text_ids(embeds.shape()[1] as usize);
        Ok((embeds, ids))
    }

    /// Edit reference conditioning for **N** images (the fork's `prepare_reference_image_conditioning`):
    /// each image → resize → VAE-encode → crop-to-even → 2×2 patchify → BN-normalize → pack, tagged
    /// with grid ids at `t = 10 + 10·i` (the per-reference time offset), then all refs concatenated
    /// on the sequence axis. Returns `(image_latents [1, Σseq_ref, 128], image_latent_ids
    /// [1, Σseq_ref, 4])`. A single reference (N = 1) reduces to the original `t = 10` path. The
    /// FLUX.2 text encoder is a dense Qwen3 LLM with no vision input, so the prompt embeds are
    /// independent of the references — multi-image conditioning flows ONLY through these tokens.
    fn encode_references(
        &self,
        vae: &Flux2Vae,
        images: &[&mlx_gen::media::Image],
        width: u32,
        height: u32,
    ) -> Result<(Array, Array)> {
        let mut packed: Vec<Array> = Vec::with_capacity(images.len());
        let mut ids: Vec<Array> = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let pre = preprocess_ref_image(image, width, height)?; // NHWC [1,H,W,3]
            let enc = vae.encode_mean(&pre)?; // NHWC [1,H/8,W/8,32]
            let enc = enc.transpose_axes(&[0, 3, 1, 2])?; // → NCHW for the pipeline helpers
            let enc = crop_to_even(&enc)?;
            let patchified = patchify_latents(&enc)?; // [1,128,h,w]
            let normed = vae.bn_normalize_nchw(&patchified)?;
            let sh = patchified.shape();
            packed.push(pack_latents(&normed)?); // [1, seq_ref, 128]
            ids.push(prepare_grid_ids(
                sh[2] as usize,
                sh[3] as usize,
                10 + 10 * i as i32,
            ));
        }
        let packed_refs: Vec<&Array> = packed.iter().collect();
        let id_refs: Vec<&Array> = ids.iter().collect();
        Ok((
            concatenate_axis(&packed_refs, 1)?,
            concatenate_axis(&id_refs, 1)?,
        ))
    }

    /// Collect the ordered edit reference images from the request: a single `Reference`, a
    /// `MultiReference { images }` (N images, sc-2645), or several `Reference`s — flattened in
    /// conditioning order then image order (the fork passes a flat `image_paths` list). At least
    /// one reference is required.
    fn collect_edit_references<'a>(
        &self,
        req: &'a GenerationRequest,
    ) -> Result<Vec<&'a mlx_gen::media::Image>> {
        let mut refs: Vec<&mlx_gen::media::Image> = Vec::new();
        for c in &req.conditioning {
            match c {
                mlx_gen::Conditioning::Reference { image, .. } => refs.push(image),
                mlx_gen::Conditioning::MultiReference { images } => refs.extend(images.iter()),
                _ => {}
            }
        }
        if refs.is_empty() {
            return Err(Error::Msg(format!(
                "{}: edit requires at least one reference image",
                self.descriptor.id
            )));
        }
        Ok(refs)
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

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        validate_request(&self.descriptor, req).map_err(Into::into)
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl Flux2 {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let (tokenizer, te, transformer, vae) = self.parts()?;
        let base_seed = req.seed.unwrap_or_else(default_seed);
        let steps = req.steps.unwrap_or(crate::config::DEFAULT_STEPS) as usize;
        let guidance = req.guidance.unwrap_or(DEFAULT_GUIDANCE);

        // Edit: build the reference-image conditioning from one `Reference` or one `MultiReference`
        // (sc-2645). The transformer sees the joint sequence `[txt, target, ref0, ref1, …]`; its
        // output keeps the leading `target_seq` image tokens.
        let reference = if self.variant.is_edit() {
            let images = self.collect_edit_references(req)?;
            Some(self.encode_references(vae, &images, req.width, req.height)?)
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

        // For an edit, the transformer's image input/ids are `[target, ref]` (or `[target]` only on
        // a cached KV step); its output keeps the image stream, of which we take the leading
        // `target_seq` tokens. txt2img has no ref, so the concat + slice are no-ops.
        // `include_ref=false` drops the reference tokens (the 9b-kv cached step); `cache` threads
        // the per-seed KV cache through the transformer.
        let run = |latents: &Array,
                   embeds: &Array,
                   ids: &Array,
                   ts: f32,
                   include_ref: bool,
                   cache: Option<&Flux2KvCache>|
         -> Result<Array> {
            let target_seq = latents.shape()[1];
            let (hidden, img_ids) = match (&reference, include_ref) {
                (Some((ref_lat, ref_ids)), true) => (
                    concatenate_axis(&[latents, ref_lat], 1)?,
                    concatenate_axis(&[&latent_ids, ref_ids], 1)?,
                ),
                _ => (latents.clone(), latent_ids.clone()),
            };
            let out = transformer.forward_with_cache(&hidden, embeds, &img_ids, ids, ts, cache)?;
            let idx = Array::from_slice(&(0..target_seq).collect::<Vec<i32>>(), &[target_seq]);
            Ok(out.take_axis(&idx, 1)?)
        };

        // 9b-kv edit: cache reference K/V on step 0, reuse on steps 1+ (the ~2.4× speedup). The
        // edit path always has a reference, so `num_ref > 0`.
        let kv_enabled = self.variant.is_kv() && reference.is_some();
        let num_ref = reference
            .as_ref()
            .map(|(r, _)| r.shape()[1] as usize)
            .unwrap_or(0);

        // sc-2963 (rollout of sc-2957): run the MMDiT's fusable elementwise glue (adaLN affine,
        // SwiGLU, gated residual, RoPE rotation) through `mx.compile` — bit-exact (`max|Δ|=0`,
        // compile_parity.rs) and a per-step win at production geometry. Process-global, idempotent.
        crate::transformer::set_compile_glue(true);

        let mut images = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let seed = base_seed.wrapping_add(i as u64);
            let noise = create_noise(seed, req.width, req.height, self.config.in_channels)?;
            // img2img: `(1-σ)·clean + σ·noise` at `σ = sigmas[start_step]`; txt2img: pure noise.
            let mut latents = match &clean_init {
                Some(clean) => add_noise_by_interpolation(clean, &noise, sched.sigmas[start_step])?,
                None => noise,
            };
            // Fresh cache per seed — the cached reference K/V depend on the step-0 target latents.
            let cache = kv_enabled.then(|| {
                Flux2KvCache::new(self.config.num_double_layers, self.config.num_single_layers)
            });
            // img2img skips the first `start_step` steps (the fork loops `range(init_time_step, n)`).
            for (t, &ts) in timesteps.iter().enumerate().skip(start_step) {
                if req.cancel.is_cancelled() {
                    return Err(Error::Msg("generation cancelled".into()));
                }
                // KV step role: the first executed step extracts the reference K/V (running the full
                // `[txt, target, ref]` forward); later steps run `[txt, target]` only and splice the
                // cached ref K/V back in. With no cache, every step includes the reference tokens.
                let (include_ref, cache_ref) = match &cache {
                    Some(c) => {
                        let mode = if t == start_step {
                            CacheMode::Extract
                        } else {
                            CacheMode::Cached
                        };
                        c.configure(mode, num_ref);
                        (mode == CacheMode::Extract, Some(c))
                    }
                    None => (true, None),
                };
                let v = run(
                    &latents,
                    &prompt_embeds,
                    &text_ids,
                    ts,
                    include_ref,
                    cache_ref,
                )?;
                let v = match &negative {
                    Some((neg_embeds, neg_ids)) => {
                        // CFG with the cache mirrors the fork: the same cache feeds both forwards
                        // (the negative extract overwrites the positive's slots). Distilled klein
                        // runs guidance 1.0 → no negative pass, so this is the base path in practice.
                        let vn = run(&latents, neg_embeds, neg_ids, ts, include_ref, cache_ref)?;
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
    // Empty-prompt first so it wins over the shared floor for a bare default request.
    if req.prompt.trim().is_empty() {
        return Err(Error::Msg(format!("{}: prompt is required", desc.id)));
    }
    // The shared capability floor (count, size range, negative/guidance/true_cfg, sampler, scheduler,
    // conditioning) — the same check chroma delegates to (F-100; this dedups flux2's near-verbatim
    // copy and adds the previously-missing scheduler validation).
    desc.capabilities.validate_request(desc.id, req)?;
    // FLUX.2-specific: latent dims must be a multiple of 16 (VAE 8× × patch 2).
    if !req.width.is_multiple_of(16) || !req.height.is_multiple_of(16) {
        return Err(Error::Msg(format!(
            "{}: width and height must be multiples of 16, got {}x{}",
            desc.id, req.width, req.height
        )));
    }
    Ok(())
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` load fns into it.
fn load_klein_9b_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_klein_9b(spec).map_err(Into::into)
}

fn load_klein_9b_edit_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_klein_9b_edit(spec).map_err(Into::into)
}

fn load_klein_9b_kv_edit_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_klein_9b_kv_edit(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b, load: load_klein_9b_registered }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b_edit, load: load_klein_9b_edit_registered }
}

inventory::submit! {
    ModelRegistration { descriptor: descriptor_klein_9b_kv_edit, load: load_klein_9b_kv_edit_registered }
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
    fn rejects_unsupported_scheduler() {
        // F-100: flux2 delegated to the shared floor now validates the scheduler (was silently
        // accepted). The advertised "flow_match_euler" passes.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let err = model
            .validate(&GenerationRequest {
                prompt: "x".into(),
                scheduler: Some("karras".into()),
                ..Default::default()
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("unsupported scheduler"), "got: {err}");
        model
            .validate(&GenerationRequest {
                prompt: "x".into(),
                scheduler: Some("flow_match_euler".into()),
                ..Default::default()
            })
            .unwrap();
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
        assert_eq!(model.collect_edit_references(&req).unwrap().len(), 1);
    }

    #[test]
    fn edit_accepts_multi_reference() {
        // sc-2645: N reference images via `MultiReference`, flattened in order.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "combine these".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        model.validate(&req).unwrap();
        assert_eq!(model.collect_edit_references(&req).unwrap().len(), 3);
    }

    #[test]
    fn edit_without_reference_errors() {
        let model = Flux2::new_for_tests(Flux2Variant::Klein9bEdit);
        let req = GenerationRequest {
            prompt: "make it night".into(),
            ..Default::default()
        };
        let err = model.collect_edit_references(&req).unwrap_err().to_string();
        assert!(err.contains("at least one reference image"));
    }

    #[test]
    fn txt2img_rejects_multi_reference() {
        // Multi-image editing belongs to the edit variant, not txt2img.
        let model = Flux2::new_for_tests(Flux2Variant::Klein9b);
        let req = GenerationRequest {
            prompt: "x".into(),
            conditioning: vec![Conditioning::MultiReference {
                images: vec![Image::default(), Image::default()],
            }],
            ..Default::default()
        };
        let err = model.validate(&req).unwrap_err().to_string();
        assert!(err.contains("conditioning"));
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
