//! `mlx-gen-svd` model entry: the `svd_xt` Stable-Video-Diffusion descriptor, the `load` from a
//! checkpoint snapshot (the `vae/` + `unet/` + `image_encoder/` subdirs), and the [`Generator`] that
//! drives the S4 [`SvdPipeline`] for image→video.
//!
//! image→video is wired via a single [`Conditioning::Reference`] image: it is CLIP-encoded for the
//! UNet cross-attention conditioning and (noise-augmented) VAE-encoded into the per-frame image
//! latent that is channel-concatenated into the UNet input. `motion_bucket_id` / `noise_aug_strength` /
//! `conditioning_fps` (the motion cadence baked into `added_time_ids`) / `decode_chunk_size` /
//! `frames` / `steps` / the CFG ceiling come from the request (each falling back to the reference
//! default); `req.fps` is the decoupled output/playback cadence (sc-3764).
//!
//! Preprocessing (sc-3412): the CLIP image is resized with the faithful diffusers
//! `_resize_with_antialiasing` (gaussian-blur + align-corners bicubic, in `[-1,1]` space — see
//! [`crate::preprocess`]); the VAE conditioning image is resized with PIL lanczos
//! (`mlx_gen::image::resize_lanczos_u8`), matching diffusers' `VideoProcessor.preprocess`
//! (`config.resample = "lanczos"`). Both are identity when the input already matches the target
//! size (the production case) and byte-faithful to the reference when they differ.

use mlx_rs::ops::{add, broadcast_to, divide, multiply, subtract};
use mlx_rs::{random, Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{
    default_seed, gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, Image, LoadSpec, Modality, ModelDescriptor, Precision, Progress,
    Result, WeightsSource,
};

use crate::config::{ImageEncoderConfig, SchedulerConfig, UnetConfig, VaeConfig};
use crate::image_encoder::SvdImageEncoder;
use crate::pipeline::{SvdParams, SvdPipeline};
use crate::scheduler::EdmSchedule;
use crate::unet::SvdUnet;
use crate::vae::SvdVae;

/// Public registry id: `mlx_gen::load("svd_xt", spec)`.
pub const MODEL_ID: &str = "svd_xt";

/// OpenCLIP ViT-H image-normalization mean/std (the SVD `feature_extractor`). The canonical CLIP
/// constants carry more digits than f32 resolves (the extra precision is harmless — they round to the
/// nearest f32 either way), so the excessive-precision lint is allowed to keep the recognizable values.
#[allow(clippy::excessive_precision)]
const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
#[allow(clippy::excessive_precision)]
const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_58, 0.275_777_11];
const CLIP_SIZE: usize = 224;
/// VAE spatial compression (8×).
const VAE_SCALE: u32 = 8;
/// Upper bound on a `Reference` image's dimensions. The reference is resized to the output size, but
/// the source buffer (and the resize's intermediate f32 work) scale with the *input* dims, so an
/// unbounded reference drives multi-GB host allocations (F-164). 8192 is far above any real photo
/// while capping the source at a few hundred MB.
const MAX_REFERENCE_DIM: u32 = 8192;
/// Output `width`/`height` must be divisible by this: VAE 8× spatial compression then the UNet's 3
/// stride-2 downsamples / nearest-2× upsamples (another 8×). An unaligned size fails deep in
/// `UpBlock::forward` on a skip-concat shape mismatch instead of at validation (F-165).
const SIZE_ALIGN: u32 = VAE_SCALE * 8;
/// Upper bound on requested output `frames`. SVD-XT is the 25-frame variant; the per-frame latents +
/// `added_time_ids` scale linearly, so an unbounded count drives a giant allocation. 64 leaves
/// generous headroom over the default without letting the request balloon memory.
const MAX_FRAMES: u32 = 64;
/// Upper bound on requested denoise `steps` — generous over the default 25; guards a pathological
/// value from pinning the GPU for an unbounded time.
const MAX_STEPS: u32 = 200;

/// Stable identity + advertised capabilities for SVD-XT (image→video, no audio).
pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: MODEL_ID,
        family: "svd",
        backend: "mlx",
        modality: Modality::Video,
        capabilities: Capabilities {
            supports_negative_prompt: false,
            // SVD uses a frame-wise guidance ramp (min→max); `req.guidance` overrides the ceiling.
            supports_guidance: true,
            supports_true_cfg: false,
            // image→video is a single `Reference` image.
            conditioning: vec![ConditioningKind::Reference],
            supports_lora: false,
            supports_lokr: false,
            samplers: Vec::new(),
            schedulers: Vec::new(),
            min_size: 256,
            max_size: 1024,
            max_count: 1,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: false,
            supported_quants: &[],
        },
    }
}

/// The loaded SVD model: the assembled pipeline + the cached descriptor.
pub struct Svd {
    pipeline: SvdPipeline,
    descriptor: ModelDescriptor,
}

/// Load every component from a checkpoint snapshot dir (`vae/` + `unet/` + `image_encoder/`).
///
/// **Precision.** The dense path (`Precision::Bf16`, the registry's dense sentinel) runs the UNet +
/// image encoder in **fp16** — the production dtype (SVD ships an fp16 variant and runs
/// `torch_dtype=float16`); this is the fast path (≈½ the resident weights + half-precision matmuls,
/// with the fp16-sensitive reductions — GroupNorm via fused `layer_norm`, attention via fused SDPA —
/// upcast internally by MLX). `Precision::Fp32` selects the full-precision quality path (the
/// S1/S3/S4 parity-validated precision). The **VAE always stays f32** (`force_upcast=True`).
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root = match &spec.weights {
        WeightsSource::Dir(p) => p,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "svd_xt: expected a checkpoint directory (vae/ + unet/ + image_encoder/)".into(),
            ))
        }
    };

    let dense = match spec.precision {
        Precision::Bf16 => Dtype::Float16,
        Precision::Fp32 => Dtype::Float32,
    };

    // Prefer the on-disk fp16 variant when running fp16 (half the load IO, already half precision);
    // otherwise the full-precision file, cast to the target dtype.
    let load = |sub: &str, stem: &str, dtype: Dtype| -> Result<Weights> {
        let dir = root.join(sub);
        let path = if dtype == Dtype::Float16 {
            let fp16 = dir.join(format!("{stem}.fp16.safetensors"));
            if fp16.exists() {
                fp16
            } else {
                dir.join(format!("{stem}.safetensors"))
            }
        } else {
            dir.join(format!("{stem}.safetensors"))
        };
        let mut w = Weights::from_file(path)?;
        w.cast_all(dtype)?;
        Ok(w)
    };

    let vae = SvdVae::from_weights(
        &load("vae", "diffusion_pytorch_model", Dtype::Float32)?,
        &VaeConfig::default(),
    )?;
    let unet = SvdUnet::from_weights(
        &load("unet", "diffusion_pytorch_model", dense)?,
        &UnetConfig::default(),
    )?;
    let image_encoder = SvdImageEncoder::from_weights(
        &load("image_encoder", "model", dense)?,
        &ImageEncoderConfig::default(),
    )?;

    Ok(Box::new(Svd {
        pipeline: SvdPipeline::new(image_encoder, vae, unet, SchedulerConfig::default()),
        descriptor: descriptor(),
    }))
}

/// The SVD-specific request validation the core `Capabilities::validate_request` leaves to each model
/// (size alignment + the allocation/compute knob bounds) — F-165. Pulled out so it is unit-testable
/// without a loaded model.
fn validate_output_params(req: &GenerationRequest) -> Result<()> {
    // The SVD UNet needs `width`/`height` divisible by 64; an unaligned size fails deep in
    // `UpBlock::forward` on a skip-concat shape mismatch mid-denoise instead of at validation.
    if !req.width.is_multiple_of(SIZE_ALIGN) || !req.height.is_multiple_of(SIZE_ALIGN) {
        return Err(Error::Msg(format!(
            "svd_xt: {}x{} must be a multiple of {SIZE_ALIGN} (VAE 8× × UNet 8×)",
            req.width, req.height
        )));
    }
    if let Some(frames) = req.frames {
        if frames == 0 || frames > MAX_FRAMES {
            return Err(Error::Msg(format!(
                "svd_xt: frames {frames} out of range 1..={MAX_FRAMES}"
            )));
        }
    }
    if let Some(steps) = req.steps {
        if steps == 0 || steps > MAX_STEPS {
            return Err(Error::Msg(format!(
                "svd_xt: steps {steps} out of range 1..={MAX_STEPS}"
            )));
        }
    }
    Ok(())
}

/// Reject a `Reference` image with zero/oversized dims or a pixel buffer that isn't `w*h*3` RGB8,
/// using usize math so the length never wraps (F-164). Run once at the request boundary in
/// [`Svd::validate`] so the downstream resizes can't panic OOB or balloon host memory.
fn validate_reference_image(img: &Image) -> Result<()> {
    if img.width == 0 || img.height == 0 {
        return Err(Error::Msg(format!(
            "svd_xt: reference image has a zero dimension ({}x{})",
            img.width, img.height
        )));
    }
    if img.width > MAX_REFERENCE_DIM || img.height > MAX_REFERENCE_DIM {
        return Err(Error::Msg(format!(
            "svd_xt: reference image {}x{} exceeds the {MAX_REFERENCE_DIM}px dimension cap",
            img.width, img.height
        )));
    }
    if img.pixels.len() != img.width as usize * img.height as usize * 3 {
        return Err(Error::Msg(format!(
            "svd_xt: reference image pixel buffer {} != {}x{}x3 (RGB8)",
            img.pixels.len(),
            img.width,
            img.height
        )));
    }
    Ok(())
}

/// An RGB8 [`Image`] → NHWC f32 `[1, out_h, out_w, 3]` in `[0, 1]` for the VAE conditioning image:
/// PIL **lanczos** resize (matching diffusers' `VideoProcessor.preprocess`, `resample = "lanczos"`),
/// then rescale to `[0, 1]`. (Identity when the input already matches the target size.)
fn image_to_unit_nhwc(img: &Image, out_h: usize, out_w: usize) -> Result<Array> {
    // usize math: `width * height * 3` in u32 wraps for large dims (e.g. 65536² → 0), which would let
    // an empty/short buffer pass and then index OOB in the resize (F-164).
    if img.pixels.len() != img.width as usize * img.height as usize * 3 {
        return Err(Error::Msg("svd_xt: reference image must be RGB8".into()));
    }
    let resized = mlx_gen::image::resize_lanczos_u8(
        &img.pixels,
        img.height as usize,
        img.width as usize,
        out_h,
        out_w,
    ); // f32 HWC in [0,255]
    let arr = Array::from_slice(&resized, &[1, out_h as i32, out_w as i32, 3]);
    Ok(divide(&arr, mlx_gen::array::scalar(255.0))?)
}

impl Svd {
    /// Resolve the single conditioning reference image (image→video input).
    fn reference<'a>(&self, req: &'a GenerationRequest) -> Result<&'a Image> {
        req.conditioning
            .iter()
            .find_map(|c| match c {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            })
            .ok_or_else(|| Error::Msg("svd_xt: image→video requires a Reference image".into()))
    }

    /// CLIP `image_embeds` `[1, 1, 1024]` from the reference: diffusers `_resize_with_antialiasing`
    /// to 224 (gaussian-blur + align-corners bicubic, in `[-1,1]`) → CLIP mean/std normalize.
    fn clip_embeds(&self, img: &Image) -> Result<Array> {
        if img.pixels.len() != img.width as usize * img.height as usize * 3 {
            return Err(Error::Msg("svd_xt: reference image must be RGB8".into()));
        }
        let unit = crate::preprocess::resize_with_antialiasing_unit(
            &img.pixels,
            img.height as usize,
            img.width as usize,
            CLIP_SIZE,
            CLIP_SIZE,
        ); // HWC [224,224,3] in [0,1]
        let unit = Array::from_slice(&unit, &[1, CLIP_SIZE as i32, CLIP_SIZE as i32, 3]);
        let mean = Array::from_slice(&CLIP_MEAN, &[1, 1, 1, 3]);
        let std = Array::from_slice(&CLIP_STD, &[1, 1, 1, 3]);
        let normed = divide(&subtract(&unit, &mean)?, &std)?;
        let embeds = self.pipeline.image_encoder.image_embeds(&normed)?; // [1, 1024]
        let s = embeds.shape();
        Ok(embeds.reshape(&[s[0], 1, s[1]])?) // [1, 1, 1024]
    }

    /// Per-frame VAE image latent `[1, F, h, w, 4]`: preprocess to `[-1,1]`, add `noise_aug·N(0,1)`,
    /// VAE-encode (`mode()`), repeat over frames.
    fn image_latents(
        &self,
        img: &Image,
        height: u32,
        width: u32,
        num_frames: i32,
        noise_aug: f32,
        seed: u64,
    ) -> Result<Array> {
        let unit = image_to_unit_nhwc(img, height as usize, width as usize)?; // [1,H,W,3] in [0,1]
        let centered = subtract(
            &multiply(&unit, mlx_gen::array::scalar(2.0))?,
            mlx_gen::array::scalar(1.0),
        )?; // [-1,1]
        let key = random::key(seed.wrapping_add(7))?;
        let noise = random::normal::<f32>(centered.shape(), None, None, Some(&key))?;
        let augmented = add(
            &centered,
            &multiply(&noise, mlx_gen::array::scalar(noise_aug))?,
        )?;
        let latent = self.pipeline.vae.encode_mode(&augmented)?; // [1,h,w,4]
        let s = latent.shape();
        let l5 = latent.reshape(&[s[0], 1, s[1], s[2], s[3]])?;
        Ok(broadcast_to(&l5, &[s[0], num_frames, s[1], s[2], s[3]])?)
    }
}

impl Generator for Svd {
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

impl Svd {
    fn validate_impl(&self, req: &GenerationRequest) -> Result<()> {
        // Shared capability floor: size range (256..=1024), count, unsupported negative-prompt /
        // true_cfg / sampler / scheduler, and conditioning (`Reference` only). `guidance` IS supported
        // — it overrides the frame-wise CFG ceiling.
        self.descriptor
            .capabilities
            .validate_request(MODEL_ID, req)?;
        // Model-specific alignment + knob bounds the core contract leaves to each model (F-165).
        validate_output_params(req)?;
        // image→video needs the single Reference input — the one input not covered by the shared
        // size-range validation, so bound it here (F-164).
        let img = self.reference(req)?;
        validate_reference_image(img)?;
        Ok(())
    }

    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        self.validate(req)?;
        let img = self.reference(req)?;

        let mut params = SvdParams::default();
        if let Some(f) = req.frames {
            params.num_frames = f as i32;
            // Default the decode chunk to the full clip (diffusers' `decode_chunk_size = num_frames`)
            // unless the request overrides it below.
            params.decode_chunk_size = f as i32;
        }
        if let Some(s) = req.steps {
            params.num_inference_steps = s as usize;
        }
        // `params.fps` is the MOTION-conditioning cadence baked into `added_time_ids` (the value the
        // model was trained on, ~7) — it comes from `conditioning_fps`, NOT `req.fps`. `req.fps` is the
        // OUTPUT/playback cadence and is applied below at return time, so the two are decoupled
        // (diffusers `StableVideoDiffusionPipeline(fps=…)` vs `export_to_video(fps=…)`; sc-3764).
        if let Some(cfps) = req.conditioning_fps {
            params.fps = cfps as i32;
        }
        if let Some(g) = req.guidance {
            params.max_guidance_scale = g;
        }
        // SVD micro-conditioning knobs (sc-3523): `motion_bucket_id` / `noise_aug_strength` drive the
        // `added_time_ids` motion conditioning; `decode_chunk_size` is the chunked-VAE memory knob.
        // Each falls back to the reference default when the request leaves it `None`.
        if let Some(m) = req.motion_bucket_id {
            params.motion_bucket_id = m;
        }
        if let Some(n) = req.noise_aug_strength {
            params.noise_aug_strength = n;
        }
        if let Some(c) = req.decode_chunk_size {
            params.decode_chunk_size = c as i32;
        }
        let seed = req.seed.unwrap_or_else(default_seed);

        // Conditioning.
        let image_embeds = self.clip_embeds(img)?;
        let image_latents = self.image_latents(
            img,
            req.height,
            req.width,
            params.num_frames,
            params.noise_aug_strength,
            seed,
        )?;
        let added_time_ids = SvdPipeline::added_time_ids(&params);

        // Seeded init noise scaled by `init_noise_sigma` (NHWC-with-frames).
        let sched = EdmSchedule::karras(params.num_inference_steps, &self.pipeline.scheduler);
        let (h, w) = (
            (req.height / VAE_SCALE) as i32,
            (req.width / VAE_SCALE) as i32,
        );
        let key = random::key(seed)?;
        let noise =
            random::normal::<f32>(&[1, params.num_frames, h, w, 4], None, None, Some(&key))?;
        let latents = multiply(&noise, mlx_gen::array::scalar(sched.init_noise_sigma()))?;

        on_progress(Progress::Step {
            current: 0,
            total: params.num_inference_steps as u32,
        });
        let final_latents = self.pipeline.denoise(
            &latents,
            &image_embeds,
            &image_latents,
            &added_time_ids,
            params.num_frames,
            params.num_inference_steps,
            params.min_guidance_scale,
            params.max_guidance_scale,
        )?;

        on_progress(Progress::Decoding);
        let decoded =
            self.pipeline
                .decode(&final_latents, params.num_frames, params.decode_chunk_size)?; // [1,F,H,W,3]
        let frames = frames_to_images(&decoded)?;

        Ok(GenerationOutput::Video {
            frames,
            // Output/playback cadence = `req.fps` (decoupled from the motion-conditioning fps in
            // `params.fps`); falls back to the conditioning fps when the caller left it unset.
            fps: req.fps.unwrap_or(params.fps as u32),
            audio: None,
        })
    }
}

/// Decoded frames NHWC `[1, F, H, W, 3]` (roughly `[-1,1]`) → `Vec<Image>` (`clip(x·0.5+0.5)·255`).
fn frames_to_images(decoded: &Array) -> Result<Vec<Image>> {
    use mlx_rs::ops::{maximum, minimum, round};
    let half = mlx_gen::array::scalar(0.5);
    let x = add(&multiply(decoded, &half)?, &half)?;
    let x = minimum(
        &maximum(&x, mlx_gen::array::scalar(0.0))?,
        mlx_gen::array::scalar(1.0),
    )?;
    let x = round(&multiply(&x, mlx_gen::array::scalar(255.0))?, 0)?;
    let sh = x.shape();
    let (f, h, w) = (sh[1], sh[2], sh[3]);
    let total: i32 = sh.iter().product();
    let flat = x.reshape(&[total])?;
    let data = flat.as_slice::<f32>();
    let per = (h * w * 3) as usize;
    let mut frames = Vec::with_capacity(f as usize);
    for fi in 0..f as usize {
        let start = fi * per;
        let pixels: Vec<u8> = data[start..start + per].iter().map(|&v| v as u8).collect();
        frames.push(Image {
            width: w as u32,
            height: h as u32,
            pixels,
        });
    }
    Ok(frames)
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load`] into it.
fn load_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load(spec).map_err(Into::into)
}

inventory::submit! {
    mlx_gen::ModelRegistration { descriptor, load: load_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn img(w: u32, h: u32, len: usize) -> Image {
        Image {
            width: w,
            height: h,
            pixels: vec![0u8; len],
        }
    }

    #[test]
    fn validate_output_params_enforces_alignment_and_bounds() {
        // F-165: 64-aligned sizes pass; an unaligned one is rejected at validation (not deep in
        // UpBlock), and frames/steps are bounded.
        let req = |w: u32, h: u32, frames: Option<u32>, steps: Option<u32>| GenerationRequest {
            width: w,
            height: h,
            frames,
            steps,
            ..Default::default()
        };
        assert!(validate_output_params(&req(1024, 576, None, None)).is_ok()); // 576 = 9×64
        let err = validate_output_params(&req(700, 704, None, None))
            .unwrap_err()
            .to_string();
        assert!(err.contains("multiple of 64"), "got: {err}");
        assert!(validate_output_params(&req(704, 700, None, None)).is_err()); // height unaligned

        // frames / steps bounds.
        assert!(validate_output_params(&req(512, 512, Some(MAX_FRAMES), None)).is_ok());
        assert!(validate_output_params(&req(512, 512, Some(MAX_FRAMES + 1), None)).is_err());
        assert!(validate_output_params(&req(512, 512, Some(0), None)).is_err());
        assert!(validate_output_params(&req(512, 512, None, Some(MAX_STEPS + 1))).is_err());
        assert!(validate_output_params(&req(512, 512, None, Some(0))).is_err());
    }

    #[test]
    fn validate_reference_image_guards_size_and_buffer() {
        // F-164: the old `(w*h*3) as u32` wraps to 0 at 65536², so an empty buffer would pass and the
        // resize would index OOB. usize math makes the length check honest and rejects it.
        let overflow = img(65536, 65536, 0);
        assert_eq!(
            65536u32.wrapping_mul(65536).wrapping_mul(3),
            0,
            "the u32 product really does wrap to 0"
        );
        assert!(validate_reference_image(&overflow).is_err());

        // Oversized but self-consistent dims are rejected by the cap (unbounded host allocation).
        let huge = MAX_REFERENCE_DIM + 1;
        assert!(validate_reference_image(&img(huge, 1, huge as usize * 3)).is_err());

        // Zero dimension and a short/long buffer are rejected.
        assert!(validate_reference_image(&img(0, 8, 0)).is_err());
        assert!(validate_reference_image(&img(8, 8, 8 * 8 * 3 - 1)).is_err());

        // A well-formed in-range RGB8 reference passes.
        assert!(validate_reference_image(&img(64, 48, 64 * 48 * 3)).is_ok());
        // Exactly at the cap is allowed.
        assert!(validate_reference_image(&img(
            MAX_REFERENCE_DIM,
            1,
            MAX_REFERENCE_DIM as usize * 3
        ))
        .is_ok());
    }
}
