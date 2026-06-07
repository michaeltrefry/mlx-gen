//! InstantID provider (epic 3109, sc-3113/3114/3115) — identity-preserving SDXL T2I.
//!
//! Composes the existing SDXL building blocks (`mlx-gen-sdxl`) and the native face stack
//! (`mlx-gen-face`) rather than re-implementing them. One denoise step applies **both** conditioning
//! paths driven by the reference face (the vendored `StableDiffusionXLInstantIDPipeline`):
//! - the **face IP tokens** (ArcFace 512 → 16×2048 via [`ResamplerConfig::instantid_face`]) injected
//!   into every UNet cross-attention at `ip_adapter_scale`;
//! - the **IdentityNet** (a stock SDXL ControlNet) on the 5-keypoint `draw_kps` control image, its
//!   cross-attention conditioned on the *same* face tokens, residuals added into the UNet.
//!
//! CFG is positive-first (`[cond, uncond]`, matching the SDXL crate): the uncond face tokens are
//! `Resampler(zeros)` — the zero embedding run through the Resampler, NOT literal zero tokens — exactly
//! as the reference `_encode_prompt_image_emb` does it.

use std::path::PathBuf;

use mlx_rs::ops::{concatenate_axis, zeros};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result, WeightsSource};

use mlx_gen_face::{Face, FaceAnalysis};
use mlx_gen_sdxl::config::DiffusionConfig;
use mlx_gen_sdxl::ip_adapter::{load_ip_kv_pairs, Resampler, ResamplerConfig};
use mlx_gen_sdxl::sampler::{AncestralEuler, EulerSampler};
use mlx_gen_sdxl::text_encoder::ClipTextEncoder;
use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_sdxl::unet::{ControlNet, UNet2DConditionModel};
use mlx_gen_sdxl::vae::Autoencoder;
use mlx_gen_sdxl::{
    decode_image, denoise_ip_control, encode_conditioning, load_controlnet,
    load_text_encoder_1_dtype, load_text_encoder_2_dtype, load_tokenizer, load_unet_dtype,
    load_vae, preprocess_control_image, seeded_prior, text_time_ids, ControlContext, Denoiser,
};

use crate::kps;

/// The InstantID compute dtype — fp16, matching the production SDXL path (the VAE stays f32 inside
/// the SDXL loader).
const DTYPE: Dtype = Dtype::Float16;

/// Default `ip_adapter_scale` (the vendored pipeline's `set_ip_adapter_scale(0.8)`).
pub const DEFAULT_IP_SCALE: f32 = 0.8;
/// Default IdentityNet `controlnet_conditioning_scale` (the vendored default 0.8).
pub const DEFAULT_CONTROLNET_SCALE: f32 = 0.8;

/// Paths to the InstantID checkpoints.
pub struct InstantIdPaths {
    /// SDXL base snapshot dir (`unet/`, `text_encoder{,_2}/`, `vae/`, `tokenizer/`).
    pub sdxl_base: PathBuf,
    /// IdentityNet `ControlNetModel` — a dir (with `diffusion_pytorch_model.safetensors`) or a file.
    pub identitynet: WeightsSource,
    /// Converted `ip-adapter.safetensors` (`image_proj.*` + `ip_adapter.*`; see
    /// `tools/convert_instantid.py`).
    pub ip_adapter: PathBuf,
}

/// One InstantID generation request.
pub struct InstantIdRequest {
    pub prompt: String,
    pub negative: String,
    pub width: u32,
    pub height: u32,
    pub steps: usize,
    /// Classifier-free guidance scale (the vendored default 5.0).
    pub guidance: f32,
    pub ip_adapter_scale: f32,
    pub controlnet_scale: f32,
    pub seed: u64,
}

impl Default for InstantIdRequest {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            negative: String::new(),
            width: 1024,
            height: 1024,
            steps: 30,
            guidance: 5.0,
            ip_adapter_scale: DEFAULT_IP_SCALE,
            controlnet_scale: DEFAULT_CONTROLNET_SCALE,
            seed: 0,
        }
    }
}

/// Loaded InstantID model: the SDXL backbone (UNet with the face IP K/V pairs installed) + the
/// IdentityNet + the face Resampler, plus an optional native face-analysis stack.
pub struct InstantId {
    tokenizer: ClipBpeTokenizer,
    te1: ClipTextEncoder,
    te2: ClipTextEncoder,
    unet: UNet2DConditionModel,
    identitynet: ControlNet,
    resampler: Resampler,
    vae: Autoencoder,
    sampler: EulerSampler,
    face: Option<FaceAnalysis>,
}

impl InstantId {
    /// Load the SDXL backbone + IdentityNet + face Resampler, installing the decoupled-cross-attn
    /// K/V pairs into the UNet. The face-analysis stack is attached separately via
    /// [`with_face`](Self::with_face) (it needs the converted SCRFD + ArcFace weights).
    pub fn load(paths: &InstantIdPaths) -> Result<Self> {
        let root = paths.sdxl_base.as_path();
        let tokenizer = load_tokenizer(root)?;
        let te1 = load_text_encoder_1_dtype(root, DTYPE)?;
        let te2 = load_text_encoder_2_dtype(root, DTYPE)?;
        let vae = load_vae(root)?; // f32
        let mut unet = load_unet_dtype(root, DTYPE)?;

        // IdentityNet — a stock diffusers SDXL ControlNet (sc-3112), no conversion.
        let identitynet = load_controlnet(&paths.identitynet, DTYPE)?;

        // Face IP-Adapter: the Resampler (`image_proj.*`) + the 70 decoupled K/V pairs (`ip_adapter.*`),
        // both from the converted bundle. Cast to the UNet dtype so the pairs quantize/run with it.
        let mut ipa = Weights::from_file(&paths.ip_adapter).map_err(|e| {
            Error::Msg(format!(
                "instantid: load ip-adapter {:?} (run tools/convert_instantid.py): {e}",
                paths.ip_adapter
            ))
        })?;
        ipa.cast_all(DTYPE)?;
        let resampler =
            Resampler::from_weights(&ipa, "image_proj", &ResamplerConfig::instantid_face())?;
        let pairs = load_ip_kv_pairs(&ipa)?;
        unet.install_ip_adapter(pairs)?;

        let cfg = DiffusionConfig::sdxl_base();
        Ok(Self {
            tokenizer,
            te1,
            te2,
            unet,
            identitynet,
            resampler,
            vae,
            sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE),
            face: None,
        })
    }

    /// Attach the native face-analysis stack (SCRFD detector + ArcFace embedder) so [`generate`] can
    /// take a raw reference image. Weights come from `tools/convert_scrfd.py` / `convert_glintr100.py`.
    pub fn with_face(mut self, scrfd: &Weights, arcface: &Weights) -> Result<Self> {
        self.face = Some(FaceAnalysis::load(scrfd, arcface)?);
        Ok(self)
    }

    /// Quantize the stack to `bits` (8 or 4) — Q8/Q4 (sc-3116), the same scope as the SDXL provider
    /// (sc-2641): the UNet (with the now-installed face IP K/V pairs), both CLIP text encoders, and the
    /// IdentityNet ControlNet. The face **Resampler** stays fp16 (tiny, runs once per generation) and
    /// the **VAE** stays f32. Call after [`load`](Self::load) (so the IP pairs quantize with the UNet),
    /// before [`with_face`](Self::with_face).
    pub fn quantize(mut self, bits: i32) -> Result<Self> {
        self.unet.quantize(bits)?;
        self.te1.quantize(bits)?;
        self.te2.quantize(bits)?;
        self.identitynet.quantize(bits)?;
        Ok(self)
    }

    /// Detect the largest face in `img` (RGB `u8` HWC, `h×w`) and return it (bbox + 5 kps + 512-d
    /// embedding). Requires [`with_face`](Self::with_face).
    pub fn largest_face(&self, img: &[u8], h: usize, w: usize) -> Result<Face> {
        let face = self
            .face
            .as_ref()
            .ok_or_else(|| Error::Msg("instantid: face stack not attached (with_face)".into()))?;
        let mut faces = face.analyze(img, h, w)?;
        if faces.is_empty() {
            return Err(Error::Msg(
                "instantid: no face detected in the reference".into(),
            ));
        }
        Ok(faces.remove(0)) // analyze() sorts largest-first
    }

    /// Full T2I: letterbox the reference to the output size (the sc-2009 kps-distortion rule), detect
    /// the largest face, then generate. Requires [`with_face`](Self::with_face).
    pub fn generate(&self, req: &InstantIdRequest, reference: &Image) -> Result<Image> {
        let canvas = kps::letterbox(reference, req.width, req.height);
        let face = self.largest_face(&canvas.pixels, req.height as usize, req.width as usize)?;
        let kps: Vec<(f32, f32)> = face.kps.iter().map(|p| (p[0], p[1])).collect();
        self.generate_with(req, &face.embedding, &kps)
    }

    /// Core generate from a precomputed ArcFace `embedding` (512-d) and 5 `kps` (output-canvas pixel
    /// coords) — the face-stack-independent path (also the engine seam: `ip_adapter_scale = 0` +
    /// `controlnet_scale = 0` reduces to plain SDXL txt2img).
    pub fn generate_with(
        &self,
        req: &InstantIdRequest,
        embedding: &[f32],
        kps: &[(f32, f32)],
    ) -> Result<Image> {
        if embedding.len() != 512 {
            return Err(Error::Msg(format!(
                "instantid: ArcFace embedding must be 512-d, got {}",
                embedding.len()
            )));
        }
        let cfg_on = req.guidance > 1.0;

        // Seed up front so the first RNG draw is the prior (conditioning draws no RNG).
        mlx_rs::random::seed(req.seed)?;
        let tokens = self
            .tokenizer
            .tokenize_batch(&req.prompt, if cfg_on { Some(&req.negative) } else { None })?;
        let (conditioning, pooled) = encode_conditioning(&self.te1, &self.te2, &tokens)?;
        let time_ids = text_time_ids(pooled.shape()[0]);

        // Face tokens via the Resampler. CFG positive-first: row 0 = Resampler(embed),
        // row 1 = Resampler(zeros) (the reference's zero-embed-through-Resampler uncond).
        let embed = Array::from_slice(embedding, &[1, 1, 512]).as_dtype(DTYPE)?;
        let proj_in = if cfg_on {
            let z = zeros::<f32>(&[1, 1, 512])?.as_dtype(DTYPE)?;
            concatenate_axis(&[&embed, &z], 0)?
        } else {
            embed
        };
        let face_tokens = self.resampler.forward(&proj_in)?; // [B, 16, 2048]

        // kps control image (sc-3111) → [0,1] NHWC, cast to the compute dtype, CFG-batched.
        let kps_image = kps::draw_kps(req.width, req.height, kps);
        let control =
            preprocess_control_image(&kps_image, req.width, req.height)?.as_dtype(DTYPE)?;
        let control = if cfg_on {
            concatenate_axis(&[&control, &control], 0)?
        } else {
            control
        };
        let control_ctx = ControlContext {
            controlnet: &self.identitynet,
            control_image: control,
            scale: req.controlnet_scale,
        };

        // Ancestral Euler over the seeded prior (fp16).
        let prior = seeded_prior(&self.sampler, req.seed, req.width, req.height)?;
        let ancestral = AncestralEuler::new(&self.sampler, req.steps, self.sampler.max_time());
        let d = Denoiser {
            unet: &self.unet,
            sampler: &ancestral,
        };
        let latents = denoise_ip_control(
            &d,
            prior,
            &conditioning,
            &pooled,
            &time_ids,
            req.guidance,
            &Default::default(),
            &mut |_| {},
            &control_ctx,
            &face_tokens, // IdentityNet cross-attn conditioning = face tokens
            &face_tokens, // UNet IP tokens = face tokens
            req.ip_adapter_scale,
        )?;
        decode_image(&self.vae, &latents)
    }
}
