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
use mlx_gen::{CancelFlag, Error, Progress, Result, WeightsSource};

use mlx_gen_face::{Face, FaceAnalysis};
use mlx_gen_sdxl::config::DiffusionConfig;
use mlx_gen_sdxl::ip_adapter::{load_ip_kv_pairs, Resampler, ResamplerConfig};
use mlx_gen_sdxl::sampler::{AncestralEuler, EulerSampler};
use mlx_gen_sdxl::text_encoder::ClipTextEncoder;
use mlx_gen_sdxl::tokenizer::ClipBpeTokenizer;
use mlx_gen_sdxl::unet::{ControlNet, UNet2DConditionModel};
use mlx_gen_sdxl::vae::Autoencoder;
use mlx_gen_sdxl::{
    decode_image, denoise_ip_control, denoise_ip_multi_control, encode_conditioning,
    load_controlnet, load_text_encoder_1_dtype, load_text_encoder_2_dtype, load_tokenizer,
    load_unet_dtype, load_vae, preprocess_control_image, seeded_prior, text_time_ids,
    ControlContext, Denoiser,
};

use mlx_gen::image::resize_lanczos_u8;

use crate::kps;
use crate::openpose::{self, BodyPoint, STICKWIDTH};
use crate::restore;

/// The InstantID compute dtype — fp16, matching the production SDXL path (the VAE stays f32 inside
/// the SDXL loader).
const DTYPE: Dtype = Dtype::Float16;

/// Number of face landmarks `draw_kps` indexes (`[left_eye, right_eye, nose, mouth_left,
/// mouth_right]`).
const FACE_KP_COUNT: usize = 5;

/// Reject a caller-supplied `kps` slice shorter than [`FACE_KP_COUNT`] with a typed error, mirroring
/// the 512-d embedding check in the public seams. Without this, `kps::draw_kps` asserts and panics the
/// process on a truncated landmark list (F-079).
fn validate_kps(kps: &[(f32, f32)]) -> Result<()> {
    if kps.len() < FACE_KP_COUNT {
        return Err(Error::Msg(format!(
            "instantid: need {FACE_KP_COUNT} face keypoints, got {}",
            kps.len()
        )));
    }
    Ok(())
}

/// Default `ip_adapter_scale` (the vendored pipeline's `set_ip_adapter_scale(0.8)`).
pub const DEFAULT_IP_SCALE: f32 = 0.8;
/// Default IdentityNet `controlnet_conditioning_scale` (the vendored default 0.8).
pub const DEFAULT_CONTROLNET_SCALE: f32 = 0.8;
/// Default OpenPose `controlnet_conditioning_scale` in pose mode (the worker's `openPoseScale` 0.7).
pub const DEFAULT_OPENPOSE_SCALE: f32 = 0.7;
/// The no-face-visible OpenPose scale floor (`instantid_adapter.py:425` `max(openPoseScale, 0.85)`).
const NO_FACE_OPENPOSE_FLOOR: f32 = 0.85;
/// Default face-restoration prompt (sc-3380). **Gender-neutral by design** — the worker's
/// `_FACE_RESTORE_PROMPT` is hardcoded "the woman's face" and applied to every character (a latent
/// bug); the native port neutralizes it ("the face"). Callers may pass a character-specific prompt.
pub const FACE_RESTORE_PROMPT: &str =
    "close-up portrait of the face, soft natural light, photorealistic, sharp focus";
/// The face-restore crop padding factor (`instantid_adapter.py:483` `* 1.9`).
const FACE_RESTORE_CROP_PAD: f32 = 1.9;

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
#[derive(Clone)]
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
    /// OpenPose `controlnet_conditioning_scale` — used only by [`InstantId::generate_pose`].
    pub openpose_scale: f32,
    pub seed: u64,
    /// Cooperative cancellation, checked before each denoise step and between phases (sc-4380;
    /// the engine contract every registry provider honors — see F-096). `Clone` shares the flag,
    /// so the caller keeps a handle to cancel an in-flight generation.
    pub cancel: CancelFlag,
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
            openpose_scale: DEFAULT_OPENPOSE_SCALE,
            seed: 0,
            cancel: CancelFlag::default(),
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
    /// The OpenPose ControlNet for pose mode (sc-3117), attached via [`with_openpose`](Self::with_openpose).
    openpose: Option<ControlNet>,
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
            openpose: None,
            resampler,
            vae,
            sampler: EulerSampler::new_with_dtype(&cfg, true, DTYPE)?,
            face: None,
        })
    }

    /// Attach the OpenPose ControlNet for pose mode (sc-3117) — a stock diffusers SDXL ControlNet
    /// (`xinsir/controlnet-openpose-sdxl-1.0`), loaded via the same generic [`load_controlnet`] as
    /// IdentityNet (no conversion). Required by [`generate_pose`](Self::generate_pose). Call after
    /// [`load`](Self::load); call before [`quantize`](Self::quantize) so it quantizes with the stack.
    pub fn with_openpose(mut self, openpose: &WeightsSource) -> Result<Self> {
        self.openpose = Some(load_controlnet(openpose, DTYPE)?);
        Ok(self)
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
        if let Some(op) = &mut self.openpose {
            op.quantize(bits)?;
        }
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
    pub fn generate(
        &self,
        req: &InstantIdRequest,
        reference: &Image,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let canvas = kps::letterbox(reference, req.width, req.height);
        let face = self.largest_face(&canvas.pixels, req.height as usize, req.width as usize)?;
        let kps: Vec<(f32, f32)> = face.kps.iter().map(|p| (p[0], p[1])).collect();
        self.generate_with(req, &face.embedding, &kps, on_progress)
    }

    /// **Multi-view angle generation** (sc-3117): rotate the reference identity to a named view from
    /// the canonical [`kps::VIEW_ANGLE_KPS`] pack. The reference supplies *identity* (its ArcFace
    /// embedding); the pack supplies the IdentityNet *pose* (the view-angle landmarks). The canvas is
    /// **square** (`req.width` is the side; the pack kps are normalized to a square — the sc-2009
    /// kps-distortion rule). `req.height` is ignored (forced to the side). Requires
    /// [`with_face`](Self::with_face); errors on an unknown `view_angle`.
    pub fn generate_angle(
        &self,
        req: &InstantIdRequest,
        reference: &Image,
        view_angle: &str,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let side = req.width;
        let view = kps::view_angle_kps(view_angle, side).ok_or_else(|| {
            Error::Msg(format!(
                "instantid: unknown view angle {view_angle:?} (see VIEW_ANGLE_KPS)"
            ))
        })?;
        // Identity from the reference (letterboxed to the square canvas).
        let canvas = kps::letterbox(reference, side, side);
        let face = self.largest_face(&canvas.pixels, side as usize, side as usize)?;
        let kps: Vec<(f32, f32)> = view.to_vec();
        let sq = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        self.generate_with(&sq, &face.embedding, &kps, on_progress)
    }

    /// **Multi-view angle generation from caller-supplied landmarks** (sc-4425): the data-driven
    /// sibling of [`generate_angle`]. Identical pipeline and square-canvas contract, but the 5-point
    /// kps come from the caller (`kps_norm`, normalized to a square `0.0..=1.0`) instead of the
    /// canonical [`kps::VIEW_ANGLE_KPS`] table. This lets SceneWorks own the angle/framing presets
    /// (built-in plus user-defined) so the engine no longer needs a hardcoded angle table. The
    /// reference supplies *identity* (its ArcFace embedding) while `kps_norm` supplies the IdentityNet
    /// pose/framing. The canvas is **square** (`req.width` is the side; `req.height` is forced to the
    /// side, per the sc-2009 kps-distortion rule). Requires [`with_face`](Self::with_face).
    pub fn generate_with_kps(
        &self,
        req: &InstantIdRequest,
        reference: &Image,
        kps_norm: &[(f32, f32)],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        validate_kps(kps_norm)?;
        let side = req.width;
        // Scale the normalized landmarks to square-canvas pixels (mirrors `kps::view_angle_kps`).
        let kps: Vec<(f32, f32)> = kps_norm
            .iter()
            .map(|(x, y)| (x * side as f32, y * side as f32))
            .collect();
        // Identity from the reference (letterboxed to the square canvas).
        let canvas = kps::letterbox(reference, side, side);
        let face = self.largest_face(&canvas.pixels, side as usize, side as usize)?;
        let sq = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        self.generate_with(&sq, &face.embedding, &kps, on_progress)
    }

    /// Core generate from a precomputed ArcFace `embedding` (512-d) and 5 `kps` (output-canvas pixel
    /// coords) — the face-stack-independent path (also the engine seam: `ip_adapter_scale = 0` +
    /// `controlnet_scale = 0` reduces to plain SDXL txt2img).
    pub fn generate_with(
        &self,
        req: &InstantIdRequest,
        embedding: &[f32],
        kps: &[(f32, f32)],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        // Honor the engine cancellation contract (sc-4380 / F-096): bail before any tensor work.
        if req.cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        if embedding.len() != 512 {
            return Err(Error::Msg(format!(
                "instantid: ArcFace embedding must be 512-d, got {}",
                embedding.len()
            )));
        }
        validate_kps(kps)?;
        let cfg_on = req.guidance > 1.0;

        // Seed up front so the first RNG draw is the prior (conditioning draws no RNG).
        mlx_rs::random::seed(req.seed)?;
        let tokens = self
            .tokenizer
            .tokenize_batch(&req.prompt, if cfg_on { Some(&req.negative) } else { None })?;
        let (conditioning, pooled) = encode_conditioning(&self.te1, &self.te2, &tokens)?;
        let time_ids = text_time_ids(pooled.shape()[0]);

        // Face tokens via the Resampler (CFG positive-first; uncond = Resampler(zeros)).
        let face_tokens = self.face_tokens(embedding, cfg_on)?; // [B, 16, 2048]

        // kps control image (sc-3111) → IdentityNet ControlContext.
        let kps_image = kps::draw_kps(req.width, req.height, kps);
        let control_ctx = self.control_ctx(
            &self.identitynet,
            &kps_image,
            req.width,
            req.height,
            req.controlnet_scale,
            cfg_on,
        )?;

        // Ancestral Euler over the seeded prior (fp16).
        let prior = seeded_prior(&self.sampler, req.seed, req.width, req.height)?;
        let ancestral = AncestralEuler::new(&self.sampler, req.steps, self.sampler.max_time())?;
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
            &req.cancel,
            on_progress,
            &control_ctx,
            &face_tokens, // IdentityNet cross-attn conditioning = face tokens
            &face_tokens, // UNet IP tokens = face tokens
            req.ip_adapter_scale,
        )?;
        on_progress(Progress::Decoding);
        decode_image(&self.vae, &latents)
    }

    /// Build the CFG-batched face tokens from a 512-d ArcFace `embedding` (positive-first; the uncond
    /// row is `Resampler(zeros)` — the reference's zero-embed-through-Resampler, not literal zeros).
    fn face_tokens(&self, embedding: &[f32], cfg_on: bool) -> Result<Array> {
        let embed = Array::from_slice(embedding, &[1, 1, 512]).as_dtype(DTYPE)?;
        let proj_in = if cfg_on {
            let z = zeros::<f32>(&[1, 1, 512])?.as_dtype(DTYPE)?;
            concatenate_axis(&[&embed, &z], 0)?
        } else {
            embed
        };
        self.resampler.forward(&proj_in) // [B, 16, 2048]
    }

    /// Build a [`ControlContext`] for a control image: rasterized image → `[0,1]` NHWC, cast to the
    /// compute dtype, CFG-batched (the negative pass sees the same control image, matching the
    /// reference's duplicated control batch).
    fn control_ctx<'a>(
        &self,
        controlnet: &'a ControlNet,
        image: &Image,
        width: u32,
        height: u32,
        scale: f32,
        cfg_on: bool,
    ) -> Result<ControlContext<'a>> {
        let c = preprocess_control_image(image, width, height)?.as_dtype(DTYPE)?;
        let control_image = if cfg_on {
            concatenate_axis(&[&c, &c], 0)?
        } else {
            c
        };
        Ok(ControlContext {
            // Precompute the step-invariant conditioning embedding once per run (F-069).
            cond_embed: controlnet.embed_cond(&control_image)?,
            controlnet,
            scale,
        })
    }

    /// **Pose mode** (sc-3117): generate the character in one library pose on a square canvas. The
    /// OpenPose skeleton (rendered from the pre-supplied `keypoints`) drives the body pose, while
    /// IdentityNet and the face IP tokens anchor the face when the head is visible. The reference
    /// supplies *identity* (its ArcFace embedding); the face landmarks are re-placed at the pose head.
    ///
    /// `req.width` is the square side (`req.height` ignored — the OpenPose/kps control images are
    /// square-canonical, the kps-distortion rule). Requires [`with_face`](Self::with_face) +
    /// [`with_openpose`](Self::with_openpose). The face-restoration pass (sc-3380) is a separate step.
    pub fn generate_pose(
        &self,
        req: &InstantIdRequest,
        reference: &Image,
        keypoints: &[BodyPoint],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        let side = req.width;
        // Identity from the reference (letterboxed to the square canvas).
        let canvas = kps::letterbox(reference, side, side);
        let face = self.largest_face(&canvas.pixels, side as usize, side as usize)?;
        // Place the reference's 5 face landmarks at the pose's head position (when the head is
        // visible). `None` ⇒ a back/occluded view: the no-face branch zeroes IdentityNet + IP.
        let face_kps = openpose::face_box_from_keypoints(keypoints)
            .map(|(cx, cy, face_h_frac)| self.place_face_kps(&face, cx, cy, face_h_frac, side));
        let sq = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        self.generate_pose_with(
            &sq,
            &face.embedding,
            face_kps.as_deref(),
            keypoints,
            on_progress,
        )
    }

    /// Re-place the reference's 5-point face landmarks at a pose's head box. Mirrors
    /// `instantid_adapter.py::_run_pose` + `_normalized_kps`: normalize the reference kps to its
    /// detected bbox, then scale/translate to the head box `(cx, cy)` (normalized canvas coords) at
    /// height `face_h_frac` of the canvas, preserving the face aspect. Returns canvas-pixel coords.
    fn place_face_kps(
        &self,
        face: &Face,
        cx: f64,
        cy: f64,
        face_h_frac: f64,
        side: u32,
    ) -> Vec<(f32, f32)> {
        let [x1, y1, x2, y2] = face.bbox;
        let (ox, oy) = (x1 as f64, y1 as f64);
        let sw = (x2 - x1).max(1.0) as f64;
        let sh = (y2 - y1).max(1.0) as f64;
        let aspect = sw / sh;
        let canvas = side as f64;
        let face_h = canvas * face_h_frac;
        let face_w = face_h * aspect;
        face.kps
            .iter()
            .map(|&[kx, ky]| {
                let nx = (kx as f64 - ox) / sw;
                let ny = (ky as f64 - oy) / sh;
                let px = cx * canvas + (nx - 0.5) * face_w;
                let py = cy * canvas + (ny - 0.5) * face_h;
                (px as f32, py as f32)
            })
            .collect()
    }

    /// Core pose-mode generate (face-stack-independent): MultiControlNet over `[IdentityNet(face_kps),
    /// OpenPose(skeleton)]` with the face IP tokens. A `Some` `face_kps` (head visible) drives
    /// IdentityNet and the IP tokens at the request scales; `None` (back/occluded) zeroes IdentityNet
    /// and the IP tokens and boosts OpenPose to `max(openpose_scale, 0.85)`, matching
    /// `instantid_adapter.py:424-426`. `req.width` is the square side; requires
    /// [`with_openpose`](Self::with_openpose).
    pub fn generate_pose_with(
        &self,
        req: &InstantIdRequest,
        embedding: &[f32],
        face_kps: Option<&[(f32, f32)]>,
        keypoints: &[BodyPoint],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        // Honor the engine cancellation contract (sc-4380 / F-096): bail before any tensor work.
        if req.cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        if embedding.len() != 512 {
            return Err(Error::Msg(format!(
                "instantid: ArcFace embedding must be 512-d, got {}",
                embedding.len()
            )));
        }
        if let Some(kps) = face_kps {
            validate_kps(kps)?;
        }
        let openpose = self.openpose.as_ref().ok_or_else(|| {
            Error::Msg("instantid: pose mode needs the OpenPose ControlNet (with_openpose)".into())
        })?;
        let side = req.width;
        let cfg_on = req.guidance > 1.0;

        // Seed up front so the first RNG draw is the prior (conditioning draws no RNG).
        mlx_rs::random::seed(req.seed)?;
        let tokens = self
            .tokenizer
            .tokenize_batch(&req.prompt, if cfg_on { Some(&req.negative) } else { None })?;
        let (conditioning, pooled) = encode_conditioning(&self.te1, &self.te2, &tokens)?;
        let time_ids = text_time_ids(pooled.shape()[0]);
        let face_tokens = self.face_tokens(embedding, cfg_on)?;

        // OpenPose skeleton control image (sc-3379), always at the requested pose scale.
        let skeleton = openpose::draw_bodypose(side, side, keypoints, STICKWIDTH);

        // Face landmark control image + per-branch scales. No visible face ⇒ blank kps, IdentityNet +
        // IP zeroed, OpenPose boosted (the shared seed/prompt carry hair/wardrobe continuity).
        let (face_image, id_scale, op_scale, ip_scale) = match face_kps {
            Some(kps) => (
                kps::draw_kps(side, side, kps),
                req.controlnet_scale,
                req.openpose_scale,
                req.ip_adapter_scale,
            ),
            None => (
                Image {
                    width: side,
                    height: side,
                    pixels: vec![0u8; (side as usize) * (side as usize) * 3],
                },
                0.0,
                req.openpose_scale.max(NO_FACE_OPENPOSE_FLOOR),
                0.0,
            ),
        };
        // MultiControlNet branch order matches the reference: [IdentityNet(kps), OpenPose(skeleton)].
        let id_ctx =
            self.control_ctx(&self.identitynet, &face_image, side, side, id_scale, cfg_on)?;
        let op_ctx = self.control_ctx(openpose, &skeleton, side, side, op_scale, cfg_on)?;
        let controls = [id_ctx, op_ctx];

        // Ancestral Euler over the seeded prior (fp16).
        let prior = seeded_prior(&self.sampler, req.seed, side, side)?;
        let ancestral = AncestralEuler::new(&self.sampler, req.steps, self.sampler.max_time())?;
        let d = Denoiser {
            unet: &self.unet,
            sampler: &ancestral,
        };
        let latents = denoise_ip_multi_control(
            &d,
            prior,
            &conditioning,
            &pooled,
            &time_ids,
            req.guidance,
            &req.cancel,
            on_progress,
            &controls,
            &face_tokens, // ControlNet cross-attn conditioning = face tokens (both branches)
            &face_tokens, // UNet IP tokens = face tokens
            ip_scale,
        )?;
        on_progress(Progress::Decoding);
        decode_image(&self.vae, &latents)
    }

    /// **Face-restoration pass** (sc-3380): ADetailer-style identity recovery at full-body framing.
    /// Detect the largest face in `base`, crop it with `1.9×` padding, re-render that crop through the
    /// InstantID pipe (IdentityNet only — the OpenPose branch is a no-op at restore time, so this is
    /// the single-control [`generate_with`] path) with the reference `embedding`, then paste it back
    /// with a feathered elliptical mask. Recovers ArcFace identity from ~0.38 to ~0.88 at full-body
    /// framing. A no-op (returns `base` unchanged) when no face is found or the crop is degenerate.
    ///
    /// `req.prompt` is the restore prompt — pass [`FACE_RESTORE_PROMPT`] (gender-neutral) or a
    /// character-specific one; `req.width` is the (square) re-render side (1024 in production). The
    /// other request fields (negative, steps, guidance, scales, seed) drive the crop re-render.
    /// Requires [`with_face`](Self::with_face).
    pub fn restore_face(
        &self,
        req: &InstantIdRequest,
        base: &Image,
        embedding: &[f32],
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<Image> {
        // Cancel between phases (sc-4380): the restore re-render is a full second denoise pass.
        if req.cancel.is_cancelled() {
            return Err(Error::Msg("generation cancelled".into()));
        }
        let face = self
            .face
            .as_ref()
            .ok_or_else(|| Error::Msg("instantid: face stack not attached (with_face)".into()))?;
        let (bw, bh) = (base.width as usize, base.height as usize);
        let mut faces = face.analyze(&base.pixels, bh, bw)?;
        if faces.is_empty() {
            return Ok(base.clone()); // no face to restore — leave the base untouched
        }
        let f = faces.remove(0); // analyze() sorts largest-first

        // Crop box: a square-ish window around the face center, padded ×1.9, clamped to the image.
        let [x1, y1, x2, y2] = f.bbox;
        let (cx, cy) = ((x1 + x2) / 2.0, (y1 + y2) / 2.0);
        let half = (x2 - x1).max(y2 - y1) * FACE_RESTORE_CROP_PAD / 2.0;
        let a = (cx - half).max(0.0) as usize;
        let b = (cy - half).max(0.0) as usize;
        let c = (cx + half).min(base.width as f32) as usize;
        let d = (cy + half).min(base.height as f32) as usize;
        let (crop_w, crop_h) = (c.saturating_sub(a), d.saturating_sub(b));
        if crop_w < 16 || crop_h < 16 {
            return Ok(base.clone()); // degenerate crop — skip
        }

        // Re-place the detected face's 5 kps into the crop, scaled to the square re-render side.
        let side = req.width;
        let (sx, sy) = (side as f32 / crop_w as f32, side as f32 / crop_h as f32);
        let kps: Vec<(f32, f32)> = f
            .kps
            .iter()
            .map(|&[kx, ky]| ((kx - a as f32) * sx, (ky - b as f32) * sy))
            .collect();

        // Re-render the crop (IdentityNet only) imposing the reference identity, then downscale back.
        let restore_req = InstantIdRequest {
            width: side,
            height: side,
            ..req.clone()
        };
        let restored = self.generate_with(&restore_req, embedding, &kps, on_progress)?;
        let small_f = resize_lanczos_u8(
            &restored.pixels,
            side as usize,
            side as usize,
            crop_h,
            crop_w,
        );
        let small: Vec<u8> = small_f.iter().map(|&v| v as u8).collect();

        // Feathered elliptical paste-back onto a copy of the base.
        let alpha = restore::feather_mask(crop_w, crop_h);
        let mut out = base.clone();
        restore::paste_alpha(&mut out, &small, crop_w, crop_h, a, b, &alpha);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_kps_rejects_short_slices() {
        // F-079: a truncated landmark list from the public seam must return a typed error rather than
        // tripping draw_kps's assert and panicking the worker.
        for len in 0..FACE_KP_COUNT {
            let kps = vec![(0.0f32, 0.0f32); len];
            let err = validate_kps(&kps).unwrap_err().to_string();
            assert!(
                err.contains("need 5 face keypoints"),
                "len {len} got: {err}"
            );
        }
        // Exactly 5 (and more) is accepted.
        assert!(validate_kps(&[(0.0, 0.0); FACE_KP_COUNT]).is_ok());
        assert!(validate_kps(&[(0.0, 0.0); FACE_KP_COUNT + 1]).is_ok());
    }
}
