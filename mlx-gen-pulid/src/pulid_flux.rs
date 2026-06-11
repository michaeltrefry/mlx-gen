//! PuLID-FLUX end-to-end generate (sc-3074) + native face wiring (sc-3073).
//!
//! Assembles the full face-identity path on top of the FLUX.1-dev backbone:
//!   1. **Face analysis** (native MLX, epic 3079): the reference face (`Conditioning::Reference`) →
//!      `mlx_gen_face::FaceAnalysis` → largest face's ArcFace embedding (512-d) + `face_features_image`
//!      (512² aligned, background-whitened grayscale). No Python/onnx.
//!   2. **EVA-CLIP** (sc-3070): `face_features_image` → resize/normalize → `id_cond_vit` (768-d,
//!      L2-normalized) + 5 hidden states.
//!   3. **IDFormer** (sc-3071): `id_cond = cat(arcface 512, id_cond_vit 768)` + hidden → `id_embedding`
//!      [1,32,2048].
//!   4. **CA injection** (sc-3072): build `PulidCa` and run the FLUX flow-match denoise through
//!      `Flux1::generate_with_injector` (fake-CFG, true_cfg=1.0) → AE decode.
//!
//! The whole conditioning path runs in **f32**: mlx-gen-flux keeps the DiT image stream in f32 (mixed
//! precision, sc-2787), so f32 CA weights/id_embedding inject cleanly into the f32 hidden tokens (no
//! dtype mismatch) and at higher accuracy than the reference's bf16 — the e2e gate is ArcFace-cosine
//! (cross-encoder, loose), so this is strictly safe. Real-CFG / uncond-id is sc-3075; quant is sc-3076.

use std::path::{Path, PathBuf};

use mlx_rs::ops::{concatenate_axis, divide, sqrt, square, sum_axes};
use mlx_rs::{Array, Dtype};

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    gen_core, Capabilities, Conditioning, ConditioningKind, Error, GenerationOutput,
    GenerationRequest, Generator, LoadSpec, Modality, ModelDescriptor, ModelRegistration, Progress,
    Quant, Result,
};
use mlx_gen_face::FaceAnalysis;
use mlx_gen_flux::config::FluxVariant;
use mlx_gen_flux::model::{load_flux1, Flux1};

use crate::ca::PulidCa;
use crate::eva_clip::{transform, EvaConfig, EvaVisionTransformer};
use crate::idformer::{IdFormer, IdFormerConfig};

/// FLUX.1-dev DiT block counts (the PuLID injection schedule is defined over these).
const NUM_DOUBLE_BLOCKS: usize = 19;
const NUM_SINGLE_BLOCKS: usize = 38;
/// Default step from which the real-CFG (and uncond-id) branch engages when the request leaves
/// `timestep_to_start_cfg` unset — the upstream PuLID default (the photoreal preset overrides to 4
/// via `req.timestep_to_start_cfg`).
const DEFAULT_TIMESTEP_TO_START_CFG: usize = 1;
/// ArcFace (antelopev2) face embedding width — the first half of the IdFormer `id_cond`
/// (`cat(arcface, id_cond_vit)`). The id_cond_vit half is the EVA head's `proj_dim`.
const ARCFACE_DIM: i32 = 512;

pub fn descriptor() -> ModelDescriptor {
    ModelDescriptor {
        id: "pulid_flux",
        family: "pulid",
        backend: "mlx",
        modality: Modality::Image,
        capabilities: Capabilities {
            supports_negative_prompt: false, // real-CFG + negative prompt = sc-3075
            supports_guidance: true,         // FLUX.1-dev guidance-distilled CFG (default ~4.0)
            supports_true_cfg: false,        // sc-3075
            conditioning: vec![ConditioningKind::Reference], // the reference face
            supports_lora: false,
            supports_lokr: false,
            samplers: vec!["flow_match"],
            schedulers: vec!["linear"],
            min_size: 256,
            max_size: 2048,
            max_count: 8,
            mac_only: true,
            supports_kv_cache: false,
            requires_sigma_shift: true, // dev
            supported_quants: &[Quant::Q4, Quant::Q8],
        },
    }
}

/// L2-normalize each row of `[B, D]` over the feature axis (the PuLID `id_cond_vit` normalization).
fn l2_normalize_rows(x: &Array) -> Result<Array> {
    let norm = sqrt(&sum_axes(&square(x)?, &[1], true)?)?; // [B, 1]
    Ok(divide(x, &norm)?)
}

pub struct PulidFlux {
    descriptor: ModelDescriptor,
    flux: Flux1,
    eva: EvaVisionTransformer,
    idformer: IdFormer,
    /// The PuLID checkpoint weights (f32) — kept to build a per-generate [`PulidCa`] bound to the
    /// computed id_embedding. `pulid_encoder.*` already consumed by `idformer`; `pulid_ca.*` here.
    pulid: Weights,
    face: FaceAnalysis,
}

impl PulidFlux {
    /// Build from already-loaded sub-models. `pulid` must hold both `pulid_encoder.*` and
    /// `pulid_ca.*` (cast to f32); `eva`/`idformer` must likewise be f32 (the conditioning path).
    /// `face` must have a parser attached (`with_parser`) for `face_features_image`.
    pub fn new(
        flux: Flux1,
        eva: EvaVisionTransformer,
        pulid: Weights,
        face: FaceAnalysis,
    ) -> Result<Self> {
        let idformer = IdFormer::from_weights(&pulid, "pulid_encoder", IdFormerConfig::default())?;
        Ok(Self {
            descriptor: descriptor(),
            flux,
            eva,
            idformer,
            pulid,
            face,
        })
    }

    /// Face image (RGB, row-major, `h×w`) → `id_embedding` `[1,32,2048]`. Mirrors PuLID's
    /// `get_id_embedding` (the conditional side; cal_uncond is sc-3075).
    pub fn compute_id_embedding(&self, pixels: &[u8], h: usize, w: usize) -> Result<Array> {
        let faces = self.face.analyze(pixels, h, w)?;
        let face = faces.first().ok_or_else(|| {
            Error::Msg("pulid_flux: no face detected in the reference image".into())
        })?;
        // ArcFace 512-d (id_ante_embedding) — raw, un-normalized, matching the reference.
        let arcface = Array::from_slice(&face.embedding, &[1, face.embedding.len() as i32]);
        // face_features_image (512² aligned, bg-whitened gray) → EVA 336² transform → tower.
        let ffi = self.face.face_features_image(pixels, h, w, face)?;
        let eva_in = transform::eva_transform(&ffi, self.eva_image_size())?;
        let eva_out = self.eva.forward(&eva_in)?;
        let id_cond_vit = l2_normalize_rows(&eva_out.id_cond_vit)?; // [1,768]
        let id_cond = concatenate_axis(&[&arcface, &id_cond_vit], 1)?; // [1,1280]
        self.idformer.forward(&id_cond, &eva_out.hidden)
    }

    /// The unconditional id_embedding — IDFormer over **zeroed** id_cond + zeroed hidden states (the
    /// PuLID `get_id_embedding(cal_uncond=True)` path), injected on the negative real-CFG branch.
    pub fn compute_uncond_id_embedding(&self) -> Result<Array> {
        // Derive the EVA token geometry from the loaded tower's config, not the default-tower
        // constants (F-082): seq = grid²+1 (CLS + patches) = 577, embed = embed_dim = 1024, and one
        // zeroed hidden state per captured block (5). `id_cond` is the IdFormer input width (ArcFace
        // 512 + the EVA head's proj_dim 768).
        let cfg = self.eva.config();
        let seq = cfg.grid() * cfg.grid() + 1;
        let embed = cfg.embed_dim;
        let id_cond_dim = ARCFACE_DIM + cfg.proj_dim;
        let id_cond = Array::from_slice(&vec![0f32; id_cond_dim as usize], &[1, id_cond_dim]);
        let hidden: Vec<Array> = cfg
            .hidden_capture
            .iter()
            .map(|_| Array::from_slice(&vec![0f32; (seq * embed) as usize], &[1, seq, embed]))
            .collect();
        self.idformer.forward(&id_cond, &hidden)
    }

    fn eva_image_size(&self) -> i32 {
        self.eva.config().image_size
    }

    fn reference_face<'a>(&self, req: &'a GenerationRequest) -> Result<(&'a Image, f32)> {
        select_reference_face(&req.conditioning)
    }
}

/// Pick the single reference-face conditioning, rejecting any other kind. PuLID-FLUX advertises only
/// `ConditioningKind::Reference`, so a stray Control/Mask/etc. attached by a worker must **error**
/// rather than be silently dropped when `generate` clears `flux_req.conditioning` (F-094).
fn select_reference_face(conditioning: &[Conditioning]) -> Result<(&Image, f32)> {
    let mut found = None;
    for c in conditioning {
        match c {
            Conditioning::Reference { image, strength } => {
                if found.is_some() {
                    return Err(Error::Msg(
                        "pulid_flux: exactly one reference face is supported".into(),
                    ));
                }
                // The reference strength is the PuLID id_weight (0–3, default 1.0).
                found = Some((image, strength.unwrap_or(1.0)));
            }
            other => {
                return Err(Error::Msg(format!(
                    "pulid_flux: unsupported conditioning {:?} — only a reference face \
                     (Conditioning::Reference) is accepted",
                    other.kind()
                )));
            }
        }
    }
    found.ok_or_else(|| {
        Error::Msg(
            "pulid_flux: a reference face image (Conditioning::Reference) is required".into(),
        )
    })
}

impl Generator for PulidFlux {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }

    fn validate(&self, req: &GenerationRequest) -> gen_core::Result<()> {
        // Require a reference face; the FLUX backbone validates the rest (size/steps/sampler).
        self.reference_face(req)?;
        Ok(())
    }

    fn generate(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> gen_core::Result<GenerationOutput> {
        self.generate_impl(req, on_progress).map_err(Into::into)
    }
}

impl PulidFlux {
    /// The rich-`Result` body behind [`Generator::generate`]. Kept on the crate's own
    /// [`mlx_gen::Error`] so the `?` operator lifts both `mlx_rs` device exceptions and the family
    /// helpers transparently; the trait wrapper bridges the tail into [`gen_core::Error`] (epic 3720).
    fn generate_impl(
        &self,
        req: &GenerationRequest,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let (image, id_weight) = self.reference_face(req)?;
        let id_embedding =
            self.compute_id_embedding(&image.pixels, image.height as usize, image.width as usize)?;
        let mk_ca = |emb: Array| {
            PulidCa::from_weights(
                &self.pulid,
                "pulid_ca",
                emb,
                id_weight,
                NUM_DOUBLE_BLOCKS,
                NUM_SINGLE_BLOCKS,
            )
        };
        // The reference face is consumed into the injector; hand the FLUX backbone a plain request
        // (it rejects conditioning + negative_prompt it doesn't itself implement — both are handled
        // here / passed to the CFG denoise directly).
        let mut flux_req = req.clone();
        flux_req.conditioning = Vec::new();
        flux_req.negative_prompt = None;
        flux_req.true_cfg = None; // PuLID drives real-CFG itself; the backbone forbids it

        let true_cfg = req.true_cfg.unwrap_or(1.0);
        if true_cfg > 1.0 + 1e-3 {
            // Real-CFG (sc-3075): positive (id) + negative (uncond id) branches + a negative prompt.
            let pos = mk_ca(id_embedding)?;
            let neg = mk_ca(self.compute_uncond_id_embedding()?)?;
            let neg_prompt = req.negative_prompt.as_deref().unwrap_or("");
            let start_cfg = req
                .timestep_to_start_cfg
                .map(|v| v as usize)
                .unwrap_or(DEFAULT_TIMESTEP_TO_START_CFG);
            self.flux.generate_with_injector_cfg(
                &flux_req,
                &pos,
                &neg,
                neg_prompt,
                true_cfg,
                start_cfg,
                on_progress,
            )
        } else {
            // Fake-CFG (true_cfg = 1.0): single forward (sc-3074), bit-identical to that path.
            self.flux
                .generate_with_injector(&flux_req, Some(&mk_ca(id_embedding)?), on_progress)
        }
    }
}

// ---- registration -------------------------------------------------------------------------------

/// Resolve a required file path from an env var, erroring with the var name if unset/missing.
fn env_path(var: &str) -> Result<PathBuf> {
    let p = std::env::var(var)
        .map_err(|_| Error::Msg(format!("pulid_flux: set {var} to the weights path")))?;
    let p = PathBuf::from(p);
    if !p.exists() {
        return Err(Error::Msg(format!(
            "pulid_flux: {var} path does not exist: {}",
            p.display()
        )));
    }
    Ok(p)
}

/// Locate `pulid_flux_v0.9.1.safetensors` — `PULID_FLUX_WEIGHTS` override, else the HF cache.
fn resolve_pulid_weights() -> Result<PathBuf> {
    // Route the override through `env_path` so a typo'd path errors with the var name up front,
    // matching the sibling weight-path helpers (F-093), instead of a bare later I/O error.
    if std::env::var_os("PULID_FLUX_WEIGHTS").is_some() {
        return env_path("PULID_FLUX_WEIGHTS");
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let glob = format!("{home}/.cache/huggingface/hub/models--guozinan--PuLID/snapshots");
    let snaps = std::fs::read_dir(&glob).map_err(|e| {
        Error::Msg(format!(
            "pulid_flux: no PuLID cache ({glob}): {e}; set PULID_FLUX_WEIGHTS"
        ))
    })?;
    for s in snaps.flatten() {
        let cand = s.path().join("pulid_flux_v0.9.1.safetensors");
        if cand.exists() {
            return Ok(cand);
        }
    }
    Err(Error::Msg(
        "pulid_flux: pulid_flux_v0.9.1.safetensors not found; set PULID_FLUX_WEIGHTS".into(),
    ))
}

/// Load EVA weights (f32) from a converted safetensors (tools/convert_eva_clip.py output). Keys are
/// bare mlx-names (no prefix).
fn load_eva(path: &Path) -> Result<EvaVisionTransformer> {
    let mut w = Weights::from_file(path)?;
    w.cast_all(Dtype::Float32)?;
    EvaVisionTransformer::from_weights(&w, "", EvaConfig::default())
}

/// Registered loader for the `pulid_flux` target. Weight sources:
///   * FLUX.1-dev snapshot dir — `spec.weights` (Dir).
///   * `PULID_FLUX_WEIGHTS` — pulid_flux_v0.9.1.safetensors (else HF cache).
///   * `PULID_EVA_WEIGHTS` — converted EVA02-CLIP-L-14-336 safetensors.
///   * `PULID_FACE_WEIGHTS_DIR` — dir with scrfd_10g / arcface_iresnet100 / bisenet_parsing.
pub fn load_pulid_flux(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    // FLUX.1-dev backbone (its loader validates the snapshot dir). Q8/Q4 (sc-3076) composes for free:
    // `spec.quantize` flows through `load_flux1`, quantizing ONLY the FLUX backbone linears. The PuLID
    // conditioning (EVA tower, IDFormer, the 20 CA modules) stays f32 — it runs once per image, not
    // per step, so the memory win is the backbone, and the f32 CA residual injects into the (still
    // f32) DiT image stream unchanged. No quant-specific wiring needed here.
    let flux = load_flux1(FluxVariant::Dev, spec)?;

    // PuLID encoder + CA weights, cast f32 (conditioning path).
    let mut pulid = Weights::from_file(resolve_pulid_weights()?)?;
    pulid.cast_all(Dtype::Float32)?;

    // EVA-CLIP tower (f32).
    let eva = load_eva(&env_path("PULID_EVA_WEIGHTS")?)?;

    // Native face stack.
    let face_dir = env_path("PULID_FACE_WEIGHTS_DIR")?;
    let face = FaceAnalysis::load(
        &Weights::from_file(face_dir.join("scrfd_10g.safetensors"))?,
        &Weights::from_file(face_dir.join("arcface_iresnet100.safetensors"))?,
    )?
    .with_parser(&Weights::from_file(
        face_dir.join("bisenet_parsing.safetensors"),
    )?)?;

    Ok(Box::new(PulidFlux::new(flux, eva, pulid, face)?))
}

/// Registry adapter: the link-time registry's `load` slot is typed on the backend-neutral
/// [`gen_core::Result`] (epic 3720); bridge the crate's rich-`Result` [`load_pulid_flux`] into it.
fn load_pulid_flux_registered(spec: &LoadSpec) -> gen_core::Result<Box<dyn Generator>> {
    load_pulid_flux(spec).map_err(Into::into)
}

inventory::submit! {
    ModelRegistration { descriptor, load: load_pulid_flux_registered }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulid_weights_override_validates_existence() {
        // F-093: a set-but-nonexistent PULID_FLUX_WEIGHTS errors up front with the var name (routed
        // through env_path), not a bare later I/O error. (RUST_TEST_THREADS=1 makes the env mutation
        // safe — see .cargo/config.toml.)
        std::env::set_var(
            "PULID_FLUX_WEIGHTS",
            "/nonexistent/pulid_flux_v0.9.1.safetensors",
        );
        let err = resolve_pulid_weights().unwrap_err().to_string();
        std::env::remove_var("PULID_FLUX_WEIGHTS");
        assert!(err.contains("PULID_FLUX_WEIGHTS"), "got: {err}");
        assert!(err.contains("does not exist"), "got: {err}");
    }

    fn img() -> Image {
        Image {
            width: 1,
            height: 1,
            pixels: vec![0u8; 3],
        }
    }

    /// F-094: a single reference face is accepted (its strength becomes the id_weight, default 1.0).
    #[test]
    fn select_reference_accepts_single_reference() {
        let cond = vec![Conditioning::Reference {
            image: img(),
            strength: Some(2.0),
        }];
        let (_, weight) = select_reference_face(&cond).unwrap();
        assert_eq!(weight, 2.0);
    }

    /// F-094: a non-Reference conditioning (e.g. a stray Mask/Control) is rejected rather than
    /// silently dropped when `generate` clears the conditioning.
    #[test]
    fn select_reference_rejects_unsupported_conditioning() {
        let cond = vec![
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
            Conditioning::Mask { image: img() },
        ];
        let err = select_reference_face(&cond).unwrap_err().to_string();
        assert!(err.contains("unsupported"), "got: {err}");
    }

    /// Two reference faces are rejected, and an empty request is rejected as missing.
    #[test]
    fn select_reference_rejects_multiple_and_missing() {
        let two = vec![
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
            Conditioning::Reference {
                image: img(),
                strength: None,
            },
        ];
        assert!(select_reference_face(&two)
            .unwrap_err()
            .to_string()
            .contains("exactly one"));
        assert!(select_reference_face(&[])
            .unwrap_err()
            .to_string()
            .contains("required"));
    }
}
