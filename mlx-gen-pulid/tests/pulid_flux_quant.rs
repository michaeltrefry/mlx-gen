//! sc-3076 — PuLID-FLUX Q8/Q4 quantization + memory footprint (real weights, `#[ignore]`d).
//!
//! Confirms the FLUX.1-dev DiT Q8/Q4 path composes with the PuLID injection: `spec.with_quant(..)`
//! flows through `load_flux1`, quantizing ONLY the backbone linears; the PuLID conditioning (EVA /
//! IDFormer / CA) stays f32 and the f32 CA residual injects into the (still f32) DiT image stream.
//! Validates that each generate completes and the identity cosine stays within the bf16 envelope
//! (sc-3074 bf16 baseline ≈ 0.68 @ 20-step/512²). Q8 and Q4 load sequentially (the model — hence the
//! quantized backbone — drops between iterations) so only one FLUX is resident at a time.
//!
//! Run:
//!   cargo test -p mlx-gen-pulid --release --test pulid_flux_quant -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, GenerationOutput, GenerationRequest, Generator, LoadSpec, Quant, WeightsSource,
};
use mlx_gen_face::FaceAnalysis;
use mlx_gen_flux::config::FluxVariant;
use mlx_gen_flux::model::load_flux1;
use mlx_gen_pulid::eva_clip::{EvaConfig, EvaVisionTransformer};
use mlx_gen_pulid::pulid_flux::PulidFlux;

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path).unwrap_or_else(|e| panic!("missing golden {path}: {e}"))
}

fn flux_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let base =
        format!("{home}/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("no FLUX.1-dev cache ({base}): {e}"))
        .flatten()
        .map(|d| d.path())
        .find(|p| p.join("transformer").is_dir())
        .expect("no FLUX.1-dev snapshot")
}

fn pulid_weights() -> Weights {
    let home = std::env::var("HOME").unwrap();
    let base = format!("{home}/.cache/huggingface/hub/models--guozinan--PuLID/snapshots");
    let path = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("no PuLID cache ({base}): {e}"))
        .flatten()
        .map(|d| d.path().join("pulid_flux_v0.9.1.safetensors"))
        .find(|p| p.exists())
        .expect("pulid_flux_v0.9.1.safetensors not in cache");
    let mut w = Weights::from_file(&path).unwrap();
    w.cast_all(mlx_rs::Dtype::Float32).unwrap();
    w
}

fn reference_face() -> Image {
    let g = golden("face_align_goldens.safetensors");
    let a = g.require("image").unwrap();
    let sh = a.shape();
    let pixels = a
        .try_as_slice::<i32>()
        .unwrap()
        .iter()
        .map(|&v| v as u8)
        .collect::<Vec<u8>>();
    Image {
        width: sh[1] as u32,
        height: sh[0] as u32,
        pixels,
    }
}

fn load_face() -> FaceAnalysis {
    FaceAnalysis::load(
        &golden("scrfd_10g.safetensors"),
        &golden("arcface_iresnet100.safetensors"),
    )
    .unwrap()
    .with_parser(&golden("bisenet_parsing.safetensors"))
    .unwrap()
}

fn build_eva() -> EvaVisionTransformer {
    EvaVisionTransformer::from_weights(
        &golden("eva_clip_golden.safetensors"),
        "w",
        EvaConfig::default(),
    )
    .unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "real-weights quant e2e: needs FLUX.1-dev + PuLID + EVA/face goldens"]
fn pulid_flux_quant_holds_identity() {
    let (steps, size) = (20u32, 512u32);
    let face_img = reference_face();
    let face = load_face();
    let ref_emb = face
        .analyze(
            &face_img.pixels,
            face_img.height as usize,
            face_img.width as usize,
        )
        .unwrap()[0]
        .embedding
        .clone();

    // bf16 backbone-weight estimate (the memory the FLUX DiT dominates): ~23.8 GB bf16,
    // ~12 GB Q8, ~6.5 GB Q4. The PuLID conditioning (EVA/IDFormer/CA, f32) is ~3.5 GB and stays
    // resident in all cases. Completion of each render confirms the quantized stack fits.
    for quant in [Quant::Q8, Quant::Q4] {
        let spec = LoadSpec::new(WeightsSource::Dir(flux_snapshot())).with_quant(quant);
        let flux = load_flux1(FluxVariant::Dev, &spec).unwrap();
        let model = PulidFlux::new(flux, build_eva(), pulid_weights(), load_face()).unwrap();

        let req = GenerationRequest {
            prompt: "a portrait photo of a person, headshot, looking at the camera".into(),
            width: size,
            height: size,
            steps: Some(steps),
            guidance: Some(4.0),
            seed: Some(42),
            conditioning: vec![Conditioning::Reference {
                image: face_img.clone(),
                strength: Some(1.0),
            }],
            ..Default::default()
        };
        let t = Instant::now();
        let out = match model.generate(&req, &mut |_| {}).unwrap() {
            GenerationOutput::Images(mut v) => v.remove(0),
            other => panic!("expected image, got {other:?}"),
        };
        let dt = t.elapsed().as_secs_f64();

        let gen = face
            .analyze(&out.pixels, out.height as usize, out.width as usize)
            .unwrap();
        let cos = gen
            .first()
            .map(|g| cosine(&g.embedding, &ref_emb))
            .unwrap_or(f32::NAN);
        println!(
            "{quant:?}: render {dt:.1}s, identity ArcFace cosine = {cos:.4} (bf16 baseline ≈0.68)"
        );
        // Quant floor: Q8 ~ bf16; Q4 degrades but identity must survive. Floor-relative, lenient.
        let floor = if quant == Quant::Q8 { 0.45 } else { 0.30 };
        assert!(
            cos > floor,
            "{quant:?} identity collapsed: cosine {cos:.4} < {floor}"
        );
        // `model` (and its quantized FLUX backbone) drops here, freeing it before the next quant.
    }
}
