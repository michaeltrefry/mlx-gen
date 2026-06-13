//! End-to-end image-mode parity gate (sc-4813): the Rust pipeline's MODEL path
//! (encode → condition → DiT one-step → decode → crop) must match the mflux reference
//! `SeedVR2.generate_image` (pre color-correction). Noise + neg-embed are injected from the golden
//! so the gate isolates the model from the RNG / color-correction. f32 both sides.
//!
//! Goldens from `tools/dump_seedvr2_goldens.py --component e2e` in `~/.cache/mlx-gen-seedvr2-golden`.

use mlx_gen::weights::Weights;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::pipeline::Seedvr2Pipeline;
use mlx_rs::Array;

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(&std::env::var("HOME").unwrap())
                .join(".cache/mlx-gen-seedvr2-golden")
        })
}

fn metrics(got: &Array, exp: &Array) -> (f32, f32) {
    let g = got
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let e = exp.reshape(&[-1]).unwrap();
    let (gs, es) = (g.as_slice::<f32>(), e.as_slice::<f32>());
    let (mut dot, mut na, mut nb, mut maxd, mut maxr) = (0f64, 0f64, 0f64, 0f32, 0f32);
    for (a, b) in gs.iter().zip(es.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
        maxd = maxd.max((a - b).abs());
        maxr = maxr.max(b.abs());
    }
    (
        (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32,
        maxd / maxr.max(1e-12),
    )
}
fn report(label: &str, got: &Array, exp: &Array) -> f32 {
    assert_eq!(
        got.shape(),
        exp.shape(),
        "{label} shape {:?} vs {:?}",
        got.shape(),
        exp.shape()
    );
    let (cos, pr) = metrics(got, exp);
    eprintln!(
        "[{label}] shape={:?} cosine={cos:.6} peak_rel={pr:.3e}",
        got.shape()
    );
    cos
}

#[test]
fn seedvr2_e2e_model_path_matches_reference() {
    let dir = golden_dir();
    if !dir.join("e2e_io_f32.safetensors").exists() || !dir.join("dit_f32.safetensors").exists() {
        eprintln!(
            "SKIP: e2e/dit goldens absent (run tools/dump_seedvr2_goldens.py --component e2e)"
        );
        return;
    }
    let vae_w = Weights::from_file(dir.join("vae_f32.safetensors")).expect("vae weights");
    let dit_w = Weights::from_file(dir.join("dit_f32.safetensors")).expect("dit weights");
    let io = Weights::from_file(dir.join("e2e_io_f32.safetensors")).expect("e2e io");
    let pipe =
        Seedvr2Pipeline::from_weights(&vae_w, &dit_w, &DitConfig::seedvr2_3b()).expect("pipeline");

    // stage: VAE encode (vs the reference's tiled encode — single tile at 256 → should match)
    let latent = pipe
        .encode(io.require("processed").unwrap())
        .expect("encode");
    report("encode", &latent, io.require("initial_latent").unwrap());

    // stage: denoise (inject reference noise + the reference's condition latent)
    let cond = Seedvr2Pipeline::condition(io.require("initial_latent").unwrap()).expect("cond");
    let latents = pipe
        .denoise(
            io.require("noise").unwrap(),
            &cond,
            io.require("neg_embed").unwrap(),
            io.require("timestep").unwrap(),
        )
        .expect("denoise");
    report("latents", &latents, io.require("latents").unwrap());

    // full model path → decoded (pre color-correction)
    let decoded = pipe
        .run_model(
            io.require("processed").unwrap(),
            io.require("noise").unwrap(),
            io.require("neg_embed").unwrap(),
            io.require("timestep").unwrap(),
            256,
            256,
        )
        .expect("run_model");
    let cos = report("decoded", &decoded, io.require("decoded").unwrap());
    assert!(cos > 0.999, "decoded cosine {cos} too low");
}

#[test]
fn seedvr2_color_correction_matches_reference() {
    let dir = golden_dir();
    if !dir.join("e2e_io_f32.safetensors").exists() {
        eprintln!("SKIP: e2e goldens absent");
        return;
    }
    let io = Weights::from_file(dir.join("e2e_io_f32.safetensors")).expect("e2e io");
    // isolate the post-process: feed the reference decoded + style, expect the reference final image
    let out = mlx_gen_seedvr2::color::apply_color_correction(
        io.require("decoded").unwrap(),
        io.require("style").unwrap(),
        0.8,
    )
    .expect("color correction");
    let cos = report("final", &out, io.require("final").unwrap());
    assert!(cos > 0.999, "color-corrected cosine {cos} too low");
}
