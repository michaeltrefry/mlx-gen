//! Registry + real-weight load/run gate (sc-4813). Loads the SeedVR2 3B pipeline from the **raw**
//! `numz/SeedVR2_comfyUI` checkpoint dir (exercising the native converter + load path on real
//! weights), then: (a) the bundled neg-embed matches the reference; (b) the full model path matches
//! the golden `decoded`; (c) `generate` runs end-to-end on an image and returns the right size.
//! Needs the HF cache + e2e golden; skips otherwise.

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::pipeline::Seedvr2Pipeline;
use mlx_rs::{Array, Dtype};

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(&std::env::var("HOME").unwrap())
                .join(".cache/mlx-gen-seedvr2-golden")
        })
}

fn raw_dir() -> Option<std::path::PathBuf> {
    let base = std::path::Path::new(&std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("seedvr2_ema_3b_fp16.safetensors")
        .exists()
        .then_some(snap)
}

fn cosine(got: &Array, exp: &Array) -> f32 {
    let g = got
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let e = exp
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let (gs, es) = (g.as_slice::<f32>(), e.as_slice::<f32>());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in gs.iter().zip(es.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
}

#[test]
fn seedvr2_loads_and_runs_from_real_checkpoint() {
    let (Some(raw), gdir) = (raw_dir(), golden_dir()) else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    if !gdir.join("e2e_io_f32.safetensors").exists() {
        eprintln!("SKIP: e2e golden absent");
        return;
    }
    // load f32 from the raw checkpoint (native convert + load), then check the model path.
    let pipe = Seedvr2Pipeline::load(
        &raw,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Float32,
    )
    .expect("load from raw checkpoint");
    let io = Weights::from_file(gdir.join("e2e_io_f32.safetensors")).expect("e2e io");

    // (a) bundled neg-embed matches the reference
    let neg_cos = cosine(
        pipe.neg_embed().expect("neg-embed"),
        io.require("neg_embed").unwrap(),
    );
    eprintln!("neg_embed cosine = {neg_cos:.6}");
    assert!(neg_cos > 0.9999, "bundled neg-embed mismatch: {neg_cos}");

    // (b) full model path on real-converted weights vs golden decoded
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
    let cos = cosine(&decoded, io.require("decoded").unwrap());
    eprintln!("real-weight decoded cosine = {cos:.6}");
    assert!(cos > 0.999, "real-weight model path diverged: {cos}");

    // (c) full generate() smoke: a small synthetic LR image → 256×256 RGB8, no panic
    let lr = Image {
        width: 96,
        height: 96,
        pixels: (0..96 * 96 * 3).map(|i| (i % 256) as u8).collect(),
    };
    let out = pipe.generate(&lr, 256, 256, 42, 0.0).expect("generate");
    assert_eq!((out.width, out.height), (256, 256));
    assert_eq!(out.pixels.len(), 256 * 256 * 3);
    eprintln!(
        "generate ok: {}x{} ({} px)",
        out.width,
        out.height,
        out.pixels.len()
    );
}
