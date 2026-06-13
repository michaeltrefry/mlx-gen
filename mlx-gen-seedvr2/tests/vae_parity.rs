//! VAE parity gate (sc-4813): the Rust `Seedvr2Vae` encode/decode must match the mflux MLX
//! reference. Both run MLX-Metal at f32, so the gate is tight (op-order is the only drift source).
//!
//! Weights + IO goldens come from `tools/dump_seedvr2_goldens.py --component vae` (real 3B VAE,
//! ~1 GB f32). Set `SEEDVR2_GOLDEN_DIR` (default `~/.cache/mlx-gen-seedvr2-golden`). Skipped when
//! absent. Run: `cargo test -p mlx-gen-seedvr2 --test vae_parity -- --nocapture`.

use mlx_gen::weights::Weights;
use mlx_gen_seedvr2::vae::Seedvr2Vae;
use mlx_rs::Array;

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            std::path::Path::new(&home).join(".cache/mlx-gen-seedvr2-golden")
        })
}

/// `(cosine, peak_rel=max|Δ|/max|ref|)` over two equal-length f32 slices.
fn metrics(got: &[f32], exp: &[f32]) -> (f32, f32) {
    let mut dot = 0f64;
    let (mut na, mut nb) = (0f64, 0f64);
    let (mut max_abs, mut max_ref) = (0f32, 0f32);
    for (g, e) in got.iter().zip(exp.iter()) {
        dot += (*g as f64) * (*e as f64);
        na += (*g as f64) * (*g as f64);
        nb += (*e as f64) * (*e as f64);
        max_abs = max_abs.max((g - e).abs());
        max_ref = max_ref.max(e.abs());
    }
    let cos = (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32;
    (cos, max_abs / max_ref.max(1e-12))
}

fn run(label: &str, got: &Array, exp: &Array) {
    assert_eq!(got.shape(), exp.shape(), "{label}: shape mismatch");
    // reshape to 1-D forces a contiguous logical-order copy (conv/transpose outputs are lazy views)
    let g = got
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let e = exp.reshape(&[-1]).unwrap();
    let (cos, pr) = metrics(g.as_slice::<f32>(), e.as_slice::<f32>());
    eprintln!(
        "[{label}] shape={:?} cosine={cos:.6} peak_rel={pr:.3e}",
        got.shape()
    );
    // f32 same-backend: every conv / groupnorm / pixel-shuffle stage is bit-exact; the only
    // non-exact op is the mid-block SDPA (~3.5e-3), which amplifies mildly through decode (~2e-2).
    assert!(cos > 0.9990, "{label}: cosine {cos} too low");
    assert!(pr < 3e-2, "{label}: peak_rel {pr} too high");
}

#[test]
fn seedvr2_vae_matches_reference() {
    let dir = golden_dir();
    let wpath = dir.join("vae_f32.safetensors");
    let iopath = dir.join("vae_io_f32.safetensors");
    if !wpath.exists() || !iopath.exists() {
        eprintln!(
            "SKIP: goldens absent at {dir:?} (run tools/dump_seedvr2_goldens.py --component vae)"
        );
        return;
    }

    let w = Weights::from_file(&wpath).expect("load vae weights");
    let io = Weights::from_file(&iopath).expect("load vae io");
    let vae = Seedvr2Vae::from_weights(&w).expect("build vae");

    // image mode (T=1)
    let enc_img = vae
        .encode(io.require("x_img").unwrap())
        .expect("encode img");
    run("encode_img", &enc_img, io.require("enc_img").unwrap());
    let dec_img = vae
        .decode(io.require("enc_img").unwrap())
        .expect("decode img");
    run("decode_img", &dec_img, io.require("dec_img").unwrap());

    // video mode (T=5 -> latentT 2 -> decodedT 8)
    let enc_vid = vae
        .encode(io.require("x_vid").unwrap())
        .expect("encode vid");
    run("encode_vid", &enc_vid, io.require("enc_vid").unwrap());
    let dec_vid = vae
        .decode(io.require("enc_vid").unwrap())
        .expect("decode vid");
    run("decode_vid", &dec_vid, io.require("dec_vid").unwrap());
}
