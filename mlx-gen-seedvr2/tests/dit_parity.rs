//! DiT parity gate (sc-4813): the Rust `Seedvr2Transformer` full forward must match the mflux MLX
//! reference. Both run MLX-Metal f32; the residual is the windowed-attention SDPA accumulated over
//! 32 layers (cosine ~0.99998). Weights + IO goldens from `tools/dump_seedvr2_goldens.py --component
//! dit` in `~/.cache/mlx-gen-seedvr2-golden` (override with `SEEDVR2_GOLDEN_DIR`). Skips when absent.

use mlx_gen::weights::Weights;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::dit::Seedvr2Transformer;
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

#[test]
fn seedvr2_dit_matches_reference() {
    let dir = golden_dir();
    if !dir.join("dit_f32.safetensors").exists() {
        eprintln!("SKIP: dit goldens absent at {dir:?} (run tools/dump_seedvr2_goldens.py --component dit)");
        return;
    }
    let w = Weights::from_file(dir.join("dit_f32.safetensors")).expect("dit weights");
    let io = Weights::from_file(dir.join("dit_io_f32.safetensors")).expect("dit io");
    let dit = Seedvr2Transformer::from_weights(&w, &DitConfig::seedvr2_3b()).expect("build dit");

    let out = dit
        .forward(
            io.require("vid").unwrap(),
            io.require("txt").unwrap(),
            io.require("timestep").unwrap(),
        )
        .expect("forward");
    let (cos, pr) = metrics(&out, io.require("dit_out").unwrap());
    eprintln!(
        "[dit_out] shape={:?} cosine={cos:.6} peak_rel={pr:.3e}",
        out.shape()
    );
    assert!(cos > 0.999, "dit cosine {cos} too low");
    assert!(pr < 3e-2, "dit peak_rel {pr} too high");
}

/// 7B parity (sc-5197): the 36-layer pixel-mode-RoPE / GELU-MLP / no-output-ada variant. Exercises
/// the `rope_pixel` path (normalized `linspace` positions, no temporal offset, `rope_on_text=false`).
/// Goldens from `--component dit --model 7b` (`dit_7b_*`).
#[test]
fn seedvr2_dit_7b_matches_reference() {
    let dir = golden_dir();
    if !dir.join("dit_7b_f32.safetensors").exists() {
        eprintln!(
            "SKIP: 7B dit goldens absent (run tools/dump_seedvr2_goldens.py --component dit --model 7b)"
        );
        return;
    }
    let w = Weights::from_file(dir.join("dit_7b_f32.safetensors")).expect("7B dit weights");
    let io = Weights::from_file(dir.join("dit_7b_io_f32.safetensors")).expect("7B dit io");
    let dit = Seedvr2Transformer::from_weights(&w, &DitConfig::seedvr2_7b()).expect("build 7B dit");

    let out = dit
        .forward(
            io.require("vid").unwrap(),
            io.require("txt").unwrap(),
            io.require("timestep").unwrap(),
        )
        .expect("forward");
    let (cos, pr) = metrics(&out, io.require("dit_out").unwrap());
    eprintln!(
        "[dit_7b_out] shape={:?} cosine={cos:.6} peak_rel={pr:.3e}",
        out.shape()
    );
    assert!(cos > 0.999, "7B dit cosine {cos} too low");
    assert!(pr < 3e-2, "7B dit peak_rel {pr} too high");
}
