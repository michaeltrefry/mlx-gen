//! Converter byte-exact gate (sc-4813): the native Rust converter applied to the raw
//! `numz/SeedVR2_comfyUI` checkpoint must reproduce the mflux-converted MLX weights key-for-key,
//! value-for-value. The dumped `vae_f32`/`dit_f32` goldens ARE the mflux-converted weights (cast to
//! f32); the converter output (fp16, cast to f32) must match them exactly (rename + conv transpose
//! are lossless). Needs the HF cache + the goldens; skips otherwise.

use mlx_gen::weights::Weights;
use mlx_gen_seedvr2::convert::{convert_dit, convert_vae};
use mlx_rs::Dtype;

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(&std::env::var("HOME").unwrap())
                .join(".cache/mlx-gen-seedvr2-golden")
        })
}

/// Locate a file inside the HF snapshot for `numz/SeedVR2_comfyUI`.
fn raw_checkpoint(name: &str) -> Option<std::path::PathBuf> {
    let base = std::path::Path::new(&std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    let p = snap.join(name);
    p.exists().then_some(p)
}

fn assert_bit_exact(conv: &Weights, golden: &Weights) {
    let gkeys: Vec<&str> = golden.keys().collect();
    assert_eq!(
        conv.keys().count(),
        gkeys.len(),
        "key count: converter {} vs golden {}",
        conv.keys().count(),
        gkeys.len()
    );
    let mut max_diff = 0f32;
    for k in gkeys {
        let g = golden.require(k).unwrap();
        let c = conv
            .get(k)
            .unwrap_or_else(|| panic!("converter missing key {k}"));
        assert_eq!(
            c.shape(),
            g.shape(),
            "shape mismatch {k}: {:?} vs {:?}",
            c.shape(),
            g.shape()
        );
        let cf = c.as_dtype(Dtype::Float32).unwrap().reshape(&[-1]).unwrap();
        let gf = g.reshape(&[-1]).unwrap();
        for (a, b) in cf.as_slice::<f32>().iter().zip(gf.as_slice::<f32>().iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
    }
    eprintln!("max|Δ| over all tensors = {max_diff:.3e}");
    assert!(
        max_diff == 0.0,
        "converter not byte-exact: max|Δ|={max_diff}"
    );
}

#[test]
fn vae_converter_byte_exact() {
    let (Some(raw), dir) = (raw_checkpoint("ema_vae_fp16.safetensors"), golden_dir()) else {
        eprintln!("SKIP: raw VAE checkpoint absent");
        return;
    };
    if !dir.join("vae_f32.safetensors").exists() {
        eprintln!("SKIP: vae golden absent");
        return;
    }
    let conv = convert_vae(&Weights::from_file(raw).unwrap()).unwrap();
    assert_bit_exact(
        &conv,
        &Weights::from_file(dir.join("vae_f32.safetensors")).unwrap(),
    );
}

#[test]
fn dit_converter_byte_exact() {
    let (Some(raw), dir) = (
        raw_checkpoint("seedvr2_ema_3b_fp16.safetensors"),
        golden_dir(),
    ) else {
        eprintln!("SKIP: raw DiT checkpoint absent");
        return;
    };
    if !dir.join("dit_f32.safetensors").exists() {
        eprintln!("SKIP: dit golden absent");
        return;
    }
    let conv = convert_dit(&Weights::from_file(raw).unwrap()).unwrap();
    assert_bit_exact(
        &conv,
        &Weights::from_file(dir.join("dit_f32.safetensors")).unwrap(),
    );
}
