//! Quantization parity gate (sc-5198). The DiT Linears quantize to group-wise affine Q4/Q8 (the
//! same `mlx::ops::quantize` MLX/mflux use, group 64, bf16-parity cast); the VAE stays dense. We
//! validate the quantized forward against the model's **own bf16 dense** forward (the acceptance:
//! Q8 ≈ lossless, Q4 coherent) — self-contained on the converted DiT golden, no extra dump needed.
//! Also proves the in-features-divisibility predicate (so `vid_in.proj`, in=132, stays dense — else
//! `quantize` would error). Skips when the goldens are absent.

use mlx_gen::weights::Weights;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::dit::Seedvr2Transformer;
use mlx_rs::{Array, Dtype};

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(&std::env::var("HOME").unwrap())
                .join(".cache/mlx-gen-seedvr2-golden")
        })
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

/// Cast a Weights map to bf16 (the dtype the quantizer + reference compute in).
fn to_bf16(src: &Weights) -> Weights {
    let mut out = Weights::empty();
    for k in src.keys().map(String::from).collect::<Vec<_>>() {
        out.insert(
            k.clone(),
            src.require(&k).unwrap().as_dtype(Dtype::Bfloat16).unwrap(),
        );
    }
    out
}

/// Quantize the DiT, compare Q8/Q4 forwards to the bf16-dense forward on the converted golden.
fn run(weights_file: &str, io_file: &str, cfg: DitConfig, label: &str) {
    let dir = golden_dir();
    if !dir.join(weights_file).exists() {
        eprintln!("SKIP: {weights_file} absent (run tools/dump_seedvr2_goldens.py)");
        return;
    }
    let w = to_bf16(&Weights::from_file(dir.join(weights_file)).expect("dit weights"));
    let io = Weights::from_file(dir.join(io_file)).expect("dit io");
    let vid = io
        .require("vid")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let txt = io
        .require("txt")
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let ts = io.require("timestep").unwrap();

    let dense = Seedvr2Transformer::from_weights(&w, &cfg).expect("dense");
    let out_dense = dense.forward(&vid, &txt, ts).expect("dense fwd");

    let mut q8 = Seedvr2Transformer::from_weights(&w, &cfg).expect("q8");
    q8.quantize(8).expect("quantize 8"); // would error on vid_in (in=132) if the predicate were wrong
    let out_q8 = q8.forward(&vid, &txt, ts).expect("q8 fwd");
    let cos8 = cosine(&out_q8, &out_dense);

    let mut q4 = Seedvr2Transformer::from_weights(&w, &cfg).expect("q4");
    q4.quantize(4).expect("quantize 4");
    let out_q4 = q4.forward(&vid, &txt, ts).expect("q4 fwd");
    let cos4 = cosine(&out_q4, &out_dense);

    eprintln!("[{label}] Q8 cosine={cos8:.6}  Q4 cosine={cos4:.6} (vs bf16 dense)");
    // Q8 is near-lossless; Q4 is coherent (not bit-exact). Thresholds carry CI margin — this tiny
    // 16-token synthetic input amplifies per-layer quant noise vs a real full-resolution render
    // (the spike's e2e Q8 ≈ 0.998 / Q4 ≈ 0.84); the point is "near-lossless / coherent, not garbage".
    assert!(cos8 > 0.985, "{label} Q8 not near-lossless: {cos8}");
    assert!(cos4 > 0.70, "{label} Q4 incoherent: {cos4}");
}

#[test]
fn seedvr2_3b_quant_near_lossless() {
    run(
        "dit_f32.safetensors",
        "dit_io_f32.safetensors",
        DitConfig::seedvr2_3b(),
        "3B",
    );
}

#[test]
fn seedvr2_7b_quant_near_lossless() {
    run(
        "dit_7b_f32.safetensors",
        "dit_7b_io_f32.safetensors",
        DitConfig::seedvr2_7b(),
        "7B",
    );
}
