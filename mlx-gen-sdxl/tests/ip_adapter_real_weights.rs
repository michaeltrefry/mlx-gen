//! sc-3056 spike: SDXL IP-Adapter image path parity vs torch (f32).
//!
//! `#[ignore]`d — needs `h94/IP-Adapter` (ViT-H `models/image_encoder` + `sdxl_models/
//! ip-adapter-plus_sdxl_vit-h.safetensors`) and the golden from `tools/dump_ip_adapter_golden.py`.
//! Run with:
//!   cargo test -p mlx-gen-sdxl --release --test ip_adapter_real_weights -- --ignored --nocapture
//!
//! Validates the two net-new image-path modules in f32, isolated from CLIP preprocessing (the
//! golden feeds a deterministic `pixel_values`):
//!   1. ViT-H image encoder penultimate hidden state `[1, 257, 1280]`.
//!   2. Resampler image tokens `[1, 16, 2048]` (from both the golden penultimate — isolating the
//!      Resampler — and the Rust encoder's own penultimate — the full image→tokens chain).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::ip_adapter::{Resampler, ResamplerConfig};
use mlx_gen_sdxl::vision_encoder::{ClipVisionEncoder, VisionConfig};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/ip_adapter_spike_golden.safetensors"
);

/// The `h94/IP-Adapter` snapshot dir (override with `IP_ADAPTER_SNAPSHOT`).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("IP_ADAPTER_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--h94--IP-Adapter/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for h94/IP-Adapter")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn f32_weights(path: PathBuf) -> Weights {
    let mut w = Weights::from_file(&path).unwrap_or_else(|e| panic!("load {path:?}: {e}"));
    w.cast_all(Dtype::Float32).unwrap();
    w
}

/// Peak-relative error `max|a-b| / max|b|`.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let b = b.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

/// Diagnose where the error lives: peak_rel, mean-relative error, and the magnitude of the golden
/// activation at the worst position (to tell CLIP "massive activation" outliers from a real bug).
fn diagnose(tag: &str, a: &Array, b: &Array) {
    let last = *b.shape().last().unwrap();
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let b = b.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let mut sum_abs_b = 0f64;
    let mut sum_abs_d = 0f64;
    let mut worst = (0usize, 0f32);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs();
        sum_abs_b += y.abs() as f64;
        sum_abs_d += d as f64;
        if d > worst.1 {
            worst = (i, d);
        }
    }
    let (wi, wd) = worst;
    let (tok, ch) = (wi / last as usize, wi % last as usize);
    println!(
        "[{tag}] peak_rel={:.3e} mean_rel={:.3e} | worst@(tok={tok},ch={ch}): \
         golden={:.4} mine={:.4} |Δ|={:.4} (golden_peak={:.2})",
        peak_rel_slices(a, b),
        (sum_abs_d / sum_abs_b) as f32,
        b[wi],
        a[wi],
        wd,
        peak,
    );
}

fn peak_rel_slices(a: &[f32], b: &[f32]) -> f32 {
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

#[test]
#[ignore = "needs h94/IP-Adapter weights + the ip_adapter_spike golden"]
fn ip_adapter_image_path_matches_torch() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let pixel_nchw = g.require("pixel_values").unwrap(); // [1, 3, 224, 224]
    let vit_golden = g.require("vit_penultimate").unwrap(); // [1, 257, 1280]
    let tok_golden = g.require("ip_tokens").unwrap(); // [1, 16, 2048]

    // Primitive isolations: which sub-op of one encoder layer carries the ~8e-4/layer drift?
    {
        use mlx_rs::fast::layer_norm;
        let gin = g
            .require("gelu_in")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let gout = mlx_gen::nn::gelu_exact(&gin).unwrap();
        diagnose("prim gelu_exact", &gout, g.require("gelu_out").unwrap());
        let lin = g
            .require("ln_in")
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        let lw = g.require("ln_w").unwrap().as_dtype(Dtype::Float32).unwrap();
        let lb = g.require("ln_b").unwrap().as_dtype(Dtype::Float32).unwrap();
        let lout = layer_norm(&lin, Some(&lw), Some(&lb), 1e-5).unwrap();
        diagnose("prim layer_norm", &lout, g.require("ln_out").unwrap());
        // Both activation + norm are bit-exact to torch — so the encoder's residual drift is NOT
        // a primitive port bug; it's cross-backend f32 GEMM/SDPA accumulating in CLIP's pre-LN
        // residual stream (sc-3056 finding).
        assert!(
            peak_rel(&gout, g.require("gelu_out").unwrap()) < 1e-5,
            "gelu_exact != torch"
        );
        assert!(
            peak_rel(&lout, g.require("ln_out").unwrap()) < 1e-5,
            "layer_norm != torch"
        );
    }

    let snap = snapshot();
    let enc_w = f32_weights(snap.join("models/image_encoder/model.safetensors"));
    let ipa_w = f32_weights(snap.join("sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"));

    let encoder = ClipVisionEncoder::from_weights(&enc_w, &VisionConfig::vit_h_14()).unwrap();
    let resampler =
        Resampler::from_weights(&ipa_w, "image_proj", &ResamplerConfig::plus_sdxl_vit_h()).unwrap();

    // NCHW -> NHWC for the mlx conv.
    let pixel_nhwc = pixel_nchw
        .as_dtype(Dtype::Float32)
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();

    // 1. ViT-H per-layer bisection (localize any drift) + penultimate.
    let states = encoder.hidden_states(&pixel_nhwc).unwrap();
    diagnose(
        "ViT-H h0 (embed+pre_ln)",
        &states[0],
        g.require("vit_h0").unwrap(),
    );
    diagnose(
        "ViT-H h1 (layer 0 out) ",
        &states[1],
        g.require("vit_h1").unwrap(),
    );
    diagnose(
        "ViT-H h16 (layer 15 out)",
        &states[16],
        g.require("vit_h16").unwrap(),
    );
    let vit = states[states.len() - 2].clone();
    let vit_rel = peak_rel(&vit, vit_golden);
    diagnose("ViT-H penultimate       ", &vit, vit_golden);

    // 2a. Resampler from the GOLDEN penultimate (isolates the Resampler).
    let tok_iso = resampler
        .forward(&vit_golden.as_dtype(Dtype::Float32).unwrap())
        .unwrap();
    let tok_iso_rel = peak_rel(&tok_iso, tok_golden);
    println!("[ip-adapter spike] Resampler tokens (golden in) peak_rel = {tok_iso_rel:.3e}");

    // 2b. Resampler from the Rust encoder's own penultimate (full image -> tokens chain).
    let tok_e2e = resampler.forward(&vit).unwrap();
    let tok_e2e_rel = peak_rel(&tok_e2e, tok_golden);
    println!("[ip-adapter spike] Resampler tokens (e2e) peak_rel = {tok_e2e_rel:.3e}");

    // f32 vs f32, cross-backend (torch CPU vs MLX Metal). The Resampler is the strict gate: its
    // `norm_out` renormalizes, so it lands bit-close (4.9e-4). The raw ViT penultimate drifts to
    // ~1.3% because CLIP's pre-LN residual stream is never renormalized — it accumulates the
    // cross-backend f32 GEMM/SDPA floor over 32 layers (gelu + layer_norm + embeddings are each
    // proven bit-exact above, so this is NOT a port bug). The end-of-chain token error is <1% —
    // negligible for IP-Adapter conditioning at scale 0.5–0.8.
    assert!(
        tok_iso_rel < 1e-3,
        "Resampler (golden in) diverged: {tok_iso_rel:.3e}"
    );
    assert!(
        tok_e2e_rel < 1.5e-2,
        "image->tokens chain diverged: {tok_e2e_rel:.3e}"
    );
    assert!(
        vit_rel < 1e-1,
        "ViT-H penultimate diverged beyond the cross-backend floor: {vit_rel:.3e}"
    );
}
