//! Kolors IP-Adapter-Plus parity (sc-3098).
//!
//! `#[ignore]`d: needs the `Kwai-Kolors/Kolors-IP-Adapter-Plus` snapshot (+ the Kolors snapshot &
//! `tokenizer.json` for the e2e gate) and `tools/dump_kolors_ip_adapter_golden.py`. Gates isolate
//! each component, then verify the wiring:
//!
//!  - **preprocess**: `preprocess_clip_image_sized(image, 336)` matches the dumped CLIP `pixels`.
//!  - **encoder**: `ClipVisionEncoder::vit_l_14_336.penultimate(pixels)` matches transformers'
//!    `hidden_states[-2]`.
//!  - **resampler**: `Resampler(kolors_plus).forward(penultimate)` matches the torch Tencent
//!    Resampler's 16×2048 tokens (pins dim/depth/heads/dim_head).
//!  - **wiring** (f32): `denoise_ip(ip_scale=0)` is byte-identical to plain T2I (the IP injection is
//!    non-destructive); `ip_scale>0` perturbs the output and renders coherently. (f32 for the
//!    byte-exact invariant — same bf16-chaos caveat as the ControlNet gate.)
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test ip_adapter_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_kolors::ip_adapter::load_kolors_ip_adapter;
use mlx_gen_kolors::Kolors;
use mlx_gen_sdxl::{
    preprocess_clip_image_sized, ClipVisionEncoder, Resampler, ResamplerConfig, VisionConfig,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_ip_adapter_golden.safetensors"
);

fn kolors_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn ip_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_IP_ADAPTER") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-IP-Adapter-Plus/snapshots");
    std::fs::read_dir(&snaps)
        .expect("IP-Adapter snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let (a, b) = (a.reshape(&[n]).unwrap(), b.reshape(&[n]).unwrap());
    let (a, b) = (
        a.as_dtype(Dtype::Float32).unwrap(),
        b.as_dtype(Dtype::Float32).unwrap(),
    );
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-9);
    let mabs = (b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32).max(1e-9);
    let max_d = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_d / peak, mean_d / mabs)
}

/// Cosine similarity (the meaningful fidelity metric for a conditioning tensor — robust to the
/// element-wise cross-backend f32 noise that inflates peak-rel on a deep transformer).
fn cosine(a: &Array, b: &Array) -> f64 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let b = b.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn golden_image(g: &Weights) -> Image {
    let gi = g.require("image").unwrap();
    let sh = gi.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let n = sh.iter().product::<i32>();
    let pixels: Vec<u8> = gi
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

#[test]
#[ignore = "needs the Kolors-IP-Adapter-Plus snapshot + tools/golden/kolors_ip_adapter_golden.safetensors"]
fn kolors_ip_components_match_reference() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let snap = ip_snapshot();

    // ---- preprocess ----
    let image = golden_image(&g);
    let pre = preprocess_clip_image_sized(&image, 336).unwrap();
    let (pp, pm) = rel(&pre, g.require("pixels").unwrap());
    println!("preprocess: peak_rel={pp:.3e} mean_rel={pm:.3e}");
    assert!(pp < 1e-2, "preprocess peak_rel {pp:.3e} exceeds 1e-2");

    // ---- encoder (feed the golden pixels so this is a pure-encoder gate) ----
    let mut enc_w = Weights::from_file(snap.join("image_encoder/model.safetensors")).unwrap();
    enc_w.cast_all(Dtype::Float32).unwrap();
    let encoder = ClipVisionEncoder::from_weights(&enc_w, &VisionConfig::vit_l_14_336()).unwrap();
    let penult = encoder.penultimate(g.require("pixels").unwrap()).unwrap();
    let (ep, em) = rel(&penult, g.require("penultimate").unwrap());
    let ec = cosine(&penult, g.require("penultimate").unwrap());
    println!("encoder penultimate: peak_rel={ep:.3e} mean_rel={em:.3e} cosine={ec:.6}");
    // The penultimate is layer 23 of a 24-layer ViT — the torch-CPU-vs-MLX-Metal f32 noise
    // accumulates over the depth to ~1.7e-2 *peak* (a single worst element), but the tensor is
    // structurally identical (cosine ~1, tiny mean-rel). Cosine is the load-bearing gate for a
    // conditioning tensor; the loose peak bound just rules out a gross activation/layout bug.
    assert!(
        ec > 0.999,
        "encoder penultimate cosine {ec:.6} below 0.999 (encoder bug)"
    );
    assert!(
        ep < 3e-2,
        "encoder penultimate peak_rel {ep:.3e} exceeds 3e-2 floor"
    );

    // ---- resampler (feed the golden penultimate so this is a pure-resampler gate) ----
    let mut ip_w = Weights::from_file(snap.join("ip_adapter_plus_general.safetensors")).unwrap();
    ip_w.cast_all(Dtype::Float32).unwrap();
    let resampler =
        Resampler::from_weights(&ip_w, "image_proj", &ResamplerConfig::kolors_plus()).unwrap();
    let tokens = resampler
        .forward(g.require("penultimate").unwrap())
        .unwrap();
    let (rp, rm) = rel(&tokens, g.require("tokens").unwrap());
    println!("resampler tokens: peak_rel={rp:.3e} mean_rel={rm:.3e}");
    assert!(
        rp < 5e-3,
        "resampler tokens peak_rel {rp:.3e} exceeds 5e-3 (dim/heads/dim_head wrong?)"
    );
    println!("✓ Kolors IP-Adapter components (preprocess + ViT-L-336 encoder + Resampler) match the reference");
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json + Kolors-IP-Adapter-Plus snapshot"]
fn kolors_ip_scale0_is_base() {
    // f32 for the byte-exact invariant (same rationale as the ControlNet gate).
    let snap = kolors_snapshot();
    let mut kolors = Kolors::load(&snap, Dtype::Float32).expect("load Kolors");
    let (ip_encoder, pairs) = load_kolors_ip_adapter(&ip_snapshot(), Dtype::Float32).unwrap();
    kolors.install_ip_adapter(pairs).unwrap();
    let (h, w, steps, cfg) = (512, 512, 8, 5.0);

    let g = Weights::from_file(GOLDEN).unwrap();
    let reference = golden_image(&g);
    let ip_tokens = ip_encoder.tokens(&reference).unwrap();

    let pos = kolors.encode("a portrait of a person").unwrap();
    let neg = kolors.encode("blurry, low quality").unwrap();
    mlx_rs::random::seed(7).unwrap();
    let init_noise =
        mlx_rs::random::normal::<f32>(&[1, h / 8, w / 8, 4], None, None, None).unwrap();

    let base = kolors
        .denoise_latents(&init_noise, &pos, &neg, steps, cfg, h, w)
        .unwrap();
    let s0 = kolors
        .denoise_ip_latents(&ip_tokens, &init_noise, &pos, &neg, steps, cfg, 0.0, h, w)
        .unwrap();
    let bytes_eq = {
        let n = base.shape().iter().product::<i32>();
        base.reshape(&[n]).unwrap().as_slice::<f32>() == s0.reshape(&[n]).unwrap().as_slice::<f32>()
    };
    let (p0, _) = rel(&s0, &base);
    println!("ip_scale-0 vs base (f32): peak_rel={p0:.3e}");
    assert!(
        bytes_eq,
        "ip_scale=0 (f32) must be byte-identical to plain T2I (IP injection not zero-clean)"
    );
    println!("✓ ip_scale=0 is byte-identical to plain T2I at f32 (decoupled-attn wiring verified)");

    let s_on = kolors
        .denoise_ip_latents(&ip_tokens, &init_noise, &pos, &neg, steps, cfg, 0.7, h, w)
        .unwrap();
    let (pon, mon) = rel(&s_on, &base);
    println!("ip_scale-0.7 vs base (f32): peak_rel={pon:.3e} mean_rel={mon:.3e}");
    assert!(
        mon > 1e-3,
        "ip_scale=0.7 should perturb the latents vs base (mean_rel {mon:.3e} too small)"
    );
    let img = kolors.decode(&s_on).unwrap();
    assert!(
        img.pixels.iter().any(|&p| p > 16) && img.pixels.iter().any(|&p| p < 239),
        "degenerate IP-Adapter render"
    );
    println!("✓ Kolors IP-Adapter (scale>0) perturbs the output and renders coherently");
}
