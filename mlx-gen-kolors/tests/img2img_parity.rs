//! Kolors img2img end-to-end parity vs diffusers `KolorsImg2ImgPipeline` (sc-3095).
//!
//! `#[ignore]`d: needs the Kolors snapshot (TE+UNet+VAE) + the materialized `tokenizer.json` + the
//! golden from `tools/dump_kolors_img2img_golden.py`.
//!
//! Structured exactly like `t2i_parity.rs` (see its note): the single U-Net forward matches diffusers
//! to ~5e-4 and the scheduler is bit-identical, but a *full* CFG trajectory cannot be bit-compared to
//! a **torch** reference (the per-step cross-backend f32 floor compounds through the chaotic sampler).
//! So the correctness gate is the **deterministic early-step latent integration** (gate A) — which
//! exercises the whole img2img loop: the VAE-encoded init, `add_noise` (raw `x₀ + noise·σ_start` at
//! the strength-derived `begin_index`), `scale_model_input`, the U-Net, CFG, and the Euler step — and
//! the full render (gate B) is a coherence + cross-backend-delta report.
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test img2img_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{DiffusionSampler, Image};
use mlx_gen_kolors::model::DEFAULT_IMG2IMG_STRENGTH;
use mlx_gen_kolors::sampler::KolorsEulerSampler;
use mlx_gen_kolors::unet::load_unet_kolors_dtype;
use mlx_gen_kolors::Kolors;
use mlx_rs::ops::{add, concatenate_axis, multiply, subtract};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_img2img_golden.safetensors"
);

fn snapshot() -> PathBuf {
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

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let (a, b) = (a.reshape(&[n]).unwrap(), b.reshape(&[n]).unwrap());
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

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json + tools/golden/kolors_img2img_golden.safetensors"]
fn kolors_img2img_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let steps: usize = g.metadata("steps").unwrap().parse().unwrap();
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let h: i32 = g.metadata("h").unwrap().parse().unwrap();
    let w: i32 = g.metadata("w").unwrap().parse().unwrap();
    let init_latents = g.require("init_latents").unwrap();
    let noise = g.require("noise").unwrap();

    // ---- Gate A: deterministic early-step latent integration (the correctness gate). ----
    let sampler = KolorsEulerSampler::kolors_img2img(steps, strength, Dtype::Float32).unwrap();
    let unet = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let cond = concatenate_axis(
        &[
            g.require("pos_context").unwrap(),
            g.require("neg_context").unwrap(),
        ],
        0,
    )
    .unwrap();
    let pooled = concatenate_axis(
        &[
            g.require("pos_pooled").unwrap(),
            g.require("neg_pooled").unwrap(),
        ],
        0,
    )
    .unwrap();
    let mut tid = Vec::new();
    for _ in 0..2 {
        tid.extend_from_slice(&[h as f32, w as f32, 0.0, 0.0, h as f32, w as f32]);
    }
    let time_ids = Array::from_slice(&tid, &[2, 6]);

    // Seed the init exactly as the pipeline: raw add_noise at begin_index (NOT scale_initial_noise).
    let mut x = sampler.add_noise(init_latents, noise).unwrap();
    for i in 0..2usize {
        let x_in = sampler.scale_model_input(&x, i).unwrap();
        let x_unet = concatenate_axis(&[&x_in, &x_in], 0).unwrap();
        let eps = unet
            .forward(&x_unet, sampler.timestep(i), &cond, &pooled, &time_ids)
            .unwrap();
        let row = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0).unwrap();
        let (text, ng) = (row(0), row(1));
        let cfg_eps = add(
            &ng,
            multiply(
                subtract(&text, &ng).unwrap(),
                Array::from_slice(&[cfg], &[1]),
            )
            .unwrap(),
        )
        .unwrap();
        x = sampler.step(&cfg_eps, &x, i).unwrap();
        x.eval().unwrap();
        let key = if i == 0 {
            "step0_latents"
        } else {
            "step1_latents"
        };
        let (p, m) = rel(&x, g.require(key).unwrap());
        println!("gate A step{i}: peak_rel={p:.3e} mean_rel={m:.3e}");
        // step0 ~4e-3, step1 ~1e-2 (single-forward 5e-4 floor × CFG-5, minimal accumulation).
        let (pt, mt) = if i == 0 {
            (1.5e-2, 6e-3)
        } else {
            (3.5e-2, 1.5e-2)
        };
        assert!(
            p < pt,
            "gate A step{i} peak_rel {p:.3e} exceeds {pt:.1e} (img2img loop wiring bug)"
        );
        assert!(
            m < mt,
            "gate A step{i} mean_rel {m:.3e} exceeds {mt:.1e} (img2img loop wiring bug)"
        );
    }
    println!("✓ gate A: img2img early-step latent integration matches diffusers (add_noise+loop+CFG correct)");

    // ---- Gate B: full Rust img2img render (coherence + cross-backend delta report). ----
    let kolors = Kolors::load(&snapshot(), Dtype::Float32).expect("load Kolors");
    let prompt = g.metadata("prompt").unwrap();
    let negative = g.metadata("negative").unwrap();

    // Rebuild the exact init image from the golden (f32 [0,1] [H,W,3] → u8).
    let gi = g.require("init_image").unwrap();
    let n = gi.shape().iter().product::<i32>();
    let px: Vec<u8> = gi
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let init_image = Image {
        width: w as u32,
        height: h as u32,
        pixels: px,
    };

    // The Rust path encodes the init itself (mean), draws its own noise, and runs the full loop. The
    // MLX noise stream differs from torch's, so this is a coherence/brightness check, not a bit gate.
    let _ = DEFAULT_IMG2IMG_STRENGTH; // (the public default; the gate pins strength to the golden's)
    let image = kolors
        .img2img(&init_image, prompt, negative, steps, strength, cfg, 0, h, w)
        .expect("img2img");

    let gimg = g.require("image").unwrap();
    let n = gimg.shape().iter().product::<i32>();
    let want: Vec<u8> = gimg
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    assert_eq!(image.pixels.len(), want.len(), "image size");
    let mean_abs: f64 = image
        .pixels
        .iter()
        .zip(&want)
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as f64)
        .sum::<f64>()
        / want.len() as f64;
    let mean_g: f64 = want.iter().map(|&v| v as f64).sum::<f64>() / want.len() as f64;
    let mean_r: f64 =
        image.pixels.iter().map(|&v| v as f64).sum::<f64>() / image.pixels.len() as f64;
    println!(
        "gate B (full render): mean|Δ|={mean_abs:.2}/255, mean(rust)={mean_r:.1} vs mean(diffusers)={mean_g:.1}"
    );
    assert!(
        image.pixels.iter().any(|&p| p > 16) && image.pixels.iter().any(|&p| p < 239),
        "degenerate render"
    );
    assert!(
        (mean_r - mean_g).abs() < 20.0,
        "render brightness off by {:.1} (gross divergence)",
        (mean_r - mean_g).abs()
    );
    println!("✓ Kolors img2img full pipeline renders coherently (cross-backend px delta is chaos-limited)");
}
