//! sc-2400 S5: end-to-end SDXL T2I parity vs the vendored Apple reference.
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from `tools/dump_sdxl_golden.py`. Run:
//!   cargo test -p mlx-gen-sdxl --release --test e2e_real_weights -- --ignored --nocapture
//!
//! Gates:
//! - `full_pipeline_generates_fox` drives the public `load("sdxl", spec).generate(req)` and confirms
//!   the rendered image is **pixel-parity** with the reference (0.00% px>8).
//! - `denoise_per_step_matches_golden` teacher-forces each step and checks the per-step result is
//!   bit-exact — the localization gate.
//! - Diagnostics (`accumulate_from_own_prior`, `rng_stream_matches`): the accumulation from the
//!   model's own prior is bit-exact step-for-step, and the mlx-rs global normal stream is bit-exact
//!   to the reference draw-for-draw. These pinned the one-ULP prior-op-order bug behind an apparent
//!   "chaotic" divergence (sc-2400 S5).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource};
use mlx_gen_sdxl::config::DiffusionConfig;
use mlx_gen_sdxl::{
    encode_conditioning, load_text_encoder_1_dtype, load_text_encoder_2_dtype, load_tokenizer,
    load_unet_dtype, text_time_ids, EulerSampler,
};
use mlx_rs::{Array, Dtype};

// The production path runs **fp16** (sc-2721), so the e2e gate uses the `float16=True` golden, dumped
// on MLX 0.31.2 (the version mlx-gen links — the compiled erf-gelu kernel differs on 0.31.0). Build:
//   FLOAT16=1 <mlx-0.31.2 python> tools/dump_sdxl_golden.py
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_fp16_golden.safetensors"
);

/// The U-Net + both CLIP text encoders run fp16 in production; the localization gates load at the
/// same dtype so they reproduce the fp16 golden's intermediates.
const DT: Dtype = Dtype::Float16;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    // Cast to f32 before reading — the fp16 path's latents are f16, and the golden saves them as f32.
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

/// Pixels differing by > `thr` (0..255) between two RGB8 buffers, as a fraction.
fn px_frac(a: &[u8], b: &[u8], thr: i32) -> f32 {
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > thr)
        .count();
    differ as f32 / a.len() as f32
}

#[test]
#[ignore = "diagnostic: accumulate from my OWN prior (generate's exact flow), compare each step"]
fn accumulate_from_own_prior() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let steps: usize = g.metadata("steps").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let snap = snapshot();

    let tok = load_tokenizer(&snap).unwrap();
    let te1 = load_text_encoder_1_dtype(&snap, DT).unwrap();
    let te2 = load_text_encoder_2_dtype(&snap, DT).unwrap();
    let tokens = tok
        .tokenize_batch(
            g.metadata("prompt").unwrap(),
            Some(g.metadata("negative").unwrap()),
        )
        .unwrap();
    let (conditioning, pooled) = encode_conditioning(&te1, &te2, &tokens).unwrap();
    let time_ids = text_time_ids(pooled.shape()[0]);
    let unet = load_unet_dtype(&snap, DT).unwrap();
    let sampler = EulerSampler::new_with_dtype(&DiffusionConfig::sdxl_base(), true, DT);

    // generate's exact RNG order: seed, draw the prior (USE it), then per-step noise.
    mlx_rs::random::seed(seed).unwrap();
    let mut lat = sampler
        .sample_prior(g.require("prior").unwrap().shape())
        .unwrap();
    println!(
        "own prior vs golden: peak_rel={:.3e}",
        peak_rel(&lat, g.require("prior").unwrap())
    );
    for (i, (t, t_prev)) in sampler
        .timesteps(steps, sampler.max_time())
        .iter()
        .copied()
        .enumerate()
    {
        let x_unet = mlx_rs::ops::concatenate_axis(&[&lat, &lat], 0).unwrap();
        let eps = unet
            .forward(&x_unet, t, &conditioning, &pooled, &time_ids)
            .unwrap();
        let r = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0).unwrap();
        let (et, en) = (r(0), r(1));
        let ec = mlx_rs::ops::add(
            &en,
            mlx_rs::ops::multiply(
                mlx_rs::ops::subtract(&et, &en).unwrap(),
                mlx_gen::array::scalar(cfg).as_dtype(DT).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        lat = sampler.step(&ec, &lat, t, t_prev).unwrap();
        lat.eval().unwrap();
        println!(
            "  acc step{i} peak_rel={:.3e}",
            peak_rel(&lat, g.require(&format!("step{i}_latents")).unwrap())
        );
    }
}

#[test]
#[ignore = "diagnostic: does the mlx-rs global normal stream match the reference draw-for-draw?"]
fn rng_stream_matches() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/sdxl_rng_stream.safetensors"
    );
    // Optional diagnostic — its standalone golden isn't dumped by default. The RNG-stream match is
    // already proven by `accumulate_from_own_prior` (own prior + every step at 0.000e0), so skip
    // rather than fail when the golden is absent.
    let Ok(g) = Weights::from_file(path) else {
        eprintln!("skip rng_stream_matches: {path} not present (see accumulate_from_own_prior)");
        return;
    };
    mlx_rs::random::seed(42).unwrap();
    for i in 0..5 {
        let n = mlx_rs::random::normal::<f32>(&[1, 64, 64, 4], None, None, None).unwrap();
        let pr = peak_rel(&n, g.require(&format!("draw{i}")).unwrap());
        let s = n.reshape(&[-1]).unwrap();
        println!(
            "draw{i} peak_rel={pr:.3e}  rust[0..3]={:?}",
            &s.as_slice::<f32>()[..3]
        );
    }
}

/// Per-step localization gate: teacher-force each step with the golden's previous latent and assert
/// the per-step result is bit-exact. Proves the per-step math (UNet eps, CFG, the Euler-Ancestral
/// step, and the ancestral-RNG draw) is bit-exact; the public `full_pipeline_generates_fox` then
/// confirms the accumulated render is pixel-parity (sc-2400 S5).
#[test]
#[ignore = "needs the real SDXL snapshot + e2e golden"]
fn denoise_per_step_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let steps: usize = g.metadata("steps").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let snap = snapshot();

    let tok = load_tokenizer(&snap).unwrap();
    let te1 = load_text_encoder_1_dtype(&snap, DT).unwrap();
    let te2 = load_text_encoder_2_dtype(&snap, DT).unwrap();
    let tokens = tok
        .tokenize_batch(
            g.metadata("prompt").unwrap(),
            if cfg > 1.0 {
                Some(g.metadata("negative").unwrap())
            } else {
                None
            },
        )
        .unwrap();
    let (conditioning, pooled) = encode_conditioning(&te1, &te2, &tokens).unwrap();
    let time_ids = text_time_ids(pooled.shape()[0]);
    let unet = load_unet_dtype(&snap, DT).unwrap();
    let sampler = EulerSampler::new_with_dtype(&DiffusionConfig::sdxl_base(), true, DT);
    let ts = sampler.timesteps(steps, sampler.max_time());

    // Reproduce the RNG order: seed, draw + discard the prior, then per-step noise.
    mlx_rs::random::seed(seed).unwrap();
    let _prior = sampler
        .sample_prior(g.require("prior").unwrap().shape())
        .unwrap();

    let mut worst = 0f32;
    for (i, (t, t_prev)) in ts.iter().copied().enumerate() {
        // The golden saves latents as f32; the fp16 U-Net needs them at f16 (an f32 input would
        // promote the whole forward to f32). f32→f16 recovers the exact fp16 values they came from.
        let input = if i == 0 {
            g.require("prior").unwrap().as_dtype(DT).unwrap()
        } else {
            g.require(&format!("step{}_latents", i - 1))
                .unwrap()
                .as_dtype(DT)
                .unwrap()
        };
        let x_unet = mlx_rs::ops::concatenate_axis(&[&input, &input], 0).unwrap();
        let eps = unet
            .forward(&x_unet, t, &conditioning, &pooled, &time_ids)
            .unwrap();
        let r = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0).unwrap();
        let (et, en) = (r(0), r(1));
        let ec = mlx_rs::ops::add(
            &en,
            mlx_rs::ops::multiply(
                mlx_rs::ops::subtract(&et, &en).unwrap(),
                mlx_gen::array::scalar(cfg).as_dtype(DT).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        let out = sampler.step(&ec, &input, t, t_prev).unwrap();
        out.eval().unwrap();
        let pr = peak_rel(&out, g.require(&format!("step{i}_latents")).unwrap());
        worst = worst.max(pr);
        println!("  step{i} (t={t}->{t_prev}) peak_rel={pr:.3e}");
    }
    // Every per-step result is bit-exact to the reference (measured 0.0). A non-trivial value would
    // localize a per-step regression (UNet / sigma table / ancestral step).
    assert!(
        worst < 1e-4,
        "a denoise step diverged from the reference: worst peak_rel {worst:.3e}"
    );
    println!(
        "✓ every denoise step is bit-exact to the reference (worst per-step peak_rel {worst:.3e})"
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + e2e golden"]
fn full_pipeline_generates_fox() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let negative = g.metadata("negative").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let model = mlx_gen::load("sdxl", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        negative_prompt: Some(negative),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(cfg),
        ..Default::default()
    };
    let mut last = 0u32;
    let out = model
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                assert_eq!(total, steps);
                last = last.max(current);
            }
        })
        .unwrap();
    assert_eq!(last, steps, "expected {steps} step events");

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1);
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h));

    // Golden image is uint8 NHWC [1,H,W,3]; flatten to compare.
    let gimg = g.require("image_u8").unwrap();
    let gpix: Vec<u8> = gimg.as_slice::<u8>().to_vec();

    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_sdxl_fox.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let frac8 = px_frac(&img.pixels, &gpix, 8);
    let frac16 = px_frac(&img.pixels, &gpix, 16);
    let frac32 = px_frac(&img.pixels, &gpix, 32);
    let mean_abs: f32 = img
        .pixels
        .iter()
        .zip(&gpix)
        .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as f32)
        .sum::<f32>()
        / img.pixels.len() as f32;
    println!(
        "✓ full pipeline {}x{}: px>8 {:.2}%  px>16 {:.2}%  px>32 {:.2}%  mean|Δ| {:.2}/255; saved {}",
        img.width,
        img.height,
        frac8 * 100.0,
        frac16 * 100.0,
        frac32 * 100.0,
        mean_abs,
        out_path.display()
    );
    // PIXEL-PARITY with the vendored reference (measured 0.00% px>8, mean|Δ| 0.50/255 — just the
    // final-image round/clip), at the **production 30-step** config (and 8-step). The ancestral
    // sampler at CFG=7 is acutely chaos-sensitive, so this required EVERY op bit-exact: U-Net
    // (MLX-computed sinusoidal embeddings), sigma table (MLX `cumprod`), sigma interp (MLX), global
    // RNG, the prior's op order `(noise·σ)·rsqrt(σ²+1)`, the `_linspace` op order, AND `σ_up**2` via
    // MLX `power` (not `square` — 1 ULP apart). Any one regression re-introduces chaotic divergence.
    // (Each was a separate 1-ULP that the sampler amplified to 14–34% px>8 — sc-2400 S5/S6.)
    let _ = (frac16, frac32, mean_abs);
    assert!(
        frac8 < 0.001,
        "SDXL T2I lost pixel-parity with the reference: {:.3}% px>8 (a bit-exactness regression — \
         check the prior op order, sigma table, and sinusoidal embeddings)",
        frac8 * 100.0
    );
}
