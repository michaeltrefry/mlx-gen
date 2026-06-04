//! sc-2638: SDXL img2img parity vs the vendored `generate_latents_from_image` reference.
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the golden from `tools/dump_sdxl_img2img_golden.py`.
//! Run: cargo test -p mlx-gen-sdxl --release --test img2img_real_weights -- --ignored --nocapture
//!
//! Two gates:
//! - `img2img_components_bit_exact` — the init pipeline (preprocess → VAE-encode mean `x_0`,
//!   `add_noise` → `x_t`, and the step-1 U-Net eps) is **bit-exact** to the reference. This is the
//!   correctness proof.
//! - `img2img_matches_vendored` — the public `generate()` render is the **same generation** as the
//!   reference. At a fractional strength the per-step sigmas are non-round, where pmetal's
//!   source-built MLX 0.31.1 and the golden's wheel MLX 0.31.0 differ by 1 ULP in f32
//!   transcendentals; the chaos-sensitive ancestral sampler amplifies that to fine-detail
//!   divergence (sc-2400 S6 — NOT a code bug; the components above are bit-exact, and strength=1.0,
//!   like 8-step T2I, is pixel-exact).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_sdxl::config::DiffusionConfig;
use mlx_gen_sdxl::{
    encode_conditioning, encode_init_latents, load_text_encoder_1_dtype, load_text_encoder_2_dtype,
    load_tokenizer, load_unet_dtype, load_vae, text_time_ids, EulerSampler,
};
use mlx_rs::{Array, Dtype};

/// Production dtype (sc-2721): U-Net + both CLIP TEs run fp16 (VAE stays f32). The component gate
/// loads the conditioning at fp16 so it matches the `float16=True` golden's f16 conditioning.
const DT: Dtype = Dtype::Float16;
// Force-link the provider so its `inventory::submit!` registers `"sdxl"` (MODEL_ARCHITECTURE.md §4).
use mlx_gen_sdxl as _;

// Production runs fp16 (sc-2721); the render gate (`load("sdxl")`) uses the `float16=True` golden,
// dumped on MLX 0.31.2. The component gate's compute is f32 either way (the VAE is always f32 and
// img2img's encoded init keeps the U-Net/step in f32), so it just reads the golden's f16-rounded
// `timesteps`. Build: FLOAT16=1 <mlx-0.31.2 python> tools/dump_sdxl_img2img_golden.py
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_img2img_fp16_golden.safetensors"
);

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
    let (a, b) = (a.reshape(&[n]).unwrap(), b.reshape(&[n]).unwrap());
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

fn init_image(g: &Weights, w: u32, h: u32) -> Image {
    let pixels: Vec<u8> = g
        .require("init_u8")
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&v| v as u8)
        .collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

#[test]
#[ignore = "needs the real SDXL snapshot + img2img golden"]
fn img2img_components_bit_exact() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();
    let snap = snapshot();
    let image = init_image(&g, w, h);

    let vae = load_vae(&snap).unwrap(); // VAE always f32
    let sampler = EulerSampler::new_with_dtype(&DiffusionConfig::sdxl_base(), true, DT);

    // x_0 = VAE-encode mean of the preprocessed init.
    let x0 = encode_init_latents(&vae, &image, w, h).unwrap();
    let pr_x0 = peak_rel(&x0, g.require("x0_mean").unwrap());

    // x_t = add_noise(x_0, max_time·strength) — the first RNG draw (#0).
    mlx_rs::random::seed(seed).unwrap();
    let xt = sampler
        .add_noise(&x0, sampler.max_time() * strength)
        .unwrap();
    let pr_xt = peak_rel(&xt, g.require("x_t").unwrap());

    // step-1 U-Net eps (CFG) for the golden step-0 input at t = timesteps[1].
    let ts = g.require("timesteps").unwrap().as_slice::<f32>().to_vec();
    let tok = load_tokenizer(&snap).unwrap();
    let tokens = tok
        .tokenize_batch(
            g.metadata("prompt").unwrap(),
            Some(g.metadata("negative").unwrap()),
        )
        .unwrap();
    let (cond, pooled) = encode_conditioning(
        &load_text_encoder_1_dtype(&snap, DT).unwrap(),
        &load_text_encoder_2_dtype(&snap, DT).unwrap(),
        &tokens,
    )
    .unwrap();
    let time_ids = text_time_ids(pooled.shape()[0]);
    let unet = load_unet_dtype(&snap, DT).unwrap();
    let s0 = g.require("step0_latents").unwrap();
    let xu = mlx_rs::ops::concatenate_axis(&[s0, s0], 0).unwrap();
    let eps = unet.forward(&xu, ts[1], &cond, &pooled, &time_ids).unwrap();
    let row = |k: i32| eps.take_axis(Array::from_slice(&[k], &[1]), 0).unwrap();
    let (et, en) = (row(0), row(1));
    let cfg_s = mlx_gen::array::scalar(cfg).as_dtype(et.dtype()).unwrap();
    let eps1 = mlx_rs::ops::add(
        &en,
        mlx_rs::ops::multiply(mlx_rs::ops::subtract(&et, &en).unwrap(), cfg_s).unwrap(),
    )
    .unwrap();
    let pr_eps = peak_rel(&eps1, g.require("eps1_cfg").unwrap());

    println!("x_0 {pr_x0:.3e}  x_t {pr_xt:.3e}  eps1 {pr_eps:.3e}");
    assert!(pr_x0 < 1e-5, "img2img x_0 (encode) diverged: {pr_x0:.3e}");
    assert!(
        pr_xt < 1e-5,
        "img2img x_t (add_noise) diverged: {pr_xt:.3e}"
    );
    assert!(pr_eps < 1e-5, "img2img step-1 eps diverged: {pr_eps:.3e}");
    println!("✓ img2img init pipeline (encode + add_noise + eps) is bit-exact to the reference");
}

#[test]
#[ignore = "needs the real SDXL snapshot + img2img golden"]
fn img2img_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let strength: f32 = g.metadata("strength").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();
    let image = init_image(&g, w, h);

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let model = mlx_gen::load("sdxl", &spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        negative_prompt: Some(g.metadata("negative").unwrap().to_string()),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(cfg),
        conditioning: vec![Conditioning::Reference {
            image,
            strength: Some(strength),
        }],
        ..Default::default()
    };
    let out = model.generate(&req, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };

    let gpix: Vec<u8> = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/golden/rust_sdxl_img2img.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let px8 = img
        .pixels
        .iter()
        .zip(&gpix)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count() as f32
        / img.pixels.len() as f32;
    println!(
        "✓ img2img (strength {strength}) {}x{}: {:.3}% px>8 (pixel-parity); saved {}",
        img.width,
        img.height,
        px8 * 100.0,
        out_path.display()
    );
    // Pixel-parity with the vendored reference (measured 0.00% px>8). Like T2I, this needed every op
    // bit-exact through the chaos-sensitive ancestral sampler — including `σ_up**2` via MLX `power`,
    // not `square` (a 1-ULP op-choice that, at fractional-strength σ, cascaded to ~16% px>8 — sc-2638).
    assert!(
        px8 < 0.001,
        "SDXL img2img lost pixel-parity: {:.3}% px>8",
        px8 * 100.0
    );
}

/// Regression (Codex adversarial review): img2img at strength 0.0 must return a clean image, not a
/// NaN render. With the old min-1-step floor, strength 0 forced a denoise step at σ=0, where the
/// ancestral `σ_up = sqrt(σ_prev²·(σ²−σ_prev²)/σ²)` divides 0/0 → NaN. Faithful behaviour (matching
/// the reference's `int(steps·strength)` = 0 steps) returns the VAE round-trip of the init image.
#[test]
#[ignore = "needs the real SDXL snapshot + img2img golden"]
fn img2img_strength_zero_returns_clean_init() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let cfg: f32 = g.metadata("cfg").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();
    let image = init_image(&g, w, h);

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    let model = mlx_gen::load("sdxl", &spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        negative_prompt: Some(g.metadata("negative").unwrap().to_string()),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        guidance: Some(cfg),
        conditioning: vec![Conditioning::Reference {
            image,
            strength: Some(0.0),
        }],
        ..Default::default()
    };
    let out = model.generate(&req, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    // No NaN/garbage: a clean RGB8 buffer of the right size with real pixel values.
    assert_eq!(img.pixels.len(), (w * h * 3) as usize);
    assert!(
        img.pixels.iter().any(|&p| p != 0),
        "strength-0 img2img produced an all-zero (likely NaN-clamped) image"
    );
    println!(
        "✓ strength-0 img2img returned a clean {}x{} init render",
        img.width, img.height
    );
}
