//! sc-2348 slice 4: Qwen-Image T2I end-to-end parity vs the frozen fork.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot (env `QWEN_IMAGE_SNAPSHOT`, else the
//! HF cache) and the local golden from `tools/dump_qwen_image_golden.py` (gitignored). The golden
//! fixes seed 42, 512×512, 4 steps, guidance 4.0, prompt "a fox sitting in a forest,
//! photorealistic", empty negative.
//!
//! Three checks, smallest-footprint first:
//!  - **noise**: seeded RNG parity (`create_noise`) — no weights.
//!  - **transformer + scheduler + CFG + VAE**: load the transformer + VAE via the slice-4 loaders,
//!    feed the golden noise + (bf16) prompt embeds, run the denoise loop, and compare the step-0
//!    latents, final latents, decoded image, and RGB8 pixels (this exercises both on-disk loaders
//!    + the pipeline; feeding dumped noise/embeds removes RNG / text-encoder precision).
//!  - **text encoder**: the on-disk text-encoder loader → prompt-embeds parity (re-validates slice
//!    2 through the slice-4 loader).
//!
//! Run (loads the ~40 GB transformer; the text-encoder check adds ~14 GB):
//!   cargo test -p mlx-gen-qwen-image --release --test e2e_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{CancelFlag, GenerationOutput, GenerationRequest, LoadSpec, Quant, WeightsSource};
use mlx_gen_qwen_image::{
    create_noise, decoded_to_image, denoise_with_progress, loader, qwen_scheduler, unpack_latents,
    FlowMatchSampler,
};
use mlx_rs::Array;

const SEED: u64 = 42;
const STEPS: usize = 4;
const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const GUIDANCE: f32 = 4.0;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_golden.safetensors"
);
const Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_q8_golden.safetensors"
);
const Q4_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_q4_golden.safetensors"
);

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// (peak-rel `max|a-b|/max|b|`, mean-rel `mean|a-b|/mean|b|`).
fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

#[test]
#[ignore = "needs the local image golden"]
fn create_noise_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let want = g.require("noise").unwrap();
    let got = create_noise(SEED, WIDTH, HEIGHT).unwrap();
    assert_eq!(got.shape(), want.shape(), "noise shape");
    let (peak, mean) = rel_errors(&got, want);
    println!("noise: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    assert!(
        peak < 1e-4,
        "seeded-RNG noise diverged: peak-rel {peak:.3e}"
    );
}

#[test]
#[ignore = "needs real Qwen-Image transformer+VAE weights + local golden"]
fn transformer_pipeline_vae_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let root = snapshot();
    let transformer = loader::load_transformer(&root).unwrap();
    let vae = loader::load_vae(&root).unwrap();

    // Feed the golden noise (f32) + golden bf16 prompt embeds through the denoise loop.
    let noise = g.require("noise").unwrap().clone();
    let pos = g.require("prompt_embeds").unwrap();
    let neg = g.require("negative_prompt_embeds").unwrap();
    let sampler = FlowMatchSampler::new(qwen_scheduler(STEPS, WIDTH, HEIGHT));
    let latents = denoise_with_progress(
        &transformer,
        &sampler,
        noise,
        pos,
        Some(neg),
        GUIDANCE,
        WIDTH,
        HEIGHT,
        0, // txt2img: denoise every step
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();

    let want_latents = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), want_latents.shape(), "final_latents shape");
    let (peak, mean) = rel_errors(&latents, want_latents);
    println!("final_latents: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    // bf16-class e2e tolerance (Metal f32 matmul runs in reduced precision; error accumulates over
    // 4 steps × 60 layers × 2 forwards).
    assert!(mean < 2e-2, "final_latents mean-rel regressed: {mean:.3e}");
    assert!(peak < 1e-1, "final_latents peak-rel regressed: {peak:.3e}");

    // VAE decode of the (Rust-computed) latents vs the fork's decoded image.
    let unpacked = unpack_latents(&latents, WIDTH, HEIGHT).unwrap();
    let decoded = vae.decode(&unpacked).unwrap();
    let want_decoded = g.require("decoded").unwrap();
    let (dpeak, dmean) = rel_errors(&decoded, want_decoded);
    println!("decoded: peak-rel = {dpeak:.3e}, mean-rel = {dmean:.3e}");
    assert!(dmean < 5e-2, "decoded mean-rel regressed: {dmean:.3e}");

    // RGB8 pixel agreement: share of pixels differing by > 8 levels (z-image-style gate).
    let got_img = decoded_to_image(&decoded).unwrap();
    let want_img = decoded_to_image(want_decoded).unwrap();
    assert_eq!(got_img.pixels.len(), want_img.pixels.len());
    let differ = got_img
        .pixels
        .iter()
        .zip(&want_img.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / got_img.pixels.len() as f32;
    println!("pixels >8 apart: {:.3}%", frac * 100.0);
    assert!(
        frac < 0.05,
        "too many divergent pixels: {:.3}%",
        frac * 100.0
    );
}

/// Transformer-isolation **diagnostic** (NOT the headline parity gate): Q-quantize the Rust
/// transformer (group_size 64 — the exact `nn.Linear` set the fork's `nn.quantize` hits), feed the
/// fork's seeded noise + its **dense** bf16 `prompt_embeds`, run the denoise loop, and confirm the
/// latents/decode match the fork's `quantize=N` golden.
///
/// This feeds the reference's own `prompt_embeds`, so it deliberately isolates the transformer from
/// the text encoder. For Qwen that is faithful, not a blind spot: the fork's `QwenWeightDefinition`
/// marks the `text_encoder` component `skip_quantization=True` ("Quantization causes significant
/// semantic degradation") and the VAE is all-conv (no `to_quantized` leaves), so the fork's
/// `quantize=N` is **transformer-only** — exactly what the Rust port does. The honest end-to-end
/// gate is [`q_full_generate_renders`] (the public `load(spec.with_quant(Q)).generate()` path,
/// which runs the dense text encoder + VAE just like the fork). Unlike Z-Image (sc-2532), whose fork
/// *does* quantize the TE+VAE, there is no scope gap here for a full-path gate to expose.
fn q_pipeline_matches_fork(golden_path: &str, bits: i32, max_latent_mean: f32, max_px_frac: f32) {
    let g = Weights::from_file(golden_path).unwrap();
    let stored: i32 = g.metadata("quantize").unwrap().parse().unwrap();
    assert_eq!(stored, bits, "golden was dumped at a different bit-width");
    let root = snapshot();
    let mut transformer = loader::load_transformer(&root).unwrap();
    transformer.quantize(bits).unwrap();
    let vae = loader::load_vae(&root).unwrap();

    let noise = g.require("noise").unwrap().clone();
    let pos = g.require("prompt_embeds").unwrap();
    let neg = g.require("negative_prompt_embeds").unwrap();
    let sampler = FlowMatchSampler::new(qwen_scheduler(STEPS, WIDTH, HEIGHT));
    let latents = denoise_with_progress(
        &transformer,
        &sampler,
        noise,
        pos,
        Some(neg),
        GUIDANCE,
        WIDTH,
        HEIGHT,
        0, // txt2img: denoise every step
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();

    let want_latents = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), want_latents.shape(), "final_latents shape");
    let (peak, mean) = rel_errors(&latents, want_latents);
    println!("Q{bits} final_latents: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    // Q vs fork-Q: packing is byte-identical (sc-2342) so this stays near the qmm-parity floor,
    // a touch looser than the bf16 path to absorb quantized-matmul accumulation over the loop.
    assert!(
        mean < max_latent_mean,
        "Q{bits} final_latents mean-rel regressed: {mean:.3e}"
    );

    let unpacked = unpack_latents(&latents, WIDTH, HEIGHT).unwrap();
    let decoded = vae.decode(&unpacked).unwrap();
    let want_decoded = g.require("decoded").unwrap();
    let (_dpeak, dmean) = rel_errors(&decoded, want_decoded);
    println!("Q{bits} decoded: mean-rel = {dmean:.3e}");

    let got_img = decoded_to_image(&decoded).unwrap();
    let want_img = decoded_to_image(want_decoded).unwrap();
    let differ = got_img
        .pixels
        .iter()
        .zip(&want_img.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / got_img.pixels.len() as f32;
    println!("Q{bits} pixels >8 apart: {:.3}%", frac * 100.0);
    assert!(
        frac < max_px_frac,
        "Q{bits}: too many divergent pixels: {:.3}%",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real Qwen-Image transformer+VAE weights + local Q8 golden"]
fn transformer_q8_pipeline_matches_fork() {
    // Measured (512², fox, seed 42): latent mean-rel 5.0e-3, px>8 0.007%.
    q_pipeline_matches_fork(Q8_GOLDEN, 8, 3e-2, 0.02);
}

#[test]
#[ignore = "needs real Qwen-Image transformer+VAE weights + local Q4 golden (QUANTIZE=4 dump_qwen_image_golden.py)"]
fn transformer_q4_pipeline_matches_fork() {
    // Measured (512², fox, seed 42): latent mean-rel 8.3e-3, px>8 0.006%.
    q_pipeline_matches_fork(Q4_GOLDEN, 4, 3e-2, 0.02);
}

/// sc-2565: the **headline** parity gate — the full public product path
/// `mlx_gen::load("qwen_image", spec.with_quant(Q)).generate()`, end-to-end at the golden's
/// (prompt, seed, size, steps, guidance), vs the fork's `quantize=N` render. Unlike the cap-fed
/// diagnostic above, this runs the Rust tokenizer + **dense** text encoder + transformer (Q) + VAE
/// — the same component set the fork quantizes (`text_encoder` is `skip_quantization=True`, the VAE
/// is all-conv). Proves there is no end-to-end scope gap, including at Q4 (the lossiest setting,
/// which is where a text-encoder descope would surface — as it did for Z-Image, ~18% px>8).
fn q_full_generate_renders(golden_path: &str, quant: Quant, bits: i32, max_px_frac: f32) {
    let g = Weights::from_file(golden_path).unwrap();
    let stored: i32 = g.metadata("quantize").unwrap().parse().unwrap();
    assert_eq!(stored, bits, "golden was dumped at a different bit-width");
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let width: u32 = g.metadata("width").unwrap().parse().unwrap();
    let height: u32 = g.metadata("height").unwrap().parse().unwrap();

    // Full pipeline through the public registry API. The dump's NEGATIVE="" maps to the fork's
    // single-space fallback, which `generate()` reproduces for `negative_prompt: None`.
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_quant(quant);
    let generator = mlx_gen::load("qwen_image", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width,
        height,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (width, height), "image size");

    let want_img = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    assert_eq!(img.pixels.len(), want_img.pixels.len());
    let differ = img
        .pixels
        .iter()
        .zip(&want_img.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "Q{bits} full generate vs fork-Q{bits}: {:.3}% px>8 ({}x{})",
        frac * 100.0,
        img.width,
        img.height
    );
    assert!(
        frac < max_px_frac,
        "Q{bits} full generate diverged: {:.3}%",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real Qwen-Image snapshot + local Q8 golden (QUANTIZE=8 dump_qwen_image_golden.py)"]
fn q8_full_generate_renders() {
    // Measured (512², fox, seed 42): 0.023% px>8 vs fork-Q8.
    q_full_generate_renders(Q8_GOLDEN, Quant::Q8, 8, 0.02);
}

#[test]
#[ignore = "needs real Qwen-Image snapshot + local Q4 golden (QUANTIZE=4 dump_qwen_image_golden.py)"]
fn q4_full_generate_renders() {
    // Measured (512², fox, seed 42): 0.072% px>8 vs fork-Q4 — NOT the ~18% Z-Image showed
    // transformer-only, because Qwen's fork runs the text encoder dense too (sc-2565).
    q_full_generate_renders(Q4_GOLDEN, Quant::Q4, 4, 0.02);
}

#[test]
#[ignore = "needs real Qwen-Image text-encoder weights + local golden"]
fn text_encoder_path_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let enc = loader::load_text_encoder(&snapshot()).unwrap();
    let out = enc
        .encode(
            g.require("input_ids_pos").unwrap(),
            g.require("attention_mask_pos").unwrap(),
        )
        .unwrap();
    // The golden embeds are bf16 (the fork casts them); compare in f32.
    let want = g
        .require("prompt_embeds")
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap();
    assert_eq!(out.shape(), want.shape(), "prompt_embeds shape");
    let (peak, mean) = rel_errors(&out, &want);
    println!("text-encoder prompt_embeds: peak-rel = {peak:.3e}, mean-rel = {mean:.3e}");
    // The Rust encoder runs f32; the fork casts these embeds to bf16, so the gap here is the bf16
    // quantization floor (~1.5e-2), not a math error — the f32 encoder parity is slice 2's 6e-4
    // gate (`text_encoder_real_weights.rs`). This check just confirms the slice-4 loader yields a
    // working encoder.
    assert!(mean < 2e-2, "prompt_embeds mean-rel regressed: {mean:.3e}");
}
