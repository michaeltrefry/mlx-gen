//! Diagnostic probe (epic 3109 sc-3116 follow-up): does the **stock** SDXL Q8 txt2img produce a
//! coherent image at 1024² vs 512²? InstantID's Q8 e2e collapses to a flat image at 1024² but is fine
//! at 512² (and fp16 is fine at 1024²); since InstantID's UNet *is* the SDXL UNet and the failure is
//! isolated to the quantized UNet, this probe confirms whether the defect is in the base SDXL quant
//! path (not InstantID).
//!
//! `#[ignore]`d — needs the SDXL base snapshot. Run:
//!   cargo test -p mlx-gen-sdxl --release --test q8_1024_probe -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
use mlx_gen_sdxl as _;

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

fn std_dev(img: &Image) -> f32 {
    let n = img.pixels.len() as f64;
    let mean = img.pixels.iter().map(|&p| p as f64).sum::<f64>() / n;
    let var = img
        .pixels
        .iter()
        .map(|&p| (p as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    var.sqrt() as f32
}

fn render(q: Quant, size: u32, steps: u32) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_quant(q);
    let model = mlx_gen::load("sdxl", &spec).unwrap();
    let req = GenerationRequest {
        prompt: "a portrait photo of a man, sharp focus, high detail".to_string(),
        negative_prompt: Some("lowres, blurry".to_string()),
        width: size,
        height: size,
        seed: Some(0),
        steps: Some(steps),
        guidance: Some(5.0),
        ..Default::default()
    };
    match model.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the SDXL base snapshot"]
fn base_sdxl_q8_coherent_at_512_and_1024() {
    let s512 = std_dev(&render(Quant::Q8, 512, 12));
    println!("[sdxl q8 probe] 512²: pixel std = {s512:.1}");
    let s1024 = std_dev(&render(Quant::Q8, 1024, 12));
    println!("[sdxl q8 probe] 1024²: pixel std = {s1024:.1}");
    // A coherent SDXL render has high pixel variance; a collapsed/flat output sits near-constant
    // (std ~10). This asserts the BASE SDXL Q8 path at 1024² — independent of InstantID.
    assert!(
        s1024 > 40.0,
        "base SDXL Q8 collapsed at 1024² (std {s1024:.1}); 512² std {s512:.1} — confirms a base-SDXL \
         quantized-UNet defect at 1024, not InstantID-specific"
    );
}
