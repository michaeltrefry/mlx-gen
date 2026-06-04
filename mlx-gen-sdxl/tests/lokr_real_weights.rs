//! sc-2640: SDXL LoKr merge — the vendored SDXL path REJECTS LoKr, so Rust is strictly more capable.
//!
//! `#[ignore]`d — needs the real SDXL snapshot, the LCM-LoRA (for the stacking gate), and the goldens
//! from `tools/dump_sdxl_lokr_golden.py` (a synthesized LoKr merged with the validated LyCORIS
//! formula — `reconstruct_lokr_delta`, proven vs the real fork in sc-2602/sc-2528).
//! Run: cargo test -p mlx-gen-sdxl --release --test lokr_real_weights -- --ignored --nocapture
//!
//! Gates: merge count (16 synthesized modules); render parity vs the reference (cross-build floor);
//! scale-0 bit-exact no-op; **stacks with LoRA** (LCM-LoRA 515 + LoKr 16 = 531, render parity).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_sdxl as _;
use mlx_gen_sdxl::{apply_sdxl_adapters, load_unet};

fn golden(name: &str) -> Weights {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../tools/golden")
        .join(name);
    Weights::from_file(&p).unwrap()
}

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

fn spec(g: &Weights, key: &str, scale: f32, kind: AdapterKind) -> AdapterSpec {
    AdapterSpec {
        path: PathBuf::from(g.metadata(key).unwrap_or_else(|| panic!("golden {key}"))),
        scale,
        kind,
    }
}

fn render(spec: &LoadSpec, g: &Weights) -> Image {
    let model = mlx_gen::load("sdxl", spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        negative_prompt: Some(g.metadata("negative").unwrap().to_string()),
        width: g.metadata("w").unwrap().parse().unwrap(),
        height: g.metadata("h").unwrap().parse().unwrap(),
        seed: Some(g.metadata("seed").unwrap().parse().unwrap()),
        steps: Some(g.metadata("steps").unwrap().parse().unwrap()),
        guidance: Some(g.metadata("cfg").unwrap().parse().unwrap()),
        ..Default::default()
    };
    match model.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

fn px8(img: &Image, g: &Weights) -> f32 {
    let gpix: Vec<u8> = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();
    img.pixels
        .iter()
        .zip(&gpix)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count() as f32
        / img.pixels.len() as f32
}

#[test]
#[ignore = "needs the real SDXL snapshot + LoKr golden"]
fn lokr_merge_count() {
    let g = golden("sdxl_lokr_fp16_golden.safetensors");
    let mut unet = load_unet(&snapshot()).unwrap();
    let report =
        apply_sdxl_adapters(&mut unet, &[spec(&g, "lokr_path", 1.0, AdapterKind::Lokr)]).unwrap();
    assert_eq!(
        report.merged, 16,
        "LoKr should merge its 16 synthesized modules"
    );
    assert_eq!(report.skipped_keys, 0);
    println!("✓ LoKr merged {} modules (0 skipped)", report.merged);
}

#[test]
#[ignore = "needs the real SDXL snapshot + LoKr golden"]
fn lokr_render_matches_reference() {
    let g = golden("sdxl_lokr_fp16_golden.safetensors");
    let s = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![spec(
        &g,
        "lokr_path",
        1.0,
        AdapterKind::Lokr,
    )]);
    let img = render(&s, &g);
    let p = px8(&img, &g);
    // Save for inspection.
    let out =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_sdxl_lokr.png");
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!(
        "✓ LoKr render {}x{}: {:.3}% px>8 vs the LyCORIS-formula reference",
        img.width,
        img.height,
        p * 100.0
    );
    // The delta is `kron(w1,w2)` (alpha=rank=scale=1, all single-multiply ops → numpy==mlx bit-exact),
    // so this is the cross-build f32 forward residual (same class as the LoRA gate), not a merge bug.
    assert!(
        p < 0.001,
        "SDXL LoKr render diverged beyond the cross-build residual: {:.3}% px>8",
        p * 100.0
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + LoKr golden"]
fn scale_zero_lokr_is_bit_exact_noop() {
    let g = golden("sdxl_lokr_fp16_golden.safetensors");
    let base = render(&LoadSpec::new(WeightsSource::Dir(snapshot())), &g);
    let zero = render(
        &LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![spec(
            &g,
            "lokr_path",
            0.0,
            AdapterKind::Lokr,
        )]),
        &g,
    );
    let differ = base
        .pixels
        .iter()
        .zip(&zero.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differ, 0,
        "scale-0 LoKr must be a bit-exact no-op ({differ} px differ)"
    );
    println!("✓ scale-0 LoKr is a bit-exact no-op");
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA + stacked golden"]
fn lora_plus_lokr_stacks() {
    let g = golden("sdxl_lokr_stacked_fp16_golden.safetensors");
    // Spec order matches the dump: LCM-LoRA merged first (515), then the LoKr (16).
    let mut unet = load_unet(&snapshot()).unwrap();
    let report = apply_sdxl_adapters(
        &mut unet,
        &[
            spec(&g, "lora_path", 1.0, AdapterKind::Lora),
            spec(&g, "lokr_path", 1.0, AdapterKind::Lokr),
        ],
    )
    .unwrap();
    assert_eq!(
        report.merged,
        515 + 16,
        "stacking should merge LoRA (515) + LoKr (16)"
    );

    // The stacked golden's LoRA is the vendored 515-module merge; `model::load` defaults to the
    // COMPLETE 809 surface (sc-2671), so opt into the vendored surface for an apples-to-apples render.
    // (Run this `#[ignore]` test on its own — it sets a process-global env.)
    std::env::set_var("SDXL_LORA_VENDORED", "1");
    let s = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![
        spec(&g, "lora_path", 1.0, AdapterKind::Lora),
        spec(&g, "lokr_path", 1.0, AdapterKind::Lokr),
    ]);
    let img = render(&s, &g);
    std::env::remove_var("SDXL_LORA_VENDORED");
    let p = px8(&img, &g);
    println!(
        "✓ LoRA+LoKr stacked render {}x{}: {:.3}% px>8 ({} modules merged)",
        img.width,
        img.height,
        p * 100.0,
        report.merged
    );
    assert!(
        p < 0.001,
        "stacked LoRA+LoKr render diverged: {:.3}% px>8",
        p * 100.0
    );
}
