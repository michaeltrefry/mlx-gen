//! sc-2909: Qwen-Image Lightning end-to-end render over the integrated public path (real weights).
//!
//! The Lightning schedule is bit-exact vs diffusers (`tests/lightning_parity.rs`), the Lightning LoRA
//! loads cleanly (`adapter_real_weights::lightning_loras_apply_cleanly`, 720/720 modules), and the
//! transformer + VAE + denoise loop are the SAME pixel-parity components as the production base path
//! (`e2e_real_weights`, 0.000% px>8 vs the fork — itself a diffusers port). What this gate adds is the
//! **integration** proof: `mlx_gen::load("qwen_image", spec.with_adapters([lightning])).generate(req
//! { sampler: "lightning", steps: 8 })` runs end-to-end and renders a coherent natural image (not
//! flat, not pure noise, all-finite). It writes the Rust render and — if present — the diffusers
//! reference (`tools/dump_qwen_lightning_golden.py render`) as PPMs for a side-by-side visual check.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot + the cached lightx2v Lightning LoRA:
//!   cargo test -p mlx-gen-qwen-image --release --test lightning_render_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec,
    WeightsSource,
};
// Referencing the provider crate links its `inventory` registration so `mlx_gen::load(MODEL_ID, …)`
// resolves (the test otherwise touches only the `mlx_gen` core and the crate would be dropped).
use mlx_gen_qwen_image::{model_edit, MODEL_ID};

const W: u32 = 512;
const H: u32 = 512;
const STEPS: u32 = 8;
const SEED: u64 = 42;
const PROMPT: &str = "a fox sitting in a forest, photorealistic";

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

fn lightning_lora() -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--lightx2v--Qwen-Image-Lightning/snapshots");
    std::fs::read_dir(&snaps)
        .expect("download lightx2v/Qwen-Image-Lightning")
        .filter_map(|e| e.ok())
        .map(|e| {
            e.path()
                .join("Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors")
        })
        .find(|p| p.exists())
        .expect("Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors not cached")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// (mean, std, distinct-level-count, mean horizontal-adjacent-|Δ|) over an RGB8 buffer. A coherent
/// natural image has a broad histogram (std well above flat) AND spatial smoothness (small adjacent
/// Δ); pure noise has a *high* adjacent Δ (~85), a flat fill has std≈0.
fn image_stats(px: &[u8], w: u32) -> (f32, f32, usize, f32) {
    let n = px.len() as f64;
    let mean = px.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = px.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let mut seen = [false; 256];
    for &v in px {
        seen[v as usize] = true;
    }
    let distinct = seen.iter().filter(|&&b| b).count();
    // mean |Δ| between horizontally adjacent pixels (per channel, skipping row wraps).
    let stride = (w * 3) as usize;
    let mut adj_sum = 0f64;
    let mut adj_n = 0u64;
    for (i, &v) in px.iter().enumerate() {
        if i >= 3 && i % stride >= 3 {
            adj_sum += (v as i32 - px[i - 3] as i32).unsigned_abs() as f64;
            adj_n += 1;
        }
    }
    let adj = (adj_sum / adj_n.max(1) as f64) as f32;
    (mean as f32, var.sqrt() as f32, distinct, adj)
}

fn save_ppm(path: &PathBuf, img: &Image) {
    let mut buf = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    buf.extend_from_slice(&img.pixels);
    std::fs::write(path, buf).unwrap();
}

#[test]
#[ignore = "needs real Qwen-Image weights + the cached lightx2v Lightning LoRA"]
fn lightning_render_is_coherent() {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![AdapterSpec::new(
        lightning_lora(),
        1.0,
        AdapterKind::Lora,
    )]);
    let generator = mlx_gen::load(MODEL_ID, &spec).unwrap();
    let req = GenerationRequest {
        prompt: PROMPT.into(),
        width: W,
        height: H,
        seed: Some(SEED),
        steps: Some(STEPS),
        sampler: Some("lightning".into()), // CFG-off static-shift Lightning recipe (sc-2909).
        ..Default::default()
    };
    let img = match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(img.pixels.len(), (W * H * 3) as usize);

    let (mean, std, distinct, adj) = image_stats(&img.pixels, W);
    println!(
        "Rust lightning {STEPS}-step: mean {mean:.1} std {std:.1} distinct {distinct} adj|Δ| {adj:.1}"
    );
    let rust_ppm = golden_dir().join("qwen_lightning_rust_8step.ppm");
    save_ppm(&rust_ppm, &img);
    println!("wrote {}", rust_ppm.display());

    // Coherent natural image: a broad histogram (not a flat fill), many levels, AND spatial structure
    // (small adjacent Δ — rules out a pure-noise render where the schedule/LoRA/CFG path is broken).
    assert!(std > 20.0, "render looks flat/degenerate (std {std:.1})");
    assert!(
        distinct > 64,
        "too few distinct levels ({distinct}) — degenerate render"
    );
    assert!(
        adj < 40.0,
        "render looks like noise (adjacent |Δ| {adj:.1} ~ uniform)"
    );

    // Visual cross-check: if the diffusers Lightning reference was dumped, write it as a PPM too and
    // report its stats (different noise sample, SAME prompt + recipe — both should be coherent foxes).
    let ref_golden = golden_dir().join("qwen_lightning_render_8step.safetensors");
    if let Ok(g) = Weights::from_file(&ref_golden) {
        if let Ok(arr) = g.require("image_u8") {
            let dims = arr.shape();
            let (dh, dw) = (dims[0] as u32, dims[1] as u32);
            let pixels: Vec<u8> = arr.as_slice::<u8>().to_vec();
            let dref = Image {
                width: dw,
                height: dh,
                pixels,
            };
            let (m, s, d, a) = image_stats(&dref.pixels, dw);
            println!("diffusers reference {STEPS}-step: mean {m:.1} std {s:.1} distinct {d} adj|Δ| {a:.1}");
            let ref_ppm = golden_dir().join("qwen_lightning_diffusers_8step.ppm");
            save_ppm(&ref_ppm, &dref);
            println!(
                "wrote {} (visual side-by-side with the Rust render)",
                ref_ppm.display()
            );
        }
    }
}

fn edit_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_EDIT_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache Qwen-Image-Edit-2511 snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn edit_lightning_lora() -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--lightx2v--Qwen-Image-Edit-2511-Lightning/snapshots");
    std::fs::read_dir(&snaps)
        .expect("download lightx2v/Qwen-Image-Edit-2511-Lightning")
        .filter_map(|e| e.ok())
        .map(|e| {
            e.path()
                .join("Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors")
        })
        .find(|p| p.exists())
        .expect("Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors not cached")
}

/// A deterministic synthetic reference image (a smooth RGB gradient — coherent, not noise) to edit.
fn synthetic_reference() -> Image {
    let (w, h) = (512u32, 512u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = ((y + x) % 256) as u8;
            pixels.push(base);
            pixels.push(base.wrapping_mul(2));
            pixels.push(base.wrapping_mul(3));
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// sc-2909: the **Edit** Lightning path end-to-end (`qwen_image_edit` + the lightx2v
/// Qwen-Image-Edit-2511-Lightning LoRA). Mirrors the T2I render gate over the dual-latent edit loop:
/// `model_edit::load(spec.with_adapters([edit_lightning])).generate(req { sampler: "lightning",
/// steps: 8, Reference })` runs end-to-end and renders a coherent natural image (CFG-off). This
/// exercises the Edit-side Lightning wiring (the `is_lightning` branch in `model_edit::generate`),
/// which is otherwise identical to T2I (the dual-latent concat is unchanged from the pixel-parity
/// base edit path).
#[test]
#[ignore = "needs the real Qwen-Image-Edit-2511 model + the cached Edit-2511 Lightning LoRA"]
fn edit_lightning_render_is_coherent() {
    let spec =
        LoadSpec::new(WeightsSource::Dir(edit_snapshot())).with_adapters(vec![AdapterSpec::new(
            edit_lightning_lora(),
            1.0,
            AdapterKind::Lora,
        )]);
    let generator = model_edit::load(&spec).unwrap();
    let req = GenerationRequest {
        prompt: "turn the background into an autumn forest".into(),
        width: W,
        height: H,
        count: 1,
        seed: Some(SEED),
        steps: Some(STEPS),
        sampler: Some("lightning".into()),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_reference(),
            strength: None,
        }],
        ..Default::default()
    };
    let img = match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.swap_remove(0),
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!(img.pixels.len(), (W * H * 3) as usize);

    let (mean, std, distinct, adj) = image_stats(&img.pixels, W);
    println!(
        "Rust edit-lightning {STEPS}-step: mean {mean:.1} std {std:.1} distinct {distinct} adj|Δ| {adj:.1}"
    );
    let ppm = golden_dir().join("qwen_edit_lightning_rust_8step.ppm");
    save_ppm(&ppm, &img);
    println!("wrote {}", ppm.display());

    assert!(
        std > 20.0,
        "edit render looks flat/degenerate (std {std:.1})"
    );
    assert!(
        distinct > 64,
        "too few distinct levels ({distinct}) — degenerate edit render"
    );
    assert!(
        adj < 40.0,
        "edit render looks like noise (adjacent |Δ| {adj:.1} ~ uniform)"
    );
}
