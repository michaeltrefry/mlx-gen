//! sc-3115: InstantID T2I end-to-end + ArcFace-cosine identity preservation.
//!
//! `#[ignore]`d — needs the SDXL base snapshot, the InstantID `ControlNetModel`, the converted
//! `tools/golden/instantid/ip-adapter.safetensors` (`tools/convert_instantid.py`), the face-stack
//! weights (`tools/convert_scrfd.py` + `tools/convert_glintr100.py`), and the reference image
//! (`tools/dump_instantid_e2e_ref.py`). Fully self-contained in Rust at test time — no torch.
//!
//! Run (tune size/steps via env for a quick smoke):
//!   INSTANTID_SIZE=512 INSTANTID_STEPS=4 cargo test -p mlx-gen-instantid --release \
//!     --test instantid_e2e -- --ignored --nocapture
//!   cargo test -p mlx-gen-instantid --release --test instantid_e2e -- --ignored --nocapture
//!
//! The gate is **directional** (per epic 3109: ArcFace-cosine + coherence, NOT bit-exact): a correctly
//! wired pipeline preserves identity (cosine well above 0), a broken one collapses to ~0.

use std::path::PathBuf;

use mlx_gen::media::Image;
use mlx_gen::weights::Weights;
use mlx_gen::WeightsSource;
use mlx_gen_instantid::{letterbox, InstantId, InstantIdPaths, InstantIdRequest};

fn golden_path(name: &str) -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../tools/golden")).join(name)
}

fn golden(name: &str) -> Weights {
    let p = golden_path(name);
    Weights::from_file(&p).unwrap_or_else(|e| panic!("missing golden {p:?}: {e}"))
}

fn sdxl_base() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("SDXL base snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn instantid_controlnet() -> WeightsSource {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--InstantX--InstantID/snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("InstantID snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    WeightsSource::Dir(snap.join("ControlNetModel"))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Cosine similarity of two (un-normalized) embeddings.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    (dot / (na * nb + 1e-12)) as f32
}

fn save_png(name: &str, img: &Image) {
    let path = golden_path(name);
    let buf = image::RgbImage::from_raw(img.width, img.height, img.pixels.clone()).unwrap();
    buf.save(&path).unwrap();
    println!("  wrote {path:?}");
}

/// Load (optionally quantized) → detect reference face → generate → re-detect → ArcFace-cosine.
/// `quant_bits`: None = fp16, Some(8)/Some(4) = Q8/Q4 (sc-3116). Returns the identity cosine.
fn run_identity(quant_bits: Option<i32>, size_override: Option<u32>, out_png: &str) -> f32 {
    let size = size_override.unwrap_or_else(|| env_usize("INSTANTID_SIZE", 1024) as u32);
    let steps = env_usize("INSTANTID_STEPS", 30);
    let label = quant_bits
        .map(|b| format!("Q{b}"))
        .unwrap_or_else(|| "fp16".into());

    let paths = InstantIdPaths {
        sdxl_base: sdxl_base(),
        identitynet: instantid_controlnet(),
        ip_adapter: golden_path("instantid/ip-adapter.safetensors"),
    };
    let scrfd = golden("scrfd_10g.safetensors");
    let arcface = golden("arcface_iresnet100.safetensors");
    let mut model = InstantId::load(&paths).expect("load InstantID");
    if let Some(bits) = quant_bits {
        model = model.quantize(bits).expect("quantize");
    }
    let model = model
        .with_face(&scrfd, &arcface)
        .expect("attach face stack");

    // Reference face.
    let g = golden("instantid_e2e_ref.safetensors");
    let wh = g.require("ref_wh").unwrap().as_slice::<i32>().to_vec();
    let (rw, rh) = (wh[0] as u32, wh[1] as u32);
    let ref_img = Image {
        width: rw,
        height: rh,
        pixels: g.require("ref_img").unwrap().as_slice::<u8>().to_vec(),
    };

    // Letterbox to the output size + detect the reference face (its embedding drives the IP path).
    let canvas = letterbox(&ref_img, size, size);
    let ref_face = model
        .largest_face(&canvas.pixels, size as usize, size as usize)
        .expect("detect reference face");
    let kps: Vec<(f32, f32)> = ref_face.kps.iter().map(|p| (p[0], p[1])).collect();
    println!(
        "[instantid {label}] ref face det_score={:.3} kps[0]=({:.1},{:.1})",
        ref_face.det_score, kps[0].0, kps[0].1
    );

    // Generate.
    let req = InstantIdRequest {
        prompt: "film still, a portrait photo of a man, cinematic lighting, sharp focus, \
                 high detail, looking at the camera"
            .into(),
        negative: "lowres, blurry, deformed, disfigured, cartoon, painting".into(),
        width: size,
        height: size,
        steps,
        guidance: 5.0,
        ip_adapter_scale: 0.8,
        controlnet_scale: 0.8,
        seed: 0,
    };
    let out = model
        .generate_with(&req, &ref_face.embedding, &kps)
        .expect("generate");
    assert_eq!((out.width, out.height), (size, size), "output dims");
    // Not a degenerate (all-zero / NaN→0) image.
    let nonzero = out.pixels.iter().filter(|&&p| p != 0).count();
    assert!(
        nonzero > out.pixels.len() / 100,
        "output looks degenerate ({nonzero} nonzero bytes)"
    );
    save_png(out_png, &out);

    // Re-detect the generated face and measure identity preservation.
    let out_face = model
        .largest_face(&out.pixels, size as usize, size as usize)
        .expect("detect generated face");
    let cos = cosine(&ref_face.embedding, &out_face.embedding);
    println!(
        "[instantid {label}] {size}x{size} steps={steps} | generated face det_score={:.3} | \
         ArcFace-cosine(ref, generated) = {cos:.4}",
        out_face.det_score
    );
    cos
}

// Directional gate (epic 3109: identity + coherence, NOT bit-exact). fp16 measures **0.8214** at the
// default 1024²/30-step settings — essentially the sc-2009 torch baseline (≈0.876). A broken pipeline
// (wrong token wiring / no IP / no IdentityNet) collapses toward 0 (the 4-step smoke sits at ~0.21).

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_t2i_preserves_identity() {
    let cos = run_identity(None, None, "instantid_e2e_out.png");
    assert!(
        cos > 0.6,
        "fp16 identity not preserved: ArcFace-cosine {cos:.4} (expected ≳0.8 at 1024²/30 steps)"
    );
}

// sc-3116 quant tests run at **512²**, not 1024²: the stock SDXL **quantized UNet collapses to a flat
// image at 1024²** (a pre-existing base-SDXL-quant defect — `mlx-gen-sdxl tests/q8_1024_probe.rs`
// reproduces it on plain SDXL Q8 txt2img, independent of InstantID; tracked separately). At 512² the
// full InstantID quant stack (UNet + IP K/V + CLIP TEs + IdentityNet) is healthy and preserves
// identity. The IdentityNet + TE quant are fine at 1024² (only the base UNet quant is affected).

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_t2i_q8_preserves_identity() {
    let cos = run_identity(Some(8), Some(512), "instantid_e2e_q8_out.png");
    assert!(
        cos > 0.5,
        "Q8 identity not preserved: ArcFace-cosine {cos:.4}"
    );
}

#[test]
#[ignore = "needs SDXL base + InstantID + converted ip-adapter + face goldens + reference"]
fn instantid_t2i_q4_preserves_identity() {
    // Q4 is more aggressive; identity should still be clearly preserved (looser floor).
    let cos = run_identity(Some(4), Some(512), "instantid_e2e_q4_out.png");
    assert!(
        cos > 0.45,
        "Q4 identity not preserved: ArcFace-cosine {cos:.4}"
    );
}
