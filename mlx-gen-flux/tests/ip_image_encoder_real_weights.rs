//! sc-3622: real-weight check for the FLUX IP-Adapter CLIP-ViT-L/14 image encoder.
//!
//! `#[ignore]`d — loads the real `openai/clip-vit-large-patch14` vision tower from the HF cache
//! (or `CLIP_VIT_L_SNAPSHOT`). Run:
//!
//! ```text
//! cargo test -p mlx-gen-flux --release --test ip_image_encoder_real_weights -- --ignored --nocapture
//! ```
//!
//! What this proves:
//!   1. The ViT-L/14 `VisionConfig` + `vision_model.*` / `visual_projection` key contract loads the
//!      real checkpoint (right dims, right names, NCHW→NHWC patch-conv transpose) and the tower runs.
//!   2. The `.image_embeds` head produces the projected pooled CLS token `[1, 768]`, finite and
//!      deterministic.
//!   3. (optional) Numeric parity vs torch `CLIPVisionModelWithProjection`: set `CLIP_VIT_L_REF_EMBEDS`
//!      to a little-endian raw-f32 file of 768 reference embeds for the same fixed gradient input
//!      and the test asserts cosine ≥ 0.999 + max|Δ| within tolerance. The encoder math is the
//!      torch-verified `SvdImageEncoder` path (CLS → post_layernorm → visual_projection); the only
//!      ViT-L delta is config dims + checkpoint keys, both exercised by (1).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_flux::FluxIpImageEncoder;
use mlx_rs::ops::{abs, multiply, sqrt, subtract, sum};
use mlx_rs::Array;

/// Resolve the `openai/clip-vit-large-patch14` `model.safetensors` (env override or HF cache).
fn clip_vit_l_weights() -> Weights {
    if let Ok(p) = std::env::var("CLIP_VIT_L_SNAPSHOT") {
        let p = PathBuf::from(p);
        let file = if p.is_dir() {
            p.join("model.safetensors")
        } else {
            p
        };
        return Weights::from_file(file).unwrap();
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots");
    let dir = std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for openai/clip-vit-large-patch14")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    Weights::from_file(dir.join("model.safetensors")).unwrap()
}

/// A fixed, deterministic RGB gradient — the parity reference input.
fn gradient(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let dot = sum(multiply(a, b).unwrap(), None).unwrap();
    let na = sqrt(sum(multiply(a, a).unwrap(), None).unwrap()).unwrap();
    let nb = sqrt(sum(multiply(b, b).unwrap(), None).unwrap()).unwrap();
    mlx_rs::ops::divide(&dot, multiply(&na, &nb).unwrap())
        .unwrap()
        .item::<f32>()
}

#[test]
#[ignore = "loads openai/clip-vit-large-patch14 vision weights; set CLIP_VIT_L_SNAPSHOT or use HF cache"]
fn ip_image_encoder_runs_on_real_weights() {
    let w = clip_vit_l_weights();
    let enc = FluxIpImageEncoder::from_weights(&w).unwrap();

    let img = gradient(384, 384);
    let embeds = enc.encode(&img).unwrap();
    assert_eq!(embeds.shape(), &[1, 768], "image_embeds must be [1, 768]");

    // Finite + non-degenerate.
    let norm = sqrt(sum(multiply(&embeds, &embeds).unwrap(), None).unwrap())
        .unwrap()
        .item::<f32>();
    assert!(
        norm.is_finite() && norm > 1e-3,
        "embeds must be finite + non-zero (‖·‖={norm})"
    );

    // Deterministic across runs (same weights, same input → byte-identical).
    let embeds2 = enc.encode(&img).unwrap();
    let d = abs(subtract(&embeds, &embeds2).unwrap()).unwrap();
    assert_eq!(
        mlx_rs::ops::max(&d, None).unwrap().item::<f32>(),
        0.0,
        "encode must be deterministic"
    );
    println!("[ip-clip-vit-l] image_embeds [1,768] ‖·‖={norm:.4}");

    // Optional torch golden.
    if let Ok(ref_path) = std::env::var("CLIP_VIT_L_REF_EMBEDS") {
        let bytes = std::fs::read(&ref_path).expect("CLIP_VIT_L_REF_EMBEDS file");
        let refv: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(refv.len(), 768, "reference must be 768 f32");
        let refa = Array::from_slice(&refv, &[1, 768]);
        let cos = cosine(&embeds, &refa);
        let maxd = abs(subtract(&embeds, &refa).unwrap()).unwrap();
        let maxd = mlx_rs::ops::max(&maxd, None).unwrap().item::<f32>();
        println!("[ip-clip-vit-l] vs torch golden: cosine={cos:.6} max|Δ|={maxd:.3e}");
        assert!(cos >= 0.999, "cosine vs torch golden {cos} < 0.999");
        assert!(maxd < 5e-2, "max|Δ| vs torch golden {maxd} too large");
    }
}

fn read_f32_le(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|_| panic!("missing {path}"))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Rigorous numeric parity vs torch `CLIPVisionModelWithProjection.image_embeds`, bypassing
/// preprocessing: both sides consume the **identical** normalized pixel tensor, so this isolates the
/// ViT-L tower + projection head. Generate the reference with the torch venv:
///
/// ```text
/// ~/mlx-flux-venv/bin/python tools/clip_vit_l_parity_ref.py /tmp/clip_l
/// CLIP_VIT_L_PIXELS=/tmp/clip_l.pixels CLIP_VIT_L_REF=/tmp/clip_l.embeds \
///   cargo test -p mlx-gen-flux --release --test ip_image_encoder_real_weights \
///   ip_image_embeds_torch_parity -- --ignored --nocapture
/// ```
#[test]
#[ignore = "torch parity: needs tools/clip_vit_l_parity_ref.py output (CLIP_VIT_L_PIXELS + CLIP_VIT_L_REF)"]
fn ip_image_embeds_torch_parity() {
    let pixels_path = std::env::var("CLIP_VIT_L_PIXELS").expect("set CLIP_VIT_L_PIXELS");
    let ref_path = std::env::var("CLIP_VIT_L_REF").expect("set CLIP_VIT_L_REF");

    let pixels = read_f32_le(&pixels_path);
    assert_eq!(
        pixels.len(),
        224 * 224 * 3,
        "pixels must be NHWC [1,224,224,3]"
    );
    let pixels = Array::from_slice(&pixels, &[1, 224, 224, 3]);

    let enc = FluxIpImageEncoder::from_weights(&clip_vit_l_weights()).unwrap();
    let embeds = enc.image_embeds(&pixels).unwrap();
    assert_eq!(embeds.shape(), &[1, 768]);

    let refv = read_f32_le(&ref_path);
    assert_eq!(refv.len(), 768);
    let refa = Array::from_slice(&refv, &[1, 768]);

    let cos = cosine(&embeds, &refa);
    let maxd = mlx_rs::ops::max(abs(subtract(&embeds, &refa).unwrap()).unwrap(), None)
        .unwrap()
        .item::<f32>();
    println!("[ip-clip-vit-l] torch image_embeds parity: cosine={cos:.6} max|Δ|={maxd:.3e}");
    // cosine is the parity verdict for an embedding (≈0.999984 measured). The residual max|Δ| is f32
    // cross-implementation accumulation (MLX's fused SDPA/LayerNorm vs torch eager) over 24 layers on
    // a ~19-norm vector — not a structural mismatch (the checkpoint + both sides are f32).
    assert!(cos >= 0.9999, "cosine vs torch {cos} < 0.9999");
    assert!(maxd < 5e-2, "max|Δ| vs torch {maxd} too large");
}
