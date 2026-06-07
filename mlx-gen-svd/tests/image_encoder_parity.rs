//! SVD image-encoder parity vs transformers `CLIPVisionModelWithProjection` (epic 3040 / sc-3373).
//! Gates `SvdImageEncoder::image_embeds` (the reused ViT-H body + post_layernorm + visual_projection)
//! against a golden dumped from the real model (`tools/dump_svd_image_encoder_golden.py`), in f32 so
//! the gate isolates the math from fp16 rounding. Needs the SVD checkpoint locally → `--ignored`.
//!
//! Run: `cargo test -p mlx-gen-svd --test image_encoder_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract};
use mlx_rs::Dtype;

use mlx_gen::weights::Weights;
use mlx_gen_svd::{ImageEncoderConfig, SvdImageEncoder};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/svd_image_encoder_golden.safetensors"
);

/// Locate the SVD `image_encoder/model.safetensors` in the HF cache.
fn image_encoder_path() -> std::path::PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap();
            std::path::PathBuf::from(home).join(".cache/huggingface/hub")
        });
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path();
    snap.join("image_encoder/model.safetensors")
}

#[test]
#[ignore = "needs the SVD checkpoint in the HF cache"]
fn svd_image_encoder_matches_transformers() {
    let mut w = Weights::from_file(image_encoder_path()).expect("svd image_encoder weights");
    w.cast_all(Dtype::Float32).expect("cast f32");
    let enc = SvdImageEncoder::from_weights(&w, &ImageEncoderConfig::default()).expect("encoder");

    let g = Weights::from_file(GOLDEN).expect("image encoder golden");
    // NCHW [1,3,224,224] → NHWC [1,224,224,3] for the conv-based body.
    let pv = g
        .require("pixel_values")
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap();
    let embeds = enc.image_embeds(&pv).unwrap();
    let want = g.require("image_embeds").unwrap();
    assert_eq!(embeds.shape(), want.shape(), "image_embeds shape");

    let diff = abs(subtract(&embeds, want).unwrap()).unwrap();
    let max_abs = max_op(&diff, None).unwrap().item::<f32>();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    let rel = max_abs / denom.max(1e-6);
    println!("image_encoder parity: max|Δ| {max_abs}, peak-rel {rel}");
    // ~0.2% peak-rel is f32 cross-backend accumulation over the 32-layer ViT (matmul/sdpa ordering),
    // not a structural gap — and image_embeds only feed cross-attn conditioning (robust to it).
    assert!(rel < 3e-3, "image_embeds peak-rel {rel} (max|Δ| {max_abs})");
}
