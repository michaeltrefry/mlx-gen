//! sc-3707 — load a real converted SAM2 checkpoint (from `tools/convert_sam2_to_mlx.py`, or the
//! `SceneWorks/sam2-mlx` mirror) and run the full box→mask forward. This is the production
//! weight-load path: a standalone `.safetensors` on disk (not the parity golden's bundle).
//!
//! Run (point at a converted checkpoint):
//!   SCENEWORKS_SAM2_WEIGHTS=/path/to/sam2.1_hiera_large.safetensors \
//!     cargo test -p mlx-gen-sam2 --release --test weights_load -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ImageEncoderConfig, Sam2Segmenter};
use mlx_rs::Array;

#[test]
#[ignore = "needs SCENEWORKS_SAM2_WEIGHTS=<converted large .safetensors>"]
fn loads_converted_checkpoint_and_segments() {
    let path = std::env::var("SCENEWORKS_SAM2_WEIGHTS").unwrap_or_else(|_| {
        panic!("set SCENEWORKS_SAM2_WEIGHTS to a converted large checkpoint (tools/convert_sam2_to_mlx.py)")
    });
    let w = Weights::from_file(&path).expect("load converted checkpoint");
    let seg = Sam2Segmenter::from_weights(&w, &Sam2ImageEncoderConfig::large())
        .expect("build segmenter from converted weights");

    // A flat (zero) preprocessed image + a centered box: proves the full encode→prompt→decode→
    // upsample→threshold path runs on real weights and yields a well-formed binary L mask.
    let pixels = Array::from_slice(&vec![0f32; 3 * 1024 * 1024], &[1, 3, 1024, 1024]);
    let mask = seg
        .segment_from_pixels(&pixels, [200.0, 150.0, 820.0, 870.0], 1024, 1024)
        .expect("segment");

    assert_eq!(mask.shape(), &[1024, 1024]);
    let vals = mask
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert!(
        vals.iter().all(|&v| v == 0.0 || v == 255.0),
        "mask must be binary 0/255"
    );
    let fg = vals.iter().filter(|&&v| v == 255.0).count();
    println!(
        "converted-checkpoint segment OK: {fg} foreground px of {}",
        vals.len()
    );
}
