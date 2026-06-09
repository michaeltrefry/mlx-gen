//! sc-3706 — SAM2 box-prompt segmenter (prompt encoder + two-way mask decoder) parity vs the
//! MLX-native reference (`avbiswas/sam2-mlx` `Sam2ImageSegmenter`, the impl this crate ports).
//!
//! Golden: `tools/dump_sam2_segmenter_golden.py` (reference encode→box-prompt→decode on a fixed
//! input + box). Both run MLX Metal, so parity is near-bit. Validates the whole box→mask path:
//! the encoder, `conv_s0/s1` high-res projection, prompt encoder, two-way transformer, mask
//! upscaling/hypernetwork, and the argmax-IoU mask selection.
//!
//! Run:
//!   PYTHONPATH=/tmp/sam2-mlx/src ~/mlx-flux-venv/bin/python tools/dump_sam2_segmenter_golden.py --size large
//!   cargo test -p mlx-gen-sam2 --release --test segmenter_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam2::{Sam2ImageEncoderConfig, Sam2Segmenter};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sam2_segmenter_golden_large.safetensors"
);

fn golden() -> Weights {
    Weights::from_file(GOLDEN).unwrap_or_else(|e| {
        panic!("missing {GOLDEN}: {e}\nRun tools/dump_sam2_segmenter_golden.py --size large first.")
    })
}

fn flat(a: &Array) -> Vec<f32> {
    let n: i32 = a.shape().iter().product();
    a.reshape(&[n])
        .unwrap()
        .as_dtype(mlx_rs::Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn metrics(got: &Array, want: &Array) -> (f32, f32, f32) {
    let a = flat(got);
    let b = flat(want);
    assert_eq!(
        a.len(),
        b.len(),
        "shape {:?} vs {:?}",
        got.shape(),
        want.shape()
    );
    let peak_ref = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_ref: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_diff: f64 = a.iter().zip(&b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(&b) {
        dot += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    let cos = (dot / (na.sqrt() * nb.sqrt())) as f32;
    (cos, max_diff / peak_ref, (sum_diff / sum_ref) as f32)
}

#[test]
#[ignore = "needs local golden from tools/dump_sam2_segmenter_golden.py --size large"]
fn segmenter_box_prompt_matches_mlx_reference_large() {
    let g = golden();
    let seg = Sam2Segmenter::from_weights(&g, &Sam2ImageEncoderConfig::large()).unwrap();

    let enc_in = g.require("enc_in").unwrap();
    let bx = flat(g.require("box_1024").unwrap());
    let box_1024 = [bx[0], bx[1], bx[2], bx[3]];

    let (best_mask, iou) = seg.best_low_res_mask(enc_in, box_1024).unwrap();
    assert_eq!(best_mask.shape(), &[256, 256]);

    // Best low-res mask logits vs the reference's argmax-selected mask.
    let want_best = g.require("ref_low_res_best").unwrap();
    let (cos, peak, mean) = metrics(&best_mask, want_best);
    println!("best low-res mask: cos {cos:.7} peak-rel {peak:.3e} mean-rel {mean:.3e}");
    assert!(cos > 0.999, "best mask cosine {cos:.7}");
    assert!(mean < 5e-3, "best mask mean-rel {mean:.3e}");

    // Selected IoU matches the reference's best IoU score.
    let ref_ious = flat(g.require("ref_ious").unwrap());
    let ref_best = flat(g.require("ref_best_idx").unwrap())[0] as usize;
    println!("selected iou {iou:.5} vs ref ious {ref_ious:?} (ref best idx {ref_best})");
    assert!(
        (iou - ref_ious[ref_best]).abs() < 1e-2,
        "selected iou {iou:.5} vs ref {:.5}",
        ref_ious[ref_best]
    );

    // The full post-process (upsample + threshold) agrees with the reference best mask thresholded.
    let mask = seg.segment_from_pixels(enc_in, box_1024, 256, 256).unwrap();
    assert_eq!(mask.shape(), &[256, 256]);
    let got_bin: Vec<u8> = flat(&mask).iter().map(|&v| (v > 127.0) as u8).collect();
    let ref_bin: Vec<u8> = flat(want_best).iter().map(|&v| (v > 0.0) as u8).collect();
    let inter: usize = got_bin
        .iter()
        .zip(&ref_bin)
        .filter(|(a, b)| **a == 1 && **b == 1)
        .count();
    let union: usize = got_bin
        .iter()
        .zip(&ref_bin)
        .filter(|(a, b)| **a == 1 || **b == 1)
        .count();
    let iou_bin = if union == 0 {
        1.0
    } else {
        inter as f32 / union as f32
    };
    println!(
        "binary mask IoU vs reference: {iou_bin:.5} (foreground px: got {}, ref {})",
        got_bin.iter().filter(|&&v| v == 1).count(),
        ref_bin.iter().filter(|&&v| v == 1).count()
    );
    assert!(iou_bin > 0.99, "binary mask IoU {iou_bin:.5}");
}
