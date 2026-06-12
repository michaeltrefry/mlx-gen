//! SAM3-E geometry/exemplar-encoder (box-prompt PVS path) parity (sc-4923).
//!
//! Two gates from `scripts/spikes/sam3_oracle/dump_geometry_fixture.py`:
//!   1. `geometry_encoder_matches_oracle` — feed the geometry encoder the exact inputs the reference
//!      module received (boxes, labels, the 72² FPN feature + its sine pos embed) and check its
//!      output prompt tokens against the torch oracle. Isolates the new `roi_align` + encoder.
//!   2. `pvs_box_prompt_matches_oracle` — run the full box-prompted segmenter end-to-end and check
//!      the post-processed instance masks against the oracle.
//!
//! Run:
//!   SAM3_WEIGHTS=.../model.safetensors \
//!   SAM3_GEOMETRY_FIXTURE=scripts/spikes/sam3_oracle/geometry_fixture.safetensors \
//!   SAM3_GEOMETRY_E2E_FIXTURE=scripts/spikes/sam3_oracle/geometry_e2e_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test geometry_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::{Sam3GeometryConfig, Sam3GeometryEncoder, Sam3ImageSegmenter};
use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::Array;

fn f32_of(a: &Array) -> f32 {
    a.as_dtype(mlx_rs::Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let dot = f32_of(&sum(multiply(a, b).unwrap(), None).unwrap());
    let na = f32_of(&sum(multiply(a, a).unwrap(), None).unwrap()).sqrt();
    let nb = f32_of(&sum(multiply(b, b).unwrap(), None).unwrap()).sqrt();
    dot / (na * nb)
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    f32_of(&max(abs(subtract(a, b).unwrap()).unwrap(), None).unwrap())
}

/// IoU of two binary `[h, w]` masks (uint8 0/1).
fn iou(a: &Array, b: &Array) -> f32 {
    let af = a.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let bf = b.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let inter = f32_of(&sum(multiply(&af, &bf).unwrap(), None).unwrap());
    let sa = f32_of(&sum(&af, None).unwrap());
    let sb = f32_of(&sum(&bf, None).unwrap());
    let union = sa + sb - inter;
    if union <= 0.0 {
        1.0
    } else {
        inter / union
    }
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_GEOMETRY_FIXTURE"]
fn geometry_encoder_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_GEOMETRY_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/geometry_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load geometry fixture");
    let geo = Sam3GeometryEncoder::from_weights(
        &w,
        "detector_model.geometry_encoder",
        &Sam3GeometryConfig::sam3(),
    )
    .expect("build geometry encoder");

    let boxes = fx.require("box_embeddings").unwrap().clone(); // [1,N,4] cxcywh
    let labels: Vec<i32> = fx.require("box_labels").unwrap().as_slice::<i32>().to_vec();

    // fpn_72 NCHW [1,256,72,72] → NHWC [1,72,72,256]
    let fpn_nchw = fx.require("fpn_72").unwrap().clone();
    let vision = fpn_nchw.transpose_axes(&[0, 2, 3, 1]).unwrap();
    // vision_pos NCHW [1,256,72,72] → flattened [1,H*W,256]
    let pos_nchw = fx.require("vision_pos_72").unwrap().clone();
    let vision_pos = pos_nchw
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap()
        .reshape(&[1, 72 * 72, 256])
        .unwrap();

    let out = geo
        .forward(&boxes, &labels, &vision, &vision_pos)
        .expect("geometry forward");

    let want = fx.require("geo_output").unwrap().clone();
    let cos = cosine(&out, &want);
    let maxabs = max_abs_diff(&out, &want);
    println!(
        "geometry prompt tokens: cosine={cos:.7} max_abs={maxabs:.5} shape={:?}",
        out.shape()
    );

    assert_eq!(out.shape(), want.shape(), "geometry output shape mismatch");
    assert!(cos > 0.9999, "geometry cosine {cos:.7} below 0.9999");
    assert!(maxabs < 1e-2, "geometry max_abs {maxabs:.5} above 1e-2");
}

#[test]
#[ignore = "needs SAM3_WEIGHTS + SAM3_GEOMETRY_E2E_FIXTURE"]
fn pvs_box_prompt_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_GEOMETRY_E2E_FIXTURE").unwrap_or_else(|_| {
        "scripts/spikes/sam3_oracle/geometry_e2e_fixture.safetensors".to_string()
    });

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load geometry e2e fixture");
    let seg = Sam3ImageSegmenter::from_weights(&w).expect("build segmenter");

    let pixel_values = fx.require("pixel_values").unwrap().clone();
    let input_ids = fx.require("input_ids").unwrap().clone();
    let mask: Vec<i32> = fx
        .require("attention_mask")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let boxes = fx.require("input_boxes").unwrap().clone();
    let labels: Vec<i32> = fx
        .require("input_boxes_labels")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();

    let got = seg
        .segment_with_boxes(
            &pixel_values,
            &input_ids,
            &mask,
            &boxes,
            &labels,
            (1.0, 1.0),
            0.5,
            0.5,
        )
        .expect("segment_with_boxes");

    let want_masks = fx.require("instance_masks").unwrap().clone(); // [m,288,288] uint8
    let want_n = want_masks.shape()[0] as usize;
    println!("PVS instances: got {} want {}", got.len(), want_n);
    assert_eq!(got.len(), want_n, "PVS instance count mismatch");

    let mut worst_iou = 1.0f32;
    for (i, inst) in got.iter().enumerate() {
        let want = want_masks
            .take_axis(Array::from_slice(&[i as i32], &[1]), 0)
            .unwrap()
            .reshape(&[288, 288])
            .unwrap();
        let m = iou(&inst.mask, &want);
        worst_iou = worst_iou.min(m);
        println!("  instance {i}: score={:.3} mask IoU={:.4}", inst.score, m);
    }
    assert!(
        worst_iou > 0.95,
        "worst PVS instance mask IoU {worst_iou:.4} below 0.95"
    );
}
