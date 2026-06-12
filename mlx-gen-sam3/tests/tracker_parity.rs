//! SAM3-F1 tracker single-frame (box-prompt PVS) parity (sc-4924): load the real `facebook/sam3`
//! weights, run the `Sam3Tracker` single-frame box path, and check it against the torch oracle
//! (`scripts/spikes/sam3_oracle/dump_tracker_fixture.py`).
//!
//! Run:
//!   SAM3_WEIGHTS=$HOME/.cache/huggingface/hub/models--facebook--sam3/snapshots/<rev>/model.safetensors \
//!   SAM3_TRACKER_FIXTURE=scripts/spikes/sam3_oracle/tracker_fixture.safetensors \
//!     cargo test -p mlx-gen-sam3 --release --test tracker_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_sam3::Sam3Tracker;
use mlx_rs::ops::{abs, max, maximum, multiply, sqrt, subtract, sum};
use mlx_rs::{Array, Dtype};

fn scalar(a: &Array) -> f32 {
    a.as_dtype(Dtype::Float32).unwrap().item::<f32>()
}

fn cosine(a: &Array, b: &Array) -> f32 {
    let a = a.reshape(&[-1]).unwrap();
    let b = b.reshape(&[-1]).unwrap();
    let dot = scalar(&sum(multiply(&a, &b).unwrap(), None).unwrap());
    let na = scalar(&sqrt(sum(multiply(&a, &a).unwrap(), None).unwrap()).unwrap());
    let nb = scalar(&sqrt(sum(multiply(&b, &b).unwrap(), None).unwrap()).unwrap());
    dot / (na * nb)
}

/// Max-abs error relative to the fixture's own dynamic range.
fn max_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let max_abs = scalar(&max(&diff, None).unwrap());
    let denom = scalar(
        &maximum(
            max(abs(want).unwrap(), None).unwrap(),
            Array::from_f32(1e-6),
        )
        .unwrap(),
    );
    max_abs / denom
}

#[test]
#[ignore = "needs SAM3_WEIGHTS=<facebook/sam3 model.safetensors> + SAM3_TRACKER_FIXTURE"]
fn tracker_single_frame_matches_oracle() {
    let weights_path =
        std::env::var("SAM3_WEIGHTS").expect("set SAM3_WEIGHTS to facebook/sam3 model.safetensors");
    let fixture_path = std::env::var("SAM3_TRACKER_FIXTURE")
        .unwrap_or_else(|_| "scripts/spikes/sam3_oracle/tracker_fixture.safetensors".to_string());

    let w = Weights::from_file(&weights_path).expect("load sam3 weights");
    let fx = Weights::from_file(&fixture_path).expect("load tracker fixture");
    let tracker = Sam3Tracker::from_weights(&w).expect("build tracker");

    let pixel_values = fx.require("pixel_values").expect("pixel_values").clone();
    let box_v = fx
        .require("box_1008")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let box_xyxy = [box_v[0], box_v[1], box_v[2], box_v[3]];

    // --- Stage check: tracker neck (shared backbone → FPN → conv_s0/s1). Isolates the neck from the
    // prompt encoder + mask decoder. ours NHWC → NCHW to match the fixture.
    let (_emb, high_res) = tracker.encode_frame(&pixel_values).expect("encode_frame");
    for (i, key) in ["high_res_s0", "high_res_s1"].iter().enumerate() {
        let got = high_res[i].transpose_axes(&[0, 3, 1, 2]).unwrap();
        let want = fx.require(key).unwrap().clone();
        assert_eq!(got.shape(), want.shape(), "{key} shape");
        let (c, r) = (cosine(&got, &want), max_rel(&got, &want));
        println!("{key}: cosine={c:.7} max_rel={r:.2e}");
        // cosine is the gate here (the FPN's transposed convs accumulate MLX Metal reduced-precision
        // matmul error, so the global-denominator max_rel has fp-noise outliers).
        assert!(c > 0.9999, "{key} cosine {c}");
    }

    // --- End-to-end: box-prompt → best low-res mask + iou + object score.
    let out = tracker.segment(&pixel_values, box_xyxy).expect("segment");
    let want_mask = fx.require("best_low_res").unwrap().clone(); // [288, 288]
    assert_eq!(out.low_res.shape(), want_mask.shape(), "mask shape");
    let c = cosine(&out.low_res, &want_mask);
    let r = max_rel(&out.low_res, &want_mask);
    // mask-logit agreement (sign/argmax is what the binary mask depends on): cosine should be ~1.
    let want_iou = fx
        .require("iou_scores")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let want_obj = scalar(fx.require("object_score").unwrap());
    println!(
        "mask: cosine={c:.7} max_rel={r:.2e}  iou ours={:.4} oracle_best={:.4}  obj ours={:.3} oracle={:.3}",
        out.iou,
        want_iou.iter().cloned().fold(f32::MIN, f32::max),
        out.object_score,
        want_obj
    );
    assert!(c > 0.999, "mask cosine {c}");
    // object score is a large logit (~24); compare relative (MLX Metal matmul is reduced-precision).
    assert!(
        (out.object_score - want_obj).abs() / want_obj.abs().max(1.0) < 0.01,
        "object score ours {} vs {want_obj}",
        out.object_score
    );
}
