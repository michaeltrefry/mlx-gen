//! BiSeNet face-parsing parity vs facexlib (torch) (sc-3084).
//!
//! Gates on the real pipeline 512² crop (`align512.0` from the sc-3083 golden):
//!   1. **mask** — argmax class mask matches the torch facexlib parse: pixel agreement and per-class
//!      mean IoU ≈ 1.0 (the parse is a coarse argmax → tolerant; the spike's acceptance).
//!   2. **face_features_image** — the PuLID consumer output (`bg → white, else gray`) matches torch.
//!
//! Logits max-abs is printed (informational; the argmax is what the consumer uses).
//!
//! Goldens from `tools/convert_bisenet.py` (gitignored under `tools/golden/`) — hence `#[ignore]`.
//!
//! Run (torch venv for the golden, then cargo):
//!   ~/.bisenet-spike/venv/bin/python tools/convert_bisenet.py
//!   cargo test -p mlx-gen-face --release --test bisenet_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_face::{bisenet, face_features_image, BiSeNet};
use mlx_rs::Array;

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path).unwrap_or_else(|e| {
        panic!("missing golden {path}: {e}\nRun tools/convert_bisenet.py first.")
    })
}

#[test]
#[ignore = "needs local goldens from tools/convert_bisenet.py"]
fn bisenet_mask_and_features_parity() {
    let net = BiSeNet::from_weights(&golden("bisenet_parsing.safetensors")).unwrap();
    let g = golden("bisenet_goldens.safetensors");

    // input: RGB int32 [512,512,3] → [0,1] NHWC [1,512,512,3]
    let rgb_i32 = g.require("input").unwrap().try_as_slice::<i32>().unwrap();
    let rgb01: Vec<f32> = rgb_i32.iter().map(|&v| v as f32 / 255.0).collect();
    let rgb01 = Array::from_slice(&rgb01, &[1, 512, 512, 3]);
    let input = bisenet::to_parse_input(&rgb01).unwrap();

    // --- logits (informational) vs torch
    let logits = net.parse_logits(&input).unwrap();
    let lg = logits.try_as_slice::<f32>().unwrap();
    let want_lg = g.require("logits").unwrap().try_as_slice::<f32>().unwrap();
    let lmax = lg
        .iter()
        .zip(want_lg)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("logits max|Δ| vs torch = {lmax:.4}");

    // --- mask parity (pixel agreement + mean IoU over classes present in gt)
    let mask = net.parse_mask(&input).unwrap();
    let pred = mask.try_as_slice::<u32>().unwrap();
    let gt: Vec<u32> = g
        .require("mask")
        .unwrap()
        .try_as_slice::<i32>()
        .unwrap()
        .iter()
        .map(|&v| v as u32)
        .collect();
    assert_eq!(pred.len(), gt.len(), "mask len");
    let n = pred.len();
    let agree = pred.iter().zip(&gt).filter(|(a, b)| a == b).count();
    let pix_acc = agree as f32 / n as f32;

    // per-class IoU
    let mut inter = [0u64; 19];
    let mut union = [0u64; 19];
    let mut gt_has = [false; 19];
    for (&p, &t) in pred.iter().zip(&gt) {
        let (p, t) = (p as usize, t as usize);
        gt_has[t] = true;
        if p == t {
            inter[p] += 1;
        } else {
            union[p] += 1; // p side of the symmetric diff
            union[t] += 1; // t side
        }
    }
    let mut ious = Vec::new();
    for c in 0..19 {
        if gt_has[c] {
            let u = inter[c] + union[c];
            if u > 0 {
                ious.push(inter[c] as f32 / u as f32);
            }
        }
    }
    let mean_iou = ious.iter().sum::<f32>() / ious.len() as f32;
    let min_iou = ious.iter().cloned().fold(1.0f32, f32::min);
    println!("mask: pixel agreement {:.4}%, mean IoU {mean_iou:.4}, min class IoU {min_iou:.4} ({} classes)", pix_acc * 100.0, ious.len());

    // --- face_features_image (consumer contract) vs torch
    let ffi = face_features_image(&rgb01, &mask).unwrap();
    let got = ffi.try_as_slice::<f32>().unwrap();
    let want = g
        .require("face_features_image")
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap();
    let (mut ffi_diff, mut ffi_max) = (0usize, 0.0f32);
    for (a, b) in got.iter().zip(want) {
        let d = (a - b).abs();
        if d > 1e-6 {
            ffi_diff += 1;
        }
        ffi_max = ffi_max.max(d);
    }
    let ffi_frac = ffi_diff as f32 / got.len() as f32;
    println!(
        "face_features_image: {ffi_diff} px differ ({:.4}%), max|Δ| {ffi_max:.4}",
        ffi_frac * 100.0
    );

    assert!(
        pix_acc >= 0.99,
        "mask pixel agreement vs torch too low: {pix_acc}"
    );
    assert!(
        mean_iou >= 0.95,
        "mask mean IoU vs torch too low: {mean_iou}"
    );
    assert!(
        ffi_frac < 0.01,
        "face_features_image diverged from torch: {:.4}%",
        ffi_frac * 100.0
    );
}
