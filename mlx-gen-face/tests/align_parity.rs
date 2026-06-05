//! 5-point alignment parity vs insightface (sc-3083).
//!
//! sc-3083 delivers the *alignment*, so the gates isolate alignment correctness:
//!   1. **M** — our [`estimate_norm`] (Umeyama) matches insightface `estimate_norm` (≤1e-3; the
//!      solver matches skimage to ~1e-13 in f64 — the residual is f32 golden-dump rounding).
//!   2. **crop** — our [`norm_crop`] bit-matches `cv2.warpAffine` (fixed-point INTER_LINEAR) except
//!      a handful of ±1-LSB pixels at sub-pixel quantization ties (our f64 Umeyama vs cv2's
//!      skimage-LAPACK M differ at machine-epsilon, flipping a few boundary roundings). 512² crop
//!      bit-exact.
//!   3. **alignment fidelity (the real gate)** — ArcFace on *our* crop yields the *same embedding*
//!      as ArcFace on the *reference cv2* crop (cos ≥ 0.99999). This proves the alignment is
//!      faithful independent of any ArcFace numerics.
//!
//! NOTE (sc-3081 finding, surfaced here): the native ArcFace matches onnx's canonical pure-numpy
//! `ReferenceEvaluator` (`emb_ref`) at cos ≥ 0.998, but matches insightface's *ONNX Runtime* output
//! (`emb_onnx`) only at ~0.98 — because ORT's MLAS kernels diverge from exact math by the same ~0.98
//! over the 100-layer iresnet100 (ORT vs its own ReferenceEvaluator is also ~0.98). That gap is an
//! ArcFace/ORT numerical-fidelity issue, *not* an alignment issue, so it is reported (not asserted)
//! here. Goldens from `tools/dump_face_align_golden.py` (+ sc-3081/3082 weight goldens), gitignored.
//!
//! Run:
//!   ~/.dwpose-spike/venv/bin/python tools/dump_face_align_golden.py
//!   cargo test -p mlx-gen-face --release --test align_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_face::{align, norm_crop, to_arcface_input, ArcFace, Scrfd};

fn golden(name: &str) -> Weights {
    let path = format!("{}/../tools/golden/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("missing golden {path}: {e}\nRun the dump_*.py tools first."))
}

fn u8_tensor(g: &Weights, key: &str) -> (Vec<u8>, Vec<i32>) {
    let a = g.require(key).unwrap();
    let bytes = a
        .try_as_slice::<i32>()
        .unwrap()
        .iter()
        .map(|&v| v as u8)
        .collect();
    (bytes, a.shape().to_vec())
}

fn kps5(g: &Weights, key: &str) -> [[f32; 2]; 5] {
    let v = g
        .require(key)
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap()
        .to_vec();
    let mut k = [[0.0f32; 2]; 5];
    for (i, p) in k.iter_mut().enumerate() {
        p[0] = v[i * 2];
        p[1] = v[i * 2 + 1];
    }
    k
}

fn vec_f32(g: &Weights, key: &str) -> Vec<f32> {
    g.require(key)
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn diff(a: &[u8], b: &[u8]) -> (usize, i32) {
    assert_eq!(a.len(), b.len(), "buffer len");
    a.iter().zip(b).fold((0, 0), |(n, m), (&x, &y)| {
        let d = (x as i32 - y as i32).abs();
        (n + (d != 0) as usize, m.max(d))
    })
}

fn embed(net: &ArcFace, crop: &[u8]) -> Vec<f32> {
    let input = to_arcface_input(std::slice::from_ref(&crop.to_vec())).unwrap();
    net.forward(&input)
        .unwrap()
        .try_as_slice::<f32>()
        .unwrap()
        .to_vec()
}

#[test]
#[ignore = "needs local goldens (tools/dump_face_align_golden.py + sc-3081/3082 goldens)"]
fn align_norm_crop_parity() {
    let g = golden("face_align_goldens.safetensors");
    let arc = ArcFace::from_weights(&golden("arcface_iresnet100.safetensors")).unwrap();
    let (img, ishape) = u8_tensor(&g, "image");
    let (ih, iw) = (ishape[0] as usize, ishape[1] as usize);
    let n = g.require("n_faces").unwrap().item::<i32>() as usize;
    println!("image {ih}x{iw}, {n} faces");

    let (mut worst_m, mut worst_max) = (0.0f32, 0i32);
    let (mut worst_frac, mut worst_512) = (0.0f32, 0usize);
    let (mut min_align_cos, mut min_ref_cos, mut min_ort_cos) = (1.0f32, 1.0f32, 1.0f32);

    for i in 0..n {
        let kps = kps5(&g, &format!("kps.{i}"));

        // (1) M vs insightface estimate_norm
        let m = align::estimate_norm(&kps).0;
        let want_m = vec_f32(&g, &format!("M.{i}"));
        let dm = m
            .iter()
            .zip(&want_m)
            .map(|(a, b)| (*a as f32 - b).abs())
            .fold(0.0, f32::max);
        worst_m = worst_m.max(dm);

        // (2) crop vs cv2.warpAffine (norm_crop 112² and the facexlib 512²)
        let (want_crop, _) = u8_tensor(&g, &format!("norm_crop.{i}"));
        let crop = norm_crop(&img, ih, iw, &kps);
        let (cd, cm) = diff(&crop, &want_crop);
        let frac = cd as f32 / want_crop.len() as f32;
        worst_max = worst_max.max(cm);
        worst_frac = worst_frac.max(frac);
        let (want_512, _) = u8_tensor(&g, &format!("align512.{i}"));
        let (d512, _) = diff(&align::align_face_512(&img, ih, iw, &kps), &want_512);
        worst_512 = worst_512.max(d512);

        // (3) alignment fidelity: ArcFace(our crop) vs ArcFace(cv2 crop) — isolates alignment.
        let emb = embed(&arc, &crop);
        let emb_cv2 = embed(&arc, &want_crop);
        let align_cos = cosine(&emb, &emb_cv2);
        min_align_cos = min_align_cos.min(align_cos);

        // (informational) native ArcFace vs canonical math (RefEval) and vs insightface ORT.
        let ref_cos = cosine(&emb, &vec_f32(&g, &format!("emb_ref.{i}")));
        let ort_cos = cosine(&emb, &vec_f32(&g, &format!("emb_onnx.{i}")));
        min_ref_cos = min_ref_cos.min(ref_cos);
        min_ort_cos = min_ort_cos.min(ort_cos);
        println!(
            "  face {i}: |ΔM| {dm:.2e}  crop {cd}px(max {cm},{:.3}%)  512diff {d512}  | align-cos {align_cos:.6}  ref-cos {ref_cos:.6}  ort-cos {ort_cos:.6}",
            frac * 100.0
        );
    }

    println!(
        "\nALIGNMENT: worst|ΔM| {worst_m:.2e}, worst crop ±{worst_max} ({:.3}%), worst 512diff {worst_512}, min align-cos {min_align_cos:.6}",
        worst_frac * 100.0
    );
    println!("ARCFACE  : min cos vs canonical-math(RefEval) {min_ref_cos:.4} | min cos vs insightface(ORT) {min_ort_cos:.4}  (ORT≠math by ~same — sc-3081/ORT numerics)");

    // alignment gates (sc-3083 scope)
    assert!(
        worst_m < 1e-3,
        "estimate_norm M drift vs insightface: {worst_m}"
    );
    assert!(
        worst_max <= 1,
        "norm_crop diff must be ≤ ±1 LSB (warp not faithful): max {worst_max}"
    );
    assert!(
        worst_frac < 1e-3,
        "too many crop pixels differ ({:.3}%) — warp not faithful",
        worst_frac * 100.0
    );
    assert_eq!(worst_512, 0, "align_face_512 must bit-match cv2.warpAffine");
    assert!(
        min_align_cos >= 0.99999,
        "alignment changes the embedding (not faithful): {min_align_cos}"
    );
    // ArcFace matches canonical math (the achievable parity; ORT divergence is the sc-3081 finding).
    assert!(
        min_ref_cos >= 0.998,
        "native ArcFace diverged from onnx ReferenceEvaluator: {min_ref_cos}"
    );
}

/// Full native loop: SCRFD detect → norm_crop → ArcFace. Validates the alignment glue closes the
/// photo→embedding path; embedding parity is measured vs canonical math (ORT divergence = sc-3081).
#[test]
#[ignore = "needs local goldens (tools/dump_face_align_golden.py + sc-3081/3082 goldens)"]
fn end_to_end_detect_align_embed() {
    let g = golden("face_align_goldens.safetensors");
    let arc = ArcFace::from_weights(&golden("arcface_iresnet100.safetensors")).unwrap();
    let scrfd = Scrfd::from_weights(&golden("scrfd_10g.safetensors")).unwrap();
    let (img, ishape) = u8_tensor(&g, "image");
    let (ih, iw) = (ishape[0] as usize, ishape[1] as usize);
    let det_scale = g.require("det_scale").unwrap().item::<f32>();
    let n = g.require("n_faces").unwrap().item::<i32>() as usize;

    let dets = scrfd
        .detect(g.require("blob").unwrap(), det_scale, 0.5, 0.4)
        .unwrap();
    println!("SCRFD detected {} faces (insightface {n})", dets.len());
    assert_eq!(dets.len(), n, "SCRFD face count vs insightface");

    let mut min_ref_cos = 1.0f32;
    let mut min_ort_cos = 1.0f32;
    for i in 0..n {
        let want_kps = kps5(&g, &format!("kps.{i}"));
        let det = dets
            .iter()
            .min_by(|a, b| kps_l2(&a.kps, &want_kps).total_cmp(&kps_l2(&b.kps, &want_kps)))
            .unwrap();
        let emb = embed(&arc, &norm_crop(&img, ih, iw, &det.kps));
        let ref_cos = cosine(&emb, &vec_f32(&g, &format!("emb_ref.{i}")));
        let ort_cos = cosine(&emb, &vec_f32(&g, &format!("emb_onnx.{i}")));
        min_ref_cos = min_ref_cos.min(ref_cos);
        min_ort_cos = min_ort_cos.min(ort_cos);
        println!(
            "  face {i}: kps L2 {:.3}px  ref-cos {ref_cos:.6}  ort-cos {ort_cos:.6}",
            kps_l2(&det.kps, &want_kps)
        );
    }
    println!(
        "e2e min cos vs canonical-math {min_ref_cos:.4} | vs insightface(ORT) {min_ort_cos:.4}"
    );
    assert!(
        min_ref_cos >= 0.998,
        "native detect→align→embed diverged from canonical math: {min_ref_cos}"
    );
}

fn kps_l2(a: &[[f32; 2]; 5], b: &[[f32; 2]; 5]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(p, q)| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt())
        .sum()
}
