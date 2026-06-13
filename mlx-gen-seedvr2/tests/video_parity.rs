//! Video-mode parity + orchestration gate (sc-4814).
//!
//! (a) **Model path:** loads the 3B pipeline from the **raw** `numz/SeedVR2_comfyUI` checkpoint
//! (native converter + load on real weights) and runs the multi-frame [`run_model_5d`] over an
//! injected (T=8 → latentT=2 → decodedT=8) clip + noise, asserting the decoded 5-D tensor matches
//! the mflux golden (`video` component of `tools/dump_seedvr2_goldens.py`). This is the video analog
//! of the e2e image gate; both sides are MLX-Metal f32 → near bit-exact.
//!
//! (b) **Orchestration:** a small synthetic clip through the full [`generate_video`] (chunked, with a
//! `chunk_override` that forces a 2-chunk overlap blend) returns the right frame count + size.
//!
//! Both need the HF cache (and (a) the video golden); they skip otherwise.

use mlx_gen::weights::Weights;
use mlx_gen::Image;
use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::pipeline::Seedvr2Pipeline;
use mlx_rs::{Array, Dtype};

fn golden_dir() -> std::path::PathBuf {
    std::env::var("SEEDVR2_GOLDEN_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::Path::new(&std::env::var("HOME").unwrap())
                .join(".cache/mlx-gen-seedvr2-golden")
        })
}

fn raw_dir() -> Option<std::path::PathBuf> {
    let base = std::path::Path::new(&std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("seedvr2_ema_3b_fp16.safetensors")
        .exists()
        .then_some(snap)
}

fn cosine(got: &Array, exp: &Array) -> f32 {
    let g = got
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let e = exp
        .as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[-1])
        .unwrap();
    let (gs, es) = (g.as_slice::<f32>(), e.as_slice::<f32>());
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (a, b) in gs.iter().zip(es.iter()) {
        dot += (*a as f64) * (*b as f64);
        na += (*a as f64).powi(2);
        nb += (*b as f64).powi(2);
    }
    (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
}

/// A synthetic LR frame whose content varies with the frame index (so a real clip's per-frame work
/// is exercised, not a constant image).
fn lr_frame(w: u32, h: u32, t: u32) -> Image {
    let pixels = (0..w * h * 3).map(|i| ((i + t * 37) % 256) as u8).collect();
    Image {
        width: w,
        height: h,
        pixels,
    }
}

#[test]
fn seedvr2_video_model_path_matches_golden() {
    let (Some(raw), gdir) = (raw_dir(), golden_dir()) else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    if !gdir.join("video_io_f32.safetensors").exists() {
        eprintln!("SKIP: video golden absent (run dump_seedvr2_goldens.py --component video)");
        return;
    }
    let pipe = Seedvr2Pipeline::load(
        &raw,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Float32,
    )
    .expect("load from raw checkpoint");
    let io = Weights::from_file(gdir.join("video_io_f32.safetensors")).expect("video io");

    // processed is (1,3,8,64,64); H=W=64 (mult-of-16) → no crop. Inject the golden noise so the
    // comparison is deterministic regardless of RNG bit-matching.
    let decoded = pipe
        .run_model_5d(
            io.require("processed").unwrap(),
            io.require("noise").unwrap(),
            io.require("neg_embed").unwrap(),
            io.require("timestep").unwrap(),
            64,
            64,
        )
        .expect("run_model_5d");
    let exp = io.require("decoded").unwrap();
    assert_eq!(decoded.shape(), exp.shape(), "decoded 5-D shape");
    let cos = cosine(&decoded, exp);
    eprintln!(
        "video model-path decoded cosine = {cos:.6} (shape {:?})",
        decoded.shape()
    );
    assert!(cos > 0.999, "multi-frame model path diverged: {cos}");
}

#[test]
fn seedvr2_generate_video_runs_end_to_end() {
    let Some(raw) = raw_dir() else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    let pipe = Seedvr2Pipeline::load(
        &raw,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Float32,
    )
    .expect("load from raw checkpoint");

    // 10 LR frames → 96×96; force chunk=8 so the clip spans two chunks ([0:8],[4:12]) and exercises
    // the overlap cross-fade assembly. Output must preserve the frame count + size.
    let n = 10usize;
    let frames: Vec<Image> = (0..n as u32).map(|t| lr_frame(48, 48, t)).collect();
    let out = pipe
        .generate_video(&frames, 96, 96, 7, 0.0, Some(8))
        .expect("generate_video");
    assert_eq!(out.len(), n, "frame count preserved");
    for (i, f) in out.iter().enumerate() {
        assert_eq!((f.width, f.height), (96, 96), "frame {i} size");
        assert_eq!(f.pixels.len(), 96 * 96 * 3, "frame {i} pixel count");
    }
    eprintln!(
        "generate_video ok: {} frames @ 96x96 (2-chunk overlap blend)",
        out.len()
    );
}
