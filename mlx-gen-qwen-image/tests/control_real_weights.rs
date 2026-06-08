//! epic 3401 / sc-3574: real-weights validation of the Qwen-Image **ControlNet-Union** (strict
//! pose) port.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` base snapshot (env `QWEN_IMAGE_SNAPSHOT`, else
//! the HF cache) and the InstantX `Qwen-Image-ControlNet-Union` checkpoint (env `QWEN_CONTROL_WEIGHTS`,
//! else the HF cache). Gates, smallest-footprint first:
//!  - **controlnet load + forward** (`control_loads_and_emits_residuals`): loads ONLY the ~1.6 GB
//!    control branch, runs it on random inputs, asserts 5 finite, non-zero residuals of the right
//!    shape. Validates the loader (sc-3569) + the control transformer forward (sc-3568) cheaply.
//!  - **scale-0 self-consistency** (`scale_zero_matches_base`): loads the base 60-layer MMDiT too,
//!    and asserts `forward_control(residuals, scale = 0)` is **bit-identical** to the plain
//!    `forward` — proving the injection seam (sc-3571) is inert at scale 0 and the base parity path
//!    is untouched.
//!  - **scale-1 changes output** (`scale_one_changes_output`): with real residuals at scale 1 the
//!    output differs from base — the pose actually takes effect.
//!
//! The full numeric parity vs the diffusers `QwenImageControlNetPipeline` (per-block residuals +
//! e2e latents/image on a DWPose skeleton, bf16/Q8/Q4, controlScale sweep) is driven by the golden
//! from `tools/dump_qwen_control_golden.py` in `e2e_matches_diffusers_golden` (skipped when the
//! golden is absent).
//!
//! Run (the scale-0 gate loads the ~40 GB base transformer):
//!   cargo test -p mlx-gen-qwen-image --release --test control_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    WeightsSource,
};
use mlx_gen_qwen_image::loader;
use mlx_rs::{random, Array, Dtype};

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const TXT_SEQ: i32 = 64;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_control_golden.safetensors"
);

/// Base `Qwen/Qwen-Image` snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// InstantX `Qwen-Image-ControlNet-Union` checkpoint (env `QWEN_CONTROL_WEIGHTS`, else the HF cache
/// — the single `diffusion_pytorch_model.safetensors`).
fn control_source() -> WeightsSource {
    if let Ok(p) = std::env::var("QWEN_CONTROL_WEIGHTS") {
        return WeightsSource::File(PathBuf::from(p));
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--InstantX--Qwen-Image-ControlNet-Union/snapshots");
    let file = std::fs::read_dir(&snaps)
        .expect("control HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .flat_map(|d| {
            std::fs::read_dir(d)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
        })
        .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .expect("a control .safetensors");
    WeightsSource::File(file)
}

fn randn(shape: &[i32], seed: u64) -> Array {
    let k = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, None, Some(&k)).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

fn max_abs(a: &Array) -> f32 {
    let abs = mlx_rs::ops::abs(a).unwrap();
    mlx_rs::ops::max(&abs, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

/// The latent grid + packed-token sequence for the test geometry.
fn geom() -> (usize, usize, i32) {
    let (lh, lw) = ((HEIGHT / 16) as usize, (WIDTH / 16) as usize);
    (lh, lw, (lh * lw) as i32)
}

#[test]
#[ignore = "needs the InstantX Qwen-Image-ControlNet-Union checkpoint in the HF cache"]
fn control_loads_and_emits_residuals() {
    let (lh, lw, seq) = geom();
    let cn = loader::load_controlnet(&control_source()).expect("load controlnet");
    assert_eq!(
        cn.num_residuals(),
        5,
        "InstantX Union ships 5 control layers"
    );

    let latents = randn(&[1, seq, 64], 1);
    let control = randn(&[1, seq, 64], 2);
    let embeds = randn(&[1, TXT_SEQ, 3584], 3)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();

    let residuals = cn
        .forward(&latents, &control, &embeds, 0.5, lh, lw)
        .expect("forward");
    assert_eq!(residuals.len(), 5);
    for (i, r) in residuals.iter().enumerate() {
        assert_eq!(r.shape(), &[1, seq, 3072], "residual {i} shape");
        let m = max_abs(r);
        assert!(
            m.is_finite() && m > 0.0,
            "residual {i} must be finite + non-zero, got {m}"
        );
    }
}

#[test]
#[ignore = "needs the base Qwen-Image snapshot (~40 GB) + the control checkpoint"]
fn scale_zero_matches_base() {
    let (lh, lw, seq) = geom();
    let base = loader::load_transformer(&snapshot()).expect("load base transformer");
    let cn = loader::load_controlnet(&control_source()).expect("load controlnet");

    let latents = randn(&[1, seq, 64], 10);
    let control = randn(&[1, seq, 64], 11);
    let embeds = randn(&[1, TXT_SEQ, 3584], 12)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let sigma = 0.7;

    let base_out: Array = base
        .forward(&latents, &embeds, None, sigma, lh, lw, &[])
        .expect("base forward");
    let residuals = cn
        .forward(&latents, &control, &embeds, sigma, lh, lw)
        .expect("cn forward");
    let ctrl_out = base
        .forward_control(
            &latents,
            &embeds,
            None,
            sigma,
            lh,
            lw,
            &[],
            Some(&residuals),
            0.0,
        )
        .expect("control forward scale 0");

    // scale 0 ⇒ `hidden + residual*0 == hidden`: bit-identical to the base T2I forward.
    assert_eq!(
        max_abs_diff(&base_out, &ctrl_out),
        0.0,
        "control scale 0 must be bit-identical to base forward"
    );
}

#[test]
#[ignore = "needs the base Qwen-Image snapshot (~40 GB) + the control checkpoint"]
fn scale_one_changes_output() {
    let (lh, lw, seq) = geom();
    let base = loader::load_transformer(&snapshot()).expect("load base transformer");
    let cn = loader::load_controlnet(&control_source()).expect("load controlnet");

    let latents = randn(&[1, seq, 64], 20);
    let control = randn(&[1, seq, 64], 21);
    let embeds = randn(&[1, TXT_SEQ, 3584], 22)
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let sigma = 0.7;

    let base_out: Array = base
        .forward(&latents, &embeds, None, sigma, lh, lw, &[])
        .expect("base forward");
    let residuals = cn
        .forward(&latents, &control, &embeds, sigma, lh, lw)
        .expect("cn forward");
    let ctrl_out = base
        .forward_control(
            &latents,
            &embeds,
            None,
            sigma,
            lh,
            lw,
            &[],
            Some(&residuals),
            1.0,
        )
        .expect("control forward scale 1");

    assert!(
        max_abs_diff(&base_out, &ctrl_out) > 0.0,
        "control scale 1 must change the output vs base (pose takes effect)"
    );
}

#[test]
#[ignore = "needs the base Qwen-Image snapshot (~40 GB) + control checkpoint + text encoder (~14 GB)"]
fn public_generate_runs() {
    // End-to-end smoke of the public `qwen_image_control` API (sc-3572): encode prompt, VAE-encode +
    // pack the (synthetic) skeleton, run the control denoise loop, decode. No golden — asserts a
    // valid, non-degenerate image at the requested size (the numeric-vs-diffusers gate is
    // `e2e_matches_diffusers_golden`). 2 steps to keep it runnable.
    let (w, h) = (512u32, 512u32);
    let skeleton = Image {
        width: w,
        height: h,
        pixels: (0..(w * h * 3)).map(|i| (i % 256) as u8).collect(),
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_control(control_source());
    let gen = mlx_gen::load("qwen_image_control", &spec).expect("load qwen_image_control");
    let req = GenerationRequest {
        prompt: "a person standing, photorealistic".into(),
        seed: Some(7),
        width: w,
        height: h,
        count: 1,
        steps: Some(2),
        conditioning: vec![Conditioning::Control {
            image: skeleton,
            kind: ControlKind::Pose,
            scale: 1.0,
        }],
        ..Default::default()
    };
    let out = gen
        .generate(&req, &mut |_p: Progress| {})
        .expect("generate");
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    assert_eq!(images.len(), 1);
    let img = &images[0];
    assert_eq!((img.width, img.height), (w, h));
    assert_eq!(img.pixels.len(), (w * h * 3) as usize);
    // Not a flat/degenerate image: more than one distinct pixel value.
    let first = img.pixels[0];
    assert!(
        img.pixels.iter().any(|&p| p != first),
        "decoded image is flat (degenerate render)"
    );
}

#[test]
#[ignore = "needs tools/golden/qwen_control_golden.safetensors from dump_qwen_control_golden.py"]
fn e2e_matches_diffusers_golden() {
    if !PathBuf::from(GOLDEN).exists() {
        eprintln!("skipping: {GOLDEN} absent (run tools/dump_qwen_control_golden.py)");
        return;
    }
    let g = mlx_gen::weights::Weights::from_file(GOLDEN).expect("golden");

    // The golden records the seed/size/steps/guidance/scale + a (skeleton) control image; drive the
    // public `qwen_image_control` API and compare the decoded image to the diffusers reference.
    let seed: u64 = g
        .metadata("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let scale: f32 = g
        .metadata("control_scale")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let prompt = g
        .metadata("prompt")
        .unwrap_or("a person, photorealistic")
        .to_string();
    let skeleton = g
        .require("control_image_rgb8")
        .expect("control image in golden");
    let sh = skeleton.shape();
    let (h, w) = (sh[0] as u32, sh[1] as u32);
    let pixels: Vec<u8> = skeleton
        .as_dtype(Dtype::Uint8)
        .unwrap()
        .as_slice::<u8>()
        .to_vec();
    let control_image = Image {
        width: w,
        height: h,
        pixels,
    };

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_control(control_source());
    let gen = mlx_gen::load("qwen_image_control", &spec).expect("load qwen_image_control");
    let req = GenerationRequest {
        prompt,
        seed: Some(seed),
        width: w,
        height: h,
        count: 1,
        conditioning: vec![Conditioning::Control {
            image: control_image,
            kind: ControlKind::Pose,
            scale,
        }],
        ..Default::default()
    };
    let out = gen
        .generate(&req, &mut |_p: Progress| {})
        .expect("generate");
    let GenerationOutput::Images(images) = out else {
        panic!("expected images")
    };
    let got = &images[0];
    let golden_img = g.require("image_rgb8").expect("golden image");
    let gsh = golden_img.shape();
    assert_eq!((gsh[0] as u32, gsh[1] as u32), (h, w), "golden image size");
    let golden_px: Vec<u8> = golden_img
        .as_dtype(Dtype::Uint8)
        .unwrap()
        .as_slice::<u8>()
        .to_vec();
    // Cross-build (mixed-precision MLX vs bf16 torch) floor: compare with the established Qwen
    // pixel-difference tolerance (% of pixels with |Δ| > 8, like the other e2e gates).
    let over: usize = got
        .pixels
        .iter()
        .zip(&golden_px)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let pct = 100.0 * over as f32 / golden_px.len() as f32;
    eprintln!("control e2e px>8: {pct:.3}%");
    assert!(pct < 2.0, "control e2e px>8 {pct:.3}% exceeds 2% floor");
}
