//! sc-2465 slice 7a: Qwen-Image-Edit pipeline parity vs the frozen fork. Micro-gated.
//!
//! - **Gate 1 (here)**: the multi-image (dual-latent) RoPE — `QwenRope3d::forward_multi` over
//!   `[noise_grid, cond_grid]` — vs the fork's `QwenEmbedRopeMLX`. Weight-free.
//!
//! Run: `cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_edit_rope_golden.py`, then
//! `cargo test -p mlx-gen-qwen-image --release --test edit_real_weights -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    CancelFlag, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_qwen_image::transformer::QwenRope3d;
use mlx_gen_qwen_image::{
    decoded_to_image, denoise_edit_with_progress, encode_reference_latents, loader, model_edit,
    qwen_scheduler, tokenize_edit, unpack_latents, FlowMatchSampler, ImageInput,
    QwenImageProcessor,
};
use mlx_rs::Array;

const ROPE_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_edit_rope_golden.safetensors"
);
const EDIT_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_edit_golden.safetensors"
);
const EDIT_Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_edit_q8_golden.safetensors"
);
const EDIT_MULTI_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_edit_multi_golden.safetensors"
);
const EDIT_MULTI_Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_image_edit_multi_q8_golden.safetensors"
);
const TOKENIZE_DEBUG_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_edit_tokenize_debug.safetensors"
);
const VISION_STAGES_DEBUG_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_edit_vision_stages_debug.safetensors"
);

// Must match tools/dump_qwen_image_edit_golden.py.
const STEPS: usize = 2;
const GUIDANCE: f32 = 4.0;

fn edit_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_EDIT_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2509/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
}

// Must match tools/dump_qwen_edit_rope_golden.py.
#[test]
#[ignore = "needs local edit-rope golden"]
fn edit_rope_multi_image_matches_fork() {
    let g = Weights::from_file(ROPE_GOLDEN).unwrap();
    let (ic, is_, tc, ts) = QwenRope3d::qwen_image()
        .forward_multi(&[(8, 12), (6, 6)], 20)
        .unwrap();
    for (name, got, key) in [
        ("img_cos", &ic, "img_cos"),
        ("img_sin", &is_, "img_sin"),
        ("txt_cos", &tc, "txt_cos"),
        ("txt_sin", &ts, "txt_sin"),
    ] {
        let want = g.require(key).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let d = max_abs_diff(got, want);
        println!("edit rope {name} {:?}: max abs diff {d:.3e}", got.shape());
        assert!(d < 1e-5, "{name} max abs diff {d:.3e}");
    }
}

/// Gate 2: the full dual-latent denoise loop (concat noise+ref → transformer with `cond_grids` →
/// slice → CFG → Euler) vs the fork's edit loop. Feeds the golden noise + prompt embeds + packed
/// reference latents + cond grid (so the tokenizer / VL encoder / VAE-encode — each separately
/// verified — are out of scope), loads the real transformer + VAE from the Edit snapshot, and
/// compares the final latents + decoded image.
#[test]
#[ignore = "needs real Qwen-Image-Edit-2509 transformer+VAE weights + local edit golden"]
fn edit_pipeline_matches_fork() {
    let g = Weights::from_file(EDIT_GOLDEN).unwrap();
    let root = edit_snapshot();
    let transformer = loader::load_transformer(&root).unwrap();
    let vae = loader::load_vae(&root).unwrap();

    let dims = g.require("out_dims").unwrap();
    let dims = dims.as_slice::<i32>();
    let (w, h) = (dims[0] as u32, dims[1] as u32);
    let cg = g.require("cond_grid").unwrap();
    let cg = cg.as_slice::<i32>();
    let cond_grids = vec![(cg[0] as usize, cg[1] as usize)];

    let noise = g.require("noise").unwrap().clone();
    let static_lat = g.require("static_image_latents").unwrap();
    let pos = g.require("pos_embeds").unwrap();
    let neg = g.require("neg_embeds").unwrap();
    let sampler = FlowMatchSampler::new(qwen_scheduler(STEPS, w, h));

    let latents = denoise_edit_with_progress(
        &transformer,
        &sampler,
        noise,
        static_lat,
        &cond_grids,
        pos,
        Some(neg),
        GUIDANCE,
        w,
        h,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();

    let want = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), want.shape(), "final_latents shape");
    let (peak, mean) = rel_errors(&latents, want);
    println!("edit final_latents: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    assert!(mean < 2e-2, "edit final_latents mean-rel {mean:.3e}");
    assert!(peak < 1e-1, "edit final_latents peak-rel {peak:.3e}");

    let unpacked = unpack_latents(&latents, w, h).unwrap();
    let decoded = vae.decode(&unpacked).unwrap();
    let want_dec = g.require("decoded").unwrap();
    let (dpeak, dmean) = rel_errors(&decoded, want_dec);
    println!("edit decoded: peak-rel {dpeak:.3e}  mean-rel {dmean:.3e}");
    assert!(dmean < 5e-2, "edit decoded mean-rel {dmean:.3e}");
}

/// The deterministic synthetic reference image (matches `tools/dump_qwen_image_edit_golden.py`).
fn synthetic_reference() -> Image {
    let (w, h) = (512u32, 512u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = (y + x) % 256;
            pixels.push(base as u8);
            pixels.push(((base * 2) % 256) as u8);
            pixels.push(((base * 3) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// A second deterministic synthetic reference (distinct pattern, same 512² size) for the
/// multi-image gates. Matches the `rgb2` gradient in `tools/dump_qwen_image_edit_golden.py`
/// (`base = (2y + x) % 256`, channels `[3·base, base, 2·base]`).
fn synthetic_reference2() -> Image {
    let (w, h) = (512u32, 512u32);
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let base = (2 * y + x) % 256;
            pixels.push(((base * 3) % 256) as u8);
            pixels.push(base as u8);
            pixels.push(((base * 2) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Gate 3: the dual-latent **reference** path (LANCZOS resize → normalize → VAE-encode → pack) vs
/// the fork's packed `static_image_latents`. VAE-only (no transformer/LM), so it isolates the new
/// `encode_reference_latents` / LANCZOS code.
#[test]
#[ignore = "needs real Qwen-Image-Edit-2509 VAE weights + local edit golden"]
fn edit_reference_latents_matches_fork() {
    let g = Weights::from_file(EDIT_GOLDEN).unwrap();
    let vae = loader::load_vae(&edit_snapshot()).unwrap();
    let img = synthetic_reference();
    // condition_resize_dims(512,512) → (384,384) = the VL/dual-latent resolution from the dump.
    let (static_lat, cond_grid) = encode_reference_latents(
        &vae,
        ImageInput {
            data: &img.pixels,
            height: img.height as usize,
            width: img.width as usize,
        },
        384,
        384,
    )
    .unwrap();
    assert_eq!(cond_grid, (24, 24), "cond grid");
    let want = g.require("static_image_latents").unwrap();
    assert_eq!(static_lat.shape(), want.shape(), "static latents shape");
    let (peak, mean) = rel_errors(&static_lat, want);
    println!("edit ref latents: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    // LANCZOS matches PIL only to ~1/255 (fixed-point resampler not bit-reproduced); the diff is
    // a few border pixels through the VAE, not a math error.
    assert!(mean < 3e-2, "ref latents mean-rel {mean:.3e}");
}

/// Debug: my `tokenize_edit` input_ids + pixel_values vs the fork's, for the synthetic reference
/// (tokenizer + processor only, no model weights). Pinpoints whether the VL-input divergence is in
/// the text tokens or the image pixels.
#[test]
#[ignore = "needs the Edit tokenizer.json + local tokenize-debug golden"]
fn edit_tokenize_debug() {
    let g = Weights::from_file(TOKENIZE_DEBUG_GOLDEN).unwrap();
    let tokenizer = loader::load_tokenizer(&edit_snapshot()).unwrap();
    let processor = QwenImageProcessor::default();
    let img = synthetic_reference();
    let inp = tokenize_edit(
        &tokenizer,
        &processor,
        "make it autumn",
        ImageInput {
            data: &img.pixels,
            height: img.height as usize,
            width: img.width as usize,
        },
    )
    .unwrap();

    let want_ids = g.require("input_ids").unwrap();
    assert_eq!(inp.input_ids.shape(), want_ids.shape(), "input_ids shape");
    let (a, b) = (inp.input_ids.as_slice::<i32>(), want_ids.as_slice::<i32>());
    let id_diffs = a.iter().zip(b).filter(|(x, y)| x != y).count();
    println!("input_ids: {id_diffs}/{} differ", b.len());
    if id_diffs > 0 {
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
        println!(
            "  first diff at {first}: mine {} fork {}",
            a[first], b[first]
        );
    }

    let want_pv = g.require("pixel_values").unwrap();
    println!(
        "pixel_values mine {:?} fork {:?}",
        inp.pixel_values.shape(),
        want_pv.shape()
    );
    let (peak, mean) = rel_errors(&inp.pixel_values, want_pv);
    println!("pixel_values: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
}

/// Debug bisection: per-stage vision activations vs the fork — finds the first divergent op.
#[test]
#[ignore = "needs vision weights + local stages-debug golden"]
fn edit_vision_stages_debug() {
    let tok = Weights::from_file(TOKENIZE_DEBUG_GOLDEN).unwrap();
    let g = Weights::from_file(VISION_STAGES_DEBUG_GOLDEN).unwrap();
    let vt = loader::load_vision_encoder(&edit_snapshot()).unwrap();
    let pixel = tok.require("pixel_values").unwrap();
    let caps = vt.forward_capture(pixel, &[[1, 28, 28]]).unwrap();
    for (name, got) in &caps {
        let want = g.require(name).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name} shape");
        let (peak, mean) = rel_errors(got, want);
        println!(
            "stage {name:>12} {:?}: peak-rel {peak:.3e}  mean-rel {mean:.3e}",
            got.shape()
        );
    }
}

/// Debug: my tokenize_edit → VL-encode embeds vs the fork's `pos_embeds` (loads only LM + vision).
/// Isolates whether the Generator's self-computed conditioning matches the fork's.
#[test]
#[ignore = "needs LM + vision weights + local edit golden"]
fn edit_pos_embeds_matches_fork() {
    let g = Weights::from_file(EDIT_GOLDEN).unwrap();
    let tokenizer = loader::load_tokenizer(&edit_snapshot()).unwrap();
    let processor = QwenImageProcessor::default();
    let vl = loader::load_vision_language_encoder(&edit_snapshot()).unwrap();
    let img = synthetic_reference();
    let inp = tokenize_edit(
        &tokenizer,
        &processor,
        "make it autumn",
        ImageInput {
            data: &img.pixels,
            height: img.height as usize,
            width: img.width as usize,
        },
    )
    .unwrap();
    let grids: Vec<[i32; 3]> = inp
        .grid_thw
        .as_slice::<i32>()
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    println!(
        "my grid_thw {:?}  input_ids {:?}",
        grids,
        inp.input_ids.shape()
    );
    let embeds = vl
        .encode(
            &inp.input_ids,
            &inp.attention_mask,
            &inp.pixel_values,
            &grids,
        )
        .unwrap();
    let want = g.require("pos_embeds").unwrap();
    println!("my embeds {:?}  fork {:?}", embeds.shape(), want.shape());
    assert_eq!(embeds.shape(), want.shape(), "pos_embeds shape");
    let (peak, mean) = rel_errors(&embeds, want);
    println!("edit pos_embeds: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
}

/// Determinism: run the SAME edit (same ref + prompt + seed) twice through the Rust Generator and
/// assert the two decoded images are BYTE-IDENTICAL. Confirms the Rust pipeline is a deterministic
/// function (so any fork divergence is cross-implementation, not run-to-run noise / NAX chaos).
#[test]
#[ignore = "needs the full real Qwen-Image-Edit-2509 model"]
fn edit_generate_is_deterministic_rust() {
    let spec = LoadSpec::new(WeightsSource::Dir(edit_snapshot()));
    let generator = model_edit::load(&spec).unwrap();
    let req = GenerationRequest {
        prompt: "make it autumn".into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(42),
        steps: Some(STEPS as u32),
        guidance: Some(GUIDANCE),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_reference(),
            strength: None,
        }],
        ..Default::default()
    };
    let grab = || match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.swap_remove(0),
        _ => panic!("expected images"),
    };
    let a = grab();
    let b = grab();
    let differ = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .filter(|(x, y)| x != y)
        .count();
    println!(
        "rust edit determinism: {differ}/{} bytes differ between two runs",
        a.pixels.len()
    );
    assert_eq!(differ, 0, "two identical Rust edits must be byte-identical");
}

/// Gate 4 (full slice 7a): the **whole** `qwen_image_edit` Generator — load → tokenize_edit →
/// VL-encode → dual-latent → edit denoise → decode → RGB8 — vs the fork's decoded edit output for
/// the same reference + prompt + seed. The heaviest test (LM + vision + transformer + VAE).
#[test]
#[ignore = "needs the full real Qwen-Image-Edit-2509 model + local edit golden"]
fn edit_generate_matches_fork() {
    let g = Weights::from_file(EDIT_GOLDEN).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(edit_snapshot()));
    let generator = model_edit::load(&spec).unwrap();

    let req = GenerationRequest {
        prompt: "make it autumn".into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(42),
        steps: Some(STEPS as u32),
        guidance: Some(GUIDANCE),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_reference(),
            strength: None,
        }],
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let got = match out {
        GenerationOutput::Images(mut v) => v.swap_remove(0),
        _ => panic!("expected images"),
    };

    let want = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    assert_eq!(got.pixels.len(), want.pixels.len(), "pixel count");
    let differ = got
        .pixels
        .iter()
        .zip(&want.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / got.pixels.len() as f32;
    println!("edit generate pixels >8 apart: {:.3}%", frac * 100.0);
    // PIXEL-PARITY with the fork (observed 0.000% px>8), like T2I: the conditioning-image resampler
    // bit-matches PIL (`resize_u8` fixed-point path) → bit-identical pixel_values, and the
    // f32-activation path keeps the 60-layer net within bf16-rounding (< 8/255). The tiny margin
    // absorbs the irreducible bf16-matmul reduction-order floor (~1e-5/op, below the u8 threshold).
    assert!(
        frac < 0.001,
        "edit generate diverges from the fork: {:.3}% px>8",
        frac * 100.0
    );
}

/// Slice 7b (denoise isolation): the dual-latent loop with a **Q8-quantized** transformer (group 64,
/// the set `nn.quantize` hits; VL encoder + VAE stay bf16) vs the fork-Q8 edit golden. Feeds the
/// golden inputs (so only the Q8 transformer is under test), mirroring the T2I
/// `transformer_q8_pipeline_matches_fork` gate.
#[test]
#[ignore = "needs real Qwen-Image-Edit-2509 transformer+VAE weights + local Q8 edit golden"]
fn edit_pipeline_q8_matches_fork() {
    let g = Weights::from_file(EDIT_Q8_GOLDEN).unwrap();
    let root = edit_snapshot();
    let mut transformer = loader::load_transformer(&root).unwrap();
    transformer.quantize(8).unwrap();
    let vae = loader::load_vae(&root).unwrap();

    let dims = g.require("out_dims").unwrap();
    let dims = dims.as_slice::<i32>();
    let (w, h) = (dims[0] as u32, dims[1] as u32);
    let cg = g.require("cond_grid").unwrap();
    let cg = cg.as_slice::<i32>();
    let cond_grids = vec![(cg[0] as usize, cg[1] as usize)];

    let noise = g.require("noise").unwrap().clone();
    let static_lat = g.require("static_image_latents").unwrap();
    let pos = g.require("pos_embeds").unwrap();
    let neg = g.require("neg_embeds").unwrap();
    let sampler = FlowMatchSampler::new(qwen_scheduler(STEPS, w, h));

    let latents = denoise_edit_with_progress(
        &transformer,
        &sampler,
        noise,
        static_lat,
        &cond_grids,
        pos,
        Some(neg),
        GUIDANCE,
        w,
        h,
        &CancelFlag::default(),
        &mut |_| {},
    )
    .unwrap();

    let want = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), want.shape(), "Q8 edit final_latents shape");
    let (peak, mean) = rel_errors(&latents, want);
    println!("Q8 edit final_latents: peak-rel {peak:.3e}  mean-rel {mean:.3e}");
    // Byte-identical Q8 packing (sc-2342) → near the qmm floor; a touch looser than bf16 to absorb
    // quantized-matmul accumulation over the loop (same bounds as the T2I Q8 gate).
    assert!(mean < 3e-2, "Q8 edit final_latents mean-rel {mean:.3e}");
    assert!(peak < 1.5e-1, "Q8 edit final_latents peak-rel {peak:.3e}");

    let unpacked = unpack_latents(&latents, w, h).unwrap();
    let decoded = vae.decode(&unpacked).unwrap();
    let want_dec = g.require("decoded").unwrap();
    let (_dpeak, dmean) = rel_errors(&decoded, want_dec);
    println!("Q8 edit decoded: mean-rel {dmean:.3e}");
    assert!(dmean < 6e-2, "Q8 edit decoded mean-rel {dmean:.3e}");
}

/// Slice 7b (full e2e): the whole `qwen_image_edit` Generator with a **Q8** `LoadSpec` — bf16
/// conditioning (pixel-parity) + Q8 transformer — vs the fork-Q8 decoded edit output. Exercises the
/// `model_edit::load` Q8 wiring end-to-end.
#[test]
#[ignore = "needs the full real Qwen-Image-Edit-2509 model + local Q8 edit golden"]
fn edit_generate_q8_matches_fork() {
    let g = Weights::from_file(EDIT_Q8_GOLDEN).unwrap();
    let spec = LoadSpec::new(WeightsSource::Dir(edit_snapshot())).with_quant(mlx_gen::Quant::Q8);
    let generator = model_edit::load(&spec).unwrap();

    let req = GenerationRequest {
        prompt: "make it autumn".into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(42),
        steps: Some(STEPS as u32),
        guidance: Some(GUIDANCE),
        conditioning: vec![Conditioning::Reference {
            image: synthetic_reference(),
            strength: None,
        }],
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let got = match out {
        GenerationOutput::Images(mut v) => v.swap_remove(0),
        _ => panic!("expected images"),
    };

    let want = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    assert_eq!(got.pixels.len(), want.pixels.len(), "pixel count");
    let differ = got
        .pixels
        .iter()
        .zip(&want.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / got.pixels.len() as f32;
    println!("Q8 edit generate pixels >8 apart: {:.3}%", frac * 100.0);
    // Q8 transformer (byte-identical packing, sc-2342) on the bf16 pixel-parity conditioning path →
    // matches fork-Q8 to the qmm floor, like T2I Q8 (~0.007% px>8).
    assert!(
        frac < 0.02,
        "Q8 edit generate diverges from fork-Q8: {:.3}% px>8",
        frac * 100.0
    );
}

/// sc-2529 (multi-image e2e): the whole `qwen_image_edit` Generator with a **`MultiReference`** of
/// two distinct references — load → text path on `references[0]` → dual-latent encode/concat of
/// **both** refs → multi-grid edit denoise → decode → RGB8 — vs the fork's decoded multi-image edit
/// (`image_paths=[ref1, ref2]`, `cond_image_grid=[(1,h,w),(1,h,w)]`). Confirms the second reference
/// is wired through the dual-latent sequence (the text/VL embeds match the single-image path).
#[test]
#[ignore = "needs the full real Qwen-Image-Edit-2509 model + local multi edit golden"]
fn edit_generate_multi_matches_fork() {
    let g = Weights::from_file(EDIT_MULTI_GOLDEN).unwrap();
    assert_eq!(
        g.require("num_images").unwrap().as_slice::<i32>(),
        &[2],
        "golden must be the 2-image dump (MULTI=1)"
    );
    let spec = LoadSpec::new(WeightsSource::Dir(edit_snapshot()));
    let generator = model_edit::load(&spec).unwrap();

    let req = GenerationRequest {
        prompt: "make it autumn".into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(42),
        steps: Some(STEPS as u32),
        guidance: Some(GUIDANCE),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![synthetic_reference(), synthetic_reference2()],
        }],
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let got = match out {
        GenerationOutput::Images(mut v) => v.swap_remove(0),
        _ => panic!("expected images"),
    };

    let want = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    assert_eq!(got.pixels.len(), want.pixels.len(), "pixel count");
    let differ = got
        .pixels
        .iter()
        .zip(&want.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / got.pixels.len() as f32;
    println!("multi edit generate pixels >8 apart: {:.3}%", frac * 100.0);
    // PIXEL-PARITY with the fork, like single-image edit: both references are VAE-encoded with the
    // PIL-exact resampler and folded into the dual-latent sequence; the f32-activation 60-layer net
    // stays within bf16-rounding (< 8/255).
    assert!(
        frac < 0.001,
        "multi edit generate diverges from the fork: {:.3}% px>8",
        frac * 100.0
    );
}

/// sc-2529 (multi-image e2e, Q8): the multi-reference Generator with a **Q8** `LoadSpec` vs the
/// fork-Q8 multi-image edit golden. Exercises the dual-latent multi-ref concat through the Q8
/// transformer end-to-end.
#[test]
#[ignore = "needs the full real Qwen-Image-Edit-2509 model + local multi Q8 edit golden"]
fn edit_generate_multi_q8_matches_fork() {
    let g = Weights::from_file(EDIT_MULTI_Q8_GOLDEN).unwrap();
    assert_eq!(
        g.require("num_images").unwrap().as_slice::<i32>(),
        &[2],
        "golden must be the 2-image Q8 dump (MULTI=1 QUANTIZE=8)"
    );
    let spec = LoadSpec::new(WeightsSource::Dir(edit_snapshot())).with_quant(mlx_gen::Quant::Q8);
    let generator = model_edit::load(&spec).unwrap();

    let req = GenerationRequest {
        prompt: "make it autumn".into(),
        width: 1024,
        height: 1024,
        count: 1,
        seed: Some(42),
        steps: Some(STEPS as u32),
        guidance: Some(GUIDANCE),
        conditioning: vec![Conditioning::MultiReference {
            images: vec![synthetic_reference(), synthetic_reference2()],
        }],
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let got = match out {
        GenerationOutput::Images(mut v) => v.swap_remove(0),
        _ => panic!("expected images"),
    };

    let want = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    assert_eq!(got.pixels.len(), want.pixels.len(), "pixel count");
    let differ = got
        .pixels
        .iter()
        .zip(&want.pixels)
        .filter(|(a, b)| (**a as i16 - **b as i16).abs() > 8)
        .count();
    let frac = differ as f32 / got.pixels.len() as f32;
    println!(
        "multi Q8 edit generate pixels >8 apart: {:.3}%",
        frac * 100.0
    );
    assert!(
        frac < 0.02,
        "multi Q8 edit generate diverges from fork-Q8: {:.3}% px>8",
        frac * 100.0
    );
}
