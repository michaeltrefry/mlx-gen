//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (Qwen-Image, T2I + Edit).
//!
//! The companion to Wan's `tests/perf.rs`. sc-2963 proved the compiled glue (modulate / gated /
//! tanh-GELU FFN / RoPE) **bit-exact** via in-crate `#[cfg(test)] sc2963` helper gates and measured
//! the fusion win with `compile_micro`, but never ran the `perf.rs`-style A/B on the real ~40 GB
//! 60-layer transformer. This file closes that gap on the SAME real transformer for **both** paths:
//! it times `QwenTransformer::forward` warm, eager (`set_compile_glue(false)`) vs compiled
//! (`set_compile_glue(true)`), and asserts the compiled forward is **bit-identical** on the real
//! weights (`max|Δ| == 0`). The Edit path additionally exercises the `zero_cond_t` dual-latent
//! `modulate_index` route (cond_grids non-empty).
//!
//! Qwen runs mixed precision: f32 latents, bf16 text embeds (`model.rs`). Timing is value-independent.
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-qwen-image --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `QWEN_PERF_SIZE` (square px, default 1024) / `QWEN_PERF_TXT` (text seq, 128).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_qwen_image::loader::{load_transformer, load_transformer_edit};
use mlx_gen_qwen_image::transformer::{set_compile_glue, QwenTransformer};
use mlx_rs::{random, Array, Dtype};

fn snapshot(env: &str, repo: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env) {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

fn env_i32(var: &str, default: i32) -> i32 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn timed(eval: &[&Array], start: Instant) -> f64 {
    mlx_rs::transforms::eval(eval.iter().copied()).unwrap();
    start.elapsed().as_secs_f64()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let a = a.as_dtype(Dtype::Float32).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap();
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(&a, &b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>()
}

/// Run the warm A/B + real-weight bit-exact check for one forward closure. `label` names the path.
fn ab<F: Fn() -> Array>(label: &str, run: F) {
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "{label} v shape");
    let max_diff = max_abs_diff(&comp0, &eager0);

    let warmup = 2usize;
    let iters = 6usize;

    set_compile_glue(false);
    let mut eager = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let v = run();
        let dt = timed(&[&v], start);
        if i >= warmup {
            eager.push(dt);
        }
    }

    set_compile_glue(true);
    let mut compiled = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let v = run();
        let dt = timed(&[&v], start);
        if i >= warmup {
            compiled.push(dt);
        }
    }
    set_compile_glue(false);

    let eag = median(eager);
    let cmp = median(compiled);
    println!(
        "[{label} warm s/step] eager={eag:.4}  compiled-glue={cmp:.4}  speedup={:.3}×  \
         (recovers {:.1}% of step)  max|Δ| compiled-vs-eager={max_diff:.3e}",
        eag / cmp,
        (eag - cmp) / eag * 100.0
    );
    assert_eq!(
        max_diff, 0.0,
        "Qwen {label} compiled glue diverged from eager on real weights"
    );
}

/// f32 packed image latents `[1, seq, 64]` and bf16 text embeds `[1, txt, 3584]` at production dtype.
fn inputs(img_seq: i32, txt_seq: i32) -> (Array, Array) {
    let key = random::key(0).unwrap();
    let hidden = random::normal::<f32>(&[1, img_seq, 64], None, None, Some(&key)).unwrap();
    let encoder = random::normal::<f32>(&[1, txt_seq, 3584], None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    mlx_rs::transforms::eval([&hidden, &encoder]).unwrap();
    (hidden, encoder)
}

#[test]
#[ignore = "needs real Qwen-Image weights (QWEN_IMAGE_SNAPSHOT or HF cache)"]
fn qwen_t2i_per_step_compiled_vs_eager() {
    let snap = match snapshot("QWEN_IMAGE_SNAPSHOT", "models--Qwen--Qwen-Image") {
        Some(p) => p,
        None => {
            eprintln!("skip: set QWEN_IMAGE_SNAPSHOT or populate the HF cache for Qwen-Image");
            return;
        }
    };
    let size = env_i32("QWEN_PERF_SIZE", 1024);
    let txt_seq = env_i32("QWEN_PERF_TXT", 128);
    let lat = size / 16; // patched grid (VAE/8 then 2×2 patch)
    let img_seq = lat * lat;
    println!(
        "Qwen-Image T2I: {size}x{size} -> img_seq={img_seq} (grid {lat}x{lat}), txt_seq={txt_seq}"
    );

    let t: QwenTransformer = load_transformer(&snap).expect("load Qwen-Image transformer");
    let (hidden, encoder) = inputs(img_seq, txt_seq);
    let (lat, sigma) = (lat as usize, 1.0f32);
    ab("T2I", || {
        t.forward(&hidden, &encoder, None, sigma, lat, lat, &[])
            .unwrap()
    });
}

#[test]
#[ignore = "needs real Qwen-Image-Edit-2511 weights (QWEN_IMAGE_EDIT_SNAPSHOT or HF cache)"]
fn qwen_edit_per_step_compiled_vs_eager() {
    let snap = match snapshot(
        "QWEN_IMAGE_EDIT_SNAPSHOT",
        "models--Qwen--Qwen-Image-Edit-2511",
    ) {
        Some(p) => p,
        None => {
            eprintln!("skip: set QWEN_IMAGE_EDIT_SNAPSHOT or populate the HF cache for Qwen-Image-Edit-2511");
            return;
        }
    };
    let size = env_i32("QWEN_PERF_SIZE", 1024);
    let txt_seq = env_i32("QWEN_PERF_TXT", 128);
    let lat = (size / 16) as usize; // noise grid; one same-size reference (cond_grids=[(lat,lat)])
    let noise_seq = (lat * lat) as i32;
    let img_seq = noise_seq * 2; // noise + one reference, concatenated (dual-latent edit)
    println!("Qwen-Image-Edit: {size}x{size} -> img_seq={img_seq} (noise {lat}x{lat} + ref {lat}x{lat}), txt_seq={txt_seq}");

    let t: QwenTransformer =
        load_transformer_edit(&snap).expect("load Qwen-Image-Edit transformer");
    let (hidden, encoder) = inputs(img_seq, txt_seq);
    let sigma = 1.0f32;
    let cond_grids = [(lat, lat)];
    ab("Edit", || {
        t.forward(&hidden, &encoder, None, sigma, lat, lat, &cond_grids)
            .unwrap()
    });
}
