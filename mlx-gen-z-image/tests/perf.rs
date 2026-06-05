//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (Z-Image turbo + control).
//!
//! The companion to Wan's `tests/perf.rs`. sc-2963 proved the compiled glue (SwiGLU / gated / RoPE,
//! plus the control `add_hint`) **bit-exact** via the committed `z_transformer` / `z_control_transformer`
//! fixtures (`compile_parity.rs`) and measured the win with `compile_micro`, but never ran the
//! `perf.rs`-style A/B on the real ~12 GB checkpoint. This file closes that gap on the SAME real
//! transformers for **both** the turbo base path and the control path. It times the production
//! `forward` warm, eager (`set_compile_glue(false)`) vs compiled (`set_compile_glue(true)`), and
//! asserts the compiled forward is **bit-identical** on the real weights (`max|Δ| == 0`).
//!
//! The control test runs the sc-2720 mixed precision (bf16 base + cap, **f32** `control_context`) and
//! also asserts the compiled output keeps the f32 dtype the f32 hint promotes it to — the dtype-flow
//! contract the glue must not perturb.
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-z-image --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `ZIMAGE_PERF_SIZE` (square px, default 1024) / `ZIMAGE_PERF_CAP` (cap len, 64).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::WeightsSource;
use mlx_gen_z_image::{load_control_transformer, load_transformer, set_compile_glue};
use mlx_rs::{random, Array, Dtype};

fn snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
}

/// The Fun-Controlnet-Union checkpoint: env `CONTROL_WEIGHTS`, else the first `.safetensors` in the
/// HF cache. `None` ⇒ the control test skips.
fn control_source() -> Option<WeightsSource> {
    if let Ok(p) = std::env::var("CONTROL_WEIGHTS") {
        return Some(WeightsSource::File(PathBuf::from(p)));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1/snapshots",
    );
    let file = std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .flat_map(|d| {
            std::fs::read_dir(d)
                .into_iter()
                .flatten()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
        })
        .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))?;
    Some(WeightsSource::File(file))
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

/// Warm A/B + real-weight bit-exact check for one forward closure. Returns the eager output dtype.
fn ab<F: Fn() -> Array>(label: &str, run: F) -> Dtype {
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "{label} v shape");
    assert_eq!(comp0.dtype(), eager0.dtype(), "{label} dtype preserved");
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
        "Z-Image {label} compiled glue diverged from eager on real weights"
    );
    eager0.dtype()
}

#[test]
#[ignore = "needs real Z-Image-Turbo weights (ZIMAGE_SNAPSHOT or HF cache)"]
fn zimage_turbo_per_step_compiled_vs_eager() {
    let snap = match snapshot() {
        Some(p) => p,
        None => {
            eprintln!("skip: set ZIMAGE_SNAPSHOT or populate the HF cache for Z-Image-Turbo");
            return;
        }
    };
    let size = env_i32("ZIMAGE_PERF_SIZE", 1024);
    let cap_len = env_i32("ZIMAGE_PERF_CAP", 64);
    let lat = size / 8;
    println!("Z-Image-Turbo: {size}x{size} -> latent [16,1,{lat},{lat}], cap_len={cap_len} (bf16)");

    let t = load_transformer(&snap).expect("load Z-Image transformer");

    // Production dtype: bf16 latents + bf16 cap (model.rs casts both to bf16).
    let key = random::key(0).unwrap();
    let bf16 = |a: Array| a.as_dtype(Dtype::Bfloat16).unwrap();
    let x = bf16(random::normal::<f32>(&[16, 1, lat, lat], None, None, Some(&key)).unwrap());
    let cap = bf16(random::normal::<f32>(&[cap_len, 2560], None, None, Some(&key)).unwrap());
    mlx_rs::transforms::eval([&x, &cap]).unwrap();

    let dt = ab("turbo", || t.forward(&x, 0.7, &cap).unwrap());
    // The forward output is f32 even with bf16 inputs: the timestep embedding is f32 and the final
    // `-1·velocity` is an f32 array, so the adaLN modulation promotes the stream from block 0 (the
    // weights stay bf16 — bf16-weight × f32-activation GEMMs). The contract sc-2963 must hold is
    // dtype-*preservation* under compile (compiled==eager dtype), already asserted inside `ab`.
    println!("Z-Image turbo forward output dtype: {dt:?}");
}

#[test]
#[ignore = "needs real Z-Image-Turbo + Fun-Controlnet weights (HF cache or env overrides)"]
fn zimage_control_per_step_compiled_vs_eager() {
    let (snap, ctrl) = match (snapshot(), control_source()) {
        (Some(s), Some(c)) => (s, c),
        _ => {
            eprintln!("skip: need Z-Image-Turbo + Fun-Controlnet-Union weights (HF cache or env)");
            return;
        }
    };
    let size = env_i32("ZIMAGE_PERF_SIZE", 1024);
    let cap_len = env_i32("ZIMAGE_PERF_CAP", 64);
    let lat = size / 8;
    println!("Z-Image-Control: {size}x{size} -> latent [16,1,{lat},{lat}] bf16 + control_context [33,1,{lat},{lat}] f32 (sc-2720 mixed precision)");

    let t = load_control_transformer(&snap, &ctrl).expect("load Z-Image control transformer");

    let key = random::key(0).unwrap();
    let bf16 = |a: Array| a.as_dtype(Dtype::Bfloat16).unwrap();
    // sc-2720 mixed precision: bf16 base latents + cap, f32 control_context.
    let x = bf16(random::normal::<f32>(&[16, 1, lat, lat], None, None, Some(&key)).unwrap());
    let cap = bf16(random::normal::<f32>(&[cap_len, 2560], None, None, Some(&key)).unwrap());
    let cc = random::normal::<f32>(&[33, 1, lat, lat], None, None, Some(&key)).unwrap();
    mlx_rs::transforms::eval([&x, &cap, &cc]).unwrap();

    let dt = ab("control", || {
        t.forward(&x, 0.7, &cap, Some(&cc), 1.0).unwrap()
    });
    // The control output is f32 (both the f32 control_context and the base stream's f32 timestep
    // path; sc-2720 mixed precision). The contract sc-2963 must hold is dtype-*preservation* under
    // compile (compiled==eager dtype, asserted inside `ab`) — i.e. the f32 control hint path is not
    // perturbed by the fused `add_hint`.
    assert_eq!(
        dt,
        Dtype::Float32,
        "control forward output is f32 (mixed-precision path)"
    );
}
