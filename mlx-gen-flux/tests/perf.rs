//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (FLUX.1).
//!
//! The companion to Wan's `tests/perf.rs`: the sc-2963 rollout proved the compiled glue **bit-exact**
//! on CI fixtures + in-crate helpers and measured the fusion win with the weight-independent
//! `compile_micro` micros, but never ran the `perf.rs`-style A/B on a real multi-GB FLUX.1 checkpoint
//! (the rollout worktree had no weights). This file closes that gap: on the SAME real transformer it
//!
//!   * times the production `FluxTransformer::forward` warm, eager (`set_compile_glue(false)`) vs
//!     compiled (`set_compile_glue(true)`), reporting warm s/step + speedup, and
//!   * asserts the compiled forward is **bit-identical** to the eager one on the real weights
//!     (`max|Δ| == 0` — the real-weight parity proof the fixture/helper gates approximate).
//!
//! FLUX.1 runs the fork's mixed precision (f32 latents/embeds/main-stream, bf16 conditioning), so the
//! synthetic inputs are f32 like the e2e golden path. Timing is value-independent; only the shapes +
//! dtypes (hence the kernels) drive it.
//!
//! Run it (schnell from the HF cache, or point at a snapshot):
//! ```text
//! cargo test --release -p mlx-gen-flux --test perf -- --ignored --nocapture
//! FLUX_VARIANT=dev MLX_GEN_FLUX_SNAPSHOT=<snapshot> \
//!   cargo test --release -p mlx-gen-flux --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `FLUX_PERF_WIDTH` / `FLUX_PERF_HEIGHT` (default 1024×1024).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_flux::config::FluxVariant;
use mlx_gen_flux::loader::load_transformer;
use mlx_gen_flux::transformer::set_compile_glue;
use mlx_rs::{random, Array};

fn variant() -> FluxVariant {
    match std::env::var("FLUX_VARIANT").as_deref() {
        Ok("dev") => FluxVariant::Dev,
        _ => FluxVariant::Schnell,
    }
}

/// Resolve the FLUX.1 snapshot: `MLX_GEN_FLUX_SNAPSHOT` override, else the first snapshot dir in the
/// HF cache for the selected variant. Returns `None` (test skips) if neither exists.
fn snapshot(variant: FluxVariant) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let repo = match variant {
        FluxVariant::Dev => "models--black-forest-labs--FLUX.1-dev",
        FluxVariant::Schnell => "models--black-forest-labs--FLUX.1-schnell",
    };
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

fn env_u32(var: &str, default: u32) -> u32 {
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
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>()
}

#[test]
#[ignore = "needs real FLUX.1 weights (MLX_GEN_FLUX_SNAPSHOT or HF cache)"]
fn flux1_per_step_compiled_vs_eager() {
    let var = variant();
    let snap = match snapshot(var) {
        Some(p) => p,
        None => {
            eprintln!("skip: set MLX_GEN_FLUX_SNAPSHOT or populate the HF cache for {var:?}");
            return;
        }
    };
    let width = env_u32("FLUX_PERF_WIDTH", 1024);
    let height = env_u32("FLUX_PERF_HEIGHT", 1024);
    let img_seq = ((height / 16) * (width / 16)) as i32;
    let txt_seq = var.max_sequence_length() as i32;
    let guidance = if var.supports_guidance() { 3.5 } else { 0.0 };
    println!(
        "FLUX.1 {var:?}: {height}x{width} -> img_seq={img_seq}, txt_seq={txt_seq}, guidance={guidance}"
    );

    let t = load_transformer(&snap, var).expect("load FLUX.1 transformer");

    // Synthetic but production-shaped f32 inputs (FLUX runs an f32 main stream; sc-2787). Timing is
    // value-independent.
    let key = random::key(0).unwrap();
    let init = random::normal::<f32>(&[1, img_seq, 64], None, None, Some(&key)).unwrap();
    let pe = random::normal::<f32>(&[1, txt_seq, 4096], None, None, Some(&key)).unwrap();
    let pooled = random::normal::<f32>(&[1, 768], None, None, Some(&key)).unwrap();
    mlx_rs::transforms::eval([&init, &pe, &pooled]).unwrap();
    let sigma = 1.0f32;

    let run = || {
        t.forward(&init, &pe, &pooled, sigma, guidance, width, height)
            .unwrap()
    };

    // --- real-weight bit-exactness (the parity contract on real weights) ---
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "v shape");
    let max_diff = max_abs_diff(&comp0, &eager0);

    let warmup = 2usize;
    let iters = 6usize;

    // --- eager ---
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

    // --- compiled glue ---
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
        "[warm s/step] eager={eag:.4}  compiled-glue={cmp:.4}  speedup={:.3}×  \
         (recovers {:.1}% of step)  max|Δ| compiled-vs-eager={max_diff:.3e}",
        eag / cmp,
        (eag - cmp) / eag * 100.0
    );
    assert_eq!(
        max_diff, 0.0,
        "FLUX.1 compiled glue diverged from eager on real weights"
    );
}
