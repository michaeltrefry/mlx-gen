//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (FLUX.2 klein-9b).
//!
//! The companion to Wan's `tests/perf.rs`. sc-2963 proved the compiled glue **bit-exact** via the
//! committed `transformer_golden` fixture (`compile_parity.rs`) and measured the fusion win with the
//! weight-independent `compile_micro` micros, but never ran the `perf.rs`-style A/B on the real
//! ~18 GB klein-9b checkpoint. This file closes that gap on the SAME real transformer: it times
//! `Flux2Transformer::forward` warm, eager (`set_compile_glue(false)`) vs compiled
//! (`set_compile_glue(true)`), and asserts the compiled forward is **bit-identical** on the real
//! weights (`max|Δ| == 0`).
//!
//! Run it (klein-9b from the HF cache, or point at a snapshot):
//! ```text
//! cargo test --release -p mlx-gen-flux2 --test perf -- --ignored --nocapture
//! MLX_GEN_FLUX2_SNAPSHOT=<snapshot> cargo test --release -p mlx-gen-flux2 --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `FLUX2_PERF_WIDTH` / `FLUX2_PERF_HEIGHT` (default 1024×1024).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_flux2::config::Flux2Config;
use mlx_gen_flux2::loader::load_transformer;
use mlx_gen_flux2::pipeline::{create_noise, prepare_grid_ids, prepare_text_ids};
use mlx_gen_flux2::transformer::set_compile_glue;
use mlx_rs::{random, Array};

fn snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
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
#[ignore = "needs real FLUX.2-klein-9b weights (MLX_GEN_FLUX2_SNAPSHOT or HF cache)"]
fn flux2_per_step_compiled_vs_eager() {
    let snap = match snapshot() {
        Some(p) => p,
        None => {
            eprintln!(
                "skip: set MLX_GEN_FLUX2_SNAPSHOT or populate the HF cache for FLUX.2-klein-9b"
            );
            return;
        }
    };
    let cfg = Flux2Config::klein_9b();
    let width = env_u32("FLUX2_PERF_WIDTH", 1024);
    let height = env_u32("FLUX2_PERF_HEIGHT", 1024);
    let (lat_h, lat_w) = ((height / 16) as usize, (width / 16) as usize);
    let txt_seq = cfg.max_sequence_length;
    println!(
        "FLUX.2 klein-9b: {height}x{width} -> seq={}, txt_seq={txt_seq}, joint_dim={}",
        lat_h * lat_w,
        cfg.joint_attention_dim
    );

    let t = load_transformer(&snap).expect("load FLUX.2 transformer");

    // Production-shaped f32 inputs (FLUX.2 runs an f32 latent stream; model.rs). Timing is
    // value-independent; the position ids + encoder shape drive the kernels.
    let key = random::key(0).unwrap();
    let hidden = create_noise(0, width, height, cfg.in_channels).expect("noise");
    let encoder = random::normal::<f32>(
        &[1, txt_seq as i32, cfg.joint_attention_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let img_ids = prepare_grid_ids(lat_h, lat_w, 0);
    let txt_ids = prepare_text_ids(txt_seq);
    mlx_rs::transforms::eval([&hidden, &encoder, &img_ids, &txt_ids]).unwrap();
    let timestep = 500.0f32;

    let run = || {
        t.forward(&hidden, &encoder, &img_ids, &txt_ids, timestep)
            .unwrap()
    };

    // --- real-weight bit-exactness ---
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "v shape");
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
        "[warm s/step] eager={eag:.4}  compiled-glue={cmp:.4}  speedup={:.3}×  \
         (recovers {:.1}% of step)  max|Δ| compiled-vs-eager={max_diff:.3e}",
        eag / cmp,
        (eag - cmp) / eag * 100.0
    );
    assert_eq!(
        max_diff, 0.0,
        "FLUX.2 compiled glue diverged from eager on real weights"
    );
}
