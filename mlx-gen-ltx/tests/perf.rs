//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (LTX-2.3, Q8/Q4).
//!
//! The companion to Wan's `tests/perf.rs`. LTX is the rollout's **expected biggest win**: the
//! `compile_micro` showed the tanh-GELU FFN fusing 10.85× at video sequence (the 285 MB f32/bf16 FFN
//! is the largest in the fleet, and it grows with the video token count). sc-2963 proved the glue
//! bit-exact via the in-crate `#[cfg(test)] sc2963` helper gate but never ran the `perf.rs`-style A/B
//! on the real ~19 GB Q8 / ~11 GB Q4 checkpoint. This file closes that gap on the SAME real DiT:
//!
//!   * `ltx_video_per_step_compiled_vs_eager` times `LtxDiT::forward` warm, eager vs compiled, at the
//!     two production geometries the story names — 512²×129f (latent 17×16×16 → seq 4352) and
//!     1280×720 (latent 17×22×40 → seq 14960, the ~245M-elem FFN) — for every present quant dir
//!     (Q8 + Q4), asserting `max|Δ| == 0` on the real weights at each.
//!   * `ltx_av_compiled_vs_eager` runs the `AvDiT` joint (video **+ audio**) forward over the
//!     committed AV golden's real inputs, eager vs compiled, asserting both velocities are
//!     bit-identical — the audio/cross-modal blocks share the same crate-global glue functions.
//!
//! Production runs `Precision::quant_bf16` (bf16 activations × quantized weights). Timing is value-
//! independent; the seq + dtype drive the kernels.
//!
//! Run it (Q8 from the prod model dir, Q4 auto-detected alongside):
//! ```text
//! cargo test --release -p mlx-gen-ltx --test perf -- --ignored --nocapture
//! LTX_BASE_DIR=… LTX_BASE_Q4_DIR=… cargo test --release -p mlx-gen-ltx --test perf -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, SplitModel};
use mlx_gen_ltx::positions::create_position_grid;
use mlx_gen_ltx::set_compile_glue;
use mlx_gen_ltx::transformer::{AvDiT, LtxDiT, Precision};
use mlx_rs::{random, Array, Dtype};

const AV_GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_av_dit_golden_bf16.safetensors"
);

/// Production model dirs to A/B over: `(label, env_override, default_subdir)`.
fn model_dirs() -> Vec<(&'static str, PathBuf)> {
    let prod = std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join("Library/Application Support/SceneWorks/data/models/mlx"));
    let resolve = |env: &str, sub: &str| -> Option<PathBuf> {
        if let Ok(p) = std::env::var(env) {
            return Some(PathBuf::from(p));
        }
        prod.as_ref().map(|p| p.join(sub))
    };
    [
        ("Q8", resolve("LTX_BASE_DIR", "ltx_2_3_base_q8")),
        ("Q4", resolve("LTX_BASE_Q4_DIR", "ltx_2_3_base_q4")),
    ]
    .into_iter()
    .filter_map(|(l, p)| {
        p.filter(|p| p.join("transformer.safetensors").exists())
            .map(|p| (l, p))
    })
    .collect()
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

fn bf16(a: Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

#[test]
#[ignore = "needs ltx_2_3_base_q8/q4 transformer.safetensors (~11-19 GB)"]
fn ltx_video_per_step_compiled_vs_eager() {
    let dirs = model_dirs();
    if dirs.is_empty() {
        eprintln!(
            "skip: set LTX_BASE_DIR (and/or LTX_BASE_Q4_DIR) or install the prod LTX model dirs"
        );
        return;
    }
    // (label, frames, height, width) in LATENT/patch units: 512²×129f and 1280×720.
    let geoms = [
        ("512²×129f", 17usize, 16usize, 16usize),
        ("1280×720", 17, 22, 40),
    ];
    let warmup = 1usize;
    let iters = 3usize;

    for (qlabel, dir) in &dirs {
        let cfg = LtxConfig::from_model_dir(dir).expect("embedded_config.json");
        let split = SplitModel::from_model_dir(dir).expect("split_model.json");
        let prec = Precision::quant_bf16(split.bits, split.group);
        let w = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
        let dit = LtxDiT::from_weights(&w, &cfg, prec).expect("build LtxDiT");
        let inner = cfg.inner_dim();

        for (glabel, f, h, ww) in geoms {
            let s = (f * h * ww) as i32;
            let key = random::key(0).unwrap();
            let latent = bf16(random::normal::<f32>(&[1, s, 128], None, None, Some(&key)).unwrap());
            let ts = bf16(Array::from_slice(&vec![0.7f32; s as usize], &[1, s]));
            let context =
                bf16(random::normal::<f32>(&[1, 128, inner], None, None, Some(&key)).unwrap());
            let positions = create_position_grid(1, f, h, ww); // f32, (1,3,S,2)
            mlx_rs::transforms::eval([&latent, &ts, &context, &positions]).unwrap();

            let run = || {
                dit.forward(&latent, &ts, &context, None, &positions)
                    .unwrap()
            };

            set_compile_glue(false);
            let eager0 = run();
            set_compile_glue(true);
            let comp0 = run();
            set_compile_glue(false);
            assert_eq!(comp0.shape(), eager0.shape(), "velocity shape");
            let max_diff = max_abs_diff(&comp0, &eager0);

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
                "[LTX {qlabel} {glabel} seq={s}] eager={eag:.4}  compiled-glue={cmp:.4}  \
                 speedup={:.3}×  (recovers {:.1}% of step)  max|Δ|={max_diff:.3e}",
                eag / cmp,
                (eag - cmp) / eag * 100.0
            );
            assert_eq!(
                max_diff, 0.0,
                "LTX {qlabel} {glabel} compiled glue diverged from eager on real weights"
            );
        }
    }
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors + committed AV golden fixture"]
fn ltx_av_compiled_vs_eager() {
    let dirs = model_dirs();
    let dir = match dirs
        .iter()
        .find(|(l, _)| *l == "Q8")
        .or_else(|| dirs.first())
    {
        Some((_, d)) => d.clone(),
        None => {
            eprintln!("skip: need an LTX model dir for the AvDiT audio-path parity check");
            return;
        }
    };
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let split = SplitModel::from_model_dir(&dir).expect("split_model.json");
    let prec = Precision::quant_bf16(split.bits, split.group);
    let w = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit = AvDiT::from_weights(&w, &cfg, prec).expect("build AvDiT");
    let g = Weights::from_file(AV_GOLDEN_BF16).expect("AV golden fixture (committed)");

    let run = || {
        dit.forward(
            g.require("video_latent").unwrap(),
            g.require("video_timestep").unwrap(),
            g.require("video_context").unwrap(),
            None,
            g.require("video_positions").unwrap(),
            g.require("audio_latent").unwrap(),
            g.require("audio_timestep").unwrap(),
            g.require("audio_context").unwrap(),
            None,
            g.require("audio_positions").unwrap(),
        )
        .expect("av dit forward")
    };

    set_compile_glue(false);
    let (ev, ea) = run();
    set_compile_glue(true);
    let (cv, ca) = run();
    set_compile_glue(false);

    let dv = max_abs_diff(&cv, &ev);
    let da = max_abs_diff(&ca, &ea);
    println!("[LTX AvDiT compiled-vs-eager] video max|Δ|={dv:.3e}  audio max|Δ|={da:.3e}");
    assert_eq!(
        dv, 0.0,
        "LTX AvDiT video glue diverged from eager on real weights"
    );
    assert_eq!(
        da, 0.0,
        "LTX AvDiT audio glue diverged from eager on real weights"
    );
}
