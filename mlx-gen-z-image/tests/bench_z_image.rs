//! sc-2352: Z-Image-turbo performance benchmarks (Rust mlx-gen), to compare against the frozen
//! Python mflux fork (`tools/bench_z_image_fork.py`).
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache. Run with:
//!   cargo test -p mlx-gen-z-image --release --test bench_z_image -- --ignored --nocapture
//!
//! Two metrics, mirroring the fork script so the numbers line up:
//!   * end-to-end `generate()` wall-clock (encode + denoise + VAE), the latency a user feels;
//!   * pure DiT per-step time with an eval forced each step (the fork `mx.eval`s per step too),
//!     which isolates the dominant cost and exposes the slower first (graph-build) step.
//!
//! MLX is lazy, so end-to-end timing is honest only because `generate()` materializes the pixels
//! (`decoded_to_image` reads the buffer). mlx-rs 0.25 exposes no peak-memory API, so memory is not
//! reported here — see the fork measurements in memory for the bf16 envelope.

use std::time::Instant;

use mlx_gen::{FlowMatchEuler, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource};
use mlx_gen_z_image::{
    create_noise, load_text_encoder, load_tokenizer, load_transformer, slice_valid,
};
use mlx_rs::fast::scaled_dot_product_attention;
use mlx_rs::ops::matmul;
use mlx_rs::{random, Array, Dtype};

/// (width, height) sweep — turbo's typical operating points.
const SIZES: &[(u32, u32)] = &[(256, 256), (512, 512), (1024, 1024)];
const STEPS: usize = 4;
const RUNS: usize = 3;
const PROMPT: &str = "a fox";

mod common;
use common::snapshot;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn fmt3(v: &[f64]) -> Vec<String> {
    v.iter().map(|t| format!("{t:.3}")).collect()
}

fn bf16_normal(shape: &[i32], seed: u64) -> Array {
    let key = random::key(seed).unwrap();
    random::normal::<f32>(shape, None, None, Some(&key))
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap()
}

/// Op-level microbench: the two ops that dominate the DiT (attention SDPA + the big GEMM), to
/// compare the MLX C++ core that mlx-rs bundles against a reference `mlx` (Python) of a different
/// version. If these diverge, the gap is the MLX kernel version, not the port's Rust.
#[test]
#[ignore = "diagnostic microbench"]
fn bench_ops_micro() {
    let bench = |f: &dyn Fn() -> Array, n: usize| -> f64 {
        for _ in 0..5 {
            f().eval().unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..n {
            f().eval().unwrap();
        }
        t0.elapsed().as_secs_f64() / n as f64 * 1000.0
    };
    println!("\n# Op microbench — MLX core bundled by mlx-rs (v0.25.1), bf16");
    for s in [1024i32, 4096] {
        let (b, h, d) = (1, 30, 128);
        let q = bf16_normal(&[b, h, s, d], 1);
        let k = bf16_normal(&[b, h, s, d], 2);
        let v = bf16_normal(&[b, h, s, d], 3);
        q.eval().unwrap();
        k.eval().unwrap();
        v.eval().unwrap();
        let scale = (d as f32).powf(-0.5);
        let t_sdpa = bench(
            &|| scaled_dot_product_attention(&q, &k, &v, scale, None, None).unwrap(),
            30,
        );

        let a = bf16_normal(&[s, 3840], 4);
        let wmat = bf16_normal(&[3840, 3840], 5);
        a.eval().unwrap();
        wmat.eval().unwrap();
        let t_mm = bench(&|| matmul(&a, &wmat).unwrap(), 30);
        println!("S={s}: sdpa {t_sdpa:.3}ms  matmul[{s},3840]x[3840,3840] {t_mm:.3}ms");
    }
}

#[test]
#[ignore = "needs real Z-Image weights"]
fn bench_generate_wall_clock() {
    let g = mlx_gen::load(
        "z_image_turbo",
        &LoadSpec::new(WeightsSource::Dir(snapshot())),
    )
    .unwrap();
    println!(
        "\n# Z-Image-turbo end-to-end generate() wall-clock — {STEPS} steps, bf16, median of {RUNS} runs (after 1 warmup)"
    );
    for &(w, h) in SIZES {
        let req = |seed: u64| GenerationRequest {
            prompt: PROMPT.into(),
            width: w,
            height: h,
            seed: Some(seed),
            steps: Some(STEPS as u32),
            ..Default::default()
        };
        // Warmup (graph build / kernel compile) — not timed.
        let _ = g.generate(&req(0), &mut |_| {}).unwrap();

        let mut times = Vec::with_capacity(RUNS);
        for r in 0..RUNS {
            let t0 = Instant::now();
            let out = g.generate(&req(r as u64 + 1), &mut |_| {}).unwrap();
            match out {
                GenerationOutput::Images(v) => assert_eq!(v.len(), 1),
                other => panic!("expected images, got {other:?}"),
            }
            times.push(t0.elapsed().as_secs_f64());
        }
        let med = median(times.clone());
        println!(
            "{w}x{h}: {med:.3}s/image  (amortized {:.3}s/step)  runs={:?}",
            med / STEPS as f64,
            fmt3(&times)
        );
    }
}

#[test]
#[ignore = "needs real Z-Image weights"]
fn bench_denoise_per_step() {
    let snap = snapshot();
    // Encode the prompt once — content doesn't affect DiT step time, only the (fixed) cap shape.
    let tok = load_tokenizer(&snap).unwrap();
    let te = load_text_encoder(&snap).unwrap();
    let t = tok.tokenize(PROMPT).unwrap();
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&t);
    let num_valid: i32 = attention_mask.as_slice::<i32>().iter().sum();
    let cap = slice_valid(&te.forward(&input_ids, &attention_mask).unwrap(), num_valid)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let transformer = load_transformer(&snap).unwrap();

    println!("\n# Z-Image-turbo DiT per-step — eval forced each step, bf16 (first step includes graph build)");
    for &(w, h) in SIZES {
        let scheduler = FlowMatchEuler::for_image(STEPS, w, h);

        // Warmup the full loop once (compile kernels for this resolution).
        {
            let mut lat = create_noise(7, w, h)
                .unwrap()
                .as_dtype(Dtype::Bfloat16)
                .unwrap();
            for s in 0..scheduler.num_steps() {
                let v = transformer
                    .forward(&lat, scheduler.timestep(s), &cap)
                    .unwrap();
                lat = scheduler.step(&lat, &v, s).unwrap();
            }
            lat.eval().unwrap();
        }

        // Timed: eval after each step to attribute time to that step.
        let mut lat = create_noise(8, w, h)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        lat.eval().unwrap();
        let mut step_times = Vec::with_capacity(scheduler.num_steps());
        for s in 0..scheduler.num_steps() {
            let t0 = Instant::now();
            let v = transformer
                .forward(&lat, scheduler.timestep(s), &cap)
                .unwrap();
            lat = scheduler.step(&lat, &v, s).unwrap();
            lat.eval().unwrap();
            step_times.push(t0.elapsed().as_secs_f64());
        }
        // Mean of the steady-state steps (drop step 0's graph-build overhead).
        let steady: f64 = step_times[1..].iter().sum::<f64>() / (step_times.len() - 1) as f64;
        println!(
            "{w}x{h}: {steady:.3}s/step steady (steps 1+), per-step={:?}",
            fmt3(&step_times)
        );
    }
}
