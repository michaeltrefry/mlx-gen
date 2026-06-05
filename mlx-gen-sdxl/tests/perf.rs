//! sc-2999 real-weight per-step A/B for the sc-2963 `mx.compile` glue rollout (SDXL base-1.0, fp16).
//!
//! The companion to Wan's `tests/perf.rs`. SDXL is the documented small-win case: its heavy
//! GELU/GEGLU were already `mx.compile`'d in sc-2721, so sc-2963 only compiles the residual 2-op
//! SiLU glue (`x·sigmoid(x)`). sc-2963 proved that SiLU fusion bit-exact in fp16 + f32 in isolation
//! (`lib.rs` unit test, `compile_micro.rs`) but never measured it on the real fp16 UNet. This file
//! times `UNet2DConditionModel::forward` warm, eager (`set_compile_glue(false)`) vs compiled
//! (`set_compile_glue(true)`) on the real CFG-batch (B=2) forward, and asserts the compiled forward
//! is **bit-identical** on the real fp16 weights (`max|Δ| == 0`).
//!
//! The fp16 golden re-gate the story also asks for — confirming the full-UNet render against the
//! MLX-0.31.2 golden is *unmoved* with SiLU fusion on — is covered by running the existing
//! `e2e_real_weights::full_pipeline_generates_fox` (which drives the production `generate`, where the
//! toggle is on). This file is the per-step + bit-exact half.
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-sdxl --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `SDXL_PERF_SIZE` (square px, default 1024) / `SDXL_PERF_BATCH` (default 2).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen_sdxl::{load_unet_dtype, set_compile_glue};
use mlx_rs::{random, Array, Dtype};

fn snapshot() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
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

#[test]
#[ignore = "needs the real SDXL snapshot (SDXL_SNAPSHOT or HF cache)"]
fn sdxl_per_step_compiled_vs_eager() {
    let snap = match snapshot() {
        Some(p) => p,
        None => {
            eprintln!("skip: set SDXL_SNAPSHOT or populate the HF cache for SDXL base-1.0");
            return;
        }
    };
    let size = env_i32("SDXL_PERF_SIZE", 1024);
    let b = env_i32("SDXL_PERF_BATCH", 2); // CFG doubles the UNet batch in production
    let lat = size / 8;
    println!("SDXL base-1.0 fp16: {size}x{size} -> latent [{b},{lat},{lat},4] (CFG B={b})");

    let unet = load_unet_dtype(&snap, Dtype::Float16).expect("load fp16 UNet");

    // Production fp16 inputs (the `StableDiffusionXL(float16=True)` dtype). Timing is value-independent.
    let key = random::key(0).unwrap();
    let f16 = |a: Array| a.as_dtype(Dtype::Float16).unwrap();
    let latents = f16(random::normal::<f32>(&[b, lat, lat, 4], None, None, Some(&key)).unwrap());
    let conditioning = f16(random::normal::<f32>(&[b, 77, 2048], None, None, Some(&key)).unwrap());
    let pooled = f16(random::normal::<f32>(&[b, 1280], None, None, Some(&key)).unwrap());
    let time_ids = Array::from_slice(
        &[size as f32, size as f32, 0.0, 0.0, size as f32, size as f32].repeat(b as usize),
        &[b, 6],
    );
    mlx_rs::transforms::eval([&latents, &conditioning, &pooled, &time_ids]).unwrap();
    let timestep = 700.0f32;

    let run = || {
        unet.forward(&latents, timestep, &conditioning, &pooled, &time_ids)
            .unwrap()
    };

    // --- real-weight bit-exactness (SiLU fusion fp16) ---
    set_compile_glue(false);
    let eager0 = run();
    set_compile_glue(true);
    let comp0 = run();
    set_compile_glue(false);
    assert_eq!(comp0.shape(), eager0.shape(), "eps shape");
    assert_eq!(comp0.dtype(), Dtype::Float16, "fp16 forward must stay f16");
    let max_diff = max_abs_diff(&comp0, &eager0);

    let warmup = 2usize;
    let iters = 8usize;

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
        "SDXL compiled SiLU diverged from eager on the real fp16 UNet"
    );
}
