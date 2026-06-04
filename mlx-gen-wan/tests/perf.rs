//! sc-2853 per-step DiT perf A/B (`#[ignore]` — needs a real converted A14B expert checkpoint).
//!
//! Measures the **warm per-step transformer cost** at the small-sequence regime where the port
//! lagged (480p×25f, seq ~10920), comparing on **identical real weights + machine**:
//!
//!   * **legacy** — the pre-sc-2853 path: two sequential B=1 forwards (cond, then uncond), each
//!     recomputing RoPE cos/sin + all blocks' cross-attention K/V (`WanTransformer::forward`);
//!   * **cached** — the sc-2853 path: one batched **B=2** forward over the stacked [cond, uncond]
//!     latent, reusing per-generate RoPE + cross-K/V caches (`forward_cached` + `prepare_*`).
//!
//! The A/B isolates exactly the two optimizations (batching + step-caching). Output is **proven
//! bit-identical** by `batched_forward.rs` (CI) + `s6_real_parity.rs` (real e2e), so this file only
//! measures wall-clock; it asserts the cached path is no slower than legacy at small seq.
//!
//! Run it:
//! ```text
//! WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
//!   cargo test --release -p mlx-gen-wan --test perf -- --ignored --nocapture
//! ```
//! Override geometry with `WAN_PERF_FRAMES` / `WAN_PERF_HEIGHT` / `WAN_PERF_WIDTH` (default 25/480/832).

use std::path::PathBuf;
use std::time::Instant;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{latent_shape, seq_len};
use mlx_gen_wan::WanTransformer;
use mlx_rs::ops::concatenate_axis;
use mlx_rs::{random, Array};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy().to_string();
        match s.strip_prefix("~/") {
            Some(rest) => match std::env::var_os("HOME") {
                Some(home) => PathBuf::from(format!("{}/{rest}", home.to_string_lossy())),
                None => PathBuf::from(s),
            },
            None => PathBuf::from(s),
        }
    })
}

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// Run `f`, force-evaluate its output arrays, and return the elapsed seconds.
fn timed(label_eval: &[&Array], start: Instant) -> f64 {
    mlx_rs::transforms::eval(label_eval.iter().copied()).unwrap();
    start.elapsed().as_secs_f64()
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
#[ignore = "needs a real converted Wan2.2-A14B expert checkpoint (WAN_A14B_MODEL_DIR)"]
fn wan_a14b_per_step_cached_vs_legacy() {
    let model_dir = match env_path("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR to the converted A14B model dir");
            return;
        }
    };

    let cfg = WanModelConfig::from_model_dir(&model_dir).expect("read config.json");
    let frames = env_usize("WAN_PERF_FRAMES", 25);
    let height = env_usize("WAN_PERF_HEIGHT", 480) as u32;
    let width = env_usize("WAN_PERF_WIDTH", 832) as u32;
    let lat = latent_shape(frames, height, width, cfg.vae_z_dim, cfg.vae_stride);
    let sl = seq_len(lat, cfg.patch_size);
    println!(
        "geometry: {frames}f {height}x{width} → latent {:?}, seq_len={sl}",
        lat
    );

    // One real expert is enough for a per-step A/B (both experts run the same dense forward).
    let w = Weights::from_file(model_dir.join("high_noise_model.safetensors")).expect("expert");
    let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");

    // Synthetic but realistically-shaped inputs (timing is input-value-independent).
    let key = random::key(0).unwrap();
    let latent = random::normal::<f32>(&lat[..], None, None, Some(&key)).unwrap();
    let raw_ctx = random::normal::<f32>(
        &[cfg.text_len as i32, cfg.text_dim as i32],
        None,
        None,
        Some(&key),
    )
    .unwrap();
    let ctx_cond = dit.embed_text(&raw_ctx).unwrap();
    let ctx_uncond = dit.embed_text(&raw_ctx).unwrap();
    mlx_rs::transforms::eval([&latent, &ctx_cond, &ctx_uncond]).unwrap();

    let grid = dit.patch_grid(&latent);
    let t = 833.0f32;

    let warmup = 2usize;
    let iters = 6usize;

    // --- legacy: two sequential B=1 forwards, each recomputing RoPE + cross-K/V ---
    let mut legacy = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let cond = dit.forward(&latent, t, &ctx_cond).unwrap();
        let uncond = dit.forward(&latent, t, &ctx_uncond).unwrap();
        let dt = timed(&[&cond, &uncond], start);
        if i >= warmup {
            legacy.push(dt);
        }
    }

    // --- cached: build per-generate caches once, then one batched B=2 forward per step ---
    let context_batch = concatenate_axis(&[&ctx_cond, &ctx_uncond], 0).unwrap();
    let cross_kv = dit.prepare_cross_kv(&context_batch).unwrap();
    let (cos, sin) = dit.prepare_rope(grid).unwrap();
    {
        let mut pre: Vec<&Array> = vec![&cos, &sin];
        for (k, v) in &cross_kv {
            pre.push(k);
            pre.push(v);
        }
        mlx_rs::transforms::eval(pre).unwrap();
    }
    let mut cached = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let preds = dit
            .forward_cached(&latent, t, &cross_kv, &cos, &sin, 2)
            .unwrap();
        let dt = timed(&[&preds[0], &preds[1]], start);
        if i >= warmup {
            cached.push(dt);
        }
    }

    // --- cached + compiled glue (sc-2957): identical B=2 cached path, but the fusable elementwise
    // chains (adaLN affine, gated residual, gated-GELU FFN activation, RoPE rotation) run through
    // `mx.compile` so MLX fuses each into one kernel. Measured +14.1%/step at 480p×25f (23.07→19.81,
    // bit-exact) — which MATCHES / beats the Python whole-model `mx.compile` ceiling (20.36 s/step,
    // `tools/bench_wan_a14b.py`): per-chain compile closes the eager-vs-compiled gap; whole-graph
    // compile would buy nothing further (sc-2957 finding — the "needs whole-graph" hunch was falsified). ---
    mlx_gen_wan::transformer::set_compile_glue(true);
    // Bit-exactness vs the eager cached path (the parity contract; batched_forward.rs CI-gates it).
    let comp0 = dit
        .forward_cached(&latent, t, &cross_kv, &cos, &sin, 2)
        .unwrap();
    let eag0 = {
        mlx_gen_wan::transformer::set_compile_glue(false);
        let e = dit
            .forward_cached(&latent, t, &cross_kv, &cos, &sin, 2)
            .unwrap();
        mlx_gen_wan::transformer::set_compile_glue(true);
        e
    };
    let max_diff: f32 = comp0
        .iter()
        .zip(eag0.iter())
        .map(|(c, e)| {
            let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(c, e).unwrap()).unwrap();
            mlx_rs::ops::max(d, None).unwrap().item::<f32>()
        })
        .fold(0.0f32, f32::max);

    let mut compiled = Vec::new();
    for i in 0..(warmup + iters) {
        let start = Instant::now();
        let preds = dit
            .forward_cached(&latent, t, &cross_kv, &cos, &sin, 2)
            .unwrap();
        let dt = timed(&[&preds[0], &preds[1]], start);
        if i >= warmup {
            compiled.push(dt);
        }
    }
    mlx_gen_wan::transformer::set_compile_glue(false);

    let leg = median(legacy);
    let cac = median(cached);
    let cmp = median(compiled);
    println!("[warm s/step] legacy(2×B1+recompute)={leg:.4}  cached(B2+stepcache)={cac:.4}  speedup={:.3}×", leg / cac);
    println!(
        "[warm s/step] cached(eager)={cac:.4}  cached(compiled-glue)={cmp:.4}  speedup={:.3}×  \
         (recovers {:.1}% of step)  max|Δ| compiled-vs-eager={max_diff:.3e}",
        cac / cmp,
        (cac - cmp) / cac * 100.0
    );

    // NOTE (sc-2853 measured finding): at the A14B's production geometries a single B=1 forward
    // already saturates the GPU, so batching B=2 buys no throughput and the cross-KV/RoPE recompute
    // is sub-1% — the two paths land within thermal noise (480p×25f: ~28.4 vs ~29.2 s/step). The
    // small-seq Rust-vs-Python gap is `mx.compile` (the eager mlx-rs port can't use it), NOT batching
    // — confirmed by `tools/bench_wan_a14b.py` (eager-Python ≈ Rust, compiled-Python ~20% faster).
    // This harness is a measurement tool, not a win-gate; assert only that batching doesn't *regress*
    // beyond the thermal band.
    assert!(
        cac <= leg * 1.10,
        "cached path materially slower than legacy: {cac:.4} vs {leg:.4} s/step (beyond noise band)"
    );
}
