//! sc-2957 compile-mechanism microbenchmark — does `mx.compile` (mlx-rs) fuse the Wan DiT's
//! elementwise *glue* into faster kernels, the way `mx.compile(model)` does for the Python reference?
//!
//! This needs **no weights** — it times the fusable elementwise chains in isolation at the A14B
//! 480p×25f production shapes (B=2, seq 10920, dim 5120, ffn_dim 13824), eager vs `compile`d, so the
//! kernel-fusion win can be measured + attributed per chain BEFORE the expensive real-weight A/B
//! (`tests/perf.rs`). The chains:
//!   * **ffn_gelu** — the gated-GELU FFN activation `gelu_tanh` on `[B, L, ffn_dim]` bf16 (~8 eager
//!     elementwise ops on a ~600 MB tensor / block, the single biggest glue cost; currently UNcompiled
//!     in `mlx_gen::nn::gelu_tanh`).
//!   * **modulate** — adaLN affine `ln_out·(1+e_scale)+e_shift` on `[B, L, dim]` f32 (the `layer_norm`
//!     itself is already a fused `mx.fast` kernel and stays eager; only the affine fuses).
//!   * **gated** — gated residual `x + y·gate` on `[B, L, dim]` f32.
//!
//! ## Three variants per chain — the erase-per-call trap
//! mlx-rs's `CompiledState` has a `Drop` that calls `mlx_detail_compile_erase(id)` (compile/mod.rs).
//! The public `compile(f, _)` returns a closure that **re-creates and drops** a `Compiled` on every
//! invocation, so it evicts the trace cache each call. We therefore measure:
//!   * **eager** — no compile.
//!   * **oneshot** — `compile(f, true)(x)` per call (what core `gelu_exact`/`gelu_quick` do): re-traces
//!     + erases every call, only the fused Metal kernel is cache-hit.
//!   * **held** — build the `Compiled` handle once via the `Compile` trait, reuse `call_mut` across
//!     calls (the efficient pattern: trace once, no per-call erase).
//!
//! The gap between oneshot and held is the re-trace/erase overhead; the gap between eager and the best
//! compiled variant is the reachable fusion win.
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-wan --test compile_micro -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_gen::array::scalar;
use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, power, tanh};
use mlx_rs::transforms::compile::{compile, CallMut, Compile};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array, Dtype};

fn env_usize(var: &str, default: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

/// Time `f` (returns the array to force-eval) over `warmup + iters`, return median ms.
fn bench(warmup: usize, iters: usize, mut f: impl FnMut() -> Array) -> f64 {
    let mut times = Vec::new();
    for i in 0..(warmup + iters) {
        let t0 = Instant::now();
        let out = f();
        eval([&out]).unwrap();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= warmup {
            times.push(dt);
        }
    }
    median(times)
}

// gelu_tanh body (mirrors mlx_gen::nn::gelu_tanh exactly), generic so eager/oneshot/held share it.
fn gelu_body(x: &Array) -> Result<Array, Exception> {
    let dt = x.dtype();
    let s = |v: f32| -> Result<Array, Exception> { scalar(v).as_dtype(dt) };
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
    let gate = add(&tanh(&inner)?, &s(1.0)?)?;
    multiply(&multiply(x, &s(0.5)?)?, &gate)
}

fn modulate_body((m, e1, e0): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(&multiply(m, &add(e1, scalar(1.0))?)?, e0)
}

fn gated_body((x, y, g): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(x, &multiply(y, g)?)
}

fn normal(shape: &[i32], dt: Dtype) -> Array {
    let key = random::key(0).unwrap();
    let x = random::normal::<f32>(shape, None, None, Some(&key)).unwrap();
    let x = if dt == Dtype::Float32 {
        x
    } else {
        x.as_dtype(dt).unwrap()
    };
    eval([&x]).unwrap();
    x
}

fn max_abs_diff(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    let m = mlx_rs::ops::max(d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    m.item::<f32>() as f64
}

#[test]
#[ignore = "perf microbenchmark (no weights) — run with --ignored --nocapture"]
fn compile_glue_micro() {
    let b = env_usize("WAN_PERF_BATCH", 2) as i32;
    let l = env_usize("WAN_PERF_SEQ", 10920) as i32;
    let dim = env_usize("WAN_DIM", 5120) as i32;
    let ffn = env_usize("WAN_FFN", 13824) as i32;
    let warmup = 3usize;
    let iters = 12usize;
    println!("shapes: B={b} L={l} dim={dim} ffn_dim={ffn}  (warmup={warmup} iters={iters})");

    // ---- ffn_gelu on [B, L, ffn_dim] bf16 (×40 layers) ----
    {
        let x = normal(&[b, l, ffn], Dtype::Bfloat16);
        let eager = bench(warmup, iters, || gelu_body(&x).unwrap());
        let oneshot = bench(warmup, iters, || compile(gelu_body, true)(&x).unwrap());
        let mut held = gelu_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut(&x).unwrap());
        let diff = max_abs_diff(
            &gelu_body(&x).unwrap(),
            &compile(gelu_body, true)(&x).unwrap(),
        );
        println!(
            "[ffn_gelu bf16 {b}x{l}x{ffn}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×40L={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 40.0
        );
    }

    // ---- modulate (adaLN affine) on [B, L, dim] f32 (×80 = 2/block × 40) ----
    {
        let m = normal(&[b, l, dim], Dtype::Float32);
        let e1 = normal(&[1, 1, dim], Dtype::Float32);
        let e0 = normal(&[1, 1, dim], Dtype::Float32);
        let eager = bench(warmup, iters, || modulate_body((&m, &e1, &e0)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(modulate_body, true)((&m, &e1, &e0)).unwrap()
        });
        let mut held = modulate_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&m, &e1, &e0)).unwrap());
        let diff = max_abs_diff(
            &modulate_body((&m, &e1, &e0)).unwrap(),
            &compile(modulate_body, true)((&m, &e1, &e0)).unwrap(),
        );
        println!(
            "[modulate f32 {b}x{l}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×80={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 80.0
        );
    }

    // ---- gated residual on [B, L, dim] f32 (×120 = 3/block × 40) ----
    {
        let x = normal(&[b, l, dim], Dtype::Float32);
        let y = normal(&[b, l, dim], Dtype::Float32);
        let g = normal(&[1, 1, dim], Dtype::Float32);
        let eager = bench(warmup, iters, || gated_body((&x, &y, &g)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(gated_body, true)((&x, &y, &g)).unwrap()
        });
        let mut held = gated_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&x, &y, &g)).unwrap());
        let diff = max_abs_diff(
            &gated_body((&x, &y, &g)).unwrap(),
            &compile(gated_body, true)((&x, &y, &g)).unwrap(),
        );
        println!(
            "[gated f32 {b}x{l}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×120={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 120.0
        );
    }
}
