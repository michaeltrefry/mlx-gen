//! sc-2963 compile-mechanism microbenchmark (rollout of the Wan sc-2957 template) — does
//! `mx.compile` fuse the LTX AvDiT's fusable elementwise *glue* into faster kernels?
//!
//! No weights — it times the fusable chains in isolation at LTX video production shapes (inner dim
//! 4096 = 32×128, FFN 16384 = 4×dim, 48 video blocks). The default seq 4352 ≈ 512²×129f **stage-2**
//! (17 latent frames × 16×16); a 1280×720 render reaches ~14960 (override with LTX_PERF_SEQ), where
//! the FFN tensor is ~245M elements — the same big-FFN driver that gave Wan +14%/step. All f32 here
//! (the `quant_f32` quality target). The chains:
//!   * **gelu_tanh** — the FFN activation on `[B, S, ffn]` (the dominant glue cost at video sequence).
//!   * **modulate** — adaLN affine `x·(1+scale)+shift` on `[B, S, dim]`.
//!   * **gated** — gated residual `x + out·gate` on `[B, S, dim]`.
//!   * **rope_rotate** — the split (rotate-halves) rotation on `[B, H, S, head_dim/2]` (q and k).
//!
//! Run it:
//! ```text
//! cargo test --release -p mlx-gen-ltx --test compile_micro -- --ignored --nocapture
//! ```

use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::ops::{add, multiply, power, subtract, tanh};
use mlx_rs::transforms::compile::{compile, CallMut, Compile};
use mlx_rs::transforms::eval;
use mlx_rs::{random, Array};

fn scalar(v: f32) -> Array {
    Array::from_slice(&[v], &[1])
}

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

// --- Pinned copies of the production glue closures (F-061) ---------------------------------------
// These intentionally duplicate the closure bodies of `src/transformer.rs::{gelu_ffn, modulate,
// gated}` and `src/rope.rs::rope_rotate`. They are NOT shared because those `fn`s are crate-private
// (an integration test can only reach `pub` items) and wrap the body in the `compile_glue()` runtime
// toggle, whereas this benchmark needs the bare closure to hand to `mx.compile` and time in isolation.
// **Keep each body byte-identical to its production source.** The `max_abs_diff` checks below confirm
// each compiled chain matches its own eager form, but they do NOT cross-check against production — so
// a change to the production glue math must be mirrored here by hand or the perf numbers go stale.
fn gelu_body(x: &Array) -> Result<Array, Exception> {
    let dt = x.dtype();
    let s = |v: f32| -> Result<Array, Exception> { scalar(v).as_dtype(dt) };
    let c = (2.0_f64 / std::f64::consts::PI).sqrt() as f32;
    let x3 = power(x, Array::from_int(3))?;
    let inner = multiply(&add(x, &multiply(&x3, &s(0.044_715)?)?)?, &s(c)?)?;
    let gate = add(&tanh(&inner)?, &s(1.0)?)?;
    multiply(&multiply(x, &s(0.5)?)?, &gate)
}

fn modulate_body((x, sc, sh): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(&multiply(x, &add(sc, scalar(1.0))?)?, sh)
}

fn gated_body((x, o, g): (&Array, &Array, &Array)) -> Result<Array, Exception> {
    add(x, &multiply(o, g)?)
}

fn rope_body(inp: &[Array]) -> Result<Vec<Array>, Exception> {
    let (a, b, c, s) = (&inp[0], &inp[1], &inp[2], &inp[3]);
    let out_first = subtract(&multiply(a, c)?, &multiply(b, s)?)?;
    let out_second = add(&multiply(b, c)?, &multiply(a, s)?)?;
    Ok(vec![out_first, out_second])
}

fn normal(shape: &[i32]) -> Array {
    let key = random::key(0).unwrap();
    let x = random::normal::<f32>(shape, None, None, Some(&key)).unwrap();
    eval([&x]).unwrap();
    x
}

fn max_abs_diff(a: &Array, b: &Array) -> f64 {
    let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
    mlx_rs::ops::max(&d, None).unwrap().item::<f32>() as f64
}

#[test]
#[ignore = "perf microbenchmark (no weights) — run with --ignored --nocapture"]
fn compile_glue_micro() {
    let b = env_usize("LTX_PERF_BATCH", 1) as i32;
    let s = env_usize("LTX_PERF_SEQ", 4352) as i32; // 512²×129f stage-2
    let dim = env_usize("LTX_DIM", 4096) as i32;
    let ffn = env_usize("LTX_FFN", 16384) as i32; // 4 × 4096
    let heads = env_usize("LTX_HEADS", 32) as i32;
    let half = env_usize("LTX_HALF", 64) as i32;
    let warmup = 3usize;
    let iters = 10usize;
    println!("shapes: B={b} S={s} dim={dim} ffn={ffn} heads={heads} half={half}  (warmup={warmup} iters={iters})");

    // ---- gelu_tanh FFN on [B, S, ffn] f32 (~48 / step — one FF per video block) ----
    {
        let x = normal(&[b, s, ffn]);
        let eager = bench(warmup, iters, || gelu_body(&x).unwrap());
        let oneshot = bench(warmup, iters, || compile(gelu_body, true)(&x).unwrap());
        let mut held = gelu_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut(&x).unwrap());
        let diff = max_abs_diff(
            &gelu_body(&x).unwrap(),
            &compile(gelu_body, true)(&x).unwrap(),
        );
        println!(
            "[gelu_tanh f32 {b}x{s}x{ffn}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×48={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 48.0
        );
    }

    // ---- modulate (adaLN affine) on [B, S, dim] f32 (~192 / step) ----
    {
        let m = normal(&[b, s, dim]);
        let sc = normal(&[b, 1, dim]);
        let sh = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || modulate_body((&m, &sc, &sh)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(modulate_body, true)((&m, &sc, &sh)).unwrap()
        });
        let mut held = modulate_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&m, &sc, &sh)).unwrap());
        let diff = max_abs_diff(
            &modulate_body((&m, &sc, &sh)).unwrap(),
            &compile(modulate_body, true)((&m, &sc, &sh)).unwrap(),
        );
        println!(
            "[modulate f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×192={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 192.0
        );
    }

    // ---- gated residual on [B, S, dim] f32 (~144 / step) ----
    {
        let x = normal(&[b, s, dim]);
        let o = normal(&[b, s, dim]);
        let g = normal(&[b, 1, dim]);
        let eager = bench(warmup, iters, || gated_body((&x, &o, &g)).unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(gated_body, true)((&x, &o, &g)).unwrap()
        });
        let mut held = gated_body.compile(true);
        let heldt = bench(warmup, iters, || held.call_mut((&x, &o, &g)).unwrap());
        let diff = max_abs_diff(
            &gated_body((&x, &o, &g)).unwrap(),
            &compile(gated_body, true)((&x, &o, &g)).unwrap(),
        );
        println!(
            "[gated f32 {b}x{s}x{dim}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×144={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 144.0
        );
    }

    // ---- rope_rotate (rotate-halves) on [B, H, S, half] f32 (q and k, ~96 / step) ----
    {
        let a = normal(&[b, heads, s, half]);
        let bb = normal(&[b, heads, s, half]);
        let c = normal(&[b, heads, s, half]);
        let sn = normal(&[b, heads, s, half]);
        let args = [a.clone(), bb.clone(), c.clone(), sn.clone()];
        let eager = bench(warmup, iters, || rope_body(&args).unwrap().pop().unwrap());
        let oneshot = bench(warmup, iters, || {
            compile(rope_body, true)(&args).unwrap().pop().unwrap()
        });
        let mut held = rope_body.compile(true);
        let heldt = bench(warmup, iters, || {
            held.call_mut(&args).unwrap().pop().unwrap()
        });
        let diff = max_abs_diff(
            &rope_body(&args).unwrap()[0],
            &compile(rope_body, true)(&args).unwrap()[0],
        );
        println!(
            "[rope_rotate f32 {b}x{heads}x{s}x{half}] eager={eager:.3} oneshot={oneshot:.3} held={heldt:.3} ms \
             | held speedup={:.2}× saved/call={:.3}ms ×96={:.1}ms | max|Δ|={diff:.2e}",
            eager / heldt,
            eager - heldt,
            (eager - heldt) * 96.0
        );
    }
}
