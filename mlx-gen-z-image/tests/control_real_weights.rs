//! sc-2349 / sc-2257: real-weights validation of the Z-Image **Fun-Controlnet-Union** port against
//! the frozen fork.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` base + the
//! `alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1` control checkpoint in the HF cache, plus the
//! golden produced by `tools/dump_z_image_control_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-z-image --release --test control_real_weights -- --ignored --nocapture
//!
//! Stage gates isolate the control transformer (feeding the fork's exact `cap_feats` +
//! `control_context`): the scale-0 self-consistency (control inert ⇒ base), the single-forward
//! velocity, and the denoise loop. The final gates drive the **public**
//! `load("z_image_turbo_control", spec).generate(req)` API with a `Conditioning::Control` (dense and
//! Q8) and confirm the render matches the fork's control golden.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, ControlKind, FlowMatchEuler, GenerationOutput, GenerationRequest, Image,
    LoadSpec, Progress, Quant, WeightsSource,
};
use mlx_gen_z_image::{
    decoded_to_image, denoise_control_with_progress, load_control_transformer, load_transformer,
    load_vae, unpack_latents, ZImageControlTransformer,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_control_golden.safetensors"
);
const Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_control_q8_golden.safetensors"
);

/// Locate the base Z-Image-Turbo snapshot dir (env override, else the HF cache).
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Locate the Fun-Controlnet-Union checkpoint (env override `CONTROL_WEIGHTS`, else the golden's
/// recorded path, else the HF cache). Returned as a single-file `WeightsSource`.
fn control_source(g: &Weights) -> WeightsSource {
    if let Ok(p) = std::env::var("CONTROL_WEIGHTS") {
        return WeightsSource::File(PathBuf::from(p));
    }
    if let Some(p) = g.metadata("control_weights") {
        let pb = PathBuf::from(p);
        if pb.exists() {
            return WeightsSource::File(pb);
        }
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home).join(
        ".cache/huggingface/hub/models--alibaba-pai--Z-Image-Turbo-Fun-Controlnet-Union-2.1/snapshots",
    );
    let file = std::fs::read_dir(&snaps)
        .expect("control HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .flat_map(|d| {
            std::fs::read_dir(d)
                .unwrap()
                .filter_map(|e| e.ok())
                .map(|e| e.path())
        })
        .find(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .expect("a control .safetensors");
    WeightsSource::File(file)
}

/// Peak-relative error `max|a-b| / max|b|`.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

fn meta_u32(g: &Weights, k: &str) -> u32 {
    g.metadata(k).unwrap().parse().unwrap()
}

fn meta_f32(g: &Weights, k: &str) -> f32 {
    g.metadata(k).unwrap().parse().unwrap()
}

/// The synthetic control image, read back from the golden so both sides use byte-identical pixels.
fn control_image(g: &Weights) -> Image {
    let w = meta_u32(g, "w");
    let h = meta_u32(g, "h");
    let arr = g.require("control_image_u8").unwrap(); // int32 HWC
    let pixels: Vec<u8> = arr.as_slice::<i32>().iter().map(|&v| v as u8).collect();
    assert_eq!(pixels.len(), (w * h * 3) as usize, "control image size");
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// Gate A (self-consistency): on real weights, `control_context_scale = 0` and `control_context =
/// None` produce the same velocity (the control branch is inert), and a non-zero scale genuinely
/// changes it (the control branch is active). Isolates the control transformer with the fork's
/// `cap_feats` + `control_context`.
#[test]
#[ignore = "needs real Z-Image + control weights and the local control golden"]
fn control_scale0_is_inert() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scale = meta_f32(&g, "control_scale");
    let t0 = 1.0 - sigmas[0];

    let transformer = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();
    let init = bf16(g.require("init").unwrap());
    let cap = bf16(g.require("cap_feats").unwrap());
    let cc = bf16(g.require("control_context").unwrap());

    let v_none = transformer.forward(&init, t0, &cap, None, 1.0).unwrap();
    let v_s0 = transformer
        .forward(&init, t0, &cap, Some(&cc), 0.0)
        .unwrap();
    let v_ctrl = transformer
        .forward(&init, t0, &cap, Some(&cc), scale)
        .unwrap();

    let inert = peak_rel(&v_s0, &v_none);
    let active = peak_rel(&v_ctrl, &v_none);
    println!(
        "control scale=0 vs None peak_rel={inert:.2e}; scale={scale} vs None peak_rel={active:.3e}"
    );
    assert!(
        inert < 1e-3,
        "control scale=0 is not inert: peak_rel {inert:.2e}"
    );
    assert!(
        active > 1e-2,
        "control branch appears inert at scale={scale}: peak_rel {active:.2e}"
    );
}

/// Gate B: the single control forward (first step) reproduces the fork's `v0`. f32 inputs rule out
/// a bf16-only divergence; the golden `v0` is the fork's bf16 control forward.
#[test]
#[ignore = "needs real Z-Image + control weights and the local control golden"]
fn control_single_forward_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scale = meta_f32(&g, "control_scale");
    let transformer = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();

    let v = transformer
        .forward(
            g.require("init").unwrap(),
            1.0 - sigmas[0],
            g.require("cap_feats").unwrap(),
            Some(g.require("control_context").unwrap()),
            scale,
        )
        .unwrap();
    let golden = g.require("v0").unwrap();
    assert_eq!(v.shape(), golden.shape(), "v0 shape");
    let pr = peak_rel(&v, golden);
    println!(
        "control single forward: v0 peak_rel={pr:.2e} shape={:?}",
        v.shape()
    );
    assert!(
        pr < 5e-2,
        "control single forward diverged: peak_rel {pr:.2e}"
    );
}

/// Gate C: the full control denoise loop reproduces the fork's final latents. The Rust control path
/// runs **f32** (the fork's bf16 path plus the source-MLX-vs-wheel toolchain residual otherwise
/// compounds over 8 steps; f32 lands closer — see `control_dtype_diag`), seeded from the fork's
/// exact bf16 noise. The latent peak-rel is a single-outlier metric (the real gate is px>8 below);
/// the 8-step control accumulation runs ~0.15, so the bound is looser than the base's 4-step 0.1.
#[test]
#[ignore = "needs real Z-Image + control weights and the local control golden"]
fn control_denoise_loop_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scale = meta_f32(&g, "control_scale");
    let scheduler = FlowMatchEuler { sigmas };
    let transformer = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();

    // f32 path, seeded from the fork's bf16 noise (bf16-round → f32), matching `generate`.
    let init = bf16(g.require("init").unwrap())
        .as_dtype(Dtype::Float32)
        .unwrap();
    let cap = g.require("cap_feats").unwrap().clone();
    let cc = g.require("control_context").unwrap().clone();
    let out = denoise_control_with_progress(
        &transformer,
        &scheduler,
        init,
        &cap,
        &cc,
        scale,
        0,
        &Default::default(),
        &mut |_| {},
    )
    .unwrap()
    .as_dtype(Dtype::Float32)
    .unwrap();

    let golden = g.require("final_latents").unwrap();
    assert_eq!(out.shape(), golden.shape(), "final latents shape");
    let pr = peak_rel(&out, golden);
    println!(
        "control denoise: final_latents peak_rel={pr:.2e} shape={:?}",
        out.shape()
    );
    assert!(
        pr < 2e-1,
        "control final latents diverged: peak_rel {pr:.2e}"
    );
}

/// Q8 control gate (sc-2257 Q8 control branch): proves the control port adds **zero** error to the
/// base Q8 path, so the Rust-vs-fork Q8 render gap (~8% px>8 over the 8-step loop) is entirely the
/// base z_image Q8 residual (sc-2532), inherited — NOT a control-branch defect.
///
/// Root-caused by stage bisection (`control_q8_bisect` + `tools/probe_z_control_q8_bisect.py`):
///   - Every control-specific stage (control embedder, refiner/main hints, threaded state) matches
///     the fork to <0.6% mean-rel. The divergence appears only in the **base** 30-layer main loop.
///   - The control model's base path (control inert, scale=0) is **byte-identical** to a freshly
///     loaded standalone base Q8 forward (asserted below, mean-rel 0) — the control adds nothing.
///   - The base Q8 single forward itself diverges ~1.26% mean-rel from the fork (vs ~0.15% dense);
///     that ~8× quantized-kernel residual is a base-model property (sc-2532), accumulated here over
///     8 steps + the deeper control stack. (NOT the activation dtype: the fork's own Q8 bf16-vs-f32
///     is only 0.24% — see `probe_z_control_q8_dtype.py`.)
///
/// The dense control path is pixel-faithful (`control_full_pipeline_matches_fork`, 0.166%).
#[test]
#[ignore = "needs real Z-Image + control weights and the local Q8 control golden \
            (QUANTIZE=8 dump_z_image_control_golden.py)"]
fn control_q8_transformer_matches_golden() {
    let g = Weights::from_file(Q8_GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scale = meta_f32(&g, "control_scale");
    let t0 = 1.0 - sigmas[0];
    let scheduler = FlowMatchEuler { sigmas };

    let mut transformer = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();
    transformer.quantize(8).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    let init = bf16(g.require("init").unwrap())
        .as_dtype(Dtype::Float32)
        .unwrap();
    let cap = g.require("cap_feats").unwrap().clone();
    let cc = g.require("control_context").unwrap().clone();

    // (a) FAITHFULNESS PROOF (pure Rust, no fork golden): the control model's base path (scale=0,
    // control inert) is byte-identical to a freshly loaded standalone base Q8 forward — so the
    // control port introduces zero error into the quantized base path.
    let mean_rel = |a: &Array, b: &Array| -> f32 {
        let n = b.shape().iter().product::<i32>();
        let a = a.reshape(&[n]).unwrap();
        let b = b.reshape(&[n]).unwrap();
        let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
        xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32 / mabs
    };
    let v_s0 = transformer
        .forward(&init, t0, &cap, Some(&cc), 0.0)
        .unwrap();
    let v_none = transformer.forward(&init, t0, &cap, None, 1.0).unwrap();
    let mut base = load_transformer(&snapshot()).unwrap();
    base.quantize(8).unwrap();
    let v_base = base.forward(&init, t0, &cap).unwrap();
    let inert = mean_rel(&v_s0, &v_none);
    let vs_base = mean_rel(&v_none, &v_base);
    println!("control Q8 composition: scale0-vs-None mean_rel={inert:.3e}  None-vs-standalone-base mean_rel={vs_base:.3e}");
    assert!(
        inert == 0.0,
        "Q8 control branch not inert at scale=0: {inert:.3e}"
    );
    assert!(
        vs_base == 0.0,
        "Q8 control base path differs from standalone base Q8 (composition bug): {vs_base:.3e}"
    );

    // (b) Single forward vs fork-Q8 v0 — faithful per-step quant (same order as the base Q8).
    let v0 = transformer
        .forward(&init, t0, &cap, Some(&cc), scale)
        .unwrap();
    let v0_pr = peak_rel(&v0, g.require("v0").unwrap());
    println!("control Q8 single forward: v0 peak_rel={v0_pr:.3e}");
    assert!(
        v0_pr < 3e-2,
        "Q8 single forward diverged: peak_rel {v0_pr:.3e}"
    );

    let latents = denoise_control_with_progress(
        &transformer,
        &scheduler,
        init,
        &cap,
        &cc,
        scale,
        0,
        &Default::default(),
        &mut |_| {},
    )
    .unwrap()
    .as_dtype(Dtype::Float32)
    .unwrap();

    let golden = g.require("final_latents").unwrap();
    println!(
        "control Q8 transformer: final_latents peak_rel={:.3e}",
        peak_rel(&latents, golden)
    );

    let unpacked = unpack_latents(&latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae
        .decode(&latent5)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "control Q8 transformer-isolated decode: {:.3}% px>8 ({differ}/{})",
        frac * 100.0,
        img.pixels.len()
    );
}

/// DIAGNOSTIC (sc-2349 Q8 root-cause): stage-by-stage bisection of the Q8 control single forward vs
/// the fork's f32 Q8 stages (`tools/probe_z_control_q8_bisect.py`). The first stage to diverge on
/// mean-rel localizes the bug. fork-f32 ≈ fork-bf16 (~0.24%), so this isolates a real bug from the
/// activation dtype. Uses `forward_capture` to read Rust's intermediates.
#[test]
#[ignore = "diagnostic; needs the Q8 bisect golden"]
fn control_q8_bisect() {
    let g = Weights::from_file(Q8_GOLDEN).unwrap();
    let bisect = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/z_control_q8_bisect.safetensors"
    ))
    .unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scale = meta_f32(&g, "control_scale");

    let mut transformer = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();
    transformer.quantize(8).unwrap();

    let init = bf16(g.require("init").unwrap())
        .as_dtype(Dtype::Float32)
        .unwrap();
    let cap = g.require("cap_feats").unwrap().clone();
    let cc = g.require("control_context").unwrap().clone();

    let (v0, stages) = transformer
        .forward_capture(&init, 1.0 - sigmas[0], &cap, &cc, scale)
        .unwrap();

    // mean-relative error (robust; not a single outlier like peak_rel).
    let mean_rel = |a: &Array, b: &Array| -> f32 {
        let n = b.shape().iter().product::<i32>();
        let a = a.reshape(&[n]).unwrap();
        let b = b.reshape(&[n]).unwrap();
        let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
        xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32 / mabs
    };

    println!("--- Q8 control single forward: Rust vs fork-f32 stages ---");
    for (name, arr) in &stages {
        let Some(golden) = bisect.get(name) else {
            continue; // "x_tokens" has no bisect-golden entry — compared to the qmm probe below
        };
        if arr.shape() != golden.shape() {
            println!(
                "  {name:>14}: SHAPE {:?} vs {:?}",
                arr.shape(),
                golden.shape()
            );
            continue;
        }
        println!(
            "  {name:>14}: mean_rel={:.3e}  peak_rel={:.3e}",
            mean_rel(arr, golden),
            peak_rel(arr, golden)
        );
    }
    println!(
        "  {:>14}: mean_rel={:.3e}  peak_rel={:.3e}",
        "v0",
        mean_rel(&v0, bisect.require("v0_scale1").unwrap()),
        peak_rel(&v0, bisect.require("v0_scale1").unwrap())
    );

    // STAGE-INJECTION (resolves the qmm_smallk=0.0 vs bisection x_emb=0.3% contradiction): is Rust's
    // own `patchify(init)` byte-identical to the fork's patchified init? `qmm_smallK_probe`'s `x` IS
    // the fork's patchify(init), and `qmm_smallk` already proved Rust's qmm on that exact `x` matches
    // the fork at 0.0 — so if Rust's `x_tokens` here also equals it, the embedder input is identical
    // and the x_emb 0.3% must be a measurement artifact; if it diverges, patchify is the source.
    let qmm_probe = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/qmm_smallK_probe.safetensors"
    ))
    .unwrap();
    let fork_x = qmm_probe.require("x").unwrap();
    match stages.iter().find(|(n, _)| *n == "x_tokens") {
        Some((_, x_tokens)) if x_tokens.shape() == fork_x.shape() => println!(
            "  patchify x_tokens: Rust vs fork(qmm-probe x) mean_rel={:.3e}  peak_rel={:.3e}",
            mean_rel(x_tokens, fork_x),
            peak_rel(x_tokens, fork_x)
        ),
        Some((_, x_tokens)) => println!(
            "  patchify x_tokens: SHAPE {:?} vs fork {:?}",
            x_tokens.shape(),
            fork_x.shape()
        ),
        None => println!("  patchify x_tokens: not captured"),
    }

    // Base-path (scale=0, control inert) vs the hints (scale=1): if scale=0 has the SAME per-forward
    // error as scale=1, the divergence is in the BASE path, not the control hints.
    let v0_s0 = transformer
        .forward(&init, 1.0 - sigmas[0], &cap, Some(&cc), 0.0)
        .unwrap();
    println!(
        "  base path (scale=0) vs fork v0_scale0: mean_rel={:.3e}  peak_rel={:.3e}",
        mean_rel(&v0_s0, bisect.require("v0_scale0").unwrap()),
        peak_rel(&v0_s0, bisect.require("v0_scale0").unwrap())
    );

    // COMPOSITION CHECK (pure Rust): does the control model's base path equal (1) its own None path
    // (which delegates to base.forward) and (2) a freshly-loaded STANDALONE base Q8 forward? If both
    // are ~0, the control port is faithful and the Q8 residual is inherited from the base Q8 path.
    let v_none = transformer
        .forward(&init, 1.0 - sigmas[0], &cap, None, 1.0)
        .unwrap();
    let mut base = load_transformer(&snapshot()).unwrap();
    base.quantize(8).unwrap();
    let v_base = base.forward(&init, 1.0 - sigmas[0], &cap).unwrap();
    println!(
        "  COMPOSITION: scale0-vs-None mean_rel={:.3e} | None-vs-standalone-base-Q8 mean_rel={:.3e} peak_rel={:.3e}",
        mean_rel(&v0_s0, &v_none),
        mean_rel(&v_none, &v_base),
        peak_rel(&v_none, &v_base),
    );
}

/// DETERMINISM (sc-2349 Q8 root-cause; Michael's falsification test): does the SAME toolchain
/// produce the SAME Q8 output run-to-run, back-to-back? My "inherited base-Q8 residual = source-MLX-
/// vs-wheel toolchain" story assumes the Rust-vs-fork gap is a STABLE cross-build difference. If a
/// single forward / the full loop differ run-to-run within one process, the Metal quantized-matmul
/// kernel uses non-fixed-order (atomic) reductions and the "8% residual" is partly nondeterminism,
/// not a stable computational difference. Reports max|Δ| for Q8 vs dense, single-forward + full-loop.
#[test]
#[ignore = "diagnostic; needs real Z-Image + control weights and the Q8 control golden"]
fn control_q8_determinism() {
    let g = Weights::from_file(Q8_GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scale = meta_f32(&g, "control_scale");
    let t0 = 1.0 - sigmas[0];
    let scheduler = FlowMatchEuler { sigmas };

    let init = bf16(g.require("init").unwrap())
        .as_dtype(Dtype::Float32)
        .unwrap();
    let cap = g.require("cap_feats").unwrap().clone();
    let cc = g.require("control_context").unwrap().clone();

    let max_abs = |a: &Array, b: &Array| -> f32 {
        let n = b.shape().iter().product::<i32>();
        let a = a.reshape(&[n]).unwrap();
        let b = b.reshape(&[n]).unwrap();
        a.as_slice::<f32>()
            .iter()
            .zip(b.as_slice::<f32>())
            .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
    };

    // (1) Q8, same loaded model, single forward twice (isolates the kernel).
    let mut q8 = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();
    q8.quantize(8).unwrap();
    let a = q8.forward(&init, t0, &cap, Some(&cc), scale).unwrap();
    let b = q8.forward(&init, t0, &cap, Some(&cc), scale).unwrap();
    println!(
        "Q8 single-forward run-to-run max|Δ|={:.3e}",
        max_abs(&a, &b)
    );

    // (2) Q8 full 8-step loop twice (run-to-run end-to-end).
    let run = |t: &ZImageControlTransformer| {
        denoise_control_with_progress(
            t,
            &scheduler,
            init.clone(),
            &cap,
            &cc,
            scale,
            0,
            &Default::default(),
            &mut |_| {},
        )
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
    };
    let la = run(&q8);
    let lb = run(&q8);
    println!("Q8 full-loop run-to-run max|Δ|={:.3e}", max_abs(&la, &lb));

    // (3) Dense control single forward twice (baseline: is even the dense path deterministic?).
    let dense = load_control_transformer(&snapshot(), &control_source(&g)).unwrap();
    let da = dense.forward(&init, t0, &cap, Some(&cc), scale).unwrap();
    let db = dense.forward(&init, t0, &cap, Some(&cc), scale).unwrap();
    println!(
        "dense single-forward run-to-run max|Δ|={:.3e}",
        max_abs(&da, &db)
    );

    // (4) ACROSS-PROCESS: persist this process's Q8 v0 + final latents; on a second invocation,
    // compare against the previous process's bytes. Rust-vs-fork is itself cross-process, so this is
    // the determinism check that actually matters for the "stable cross-build residual" claim.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    let read_f32 = |p: &std::path::Path| -> Option<Vec<f32>> {
        std::fs::read(p).ok().map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        })
    };
    let write_f32 = |p: &std::path::Path, a: &Array| {
        let n = a.shape().iter().product::<i32>();
        let v = a.reshape(&[n]).unwrap();
        let bytes: Vec<u8> = v
            .as_slice::<f32>()
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        std::fs::write(p, bytes).unwrap();
    };
    let cmp_prev = |label: &str, p: &std::path::Path, cur: &Array| {
        if let Some(prev) = read_f32(p) {
            let n = cur.shape().iter().product::<i32>();
            let cur = cur.reshape(&[n]).unwrap();
            let md = cur
                .as_slice::<f32>()
                .iter()
                .zip(&prev)
                .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
            println!("{label} ACROSS-PROCESS (vs previous invocation) max|Δ|={md:.3e}");
        } else {
            println!("{label} ACROSS-PROCESS: no previous file — run this test again to compare");
        }
    };
    let v0_path = dir.join("rust_q8_v0.bin");
    let final_path = dir.join("rust_q8_final.bin");
    cmp_prev("Q8 single-forward", &v0_path, &a);
    cmp_prev("Q8 full-loop", &final_path, &la);
    write_f32(&v0_path, &a);
    write_f32(&final_path, &la);
}

/// Drive the full control pipeline through the public API and compare to the fork's golden render.
fn full_pipeline(golden_path: &str, quant: Option<Quant>, bits_label: &str, max_px_frac: f32) {
    let g = Weights::from_file(golden_path).unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = meta_u32(&g, "steps");
    let (w, h) = (meta_u32(&g, "w"), meta_u32(&g, "h"));
    let scale = meta_f32(&g, "control_scale");

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_control(control_source(&g));
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    let generator = mlx_gen::load("z_image_turbo_control", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        conditioning: vec![Conditioning::Control {
            image: control_image(&g),
            kind: ControlKind::Pose,
            scale,
        }],
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                assert_eq!(total, steps, "control runs all steps (txt2img start)");
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    assert_eq!(last_step, steps, "expected {steps} step events");

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1);
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../tools/golden/rust_z_image_control{bits_label}.png"
    ));
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "✓ control{bits_label} public generate: {}x{}; {differ}/{} px differ by >8 ({:.3}%); saved {}",
        img.width,
        img.height,
        img.pixels.len(),
        frac * 100.0,
        out_path.display()
    );
    assert!(
        frac < max_px_frac,
        "control{bits_label} render diverged from the fork: {:.3}% px>8 >= {:.3}%",
        frac * 100.0,
        max_px_frac * 100.0
    );
}

/// The integration proof: dense control render vs the fork's control golden.
#[test]
#[ignore = "needs real Z-Image + control weights and the local control golden"]
fn control_full_pipeline_matches_fork() {
    // Measured (1024², 8-step, scale 1.0): 0.166% px>8 (the source-MLX-vs-wheel toolchain residual
    // accumulated over the loop). 1% leaves headroom for run variance.
    full_pipeline(GOLDEN, None, "", 0.01);
}

/// sc-2257 Q8 control branch: the **full public Q8 path** (`with_quant(Q8)`) vs the fork's Q8
/// control golden. Base + control quantized together; the control patch embedder stays dense.
///
/// Measured ~8% px>8 — the **inherited base z_image Q8 residual** (sc-2532) accumulated over the
/// control's 8-step loop + deeper stack, NOT a control-branch defect. Proven by
/// `control_q8_transformer_matches_golden`: the control's base path is byte-identical to a
/// standalone base Q8 forward (composition adds zero error) and every control-specific stage matches
/// the fork to <0.6%. The base Q8 per-forward residual (~1.26% mean-rel, vs ~0.15% dense) is a
/// quantized-kernel property of the base model — see [[zimage-q4q8-sc2532]] / the base Q8 gates. The
/// dense control path is the pixel-faithful parity gate (`control_full_pipeline_matches_fork`,
/// 0.166%). The bound (12%) accommodates the inherited residual with run-variance headroom.
#[test]
#[ignore = "needs real Z-Image + control weights and the local Q8 control golden \
            (QUANTIZE=8 dump_z_image_control_golden.py)"]
fn control_q8_full_pipeline_matches_fork() {
    full_pipeline(Q8_GOLDEN, Some(Quant::Q8), "_q8", 0.12);
}
