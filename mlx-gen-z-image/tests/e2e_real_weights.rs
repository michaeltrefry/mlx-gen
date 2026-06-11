//! sc-2352 / sc-2344: end-to-end validation of the Z-Image port against a real-weights golden run.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the
//! golden produced by `tools/dump_z_image_golden.py` (gitignored, local). Run with:
//!   cargo test -p mlx-gen-z-image --release --test e2e_real_weights -- --ignored --nocapture
//!
//! The stage tests validate each pipeline stage on real bf16 weights against the fork's
//! intermediates; the final test drives the **public** `load(id, spec).generate(req)` API and
//! confirms the rendered image matches the fork's golden.

use mlx_gen::weights::Weights;
use mlx_gen::{
    FlowMatchEuler, GenerationOutput, GenerationRequest, LoadSpec, Progress, WeightsSource,
};
use mlx_gen_z_image::{
    decoded_to_image, denoise, load_text_encoder, load_tokenizer, load_transformer, load_vae,
    slice_valid, unpack_latents,
};
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_golden.safetensors"
);
const Q8_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_q8_golden.safetensors"
);
const Q4_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/z_image_q4_golden.safetensors"
);

/// Locate the Z-Image-Turbo snapshot dir (env override, else the HF cache).
mod common;
use common::snapshot;

/// Peak-relative error `max|a-b| / max|b|` — the meaningful metric for high-dynamic-range
/// tensors compared against a bf16 golden.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    // reshape to 1-D forces C-order materialization (decode/transpose views would otherwise
    // expose physical, not logical, order through as_slice).
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

/// STALE-GOLDEN GUARD (sc-3007). The golden is a gitignored local fixture that can silently drift
/// from the production schedule: the per-stage `e2e_denoise_loop` / `q_pipeline_matches_fork` tests
/// feed the golden's *own* embedded `sigmas`, so a golden dumped at the wrong time-shift passes
/// every stage yet makes only the public-`generate` tests (which recompute the schedule) diverge
/// — ~40% px. That is exactly how `e2e_full_pipeline_generates_fox` was failing on `main`: the dense
/// golden was dumped pre-sc-2536 with the empirical-mu schedule (shift≈9.89) while production
/// switched to static shift=3.0. Assert the golden's schedule matches production up front so a stale
/// golden fails loudly with a re-dump hint instead of a cryptic pixel mismatch. (3.0 mirrors the
/// model's `SCHEDULE_SHIFT`, which is `pub(crate)` and so not reachable from this integration test.)
fn assert_golden_schedule_is_production(
    g: &Weights,
    steps: u32,
    w: u32,
    h: u32,
    seed: u64,
    prompt: &str,
) {
    let golden_sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let prod_sigmas = FlowMatchEuler::for_static_shift(steps as usize, 3.0).sigmas;
    assert_eq!(
        golden_sigmas.len(),
        prod_sigmas.len(),
        "STALE GOLDEN (sc-3007): golden sigma count {} != production schedule length {} — re-dump",
        golden_sigmas.len(),
        prod_sigmas.len()
    );
    for (i, (gs, ps)) in golden_sigmas.iter().zip(&prod_sigmas).enumerate() {
        assert!(
            (gs - ps).abs() < 1e-4,
            "STALE GOLDEN (sc-3007): golden sigma[{i}]={gs} != production for_static_shift({steps}, \
             3.0)[{i}]={ps}. The golden was dumped at a different schedule than the public `generate` \
             path uses — re-dump it with the current `tools/dump_z_image_golden.py` (mlx 0.31.2), e.g. \
             `ZIMAGE_W={w} ZIMAGE_H={h} ZIMAGE_STEPS={steps} ZIMAGE_SEED={seed} ZIMAGE_PROMPT=\"{prompt}\" \
             [QUANTIZE=N] <mflux-0.31.2-venv>/bin/python tools/dump_z_image_golden.py`"
        );
    }
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_text_encoder_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    let enc = load_text_encoder(&snapshot()).unwrap();
    let out = enc
        .forward(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let cap = slice_valid(&out, num_valid).unwrap();

    let golden = g.require("cap_feats").unwrap();
    assert_eq!(cap.shape(), golden.shape(), "cap_feats shape");

    let a = cap.as_slice::<f32>();
    let b = golden.as_slice::<f32>();
    let max_abs_g = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff: f32 =
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    // Peak-relative error: the meaningful metric for a high-dynamic-range tensor (values reach
    // ~1.4e4) compared against a bf16 golden after a 35-layer f32 forward.
    let peak_rel = max_diff / max_abs_g;
    println!(
        "cap_feats: max|golden|={max_abs_g:.1} max|diff|={max_diff:.3} peak_rel={peak_rel:.2e} mean|diff|={mean_diff:.5}"
    );
    assert!(
        peak_rel < 2e-3,
        "cap_feats diverged from the fork: peak-relative error {peak_rel:.2e} >= 2e-3"
    );
    println!(
        "✓ text encoder: cap_feats {:?} matches the fork golden (peak-rel {peak_rel:.2e})",
        cap.shape()
    );
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_transformer_single_forward_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let transformer = load_transformer(&snapshot()).unwrap();

    // First step in f32 (rules out bf16): v0 = transformer(init, 1 - sigma[0], cap_feats).
    let timestep0 = 1.0 - sigmas[0];
    let v = transformer
        .forward(
            g.require("init").unwrap(),
            timestep0,
            g.require("cap_feats").unwrap(),
        )
        .unwrap();
    let golden = g.require("v0").unwrap();
    assert_eq!(v.shape(), golden.shape(), "v0 shape");
    let pr = peak_rel(&v, golden);
    println!(
        "transformer single forward: v0 peak_rel={pr:.2e} shape={:?}",
        v.shape()
    );
    assert!(
        pr < 5e-2,
        "single transformer forward diverged at real resolution: peak_rel {pr:.2e}"
    );
    println!("✓ transformer single forward matches golden");
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_denoise_loop_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    // Use the fork's exact sigmas (not a recomputed schedule) so this isolates the loop, not mu.
    let scheduler = FlowMatchEuler { sigmas };
    let transformer = load_transformer(&snapshot()).unwrap();

    // Match the fork's bf16 path: init noise + cap_feats fed to the DiT as bf16.
    let init = bf16(g.require("init").unwrap());
    let cap = bf16(g.require("cap_feats").unwrap());
    let out = denoise(&transformer, &scheduler, init, &cap).unwrap();
    let out = out.as_dtype(Dtype::Float32).unwrap();

    let golden = g.require("final_latents").unwrap();
    assert_eq!(out.shape(), golden.shape(), "final latents shape");
    let pr = peak_rel(&out, golden);
    println!(
        "denoise: final_latents peak_rel={pr:.2e} shape={:?}",
        out.shape()
    );
    // bf16 accumulation over 4 iterative steps (each feeding the next) compounds; the decoded
    // image is near-pixel-perfect, so this peak-relative latent drift is benign.
    assert!(pr < 1e-1, "final latents diverged: peak_rel {pr:.2e}");
    println!("✓ denoise loop matches golden (peak-rel {pr:.2e})");
}

#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_vae_and_image_matches_golden() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vae = load_vae(&snapshot()).unwrap();

    // golden final_latents [16,1,H,W] -> unpack [1,16,H,W] -> [1,16,1,H,W] for decode.
    let latents = g.require("final_latents").unwrap();
    let unpacked = unpack_latents(latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae.decode(&latent5).unwrap(); // f32 (latents f32, weights bf16 -> promote)
    let decoded = decoded.as_dtype(Dtype::Float32).unwrap();

    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");

    // The authoritative guardrail: the per-pixel RGB8 diff (this is what the rendered image is
    // actually judged on). Assert it FIRST so a genuine decode regression is caught here and is
    // neither masked by — nor able to mask — the single-pixel HDR-outlier `peak_rel` below.
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 2)
        .count();
    println!(
        "✓ vae+image: {}x{}, {} / {} pixels differ by >2",
        img.width,
        img.height,
        differ,
        img.pixels.len()
    );
    assert!(
        differ < img.pixels.len() / 50,
        "too many pixel diffs: {differ}"
    );

    // Secondary HDR-outlier metric. `peak_rel = max|Δ| / max|golden|` is dominated by a single
    // high-dynamic-range pixel, so it drifts whenever the golden latents change: 2e-2 (pre-bump) →
    // ~2.8e-2 (MLX 0.31.1 bump, sc-2517) → 3.73e-2 after the sc-3007 golden re-dump (PR #129, which
    // switched the golden to the production `for_static_shift(3.0)` sigmas). The RGB8 guardrail above
    // is authoritative; this only catches a gross decode shift, so the threshold tracks the re-dump
    // with the usual cushion (history: 2e-2 → 2.8e-2 → 3.5e-2 → 4.5e-2).
    let pr = peak_rel(&decoded, golden);
    println!("vae: decoded peak_rel={pr:.2e} shape={:?}", decoded.shape());
    assert!(pr < 4.5e-2, "VAE decode diverged: peak_rel {pr:.2e}");
}

/// The integration proof: the full prompt→image pipeline through the **public** Generator API
/// (`mlx_gen::load("z_image_turbo", …).generate(req)`), compared to the fork's golden render.
#[test]
#[ignore = "needs real Z-Image weights + local golden"]
fn e2e_full_pipeline_generates_fox() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let snap = snapshot();
    let num_valid: i32 = g.metadata("num_valid").unwrap().parse().unwrap();

    // Drive the request from the golden's own metadata so this test tracks whatever
    // (prompt, seed, steps, size) the golden was dumped at — no separate hardcoding to
    // drift. dump_z_image_golden.py honors ZIMAGE_W/H/STEPS/SEED/PROMPT; this reads them back.
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    // STALE-GOLDEN GUARD (sc-3007): fail fast if the golden's embedded schedule doesn't match the
    // production `for_static_shift(steps, 3.0)` path (see `assert_golden_schedule_is_production`).
    assert_golden_schedule_is_production(&g, steps, w, h, seed, &prompt);

    // Tokenizer parity: the prompt with the Qwen chat template reproduces the fork's ids exactly.
    let tok = load_tokenizer(&snap).unwrap();
    let t = tok.tokenize(&prompt).unwrap();
    let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&t);
    let take_n =
        |a: &Array| a.reshape(&[-1]).unwrap().as_slice::<i32>()[..num_valid as usize].to_vec();
    assert_eq!(
        take_n(&input_ids),
        take_n(g.require("input_ids").unwrap()),
        "tokenizer input_ids diverge from the fork"
    );

    // Full pipeline through the public API: load(id, spec) -> generate(req).
    let spec = LoadSpec::new(WeightsSource::Dir(snap));
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt: prompt.clone(),
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, total } = p {
                assert_eq!(total, steps, "step total");
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    assert_eq!(
        last_step, steps,
        "expected {steps} denoise-step progress events"
    );

    let img = match out {
        GenerationOutput::Images(mut v) => {
            assert_eq!(v.len(), 1, "count=1 -> one image");
            v.pop().unwrap()
        }
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    // Save the Rust render for visual inspection.
    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_fox.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    // Compare to the fork's golden image (bf16-loop drift allows a small fraction of pixels to
    // differ).
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "✓ full pipeline (public generate): prompt->image {}x{}; {} / {} pixels differ by >8 from the fork; saved {}",
        img.width,
        img.height,
        differ,
        img.pixels.len(),
        out_path.display()
    );
    assert!(
        differ < img.pixels.len() / 20,
        "full-pipeline image diverges: {differ} pixels"
    );
}

/// sc-2532: the Q4/Q8 **transformer**-isolation diagnostic. Quantize the Rust transformer
/// (group_size 64 — the fork's `nn.quantize` set, all 276 transformer Linears), feed the fork's
/// seeded init noise + quantized-text-encoder `cap_feats`, run the denoise loop on the fork's exact
/// sigmas, and confirm the latents + decoded image match the fork's quantized golden. (The full
/// product path — quantizing the text encoder + VAE too — is gated by `q{8,4}_full_generate_renders`;
/// feeding `cap_feats` here isolates the transformer from the text encoder.)
///
/// The quantization is byte-identical to the fork *once the weight is quantized at bf16* — the fork's
/// compute dtype. Z-Image-Turbo ships an **f32** transformer checkpoint; quantizing it as-loaded
/// (f32) yields group `scales` ~0.13% off the fork's bf16 scales, which compounded into a ~0.78%
/// px>8 base-Q8 residual (sc-2604, previously misattributed to "source-MLX-vs-wheel toolchain").
/// `AdaptableLinear::quantize` now casts to bf16 first, so the residual collapses to the dense floor
/// (~0.03% px>8 @1024²). Golden is regenerated at 1024² (see the `#[ignore]` message).
fn q_pipeline_matches_fork(
    golden_path: &str,
    bits: i32,
    max_latent_mean_rel: f32,
    max_px_frac: f32,
) {
    let g = Weights::from_file(golden_path).unwrap();
    let stored: i32 = g.metadata("quantize").unwrap().parse().unwrap();
    assert_eq!(stored, bits, "golden was dumped at a different bit-width");
    let snap = snapshot();

    let mut transformer = load_transformer(&snap).unwrap();
    transformer.quantize(bits).unwrap();
    let vae = load_vae(&snap).unwrap();

    // Fork's exact sigmas (isolate the loop from any schedule recompute) + its bf16 init/cap.
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let scheduler = FlowMatchEuler { sigmas };
    let init = bf16(g.require("init").unwrap());
    let cap = bf16(g.require("cap_feats").unwrap());
    let latents = denoise(&transformer, &scheduler, init, &cap)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let golden = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), golden.shape(), "final latents shape");
    // mean-rel is the stable metric (peak_rel is a single high-dynamic-range outlier); print both.
    let a = latents.reshape(&[-1]).unwrap();
    let b = golden.reshape(&[-1]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mabs: f32 = ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32;
    let mean_rel: f32 =
        xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32 / mabs;
    println!(
        "Q{bits} final_latents: mean_rel={mean_rel:.3e} peak_rel={:.3e} shape={:?}",
        peak_rel(&latents, golden),
        latents.shape()
    );
    assert!(
        mean_rel < max_latent_mean_rel,
        "Q{bits} final latents diverged from fork-Q{bits}: mean_rel {mean_rel:.3e} >= {max_latent_mean_rel:.3e}"
    );

    // Dense-VAE decode of the Rust Q-latents vs the fork's quantized decode, compared as RGB8. (The
    // VAE mid-block-attention Linears the fork also quantizes are pixel-irrelevant here: decoding
    // the fork's *exact* Q8 latents through the dense Rust VAE reproduces the fork's quantized-VAE
    // decode to 0 px>8 — measured during the sc-2532 investigation.)
    let unpacked = unpack_latents(&latents).unwrap();
    let sh = unpacked.shape();
    let latent5 = unpacked.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap();
    let decoded = vae
        .decode(&latent5)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let golden_dec = g.require("decoded").unwrap();

    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden_dec).unwrap();

    // Save the Rust Q-render next to the fork's golden PNG for visual inspection (like the dense
    // `rust_fox.png`).
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/rust_q{bits}.png"));
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "Q{bits} pixels >8 apart: {:.3}% ({} / {}); saved {}",
        frac * 100.0,
        differ,
        img.pixels.len(),
        out_path.display()
    );
    assert!(
        frac < max_px_frac,
        "Q{bits}: too many divergent pixels: {:.3}% >= {:.3}%",
        frac * 100.0,
        max_px_frac * 100.0
    );
    println!("✓ Q{bits} pipeline matches fork-Q{bits}");
}

/// sc-2532: prove the Q8 quantization is byte-identical to the fork on a **real bf16 model weight**
/// (the existing `quant_parity.rs` covers an f32 weight; the model quantizes bf16). Quantizing the
/// same `layers.0.attention.to_q` weight with mlx-rs reproduces the fork's `mx.quantize` wq/scales/
/// biases exactly and `quantized_matmul` to 0 — so the Q8 e2e residual is the Q8 mode's sensitivity,
/// not a packing/qmm difference. Golden from `tools/dump_z_image_q8_pack_probe.py`.
#[test]
#[ignore = "needs the zq8_pack_probe golden (tools/dump_z_image_q8_pack_probe.py)"]
fn q8_packing_byte_identical_to_fork() {
    use mlx_rs::ops::{quantize, quantized_matmul};
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../tools/golden/zq8_pack_probe.safetensors"
    );
    // The probe golden is a gitignored local fixture (regenerable via
    // tools/dump_z_image_q8_pack_probe.py). Skip rather than panic-`unwrap()` when it's absent, like
    // the other optional-golden real-weight tests — Q8 packing is also exercised end-to-end by
    // `transformer_q8_pipeline_matches_fork`.
    let Ok(g) = Weights::from_file(path) else {
        eprintln!(
            "skip q8_packing_byte_identical_to_fork: {path} not present \
             (generate it with tools/dump_z_image_q8_pack_probe.py); \
             Q8 packing is also covered by transformer_q8_pipeline_matches_fork"
        );
        return;
    };
    // f32→bf16 of an already-bf16 value is exact, so this is the fork's exact quantized weight.
    let w = bf16(g.require("w").unwrap());
    let x = bf16(g.require("x").unwrap());

    let (wq, scales, biases) = quantize(&w, 64, 8).unwrap();
    let wq_match = mlx_rs::ops::eq(&wq, g.require("wq").unwrap())
        .unwrap()
        .all(None)
        .unwrap()
        .item::<bool>();
    let sc_pr = peak_rel(
        &scales.as_dtype(Dtype::Float32).unwrap(),
        g.require("scales").unwrap(),
    );
    let bi_pr = peak_rel(
        &biases.as_dtype(Dtype::Float32).unwrap(),
        g.require("biases").unwrap(),
    );
    let qmm = quantized_matmul(&x, &wq, &scales, &biases, true, 64, 8).unwrap();
    let qmm_pr = peak_rel(
        &qmm.as_dtype(Dtype::Float32).unwrap(),
        g.require("qmm").unwrap(),
    );
    println!(
        "Q8 packing vs fork: wq_exact={wq_match} scales_pr={sc_pr:.2e} biases_pr={bi_pr:.2e} qmm_pr={qmm_pr:.2e}"
    );
    assert!(
        wq_match,
        "Q8 packed weight is not byte-identical to the fork"
    );
    assert!(
        sc_pr == 0.0 && bi_pr == 0.0,
        "Q8 scales/biases differ from the fork"
    );
    assert!(
        qmm_pr < 1e-6,
        "Q8 quantized_matmul differs from the fork: {qmm_pr:.2e}"
    );
}

#[test]
#[ignore = "needs real Z-Image weights + Q8 golden @1024² (QUANTIZE=8 ZIMAGE_W=1024 ZIMAGE_H=1024 dump_z_image_golden.py)"]
fn transformer_q8_pipeline_matches_fork() {
    // Measured (1024², fox, seed 42) AFTER sc-2604: latent mean_rel 3.6e-3, px>8 0.028% (was
    // 2.0e-2 / 0.78% when the f32 checkpoint was quantized as f32 — see `quantize` in adapters.rs).
    // Now at the dense-path floor; thresholds catch a regression to the old f32-quantize residual.
    q_pipeline_matches_fork(Q8_GOLDEN, 8, 1e-2, 0.005);
}

#[test]
#[ignore = "needs real Z-Image weights + Q4 golden @1024² (QUANTIZE=4 ZIMAGE_W=1024 ZIMAGE_H=1024 dump_z_image_golden.py)"]
fn transformer_q4_pipeline_matches_fork() {
    // Measured (1024², fox, seed 42) AFTER sc-2604: latent mean_rel 4.5e-3, px>8 0.137% (was
    // 1.6e-2 / 0.64% with the f32-quantize bug). Now at the dense-path floor.
    q_pipeline_matches_fork(Q4_GOLDEN, 4, 1e-2, 0.005);
}

/// sc-2532: the **full public product path** — `load("z_image_turbo", spec.with_quant(Q)).generate()`
/// — end-to-end at the golden's (prompt, seed, size), vs the fork's `quantize=N` render, saving
/// `rust_q{bits}_full.png` for inspection. This exercises the whole quantized model (transformer +
/// text encoder + VAE), matching the fork's `nn.quantize`. It's the honest e2e parity gate (the
/// cap_feats-fed `q_pipeline_matches_fork` can't see a text-encoder scope gap). Quantizing the text
/// encoder is what drops Q4 from ~18% (transformer-only, dense TE) to sub-1% here (1024²).
fn q_full_generate_renders(golden_path: &str, quant: mlx_gen::Quant, bits: i32, max_px_frac: f32) {
    let g = Weights::from_file(golden_path).unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    // STALE-GOLDEN GUARD (sc-3007): same schedule-drift check as the dense full-pipeline test — the
    // public quantized `generate` path recomputes `for_static_shift(steps, 3.0)`, so a golden dumped
    // at a different shift would silently diverge here too.
    assert_golden_schedule_is_production(&g, steps, w, h, seed, &prompt);

    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_quant(quant);
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/rust_q{bits}_full.png"));
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
        "Q{bits} full generate vs fork-Q{bits}: {:.3}% px>8 ({}x{}); saved {}",
        frac * 100.0,
        img.width,
        img.height,
        out_path.display()
    );
    assert!(
        frac < max_px_frac,
        "Q{bits} full generate diverged: {:.3}%",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real Z-Image weights + Q8 golden @1024² (QUANTIZE=8 ZIMAGE_W=1024 ZIMAGE_H=1024)"]
fn q8_full_generate_renders() {
    // Measured (1024²) AFTER sc-2604: 0.024% px>8 vs fork-Q8 (whole-model quant: transformer + text
    // encoder + VAE — all f32 on disk, all fixed by the bf16-cast-before-quantize). Was 0.81%.
    q_full_generate_renders(Q8_GOLDEN, mlx_gen::Quant::Q8, 8, 0.005);
}

#[test]
#[ignore = "needs real Z-Image weights + Q4 golden @1024² (QUANTIZE=4 ZIMAGE_W=1024 ZIMAGE_H=1024)"]
fn q4_full_generate_renders() {
    // Measured (1024²) AFTER sc-2604: 0.088% px>8 vs fork-Q4 (whole-model quant). Was 0.73%
    // (transformer-only-dense-TE was ~18%, fixed in sc-2532; the f32-quantize residual in sc-2604).
    q_full_generate_renders(Q4_GOLDEN, mlx_gen::Quant::Q4, 4, 0.005);
}
