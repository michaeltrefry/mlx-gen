//! S5 two-stage T2V pipeline parity vs the reference `generate_av.py` video path (sc-2679 S5).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `transformer.safetensors` (~20 GB) + `upsampler` +
//! `vae_decoder`. The committed golden (`tests/fixtures/ltx_pipeline_golden.safetensors`, from
//! `tools/dump_ltx_pipeline_golden.py`) holds the reference **f32** 2-stage I/O over injected inputs
//! (initial noise, re-noise sample, synthetic text embeddings, position grids); this test loads the
//! SAME weights and checks the Rust `pipeline::generate_t2v` reproduces the stage-1 latents, the final
//! latents, and the decoded uint8 frames.
//!
//! **The golden MUST be mlx 0.31.2** (the Rust build): `quantized_matmul` changed 0.31.0→0.31.2.
//! Run in the **f32** regime (`Precision::F32Q8`, f32 latents) — gates the pipeline *math* (legacy
//! Euler, re-noise, 2-stage orchestration, uint8 conversion) isolated from bf16 rounding, mirroring
//! the S3b DiT gate. The bf16-production px>8 verdict is S6. Honors "divergence is not rounding":
//! stage-1 + final latents + frames are gated separately to localize any gap.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test pipeline_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, LtxVaeConfig};
use mlx_gen_ltx::pipeline::{
    decode_to_frames, denoise, generate_t2v_latents, renoise, STAGE1_SIGMAS, STAGE2_SIGMAS,
};
use mlx_gen_ltx::transformer::{LtxDiT, Precision};
use mlx_gen_ltx::upsampler::{upsample_latents, LatentUpsampler};
use mlx_gen_ltx::vae::LtxVideoVae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_pipeline_golden.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let denom = max_op(abs(f32(want)).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|` — robust to the output-LayerNorm-amplified massive-activation channels.
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), f32(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// Fraction of uint8 frame pixels differing by more than 8 (the e2e px>8 metric).
fn px_gt8(got: &Array, want: &Array) -> f32 {
    let g = f32(got);
    let w = f32(want);
    let diff = abs(subtract(&g, &w).unwrap()).unwrap();
    let over = mlx_rs::ops::gt(&diff, Array::from_int(8)).unwrap();
    let count = sum(over.as_dtype(Dtype::Float32).unwrap(), None)
        .unwrap()
        .item::<f32>();
    count / (g.size() as f32)
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB) + upsampler + vae_decoder"]
fn pipeline_matches_reference() {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit = LtxDiT::from_weights(&tw, &cfg, Precision::F32Q8).expect("build LtxDiT");

    let uw = Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler");
    let up = LatentUpsampler::from_weights(&uw).expect("build LatentUpsampler");

    let vcfg = LtxVaeConfig::from_model_dir(&dir).expect("vae config");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae_decoder");
    let vae = LtxVideoVae::from_weights(&dec, None, &vcfg).expect("build LtxVideoVae");

    let g = Weights::from_file(GOLDEN).expect("golden (run tools/dump_ltx_pipeline_golden.py)");

    // --- Localize the NEW S5 orchestration (gated tight; the inherited DiT residual is below). ---

    // Stage-1 denoise loop (8 steps) in isolation: the legacy-Euler / flatten / forward / unflatten
    // wiring. Tight — the F32Q8 per-forward residual mostly cancels per-token at this stage's S=8.
    let s1 = denoise(
        &dit,
        g.require("stage1_noise").unwrap(),
        g.require("context").unwrap(),
        g.require("stage1_positions").unwrap(),
        &STAGE1_SIGMAS,
        None,
        &mut |_| {},
    )
    .expect("stage1 denoise");
    let s1_mr = mean_rel(&s1, g.require("stage1_out").unwrap());
    eprintln!("stage1 latents mean_rel = {s1_mr:.3e}");
    assert!(
        s1_mr < 5e-3,
        "stage1 denoise loop diverged: mean_rel {s1_mr:.3e}"
    );

    // The stage transition is exact: upsample (S4 bit-exact) + re-noise (formula) reproduce the
    // reference's `upsampled`/`renoised` byte-for-byte from the golden's stage-1 latents.
    let ups = upsample_latents(
        g.require("stage1_out").unwrap(),
        &up,
        g.require("latent_mean").unwrap(),
        g.require("latent_std").unwrap(),
    )
    .expect("upsample");
    let ups_pr = peak_rel(&ups, g.require("upsampled").unwrap());
    let rn = renoise(
        g.require("upsampled").unwrap(),
        g.require("stage2_noise").unwrap(),
        STAGE2_SIGMAS[0],
    )
    .expect("renoise");
    let rn_pr = peak_rel(&rn, g.require("renoised").unwrap());
    eprintln!("upsample peak_rel = {ups_pr:.3e} | renoise peak_rel = {rn_pr:.3e}");
    assert!(ups_pr == 0.0, "upsample not bit-exact: {ups_pr:.3e}");
    assert!(rn_pr == 0.0, "renoise not bit-exact: {rn_pr:.3e}");

    let mut steps = 0usize;
    let latents = generate_t2v_latents(
        &dit,
        &up,
        g.require("stage1_noise").unwrap(),
        g.require("stage1_positions").unwrap(),
        g.require("stage2_noise").unwrap(),
        g.require("stage2_positions").unwrap(),
        g.require("context").unwrap(),
        g.require("latent_mean").unwrap(),
        g.require("latent_std").unwrap(),
        &mut |_| steps += 1,
    )
    .expect("generate_t2v_latents");
    assert_eq!(steps, 11, "8 stage-1 + 3 stage-2 denoise steps");

    let want_final = g.require("final_latents").unwrap();
    assert_eq!(latents.shape(), want_final.shape(), "final latent shape");
    let (fpr, fmr) = (
        peak_rel(&latents, want_final),
        mean_rel(&latents, want_final),
    );
    eprintln!(
        "final latents peak_rel = {fpr:.3e} mean_rel = {fmr:.3e} shape={:?}",
        latents.shape()
    );

    let frames = decode_to_frames(&vae, &latents).expect("decode_to_frames");
    let want_frames = g.require("frames").unwrap();
    assert_eq!(frames.shape(), want_frames.shape(), "frame shape");
    assert_eq!(frames.dtype(), Dtype::Uint8, "frames uint8");
    let px = px_gt8(&frames, want_frames);
    eprintln!("frames px>8 = {:.4}% ({:?})", px * 100.0, frames.shape());

    // The NEW S5 code is gated tight above (stage-1 0.17%, upsample/renoise bit-exact 0.0). The
    // full-pipeline numbers below carry the **inherited S3b F32Q8 per-forward DiT residual** — the
    // f32 SDPA-accumulation floor (~5.7e-4/block, S3a) amplified by the output LayerNorm to ~0.9%/
    // forward at S=32 (S3b), which compounds over the 11-step 2-stage trajectory (stage-2 from the
    // exact reference input is already ~3% over its 3 steps) plus the chaotic-sampler sensitivity.
    // Measured: final mean_rel ~7.4e-2, frames px>8 ~2.0% at 128² (the known low-res px>8 floor —
    // it shrinks at production res). NOT a pipeline bug; the mechanism is named (divergence-is-not-
    // rounding). The production-resolution bf16 px>8 verdict — and any tightening of the per-forward
    // DiT residual — is S6. Gated loosely here only as a bound on the compounding.
    assert!(
        fmr < 1.2e-1,
        "final latents mean_rel {fmr:.3e} above the F32Q8 floor"
    );
    assert!(
        px < 4e-2,
        "frames px>8 {:.4}% above the low-res F32Q8 floor",
        px * 100.0
    );
}
