//! S6 end-to-end T2V parity vs the reference `generate_av.py` video path (sc-2679 S6).
//!
//! Two gates, both against the committed golden (`tests/fixtures/ltx_e2e_golden.safetensors`, from
//! `tools/dump_ltx_e2e_{te,golden}.py` — a real prompt at 256×256, 9 frames):
//!
//!  1. `tokenizer_matches_reference` — the Rust Gemma tokenizer ([`LtxTokenizer`]) reproduces the
//!     reference `AutoTokenizer` `input_ids` byte-for-byte (left-pad to 128, `<bos>`, no EOS). Needs
//!     only the Gemma `tokenizer.json` (`$LTX_GEMMA_DIR` / HF cache), no model weights.
//!  2. `e2e_frames_match_reference` (`#[ignore]`, ~22 GB) — the pipeline from the reference's
//!     **injected** `video_embeddings` + noise (the Gemma backbone need not be reloaded; the text
//!     encoder is gated by S1 `te_parity`, the tokenizer by gate 1). **GATES** the full 2-stage
//!     F32Q8 e2e (position grid + 2× upsample + re-noise bit-exact, stage-2 from the reference's exact
//!     input bit-exact, full final latents bit-exact, frames px>8 = 0.00% — the sc-2679 S6 acceptance).
//!
//! **Golden is mlx 0.31.2** (the Q8 path), f32 regime (`Precision::F32Q8`). The distilled **stage-1**
//! (8 steps from pure noise) is **chaos-sensitive** (like SDXL's ancestral sampler): any per-forward
//! seed is amplified into a large latent divergence. sc-2842 drove the per-forward DiT to **bit-exact**
//! by fixing the last seed — the adaLN timestep sinusoid was built on the host in f64 then cast to f32,
//! while the reference `get_timestep_embedding` builds it in MLX f32 (~1e-7/elem; invisible in bf16 but
//! it modulates every block in F32Q8, compounding over 48 layers to ~0.9% → chaos-amplified to ~31%).
//! With the table in MLX f32 the per-forward is bit-exact, stage-1 deterministic-identical, and the
//! e2e **pixel-exact**. Honors "divergence is not rounding": the gap was a real, named, fixed op — not
//! irreducible f32 accumulation.
//!
//! Run: `LTX_BASE_DIR=… LTX_GEMMA_DIR=… cargo test -p mlx-gen-ltx --test e2e_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, gt, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::{LtxConfig, LtxVaeConfig};
use mlx_gen_ltx::pipeline::{
    decode_to_frames, denoise, generate_t2v_latents, renoise, STAGE2_SIGMAS,
};
use mlx_gen_ltx::positions::create_position_grid;
use mlx_gen_ltx::tokenizer::LtxTokenizer;
use mlx_gen_ltx::transformer::{LtxDiT, Precision};
use mlx_gen_ltx::upsampler::{upsample_latents, LatentUpsampler};
use mlx_gen_ltx::vae::LtxVideoVae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_e2e_golden.safetensors"
);
/// The reference's **native bf16+Q8** e2e golden (`LTX_BF16=1 dump_ltx_e2e_golden.py`) — the
/// production-precision target ([`Precision::Bf16Q8`] DiT + bf16 upsampler/statistics; f32 VAE decode).
const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_e2e_golden_bf16.safetensors"
);
/// The prompt PHASE A (`dump_ltx_e2e_te.py`) tokenized.
const PROMPT: &str = "A cat playing a grand piano on a city rooftop at sunset.";
const MAX_TOKENS: usize = 128;

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn gemma_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_GEMMA_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    let base = std::path::PathBuf::from(home)
        .join(".cache/huggingface/hub/models--mlx-community--gemma-3-12b-it-bf16/snapshots");
    std::fs::read_dir(&base)
        .expect("gemma snapshot dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
        .expect("a gemma snapshot")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let denom = max_op(abs(f32(want)).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), f32(want)).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(f32(want)).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

/// Fraction of uint8 pixels differing by > 8 (the e2e px>8 metric).
fn px_gt8(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let over = gt(&diff, Array::from_int(8)).unwrap();
    sum(over.as_dtype(Dtype::Float32).unwrap(), None)
        .unwrap()
        .item::<f32>()
        / (got.size() as f32)
}

#[test]
#[ignore = "needs the Gemma tokenizer.json (set LTX_GEMMA_DIR or HF cache)"]
fn tokenizer_matches_reference() {
    let tok = LtxTokenizer::from_dir(&gemma_dir()).expect("load gemma tokenizer.json");
    let (ids, mask) = tok.encode(PROMPT, MAX_TOKENS).expect("encode");
    let g = Weights::from_file(GOLDEN).expect("e2e golden");
    let want = g.require("input_ids").unwrap();
    assert_eq!(ids.shape(), &[1, MAX_TOKENS as i32], "ids shape");
    let got = ids.as_slice::<i32>();
    let exp = want.as_dtype(Dtype::Int32).unwrap();
    let exp = exp.as_slice::<i32>();
    assert_eq!(
        got, exp,
        "Gemma input_ids must match the reference AutoTokenizer"
    );
    // Left-pad: leading pads are mask 0, the valid tail mask 1; valid count = non-pad ids.
    let valid: i32 = mask.as_slice::<i32>().iter().sum();
    let nonzero = got.iter().filter(|&&x| x != 0).count() as i32;
    eprintln!("tokenizer: {valid} valid tokens (bos+{} prompt)", valid - 1);
    assert_eq!(
        valid, nonzero,
        "attention mask marks exactly the non-pad tokens"
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer (~20 GB) + upsampler + vae_decoder"]
fn e2e_frames_match_reference() {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("config");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit = LtxDiT::from_weights(&tw, &cfg, Precision::F32Q8).expect("dit");
    let uw = Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler");
    let up = LatentUpsampler::from_weights(&uw).expect("upsampler");
    let vcfg = LtxVaeConfig::from_model_dir(&dir).expect("vae cfg");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae");
    let vae = LtxVideoVae::from_weights(&dec, None, &vcfg).expect("vae");

    let g = Weights::from_file(GOLDEN).expect("e2e golden");
    let ctx = g.require("video_embeddings").unwrap();
    let (mean, std) = (latent_stat(&dec, "mean"), latent_stat(&dec, "std"));

    // --- GATED: the verified-correct components (the port reproduces these to within tolerance). ---

    // Position grid (generate()'s `create_position_grid`) == the reference `create_video_position_grid`.
    let pos1 = create_position_grid(1, 2, 4, 4);
    let pos2 = create_position_grid(1, 2, 8, 8);
    assert!(
        peak_rel(&pos1, g.require("stage1_positions").unwrap()) == 0.0,
        "stage1 positions"
    );
    assert!(
        peak_rel(&pos2, g.require("stage2_positions").unwrap()) == 0.0,
        "stage2 positions"
    );

    // Upsample + re-noise are bit-exact (S4 + the formula) from the reference's stage-1 latents.
    let ups = upsample_latents(g.require("stage1_out").unwrap(), &up, &mean, &std).expect("ups");
    assert!(
        peak_rel(&ups, g.require("upsampled").unwrap()) == 0.0,
        "upsample bit-exact"
    );
    let rn = renoise(
        g.require("upsampled").unwrap(),
        g.require("stage2_noise").unwrap(),
        STAGE2_SIGMAS[0],
    )
    .expect("renoise");
    assert!(
        peak_rel(&rn, g.require("renoised").unwrap()) == 0.0,
        "renoise bit-exact"
    );

    // Stage-2 denoise (3 steps) from the reference's exact `renoised` input — **bit-exact** now that
    // the per-forward DiT is bit-exact (sc-2842: the timestep freq table runs in MLX f32, not host
    // f64). Was 0.16% when the host-f64 seed accumulated over the 3 steps.
    let s2 = denoise(
        &dit,
        g.require("renoised").unwrap(),
        ctx,
        &pos2,
        &STAGE2_SIGMAS,
        None,
        &mut |_| {},
    )
    .expect("stage2");
    let s2_mr = mean_rel(&s2, g.require("final_latents").unwrap());
    eprintln!("stage2 (from ref input) mean_rel = {s2_mr:.3e}");
    assert!(
        s2_mr == 0.0,
        "stage2 denoise from correct input must be bit-exact: {s2_mr:.3e}"
    );

    // --- GATED: the full 2-stage F32Q8 e2e — the sc-2679 S6 / task #2701 acceptance. ---
    // The distilled stage-1 (8 steps from pure noise) is chaos-sensitive (the SDXL ancestral-sampler
    // phenomenon), so it requires a fully **bit-exact** per-forward DiT. sc-2842 found and fixed the
    // last seed: the adaLN timestep sinusoid was tabulated on the host in f64 then cast to f32, while
    // the reference `get_timestep_embedding` builds it in MLX f32 — a ~1e-7/elem mismatch that, fed
    // into the f32 adaLN modulating every block, compounded over 48 layers to ~0.9% and then chaos-
    // amplified to ~31% px>8. With the table in MLX f32 the per-forward is bit-exact, stage-1 is
    // deterministic-identical, and the e2e is **pixel-exact** (px>8 = 0.00%). Acceptance: px>8 < 1%.
    let latents = generate_t2v_latents(
        &dit,
        &up,
        g.require("stage1_noise").unwrap(),
        &pos1,
        g.require("stage2_noise").unwrap(),
        &pos2,
        ctx,
        &mean,
        &std,
        &mut |_| {},
    )
    .expect("generate_t2v_latents");
    let fmr = mean_rel(&latents, g.require("final_latents").unwrap());
    let frames = decode_to_frames(&vae, &latents).expect("decode");
    let want_frames = g.require("frames").unwrap();
    assert_eq!(frames.shape(), want_frames.shape(), "frame shape");
    assert_eq!(frames.dtype(), Dtype::Uint8);
    let px = px_gt8(&frames, want_frames);
    eprintln!(
        "FULL e2e: final latents mean_rel = {fmr:.3e}, frames px>8 = {:.2}%",
        px * 100.0
    );
    assert!(
        fmr == 0.0,
        "full e2e final latents must be bit-exact (per-forward is bit-exact): {fmr:.3e}"
    );
    assert!(
        px < 1e-2,
        "e2e frames px>8 {:.2}% exceeds the 1% acceptance (sc-2679 S6)",
        px * 100.0
    );
}

/// The **production-precision** e2e: the reference's native bf16+Q8 path ([`Precision::Bf16Q8`] DiT +
/// bf16 upsampler/statistics) vs the bf16 golden. The DiT per-forward is bit-exact (sc-2842, incl. the
/// bf16 timestep-scale fix), so the bf16 stage-1/stage-2 latents are bit-exact too; the VAE decode runs
/// f32 (the quality island, post-sampling, no chaos amplification), so frames are pixel-parity within
/// the <1% acceptance rather than byte-identical.
#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer (~20 GB) + upsampler + vae_decoder"]
fn e2e_frames_match_reference_bf16() {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("config");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit = LtxDiT::from_weights(&tw, &cfg, Precision::Bf16Q8).expect("dit");
    let uw = Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler");
    let up = LatentUpsampler::from_weights(&uw).expect("upsampler");
    let vcfg = LtxVaeConfig::from_model_dir(&dir).expect("vae cfg");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae");
    let vae = LtxVideoVae::from_weights(&dec, None, &vcfg).expect("vae");

    let g = Weights::from_file(GOLDEN_BF16).expect("e2e bf16 golden");
    let ctx = g.require("video_embeddings").unwrap(); // bf16 (the DiT keeps it bf16)
                                                      // bf16 latent statistics — the upsampler + re-normalize run native bf16 in the production path.
    let (mean, std) = (
        latent_stat_dt(&dec, "mean", Dtype::Bfloat16),
        latent_stat_dt(&dec, "std", Dtype::Bfloat16),
    );
    let pos1 = create_position_grid(1, 2, 4, 4);
    let pos2 = create_position_grid(1, 2, 8, 8);

    // Upsample + re-noise are bit-exact (bf16) from the reference's stage-1 latents.
    let ups = upsample_latents(g.require("stage1_out").unwrap(), &up, &mean, &std).expect("ups");
    assert!(
        peak_rel(&ups, g.require("upsampled").unwrap()) == 0.0,
        "bf16 upsample bit-exact"
    );
    let rn = renoise(
        g.require("upsampled").unwrap(),
        g.require("stage2_noise").unwrap(),
        STAGE2_SIGMAS[0],
    )
    .expect("renoise");
    assert!(
        peak_rel(&rn, g.require("renoised").unwrap()) == 0.0,
        "bf16 renoise bit-exact"
    );

    // Stage-2 (3 steps) from the reference's exact bf16 input — bit-exact (the bf16 per-forward is).
    let s2 = denoise(
        &dit,
        g.require("renoised").unwrap(),
        ctx,
        &pos2,
        &STAGE2_SIGMAS,
        None,
        &mut |_| {},
    )
    .expect("stage2");
    let s2_mr = mean_rel(&s2, g.require("final_latents").unwrap());
    eprintln!("bf16 stage2 (from ref input) mean_rel = {s2_mr:.3e}");
    assert!(
        s2_mr == 0.0,
        "bf16 stage2 from correct input must be bit-exact: {s2_mr:.3e}"
    );

    // Full 2-stage bf16 e2e → frames.
    let latents = generate_t2v_latents(
        &dit,
        &up,
        g.require("stage1_noise").unwrap(),
        &pos1,
        g.require("stage2_noise").unwrap(),
        &pos2,
        ctx,
        &mean,
        &std,
        &mut |_| {},
    )
    .expect("generate_t2v_latents");
    let fmr = mean_rel(&latents, g.require("final_latents").unwrap());
    let frames = decode_to_frames(&vae, &latents).expect("decode");
    let want_frames = g.require("frames").unwrap();
    assert_eq!(frames.shape(), want_frames.shape(), "frame shape");
    assert_eq!(frames.dtype(), Dtype::Uint8);
    let px = px_gt8(&frames, want_frames);
    eprintln!(
        "bf16 FULL e2e: final latents mean_rel = {fmr:.3e}, frames px>8 = {:.2}%",
        px * 100.0
    );
    assert!(
        fmr == 0.0,
        "bf16 full e2e final latents must be bit-exact (bf16 per-forward is): {fmr:.3e}"
    );
    assert!(
        px < 1e-2,
        "bf16 e2e frames px>8 {:.2}% exceeds the 1% acceptance",
        px * 100.0
    );
}

/// Load a VAE `per_channel_statistics.{mean,std}` at `dt` (the upsampler's latent norm). f32 for the
/// quality path; bf16 for the production path (the upsampler + re-norm then run native bf16).
fn latent_stat_dt(dec: &Weights, which: &str, dt: Dtype) -> Array {
    dec.require(&format!("per_channel_statistics.{which}"))
        .unwrap()
        .as_dtype(dt)
        .unwrap()
}

fn latent_stat(dec: &Weights, which: &str) -> Array {
    latent_stat_dt(dec, which, Dtype::Float32)
}
