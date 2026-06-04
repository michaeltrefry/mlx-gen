//! I2V single-image conditioning parity vs the reference `generate.py` / `generate_av.py` video path
//! (sc-2685). Gates the I2V-new code — `apply_conditioning`, the stage noiser, per-token `σ·mask`
//! timesteps, `apply_denoise_mask`, and conditioned-frame preservation — plus the full 2-stage
//! conditioned pipeline → frames, against the committed golden (`tests/fixtures/ltx_i2v_golden.safetensors`,
//! from `tools/dump_ltx_i2v_golden.py`).
//!
//! The conditioning **image latent** is a deterministic synthetic latent injected by the dump (the
//! VAE *encoder* is gated separately by `vae_parity::encode_matches_reference`), so this isolates the
//! conditioning + conditioned-denoise math. Strength = 1.0, frame 0 (the reference defaults) → the
//! conditioned frame is fully pinned: its value in the final latent must equal the clean image latent.
//!
//! Both precisions, like `e2e_parity`: `f32` (`Precision::F32Q8`, the quality gate) and the native
//! `bf16+Q8` production path (`Precision::Bf16Q8`). The per-forward DiT is bit-exact (sc-2842), so the
//! conditioned latents are bit-exact and frames are pixel-parity (px>8 < 1%).
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test i2v_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, gt, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::conditioning::apply_conditioning;
use mlx_gen_ltx::config::LtxConfig;
use mlx_gen_ltx::pipeline::{decode_to_frames, denoise, generate_i2v_latents, STAGE1_SIGMAS};
use mlx_gen_ltx::positions::create_position_grid;
use mlx_gen_ltx::transformer::{LtxDiT, Precision};
use mlx_gen_ltx::upsampler::{upsample_latents, LatentUpsampler};
use mlx_gen_ltx::vae::LtxVideoVae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_i2v_golden.safetensors"
);
const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_i2v_golden_bf16.safetensors"
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

fn px_gt8(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), f32(want)).unwrap()).unwrap();
    let over = gt(&diff, Array::from_int(8)).unwrap();
    sum(over.as_dtype(Dtype::Float32).unwrap(), None)
        .unwrap()
        .item::<f32>()
        / (got.size() as f32)
}

/// Temporal frame slice `x[:, :, i:i+1]` (axis 2).
fn frame(x: &Array, i: i32) -> Array {
    x.take_axis(Array::from_int(i), 2).unwrap()
}

fn latent_stat(dec: &Weights, which: &str, dt: Dtype) -> Array {
    dec.require(&format!("per_channel_statistics.{which}"))
        .unwrap()
        .as_dtype(dt)
        .unwrap()
}

/// Shared I2V gate: build the model + the conditioning state from the golden's injected samples, gate
/// `apply_conditioning` + the noiser + the conditioned denoise + the full e2e against `golden_path`.
fn run_i2v_gate(golden_path: &str, prec: Precision, stat_dt: Dtype) {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("config");
    let tw = Weights::from_file(dir.join("transformer.safetensors")).expect("transformer");
    let dit = LtxDiT::from_weights(&tw, &cfg, prec).expect("dit");
    let uw = Weights::from_file(dir.join("upsampler.safetensors")).expect("upsampler");
    let up = LatentUpsampler::from_weights(&uw).expect("upsampler");
    let vcfg = mlx_gen_ltx::config::LtxVaeConfig::from_model_dir(&dir).expect("vae cfg");
    let dec = Weights::from_file(dir.join("vae_decoder.safetensors")).expect("vae");
    let vae = LtxVideoVae::from_weights(&dec, None, &vcfg).expect("vae");

    let g = Weights::from_file(golden_path).expect("i2v golden (run tools/dump_ltx_i2v_golden.py)");
    let ctx = g.require("video_embeddings").unwrap();
    let (mean, std) = (
        latent_stat(&dec, "mean", stat_dt),
        latent_stat(&dec, "std", stat_dt),
    );
    let strength = f32(g.require("strength").unwrap()).as_slice::<f32>()[0];
    let frame_idx = g
        .require("frame_idx")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()[0];

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

    let stage1_image = g.require("stage1_image_latent").unwrap();
    let stage1_noise = g.require("stage1_noise").unwrap();

    // --- apply_conditioning: mask / clean / latent-frame placement bit-exact vs the reference. ---
    let zeros1 = Array::zeros::<f32>(stage1_noise.shape())
        .unwrap()
        .as_dtype(stage1_noise.dtype())
        .unwrap();
    let st1 =
        apply_conditioning(&zeros1, stage1_image, frame_idx, strength).expect("apply_conditioning");
    assert!(
        peak_rel(&st1.denoise_mask, g.require("stage1_mask").unwrap()) == 0.0,
        "stage1 mask"
    );
    assert!(
        peak_rel(&st1.clean_latent, g.require("stage1_clean").unwrap()) == 0.0,
        "stage1 clean"
    );
    // The conditioned frame's clean latent == the injected image latent (single-frame cond).
    assert!(
        peak_rel(
            &frame(&st1.clean_latent, frame_idx),
            &frame(stage1_image, 0)
        ) == 0.0,
        "conditioned clean frame == image latent"
    );

    // --- The stage noiser: σ₀ = 1.0 (stage 1). ---
    let st1 = st1.noised(stage1_noise, STAGE1_SIGMAS[0]).expect("noiser");
    assert!(
        peak_rel(&st1.latent, g.require("stage1_state_latent").unwrap()) == 0.0,
        "stage1 noiser"
    );
    // strength=1 → the conditioned frame is pinned to the image latent through the noiser.
    assert!(
        peak_rel(&frame(&st1.latent, frame_idx), &frame(stage1_image, 0)) == 0.0,
        "conditioned frame pinned after noiser"
    );

    // --- Stage-1 conditioned denoise: per-token σ·mask + apply_denoise_mask, bit-exact. ---
    let s1 = denoise(
        &dit,
        &st1.latent,
        ctx,
        &pos1,
        &STAGE1_SIGMAS,
        Some(&st1),
        &mut |_| {},
    )
    .expect("stage1 denoise");
    let s1_mr = mean_rel(&s1, g.require("stage1_out").unwrap());
    eprintln!("stage1 conditioned denoise mean_rel = {s1_mr:.3e}");
    assert!(
        s1_mr == 0.0,
        "stage1 conditioned denoise must be bit-exact: {s1_mr:.3e}"
    );
    // Conditioned-frame preservation: frame 0 of the stage-1 output == the image latent.
    assert!(
        peak_rel(&frame(&s1, frame_idx), &frame(stage1_image, 0)) == 0.0,
        "stage1 conditioned frame preserved"
    );

    // --- Upsample bit-exact (S4). ---
    let ups = upsample_latents(&s1, &up, &mean, &std).expect("upsample");
    assert!(
        peak_rel(&ups, g.require("upsampled").unwrap()) == 0.0,
        "upsample bit-exact"
    );

    // --- Full 2-stage I2V e2e (the sc-2685 acceptance) → final latents + frames. ---
    let mut step = 0usize;
    let latents = generate_i2v_latents(
        &dit,
        &up,
        stage1_image,
        stage1_noise,
        &pos1,
        g.require("stage2_image_latent").unwrap(),
        g.require("stage2_noise").unwrap(),
        &pos2,
        ctx,
        &mean,
        &std,
        frame_idx,
        strength,
        &mut |_| step += 1,
    )
    .expect("generate_i2v_latents");
    let fmr = mean_rel(&latents, g.require("final_latents").unwrap());
    // Conditioned-frame preservation through both stages: final frame 0 == the stage-2 image latent.
    let cond_pr = peak_rel(
        &frame(&latents, frame_idx),
        &frame(g.require("stage2_image_latent").unwrap(), 0),
    );
    let frames = decode_to_frames(&vae, &latents).expect("decode");
    let want_frames = g.require("frames").unwrap();
    assert_eq!(frames.shape(), want_frames.shape(), "frame shape");
    assert_eq!(frames.dtype(), Dtype::Uint8);
    let px = px_gt8(&frames, want_frames);
    eprintln!(
        "FULL I2V e2e: final latents mean_rel = {fmr:.3e}, conditioned-frame peak_rel = {cond_pr:.3e}, frames px>8 = {:.2}%",
        px * 100.0
    );
    assert_eq!(
        step,
        STAGE1_SIGMAS.len() - 1 + 3,
        "step callbacks fired per denoise step"
    );
    assert!(
        fmr == 0.0,
        "full I2V e2e final latents must be bit-exact: {fmr:.3e}"
    );
    assert!(
        cond_pr == 0.0,
        "conditioned frame must be preserved exactly in the final latent: {cond_pr:.3e}"
    );
    assert!(
        px < 1e-2,
        "I2V e2e frames px>8 {:.2}% exceeds the 1% acceptance",
        px * 100.0
    );
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer (~20 GB) + upsampler + vae_decoder + the I2V golden"]
fn i2v_frames_match_reference() {
    run_i2v_gate(GOLDEN, Precision::F32Q8, Dtype::Float32);
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer (~20 GB) + upsampler + vae_decoder + the bf16 I2V golden"]
fn i2v_frames_match_reference_bf16() {
    run_i2v_gate(GOLDEN_BF16, Precision::Bf16Q8, Dtype::Bfloat16);
}
