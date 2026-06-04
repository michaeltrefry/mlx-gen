//! sc-2680 parity gate: the image-conditioned **TI2V mask-blend** denoise (`pipeline::denoise_ti2v`
//! plus the DiT's per-token-timestep `forward_tokens`) must reproduce the `mlx_video` reference's
//! `is_i2v_mask_blend` loop.
//!
//! Self-contained committed fixture (`tools/dump_ti2v_fixtures.py`): a tiny seeded dense `WanModel`,
//! with **injected** context + initial noise + encoded-image latent `z_img`, run through the
//! reference's per-token-timestep CFG loop (Euler, first-frame tokens frozen at `t=0`, mask
//! re-applied each step). Runs in CI, no real weights. Also checks the Rust `build_ti2v_mask` mask +
//! per-token mask against the reference `build_i2v_mask`, and `ti2v_blend_init` against the reference
//! `(1−mask)·z_img + mask·noise`.
//!
//! The DiT runs bf16 (the production regime), so the final-latent gap is the known cross-build bf16
//! kernel delta (MLX 0.31.1+patches vs the reference's 0.31.2) accumulated over the loop — bounded,
//! not a code bug (same envelope as the S4 dense gate). The mask logic is gated bit-tight separately.

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::{build_ti2v_mask, denoise_ti2v, ti2v_blend_init};
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::WanTransformer;

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ti2v_pipeline.safetensors"
    );
    Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_ti2v_fixtures.py)"))
}

/// The tiny dense config the fixture was dumped with (mirrors `dump_ti2v_fixtures.py` / S4).
fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan21_t2v_1_3b();
    c.dim = 128;
    c.num_heads = 1; // head_dim 128
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 16;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c.dual_model = false;
    c
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

#[test]
fn build_ti2v_mask_matches_reference() {
    let w = fixture();
    // Fixture dims: z=16, t_lat=2, h_lat=w_lat=2, patch (1,2,2).
    let (mask, tokens) = build_ti2v_mask(16, 2, 2, 2, (1, 2, 2));
    let exp_mask = w.require("mask").unwrap();
    let exp_tokens = w.require("mask_tokens").unwrap();
    assert_eq!(mask.shape(), exp_mask.shape(), "mask shape");
    assert_eq!(tokens.shape(), exp_tokens.shape(), "mask_tokens shape");
    assert_eq!(
        mask.as_slice::<f32>(),
        exp_mask.as_slice::<f32>(),
        "mask must match reference build_i2v_mask"
    );
    assert_eq!(
        tokens.as_slice::<f32>(),
        exp_tokens.as_slice::<f32>(),
        "mask_tokens must match reference"
    );
}

#[test]
fn wan_ti2v_mask_blend_matches_reference() {
    let w = fixture();
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");

    let ctx_cond = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
    let ctx_uncond = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
    let init_noise = w.require("init_noise").unwrap();
    let z_img = w.require("z_img").unwrap();
    let mask = w.require("mask").unwrap();
    let mask_tokens = w.require("mask_tokens").unwrap();

    // Blend the noise init (gates ti2v_blend_init): (1−mask)·z_img + mask·noise.
    let init_latents = ti2v_blend_init(z_img, mask, init_noise).unwrap();

    let mut steps_seen = 0usize;
    let latents = denoise_ti2v(
        &dit,
        SolverKind::Euler,
        cfg.num_train_timesteps,
        4,   // steps
        5.0, // shift
        3.0, // guidance
        &ctx_cond,
        Some(&ctx_uncond),
        &init_latents,
        z_img,
        mask,
        mask_tokens,
        &mut |_| steps_seen += 1,
    )
    .expect("denoise_ti2v");
    assert_eq!(steps_seen, 4, "progress callback fired per step");

    let exp = w.require("final_latents").unwrap();
    assert_eq!(latents.shape(), exp.shape(), "final latent shape");
    let (max_abs, mean_rel) = diff(latents.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[ti2v latents] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        latents.shape()
    );

    // The first latent temporal frame must stay frozen to z_img (mask-blend invariant).
    let lat = latents.as_slice::<f32>();
    let zexp = z_img.as_slice::<f32>(); // [16,1,2,2] = 64 vals (frame 0 for each channel)
    let (t_lat, plane) = (2usize, 4usize); // h_lat·w_lat
    let mut frame0_max = 0f32;
    for c in 0..16 {
        for p in 0..plane {
            let got = lat[c * t_lat * plane + p]; // temporal index 0
            frame0_max = frame0_max.max((got - zexp[c * plane + p]).abs());
        }
    }
    assert!(
        frame0_max < 1e-5,
        "first frame must stay pinned to z_img (max|Δ|={frame0_max:.3e})"
    );

    // Same bf16 cross-build envelope as S4 (gate at 2e-2; a logic bug gives mean_rel ~O(1)).
    assert!(
        mean_rel < 2e-2,
        "ti2v latents diverged: mean_rel={mean_rel:.3e}"
    );
}
