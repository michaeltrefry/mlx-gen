//! S5 parity gate: the dual-expert **MoE** denoise loop (`pipeline::denoise_moe`) must reproduce the
//! `mlx_video` reference's `is_dual` boundary-switched loop.
//!
//! Self-contained committed fixture (`tools/dump_s5_fixtures.py`): two tiny seeded dense `WanModel`s
//! (high/low noise, different seeds) + a tiny z16 `WanVAE`, with injected noise + context, run
//! through the reference's per-step expert select (`t ≥ boundary·num_train`), per-expert embeds /
//! cross-KV / RoPE / guidance. The fixture's 4 steps route `high@999, high@937, low@833, low@624`
//! — both experts exercised across the boundary (875). Runs in CI, no real weights.
//!
//! Same bf16 cross-build envelope as S4 (the DiT runs bf16); gated at 2e-2.

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::denoise_moe;
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{Expert, WanTransformer, WanVae};

fn load(name: &str) -> Weights {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_s5_fixtures.py)"))
}

/// The tiny dual config the fixture was dumped with (`dump_s5_fixtures.py`).
fn tiny_cfg() -> WanModelConfig {
    let mut c = WanModelConfig::wan21_t2v_1_3b();
    c.dim = 128;
    c.num_heads = 1;
    c.num_layers = 2;
    c.ffn_dim = 256;
    c.freq_dim = 256;
    c.text_dim = 32;
    c.text_len = 8;
    c.in_dim = 16;
    c.out_dim = 16;
    c.vae_z_dim = 16;
    c.boundary = 0.875;
    c.num_train_timesteps = 1000;
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

fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

#[test]
fn wan_moe_denoise_matches_reference() {
    let low_w = load("s5_low.safetensors");
    let high_w = load("s5_high.safetensors");
    let cfg = tiny_cfg();

    let low_dit = WanTransformer::from_weights(&low_w, &cfg).expect("low DiT");
    let high_dit = WanTransformer::from_weights(&high_w, &cfg).expect("high DiT");
    let vae = WanVae::from_weights(&low_w).expect("VAE");

    let ctx_cond = low_w.require("ctx_cond").unwrap();
    let ctx_uncond = low_w.require("ctx_uncond").unwrap();
    let init_noise = low_w.require("init_noise").unwrap();

    // Each expert embeds the shared raw context through its own text_embedding.
    let low = Expert {
        transformer: &low_dit,
        ctx_cond: low_dit.embed_text(ctx_cond).unwrap(),
        ctx_uncond: Some(low_dit.embed_text(ctx_uncond).unwrap()),
        guidance: 3.0,
    };
    let high = Expert {
        transformer: &high_dit,
        ctx_cond: high_dit.embed_text(ctx_cond).unwrap(),
        ctx_uncond: Some(high_dit.embed_text(ctx_uncond).unwrap()),
        guidance: 4.0,
    };
    let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32; // 875

    let latents = denoise_moe(
        &low,
        &high,
        boundary_timestep,
        SolverKind::Euler,
        cfg.num_train_timesteps,
        4,
        5.0,
        init_noise,
        &mut |_| {},
    )
    .expect("denoise_moe");

    let exp_lat = low_w.require("final_latents").unwrap();
    assert_eq!(latents.shape(), exp_lat.shape(), "final latent shape");
    let (la_max, la_mr) = diff(latents.as_slice::<f32>(), exp_lat.as_slice::<f32>());
    println!(
        "[moe latents] shape={:?} max|Δ|={la_max:.3e} mean_rel={la_mr:.3e}",
        latents.shape()
    );

    let video = vae
        .decode(&latents.reshape(&prepend1(latents.shape())).unwrap())
        .unwrap();
    let exp_vid = low_w.require("video").unwrap();
    assert_eq!(video.shape(), exp_vid.shape(), "video shape");
    let (vid_max, vid_mr) = diff(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    println!(
        "[moe video]   shape={:?} max|Δ|={vid_max:.3e} mean_rel={vid_mr:.3e}",
        video.shape()
    );

    // bf16 DiT cross-build envelope (see S4); a routing/boundary bug gives mean_rel ~O(1).
    assert!(la_mr < 2e-2, "moe latents diverged: mean_rel={la_mr:.3e}");
    assert!(vid_mr < 2e-2, "moe video diverged: mean_rel={vid_mr:.3e}");
}
