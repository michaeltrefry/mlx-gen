//! sc-2853 invariant: the **batched B=2 CFG forward** ([`WanTransformer::forward_cached`] with
//! `batch = 2`) is **bit-identical** to two sequential B=1 forwards (one per CFG branch). This is the
//! correctness precondition for the small-seq CFG optimization — attention never mixes batch
//! elements, so stacking cond + uncond on the batch axis must not change either branch's output.
//!
//! Runs in CI on the tiny seeded S5 weights (no real checkpoint). It also exercises the cached path
//! (`prepare_cross_kv` / `prepare_rope` / `forward_cached`) against the legacy recompute `forward`.

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::WanTransformer;
use mlx_rs::ops::concatenate_axis;

/// The tiny dual config the S5 fixture was dumped with (mirrors `s5_parity.rs::tiny_cfg`).
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

fn load(name: &str) -> Weights {
    let path = format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    Weights::from_file(&path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run dump_s5_fixtures.py)"))
}

fn max_abs(got: &[f32], exp: &[f32]) -> f32 {
    got.iter()
        .zip(exp.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max)
}

#[test]
fn batched_b2_forward_bit_identical_to_two_b1_forwards() {
    let w = load("s5_low.safetensors");
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");

    let latent = w.require("init_noise").unwrap(); // [16, 2, 2, 2]
    let ctx_cond = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap(); // [1, text_len, dim]
    let ctx_uncond = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
    let t = 833.0f32; // a representative integer timestep

    // Legacy B=1 forwards, one per branch (the prior CFG path).
    let cond_b1 = dit.forward(latent, t, &ctx_cond).unwrap();
    let uncond_b1 = dit.forward(latent, t, &ctx_uncond).unwrap();

    // Batched B=2 forward over the stacked [cond, uncond] context.
    let context_batch = concatenate_axis(&[&ctx_cond, &ctx_uncond], 0).unwrap(); // [2, text_len, dim]
    let cross_kv = dit.prepare_cross_kv(&context_batch).unwrap();
    let (cos, sin) = dit.prepare_rope(dit.patch_grid(latent)).unwrap();
    let preds = dit
        .forward_cached(latent, t, &cross_kv, &cos, &sin, 2)
        .unwrap();
    assert_eq!(preds.len(), 2, "B=2 forward yields two outputs");

    assert_eq!(preds[0].shape(), cond_b1.shape());
    assert_eq!(preds[1].shape(), uncond_b1.shape());

    let cond_d = max_abs(preds[0].as_slice::<f32>(), cond_b1.as_slice::<f32>());
    let uncond_d = max_abs(preds[1].as_slice::<f32>(), uncond_b1.as_slice::<f32>());
    println!("[batched B=2 vs 2×B=1] cond max|Δ|={cond_d:.3e} uncond max|Δ|={uncond_d:.3e}");

    // Bit-exact: stacking the branches on the batch axis must not perturb either output.
    assert_eq!(cond_d, 0.0, "cond branch diverged under batching");
    assert_eq!(uncond_d, 0.0, "uncond branch diverged under batching");
}
