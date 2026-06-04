//! sc-2957 invariant: the **compiled elementwise glue** ([`set_compile_glue(true)`]) produces a
//! forward that is **bit-identical** to the eager forward. `mx.compile` fuses the adaLN affine, gated
//! residual, gated-GELU FFN activation, and RoPE rotation into single kernels; the fusion must not
//! perturb the result (it didn't in the standalone microbench — `tests/compile_micro.rs` — `max|Δ|=0`,
//! and this gates the whole-forward composition on the tiny seeded S5 weights, in CI, no real
//! checkpoint).

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::transformer::set_compile_glue;
use mlx_gen_wan::WanTransformer;
use mlx_rs::ops::concatenate_axis;

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
fn compiled_glue_bit_identical_to_eager() {
    let w = load("s5_low.safetensors");
    let cfg = tiny_cfg();
    let dit = WanTransformer::from_weights(&w, &cfg).expect("DiT");

    let latent = w.require("init_noise").unwrap();
    let ctx_cond = dit.embed_text(w.require("ctx_cond").unwrap()).unwrap();
    let ctx_uncond = dit.embed_text(w.require("ctx_uncond").unwrap()).unwrap();
    let t = 833.0f32;

    let context_batch = concatenate_axis(&[&ctx_cond, &ctx_uncond], 0).unwrap();
    let cross_kv = dit.prepare_cross_kv(&context_batch).unwrap();
    let (cos, sin) = dit.prepare_rope(dit.patch_grid(latent)).unwrap();

    set_compile_glue(false);
    let eager = dit
        .forward_cached(latent, t, &cross_kv, &cos, &sin, 2)
        .unwrap();
    set_compile_glue(true);
    let compiled = dit
        .forward_cached(latent, t, &cross_kv, &cos, &sin, 2)
        .unwrap();
    set_compile_glue(false);

    assert_eq!(compiled.len(), eager.len());
    for (i, (c, e)) in compiled.iter().zip(eager.iter()).enumerate() {
        assert_eq!(c.shape(), e.shape());
        let d = max_abs(c.as_slice::<f32>(), e.as_slice::<f32>());
        println!("[compiled vs eager] batch[{i}] max|Δ|={d:.3e}");
        assert_eq!(
            d, 0.0,
            "compiled glue diverged from eager on batch element {i}"
        );
    }
}
