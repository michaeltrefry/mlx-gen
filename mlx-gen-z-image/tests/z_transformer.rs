//! sc-2344 (denoiser PR 2): full Z-Image DiT forward parity vs the fork.
//! Fixture `tests/fixtures/z_transformer.safetensors` ← `tools/dump_z_transformer.py`
//! (tiny synthetic model: dim=96, 4 heads, 1 refiner + 2 main layers, in_ch=4, patch=2).
//! Tol 1e-2 — Metal fp32 across a 30+-matmul forward.

use mlx_gen::weights::Weights;
use mlx_gen_z_image::{ZImageTransformer, ZImageTransformerConfig};
use mlx_rs::ops::all_close;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_transformer.safetensors"
);

fn small_cfg() -> ZImageTransformerConfig {
    ZImageTransformerConfig {
        patch_size: 2,
        f_patch_size: 1,
        in_channels: 4,
        dim: 96,
        n_layers: 2,
        n_refiner_layers: 1,
        n_heads: 4,
        norm_eps: 1e-5,
        cap_feat_dim: 32,
        rope_theta: 256.0,
        t_scale: 1000.0,
        axes_dims: vec![8, 8, 8],
        axes_lens: vec![64, 64, 64],
        frequency_embedding_size: 256,
    }
}

#[test]
fn full_forward_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let model = ZImageTransformer::from_weights(&w, "w", small_cfg()).unwrap();

    let x = w.require("in.x").unwrap();
    let cap_feats = w.require("in.cap_feats").unwrap();
    let y = model.forward(x, 0.7, cap_feats).unwrap();

    let want = w.require("out.y").unwrap();
    assert_eq!(y.shape(), want.shape(), "output shape");
    assert!(
        all_close(&y, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "full Z-Image denoiser forward diverged from the fork"
    );
}
