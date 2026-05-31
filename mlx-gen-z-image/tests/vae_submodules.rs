//! sc-2344: parity for the Z-Image VAE decoder sub-modules vs the fork — the convolutional
//! op family (Conv2d + pytorch-compatible GroupNorm + nearest upsample + spatial attention).
//! Fixture `tests/fixtures/vae_submodules.safetensors` ← `tools/dump_vae_submodules.py`.
//! Tol 1e-2 (Metal fp32 convs).

use mlx_gen::weights::Weights;
use mlx_gen_z_image::vae::{ResnetBlock2D, UpSampler, VaeAttention};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vae_submodules.safetensors"
);

fn close(a: &Array, b: &Array) -> bool {
    all_close(a, b, 1e-2, 1e-2, false).unwrap().item::<bool>()
}

#[test]
fn resnet_block_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let rb = ResnetBlock2D::from_weights(&w, "rb.w").unwrap();
    let out = rb.forward(w.require("rb.in").unwrap()).unwrap();
    let want = w.require("rb.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(
        close(&out, want),
        "ResnetBlock2D (incl. 1x1 shortcut) diverged"
    );
}

#[test]
fn vae_attention_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let attn = VaeAttention::from_weights(&w, "attn.w").unwrap();
    let out = attn.forward(w.require("attn.in").unwrap()).unwrap();
    let want = w.require("attn.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want), "VAE mid-block attention diverged");
}

#[test]
fn up_sampler_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let up = UpSampler::from_weights(&w, "up.w").unwrap();
    let out = up.forward(w.require("up.in").unwrap()).unwrap();
    let want = w.require("up.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want), "UpSampler (nearest-2x + conv) diverged");
}
