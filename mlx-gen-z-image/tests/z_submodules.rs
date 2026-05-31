//! sc-2344 (denoiser PR 1): parity for the Z-Image DiT sub-modules vs the fork.
//! Fixture `tests/fixtures/z_submodules.safetensors` ← `tools/dump_z_submodules.py`.
//! Tolerance 1e-2 for matmul-bearing modules (Metal fp32); the weightless RoPE table is
//! pure trig + gather, so it's checked tight at 1e-4.

use mlx_gen::weights::Weights;
use mlx_gen_z_image::{FinalLayer, RopeEmbedder, TimestepEmbedder, ZImageContextBlock};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_submodules.safetensors"
);

fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

#[test]
fn timestep_embedder_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let te = TimestepEmbedder::from_weights(&w, "te.w", 256).unwrap();
    let out = te.forward(w.require("te.in_t").unwrap()).unwrap();
    let want = w.require("te.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-2, 1e-2));
}

#[test]
fn rope_embedder_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let rope = RopeEmbedder::new(256.0, &[32, 48, 48], &[1024, 512, 512]);
    let out = rope.forward(w.require("rope.ids").unwrap()).unwrap();
    let want = w.require("rope.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-4, 1e-4), "RoPE table diverged");
}

#[test]
fn final_layer_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let fl = FinalLayer::from_weights(&w, "fl.w").unwrap();
    let out = fl
        .forward(w.require("fl.in_x").unwrap(), w.require("fl.in_c").unwrap())
        .unwrap();
    let want = w.require("fl.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-2, 1e-2));
}

#[test]
fn context_block_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let cb = ZImageContextBlock::from_weights(&w, "cb.w", 96, 4, 1e-5).unwrap();
    let out = cb
        .forward(
            w.require("cb.in_x").unwrap(),
            w.require("cb.in_freqs_cis").unwrap(),
        )
        .unwrap();
    let want = w.require("cb.out").unwrap();
    assert_eq!(out.shape(), want.shape());
    assert!(close(&out, want, 1e-2, 1e-2));
}
