//! Wan-VACE conditioning host-op parity (epic 3040 / sc-3388, S2 / sc-3435).
//!
//! Byte-validates the **pure host** pieces of the VACE control construction against a torch golden
//! (`tools/dump_wanvace_cond_golden.py`, which replicates diffusers `WanVACEPipeline.prepare_masks` +
//! the `prepare_video_latents` masking exactly, no VAE):
//!   - `prepare_masks` — the 8×8 spatial unfold (`view/permute/flatten`) → 64 ch + nearest-exact
//!     temporal resample + reference zero-frame prepend.
//!   - `binarize_mask` + the inactive/reactive split (`where(mask>0.5)`, `video·(1−m)`, `video·m`).
//!
//! These are reshape/gather/elementwise ops → **bit-exact** vs torch (no matmul, so none of the
//! cross-backend f32 floor that bounds the S1 transformer gate). The VAE-encode + `(x−mean)·std`
//! normalization that turns these into the final 32/96-ch control is the already-validated
//! [`WanVae::encode`] (sc-2678), exercised end-to-end (checkpoint-gated) at S3.

use mlx_gen::weights::Weights;
use mlx_gen_wan::vace::{binarize_mask, prepare_masks};
use mlx_rs::ops::{multiply, subtract};
use mlx_rs::Array;

fn golden() -> Weights {
    Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wanvace_cond_golden.safetensors"
    ))
    .expect("run tools/dump_wanvace_cond_golden.py")
}

fn max_abs(got: &[f32], exp: &[f32]) -> f32 {
    assert_eq!(got.len(), exp.len(), "length mismatch");
    got.iter()
        .zip(exp.iter())
        .map(|(g, e)| (g - e).abs())
        .fold(0f32, f32::max)
}

#[test]
fn prepare_masks_matches_diffusers() {
    let g = golden();
    let mask = g.require("in.mask").unwrap(); // [3, 13, 32, 32]
                                              // Golden params: vae_t=4, vae_s=8, patch=2, num_ref=1.
    let out = prepare_masks(mask, 4, 8, 2, 1).expect("prepare_masks");
    let exp = g.require("out.mask_latent").unwrap();
    assert_eq!(out.shape(), exp.shape(), "mask_latent shape");
    let d = max_abs(out.as_slice::<f32>(), exp.as_slice::<f32>());
    println!("[prepare_masks] shape={:?} max|Δ|={d:.3e}", out.shape());
    assert_eq!(d, 0.0, "prepare_masks not bit-exact: {d:.3e}");
}

#[test]
fn masking_matches_diffusers() {
    let g = golden();
    let video = g.require("in.video").unwrap();
    let mask = g.require("in.mask").unwrap();

    let m_bin = binarize_mask(mask).expect("binarize");
    let db = max_abs(
        m_bin.as_slice::<f32>(),
        g.require("out.m_bin").unwrap().as_slice::<f32>(),
    );
    println!("[binarize] max|Δ|={db:.3e}");
    assert_eq!(db, 0.0, "binarize not bit-exact");

    let one = Array::from_slice(&[1.0f32], &[1]);
    let inactive = multiply(video, subtract(&one, &m_bin).unwrap()).unwrap();
    let reactive = multiply(video, &m_bin).unwrap();
    let di = max_abs(
        inactive.as_slice::<f32>(),
        g.require("out.inactive").unwrap().as_slice::<f32>(),
    );
    let dr = max_abs(
        reactive.as_slice::<f32>(),
        g.require("out.reactive").unwrap().as_slice::<f32>(),
    );
    println!("[inactive] max|Δ|={di:.3e}  [reactive] max|Δ|={dr:.3e}");
    assert_eq!(di, 0.0, "inactive not bit-exact");
    assert_eq!(dr, 0.0, "reactive not bit-exact");
}
