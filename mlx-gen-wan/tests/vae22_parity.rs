//! sc-2680 parity gate: the Wan 2.2 z48 `Wan22Vae` (vae22) must reproduce the `mlx_video`
//! reference's decode (causal `first_chunk`) + chunked encode.
//!
//! Like the S2 z16 gate, the 5B's production VAE weights are heavy, so this runs against a
//! **self-contained committed fixture**: a tiny `dec_dim=8`/`enc_dim=8` instance with the real
//! `z_dim=48`, seeded random weights, + reference decode/encode IO (`tools/dump_vae22_fixtures.py`,
//! ~4.4 MB). The architecture is width-parametric, so this exercises every vae22 path (channels-last
//! causal 3-D conv, channel-L2 `RMS_norm` eps 1e-24, per-frame attention, `DupUp3D`/`AvgDown3D`,
//! up/down `Resample` `time_conv` incl. the `first_chunk` interleave + chunk-cache, spatial 2×2
//! patchify, the chunked-encode `feat_cache`, mean/std denorm). It runs on Metal in CI — no `#[ignore]`.
//!
//! Honors "divergence is not rounding": the reference runs the VAE in f32; this port does too. The
//! only expected gap is the float-summation order between mlx `conv3d` and the reference's
//! conv2d-per-temporal-slice decomposition of the same convolution (bounded, like the 2.1 gate).

use mlx_gen::weights::Weights;
use mlx_gen_wan::Wan22Vae;

fn fixture() -> Weights {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/vae22.safetensors"
    );
    Weights::from_file(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run tools/dump_vae22_fixtures.py)"))
}

/// `(max|Δ|, Σ|Δ| / Σ|ref|)` over two equal-length f32 slices.
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

/// The tiny fixture uses `dec_dim = enc_dim = 8` with the real `z_dim = 48`.
fn vae(w: &Weights) -> Wan22Vae {
    Wan22Vae::from_weights_dims(w, 8, 8, 48).expect("build Wan22Vae")
}

#[test]
fn vae22_decode_matches_reference() {
    let w = fixture();
    let vae = vae(&w);

    let dec_in = w.require("dec_in").expect("dec_in"); // [48, T, H, W] (channels-first, normalized)
    let exp = w.require("dec_out").expect("dec_out"); // [1, T', 16H, 16W, 3]
    let got = vae.decode(dec_in).expect("decode");
    assert_eq!(got.shape(), exp.shape(), "decode output shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[vae22 decode] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "decode diverged: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

#[test]
fn vae22_encode_single_frame_matches_reference() {
    // T=1 single-image encode (the TI2V conditioning path) — distinct chunking from the T=5 case.
    let w = fixture();
    let vae = vae(&w);
    let enc_in = w.require("enc_in1").expect("enc_in1");
    let exp = w.require("enc_out1").expect("enc_out1");
    let got = vae.encode(enc_in).expect("encode T=1");
    assert_eq!(got.shape(), exp.shape(), "T=1 encode output shape");
    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[vae22 encode T=1] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "T=1 encode diverged: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}

#[test]
fn vae22_encode_matches_reference() {
    let w = fixture();
    let vae = vae(&w);

    let enc_in = w.require("enc_in").expect("enc_in"); // [1, T, H, W, 3] (channels-last, [-1,1])
    let exp = w.require("enc_out").expect("enc_out"); // [1, T_lat, H_lat, W_lat, 48]
    let got = vae.encode(enc_in).expect("encode");
    assert_eq!(got.shape(), exp.shape(), "encode output shape");

    let (max_abs, mean_rel) = diff(got.as_slice::<f32>(), exp.as_slice::<f32>());
    println!(
        "[vae22 encode] shape={:?} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}",
        got.shape()
    );
    assert!(
        mean_rel < 1e-3,
        "encode diverged: mean_rel={mean_rel:.3e} max|Δ|={max_abs:.3e}"
    );
}
