//! sc-5136: VAE-input preprocessing matches `VAEVideoTransform` on the exactly-matchable pieces.
//!
//! Golden (`tools/dump_bernini_vae_preprocess_golden.py`): `MaxLongEdgeMinShortEdgeResize` target
//! dims over 7 cases — **bit-exact** (integer, banker's round) — and the `ToTensor` + `Normalize(0.5)`
//! → `[-1,1]` on a fixed uint8 image — **bit-exact** (elementwise). The PIL-bicubic resize
//! interpolation is excluded (the port uses the `image` crate; dims exact, pixels differ slightly).
//!
//! Run: `cargo test -p mlx-gen-bernini --test vae_preprocess_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::vae_preprocess::{normalize_chw, resize_dims};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vae_preprocess_golden.safetensors"
);

#[test]
fn vae_preprocess_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let max_size: i64 = w.metadata("max_size").unwrap().parse().unwrap();
    let min_size: i64 = w.metadata("min_size").unwrap().parse().unwrap();
    let stride: i64 = w.metadata("stride").unwrap().parse().unwrap();

    // --- resize dims: bit-exact ---
    let win = w
        .require("resize.in_wh")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let wout = w
        .require("resize.out_wh")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let n = win.len() / 2;
    for i in 0..n {
        let (wd, h) = (win[i * 2] as i64, win[i * 2 + 1] as i64);
        let got = resize_dims(wd, h, max_size, min_size, stride);
        let want = (wout[i * 2] as i64, wout[i * 2 + 1] as i64);
        assert_eq!(got, want, "resize_dims({wd},{h})");
    }
    println!("resize_dims: {n} cases bit-exact");

    // --- normalize -> [-1,1] ---
    let h: i64 = w.metadata("norm_h").unwrap().parse().unwrap();
    let wd: i64 = w.metadata("norm_w").unwrap().parse().unwrap();
    let img: Vec<u8> = w
        .require("norm.image_hwc_u8")
        .unwrap()
        .as_slice::<i32>()
        .iter()
        .map(|&x| x as u8)
        .collect();
    let got = normalize_chw(&img, h, wd);
    let want = w.require("norm.chw").unwrap();
    assert_eq!(got.shape(), want.shape(), "norm shape");
    let g: Vec<f32> = got.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
    let want_v: Vec<f32> = want.flatten(None, None).unwrap().as_slice::<f32>().to_vec();
    let max_diff = g
        .iter()
        .zip(&want_v)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!("normalize: peak|Δ|={max_diff:.3e}");
    assert!(
        max_diff < 1e-6,
        "normalize peak|Δ| {max_diff:.3e} exceeds 1e-6"
    );
}
