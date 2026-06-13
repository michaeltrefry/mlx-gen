//! sc-5142: the renderer's ViT-conditioned guidance-combine modes match the reference (f32).
//!
//! Synthetic-fixture parity (`tools/dump_bernini_vit_guidance_golden.py`): `apg_delta` (verbatim
//! reference) + the `sample_one_step` combine arms (`vae_txt_vit` / `vae_txt_vit_wapg` /
//! `rv2v_wapg` / `r2v_wapg`) over random `[1, n, C]` target-sliced packed-token predictions. Pure
//! elementwise + a single-scalar projection per delta, so this matches to the f32 floor.
//!
//! Run: `cargo test -p mlx-gen-bernini --test vit_guidance_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_bernini::guidance::apg_delta;
use mlx_gen_bernini::vit_guidance::{rv2v_chain, vae_txt_vit};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vit_guidance_golden.safetensors"
);

fn check(name: &str, got: &Array, want: &Array, tol: f32) {
    assert_eq!(got.shape(), want.shape(), "{name} shape");
    let n = want.shape().iter().product::<i32>();
    let g = got.reshape(&[n]).unwrap();
    let wv = want.reshape(&[n]).unwrap();
    let (g, wv) = (g.as_slice::<f32>(), wv.as_slice::<f32>());
    let peak = wv.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = g
        .iter()
        .zip(wv)
        .fold(0f32, |m, (&a, &b)| m.max((a - b).abs()));
    println!(
        "{name:>12}: peak|Δ|={max_diff:.3e} peak-rel={:.3e}",
        max_diff / peak
    );
    assert!(max_diff / peak < tol, "{name} peak-rel exceeds {tol:.1e}");
}

#[test]
fn vit_guidance_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let g = |k: &str| w.require(k).unwrap().clone();
    let (w_img, w_txt, w_tgt, w_vid) = (4.5, 4.0, 3.0, 1.25);

    // bare apg_delta (projection only): apg_delta(img - base, ref = img, 0.2, 1.0).
    let delta = mlx_rs::ops::subtract(g("io.img"), g("io.base")).unwrap();
    let apg = apg_delta(&delta, &g("io.img"), 0.2, 1.0).expect("apg_delta");
    check("apg_only", &apg, w.require("out.apg_only").unwrap(), 1e-5);

    // vae_txt_vit (plain) + vae_txt_vit_wapg (apg, ref = "to" pred).
    let vtv_plain = vae_txt_vit(
        &g("io.base"),
        &g("io.img"),
        &g("io.txt"),
        &g("io.vit"),
        w_img,
        w_txt,
        w_tgt,
        false,
    )
    .unwrap();
    check(
        "vtv_plain",
        &vtv_plain,
        w.require("out.vtv_plain").unwrap(),
        1e-5,
    );
    let vtv_apg = vae_txt_vit(
        &g("io.base"),
        &g("io.img"),
        &g("io.txt"),
        &g("io.vit"),
        w_img,
        w_txt,
        w_tgt,
        true,
    )
    .unwrap();
    check("vtv_apg", &vtv_apg, w.require("out.vtv_apg").unwrap(), 1e-5);

    // rv2v_wapg (plain) + r2v_wapg (apg, ref = "from" pred).
    let rv2v_plain = rv2v_chain(
        &g("io.base"),
        &g("io.eps_v"),
        &g("io.eps_vi"),
        &g("io.eps_vti"),
        &g("io.eps_vtic"),
        w_vid,
        w_img,
        w_txt,
        w_tgt,
        false,
    )
    .unwrap();
    check(
        "rv2v_plain",
        &rv2v_plain,
        w.require("out.rv2v_plain").unwrap(),
        1e-5,
    );
    let r2v_apg = rv2v_chain(
        &g("io.base"),
        &g("io.eps_v"),
        &g("io.eps_vi"),
        &g("io.eps_vti"),
        &g("io.eps_vtic"),
        w_vid,
        w_img,
        w_txt,
        w_tgt,
        true,
    )
    .unwrap();
    check("r2v_apg", &r2v_apg, w.require("out.r2v_apg").unwrap(), 1e-5);
}
