//! sc-2344: full VAE decoder-assembly parity vs the fork, plus the `Vae::decode` scale/shift
//! wrapper. Fixture `tests/fixtures/vae_decoder.safetensors` ← `tools/dump_vae_decoder.py`
//! (small decoder mirroring `Decoder.__call__`: conv_in → mid → 2 up-blocks → norm-out →
//! SiLU → conv_out). Tol 1e-2 (Metal fp32 convs).

use mlx_gen::weights::Weights;
use mlx_gen_z_image::vae::{Decoder, Vae, VaeDecoderConfig};
use mlx_rs::ops::{add, all_close, multiply};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vae_decoder.safetensors"
);

fn small_cfg() -> VaeDecoderConfig {
    VaeDecoderConfig {
        up_blocks: vec![(3, true), (3, false)],
    }
}

#[test]
fn decoder_assembly_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let decoder = Decoder::from_weights(&w, "", &small_cfg()).unwrap();

    let latent = w.require("in.latent").unwrap();
    let image = decoder.forward(latent).unwrap();
    let want = w.require("out.image").unwrap();

    assert_eq!(image.shape(), want.shape(), "decoder output shape");
    assert!(
        all_close(&image, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "VAE decoder assembly diverged from the fork"
    );
}

#[test]
fn vae_decode_applies_scale_shift_and_frame_axis() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let vae = Vae::from_weights(&w, "", &small_cfg()).unwrap();

    let latent = w.require("in.latent").unwrap(); // (1,16,8,8)
    let sh = latent.shape();
    let latent5 = latent.reshape(&[sh[0], sh[1], 1, sh[2], sh[3]]).unwrap(); // (1,16,1,8,8)

    let decoded = vae.decode(&latent5).unwrap();
    assert_eq!(
        decoded.shape(),
        &[1, 3, 1, 16, 16],
        "decode restores the frame axis"
    );

    // Reference: decoder.forward((latent / scaling) + shift), with the frame axis added back.
    let scaled = add(
        multiply(
            latent,
            Array::from_slice(&[1.0 / Vae::SCALING_FACTOR], &[1]),
        )
        .unwrap(),
        Array::from_slice(&[Vae::SHIFT_FACTOR], &[1]),
    )
    .unwrap();
    let ref_img = vae.decoder().forward(&scaled).unwrap();
    let d = ref_img.shape();
    let ref5 = ref_img.reshape(&[d[0], d[1], 1, d[2], d[3]]).unwrap();

    assert!(
        all_close(&decoded, &ref5, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "Vae::decode scale/shift/frame-axis wrapper is wrong"
    );
}
