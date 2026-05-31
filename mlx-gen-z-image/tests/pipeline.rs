//! sc-2344: Z-Image latent-lifecycle parity vs the Python mflux fork.
//!
//! Fixture `tests/fixtures/z_latents.safetensors` ← `tools/dump_z_latents.py`:
//! - `noise` — seeded `mx.random.normal` (the version-drift RNG-parity gate: mlx-rs 0.25 bundled
//!   MLX vs the fork's 0.31). Same seed must reproduce the same noise.
//! - `decoded` + `image_i32` — a VAE-output tensor and the fork's RGB8 encoding of it.

use mlx_gen::weights::Weights;
use mlx_gen_z_image::pipeline::{create_noise, decoded_to_image, unpack_latents};
use mlx_rs::ops::all_close;
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/z_latents.safetensors"
);

#[test]
fn seeded_noise_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let cfg: Vec<u32> = w
        .metadata("noise_cfg")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let (seed, width, height) = (cfg[0] as u64, cfg[1], cfg[2]);

    let mine = create_noise(seed, width, height).unwrap();
    let golden = w.require("noise").unwrap();
    assert_eq!(mine.shape(), golden.shape(), "noise shape");
    assert!(
        all_close(&mine, golden, 1e-5, 1e-5, false)
            .unwrap()
            .item::<bool>(),
        "seeded noise diverged from the fork — mlx-rs 0.25 RNG differs from mlx 0.31; \
         seeded reproduction would not bit-match the Python pipeline"
    );
}

#[test]
fn decoded_to_image_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let decoded = w.require("decoded").unwrap();

    let img = decoded_to_image(decoded).unwrap();

    let hwc: Vec<u32> = w
        .metadata("image_hwc")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!((img.height, img.width), (hwc[0], hwc[1]), "image dims");

    // Compare RGB8 bytes against the fork's uint8 encoding (stored as int32).
    let golden = w.require("image_i32").unwrap();
    let golden: Vec<u8> = golden.as_slice::<i32>().iter().map(|&v| v as u8).collect();
    assert_eq!(
        img.pixels, golden,
        "RGB8 bytes diverged from the fork's ImageUtil encoding"
    );
}

#[test]
fn unpack_adds_batch_drops_temporal() {
    // [C,1,H,W] -> [1,C,H,W]
    let latents = Array::from_slice(&vec![0.0f32; 16 * 8 * 8], &[16, 1, 8, 8]);
    let out = unpack_latents(&latents).unwrap();
    assert_eq!(out.shape(), &[1, 16, 8, 8]);
}
