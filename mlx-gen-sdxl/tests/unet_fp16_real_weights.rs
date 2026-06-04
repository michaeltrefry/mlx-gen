//! sc-2721: SDXL U-Net single-forward parity in **fp16** vs the vendored reference (the production
//! `StableDiffusionXL(float16=True)` dtype). De-risks the fp16 migration: it answers whether the
//! fp16 U-Net forward is byte-identical across builds (mlx-gen runs the NAX MLX 0.31.2 build; the
//! golden is dumped on the mflux pip wheel 0.31.0). A bit-exact result means an e2e fp16 byte-parity
//! target is reachable through the chaos-sensitive ancestral sampler; a 1-ULP-ish gap means it is
//! not (cross-version NAX-vs-wheel residual) and the deliverable is within-tolerance parity.
//!
//! `#[ignore]`d — needs the SDXL snapshot + the golden from `tools/dump_sdxl_unet_golden_fp16.py`.
//! Run with:
//!   cargo test -p mlx-gen-sdxl --release --test unet_fp16_real_weights -- --ignored --nocapture

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_sdxl::load_unet_dtype;
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_unet_golden_fp16.safetensors"
);

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn f32v(a: &Array) -> Vec<f32> {
    let n = a.shape().iter().product::<i32>();
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .reshape(&[n])
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

#[test]
#[ignore = "needs the real SDXL snapshot + fp16 U-Net golden"]
fn unet_single_forward_matches_vendored_fp16() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let timestep: f32 = g.metadata("timestep").unwrap().parse().unwrap();

    // The golden inputs are saved at their native compute dtype (f16 latents/conditioning/pooled,
    // f32 time_ids). Feed them as-is so the U-Net runs fp16 end to end, matching the reference.
    let latents = g
        .require("latents")
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let conditioning = g
        .require("conditioning")
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let pooled = g
        .require("pooled")
        .unwrap()
        .as_dtype(Dtype::Float16)
        .unwrap();
    let time_ids = g
        .require("time_ids")
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();

    let unet = load_unet_dtype(&snapshot(), Dtype::Float16).unwrap();
    let eps = unet
        .forward(&latents, timestep, &conditioning, &pooled, &time_ids)
        .unwrap();

    let golden = g.require("eps").unwrap();
    assert_eq!(eps.shape(), golden.shape(), "eps shape");
    assert_eq!(eps.dtype(), Dtype::Float16, "fp16 forward must stay f16");

    let (a, b) = (f32v(&eps), f32v(golden));
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(&b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mabs = b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32;
    let mean_rel =
        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32 / mabs;
    let exact = a.iter().zip(&b).filter(|(x, y)| x == y).count();
    let total = a.len();
    println!(
        "fp16 unet eps {:?}: peak_rel={:.3e} mean_rel={mean_rel:.3e} byte-exact {exact}/{total} ({:.2}%)",
        eps.shape(),
        max_diff / peak,
        100.0 * exact as f32 / total as f32,
    );
    // Loose correctness bound (the same scale as the f32 gate); the printed byte-exact fraction is
    // the real signal for whether e2e fp16 byte-parity is reachable.
    assert!(
        max_diff / peak < 5e-2,
        "fp16 U-Net forward diverged badly: peak_rel {:.3e}",
        max_diff / peak
    );
}
