//! sc-3057: SDXL masked-inpaint correctness on real weights.
//!
//! `#[ignore]`d — needs the real SDXL snapshot. Run:
//!   cargo test -p mlx-gen-sdxl --release --test inpaint_real_weights -- --ignored --nocapture
//!
//! The inpaint blend rides the already-validated img2img path (sc-2638, pixel-parity to the vendored
//! reference), so correctness is proven by two end-to-end invariants of the blend itself — no new
//! cross-impl golden needed (diffusers↔mlx_sd sampler parity is a pre-existing base condition):
//!   1. **mask = all-white ⇒ inpaint ≡ img2img, byte-for-byte.** The blend is identity where mask=1
//!      and draws no RNG, so it must reproduce plain img2img exactly (proves wiring + RNG-neutrality).
//!   2. **mask = all-black ⇒ output = VAE round-trip of the init.** Every step pins the latent to the
//!      init (final step to the clean `x₀`), so decoding equals `decode(encode(init))`.

use std::path::PathBuf;

use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_sdxl as _;
use mlx_gen_sdxl::{decode_image, encode_init_latents, load_vae}; // force-link the provider so `inventory` registers "sdxl"

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

/// Deterministic init image (a diagonal RGB gradient), `w`×`h` RGB8.
fn init_image(w: u32, h: u32) -> Image {
    let mut pixels = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            pixels.push((x % 256) as u8);
            pixels.push((y % 256) as u8);
            pixels.push(((x + y) % 256) as u8);
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

fn solid(w: u32, h: u32, v: u8) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![v; (w * h * 3) as usize],
    }
}

fn req(prompt: &str, init: &Image, mask: Option<&Image>, strength: f32) -> GenerationRequest {
    let mut conditioning = vec![Conditioning::Reference {
        image: init.clone(),
        strength: Some(strength),
    }];
    if let Some(m) = mask {
        conditioning.push(Conditioning::Mask { image: m.clone() });
    }
    GenerationRequest {
        prompt: prompt.to_string(),
        width: 512,
        height: 512,
        seed: Some(7),
        steps: Some(6),
        conditioning,
        ..Default::default()
    }
}

fn run(model: &dyn mlx_gen::Generator, req: &GenerationRequest) -> Image {
    match model.generate(req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => unreachable!("sdxl returns images"),
    }
}

#[test]
#[ignore = "needs the real SDXL snapshot"]
fn inpaint_blend_invariants() {
    let snap = snapshot();
    let model = mlx_gen_sdxl::load(&LoadSpec::new(WeightsSource::Dir(snap.clone()))).unwrap();
    let init = init_image(512, 512);
    let white = solid(512, 512, 255);
    let black = solid(512, 512, 0);

    // Invariant 1: all-white mask ⇒ inpaint ≡ img2img, byte-for-byte.
    let img2img = run(model.as_ref(), &req("a fox in a field", &init, None, 0.85));
    let inpaint_white = run(
        model.as_ref(),
        &req("a fox in a field", &init, Some(&white), 0.85),
    );
    let diff1 = inpaint_white
        .pixels
        .iter()
        .zip(&img2img.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "[inpaint] white-mask vs img2img: {diff1} / {} px bytes differ",
        img2img.pixels.len()
    );
    assert_eq!(diff1, 0, "all-white inpaint must equal plain img2img");

    // Invariant 2: all-black mask ⇒ output = VAE round-trip of the init.
    let inpaint_black = run(
        model.as_ref(),
        &req("a fox in a field", &init, Some(&black), 0.85),
    );
    let vae = load_vae(&snap).unwrap();
    let roundtrip =
        decode_image(&vae, &encode_init_latents(&vae, &init, 512, 512).unwrap()).unwrap();
    let diff2 = inpaint_black
        .pixels
        .iter()
        .zip(&roundtrip.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!(
        "[inpaint] black-mask vs VAE round-trip: {diff2} / {} px bytes differ",
        roundtrip.pixels.len()
    );
    assert_eq!(
        diff2, 0,
        "all-black inpaint must equal the init VAE round-trip"
    );
}
