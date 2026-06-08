//! sc-3624: end-to-end real-weight check for the FLUX.1 XLabs IP-Adapter generate path.
//!
//! `#[ignore]`d — loads a full FLUX.1-schnell snapshot + the XLabs adapter + CLIP-ViT-L tower. Run:
//!
//! ```text
//! MLX_GEN_FLUX_SNAPSHOT=/path/to/FLUX.1-schnell/snapshot \
//!   cargo test -p mlx-gen-flux --release --test ip_generate_real_weights -- --ignored --nocapture
//! ```
//!
//! This is the engine-side structural acceptance for sc-3624 (the torch A/B numeric parity is the
//! separate manual step noted on the story — it needs a torch box):
//!   1. A `Conditioning::Reference { strength: 0 }` reproduces plain txt2img **byte-for-byte** (the
//!      IP branch is zeroed and draws no RNG) — proving the reference path leaves the base render
//!      untouched and the dispatch/encode/inject plumbing is wired with no stray perturbation.
//!   2. `strength > 0` changes the image (the IP branch is actually applied).

use std::path::PathBuf;

use mlx_gen::{Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};

fn flux_schnell_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots");
    std::fs::read_dir(&snaps)
        .expect("FLUX.1-schnell HF snapshot (or set MLX_GEN_FLUX_SNAPSHOT)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
        .expect("a complete FLUX.1-schnell snapshot dir")
}

fn hf_file(repo: &str, file: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(format!(".cache/huggingface/hub/models--{repo}/snapshots"));
    let dir = std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache snapshots dir for {repo}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    dir.join(file)
}

/// Stage the engine's `ip_adapter` dir contract: `ip_adapter.safetensors` + `image_encoder/
/// model.safetensors`, symlinked from the HF caches (the layout SceneWorks stages in sc-3625).
fn staged_ip_dir() -> PathBuf {
    let dir = std::env::temp_dir().join("mlx_gen_flux_ip_e2e");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("image_encoder")).unwrap();
    let ip = std::env::var("FLUX_IP_ADAPTER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| hf_file("XLabs-AI--flux-ip-adapter", "ip_adapter.safetensors"));
    let clip = hf_file("openai--clip-vit-large-patch14", "model.safetensors");
    std::os::unix::fs::symlink(ip, dir.join("ip_adapter.safetensors")).unwrap();
    std::os::unix::fs::symlink(clip, dir.join("image_encoder/model.safetensors")).unwrap();
    dir
}

fn gradient(w: u32, h: u32) -> Image {
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

#[test]
#[ignore = "loads a full FLUX.1-schnell snapshot + XLabs IP-Adapter + CLIP-ViT-L"]
fn flux_ip_scale_zero_equals_txt2img() {
    let model = mlx_gen_flux::load_schnell(
        &LoadSpec::new(WeightsSource::Dir(flux_schnell_snapshot()))
            .with_ip_adapter(WeightsSource::Dir(staged_ip_dir())),
    )
    .unwrap();

    let refimg = gradient(512, 512);
    let req = |strength: Option<f32>| {
        let conditioning = match strength {
            Some(s) => vec![Conditioning::Reference {
                image: refimg.clone(),
                strength: Some(s),
            }],
            None => vec![],
        };
        GenerationRequest {
            prompt: "a portrait of a fox".to_string(),
            width: 512,
            height: 512,
            seed: Some(5),
            steps: Some(4),
            conditioning,
            ..Default::default()
        }
    };
    let run = |r: &GenerationRequest| match model.generate(r, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => unreachable!(),
    };

    let plain = run(&req(None));
    let ip0 = run(&req(Some(0.0)));
    let ip = run(&req(Some(0.7)));

    let diff0 = plain
        .pixels
        .iter()
        .zip(&ip0.pixels)
        .filter(|(a, b)| a != b)
        .count();
    let diffi = plain
        .pixels
        .iter()
        .zip(&ip.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!("[flux-ip] txt2img vs IP(scale=0): {diff0} px bytes differ");
    println!("[flux-ip] txt2img vs IP(scale=0.7): {diffi} px bytes differ");
    assert_eq!(
        diff0, 0,
        "IP scale=0 must equal plain txt2img byte-for-byte"
    );
    assert!(diffi > 0, "IP scale=0.7 must change the image");
}

fn flux_dev_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX_DEV_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots");
    std::fs::read_dir(&snaps)
        .expect("FLUX.1-dev HF snapshot (or set MLX_GEN_FLUX_DEV_SNAPSHOT)")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
        .expect("a complete FLUX.1-dev snapshot dir")
}

/// sc-3624 A/B helper: render MLX FLUX + XLabs IP-Adapter conditioned on a reference PNG, saving the
/// output PNG, so it can be compared to the diffusers `tools/flux_ip_torch_ab.py` render. Env:
/// `FLUX_IP_REF_PNG` (in), `FLUX_IP_OUT_PNG` (out), `FLUX_IP_PROMPT`, `FLUX_IP_MODEL` (dev|schnell),
/// `FLUX_IP_SCALE` (0.7), `FLUX_IP_TRUE_CFG` (dev only), `FLUX_IP_SEED` (2), `FLUX_IP_STEPS` (16).
#[test]
#[ignore = "sc-3624 A/B: MLX flux+IP render on a reference PNG; set FLUX_IP_REF_PNG + FLUX_IP_OUT_PNG"]
fn flux_ip_render_to_png() {
    let ref_png = std::env::var("FLUX_IP_REF_PNG").expect("set FLUX_IP_REF_PNG");
    let out_png = std::env::var("FLUX_IP_OUT_PNG").expect("set FLUX_IP_OUT_PNG");
    let prompt = std::env::var("FLUX_IP_PROMPT")
        .unwrap_or_else(|_| "an oil painting in the bold swirling style of Van Gogh".into());
    let dev = std::env::var("FLUX_IP_MODEL").as_deref() == Ok("dev");
    let scale: f32 = std::env::var("FLUX_IP_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.7);
    let true_cfg: Option<f32> = std::env::var("FLUX_IP_TRUE_CFG")
        .ok()
        .and_then(|s| s.parse().ok());
    let seed: u64 = std::env::var("FLUX_IP_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let steps: u32 = std::env::var("FLUX_IP_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let decoded = image::open(&ref_png)
        .expect("open FLUX_IP_REF_PNG")
        .to_rgb8();
    let reference = Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    };

    let snapshot = if dev {
        flux_dev_snapshot()
    } else {
        flux_schnell_snapshot()
    };
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot))
        .with_ip_adapter(WeightsSource::Dir(staged_ip_dir()));
    let model = if dev {
        mlx_gen_flux::load_dev(&spec)
    } else {
        mlx_gen_flux::load_schnell(&spec)
    }
    .unwrap();

    let negative_prompt = std::env::var("FLUX_IP_NEG").ok().filter(|s| !s.is_empty());
    let req = GenerationRequest {
        prompt: prompt.clone(),
        negative_prompt,
        width: 512,
        height: 512,
        seed: Some(seed),
        steps: Some(steps),
        true_cfg,
        conditioning: vec![Conditioning::Reference {
            image: reference,
            strength: Some(scale),
        }],
        ..Default::default()
    };
    let out = match model.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => unreachable!(),
    };
    image::RgbImage::from_raw(out.width, out.height, out.pixels)
        .expect("rgb buffer")
        .save(&out_png)
        .expect("save FLUX_IP_OUT_PNG");
    println!(
        "[mlx-ip] wrote {out_png} (model={}, scale={scale}, true_cfg={true_cfg:?})",
        if dev { "dev" } else { "schnell" }
    );
}
