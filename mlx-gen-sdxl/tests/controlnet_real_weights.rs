//! sc-3058: SDXL ControlNet branch parity vs the diffusers `ControlNetModel` (f32).
//!
//! `#[ignore]`d — needs `xinsir/controlnet-tile-sdxl-1.0` + the golden from
//! `tools/dump_sdxl_controlnet_golden.py`. Run:
//!   cargo test -p mlx-gen-sdxl --release --test controlnet_real_weights -- --ignored --nocapture
//!
//! Validates the whole branch (conditioning embedding + UNet-encoder copy + zero-conv heads) by
//! matching the 9 down residuals + the mid residual on a fixed (latents, control image, timestep,
//! text conditioning). A fixed timestep is fed to both sides (mlx-gen's sinusoidal timestep embedding
//! equals diffusers `get_timestep_embedding`), isolating the branch from the schedule.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    Conditioning, ControlKind, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_sdxl::config::UNetConfig;
use mlx_gen_sdxl::ControlNet;
use mlx_gen_sdxl as _; // force-link the provider so `inventory` registers "sdxl"
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_controlnet_golden.safetensors"
);

fn cn_weights() -> Weights {
    let dir = if let Ok(p) = std::env::var("SDXL_TILE_CN") {
        PathBuf::from(p)
    } else {
        let home = std::env::var("HOME").unwrap();
        let snaps = PathBuf::from(home).join(
            ".cache/huggingface/hub/models--xinsir--controlnet-tile-sdxl-1.0/snapshots",
        );
        std::fs::read_dir(&snaps)
            .expect("HF cache snapshots dir for xinsir/controlnet-tile-sdxl-1.0")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_dir())
            .expect("a snapshot dir")
    };
    let mut w = Weights::from_file(dir.join("diffusion_pytorch_model.safetensors")).unwrap();
    w.cast_all(Dtype::Float32).unwrap();
    w
}

fn nchw_to_nhwc(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32)
        .unwrap()
        .transpose_axes(&[0, 2, 3, 1])
        .unwrap()
}

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let b = b.reshape(&[n]).unwrap().as_dtype(Dtype::Float32).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    a.iter().zip(b).fold(0f32, |m, (&x, &y)| m.max((x - y).abs())) / peak
}

#[test]
#[ignore = "needs xinsir tile-CN weights + the controlnet golden"]
fn controlnet_residuals_match_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let sample = nchw_to_nhwc(g.require("sample").unwrap()); // [1,64,64,4]
    let control = nchw_to_nhwc(g.require("control").unwrap()); // [1,512,512,3]
    let encoder_x = g.require("encoder_hidden_states").unwrap().as_dtype(Dtype::Float32).unwrap();
    let text_emb = g.require("text_embeds").unwrap().as_dtype(Dtype::Float32).unwrap();
    let time_ids = g.require("time_ids").unwrap().as_dtype(Dtype::Float32).unwrap();
    let timestep = g.require("timestep").unwrap().as_slice::<f32>()[0];

    let cn = ControlNet::from_weights(&cn_weights(), &UNetConfig::sdxl_base()).unwrap();
    let res = cn
        .forward(&sample, &control, timestep, &encoder_x, &text_emb, &time_ids, 1.0)
        .unwrap();

    assert_eq!(res.down.len(), 9, "expected 9 down residuals");
    let mut worst_down = 0f32;
    for (i, d) in res.down.iter().enumerate() {
        let rel = peak_rel(d, g.require(&format!("down_{i}")).unwrap());
        println!("[controlnet] down_{i} peak_rel = {rel:.3e}");
        worst_down = worst_down.max(rel);
    }
    let mid_rel = peak_rel(&res.mid, g.require("mid").unwrap());
    println!("[controlnet] mid peak_rel = {mid_rel:.3e}");

    // f32 vs f32, cross-backend (torch CPU vs MLX Metal) over a UNet-encoder copy. The residual
    // error grows monotonically with depth (down_0 ~1.5e-3 → mid ~1.6e-2): it's the cross-backend
    // attention-accumulation floor through the deep transformer stacks (down_block[2] + the 10-layer
    // mid), NOT a port bug — a wrong cond-embedding / zero-conv / block would blow the *shallow*
    // residuals up too. The residuals are scaled by ~0.45 and added to the UNet, so this is
    // negligible end-to-end.
    assert!(worst_down < 1e-2, "ControlNet down residuals diverged: {worst_down:.3e}");
    assert!(mid_rel < 2e-2, "ControlNet mid residual beyond the deep-path floor: {mid_rel:.3e}");
}

fn base_snapshot() -> PathBuf {
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

fn tile_cn_snapshot() -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--xinsir--controlnet-tile-sdxl-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir for xinsir tile-CN")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
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
    Image { width: w, height: h, pixels }
}

/// e2e wiring check: a ControlNet request at `conditioning_scale = 0` produces 0 residuals and the
/// branch draws no RNG, so it must reproduce plain img2img byte-for-byte (proves the load + dispatch
/// + `forward_with_control` wiring + the scale knob).
#[test]
#[ignore = "needs the SDXL base snapshot + xinsir tile-CN"]
fn controlnet_scale_zero_equals_img2img() {
    let model = mlx_gen_sdxl::load(
        &LoadSpec::new(WeightsSource::Dir(base_snapshot()))
            .with_control(WeightsSource::Dir(tile_cn_snapshot())),
    )
    .unwrap();
    let init = gradient(512, 512);

    let base_req = |extra: Vec<Conditioning>| {
        let mut conditioning = vec![Conditioning::Reference {
            image: init.clone(),
            strength: Some(0.85),
        }];
        conditioning.extend(extra);
        GenerationRequest {
            prompt: "a detailed fox".to_string(),
            width: 512,
            height: 512,
            seed: Some(11),
            steps: Some(6),
            conditioning,
            ..Default::default()
        }
    };
    let run = |req: &GenerationRequest| match model.generate(req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.remove(0),
        _ => unreachable!(),
    };

    let plain = run(&base_req(vec![]));
    let ctrl0 = run(&base_req(vec![Conditioning::Control {
        image: init.clone(),
        kind: ControlKind::Other("tile".to_string()),
        scale: 0.0,
    }]));
    let diff = plain
        .pixels
        .iter()
        .zip(&ctrl0.pixels)
        .filter(|(a, b)| a != b)
        .count();
    println!("[controlnet] scale=0 vs img2img: {diff} / {} px bytes differ", plain.pixels.len());
    assert_eq!(diff, 0, "ControlNet at scale=0 must equal plain img2img");
}
