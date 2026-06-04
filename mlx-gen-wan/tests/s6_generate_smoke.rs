//! S6 product-path smoke (`#[ignore]` — needs the 54 GB converted A14B checkpoint).
//!
//! Drives the **public** product entry point end-to-end — `mlx_gen::load("wan2_2_t2v_14b", spec)`
//! then `Generator::generate` — to confirm the registry → staged-load → seeded-noise →
//! `denoise_moe` → z16 VAE decode → `Vec<Image>` path runs and yields a *coherent* video (not noise,
//! not a flat frame). This complements `s6_real_parity` (which drives the pipeline directly against
//! a reference golden); here we exercise the seeded-noise path + `GenerationOutput::Video` assembly
//! the parity test bypasses. Frame 0 is written to `$WAN_A14B_OUT/frame00.ppm` (when `WAN_A14B_OUT`
//! is set) so the result can be eyeballed.
//!
//! ```text
//! WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
//! WAN_A14B_OUT=/tmp/wan_a14b_smoke \
//!   cargo test -p mlx-gen-wan --test s6_generate_smoke -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::{registry, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource};
use mlx_gen_wan::MODEL_ID_T2V_14B;

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| {
        let s = s.to_string_lossy();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(format!("{}/{rest}", home.to_string_lossy()));
            }
        }
        PathBuf::from(s.to_string())
    })
}

/// Minimal binary PPM (P6) writer — no image-crate dependency, just to eyeball a frame.
fn write_ppm(path: &std::path::Path, img: &Image) {
    let mut buf = format!("P6\n{} {}\n255\n", img.width, img.height).into_bytes();
    buf.extend_from_slice(&img.pixels);
    std::fs::write(path, buf).expect("write ppm");
}

#[test]
#[ignore = "needs the 54 GB converted Wan2.2-T2V-A14B checkpoint (WAN_A14B_MODEL_DIR)"]
fn wan_a14b_generate_produces_coherent_video() {
    let model_dir = match env_path("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR to the converted A14B model dir");
            return;
        }
    };

    let gen = registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(model_dir)),
    )
    .expect("load wan2_2_t2v_14b");
    assert_eq!(gen.descriptor().id, MODEL_ID_T2V_14B);

    let req = GenerationRequest {
        prompt: "a red fox trotting across a snowy meadow at sunrise, cinematic".into(),
        width: 128,
        height: 128,
        frames: Some(5),
        steps: Some(6),
        seed: Some(42),
        sampler: Some("unipc".into()),
        ..Default::default()
    };
    gen.validate(&req).expect("validate");

    let mut last = 0u32;
    let mut on_progress = |p: mlx_gen::Progress| {
        if let mlx_gen::Progress::Step { current, total } = p {
            assert!(current >= last, "progress went backwards");
            last = current;
            println!("  step {current}/{total}");
        }
    };
    let out = gen.generate(&req, &mut on_progress).expect("generate");

    let (frames, fps) = match out {
        GenerationOutput::Video { frames, fps, audio } => {
            assert!(audio.is_none(), "Wan T2V has no audio track");
            (frames, fps)
        }
        other => panic!("expected Video, got {other:?}"),
    };
    assert_eq!(fps, 16, "default Wan2.2 sample_fps");
    assert!(!frames.is_empty(), "no frames produced");
    println!(
        "produced {} frames @ {}x{}, fps={fps}",
        frames.len(),
        frames[0].width,
        frames[0].height
    );

    // Coherence checks: every frame is the right size, pixels span a real range (not flat / not
    // saturated noise), and the temporal sequence isn't a single repeated frame.
    let mut prev: Option<&Image> = None;
    let mut any_temporal_change = false;
    for (i, img) in frames.iter().enumerate() {
        assert_eq!(img.width, 128);
        assert_eq!(img.height, 128);
        assert_eq!(img.pixels.len(), (img.width * img.height * 3) as usize);
        let min = *img.pixels.iter().min().unwrap();
        let max = *img.pixels.iter().max().unwrap();
        let mean: f64 = img.pixels.iter().map(|&b| b as f64).sum::<f64>() / img.pixels.len() as f64;
        println!("  frame {i}: min={min} max={max} mean={mean:.1}");
        assert!(
            max > min,
            "frame {i} is flat (min==max) — VAE decode produced a constant image"
        );
        assert!(
            (16.0..240.0).contains(&mean),
            "frame {i} mean {mean:.1} is implausible (all-black/all-white → decode bug)"
        );
        if let Some(p) = prev {
            if p.pixels != img.pixels {
                any_temporal_change = true;
            }
        }
        prev = Some(img);
    }
    if frames.len() > 1 {
        assert!(
            any_temporal_change,
            "all frames identical — temporal decode is degenerate"
        );
    }

    if let Some(out_dir) = env_path("WAN_A14B_OUT") {
        std::fs::create_dir_all(&out_dir).ok();
        for (i, img) in frames.iter().enumerate() {
            write_ppm(&out_dir.join(format!("frame{i:02}.ppm")), img);
        }
        println!("wrote {} PPM frames to {}", frames.len(), out_dir.display());
    }
}
