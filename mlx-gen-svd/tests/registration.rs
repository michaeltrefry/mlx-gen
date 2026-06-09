//! Registry wiring for `svd_xt` (epic 3040 / sc-3375): the model self-registers into the `mlx-gen`
//! model registry with the right descriptor, advertises image→video via `Reference`-only
//! conditioning, and `load` rejects a single-file source (it needs the multi-component snapshot dir).
//! The full-model load + generate is exercised by the deterministic `pipeline_parity` gate.

use mlx_gen::{
    registry, Conditioning, ConditioningKind, GenerationOutput, GenerationRequest, Image, LoadSpec,
    Modality, WeightsSource,
};
use mlx_gen_svd::MODEL_ID;

#[test]
fn svd_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("svd_xt not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "svd_xt");
    assert_eq!(d.family, "svd");
    assert_eq!(d.modality, Modality::Video);
    // image→video is Reference-only.
    assert!(d.capabilities.accepts(ConditioningKind::Reference));
    assert!(!d.capabilities.accepts(ConditioningKind::Keyframe));
    assert!(!d.capabilities.accepts(ConditioningKind::Control));
    // SVD uses a frame-wise guidance ramp; the ceiling is request-overridable.
    assert!(d.capabilities.supports_guidance);
}

#[test]
fn load_rejects_single_file() {
    let dir = std::env::temp_dir().join(format!("svd_reg_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join("model.safetensors");
    std::fs::write(&f, b"not a real checkpoint").unwrap();
    assert!(
        registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::File(f))).is_err(),
        "svd_xt must require a checkpoint directory, not a single file"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// End-to-end provider smoke (real weights): load via the registry, generate a tiny clip from a
/// synthetic reference image, and assert the output shape. Proves the full provider path (load →
/// CLIP/VAE preprocess → seeded init noise → denoise → chunked decode → `Image` frames) runs; the
/// numeric correctness of the deterministic core is gated separately by `pipeline_parity`.
#[test]
#[ignore = "needs the SVD checkpoint in the HF cache (loads the full f32 model)"]
fn svd_provider_generates_video() {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    let snap = std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path();

    let gen = registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(snap))).expect("load svd");

    // A 48×48 RGB gradient reference image.
    let (iw, ih) = (48u32, 48u32);
    let mut pixels = vec![0u8; (iw * ih * 3) as usize];
    for y in 0..ih {
        for x in 0..iw {
            let i = ((y * iw + x) * 3) as usize;
            pixels[i] = (x * 255 / iw) as u8;
            pixels[i + 1] = (y * 255 / ih) as u8;
            pixels[i + 2] = 128;
        }
    }
    let image = Image {
        width: iw,
        height: ih,
        pixels,
    };

    // Smallest size the descriptor advertises (`min_size`); `validate` now enforces the 256..=1024
    // range, so a sub-256 smoke size would be (correctly) rejected.
    let req = GenerationRequest {
        width: 256,
        height: 256,
        frames: Some(3),
        steps: Some(2),
        fps: Some(7),
        seed: Some(7),
        conditioning: vec![Conditioning::Reference {
            image,
            strength: None,
        }],
        ..Default::default()
    };

    let out = gen.generate(&req, &mut |_| {}).expect("generate");
    match out {
        GenerationOutput::Video { frames, fps, audio } => {
            assert_eq!(frames.len(), 3, "expected 3 frames");
            assert_eq!((frames[0].width, frames[0].height), (256, 256));
            assert_eq!(fps, 7);
            assert!(audio.is_none(), "svd_xt produces no audio");
        }
        other => panic!("expected Video, got {other:?}"),
    }
}

/// Locate the cached SVD checkpoint snapshot dir (shared by the real-weight tests).
#[cfg(test)]
fn svd_snapshot_dir() -> std::path::PathBuf {
    let cache = std::env::var("HF_HUB_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/huggingface/hub")
        });
    let snaps = cache
        .join("models--stabilityai--stable-video-diffusion-img2vid-xt")
        .join("snapshots");
    std::fs::read_dir(&snaps)
        .expect("svd snapshot dir")
        .next()
        .unwrap()
        .unwrap()
        .path()
}

/// A small RGB gradient reference image for the real-weight smokes.
#[cfg(test)]
fn gradient_image(w: u32, h: u32) -> Image {
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            pixels[i] = (x * 255 / w) as u8;
            pixels[i + 1] = (y * 255 / h) as u8;
            pixels[i + 2] = 128;
        }
    }
    Image {
        width: w,
        height: h,
        pixels,
    }
}

/// The sc-3523 request knobs (`motion_bucket_id` / `noise_aug_strength` / `decode_chunk_size`) are
/// actually READ by the provider: a different `motion_bucket_id` changes the generated frames (it
/// feeds the `added_time_ids` motion conditioning), and a `decode_chunk_size` smaller than the clip
/// still decodes every frame (chunked temporal VAE). Without the plumbing both runs would be
/// byte-identical / the chunk override a no-op, so this fails closed if a future edit drops it.
#[test]
#[ignore = "needs the SVD checkpoint in the HF cache (loads the full f32 model)"]
fn svd_request_knobs_drive_generation() {
    let snap = svd_snapshot_dir();
    let gen = registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(snap))).expect("load svd");

    let base = |motion: f32, chunk: Option<u32>| GenerationRequest {
        width: 256,
        height: 256,
        frames: Some(4),
        steps: Some(2),
        fps: Some(7),
        seed: Some(7),
        motion_bucket_id: Some(motion),
        decode_chunk_size: chunk,
        conditioning: vec![Conditioning::Reference {
            image: gradient_image(48, 48),
            strength: None,
        }],
        ..Default::default()
    };

    let frames_of =
        |req: &GenerationRequest| match gen.generate(req, &mut |_| {}).expect("generate") {
            GenerationOutput::Video { frames, .. } => frames,
            other => panic!("expected Video, got {other:?}"),
        };

    // Low vs high motion bucket → the conditioning differs → the pixels differ.
    let low = frames_of(&base(20.0, None));
    let high = frames_of(&base(220.0, None));
    assert_eq!(low.len(), 4);
    assert_eq!(high.len(), 4);
    assert!(
        low[3].pixels != high[3].pixels,
        "motion_bucket_id must change the generated frames"
    );

    // A decode chunk smaller than the clip still yields every frame (chunked temporal decode).
    let chunked = frames_of(&base(127.0, Some(2)));
    assert_eq!(
        chunked.len(),
        4,
        "decode_chunk_size override must keep all frames"
    );
}
