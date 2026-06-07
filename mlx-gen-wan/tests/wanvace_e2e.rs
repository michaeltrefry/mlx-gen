//! Wan-VACE end-to-end smoke (epic 3040 / sc-3388, S3 / sc-3436) — **checkpoint-gated** (`#[ignore]`).
//!
//! No Wan-VACE checkpoint is in the local HF cache yet (the cache holds the Wan T2V/I2V/TI2V
//! checkpoints, not VACE), so the real-weight e2e is a **provisioning dependency** — exactly like the
//! LTX IC-LoRA weights were for sc-3052. The engine pieces are validated component-wise without it:
//! the VACE transformer structurally (`wanvace_transformer_parity.rs`, S1) and the conditioning host
//! ops byte-exact (`wanvace_cond_parity.rs`, S2).
//!
//! To run once a converted `wan_vace` snapshot exists (the cutover, sc-3055, produces it — the
//! diffusers VACE transformer + the shared native UMT5 + z16 VAE + tokenizer):
//!   `WANVACE_DIR=/path/to/wan_vace cargo test -p mlx-gen-wan --test wanvace_e2e -- --ignored`
//! This is a **smoke** check (runs `generate` on a synthetic control clip → asserts a coherent video
//! of the right frame count); a full bit-parity gate vs diffusers `WanVACEPipeline` additionally
//! needs a committed golden dumped on that checkpoint (follow-on, once the weights land).

use std::path::PathBuf;

use mlx_gen::{
    registry, Conditioning, GenerationOutput, GenerationRequest, Image, LoadSpec, Progress,
    ReplacementMode, WeightsSource,
};
use mlx_gen_wan::MODEL_ID_VACE;

fn snapshot_dir() -> Option<PathBuf> {
    std::env::var("WANVACE_DIR").ok().map(PathBuf::from)
}

/// A solid mid-gray RGB frame.
fn frame(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![128u8; (w * h * 3) as usize],
    }
}

/// A white (fully-active) mask frame — VACE regenerates the whole frame (pose/depth-control style).
fn mask_frame(w: u32, h: u32) -> Image {
    Image {
        width: w,
        height: h,
        pixels: vec![255u8; (w * h * 3) as usize],
    }
}

#[test]
#[ignore = "needs a converted wan_vace snapshot — set WANVACE_DIR (provisioning dependency)"]
fn wan_vace_generate_smoke() {
    let dir = snapshot_dir().expect("set WANVACE_DIR to the converted wan_vace snapshot");
    let g = registry::load(MODEL_ID_VACE, &LoadSpec::new(WeightsSource::Dir(dir)))
        .expect("load wan_vace");

    let (w, h, n) = (256u32, 256u32, 13usize); // 13 = 1 + 4·3 → 4 latent frames
    let req = GenerationRequest {
        prompt: "a person walking".into(),
        width: w,
        height: h,
        frames: Some(n as u32),
        steps: Some(8),
        conditioning: vec![Conditioning::ControlClip {
            frames: (0..n).map(|_| frame(w, h)).collect(),
            mask: (0..n).map(|_| mask_frame(w, h)).collect(),
            masking_strength: 1.0,
            start_frame: 0,
            mode: ReplacementMode::FaceOnly,
        }],
        ..Default::default()
    };

    let mut on_progress = |_p: Progress| {};
    let out = g.generate(&req, &mut on_progress).expect("vace generate");
    match out {
        GenerationOutput::Video { frames, .. } => {
            assert_eq!(
                frames.len(),
                n,
                "expected {n} output frames, got {}",
                frames.len()
            );
            for (i, f) in frames.iter().enumerate() {
                assert_eq!((f.width, f.height), (w, h), "frame {i} size");
                assert_eq!(
                    f.pixels.len(),
                    (w * h * 3) as usize,
                    "frame {i} pixel buffer"
                );
            }
        }
        other => panic!("expected Video output, got {other:?}"),
    }
}
