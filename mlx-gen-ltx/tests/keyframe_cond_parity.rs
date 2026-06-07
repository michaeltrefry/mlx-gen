//! Keyframe-append (IC-LoRA in-context) conditioning parity vs the torch reference `ltx_core`
//! (epic 3040 / sc-3052). Gates `append_keyframe_clip` + `keyframe_append_positions` (the extend_clip
//! / video_bridge / replace_person conditioning mechanism) against a committed golden dumped from the
//! *actual* `VideoConditionByKeyframeIndex.apply_to` (`tools/dump_ltx_keyframe_cond_golden.py`).
//!
//! The golden uses a trivial 1-token base state, so the appended slice fully isolates the new math:
//! the patchified keyframe tokens, the per-token denoise mask (`1 − strength`), and — the novel bit —
//! the frame-offset RoPE positions (causal-fixed only at `frame_idx == 0`, `+= frame_idx`, `÷ fps`).
//! No model weights / IC-LoRA needed: it's a pure tensor op, so this runs without `--ignored`.

use mlx_rs::ops::{abs, max as max_op, subtract};
use mlx_rs::Array;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::{append_keyframe_clip, VideoTokenState};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_keyframe_cond_golden.safetensors"
);

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = abs(subtract(a, b).unwrap()).unwrap();
    max_op(&d, None).unwrap().item::<f32>()
}

/// A trivial 1-token zero base state (matching the dump's base) with `C` channels.
fn base_state(c: i32) -> VideoTokenState {
    VideoTokenState {
        latent: Array::zeros::<f32>(&[1, 1, c]).unwrap(),
        clean_latent: Array::zeros::<f32>(&[1, 1, c]).unwrap(),
        denoise_mask: Array::ones::<f32>(&[1, 1, 1]).unwrap(),
        positions: Array::zeros::<f32>(&[1, 3, 1, 2]).unwrap(),
        target_tokens: 1,
    }
}

fn check_case(g: &Weights, keyframe: &Array, tag: &str, frame_idx: i32, strength: f32) {
    let st = base_state(keyframe.shape()[1]);
    let out = append_keyframe_clip(&st, keyframe, frame_idx, strength, 8, 32, 24.0).unwrap();
    // Drop the 1-token base (the appended slice is what the golden holds).
    let take = |x: &Array, axis: i32| {
        let n = x.shape()[axis as usize];
        let idx: Vec<i32> = (1..n).collect();
        x.take_axis(Array::from_slice(&idx, &[n - 1]), axis)
            .unwrap()
    };
    let app_latent = take(&out.latent, 1);
    let app_mask = take(&out.denoise_mask, 1);
    let app_pos = take(&out.positions, 2);

    let d_lat = max_abs_diff(&app_latent, g.require(&format!("{tag}_latent")).unwrap());
    let d_mask = max_abs_diff(&app_mask, g.require(&format!("{tag}_mask")).unwrap());
    let d_pos = max_abs_diff(&app_pos, g.require(&format!("{tag}_positions")).unwrap());
    assert!(d_lat < 1e-5, "{tag} latent max|Δ| {d_lat}");
    assert!(d_mask < 1e-6, "{tag} mask max|Δ| {d_mask}");
    assert!(d_pos < 1e-5, "{tag} positions max|Δ| {d_pos}");
    println!(
        "{tag} (frame_idx={frame_idx}, s={strength}): Δlat {d_lat} Δmask {d_mask} Δpos {d_pos}"
    );
}

#[test]
fn keyframe_append_matches_reference() {
    let g = Weights::from_file(GOLDEN)
        .expect("keyframe-cond golden (run tools/dump_ltx_keyframe_cond_golden.py)");
    let keyframe = g.require("keyframe").unwrap();
    // frame_idx=0 (causal-fixed) and frame_idx=5 (no causal fix), matching the dump.
    check_case(&g, keyframe, "f0", 0, 1.0);
    check_case(&g, keyframe, "f5", 5, 0.8);
}
