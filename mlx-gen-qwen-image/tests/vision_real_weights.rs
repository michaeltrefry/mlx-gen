//! sc-2465 slice 6a: Qwen2.5-VL vision-transformer parity vs the frozen fork.
//!
//! `#[ignore]`d — needs the local golden from `tools/dump_qwen_vision_golden.py` (gitignored).
//! Micro-gated, gates added as the modules land:
//!
//! - **Gate 1 (here)**: the weight-free index/RoPE math — `window_index`, full-attn `cu_seqlens`,
//!   and `rot_pos_emb` — checked byte-exact on grids that exercise window padding, the
//!   exact-multiple edge, and multi-image.
//!
//! Run: `cd ~/repos/mflux && uv run python ~/repos/mlx-gen/tools/dump_qwen_vision_golden.py`, then
//! `cargo test -p mlx-gen-qwen-image --release --test vision_real_weights -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_qwen_image::text_encoder::vision::grid::{
    cu_seqlens, rot_pos_emb, window_index, Grid, VisionGridConfig,
};
use mlx_gen_qwen_image::text_encoder::vision::{VisionConfig, VisionTransformer};
use mlx_gen_qwen_image::vl_tokenizer::build_edit_text;
use mlx_gen_qwen_image::{load_tokenizer, load_vision_encoder, load_vision_language_encoder};
use mlx_rs::Array;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_vision_golden.safetensors"
);
const VL_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_vl_encoder_golden.safetensors"
);
const VL_TOKENIZE_GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_vl_tokenize_golden.safetensors"
);

const GRIDS: [&str; 4] = ["g0", "g1", "g2", "g3"];

fn ints(a: &Array) -> Vec<i32> {
    a.as_slice::<i32>().to_vec()
}

/// `(peak-rel, mean-rel)` vs the golden — mirrors the transformer parity test.
fn rel_errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let sum_abs_b: f64 = b.iter().map(|&v| v.abs() as f64).sum();
    let sum_abs_diff: f64 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs() as f64).sum();
    (max_diff / peak, (sum_abs_diff / sum_abs_b) as f32)
}

fn grids_of(g: &Weights, name: &str) -> Vec<Grid> {
    ints(g.require(&format!("{name}_grid")).unwrap())
        .chunks(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

#[test]
#[ignore = "needs local vision golden"]
fn window_index_and_cu_window_match_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let cfg = VisionGridConfig::default();
    for name in GRIDS {
        let grids = grids_of(&g, name);
        let (wi, cu) = window_index(&grids, &cfg);
        let want_wi = ints(g.require(&format!("{name}_window_index")).unwrap());
        let want_cu = ints(g.require(&format!("{name}_cu_window")).unwrap());
        assert_eq!(wi, want_wi, "{name} window_index");
        assert_eq!(cu, want_cu, "{name} cu_window_seqlens");
        println!(
            "{name}: window_index ({} groups) + cu_window {cu:?} OK",
            wi.len()
        );
    }
}

#[test]
#[ignore = "needs local vision golden"]
fn cu_seqlens_match_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    for name in GRIDS {
        let grids = grids_of(&g, name);
        let got = cu_seqlens(&grids);
        let want = ints(g.require(&format!("{name}_cu_seqlens")).unwrap());
        assert_eq!(got, want, "{name} cu_seqlens");
        println!("{name}: cu_seqlens {got:?} OK");
    }
}

#[test]
#[ignore = "needs local vision golden"]
fn rot_pos_emb_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let cfg = VisionGridConfig::default();
    for name in GRIDS {
        let grids = grids_of(&g, name);
        let got = rot_pos_emb(&grids, &cfg).unwrap();
        let want = g.require(&format!("{name}_rope")).unwrap();
        assert_eq!(got.shape(), want.shape(), "{name} rope shape");
        let (ga, wa) = (got.as_slice::<f32>(), want.as_slice::<f32>());
        let max_diff = ga
            .iter()
            .zip(wa)
            .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
        println!("{name}: rope {:?} max abs diff {max_diff:.3e}", got.shape());
        assert!(max_diff < 1e-5, "{name} rope max abs diff {max_diff:.3e}");
    }
}

/// Gate A: a small synthetic VisionTransformer end-to-end (no snapshot). Exercises patch_embed,
/// full + windowed (block-diagonal) SDPA, the window reorder/reverse, and the merger together, plus
/// the `from_weights` key mapping. Real-weight parity follows in slice 6b.
#[test]
#[ignore = "needs local vision golden"]
fn small_vision_transformer_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let cfg = VisionConfig {
        patch_size: 14,
        temporal_patch_size: 2,
        in_channels: 3,
        embed_dim: 64,
        depth: 4,
        num_heads: 4,
        mlp_hidden: 128, // int(64 * 2.0)
        out_hidden_size: 32,
        spatial_merge_size: 2,
        window_size: 112,
        fullatt_block_indexes: vec![1, 3],
        rope_theta: 10000.0,
    };
    let vt = VisionTransformer::from_weights(&g, "vt", &cfg).unwrap();
    let grids = grids_of(&g, "io");
    let pixel = g.require("io_pixel_values").unwrap();
    let got = vt.forward(pixel, &grids).unwrap();
    let want = g.require("io_out").unwrap();
    assert_eq!(got.shape(), want.shape(), "vt out shape");
    let (peak, mean) = rel_errors(&got, want);
    println!(
        "small VT {:?}: peak-rel {peak:.3e}  mean-rel {mean:.3e}",
        got.shape()
    );
    assert!(mean < 2e-3, "vt mean-rel {mean:.3e}");
    assert!(peak < 1e-2, "vt peak-rel {peak:.3e}");
}

/// Locate the Qwen-Image-Edit-2511 snapshot dir (env override, else the HF cache).
fn edit_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_EDIT_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Gate B: the **real** depth-32 vision transformer loaded from the Edit-2511 snapshot (bf16 weights,
/// f32 activations) vs the fork's f32 output. Validates `load_vision_encoder` (the `visual.*` remap +
/// patch-embed transpose + merger rename) and the full-scale forward.
#[test]
#[ignore = "needs real Qwen-Image-Edit-2511 vision weights + local golden"]
fn vision_transformer_real_weights_matches_fork() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let vt = load_vision_encoder(&edit_snapshot()).unwrap();
    let grids = grids_of(&g, "real");
    let pixel = g.require("real_pixel_values").unwrap();
    let got = vt.forward(pixel, &grids).unwrap();
    let want = g.require("real_out").unwrap();
    assert_eq!(got.shape(), want.shape(), "real vt out shape");
    let (peak, mean) = rel_errors(&got, want);
    println!(
        "real VT {:?}: peak-rel {peak:.3e}  mean-rel {mean:.3e}",
        got.shape()
    );
    assert!(mean < 2e-3, "real vt mean-rel {mean:.3e}");
    assert!(peak < 1e-2, "real vt peak-rel {peak:.3e}");
}

/// Gate C (slice 6b-3): the full **VL conditioning encoder** — vision embeds spliced into the text
/// stream, 28 LM layers, drop-64 — loaded from the Edit-2511 snapshot, vs the fork's f32
/// `prompt_embeds`. Validates `load_vision_language_encoder` + the splice + drop end-to-end.
#[test]
#[ignore = "needs real Qwen-Image-Edit-2511 weights + local VL golden"]
fn vl_encoder_real_weights_matches_fork() {
    let g = Weights::from_file(VL_GOLDEN).unwrap();
    let enc = load_vision_language_encoder(&edit_snapshot()).unwrap();
    let grids = grids_of(&g, "vl");
    let got = enc
        .encode(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
            g.require("pixel_values").unwrap(),
            &grids,
        )
        .unwrap();
    let want = g.require("prompt_embeds").unwrap();
    assert_eq!(got.shape(), want.shape(), "prompt_embeds shape");
    let (peak, mean) = rel_errors(&got, want);
    println!(
        "VL encoder {:?}: peak-rel {peak:.3e}  mean-rel {mean:.3e}",
        got.shape()
    );
    // Looser than the single-component gates: this stacks bf16 weight rounding through the vision
    // tower AND the 28 LM layers (vs f32 fork weights). The observed ~2.4e-3 is uniform *relative*
    // error (peak ≈ mean) — the bf16 mantissa signature, not a logic error.
    assert!(mean < 5e-3, "VL encoder mean-rel {mean:.3e}");
    assert!(peak < 1e-2, "VL encoder peak-rel {peak:.3e}");
}

/// Slice 6b-2: the edit chat template + `<|image_pad|>` expansion + special-token tokenization,
/// byte-exact vs the fork. Image/weight-free — `build_edit_text` is reconstructed for the same fixed
/// (prompt, n_image_tokens) the golden was dumped with, then tokenized via the materialized
/// `tokenizer.json`. Must match `tools/dump_qwen_vl_tokenize_golden.py`.
#[test]
#[ignore = "needs the Edit tokenizer.json + local VL-tokenize golden"]
fn vl_tokenize_matches_fork() {
    let g = Weights::from_file(VL_TOKENIZE_GOLDEN).unwrap();
    let tok = load_tokenizer(&edit_snapshot()).unwrap();
    let text = build_edit_text("make the sky purple at sunset", 36);
    let out = tok.tokenize_preformatted(&text).unwrap();
    let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&out);
    let want = g.require("input_ids").unwrap();
    assert_eq!(input_ids.shape(), want.shape(), "input_ids shape");
    assert_eq!(
        input_ids.as_slice::<i32>(),
        want.as_slice::<i32>(),
        "input_ids must be byte-exact vs the fork"
    );
    println!("VL tokenize: input_ids {:?} byte-exact OK", want.shape());
}
