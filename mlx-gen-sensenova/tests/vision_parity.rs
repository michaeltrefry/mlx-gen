//! sc-3183: the NEO vision embedder matches the reference forward (near-bit, f32).
//!
//! Synthetic-fixture parity (`tools/dump_sensenova_vision_golden.py`): a tiny `NEOVisionConfig` run
//! on a 4×4 patch grid. Exercises the full-kernel `patch_embedding` + GELU, the interleaved 2D
//! RoPE, and the 2×2-strided `dense_embedding` patch-merge.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test vision_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, NeoVisionEmbedder};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vision_golden.safetensors"
);

fn config_from_meta(w: &Weights) -> NeoChatConfig {
    let m = |k: &str| {
        w.metadata(k)
            .unwrap_or_else(|| panic!("missing metadata {k}"))
    };
    let vision = serde_json::json!({
        "hidden_size": m("hidden_size").parse::<u64>().unwrap(),
        "llm_hidden_size": m("llm_hidden_size").parse::<u64>().unwrap(),
        "num_channels": m("num_channels").parse::<u64>().unwrap(),
        "patch_size": m("patch_size").parse::<u64>().unwrap(),
        "downsample_ratio": m("downsample_ratio").parse::<f64>().unwrap(),
        "rope_theta_vision": m("rope_theta_vision").parse::<f64>().unwrap(),
    });
    NeoChatConfig::from_config_json(
        &serde_json::json!({ "llm_config": {}, "vision_config": vision }),
    )
    .expect("synthetic vision parity config is valid")
}

fn errors(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    (max_diff, max_diff / peak)
}

#[test]
fn vision_embedder_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let emb = NeoVisionEmbedder::from_weights(&w, &cfg, "vision_model.embeddings")
        .expect("build embedder");

    let pixel_values = w.require("input.pixel_values").unwrap().clone();
    // The fixture is a single 4×4 patch grid.
    let grid_raw = w
        .require("input.grid_hw")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let grid = vec![(grid_raw[0] as usize, grid_raw[1] as usize)];

    let got = emb.forward(&pixel_values, &grid).unwrap();
    let want = w.require("vis.embeds").unwrap();
    let (abs, rel) = errors(&got, want);
    println!(
        "vis.embeds: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}  shape={:?}",
        got.shape()
    );
    assert_eq!(got.shape(), want.shape());
    assert!(rel < 5e-3, "vision peak-rel {rel:.3e} exceeds 5e-3");
}
