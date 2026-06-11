//! sc-3193: backbone Q4/Q8 quantization plumbing + graceful degradation (weight-free).
//!
//! Reuses the sc-3182 backbone golden (tiny dense dual-path Qwen3). Builds the backbone twice,
//! quantizes one (Q8, then Q4), and compares the understanding-path forward to the dense bf16
//! forward: the quantized output must stay **directionally close** (Q8 tighter than Q4), be finite,
//! and the quantize op must be deterministic. This validates the `AdaptableLinear` quant seam wired
//! through the backbone (attention projections + SwiGLU, both paths); real-weight Q8≈bf16 e2e is the
//! `#[ignore]` `quant_realweight` test.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test quant_smoke -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, Path, Qwen3Backbone};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/backbone_golden.safetensors"
);

fn config_from_meta(w: &Weights) -> NeoChatConfig {
    let m = |k: &str| {
        w.metadata(k)
            .unwrap_or_else(|| panic!("missing metadata {k}"))
    };
    let llm = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": m("hidden_size").parse::<u64>().unwrap(),
        "intermediate_size": m("intermediate_size").parse::<u64>().unwrap(),
        "num_hidden_layers": m("num_hidden_layers").parse::<u64>().unwrap(),
        "num_attention_heads": m("num_attention_heads").parse::<u64>().unwrap(),
        "num_key_value_heads": m("num_key_value_heads").parse::<u64>().unwrap(),
        "head_dim": m("head_dim").parse::<u64>().unwrap(),
        "rms_norm_eps": m("rms_norm_eps").parse::<f64>().unwrap(),
        "rope_theta": m("rope_theta").parse::<f64>().unwrap(),
        "rope_theta_hw": m("rope_theta_hw").parse::<f64>().unwrap(),
        "vocab_size": m("vocab_size").parse::<u64>().unwrap(),
        "attention_bias": false,
    });
    let v = serde_json::json!({ "model_type": "neo_chat", "tie_word_embeddings": false, "llm_config": llm, "vision_config": {} });
    NeoChatConfig::from_config_json(&v).expect("synthetic parity config is valid")
}

fn index_rows(idx: &Array) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let s = idx.shape()[1] as usize;
    let flat = idx.as_slice::<i32>();
    let row = |r: usize| flat[r * s..(r + 1) * s].to_vec();
    (row(0), row(1), row(2))
}

fn flat(a: &Array) -> Vec<f32> {
    let n = a.shape().iter().product::<i32>();
    a.reshape(&[n]).unwrap().as_slice::<f32>().to_vec()
}

fn cosine(a: &Array, b: &Array) -> f64 {
    let (g, w) = (flat(a), flat(b));
    let dot: f64 = g.iter().zip(&w).map(|(&x, &y)| x as f64 * y as f64).sum();
    let na: f64 = g.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = w.iter().map(|&y| (y as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb + 1e-12)
}

fn forward(w: &Weights, cfg: &NeoChatConfig, bits: Option<i32>) -> Array {
    let mut model = Qwen3Backbone::from_weights(w, cfg, "language_model").expect("build");
    if let Some(b) = bits {
        model.quantize(b).expect("quantize");
    }
    let embeds = w.require("input.embeds").unwrap().clone();
    let (t, h, wid) = index_rows(w.require("und.indexes").unwrap());
    model
        .forward_path(&embeds, &t, &h, &wid, Path::Und)
        .unwrap()
}

#[test]
fn quantized_forward_stays_close_to_dense() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);

    let dense = forward(&w, &cfg, None);
    let q8 = forward(&w, &cfg, Some(8));
    let q4 = forward(&w, &cfg, Some(4));

    assert!(
        flat(&q8).iter().all(|v| v.is_finite()),
        "Q8 forward has non-finite values"
    );
    assert!(
        flat(&q4).iter().all(|v| v.is_finite()),
        "Q4 forward has non-finite values"
    );

    let c8 = cosine(&q8, &dense);
    let c4 = cosine(&q4, &dense);
    println!("quant vs dense: Q8 cosine={c8:.5}  Q4 cosine={c4:.5}");
    // Q8 is near-lossless; Q4 is coarser but still directionally faithful. (Tiny random weights +
    // group 64 → looser than a real model, hence conservative bounds.)
    assert!(c8 > 0.99, "Q8 cosine {c8:.5} below 0.99");
    assert!(c4 > 0.9, "Q4 cosine {c4:.5} below 0.9");
    assert!(
        c8 >= c4 - 1e-3,
        "Q8 should be at least as faithful as Q4 (Q8 {c8:.5} < Q4 {c4:.5})"
    );

    // Quantize is deterministic (byte-stable packing) → identical forward across two runs.
    let q8b = forward(&w, &cfg, Some(8));
    assert_eq!(flat(&q8), flat(&q8b), "Q8 quantize is not deterministic");
}
