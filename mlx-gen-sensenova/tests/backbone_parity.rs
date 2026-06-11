//! sc-3182: the dense dual-path Qwen3 backbone matches the reference forward (near-bit, f32).
//!
//! Synthetic-fixture parity (the repo's weight-free golden pattern): a tiny `NEOLLMConfig` with
//! random weights, dumped from the reference via `tools/dump_sensenova_backbone_golden.py`. This
//! exercises the full forward MATH — the temporal/H/W head split, the dual QK-norms, the three RoPE
//! axes, GQA, the block-causal mask, the residual stack, the dual final norm, and `lm_head` — on
//! both the understanding (`forward_und`) and generation (`forward_gen`) paths, without the 41 GB
//! checkpoint. f32 throughout; the tolerance reflects the MLX-Metal-vs-torch f32 matmul floor.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test backbone_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, Path, Qwen3Backbone};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/backbone_golden.safetensors"
);

/// Rebuild the synthetic `NeoChatConfig` from the fixture's metadata.
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

/// The three position rows of an int32 `[3, S]` index tensor.
fn index_rows(idx: &Array) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let s = idx.shape()[1] as usize;
    let flat = idx.as_slice::<i32>();
    let row = |r: usize| flat[r * s..(r + 1) * s].to_vec();
    (row(0), row(1), row(2))
}

/// (peak abs diff, peak-relative `max|Δ|/max|b|`).
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

fn check(name: &str, got: &Array, want: &Array) {
    let (abs, rel) = errors(got, want);
    println!("{name:>12}: peak|Δ|={abs:.3e}  peak-rel={rel:.3e}");
    // f32 MLX-Metal vs torch matmul floor is ~2.4e-3; allow headroom over the 2-layer stack.
    assert!(rel < 5e-3, "{name} peak-rel {rel:.3e} exceeds 5e-3");
}

/// F-138: a tied-embedding NEO checkpoint has no `lm_head` tensor (config + `expected_keys` already
/// model this). The backbone must reuse `embed_tokens` as the output projection rather than failing
/// at construction. Mirror the fixture with its `lm_head` removed and check both branches.
#[test]
fn tied_embeddings_build_without_lm_head() {
    const LM_HEAD: &str = "language_model.lm_head.weight";
    let full = Weights::from_file(FIXTURE).expect("load fixture");

    // A Weights identical to the fixture but with the `lm_head` tensor removed — a tied checkpoint.
    let keys: Vec<String> = full.keys().map(|k| k.to_string()).collect();
    assert!(
        keys.iter().any(|k| k == LM_HEAD),
        "fixture should ship an untied lm_head"
    );
    let mut tied_w = Weights::empty();
    for k in &keys {
        if k != LM_HEAD {
            tied_w.insert(k.clone(), full.require(k).unwrap().clone());
        }
    }

    // Untied (the fixture default) must still require the now-missing lm_head → construction fails.
    let mut cfg = config_from_meta(&full);
    assert!(!cfg.tie_word_embeddings);
    assert!(
        Qwen3Backbone::from_weights(&tied_w, &cfg, "language_model").is_err(),
        "untied config must still require an lm_head tensor"
    );

    // Tied reuses embed_tokens as the output projection → constructs without an lm_head tensor.
    cfg.tie_word_embeddings = true;
    let model = Qwen3Backbone::from_weights(&tied_w, &cfg, "language_model")
        .expect("tied backbone must build without an lm_head tensor");

    // The tied lm_head is functional and projects to the vocab dimension.
    let vocab = tied_w
        .require("language_model.model.embed_tokens.weight")
        .unwrap()
        .shape()[0];
    let embeds = full.require("input.embeds").expect("embeds").clone();
    let (t, h, wid) = index_rows(full.require("und.indexes").unwrap());
    let hidden = model
        .forward_path(&embeds, &t, &h, &wid, Path::Und)
        .unwrap();
    let logits = model.lm_head(&hidden).unwrap();
    assert_eq!(*logits.shape().last().unwrap(), vocab);
}

#[test]
fn backbone_matches_reference_both_paths() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let model = Qwen3Backbone::from_weights(&w, &cfg, "language_model").expect("build backbone");

    let embeds = w.require("input.embeds").expect("embeds").clone();

    // Understanding path.
    let (t, h, wid) = index_rows(w.require("und.indexes").unwrap());
    let und_hidden = model
        .forward_path(&embeds, &t, &h, &wid, Path::Und)
        .unwrap();
    check("und.hidden", &und_hidden, w.require("und.hidden").unwrap());
    check(
        "und.logits",
        &model.lm_head(&und_hidden).unwrap(),
        w.require("und.logits").unwrap(),
    );

    // Generation path (image-grid positions, bidirectional block).
    let (t, h, wid) = index_rows(w.require("gen.indexes").unwrap());
    let gen_hidden = model
        .forward_path(&embeds, &t, &h, &wid, Path::Gen)
        .unwrap();
    check("gen.hidden", &gen_hidden, w.require("gen.hidden").unwrap());
    check(
        "gen.logits",
        &model.lm_head(&gen_hidden).unwrap(),
        w.require("gen.logits").unwrap(),
    );
}
