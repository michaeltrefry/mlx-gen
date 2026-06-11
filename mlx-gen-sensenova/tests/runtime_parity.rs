//! sc-3187: the AR text-generation runtime (KV cache + greedy decode) matches the reference.
//!
//! Synthetic-fixture parity (the repo's weight-free golden pattern, shared tiny config with
//! `backbone_parity`). The fixture (`tools/dump_sensenova_runtime_golden.py`) prefills a text prefix
//! into the reference's HF cache and greedy-decodes N tokens via the exact `_generate_think`
//! single-token mechanics. This test replays the same through [`Qwen3Backbone::forward_cached`] /
//! [`Qwen3Backbone::generate`] and asserts:
//!   * prefill logits match the dense `forward_path` (cache-equivalence, no reference needed),
//!   * prefill logits match the reference,
//!   * each incremental-decode step's logits match the reference (peak-rel within the f32 floor),
//!   * the greedy token stream is bit-identical to the reference's.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test runtime_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, Path, Qwen3Backbone, Sampler, ThinkRollout};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/runtime_golden.safetensors"
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

/// peak-relative `max|Δ|/max|b|` between two same-shaped arrays.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as i32
}

#[test]
fn ar_runtime_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let model = Qwen3Backbone::from_weights(&w, &cfg, "language_model").expect("build backbone");

    let prefix_ids = w.require("prefix.input_ids").unwrap().clone();
    let (pt, ph, pw) = index_rows(w.require("prefix.indexes").unwrap());
    let prefix_len = pt.len() as i32;
    let want_prefix_logits = w.require("prefix.logits").unwrap();
    let want_tokens: Vec<i32> = w
        .require("decode.tokens")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let want_decode_logits = w.require("decode.logits").unwrap();
    let vocab = want_decode_logits.shape()[1];

    // ---- Prefill (cached, empty cache) and cache-equivalence vs the dense forward_path. ----
    let embeds = model.embed(&prefix_ids).unwrap();
    let mut cache = model.new_cache();
    let prefill_hidden = model
        .forward_cached(&embeds, &pt, &ph, &pw, Path::Und, &mut cache, true)
        .unwrap();
    let prefill_logits = model.lm_head(&prefill_hidden).unwrap();

    let dense_hidden = model
        .forward_path(&embeds, &pt, &ph, &pw, Path::Und)
        .unwrap();
    let dense_logits = model.lm_head(&dense_hidden).unwrap();
    let equiv = peak_rel(&prefill_logits, &dense_logits);
    println!("cache-equivalence (prefill vs forward_path): peak-rel={equiv:.3e}");
    assert!(
        equiv < 1e-5,
        "cached prefill must equal forward_path (same backend), got {equiv:.3e}"
    );

    let pr = peak_rel(&prefill_logits, want_prefix_logits);
    println!("prefill logits vs reference: peak-rel={pr:.3e}");
    assert!(pr < 5e-3, "prefill logits peak-rel {pr:.3e} exceeds 5e-3");
    assert_eq!(cache.len(), prefix_len);

    // The first generated token comes from the prefix's last-position logits.
    let last_idx = Array::from_slice(&[prefix_len - 1], &[1]);
    let first_logits = prefill_logits
        .take_axis(&last_idx, 1)
        .unwrap()
        .reshape(&[vocab])
        .unwrap();
    let first_logits_vec = first_logits.as_slice::<f32>().to_vec();

    // ---- Greedy decode, step by step, comparing logits + token picks to the reference. ----
    let mut prev = first_logits_vec.clone();
    let mut tokens = Vec::new();
    let mut t = prefix_len - 1;
    let n = want_tokens.len();
    for k in 0..n {
        let tok = argmax(&prev);
        tokens.push(tok);
        t += 1;
        let row = model.decode_logits(tok, t, &mut cache).unwrap();
        let row_arr = Array::from_slice(&row, &[1, vocab]);
        let want_row = want_decode_logits
            .take_axis(Array::from_slice(&[k as i32], &[1]), 0)
            .unwrap();
        let rel = peak_rel(&row_arr, &want_row);
        assert!(
            rel < 5e-3,
            "decode step {k} logits peak-rel {rel:.3e} exceeds 5e-3"
        );
        prev = row;
    }
    println!("greedy stream: {tokens:?}");
    assert_eq!(
        tokens, want_tokens,
        "greedy token stream must match the reference"
    );
    assert_eq!(cache.len(), prefix_len + n as i32);

    // ---- The `generate` wrapper yields the same stream (fresh prefill cache). ----
    let mut cache2 = model.new_cache();
    model
        .forward_cached(&embeds, &pt, &ph, &pw, Path::Und, &mut cache2, true)
        .unwrap();
    let gen = model
        .generate(
            &first_logits_vec,
            &mut cache2,
            prefix_len - 1,
            &[],
            n,
            Sampler::Greedy,
        )
        .unwrap();
    assert_eq!(
        gen, want_tokens,
        "generate() must reproduce the greedy stream"
    );
}

/// The `_generate_think` rollout: the greedy think loop emits the stream up to and including the
/// `</think>` token, forwards every emitted token into the cache, then appends `\n\n<img>`. Built on
/// the already-numerically-validated greedy decode, so this asserts the stop/append control flow and
/// the temporal/cache bookkeeping against the known greedy stream.
#[test]
fn generate_think_stops_and_appends() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let model = Qwen3Backbone::from_weights(&w, &cfg, "language_model").expect("build backbone");

    let prefix_ids = w.require("prefix.input_ids").unwrap().clone();
    let (pt, ph, pw) = index_rows(w.require("prefix.indexes").unwrap());
    let prefix_len = pt.len() as i32;
    let want_tokens: Vec<i32> = w
        .require("decode.tokens")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let embeds = model.embed(&prefix_ids).unwrap();

    let prefill = |model: &Qwen3Backbone| {
        let mut cache = model.new_cache();
        let h = model
            .forward_cached(&embeds, &pt, &ph, &pw, Path::Und, &mut cache, true)
            .unwrap();
        let logits = model.lm_head(&h).unwrap();
        let last = logits
            .take_axis(Array::from_slice(&[prefix_len - 1], &[1]), 1)
            .unwrap()
            .reshape(&[cfg.llm.vocab_size as i32])
            .unwrap();
        (cache, last.as_slice::<f32>().to_vec())
    };

    let append_ids = [5i32, 6, 7]; // stands in for the tokenizer's `\n\n<img>` ids.

    // ---- think_end mid-stream: stop at the first occurrence of the chosen `</think>` id. ----
    let think_end = want_tokens[1]; // 25 — first occurs at index 1
    let first = want_tokens.iter().position(|&t| t == think_end).unwrap();
    let (mut cache, first_logits) = prefill(&model);
    let ThinkRollout {
        think_token_ids,
        t_idx,
    } = model
        .generate_think(
            &first_logits,
            &mut cache,
            prefix_len - 1,
            think_end,
            -1,
            &append_ids,
            64,
        )
        .unwrap();
    // Emitted ids are the greedy stream up to and including the `</think>` token.
    assert_eq!(think_token_ids, want_tokens[..=first].to_vec());
    let forwards = (first + 1) as i32; // each emitted token (incl. </think>) is forwarded once
    assert_eq!(t_idx, (prefix_len - 1) + forwards + append_ids.len() as i32);
    assert_eq!(cache.len(), prefix_len + forwards + append_ids.len() as i32);

    // ---- immediate EOS: the first token is EOS → empty think, only the append lands. ----
    let eos = want_tokens[0]; // 37 — the very first greedy pick
    let (mut cache, first_logits) = prefill(&model);
    let roll = model
        .generate_think(
            &first_logits,
            &mut cache,
            prefix_len - 1,
            -1,
            eos,
            &append_ids,
            64,
        )
        .unwrap();
    assert!(
        roll.think_token_ids.is_empty(),
        "EOS on the first token → no think tokens"
    );
    assert_eq!(roll.t_idx, (prefix_len - 1) + append_ids.len() as i32);
    assert_eq!(cache.len(), prefix_len + append_ids.len() as i32);
}
