//! S1 parity gate: the UMT5-XXL text encoder must reproduce the `mlx_video` reference's prompt
//! embeddings, and `clean_text` must reproduce the reference's `_clean_text`.
//!
//! Two tiers (mirrors the LTX/FLUX convention):
//!   - **Always-on:** `clean_text` vs the reference's cleaned strings (committed in `s1.json`). This
//!     gates the ftfy port (the fullwidth-comma fold for the Chinese negative prompt) with no weights.
//!   - **`#[ignore]` heavy:** load the real `t5_encoder.safetensors` (~11 GB) + `tokenizer.json` and
//!     compare encoder output to the committed golden embeds. Resolve the snapshot via `WAN_5B_DIR`
//!     (default `~/Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b`); both are
//!     produced by `tools/dump_s1_fixtures.py`.
//!
//! Honors "divergence is not rounding": gate against the real reference, root-cause any real gap.
//! The encoder is **bit-exact** to the reference (max|Δ| = 0.0 on every prompt incl. the 126-token
//! Chinese negative). An early ~1e-3 gap was *not* the cross-build f32 accumulation it first looked
//! like — a controlled per-op floor test (matmul / rms_norm / softmax all 0.0 between the 0.31.2
//! wheel and pmetal-0.31.1 on bit-identical inputs) localized it to `gelu`: mlx-rs computes the
//! `√(2/π)` constant with an f32 MLX op vs MLX-Python's f64 host constant (1 ULP). The hand-rolled
//! `gelu_tanh` (f64 constant) closed it to zero. See `text_encoder::gelu_tanh`.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::{clean_text, load_tokenizer, Umt5Encoder};
use serde_json::Value;

fn fixtures() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/s1.json");
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (run tools/dump_s1_fixtures.py)"));
    serde_json::from_str(&text).expect("parse s1.json")
}

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_5B_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b")
}

/// Always-on: `clean_text` reproduces the reference `_clean_text` for every fixture prompt. Catches
/// any drift in the ftfy/HTML/whitespace port — including the load-bearing fullwidth-comma fold on
/// the Chinese negative prompt — without needing the encoder weights.
#[test]
fn clean_text_matches_reference() {
    let fx = fixtures();
    for (name, p) in fx["prompts"].as_object().unwrap() {
        let prompt = p["prompt"].as_str().unwrap();
        let expected = p["cleaned"].as_str().unwrap();
        let got = clean_text(prompt);
        assert_eq!(got, expected, "[{name}] clean_text mismatch");
    }
}

#[test]
#[ignore = "needs t5_encoder.safetensors (~11 GB) + tokenizer.json — run tools/dump_s1_fixtures.py"]
fn umt5_embeds_match_reference() {
    let dir = snapshot_dir();
    let cfg = WanModelConfig::wan22_ti2v_5b();

    let w = Weights::from_file(dir.join("t5_encoder.safetensors")).expect("t5_encoder.safetensors");
    let encoder = Umt5Encoder::from_weights(&w, &cfg).expect("build encoder");
    let tok = load_tokenizer(dir.join("tokenizer.json"), cfg.text_len).expect("tokenizer.json");

    let golden = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s1_t5_golden.safetensors"
    ))
    .expect("s1 golden");

    // Block-0 gate: the per-op math (rms_norm + unscaled f32 attention + per-layer bias + gated
    // GELU) is bit-exact to the reference on bit-identical input (the token embedding is byte-equal).
    {
        let out = tok
            .tokenize_preformatted(&clean_text("a cat playing the piano"))
            .unwrap();
        let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&out);
        let stages = encoder
            .forward_capture(&input_ids, &attention_mask)
            .unwrap();
        let dim = 4096i32;
        let flat = stages[1].reshape(&[512, dim]).unwrap();
        let idx = mlx_rs::Array::from_slice(&(0..6i32).collect::<Vec<i32>>(), &[6]);
        let got = flat.take_axis(&idx, 0).unwrap().as_slice::<f32>().to_vec();
        let exp = golden
            .require("block0_english")
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let (max_abs, mean_rel) = diff(&got, &exp);
        println!("[block0] max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
        assert_eq!(max_abs, 0.0, "block-0 not bit-exact: max|Δ|={max_abs:.3e}");
    }

    let fx = fixtures();
    for (name, p) in fx["prompts"].as_object().unwrap() {
        let prompt = p["prompt"].as_str().unwrap();
        let ref_ids: Vec<i32> = p["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_i64().unwrap() as i32)
            .collect();
        let seq_len = p["seq_len"].as_u64().unwrap() as usize;

        // Tokenization parity: cleaned + tokenized ids match the reference's (the non-pad prefix).
        let out = tok.tokenize_preformatted(&clean_text(prompt)).unwrap();
        let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&out);
        let ids: Vec<i32> = input_ids.as_slice::<i32>()[..seq_len].to_vec();
        assert_eq!(ids, ref_ids, "[{name}] token ids differ");

        // Embedding parity.
        let got = encoder.encode(&tok, prompt).unwrap();
        let got = got.as_slice::<f32>().to_vec();
        let exp = golden
            .require(&format!("embeds_{name}"))
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        assert_eq!(got.len(), exp.len(), "[{name}] embed length");

        let (max_abs, mean_rel) = diff(&got, &exp);
        println!("[{name}] seq_len={seq_len} max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
        // Bit-exact across all 24 layers (incl. the 126-token Chinese negative). The whole op set
        // — token-embed gather, rms_norm, unscaled f32 attention + per-layer bias + masked softmax,
        // gated GELU, and the batched matmuls — matches the reference byte-for-byte.
        assert_eq!(max_abs, 0.0, "[{name}] not bit-exact: max|Δ|={max_abs:.3e}");
    }
}

/// `(max|Δ|, Σ|Δ| / Σ|ref|)` over two equal-length f32 slices.
fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}
