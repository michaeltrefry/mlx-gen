//! sc-3191: the VQA / understanding spine matches the reference greedy decode.
//!
//! Synthetic-fixture parity (weight-free; tiny config incl. the understanding vision embedder). The
//! fixture (`tools/dump_sensenova_vqa_golden.py`) prefills an **image-conditioned** question prefix
//! (und-vision splice + `get_thw_indexes`) and greedy-decodes N tokens via the reference's
//! single-token mechanic. This test drives [`T2iModel::prefill_it2i_logits`] +
//! [`T2iModel::decode_text`] (the pieces `vqa` composes) and asserts the greedy token stream is
//! bit-identical. The tiny vocab can't hold the real special-token ids, so the model is built with
//! [`T2iModel::with_image_token_ids`] (10 / 11), mirroring the dump.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test vqa_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, Sampler, T2iModel};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/vqa_golden.safetensors"
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
    let vision = serde_json::json!({
        "hidden_size": m("vision_hidden_size").parse::<u64>().unwrap(),
        "llm_hidden_size": m("hidden_size").parse::<u64>().unwrap(),
        "num_channels": 3, "patch_size": m("patch_size").parse::<u64>().unwrap(),
        "downsample_ratio": 0.5, "rope_theta_vision": 10000.0,
    });
    let v = serde_json::json!({
        "model_type": "neo_chat", "tie_word_embeddings": false,
        "patch_size": m("patch_size").parse::<u64>().unwrap(), "downsample_ratio": 0.5,
        "noise_scale_mode": "resolution", "noise_scale": 1.0, "noise_scale_max_value": 8.0,
        "noise_scale_base_image_seq_len": 64, "add_noise_scale_embedding": true,
        "llm_config": llm, "vision_config": vision,
    });
    NeoChatConfig::from_config_json(&v).expect("synthetic parity config is valid")
}

#[test]
fn vqa_greedy_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let img_context_id: i32 = w.metadata("img_context_id").unwrap().parse().unwrap();
    let img_start_id: i32 = w.metadata("img_start_id").unwrap().parse().unwrap();
    let model = T2iModel::from_weights(&w, &cfg)
        .expect("build")
        .with_image_token_ids(img_context_id, img_start_id, 12);

    let src_gh: i32 = w.metadata("src_grid_h").unwrap().parse().unwrap();
    let src_gw: i32 = w.metadata("src_grid_w").unwrap().parse().unwrap();
    let ids: Vec<i32> = w
        .require("prefix.input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let pixel_values = w.require("pixel_values").unwrap().clone();
    let want: Vec<i32> = w
        .require("decode.tokens")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();

    let (mut cache, first_logits, t_idx) = model
        .prefill_it2i_logits(&ids, Some(&pixel_values), &[(src_gh, src_gw)])
        .expect("prefill");
    // No EOS so the loop runs the full token budget, matching the reference's fixed N steps.
    let got = model
        .decode_text(
            &first_logits,
            &mut cache,
            t_idx,
            &[],
            want.len(),
            Sampler::Greedy,
        )
        .expect("decode");

    println!("vqa greedy stream: {got:?}");
    assert_eq!(
        got, want,
        "image-conditioned greedy stream must match the reference"
    );
}
