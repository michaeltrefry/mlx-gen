//! sc-3190: interleave's `append_generated_image` mechanic matches the reference.
//!
//! Synthetic-fixture parity (weight-free; tiny config incl. the understanding vision embedder). The
//! fixture (`tools/dump_sensenova_interleave_golden.py`) prefills a text cache, then replays the
//! reference `interleave_gen`'s inner `append_image_to_cache` on a fixed image (re-encode through the
//! understanding vision embedder + `</img>` at the `t+1`/`t+2` temporal layout under the
//! image-doesn't-see-`</img>` mask) and decodes one token. This test drives
//! [`T2iModel::append_generated_image`] on the same prefix + image and matches the next-token logits
//! and greedy pick — the one genuinely new numeric piece of the interleave loop (the rest being the
//! sc-3187 decode and sc-3188/3189 image gen). Small image-token ids (10/11) via
//! [`T2iModel::with_image_token_ids`], mirroring the dump.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test interleave_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, T2iModel};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/interleave_golden.safetensors"
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

fn argmax(v: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best as i32
}

#[test]
fn append_generated_image_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let img_context_id: i32 = w.metadata("img_context_id").unwrap().parse().unwrap();
    let img_start_id: i32 = w.metadata("img_start_id").unwrap().parse().unwrap();
    let token_h: i32 = w.metadata("token_h").unwrap().parse().unwrap();
    let token_w: i32 = w.metadata("token_w").unwrap().parse().unwrap();
    let want_token: i32 = w.metadata("next_token").unwrap().parse().unwrap();
    let model = T2iModel::from_weights(&w, &cfg)
        .expect("build")
        .with_image_token_ids(img_context_id, img_start_id, 12);

    let ids: Vec<i32> = w
        .require("prefix.input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let image = w.require("image").unwrap().clone();
    let want_logits = w.require("next_logits").unwrap().clone();

    // Prefill the text prefix, then append the generated image and read the next-token logits.
    let (mut cache, _, t_idx) = model.prefill_it2i_logits(&ids, None, &[]).expect("prefill");
    let (logits, new_t) = model
        .append_generated_image(&image, token_h, token_w, t_idx, &mut cache)
        .expect("append");
    assert_eq!(
        new_t,
        t_idx + 2,
        "t_idx must advance by 2 (image block + </img>)"
    );

    let got = argmax(&logits);
    println!("append_generated_image: next_token got {got} want {want_token}");
    assert_eq!(
        got, want_token,
        "next token after append must match the reference"
    );

    // Logits within the f32 floor.
    let vocab = want_logits.shape().iter().product::<i32>();
    let got_arr = Array::from_slice(&logits, &[vocab]);
    let a = got_arr.as_slice::<f32>();
    let b = want_logits.reshape(&[vocab]).unwrap();
    let b = b.as_slice::<f32>();
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    let rel = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak;
    println!("append logits peak-rel={rel:.3e}");
    assert!(rel < 5e-3, "append logits peak-rel {rel:.3e} exceeds 5e-3");
}
