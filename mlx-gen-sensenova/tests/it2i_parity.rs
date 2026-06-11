//! sc-3189: the it2i (image-conditioned) spine matches the reference `it2i_generate` loop.
//!
//! Synthetic-fixture parity (weight-free; tiny config incl. the understanding vision embedder).
//! The fixture (`tools/dump_sensenova_it2i_golden.py`) replays the genuine reference image-
//! conditioned prefill (`extract_feature` understanding + `get_thw_indexes` + splice into
//! `<IMG_CONTEXT>` + `_it2i_prefix_forward`) and the cond-only denoise on a random prefix + source
//! image + fixed noise. This test drives [`T2iModel::prefill_it2i`] + [`T2iModel::it2i_denoise`]
//! with the same inputs and matches the per-step trajectory. The tiny vocab can't hold the real
//! special-token ids, so the model is built with [`T2iModel::with_image_token_ids`] (10 / 11),
//! mirroring the dump.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test it2i_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, T2iModel, T2iOptions};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/it2i_golden.safetensors"
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
        "num_channels": 3,
        "patch_size": m("patch_size").parse::<u64>().unwrap(),
        "downsample_ratio": 0.5,
        "rope_theta_vision": 10000.0,
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

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

#[test]
fn it2i_denoise_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let img_context_id: i32 = w.metadata("img_context_id").unwrap().parse().unwrap();
    let img_start_id: i32 = w.metadata("img_start_id").unwrap().parse().unwrap();
    let model = T2iModel::from_weights(&w, &cfg)
        .expect("build T2iModel")
        .with_image_token_ids(img_context_id, img_start_id, 12);

    let width: i32 = w.metadata("width").unwrap().parse().unwrap();
    let height: i32 = w.metadata("height").unwrap().parse().unwrap();
    let num_steps: usize = w.metadata("num_steps").unwrap().parse().unwrap();
    let src_gh: i32 = w.metadata("src_grid_h").unwrap().parse().unwrap();
    let src_gw: i32 = w.metadata("src_grid_w").unwrap().parse().unwrap();

    let ids: Vec<i32> = w
        .require("prefix.input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let pixel_values = w.require("pixel_values").unwrap().clone();
    let raw_noise = w.require("raw_noise").unwrap().clone();
    let want_traj = w.require("traj").unwrap();

    let (mut cache, img_temporal) = model
        .prefill_it2i(&ids, Some(&pixel_values), &[(src_gh, src_gw)])
        .expect("prefill_it2i");

    let opts = T2iOptions {
        cfg_scale: 1.0,
        num_steps,
        ..Default::default()
    };
    let traj = model
        .it2i_denoise(
            (&mut cache, img_temporal),
            None,
            None,
            width,
            height,
            &raw_noise,
            &opts,
            None,
        )
        .expect("it2i_denoise");
    assert_eq!(traj.len(), num_steps);

    let mut worst = 0f32;
    for (i, got) in traj.iter().enumerate() {
        let want = want_traj
            .take_axis(Array::from_slice(&[i as i32], &[1]), 0)
            .unwrap()
            .reshape(&[1, 3, height, width])
            .unwrap();
        let rel = peak_rel(got, &want);
        println!("step {i}: peak-rel={rel:.3e}");
        worst = worst.max(rel);
    }
    assert!(
        worst < 2e-2,
        "it2i trajectory peak-rel {worst:.3e} exceeds 2e-2"
    );
}
