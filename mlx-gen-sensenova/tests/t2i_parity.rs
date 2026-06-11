//! sc-3188: the T2I denoise spine matches the reference `t2i_generate` loop.
//!
//! Synthetic-fixture parity (weight-free; tiny dense dual-path config + NEO vision embedder + shallow
//! `fm_head` + timestep/noise-scale embedders). The fixture (`tools/dump_sensenova_t2i_golden.py`)
//! replays the genuine reference denoise loop — `_t2i_prefix_forward` → per-step
//! patchify/`extract_feature`(gen)/timestep+noise embed/`_t2i_predict_v`/`_euler_step`/`unpatchify`
//! — on a random prefix and a fixed initial noise. This test drives [`T2iModel::prefill_ids`] +
//! [`T2iModel::denoise`] with the same prefix + noise and matches the full per-step image
//! trajectory. Exercises the new sc-3188 wiring (channel-first patchify, gen-path vision embed +
//! timestep/noise conditioning, gen-path cached forward use-only, fm_head→velocity→euler→unpatchify,
//! the resolution noise_scale) end to end. Cross-build: f32, tolerance covers the SDPA-vs-eager +
//! MLX-Metal-vs-torch f32 floor accumulated over the steps.
//!
//! Run: `cargo test -p mlx-gen-sensenova --test t2i_parity -- --nocapture`

use mlx_gen::weights::Weights;
use mlx_gen_sensenova::{NeoChatConfig, T2iModel, T2iOptions};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/t2i_golden.safetensors"
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
        "model_type": "neo_chat",
        "tie_word_embeddings": false,
        "patch_size": m("patch_size").parse::<u64>().unwrap(),
        "downsample_ratio": 0.5,
        "noise_scale_mode": "resolution",
        "noise_scale": 1.0,
        "noise_scale_max_value": 8.0,
        "noise_scale_base_image_seq_len": 64,
        "add_noise_scale_embedding": true,
        "llm_config": llm,
        "vision_config": vision,
    });
    NeoChatConfig::from_config_json(&v).expect("synthetic parity config is valid")
}

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

#[test]
fn t2i_denoise_matches_reference() {
    let w = Weights::from_file(FIXTURE).expect("load fixture");
    let cfg = config_from_meta(&w);
    let model = T2iModel::from_weights(&w, &cfg).expect("build T2iModel");

    let width: i32 = w.metadata("width").unwrap().parse().unwrap();
    let height: i32 = w.metadata("height").unwrap().parse().unwrap();
    let num_steps: usize = w.metadata("num_steps").unwrap().parse().unwrap();

    let prefix_ids: Vec<i32> = w
        .require("prefix.input_ids")
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let raw_noise = w.require("raw_noise").unwrap().clone();
    let want_traj = w.require("traj").unwrap(); // [num_steps, 3, H, W]

    let (mut cache, text_len) = model.prefill_ids(&prefix_ids).expect("prefill");

    let opts = T2iOptions {
        cfg_scale: 1.0,
        num_steps,
        timestep_shift: 1.0,
        enable_timestep_shift: true,
        t_eps: 0.02,
        ..Default::default()
    };
    let traj = model
        .denoise(
            &mut cache, text_len, None, width, height, &raw_noise, &opts, None,
        )
        .expect("denoise");
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
    // Cross-build (SDPA-vs-eager + MLX-Metal f32 matmul floor) accumulated over the denoise steps.
    assert!(
        worst < 2e-2,
        "T2I trajectory peak-rel {worst:.3e} exceeds 2e-2"
    );
}
