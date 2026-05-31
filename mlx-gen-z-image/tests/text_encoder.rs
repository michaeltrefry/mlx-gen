//! sc-2344: full Z-Image TextEncoder forward parity vs the fork (tiny random config).
//!
//! Fixture `tests/fixtures/text_encoder.safetensors` ← `tools/dump_text_encoder.py` (precision
//! pinned to f32 for the dump). Exercises embed lookup → N pre-norm decoder layers → the
//! second-to-last layer's hidden states. attention_mask is all-ones (causal path; the padding
//! combination is unit-tested in `text_encoder::encoder`). 1e-2 tolerance (Metal fp32 matmul).

use mlx_gen::weights::Weights;
use mlx_gen_z_image::text_encoder::{TextEncoder, ZTextEncoderConfig};
use mlx_rs::ops::all_close;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/text_encoder.safetensors"
);

#[test]
fn text_encoder_forward_matches_fork() {
    let w = Weights::from_file(FIXTURE).unwrap();
    // cfg = "VOCAB,H,NL,NH,NKV,HD,INTER,SEQ"
    let p: Vec<i32> = w
        .metadata("cfg")
        .unwrap()
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let cfg = ZTextEncoderConfig {
        vocab_size: p[0],
        hidden_size: p[1],
        n_layers: p[2] as usize,
        n_heads: p[3],
        n_kv_heads: p[4],
        head_dim: p[5],
        intermediate_size: p[6],
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
    };

    let enc = TextEncoder::from_weights(&w, "", &cfg).unwrap();
    let out = enc
        .forward(
            w.require("input_ids").unwrap(),
            w.require("attention_mask").unwrap(),
        )
        .unwrap();

    let golden = w.require("out").unwrap();
    assert_eq!(out.shape(), golden.shape(), "encoder output shape");
    assert!(
        all_close(&out, golden, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "TextEncoder forward (embed + layers + second-to-last selection) diverged from the fork"
    );
}
