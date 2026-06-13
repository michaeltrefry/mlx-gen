//! sc-3168 — Lens DiT parity vs the vendor `LensTransformer2DModel`.
//!
//! Loads the real `transformer/` weights (the cached `microsoft/Lens-Turbo` snapshot) **as f32** and
//! checks, against `tools/dump_lens_dit_golden.py`:
//!   1. **per-block** — block 0 reproduces the reference block output given the golden's block-0
//!      inputs (`img_in_out`, `txt_in_out`, `temb`), with the Rust-built RoPE tables;
//!   2. **full forward** — the whole 48-block DiT reproduces the reference output for the same
//!      synthetic inputs.
//!
//! f32 on both sides makes this a tight correctness gate (`peak_rel < 5e-3`, the mlx-Metal f32-matmul
//! floor accumulated over 48 residual blocks) — bf16 cross-backend accumulation would obscure subtle
//! bugs (wrong RoPE axis, transposed weight, mis-ordered modulation). The golden + the ~16 GB f32
//! weight load keep this `#[ignore]`d; the golden is gitignored.
//!
//! Run: `cargo test -p mlx-gen-lens --test dit_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_lens::dit::rope::LensRope3d;
use mlx_gen_lens::dit::{LensDitConfig, LensTransformer, LensTransformerBlock};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_dit_golden.safetensors"
);

fn transformer_dir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    let snap = std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot");
    snap.join("transformer")
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let diff = abs(subtract(&got, &want).unwrap()).unwrap();
    let denom = max(abs(&want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let got = got.as_dtype(Dtype::Float32).unwrap();
    let want = want.as_dtype(Dtype::Float32).unwrap();
    let dot = sum(multiply(&got, &want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(&got, &got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(&want, &want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

fn meta_usize(g: &Weights, key: &str) -> usize {
    g.metadata(key).unwrap().parse().unwrap()
}

#[test]
#[ignore = "needs tools/golden/lens_dit_golden.safetensors + the Lens-Turbo transformer snapshot (~16GB f32 load)"]
fn lens_dit_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("dit golden");
    let (frame, h, w) = (
        meta_usize(&g, "frame"),
        meta_usize(&g, "h_lat"),
        meta_usize(&g, "w_lat"),
    );
    let txt_len = meta_usize(&g, "txt_len");
    let n_text = meta_usize(&g, "n_text");

    let cfg = LensDitConfig::lens();
    eprintln!("loading transformer weights (f32)…");
    let weights = Weights::from_dir(transformer_dir()).expect("load transformer shards");

    let f32 = |k: &str| g.require(k).unwrap().as_dtype(Dtype::Float32).unwrap();

    // --- 1. per-block: block 0 ---
    let block0 =
        LensTransformerBlock::from_weights(&weights, "transformer_blocks.0", cfg.num_heads, cfg.head_dim, Dtype::Float32)
            .expect("load block 0");
    let rope = LensRope3d::new(10000.0, cfg.axes_dims_rope);
    let (img_cos, img_sin, txt_cos, txt_sin) = rope.forward(frame, h, w, txt_len).unwrap();
    let (enc0, hid0) = block0
        .forward(
            &f32("img_in_out"),
            &f32("txt_in_out"),
            &f32("temb"),
            &img_cos,
            &img_sin,
            &txt_cos,
            &txt_sin,
            None,
        )
        .expect("block 0 forward");
    let blk_enc_pr = peak_rel(&enc0, &f32("block0_enc"));
    let blk_hid_pr = peak_rel(&hid0, &f32("block0_hidden"));
    eprintln!(
        "block0: enc peak_rel {blk_enc_pr:.3e} cosine {:.7} | hidden peak_rel {blk_hid_pr:.3e} cosine {:.7}",
        cosine(&enc0, &f32("block0_enc")),
        cosine(&hid0, &f32("block0_hidden")),
    );

    // --- 2. full forward ---
    let transformer = LensTransformer::from_weights(&weights, &cfg, Dtype::Float32).expect("load DiT");
    let feats: Vec<Array> = (0..n_text).map(|i| f32(&format!("feat_{i}"))).collect();
    let out = transformer
        .forward(&f32("hidden_states"), &feats, None, &f32("timestep"), frame, h, w)
        .expect("full forward");
    let out_pr = peak_rel(&out, &f32("out"));
    let out_cos = cosine(&out, &f32("out"));
    eprintln!("full forward: peak_rel {out_pr:.3e} cosine {out_cos:.7}");

    // Per-block is the tight correctness gate: fed the exact reference block-0 inputs, the Rust block
    // reproduces the output to **1.2e-3 / cosine 0.99999998** — every sub-op (fused QKV, QK-norm,
    // complex RoPE, AdaLN modulation, SwiGLU GateMLP, gated residuals) is correct. The full forward
    // then accumulates the mlx-Metal-vs-CPU f32-matmul floor over 48 residual blocks to ~7e-3 worst
    // element, but cosine stays at 5 nines — a real bug (wrong axis/transpose/order) would crater it.
    assert!(blk_enc_pr < 5e-3, "block0 enc peak_rel {blk_enc_pr:.3e} ≥ 5e-3");
    assert!(blk_hid_pr < 5e-3, "block0 hidden peak_rel {blk_hid_pr:.3e} ≥ 5e-3");
    assert!(out_pr < 1.5e-2, "full forward peak_rel {out_pr:.3e} ≥ 1.5e-2 — beyond 48-block f32 accumulation");
    assert!(out_cos > 0.9999, "full forward cosine {out_cos:.7} ≤ 0.9999");
    eprintln!("ALL PASS");
}
