//! sc-2346 S1: real-weights smoke for the FLUX.2 Qwen3 text encoder + tokenizer.
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.2-klein-9b` snapshot in the HF cache and
//! the golden produced by `tools/dump_flux2_te_real_golden.py` (gitignored, local):
//!
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_te_real_golden.py
//!   cargo test -p mlx-gen-flux2 --test te_real_weights -- --ignored --nocapture
//!
//! The committed `te_parity.rs` proves the encoder *math* bit-tight in f32 on a tiny config; this
//! proves the *loader* (shard reading + `model.` key mapping) and the *tokenizer* (chat template +
//! padding) on the real checkpoint. The Rust TE runs f32 activations while the fork golden is the
//! fork's production **bf16** — so the gate is a generous mean-relative bound (a gross loader/key
//! bug diverges by ~100%; the residual here is the expected bf16-vs-f32 accumulation over 36
//! layers, with Rust f32 the more-accurate side).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_flux2::{load_text_encoder, load_tokenizer};
use mlx_rs::Array;
use mlx_rs::Dtype;

const PROMPT: &str = "a red fox resting in fresh snow under soft winter light";

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot under models--black-forest-labs--FLUX.2-klein-9b/snapshots")
}

/// The f32 fork golden (`FLUX2_TE_F32=1`) — the correctness reference (same precision as the Rust
/// port). Falls back to the bf16 golden if only that exists.
fn golden() -> (Weights, bool) {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    let f32 = base.join("flux2_te_real_f32.safetensors");
    if let Ok(w) = Weights::from_file(&f32) {
        return (w, true);
    }
    let bf16 = base.join("flux2_te_real.safetensors");
    let w = Weights::from_file(&bf16).unwrap_or_else(|_| {
        panic!(
            "missing {} — run tools/dump_flux2_te_real_golden.py (FLUX2_TE_F32=1 for the f32 ref)",
            bf16.display()
        )
    });
    (w, false)
}

/// `(peak-relative, mean-relative)` error vs golden `b`.
fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (xs, ys) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = ys.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-12);
    let mabs = (ys.iter().map(|y| y.abs()).sum::<f32>() / ys.len() as f32).max(1e-12);
    let max_diff = xs
        .iter()
        .zip(ys)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_diff = xs.iter().zip(ys).map(|(x, y)| (x - y).abs()).sum::<f32>() / xs.len() as f32;
    (max_diff / peak, mean_diff / mabs)
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot"]
fn tokenizer_ids_match_fork() {
    let tok = load_tokenizer(&snapshot()).unwrap();
    let out = tok.tokenize(PROMPT).unwrap();
    let (input_ids, _) = mlx_gen::tokenizer::to_arrays(&out);
    let (g, _) = golden();
    let want = g.require("input_ids").unwrap();
    assert_eq!(input_ids.shape(), want.shape(), "input_ids shape (512)");
    // Exact integer match: the chat template + BPE + padding must reproduce the fork byte-for-byte.
    let got = input_ids.as_dtype(Dtype::Float32).unwrap();
    let want_f = want.as_dtype(Dtype::Float32).unwrap();
    let eq = mlx_rs::ops::all_close(&got, &want_f, 0.0, 0.0, false)
        .unwrap()
        .item::<bool>();
    assert!(eq, "tokenizer input_ids diverged from the fork");
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b snapshot + tools/golden/flux2_te_real.safetensors"]
fn text_encoder_prompt_embeds_match_fork() {
    let te = load_text_encoder(&snapshot()).unwrap();
    let (g, is_f32) = golden();
    let out = te
        .prompt_embeds(
            g.require("input_ids").unwrap(),
            g.require("attention_mask").unwrap(),
        )
        .unwrap();
    let want = g.require("prompt_embeds").unwrap();
    assert_eq!(out.shape(), want.shape(), "prompt_embeds shape");
    let (peak, mean) = rel(&out, want);
    let ref_kind = if is_f32 { "fork f32" } else { "fork bf16" };
    println!(
        "flux2 TE real-weights: peak_rel={peak:.5} mean_rel={mean:.5} (Rust f32 vs {ref_kind})"
    );
    // Against the f32 golden (same precision as the Rust port) the gate is tight — this isolates
    // port correctness from the fork's production bf16. Against the bf16 golden the residual is the
    // expected bf16-vs-f32 accumulation over 36 layers (Rust f32 the more-accurate side).
    let bound = if is_f32 { 5e-3 } else { 3.5e-2 };
    assert!(
        mean < bound,
        "TE prompt_embeds diverged: mean_rel={mean} (ref={ref_kind})"
    );
}
