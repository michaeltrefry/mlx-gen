//! Kolors Q8/Q4 quantization validation (sc-3096).
//!
//! `#[ignore]`d: needs the Kolors snapshot + the materialized `tokenizer.json`. **No torch/diffusers
//! golden** — Kolors is not in any quantized-reference fork, so (per the other providers' floor-
//! relative gate) the reference is the **same-backend dense bf16 Kolors**. Two gates:
//!
//!  - `kolors_quant_encoder_floor`: the ChatGLM3 conditioning (`context` + `pooled`) under load-time
//!    Q8/Q4 stays within a documented floor of the dense bf16 conditioning (cosine + rel-L2). Q8 is
//!    near-lossless; Q4 is looser but still high-cosine. Records the encoder's resident footprint.
//!  - `kolors_quant_generate_runs`: a full Q8 **and** Q4 `generate` completes and renders coherently
//!    (non-degenerate, brightness tracks the bf16 render). Records peak memory.
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test quant_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_kolors::chatglm3::{ChatGlmConfig, ChatGlmModel};
use mlx_gen_kolors::tokenizer::KolorsTokenizer;
use mlx_gen_kolors::Kolors;
use mlx_rs::memory::{get_active_memory, get_peak_memory, reset_peak_memory};
use mlx_rs::{Array, Dtype};

const PROMPT: &str = "A cat playing a grand piano on a city rooftop at sunset.";

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Cosine similarity + relative-L2 of two same-shape tensors (flattened, f32).
fn cos_rel(a: &Array, b: &Array) -> (f64, f64) {
    let n = a.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (
        a.as_dtype(Dtype::Float32).unwrap(),
        b.as_dtype(Dtype::Float32).unwrap(),
    );
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    let mut diff = 0f64;
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
        diff += ((x - y) as f64) * ((x - y) as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    let rel = diff.sqrt() / nb.sqrt().max(1e-30);
    (cos, rel)
}

fn load_chatglm(snap: &std::path::Path) -> ChatGlmModel {
    let te = Weights::from_dir(snap.join("text_encoder")).unwrap();
    ChatGlmModel::from_weights(&te, ChatGlmConfig::chatglm3_6b(), None, Dtype::Bfloat16).unwrap()
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json (loads the 12.5GB ChatGLM3 encoder)"]
fn kolors_quant_encoder_floor() {
    let snap = snapshot();
    let tok = KolorsTokenizer::from_dir(snap.join("tokenizer")).unwrap();
    let t = tok.encode(PROMPT).unwrap();
    let enc = |m: &ChatGlmModel| {
        let (c, p) = m
            .encode_prompt(&t.input_ids, &t.attention_mask, Some(&t.position_ids))
            .unwrap();
        c.eval().unwrap();
        p.eval().unwrap();
        (c, p)
    };

    // --- dense bf16 reference ---
    let mut m = load_chatglm(&snap);
    let (ctx_ref, pooled_ref) = enc(&m);
    let mem_bf16 = get_active_memory() as f64 / 1e9;
    println!("bf16 encoder active mem ≈ {mem_bf16:.2} GB");

    // --- Q8 in place (the dense weights drop as each projection is packed) ---
    m.quantize(8).unwrap();
    let (ctx8, pooled8) = enc(&m);
    let mem_q8 = get_active_memory() as f64 / 1e9;
    let (cc8, cr8) = cos_rel(&ctx8, &ctx_ref);
    let (pc8, pr8) = cos_rel(&pooled8, &pooled_ref);
    println!("Q8: context cos={cc8:.6} rel={cr8:.4} | pooled cos={pc8:.6} rel={pr8:.4} | active mem ≈ {mem_q8:.2} GB");
    drop(m);

    // --- Q4 (fresh dense → quantize) ---
    let mut m4 = load_chatglm(&snap);
    m4.quantize(4).unwrap();
    let (ctx4, pooled4) = enc(&m4);
    let mem_q4 = get_active_memory() as f64 / 1e9;
    let (cc4, cr4) = cos_rel(&ctx4, &ctx_ref);
    let (pc4, pr4) = cos_rel(&pooled4, &pooled_ref);
    println!("Q4: context cos={cc4:.6} rel={cr4:.4} | pooled cos={pc4:.6} rel={pr4:.4} | active mem ≈ {mem_q4:.2} GB");

    // Documented floor (measured on the Kolors snapshot, group 64):
    //   Q8 — context cos ≈ 0.9990 / rel ≈ 0.045, pooled cos ≈ 0.9999 (near-lossless);
    //   Q4 — context cos ≈ 0.950  / rel ≈ 0.31,  pooled cos ≈ 0.992.
    // The penultimate `context` is the most quant-sensitive tensor (a deep residual-stream state
    // accumulating per-group error across 26 layers, with large-magnitude outlier channels that
    // inflate rel-L2), yet the Q4 render stays coherent (see `kolors_quant_generate_runs`) — the
    // SDXL-family cross-attention is robust to that conditioning perturbation. So the binding quality
    // gate is the e2e render; these cosines guard against gross packing/dequant corruption (which
    // would tank `pooled` too and break the render), not exact fidelity.
    assert!(
        cc8 > 0.998,
        "Q8 context cosine {cc8:.6} below floor (packing/dequant bug?)"
    );
    assert!(
        pc8 > 0.999,
        "Q8 pooled cosine {pc8:.6} below floor (packing/dequant bug?)"
    );
    assert!(
        cc4 > 0.93,
        "Q4 context cosine {cc4:.6} below floor (packing/dequant bug?)"
    );
    assert!(
        pc4 > 0.98,
        "Q4 pooled cosine {pc4:.6} below floor (packing/dequant bug?)"
    );
    println!(
        "✓ Q8/Q4 ChatGLM3 conditioning within the documented floor; encoder footprint recorded"
    );
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json (loads the full pipeline per bit-width)"]
fn kolors_quant_generate_runs() {
    let snap = snapshot();
    let (h, w, steps, cfg) = (512, 512, 8, 5.0);

    // bf16 render brightness as the coherence anchor.
    let bf16 = Kolors::load(&snap, Dtype::Bfloat16).unwrap();
    let img_bf16 = bf16
        .generate(PROMPT, "blurry, low quality", steps, cfg, 0, h, w)
        .unwrap();
    let mean_bf16: f64 =
        img_bf16.pixels.iter().map(|&v| v as f64).sum::<f64>() / img_bf16.pixels.len() as f64;
    drop(bf16);
    println!("bf16 render mean brightness {mean_bf16:.1}");

    for bits in [8, 4] {
        reset_peak_memory();
        let m = Kolors::load_quantized(&snap, Dtype::Bfloat16, bits).unwrap();
        let img = m
            .generate(PROMPT, "blurry, low quality", steps, cfg, 0, h, w)
            .unwrap();
        let peak = get_peak_memory() as f64 / 1e9;
        let mean: f64 = img.pixels.iter().map(|&v| v as f64).sum::<f64>() / img.pixels.len() as f64;
        println!("Q{bits}: render mean brightness {mean:.1} | peak mem ≈ {peak:.2} GB");
        assert_eq!(img.pixels.len(), (h * w * 3) as usize, "Q{bits} image size");
        assert!(
            img.pixels.iter().any(|&p| p > 16) && img.pixels.iter().any(|&p| p < 239),
            "Q{bits} degenerate render"
        );
        assert!(
            (mean - mean_bf16).abs() < 25.0,
            "Q{bits} brightness {mean:.1} diverges from bf16 {mean_bf16:.1}"
        );
        drop(m);
    }
    println!(
        "✓ Q8 + Q4 full Kolors generate completes and renders coherently; peak memory recorded"
    );
}
