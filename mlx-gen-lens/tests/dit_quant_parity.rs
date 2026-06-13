//! sc-3175 — Lens DiT Q4/Q8 quantization parity vs the dense bf16 DiT.
//!
//! Loads the real `transformer/` weights at **bf16**, runs the dense forward over the `dit_parity`
//! golden's synthetic inputs, then quantizes the DiT ([`LensTransformer::quantize`]) and re-runs —
//! asserting the Q8 output is near-lossless and the Q4 output stays coherent vs the dense bf16 DiT
//! (the standard load-time-quant gate across the codebase; no torch reference needed). `#[ignore]`d —
//! needs the golden + the ~8 GB bf16 transformer snapshot.
//!
//! Run: `cargo test -p mlx-gen-lens --test dit_quant_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::Quant;
use mlx_gen_lens::dit::{LensDitConfig, LensTransformer};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_dit_golden.safetensors"
);

fn transformer_dir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
        .join("transformer")
}

fn meta_usize(g: &Weights, key: &str) -> usize {
    g.metadata(key).unwrap().parse().unwrap()
}

fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

#[test]
#[ignore = "needs tools/golden/lens_dit_golden.safetensors + the Lens-Turbo transformer snapshot (~8GB bf16 load)"]
fn lens_dit_quant_matches_dense() {
    let g = Weights::from_file(GOLDEN).expect("dit golden");
    let (frame, h, w) = (
        meta_usize(&g, "frame"),
        meta_usize(&g, "h_lat"),
        meta_usize(&g, "w_lat"),
    );
    let n_text = meta_usize(&g, "n_text");
    let cfg = LensDitConfig::lens();

    // Inputs are bf16 (the production DiT dtype the quant path runs at).
    let bf16 = |k: &str| g.require(k).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    let hidden = bf16("hidden_states");
    let feats: Vec<Array> = (0..n_text).map(|i| bf16(&format!("feat_{i}"))).collect();
    let timestep = bf16("timestep");

    let run = |dit: &LensTransformer| -> Array {
        dit.forward(&hidden, &feats, None, &timestep, frame, h, w)
            .expect("forward")
    };

    eprintln!("loading transformer (bf16)…");
    let weights = Weights::from_dir(transformer_dir()).expect("transformer shards");

    // Dense bf16 reference, then Q8 (same instance — quantize is in-place).
    let mut dit = LensTransformer::from_weights(&weights, &cfg, Dtype::Bfloat16).expect("load DiT");
    let dense = run(&dit);
    dit.quantize(Quant::Q8.bits()).expect("quantize Q8");
    let q8 = run(&dit);
    let q8_cos = cosine(&q8, &dense);
    eprintln!(
        "Q8 vs dense bf16: cosine {q8_cos:.6}  peak_rel {:.3e}",
        peak_rel(&q8, &dense)
    );

    // Fresh dense DiT → Q4.
    let mut dit = LensTransformer::from_weights(&weights, &cfg, Dtype::Bfloat16).expect("load DiT");
    dit.quantize(Quant::Q4.bits()).expect("quantize Q4");
    let q4 = run(&dit);
    let q4_cos = cosine(&q4, &dense);
    eprintln!(
        "Q4 vs dense bf16: cosine {q4_cos:.6}  peak_rel {:.3e}",
        peak_rel(&q4, &dense)
    );

    // Q8 is near-lossless. Q4 is lossier — a single full DiT forward quantizes `img_in`/`txt_in`/
    // `proj_out` + the attention projections + SwiGLU MLPs across all 48 blocks, so the per-forward
    // cosine sits at the 4-bit floor (~0.86 here), in line with the Q4 precedent elsewhere (e.g.
    // SenseNova T2I Q4 ~0.84) — coherent, not collapsed. The denoise then runs many such forwards;
    // the e2e render stays coherent (the registry exposes Q4 for the memory-constrained tier).
    assert!(
        q8_cos > 0.99,
        "Q8 DiT cosine {q8_cos:.6} ≤ 0.99 — not near-lossless"
    );
    assert!(
        q4_cos > 0.80,
        "Q4 DiT cosine {q4_cos:.6} ≤ 0.80 — collapsed, not a coherent quantization"
    );
    eprintln!("ALL PASS");
}
