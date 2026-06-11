//! **RESOLVED — not an open numeric question (F-044).** The sc-2349/2532 Q8 investigation is closed
//! (re-validated on MLX 0.31.2 in sc-2782); the live Q8 regression is covered by the e2e Q8 gate.
//! This `#[ignore]`d diagnostic is kept only as a localization probe for `quantized_matmul` at small
//! K, run if that gate ever reddens. It needs the original probe golden
//! (`tools/golden/qmm_smallK_probe.safetensors`, from `probe_qmm_smallK.py`), so it never runs in CI.
//!
//! sc-2349 → sc-2532 decisive probe: is the base z_image Q8 residual the `quantized_matmul`
//! **kernel**, and is it **K-dependent**?
//!
//! sc-2532's `q8_packing_byte_identical_to_fork` proved, for a K=3840 weight, that `quantize` is
//! byte-identical and `quantized_matmul` matches the fork to ~1e-6. But the sc-2349 Q8 bisection
//! found the **K=64** x-embedder's output diverges ~0.3% (→ ~1.26%/forward → ~8% over 8 steps). This
//! test settles it at K=64 with the **real** x-embedder weight + the **real** activation that feeds
//! it (`tools/probe_qmm_smallK.py`):
//!   (1) re-quantize the same weight → `wq` must be byte-identical + scales/biases exact (quantize), and
//!   (2) run `quantized_matmul` on the FORK's **exact** `wq/scales/biases/x` — byte-identical inputs,
//!       so any divergence is purely the kernel (our source-built MLX vs the fork's wheel), K-dependent.
//!
//! `#[ignore]`d — needs the probe golden. Run:
//!   cargo test -p mlx-gen-z-image --release --test qmm_smallk -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_rs::ops::{eq, quantize, quantized_matmul};
use mlx_rs::{Array, Dtype};

const PROBE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qmm_smallK_probe.safetensors"
);

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

/// `(peak-relative, mean-relative)` error vs golden `b`.
mod common;
use common::rel;

#[test]
#[ignore = "needs tools/golden/qmm_smallK_probe.safetensors (tools/probe_qmm_smallK.py)"]
fn qmm_smallk_matches_fork() {
    let g = Weights::from_file(PROBE).unwrap();
    let w = bf16(g.require("w").unwrap()); // f32→bf16 is exact (on-disk weight is bf16)
    let x = g.require("x").unwrap().clone(); // f32 — the dtype the divergent Q8 control path runs
    let fork_wq = g.require("wq").unwrap();
    let fork_scales = bf16(g.require("scales").unwrap());
    let fork_biases = bf16(g.require("biases").unwrap());
    let fork_qmm = g.require("qmm").unwrap();

    // (1) QUANTIZE at K=64: re-quantize the same weight and check the packing vs the fork.
    let (wq, scales, biases) = quantize(&w, 64, 8).unwrap();
    let wq_match = eq(&wq, fork_wq).unwrap().all(None).unwrap().item::<bool>();
    let (sc_pr, _) = rel(&scales, g.require("scales").unwrap());
    let (bi_pr, _) = rel(&biases, g.require("biases").unwrap());
    println!("quantize @K=64: wq_byte_identical={wq_match}  scales_peak_rel={sc_pr:.2e}  biases_peak_rel={bi_pr:.2e}");

    // (2) KERNEL: quantized_matmul on the FORK's exact wq/scales/biases/x — byte-identical inputs.
    let qmm_kernel =
        quantized_matmul(&x, fork_wq, &fork_scales, &fork_biases, true, 64, 8).unwrap();
    let (k_peak, k_mean) = rel(&qmm_kernel, fork_qmm);
    println!(
        "qmm KERNEL @K=64 (byte-identical inputs): peak_rel={k_peak:.3e}  mean_rel={k_mean:.3e}"
    );

    // (3) FULL Rust path: Rust's own quantize feeding Rust's qmm.
    let qmm_full = quantized_matmul(&x, &wq, &scales, &biases, true, 64, 8).unwrap();
    let (f_peak, f_mean) = rel(&qmm_full, fork_qmm);
    println!("qmm FULL (rust quantize+qmm) @K=64: peak_rel={f_peak:.3e}  mean_rel={f_mean:.3e}");

    // Interpretation (printed, not asserted — this is a diagnostic):
    //   wq_match=true & kernel mean_rel≈0  → neither; the x_emb 0.3% came from elsewhere.
    //   wq_match=true & kernel mean_rel≫0  → the qmm KERNEL is K-dependent (build difference).
    //   wq_match=false                      → quantize itself is K-dependent.
}
