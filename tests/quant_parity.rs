//! sc-2342: Q4/Q8 quantization parity vs the Python mflux fork (mlx 0.31), which quantizes
//! via `nn.quantize(model, bits=bits)` at group_size=64. The crate links mlx-rs 0.25 (an
//! OLDER bundled MLX), so this checks the epic's flagged version-drift risk at two levels:
//!   * byte-level: the packed `wq` / `scales` / `biases` match exactly;
//!   * semantic: `dequantize` and `quantized_matmul` agree within tolerance.
//!
//! Fixture `tests/fixtures/quant_q4q8.safetensors` ← `tools/dump_quant.py`.

use mlx_gen::adapters::AdaptableLinear;
use mlx_gen::weights::Weights;
use mlx_rs::ops::{all_close, array_eq, dequantize, quantize, quantized_matmul};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/quant_q4q8.safetensors"
);
const GROUP_SIZE: i32 = 64;

fn exact(a: &Array, b: &Array) -> bool {
    array_eq(a, b, false).unwrap().item::<bool>()
}
fn close(a: &Array, b: &Array, rtol: f64, atol: f64) -> bool {
    all_close(a, b, rtol, atol, false).unwrap().item::<bool>()
}

#[test]
fn quantize_packs_identically_to_mflux() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let weight = w.require("w").unwrap();

    for bits in [8, 4] {
        let (wq, scales, biases) = quantize(weight, GROUP_SIZE, bits).unwrap();
        let p = |s: &str| format!("q{bits}.{s}");

        // Byte-level packing parity (the version-drift gate).
        assert!(
            exact(&wq, w.require(&p("wq")).unwrap()),
            "q{bits}: packed wq diverged from mlx 0.31"
        );
        assert!(
            close(&scales, w.require(&p("scales")).unwrap(), 1e-4, 1e-5),
            "q{bits}: scales diverged"
        );
        assert!(
            close(&biases, w.require(&p("biases")).unwrap(), 1e-4, 1e-5),
            "q{bits}: biases diverged"
        );

        // Semantic parity: dequantized weight and quantized matmul.
        let deq = dequantize(&wq, &scales, &biases, GROUP_SIZE, bits).unwrap();
        assert!(
            close(&deq, w.require(&p("deq")).unwrap(), 1e-4, 1e-5),
            "q{bits}: dequant diverged"
        );

        let x = w.require("x").unwrap();
        let qmm = quantized_matmul(x, &wq, &scales, &biases, true, GROUP_SIZE, bits).unwrap();
        assert!(
            close(&qmm, w.require(&p("qmm")).unwrap(), 1e-2, 1e-2),
            "q{bits}: quantized_matmul diverged"
        );
    }
}

#[test]
fn adaptable_linear_quantize_matches_reference() {
    let w = Weights::from_file(FIXTURE).unwrap();
    let weight = w.require("w").unwrap().clone();
    let x = w.require("x").unwrap();

    for bits in [8, 4] {
        let mut lin = AdaptableLinear::dense(weight.clone(), None);
        assert!(!lin.is_quantized());
        lin.quantize(bits, None).unwrap();
        assert!(lin.is_quantized());

        // Quantized AdaptableLinear.forward equals the fork's quantized_matmul reference.
        let out = lin.forward(x).unwrap();
        assert!(
            close(
                &out,
                w.require(&format!("q{bits}.qmm")).unwrap(),
                1e-2,
                1e-2
            ),
            "q{bits}: AdaptableLinear quantized forward diverged from reference"
        );
    }
}

#[test]
fn quantized_forward_approximates_dense() {
    // Peak-relative error vs dense: Q8 ≈ 0.4%, Q4 ≈ 6.5% on N(0,1) weights at group_size 64
    // (measured). Assert with headroom so the gate is meaningful but not brittle.
    let w = Weights::from_file(FIXTURE).unwrap();
    let weight = w.require("w").unwrap().clone();
    let x = w.require("x").unwrap();

    let dense_out = AdaptableLinear::dense(weight.clone(), None)
        .forward(x)
        .unwrap();
    let dense = dense_out.as_slice::<f32>();
    let peak = dense.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));

    for (bits, max_rel) in [(8, 0.02_f32), (4, 0.12_f32)] {
        let mut q = AdaptableLinear::dense(weight.clone(), None);
        q.quantize(bits, None).unwrap();
        let qout = q.forward(x).unwrap();
        let out = qout.as_slice::<f32>();
        let max_err = out
            .iter()
            .zip(dense)
            .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
        let rel = max_err / peak;
        assert!(
            rel < max_rel,
            "q{bits}: peak-relative error {rel:.4} exceeds {max_rel}"
        );
    }
}
