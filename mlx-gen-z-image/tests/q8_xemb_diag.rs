//! **RESOLVED — not an open numeric question (F-044).** The sc-2349/2532/2604 Q8 investigation is
//! closed (re-validated on MLX 0.31.2 in sc-2782); the live Q8 regression is covered by the e2e Q8
//! gate. This `#[ignore]`d diagnostic is kept only as a localization probe — it pinpoints which op a
//! residual lives in *if* that gate ever reddens. It needs the original investigation's probe golden
//! (`tools/golden/q8_xemb_probe.safetensors`, from `probe_q8_xemb_loaded.py`), so it never runs in CI.
//!
//! sc-2604 Q8 root-cause diagnostic: localize the base z_image Q8 per-op residual by comparing the
//! *loaded* Rust x-embedder against the fork's loaded x-embedder (`tools/probe_q8_xemb_loaded.py`),
//! component by component. The fork side proved (within one build): loaded wq == fresh mx.quantize,
//! in-model forward == bare qmm+bias, striding irrelevant. So any Rust-vs-fork gap here is the
//! source-built MLX vs the fork wheel, isolated to ONE of:
//!   (1) the loaded quantization bytes (does `try_from_linear` == `mx.quantize`?),
//!   (2) the bare `quantized_matmul` kernel at the embedder shape (M=4096, K=64) with bias,
//!   (3) the `AdaptableLinear` in-model forward wrapper.
//!
//! Run: cargo test -p mlx-gen-z-image --release --test q8_xemb_diag -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen_z_image::load_transformer;
use mlx_rs::ops::{eq, quantize, quantized_matmul};
use mlx_rs::{Array, Dtype};

const PROBE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/q8_xemb_probe.safetensors"
);

mod common;
use common::{rel, snapshot};

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

fn f32(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

/// `(peak_rel, mean_rel)` vs golden `b`, both read as f32.
fn all_eq(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape() && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
}

#[test]
#[ignore = "needs real Z-Image weights + tools/golden/q8_xemb_probe.safetensors (probe_q8_xemb_loaded.py)"]
fn q8_xemb_loaded_vs_fork() {
    let g = Weights::from_file(PROBE).unwrap();
    let gs = g.metadata("group_size").unwrap().parse::<i32>().unwrap();
    let bits = g.metadata("bits").unwrap().parse::<i32>().unwrap();

    // Fork-dumped tensors.
    let x_tokens = f32(g.require("x_tokens").unwrap()); // [4096,64] f32, the embedder input
    let f_wq = g.require("xe_wq").unwrap().clone(); // uint32 packed
    let f_scales = bf16(g.require("xe_scales").unwrap()); // f32→bf16 exact (orig bf16)
    let f_biases = bf16(g.require("xe_biases").unwrap());
    let f_bias = bf16(g.require("xe_bias").unwrap());
    let f_inmodel = g.require("xe_inmodel").unwrap();
    let f_bare = g.require("xe_bare").unwrap();

    // Load + quantize the real base transformer (mirror the generate path).
    let mut t = load_transformer(&snapshot()).unwrap();
    t.quantize(bits).unwrap();
    let xe = t.x_embedder();
    let (r_wq, r_scales, r_biases, r_bias, r_gs, r_bits) =
        xe.quantized_params().expect("x_embedder is quantized");
    assert_eq!((r_gs, r_bits), (gs, bits), "group_size/bits");

    println!("=== (1) loaded quantization: Rust try_from_linear vs fork mx.quantize ===");
    let (wq_eq, sc_eq, bi_eq) = (
        all_eq(r_wq, &f_wq),
        all_eq(r_scales, &f_scales),
        all_eq(r_biases, &f_biases),
    );
    println!("  wq byte-identical        : {wq_eq}");
    println!("  scales byte-identical    : {sc_eq}");
    println!("  biases byte-identical    : {bi_eq}");
    println!(
        "  bias byte-identical      : {}",
        r_bias.map(|b| all_eq(b, &f_bias)).unwrap_or(false)
    );
    println!(
        "  scales peak/mean_rel     : {:?}",
        rel(r_scales, &f_scales)
    );
    println!(
        "  biases peak/mean_rel     : {:?}",
        rel(r_biases, &f_biases)
    );
    // REGRESSION GATE (sc-2604): the loaded model must quantize the bf16-cast weight, byte-identical
    // to the fork. If the loader ever stops downcasting f32 checkpoints to bf16 before quantizing,
    // the scales drift ~0.13% (wq/biases survive) → the base-Q8 e2e residual returns. Fails loudly.
    assert!(wq_eq, "loaded Q8 wq is not byte-identical to the fork");
    assert!(
        sc_eq,
        "loaded Q8 SCALES differ from the fork — the bf16-cast-before-quantize fix regressed \
         (sc-2604: f32 checkpoint quantized as f32 → divergent scales)"
    );
    assert!(bi_eq, "loaded Q8 biases differ from the fork");

    println!("\n=== (2) bare quantized_matmul KERNEL at embedder shape (M=4096,K=64) ===");
    // With the FORK's exact byte-identical wq/scales/biases/bias + the fork's x_tokens.
    let bare_fork_w =
        quantized_matmul(&x_tokens, &f_wq, &f_scales, &f_biases, true, gs, bits).unwrap();
    let bare_fork_w = mlx_rs::ops::add(&bare_fork_w, &f_bias).unwrap();
    println!(
        "  qmm(fork wq/scales/biases)+bias vs fork xe_bare   : peak/mean_rel={:?}",
        rel(&bare_fork_w, f_bare)
    );
    // With RUST's loaded wq/scales/biases/bias + the fork's x_tokens.
    let bare_rust_w =
        quantized_matmul(&x_tokens, r_wq, r_scales, r_biases, true, gs, bits).unwrap();
    let bare_rust_w = match r_bias {
        Some(b) => mlx_rs::ops::add(&bare_rust_w, b).unwrap(),
        None => bare_rust_w,
    };
    println!(
        "  qmm(rust loaded params)+bias    vs fork xe_bare   : peak/mean_rel={:?}",
        rel(&bare_rust_w, f_bare)
    );

    println!("\n=== (3) AdaptableLinear in-model forward ===");
    let r_inmodel = xe.forward(&x_tokens).unwrap();
    println!(
        "  Rust x_embedder.forward(fork x) vs fork xe_inmodel: peak/mean_rel={:?}",
        rel(&r_inmodel, f_inmodel)
    );
    println!(
        "  Rust x_embedder.forward(fork x) vs Rust bare      : peak/mean_rel={:?}",
        rel(&r_inmodel, &bare_rust_w)
    );

    println!("\n=== (4) FIX CHECK: quantize the bf16-cast weight (mirror the fork's load) ===");
    // Load a SECOND, un-quantized transformer to read the dense weight + its dtype.
    let dense_t = load_transformer(&snapshot()).unwrap();
    let (w_dense, _) = dense_t.x_embedder().dense_weight().expect("dense");
    println!(
        "  loaded dense x_embedder weight dtype: {:?}",
        w_dense.dtype()
    );
    // (a) quantize AS LOADED (f32) — should reproduce the divergent scales.
    let (_, sc_asis, _) = quantize(w_dense, gs, bits).unwrap();
    println!(
        "  quantize(loaded as-is) scales vs fork : {:?}",
        rel(&sc_asis, &f_scales)
    );
    // (b) quantize the bf16-cast weight — should MATCH the fork's bf16 scales.
    let w_bf16 = w_dense.as_dtype(Dtype::Bfloat16).unwrap();
    let (wq_bf16, sc_bf16, bi_bf16) = quantize(&w_bf16, gs, bits).unwrap();
    println!(
        "  quantize(bf16(loaded)) scales vs fork : {:?}",
        rel(&sc_bf16, &f_scales)
    );
    println!(
        "  quantize(bf16(loaded)) wq==fork       : {}",
        all_eq(&wq_bf16, &f_wq)
    );
    // (c) the bf16-quantized qmm vs the fork golden.
    let bare_bf16 = quantized_matmul(
        &x_tokens,
        &wq_bf16,
        bf16(&sc_bf16),
        &bf16(&bi_bf16),
        true,
        gs,
        bits,
    )
    .unwrap();
    let bare_bf16 = mlx_rs::ops::add(&bare_bf16, &f_bias).unwrap();
    println!(
        "  qmm(bf16-quantized)+bias vs fork xe_bare : peak/mean_rel={:?}",
        rel(&bare_bf16, f_bare)
    );
}
