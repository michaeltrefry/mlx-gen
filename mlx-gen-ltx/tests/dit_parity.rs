//! S3b full-DiT (velocity) parity vs the reference video-only `LTXModel` (sc-2679 S3b).
//!
//! `#[ignore]`d: needs the real `ltx_2_3_base_q8` `transformer.safetensors` (~20 GB). The committed
//! golden (`tests/fixtures/ltx_dit_golden.safetensors`, from `tools/dump_ltx_dit_golden.py`) holds
//! the reference **f32-activation × Q8** velocity over synthetic inputs; this test loads the SAME Q8
//! weights (kept quantized → `quantized_matmul`) and checks the Rust `LtxDiT` reproduces it.
//!
//! **The golden MUST be mlx 0.31.2** (the Rust build): `quantized_matmul` changed 0.31.0→0.31.2, so
//! a 0.31.0 golden mismatches by ~5e-4/op. At matched 0.31.2 the **full 48-layer velocity is
//! bit-exact** (peak_rel = mean_rel = 0.0). It was not until sc-2842: the adaLN timestep sinusoid was
//! tabulated on the host in f64 then cast to f32 (the reference `get_timestep_embedding` builds it in
//! MLX f32), a ~1e-7/elem seed that — fed into the f32 adaLN modulating every block — compounded over
//! the 48-layer residual to ~0.9% mean_rel. Building the table in MLX f32 makes it bit-exact. Honors
//! "divergence is not rounding": the residual was a real, named, fixed op, not f32 accumulation.
//!
//! Run: `LTX_BASE_DIR=… cargo test -p mlx-gen-ltx --test dit_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max as max_op, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_ltx::config::LtxConfig;
use mlx_gen_ltx::transformer::{LtxDiT, Precision};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_dit_golden.safetensors"
);
/// The reference's **native bf16+Q8** velocity golden (`LTX_BF16=1 dump_ltx_dit_golden.py`) — the
/// production-precision target for [`Precision::Bf16Q8`].
const GOLDEN_BF16: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ltx_dit_golden_bf16.safetensors"
);

fn base_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("LTX_BASE_DIR") {
        return d.into();
    }
    let home = std::env::var("HOME").unwrap();
    std::path::PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_base_q8")
}

fn f32(x: &Array) -> Array {
    x.as_dtype(Dtype::Float32).unwrap()
}

/// `max|Δ| / max|ref|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(f32(got), want).unwrap()).unwrap();
    let denom = max_op(abs(want).unwrap(), None).unwrap().item::<f32>();
    max_op(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// `Σ|Δ| / Σ|ref|` — robust to the output-LayerNorm-amplified massive-activation channels.
fn mean_rel(got: &Array, want: &Array) -> f32 {
    let num = sum(abs(subtract(f32(got), want).unwrap()).unwrap(), None).unwrap();
    let den = sum(abs(want).unwrap(), None).unwrap();
    num.item::<f32>() / den.item::<f32>().max(1e-12)
}

fn build_prec(prec: Precision, golden: &str) -> (LtxDiT, Weights) {
    let dir = base_dir();
    let cfg = LtxConfig::from_model_dir(&dir).expect("embedded_config.json");
    let w =
        Weights::from_file(dir.join("transformer.safetensors")).expect("transformer.safetensors");
    let dit = LtxDiT::from_weights(&w, &cfg, prec).expect("build LtxDiT");
    let g = Weights::from_file(golden).expect("golden (run tools/dump_ltx_dit_golden.py)");
    (dit, g)
}

fn build() -> (LtxDiT, Weights) {
    build_prec(Precision::F32Q8, GOLDEN)
}

#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn dit_velocity_matches_reference() {
    let (dit, g) = build();
    let got = dit
        .forward(
            g.require("latent").unwrap(),
            g.require("timestep").unwrap(),
            g.require("context").unwrap(),
            None,
            g.require("positions").unwrap(),
        )
        .expect("dit forward");
    let want = g.require("velocity").unwrap();
    assert_eq!(got.shape(), want.shape(), "velocity shape");
    let (pr, mr) = (peak_rel(&got, want), mean_rel(&got, want));
    eprintln!("dit velocity peak_rel = {pr:.3e} mean_rel = {mr:.3e}");
    // The per-forward DiT is bit-exact at matched mlx 0.31.2 (sc-2842 fixed the last seed, the
    // host-f64 timestep table). A non-zero residual here means a per-op divergence has crept back.
    assert!(
        pr == 0.0,
        "dit velocity peak_rel {pr:.3e} must be bit-exact"
    );
    assert!(
        mr == 0.0,
        "dit velocity mean_rel {mr:.3e} must be bit-exact"
    );
}

/// The reference's **native bf16+Q8** per-forward — the production-speed path ([`Precision::Bf16Q8`]).
/// Bit-exact at matched mlx 0.31.2 (the distilled stage-1 sampler is chaos-sensitive, so the bf16
/// per-forward must be as tight as the f32 one). The same sc-2842 timestep-table fix applies, plus
/// the `timestep × 1000` scaling must run in the **input (bf16) dtype** — `denoise_av` feeds a bf16
/// timestep, so a pre-upcast-to-f32 would round differently (`f32(σ)·1000` ≠ `bf16(σ·1000)`).
#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn dit_velocity_matches_reference_bf16() {
    let (dit, g) = build_prec(Precision::Bf16Q8, GOLDEN_BF16);
    let got = dit
        .forward(
            g.require("latent").unwrap(),
            g.require("timestep").unwrap(),
            g.require("context").unwrap(),
            None,
            g.require("positions").unwrap(),
        )
        .expect("dit forward");
    let want = g.require("velocity").unwrap();
    assert_eq!(got.shape(), want.shape(), "velocity shape");
    let (pr, mr) = (peak_rel(&got, want), mean_rel(&got, want));
    eprintln!("dit velocity (bf16) peak_rel = {pr:.3e} mean_rel = {mr:.3e}");
    assert!(
        pr == 0.0,
        "dit velocity (bf16) peak_rel {pr:.3e} must be bit-exact"
    );
    assert!(
        mr == 0.0,
        "dit velocity (bf16) mean_rel {mr:.3e} must be bit-exact"
    );
}

/// Sanity that the output head is exact: feed the reference post-block hidden through the Rust head
/// and compare the velocity — isolates the head from the 48-layer accumulation (was bit-exact at
/// bring-up).
#[test]
#[ignore = "needs ltx_2_3_base_q8 transformer.safetensors (~20 GB)"]
fn dit_output_head_exact() {
    let (dit, g) = build();
    let head = dit
        .output_head(
            g.require("tap_h").unwrap(),
            g.require("tap_emb_ts").unwrap(),
        )
        .expect("output_head");
    let pr = peak_rel(&head, g.require("velocity").unwrap());
    eprintln!("output_head(golden h) peak_rel = {pr:.3e}");
    assert!(pr < 5e-3, "output head peak_rel {pr:.3e} too high");
}
