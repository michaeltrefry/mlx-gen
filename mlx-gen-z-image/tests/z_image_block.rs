//! sc-2344: whole-block parity for the ported Z-Image transformer block against the Python
//! mflux fork, plus the path-addressed adapter install (sc-2343 seam).
//!
//! Fixture `tests/fixtures/zblock_small.safetensors` is produced by `tools/dump_zblock_small.py`
//! from the fork's real `ZImageTransformerBlock` at tiny dims (dim=96, heads=4, seq=4).
//! Tolerance 1e-2 matches the spike + mflux's own suite: MLX runs fp32 matmul in reduced
//! precision on Metal, so matmul chains agree to ~3–4 sig figs, not bit-exactly.

use mlx_gen::adapters::{install_adapter, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen_z_image::{ZImageBlockConfig, ZImageTransformerBlock};
use mlx_rs::ops::{all_close, array_eq};
use mlx_rs::Array;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/zblock_small.safetensors"
);
const CFG: ZImageBlockConfig = ZImageBlockConfig {
    dim: 96,
    n_heads: 4,
    norm_eps: 1e-5,
};

fn load() -> Weights {
    Weights::from_file(FIXTURE).expect("load zblock fixture")
}

fn inputs(w: &Weights) -> (Array, Array, Array) {
    (
        w.require("in.x").unwrap().clone(),
        w.require("in.freqs_cis").unwrap().clone(),
        w.require("in.t_emb").unwrap().clone(),
    )
}

#[test]
fn block_matches_python_reference() {
    let w = load();
    let block = ZImageTransformerBlock::from_weights(&w, "w", CFG).unwrap();
    let (x, freqs_cis, t_emb) = inputs(&w);

    let y = block.forward(&x, &freqs_cis, &t_emb).unwrap();
    let want = w.require("out.y").unwrap();

    assert_eq!(y.shape(), want.shape());
    let close = all_close(&y, want, 1e-2, 1e-2, false)
        .unwrap()
        .item::<bool>();
    assert!(
        close,
        "block output diverged from Python reference beyond 1e-2"
    );
}

#[test]
fn zero_scale_adapter_install_is_noop() {
    let w = load();
    let (x, freqs_cis, t_emb) = inputs(&w);

    let mut block = ZImageTransformerBlock::from_weights(&w, "w", CFG).unwrap();
    let base = block.forward(&x, &freqs_cis, &t_emb).unwrap();

    // Install a rank-4 LoRA on attention.to_q with scale 0 → residual is exactly zero.
    install_adapter(
        &mut block,
        "attention.to_q",
        Adapter::Lora {
            a: Array::from_slice(&vec![0.3f32; (CFG.dim * 4) as usize], &[CFG.dim, 4]),
            b: Array::from_slice(&vec![0.2f32; (4 * CFG.dim) as usize], &[4, CFG.dim]),
            scale: 0.0,
        },
    )
    .unwrap();

    let out = block.forward(&x, &freqs_cis, &t_emb).unwrap();
    assert!(array_eq(&out, &base, false).unwrap().item::<bool>());
}

#[test]
fn nonzero_adapter_install_changes_output() {
    let w = load();
    let (x, freqs_cis, t_emb) = inputs(&w);

    let mut block = ZImageTransformerBlock::from_weights(&w, "w", CFG).unwrap();
    let base = block.forward(&x, &freqs_cis, &t_emb).unwrap();

    install_adapter(
        &mut block,
        "feed_forward.w1",
        Adapter::Lora {
            a: Array::from_slice(&vec![0.3f32; (CFG.dim * 4) as usize], &[CFG.dim, 4]),
            // w1 maps dim -> hidden (int(dim/3*8) = 256 at dim=96).
            b: Array::from_slice(&vec![0.2f32; (4 * 256) as usize], &[4, 256]),
            scale: 0.5,
        },
    )
    .unwrap();

    let out = block.forward(&x, &freqs_cis, &t_emb).unwrap();
    assert!(!array_eq(&out, &base, false).unwrap().item::<bool>());
}

#[test]
fn install_on_unknown_path_errors() {
    let w = load();
    let mut block = ZImageTransformerBlock::from_weights(&w, "w", CFG).unwrap();
    let err = install_adapter(
        &mut block,
        "attention.no_such_proj",
        Adapter::Lora {
            a: Array::from_slice(&[0.0f32], &[1, 1]),
            b: Array::from_slice(&[0.0f32], &[1, 1]),
            scale: 1.0,
        },
    );
    assert!(err.is_err());
}
