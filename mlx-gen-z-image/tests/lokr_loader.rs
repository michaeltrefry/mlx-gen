//! sc-2343: end-to-end LoKr-loader parity vs the fork's `LoKrLoader`.
//!
//! Fixtures (`tools/dump_lokr_loader.py`): `lokr_loader.safetensors` = a small Z-Image block
//! (`w.*`) + inputs + the block output AFTER a synthetic LoKr was applied through the fork's
//! real loader; `lokr_adapter.safetensors` = that adapter in on-disk form (bare-path
//! `lokr_w1`/`w2` keys, `networkType=lokr` / `alpha` / `rank` metadata). The crate loads the
//! same adapter file and must reproduce the fork's post-adapter output (tol 1e-2 — Metal fp32).

use mlx_gen::adapters::loader::{apply_lokr, is_lokr};
use mlx_gen::weights::Weights;
use mlx_gen_z_image::{ZImageBlockConfig, ZImageTransformerBlock};
use mlx_rs::ops::all_close;

const BASE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lokr_loader.safetensors"
);
const ADAPTER: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lokr_adapter.safetensors"
);
const CFG: ZImageBlockConfig = ZImageBlockConfig {
    dim: 96,
    n_heads: 4,
    norm_eps: 1e-5,
};
const SCALE: f32 = 0.7;

#[test]
fn lokr_file_loads_and_matches_fork() {
    let base = Weights::from_file(BASE).unwrap();
    let adapter = Weights::from_file(ADAPTER).unwrap();
    assert!(
        is_lokr(&adapter),
        "fixture should be detected as a LoKr adapter"
    );

    let mut block = ZImageTransformerBlock::from_weights(&base, "w", CFG).unwrap();
    let report = apply_lokr(&mut block, &adapter, SCALE).unwrap();
    assert_eq!(
        report.applied, 2,
        "should adapt attention.to_q and feed_forward.w1"
    );
    assert!(
        report.unmatched_paths.is_empty(),
        "no adapter keys should miss the model"
    );

    let x = base.require("in.x").unwrap();
    let freqs_cis = base.require("in.freqs_cis").unwrap();
    let t_emb = base.require("in.t_emb").unwrap();
    let y = block.forward(x, freqs_cis, t_emb).unwrap();

    let want = base.require("out.y").unwrap();
    assert!(
        all_close(&y, want, 1e-2, 1e-2, false)
            .unwrap()
            .item::<bool>(),
        "LoKr-adapted output diverged from the fork"
    );
}
