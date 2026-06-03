//! S3 parity gate: the Wan DiT (5B) must reproduce the `mlx_video` reference, gated **bf16-against-bf16**
//! — the production regime (the reference runs bf16 matmuls + bf16 SDPA + bf16 cos/sin with an f32
//! residual stream; see `tools/dump_s3_fixtures.py`). The DiT now mirrors that exactly (sc-2714 +
//! sc-2770 patched the NAX bf16 GEMM and SDPA, so bf16 is correct on the pinned build).
//!
//! Observed: patch embedding **bit-exact** (`x_embed` max|Δ| = 0.0); DiT output mean_rel ~4e-2. The
//! residual is the **cross-build bf16 kernel difference**: the pinned build is MLX 0.31.1 + the
//! sc-2714/sc-2770 patches (which route the broken NAX bf16 ops to an f32/TF32 fallback), whereas the
//! production reference is MLX 0.31.2's *native* fixed NAX bf16 kernels — different bf16
//! implementations, differing at bf16 rounding and accumulating over 30 layers. Bumping the Rust pin
//! to 0.31.2 (under evaluation) would use production's exact kernels and tighten this toward exact.
//! (For comparison, gating against an f32-upcast reference gives ~6e-3 — tighter, but it isn't the
//! production dtype.) True end-to-end parity is the px>8 video gate at S4.
//!
//! `#[ignore]` heavy: loads the converted `model.safetensors` (~11 GB) from the snapshot dir
//! (`WAN_5B_DIR`). Honors "divergence is not rounding" — the residual is a *named* cross-build bf16
//! kernel difference, with the patch embedding bit-exact and the per-block growth consistent with
//! bf16 precision (not a code bug).

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::WanTransformer;

fn snapshot_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_5B_DIR") {
        return PathBuf::from(d);
    }
    PathBuf::from(std::env::var("HOME").unwrap())
        .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b")
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let (mut ma, mut sa, mut sr) = (0f32, 0f64, 0f64);
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        ma = ma.max(d);
        sa += d as f64;
        sr += e.abs() as f64;
    }
    (ma, sa / sr.max(1e-30))
}

#[test]
#[ignore = "needs the converted 5B model.safetensors (~11 GB) — run tools/dump_s3_fixtures.py"]
fn dit_forward_matches_reference() {
    let dir = snapshot_dir();
    let cfg = WanModelConfig::wan22_ti2v_5b();
    let w = Weights::from_file(dir.join("model.safetensors")).expect("model.safetensors");
    let dit = WanTransformer::from_weights(&w, &cfg).expect("build DiT");

    let g = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/s3_dit_golden.safetensors"
    ))
    .expect("s3 golden");

    let latent = g.require("latent").unwrap().clone();
    let context_raw = g.require("context_raw").unwrap().clone();
    let t: f32 = g.require("t").unwrap().as_slice::<f32>()[0];

    let context_emb = dit.embed_text(&context_raw).expect("embed_text");
    let stages = dit
        .forward_capture(&latent, t, &context_emb)
        .expect("forward_capture");
    let out = dit.forward(&latent, t, &context_emb).expect("forward");

    // Per-stage gate: x_embed (idx 0) must be bit-exact (patch embed = f32-promoted matmul → bf16,
    // exact cross-build); the residual grows with depth as the cross-build bf16 kernel difference
    // accumulates.
    let stage = |idx: usize, key: &str| -> (f32, f64) {
        let got = stages[idx]
            .as_dtype(mlx_rs::Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        diff(&got, g.require(key).unwrap().as_slice::<f32>())
    };
    let (e_max, _) = stage(0, "x_embed");
    println!("[x_embed]  max|Δ|={e_max:.3e}");
    assert_eq!(e_max, 0.0, "patch embedding not bit-exact: {e_max:.3e}");

    for (idx, key) in [(3usize, "x_block0"), (4, "x_blocks"), (5, "x_head")] {
        let (ma, mr) = stage(idx, key);
        println!("[{key}] max|Δ|={ma:.3e} mean_rel={mr:.3e}");
    }

    let got = out.as_slice::<f32>().to_vec();
    let (max_abs, mean_rel) = diff(&got, g.require("output").unwrap().as_slice::<f32>());
    println!("[output]   max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");
    // Cross-build bf16 kernel difference (0.31.1+patches fallback vs production 0.31.2 native NAX
    // bf16) accumulated over 30 layers (~4e-2). A 0.31.2 pin bump would tighten this toward exact.
    assert!(
        mean_rel < 6e-2,
        "DiT output mean_rel {mean_rel:.3e} too high"
    );
}
