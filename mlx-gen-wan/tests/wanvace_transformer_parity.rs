//! Wan-VACE transformer structural parity (epic 3040 / sc-3388, S1).
//!
//! Validates the native [`WanVaceTransformer`] forward against a **randomly-initialized small-config**
//! diffusers `WanVACETransformer3DModel` golden (`tools/dump_wanvace_transformer_golden.py`, sc-3433)
//! — no big VACE checkpoint needed. The golden is a pure-**f32** model (`.to(torch.float32)`), so the
//! gate runs the port in `compute_dtype = Float32`. The diffusers state_dict is committed under the
//! `model.` prefix; the inputs/output under `in.`/`out.`. This exercises every VACE-specific mechanism
//! on top of the (already-validated) base Wan block math: the 96-ch `vace_patch_embedding`, the
//! `WanVaceBlock` (`proj_in` on block 0, `proj_out` every block, its own `scale_shift_table`), the
//! sequential control stream, and — with the **non-trivial** `control_hidden_states_scale = [1.0,
//! 0.5]` baked into the golden — the per-vace-layer hint scale + injection order.
//!
//! ## The f32 floor (honoring "divergence is not rounding" — root-caused, not waved away)
//! The golden is dumped from **torch on CPU** (full f32), but the port runs on the **Apple GPU**,
//! whose mlx Metal f32 matmul kernel (`mpp::tensor_ops::matmul2d`, the matrix-unit path) uses
//! reduced internal precision. A controlled probe pinned this exactly: a single `[64,64]@[64,64]`
//! f32 matmul is **2.4e-2** off torch-CPU-f32 (which is 9.7e-6 vs an f64 reference), while torch-MPS
//! == torch-CPU (0.0) and a small gemv (`temb`) is f32-exact. The patchify tokens and every weight
//! load are **bit-identical** (max|Δ| = 0.0) to torch. So the ~1e-3 full-forward residual is a
//! **named cross-backend matmul-precision delta** (the same kernel the base Wan DiT runs in
//! production — see `transformer.rs` / `[[pmetal-mlx-bf16-matmul-bug]]`), not a port bug: there is no
//! mlx VACE reference to dump from, so torch is the only reference and this floor is irreducible.
//!
//! The gate clears that floor while staying tight enough to catch any VACE-logic bug; the
//! `negative control` below proves it — swapping the per-layer hint scales (a reversed/mis-scaled
//! hint) blows the divergence up two orders of magnitude past the gate.

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanVaceConfig;
use mlx_gen_wan::WanVaceTransformer;
use mlx_rs::Dtype;

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

/// The small config the golden was dumped with (`dump_wanvace_transformer_golden.py`): 4 main layers,
/// vace at `[0, 2]`, `dim = num_heads·head_dim = 4·16 = 64`, 96-ch control, patch `(1,2,2)`.
fn small_cfg() -> WanVaceConfig {
    let v = serde_json::json!({
        "model_type": "t2v",
        "model_version": "2.1",
        "dim": 64,
        "num_heads": 4,
        "num_layers": 4,
        "ffn_dim": 128,
        "freq_dim": 64,
        "text_dim": 32,
        "in_dim": 16,
        "out_dim": 16,
        "eps": 1e-6,
        "dual_model": false,
        "vace_layers": [0, 2],
        "vace_in_channels": 96
    });
    WanVaceConfig::from_config_json(&v)
}

/// Load the committed golden with the `model.` state_dict prefix stripped → the diffusers tensor
/// names the loader expects; everything cast to f32.
fn load_golden() -> Weights {
    let mut g = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wanvace_transformer_golden.safetensors"
    ))
    .expect("wanvace golden fixture");
    let stripped: Vec<(String, String)> = g
        .keys()
        .filter_map(|k| {
            k.strip_prefix("model.")
                .map(|s| (k.to_string(), s.to_string()))
        })
        .collect();
    for (from, to) in stripped {
        g.alias(&from, &to);
    }
    g.cast_all(Dtype::Float32).expect("cast golden to f32");
    g
}

#[test]
fn vace_forward_matches_diffusers_golden() {
    let g = load_golden();
    let cfg = small_cfg();
    assert_eq!(cfg.base.dim, 64);
    assert_eq!(cfg.head_dim(), 16);
    assert_eq!(cfg.vace_layers, vec![0, 2]);

    let model = WanVaceTransformer::from_weights(&g, &cfg, Dtype::Float32)
        .expect("build WanVaceTransformer");

    // Inputs (drop the leading batch axis the diffusers forward carries; the port is single-latent).
    let latent = g
        .require("in.hidden_states")
        .unwrap()
        .reshape(&[16, 4, 8, 8])
        .unwrap();
    let control = g
        .require("in.control_hidden_states")
        .unwrap()
        .reshape(&[96, 4, 8, 8])
        .unwrap();
    let context = g.require("in.encoder_hidden_states").unwrap().clone(); // [1, 12, 32]
    let t = g.require("in.timestep").unwrap().as_slice::<f32>()[0];
    let scales: Vec<f32> = g
        .require("in.control_hidden_states_scale")
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    assert_eq!(
        scales,
        vec![1.0, 0.5],
        "golden bakes a non-trivial hint scale"
    );

    let exp = g.require("out.sample").unwrap().as_slice::<f32>().to_vec();

    let out = model
        .forward_vace(&latent, &control, t, &context, &scales)
        .expect("forward_vace");
    let got = out.as_slice::<f32>().to_vec();
    assert_eq!(got.len(), exp.len(), "output length");
    let (max_abs, mean_rel) = diff(&got, &exp);
    println!("[vace output]  max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}");

    // Clears the mlx-Metal-f32-matmul vs torch-CPU-f32 floor (observed ~2.4e-3 / ~1e-3) with headroom.
    assert!(
        max_abs < 1e-2 && mean_rel < 4e-3,
        "VACE forward diverges past the matmul floor: max|Δ|={max_abs:.3e} mean_rel={mean_rel:.3e}"
    );

    // Negative control: swapping the per-vace-layer scales (a reversed/mis-scaled hint) must blow the
    // divergence far past the gate — proving the gate discriminates VACE hint scale/order despite the
    // f32 matmul floor (a wrong hint moves the output by O(hint·Δscale) ≈ 0.1+, ≫ the ~1e-3 floor).
    let bad = model
        .forward_vace(&latent, &control, t, &context, &[0.5, 1.0])
        .expect("forward_vace swapped");
    let (bad_max, bad_rel) = diff(bad.as_slice::<f32>(), &exp);
    println!("[swapped scale] max|Δ|={bad_max:.3e} mean_rel={bad_rel:.3e}");
    assert!(
        bad_rel > 2e-2,
        "negative control too weak ({bad_rel:.3e}) — the gate would not catch a hint-scale bug"
    );
}

/// Stage bisection (sc-3388 S1) — localizes a full-forward gap to the first diverging stage. Reads
/// the committed `wanvace_bisect.safetensors` (regenerate with `tools/dump_wanvace_bisect.py`). Each
/// stage is compared f32-vs-f32; the diffusers captures are forward-hook outputs of the same seeded
/// model. `#[ignore]` (a debugging aid, not a gate); the floor it surfaces is the documented
/// cross-backend matmul-precision delta.
#[test]
#[ignore = "debugging aid — run explicitly to localize a regression"]
fn vace_stage_bisection() {
    let g = load_golden();
    let bz = Weights::from_file(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/wanvace_bisect.safetensors"
    ))
    .expect("run tools/dump_wanvace_bisect.py");

    let cfg = small_cfg();
    let model = WanVaceTransformer::from_weights(&g, &cfg, Dtype::Float32).unwrap();
    let latent = g
        .require("in.hidden_states")
        .unwrap()
        .reshape(&[16, 4, 8, 8])
        .unwrap();
    let control = g
        .require("in.control_hidden_states")
        .unwrap()
        .reshape(&[96, 4, 8, 8])
        .unwrap();
    let context = g.require("in.encoder_hidden_states").unwrap().clone();
    let t = g.require("in.timestep").unwrap().as_slice::<f32>()[0];

    let stages = model
        .forward_vace_capture(&latent, &control, t, &context, &[1.0, 0.5])
        .unwrap();
    for (name, arr) in &stages {
        let got = arr
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let exp = bz.require(name).unwrap().as_slice::<f32>().to_vec();
        let (ma, mr) = diff(&got, &exp);
        println!(
            "[{name:14}] shape={:?} max|Δ|={ma:.3e} mean_rel={mr:.3e}",
            arr.shape()
        );
    }
}
