//! sc-2602: Z-Image adapter key→module routing for the targets whose trained-file (diffusers)
//! naming differs from the crate's internal field names. The transformer-block hosts are covered
//! by `z_image_block.rs`; this locks the non-obvious global translations — `t_embedder.mlp.{0,2}`
//! → `linear{1,2}` and the final layer's `adaLN_modulation.1` (Sequential index 1, vs index 0 for
//! blocks). Synthetic temp fixtures (no real weights), so this runs in CI.

use std::collections::HashMap;
use std::path::PathBuf;

use mlx_gen::adapters::{install_adapter, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen_z_image::{FinalLayer, TimestepEmbedder};
use mlx_rs::Array;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join("mlx_gen_z_image_routing_test");
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A dotted-path install that resolves to no module is rejected; one that resolves succeeds. We
/// install a trivial scale-0 LoRA (install only checks the path resolves, not shapes).
fn dummy() -> Adapter {
    Adapter::Lora {
        a: Array::from_slice(&[0.0f32], &[1, 1]),
        b: Array::from_slice(&[0.0f32], &[1, 1]),
        scale: 0.0,
    }
}

fn write(path: &PathBuf, arrays: Vec<(&str, &Array)>) {
    Array::save_safetensors(arrays, None as Option<&HashMap<String, String>>, path).unwrap();
}

#[test]
fn timestep_embedder_routes_mlp_indices() {
    // linear1: [out=8, in=freq=8], linear2: [out=8, in=8]; bias [out].
    let w8 = Array::from_slice(&vec![0.1f32; 64], &[8, 8]);
    let b8 = Array::from_slice(&[0.0f32; 8], &[8]);
    let path = tmp("t_embedder.safetensors");
    write(
        &path,
        vec![
            ("t.linear1.weight", &w8),
            ("t.linear1.bias", &b8),
            ("t.linear2.weight", &w8),
            ("t.linear2.bias", &b8),
        ],
    );
    let w = Weights::from_file(&path).unwrap();
    let mut te = TimestepEmbedder::from_weights(&w, "t", 8).unwrap();

    // Trained-file naming: mlp.0 → linear1, mlp.2 → linear2.
    assert!(install_adapter(&mut te, "mlp.0", dummy()).is_ok());
    assert!(install_adapter(&mut te, "mlp.2", dummy()).is_ok());
    // The SiLU slot and the internal field names must NOT resolve.
    assert!(install_adapter(&mut te, "mlp.1", dummy()).is_err());
    assert!(install_adapter(&mut te, "linear1", dummy()).is_err());
}

#[test]
fn final_layer_routes_adaln_index_one() {
    // linear: [16, 8], adaLN_modulation.0 (the Linear): [16, 8]; bias [16].
    let w = Array::from_slice(&vec![0.1f32; 128], &[16, 8]);
    let b = Array::from_slice(&[0.0f32; 16], &[16]);
    let path = tmp("final_layer.safetensors");
    write(
        &path,
        vec![
            ("f.linear.weight", &w),
            ("f.linear.bias", &b),
            ("f.adaLN_modulation.0.weight", &w),
            ("f.adaLN_modulation.0.bias", &b),
        ],
    );
    let wts = Weights::from_file(&path).unwrap();
    let mut fl = FinalLayer::from_weights(&wts, "f").unwrap();

    // Trained-file naming: final layer's adaLN Linear is Sequential index 1 (SiLU at 0), unlike
    // the transformer blocks whose adaLN file key is index 0.
    assert!(install_adapter(&mut fl, "linear", dummy()).is_ok());
    assert!(install_adapter(&mut fl, "adaLN_modulation.1", dummy()).is_ok());
    // The block convention (index 0) must NOT resolve on the final layer.
    assert!(install_adapter(&mut fl, "adaLN_modulation.0", dummy()).is_err());
    assert!(install_adapter(&mut fl, "ada", dummy()).is_err());
}
