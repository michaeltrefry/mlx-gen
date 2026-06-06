//! sc-3238 byte-parity validation for the native Wan2.2 TI2V-5B converter
//! ([`mlx_gen_wan::convert::convert_ti2v_5b`]).
//!
//! `#[ignore]`d + heavy: needs the native `Wan-AI/Wan2.2-TI2V-5B` checkpoint (3 f32 transformer
//! shards ~20 GB + `models_t5_umt5-xxl-enc-bf16.pth` ~11 GB + `Wan2.2_VAE.pth` ~2.8 GB) and the
//! golden `wan_2_2_ti2v_5b` dir. Runs the full converter in-process and asserts `model.safetensors`
//! (825 bf16), `t5_encoder.safetensors` (242 bf16), and `vae.safetensors` (196 f32) reproduce the
//! golden byte-for-byte, with `config.json` semantically equal. Peak RSS ~30 GB (the f32 transformer).
//!
//! Run with: `cargo test -p mlx-gen-wan --test convert_5b_parity -- --ignored --nocapture`
//! Override paths with `WAN_TI2V_5B_DIR` (golden) / `WAN_5B_CKPT` (native checkpoint dir).

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::convert::convert_ti2v_5b;
use mlx_rs::ops::array_eq;

fn golden_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_TI2V_5B_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home)
        .join("Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b")
}

fn checkpoint_dir() -> PathBuf {
    if let Ok(d) = std::env::var("WAN_5B_CKPT") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap();
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Wan-AI--Wan2.2-TI2V-5B/snapshots");
    std::fs::read_dir(&snapshots)
        .unwrap_or_else(|_| panic!("no HF snapshots at {}", snapshots.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("models_t5_umt5-xxl-enc-bf16.pth").is_file())
        .unwrap_or_else(|| {
            panic!(
                "native TI2V-5B checkpoint not found under {}",
                snapshots.display()
            )
        })
}

fn assert_component_parity(golden: &std::path::Path, produced: &std::path::Path, name: &str) {
    let g =
        Weights::from_file(golden.join(name)).unwrap_or_else(|e| panic!("load golden {name}: {e}"));
    let p = Weights::from_file(produced.join(name))
        .unwrap_or_else(|e| panic!("load produced {name}: {e}"));
    let gk: BTreeSet<&str> = g.keys().collect();
    let pk: BTreeSet<&str> = p.keys().collect();
    let missing: Vec<&&str> = gk.difference(&pk).collect();
    let extra: Vec<&&str> = pk.difference(&gk).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "{name}: keyset mismatch — {} missing {:?}, {} extra {:?}",
        missing.len(),
        &missing[..missing.len().min(8)],
        extra.len(),
        &extra[..extra.len().min(8)],
    );
    let mut diffs = 0usize;
    for k in &gk {
        let (gt, pt) = (g.require(k).unwrap(), p.require(k).unwrap());
        if gt.shape() != pt.shape() {
            eprintln!("  {name}/{k}: shape {:?} != {:?}", gt.shape(), pt.shape());
            diffs += 1;
        } else if gt.dtype() != pt.dtype() {
            eprintln!("  {name}/{k}: dtype {:?} != {:?}", gt.dtype(), pt.dtype());
            diffs += 1;
        } else if !array_eq(gt, pt, false).unwrap().item::<bool>() {
            eprintln!("  {name}/{k}: bytes differ (dtype {:?})", gt.dtype());
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "{name}: {diffs} of {} tensors differ", gk.len());
    eprintln!("  ✓ {name}: {} tensors byte-identical to golden", gk.len());
}

#[test]
#[ignore = "needs native Wan2.2-TI2V-5B checkpoint (~34 GB) + golden wan_2_2_ti2v_5b"]
fn ti2v_5b_convert_matches_golden() {
    let golden = golden_dir();
    let ckpt = checkpoint_dir();
    assert!(golden.is_dir(), "golden dir missing: {}", golden.display());
    assert!(ckpt.is_dir(), "checkpoint dir missing: {}", ckpt.display());

    let out = std::env::temp_dir().join("mlx_gen_wan_5b_parity_out");
    let _ = std::fs::remove_dir_all(&out);
    eprintln!("converting {} → {}", ckpt.display(), out.display());

    convert_ti2v_5b(&ckpt, &out).unwrap();

    for name in [
        "model.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
    ] {
        assert_component_parity(&golden, &out, name);
    }

    // config.json semantic equality.
    let parse = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
    };
    assert_eq!(
        parse(golden.join("config.json")),
        parse(out.join("config.json")),
        "config.json semantic mismatch"
    );
    eprintln!("  ✓ config.json: semantically equal to golden");

    eprintln!("\nALL Wan TI2V-5B components byte-identical to golden ✓");
}
