//! sc-3239 byte-parity validation for the native Wan2.2 I2V-A14B converter
//! ([`mlx_gen_wan::convert::convert_i2v_14b`]).
//!
//! ⚠ **No reference exists yet.** Unlike the TI2V-5B path, there is no golden `wan_2_2_i2v_14b` dir
//! (the on-disk one is a config-only stub) and the native source (~114 GB, fp32 dual experts) is not
//! cached. This test is the *ready* validation harness: point it at a native checkpoint via
//! `WAN_I2V_14B_CKPT` and a reference MLX dir via `WAN_I2V_14B_GOLDEN` (e.g. one produced once by the
//! Python `mlx_video.convert_wan` dual arm) and it asserts the Rust converter reproduces it
//! byte-for-byte: `low_noise_model.safetensors` + `high_noise_model.safetensors` + `t5_encoder` +
//! `vae` + `config.json`. The new logic (z16 VAE sanitizer, quant predicate, config round-trip) is
//! covered by the fast unit tests in `convert.rs`; the transformer sanitizer + pickle reader +
//! `quantize` are byte-proven by the TI2V-5B + LTX parity tests.
//!
//! Run with: `WAN_I2V_14B_CKPT=… WAN_I2V_14B_GOLDEN=… cargo test -p mlx-gen-wan
//!   --test convert_i2v_14b_parity -- --ignored --nocapture`

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::convert::convert_i2v_14b;
use mlx_rs::ops::array_eq;

fn env_dir(key: &str) -> PathBuf {
    PathBuf::from(std::env::var(key).unwrap_or_else(|_| {
        panic!("set {key} to the native checkpoint / reference dir to run this validation (no golden ships for I2V-14B)")
    }))
}

fn assert_component_parity(golden: &std::path::Path, produced: &std::path::Path, name: &str) {
    let g =
        Weights::from_file(golden.join(name)).unwrap_or_else(|e| panic!("load golden {name}: {e}"));
    let p = Weights::from_file(produced.join(name))
        .unwrap_or_else(|e| panic!("load produced {name}: {e}"));
    let gk: BTreeSet<&str> = g.keys().collect();
    let pk: BTreeSet<&str> = p.keys().collect();
    assert!(
        gk == pk,
        "{name}: keyset mismatch ({} vs {} keys)",
        gk.len(),
        pk.len()
    );
    let mut diffs = 0usize;
    for k in &gk {
        let (gt, pt) = (g.require(k).unwrap(), p.require(k).unwrap());
        if gt.shape() != pt.shape()
            || gt.dtype() != pt.dtype()
            || !array_eq(gt, pt, false).unwrap().item::<bool>()
        {
            eprintln!("  {name}/{k}: differs");
            diffs += 1;
        }
    }
    assert_eq!(diffs, 0, "{name}: {diffs} of {} tensors differ", gk.len());
    eprintln!("  ✓ {name}: {} tensors byte-identical", gk.len());
}

#[test]
#[ignore = "no I2V-14B reference exists (no golden + ~114 GB uncached source); set WAN_I2V_14B_{CKPT,GOLDEN}"]
fn i2v_14b_convert_matches_reference() {
    let ckpt = env_dir("WAN_I2V_14B_CKPT");
    let golden = env_dir("WAN_I2V_14B_GOLDEN");
    assert!(ckpt.is_dir(), "checkpoint dir missing: {}", ckpt.display());
    assert!(
        golden.is_dir(),
        "reference dir missing: {}",
        golden.display()
    );

    let out = std::env::temp_dir().join("mlx_gen_wan_i2v_14b_parity_out");
    let _ = std::fs::remove_dir_all(&out);

    // Honour a reference quant geometry if the golden carries one.
    let quant = golden
        .join("config.json")
        .exists()
        .then(|| {
            let v: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(golden.join("config.json")).unwrap())
                    .unwrap();
            v.get("quantization").and_then(|q| {
                Some((
                    q.get("bits")?.as_i64()? as i32,
                    q.get("group_size")?.as_i64()? as i32,
                ))
            })
        })
        .flatten();

    convert_i2v_14b(&ckpt, &out, quant).unwrap();

    for name in [
        "low_noise_model.safetensors",
        "high_noise_model.safetensors",
        "t5_encoder.safetensors",
        "vae.safetensors",
    ] {
        assert_component_parity(&golden, &out, name);
    }
    let parse = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap()
    };
    assert_eq!(
        parse(golden.join("config.json")),
        parse(out.join("config.json")),
        "config.json semantic mismatch"
    );
    eprintln!("\nALL Wan I2V-A14B components byte-identical to reference ✓");
}
