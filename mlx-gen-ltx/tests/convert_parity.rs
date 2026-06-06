//! sc-3233 byte-parity validation for the native LTX-2.3 converter
//! ([`mlx_gen_ltx::convert_and_assemble`]), generic over the LTX-2.3 family.
//!
//! `#[ignore]`d: converts a real single-file checkpoint in-process and asserts the produced split
//! components reproduce the corresponding golden **byte-for-byte** — every component's keyset, and
//! every tensor's shape, dtype, and exact bytes (bf16 bit-identical; the quantized transformer's
//! u32-packed weights + bf16 scales/biases exact); configs compared semantically. Covered:
//!   * the base `Lightricks/LTX-2.3` distilled checkpoint at Q4 **and** Q8 (`ltx_2_3_base_q4`/`q8`),
//!   * a community fine-tune at Q4 (`TenStrip/LTX2.3-10Eros` → `ltx_2_3_eros`).
//!
//! Run with: `cargo test -p mlx-gen-ltx --test convert_parity -- --ignored --nocapture`
//! Path overrides: `LTX_BASE_SRC`, `LTX_EROS_DIR` / `LTX_EROS_SRC`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_ltx::convert::{convert_and_assemble, LtxConvertOpts};
use mlx_rs::ops::array_eq;

/// The eros golden split dir (`LTX_EROS_DIR` or the default SceneWorks data path).
fn eros_golden_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LTX_EROS_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home).join("Library/Application Support/SceneWorks/data/models/mlx/ltx_2_3_eros")
}

/// The eros single-file source (`LTX_EROS_SRC` or the HF cache snapshot).
fn eros_source_file() -> PathBuf {
    if let Ok(p) = std::env::var("LTX_EROS_SRC") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--TenStrip--LTX2.3-10Eros/snapshots");
    // pick the first snapshot dir containing the bf16 file
    let snap = std::fs::read_dir(&snapshots)
        .unwrap_or_else(|_| panic!("no HF snapshots at {}", snapshots.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("10Eros_v1_bf16.safetensors").is_file())
        .unwrap_or_else(|| {
            panic!(
                "10Eros_v1_bf16.safetensors not found under {}",
                snapshots.display()
            )
        });
    snap.join("10Eros_v1_bf16.safetensors")
}

/// Assert produced `<dir>/<name>.safetensors` reproduces the golden byte-for-byte.
fn assert_component_parity(golden: &std::path::Path, produced: &std::path::Path, name: &str) {
    let g = Weights::from_file(golden.join(format!("{name}.safetensors")))
        .unwrap_or_else(|e| panic!("load golden {name}: {e}"));
    let p = Weights::from_file(produced.join(format!("{name}.safetensors")))
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
            continue;
        }
        if gt.dtype() != pt.dtype() {
            eprintln!("  {name}/{k}: dtype {:?} != {:?}", gt.dtype(), pt.dtype());
            diffs += 1;
            continue;
        }
        if !array_eq(gt, pt, false).unwrap().item::<bool>() {
            eprintln!(
                "  {name}/{k}: bytes differ (dtype {:?}, shape {:?})",
                gt.dtype(),
                gt.shape()
            );
            diffs += 1;
        }
    }
    assert_eq!(
        diffs,
        0,
        "{name}: {diffs} tensor(s) differ from golden (of {})",
        gk.len()
    );
    eprintln!("  ✓ {name}: {} tensors byte-identical to golden", gk.len());
}

/// Parse two JSON files and assert they are semantically equal (key order is irrelevant).
fn assert_json_eq(golden: &std::path::Path, produced: &std::path::Path, name: &str) {
    let parse = |p: PathBuf| -> serde_json::Value {
        serde_json::from_str(
            &std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display())),
        )
        .unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
    };
    let g = parse(golden.join(name));
    let p = parse(produced.join(name));
    assert_eq!(g, p, "{name}: semantic mismatch vs golden");
    eprintln!("  ✓ {name}: semantically equal to golden");
}

#[test]
#[ignore = "needs 10Eros_v1_bf16.safetensors (~46 GB) + golden ltx_2_3_eros (~42 GB)"]
fn eros_q4_convert_matches_golden() {
    let golden = eros_golden_dir();
    let source = eros_source_file();
    assert!(golden.is_dir(), "golden dir missing: {}", golden.display());
    assert!(
        source.is_file(),
        "source file missing: {}",
        source.display()
    );

    let out = std::env::temp_dir().join("mlx_gen_ltx_convert_parity_out");
    let _ = std::fs::remove_dir_all(&out);
    eprintln!("converting {} → {}", source.display(), out.display());

    // No upscaler dir: validate the six core components + configs (the upsampler components are raw
    // copies of an external file, exercised separately below).
    convert_and_assemble(
        &source,
        None::<&std::path::Path>,
        &out,
        &LtxConvertOpts::audio_quant(4),
    )
    .unwrap();

    for name in [
        "transformer",
        "connector",
        "vae_decoder",
        "vae_encoder",
        "audio_vae",
        "vocoder",
    ] {
        assert_component_parity(&golden, &out, name);
    }

    assert_json_eq(&golden, &out, "config.json");
    assert_json_eq(&golden, &out, "embedded_config.json");
    assert_json_eq(&golden, &out, "quantize_config.json");

    eprintln!("\nALL components byte-identical to golden ltx_2_3_eros ✓");
}

/// The upsampler components are raw re-saves (no transform): feeding the golden's own
/// `spatial_upscaler_x2_v1_1.safetensors` back through the converter must reproduce both
/// `upsampler.safetensors` and `spatial_upscaler_x2_v1_1.safetensors` byte-for-byte.
#[test]
#[ignore = "needs golden ltx_2_3_eros upscaler component (~1 GB)"]
fn eros_upscaler_roundtrip_matches_golden() {
    let golden = eros_golden_dir();
    assert!(golden.is_dir(), "golden dir missing: {}", golden.display());

    // Stage a fake upscaler dir holding the golden's x2-1.1 component under the source filename.
    let updir = std::env::temp_dir().join("mlx_gen_ltx_upscaler_src");
    let _ = std::fs::remove_dir_all(&updir);
    std::fs::create_dir_all(&updir).unwrap();
    std::fs::copy(
        golden.join("spatial_upscaler_x2_v1_1.safetensors"),
        updir.join("ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
    )
    .unwrap();

    // A tiny source file so the converter has a (trivial) transformer to emit; we only check the
    // upscaler components here.
    let src = std::env::temp_dir().join("mlx_gen_ltx_tiny_src.safetensors");
    let a = mlx_rs::Array::ones::<f32>(&[2, 2]).unwrap();
    mlx_rs::Array::save_safetensors(
        vec![("model.diffusion_model.proj_out.weight", &a)],
        None::<&std::collections::HashMap<String, String>>,
        &src,
    )
    .unwrap();

    let out = std::env::temp_dir().join("mlx_gen_ltx_upscaler_out");
    let _ = std::fs::remove_dir_all(&out);
    let opts = LtxConvertOpts {
        include_audio: false,
        quantize: false,
        bits: 4,
        group_size: 64,
    };
    convert_and_assemble(&src, Some(&updir), &out, &opts).unwrap();

    for name in ["upsampler", "spatial_upscaler_x2_v1_1"] {
        assert_component_parity(&golden, &out, name);
    }
}

/// The base `Lightricks/LTX-2.3` distilled checkpoint (`LTX_BASE_SRC` or the HF cache).
fn base_source_file() -> PathBuf {
    if let Ok(p) = std::env::var("LTX_BASE_SRC") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots");
    std::fs::read_dir(&snapshots)
        .unwrap_or_else(|_| panic!("no HF snapshots at {}", snapshots.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.join("ltx-2.3-22b-distilled.safetensors").is_file())
        .unwrap_or_else(|| {
            panic!(
                "ltx-2.3-22b-distilled.safetensors not found under {}",
                snapshots.display()
            )
        })
        .join("ltx-2.3-22b-distilled.safetensors")
}

/// Convert the base distilled checkpoint at `bits` and assert the six core components + configs match
/// the golden `<id>` (the upsampler components are raw copies, exercised by the eros roundtrip test).
fn run_base_parity(golden_id: &str, bits: i32) {
    let home = std::env::var("HOME").unwrap();
    let golden = PathBuf::from(&home).join(format!(
        "Library/Application Support/SceneWorks/data/models/mlx/{golden_id}"
    ));
    let source = base_source_file();
    assert!(golden.is_dir(), "golden dir missing: {}", golden.display());
    assert!(source.is_file(), "source missing: {}", source.display());

    let out = std::env::temp_dir().join(format!("mlx_gen_ltx_{golden_id}_out"));
    let _ = std::fs::remove_dir_all(&out);
    eprintln!(
        "converting {} (Q{bits}) → {}",
        source.display(),
        out.display()
    );

    convert_and_assemble(
        &source,
        None::<&std::path::Path>,
        &out,
        &LtxConvertOpts::audio_quant(bits),
    )
    .unwrap();

    for name in [
        "transformer",
        "connector",
        "vae_decoder",
        "vae_encoder",
        "audio_vae",
        "vocoder",
    ] {
        assert_component_parity(&golden, &out, name);
    }
    assert_json_eq(&golden, &out, "config.json");
    assert_json_eq(&golden, &out, "embedded_config.json");
    assert_json_eq(&golden, &out, "quantize_config.json");
    eprintln!("\nALL base components byte-identical to golden {golden_id} ✓");
}

/// Base LTX-2.3 distilled, **Q4** — byte-parity vs golden `ltx_2_3_base_q4`. Confirms the converter is
/// generic over the LTX-2.3 family (base + eros share the single-file format), not eros-specific.
#[test]
#[ignore = "needs ltx-2.3-22b-distilled.safetensors (~46 GB) + golden ltx_2_3_base_q4"]
fn base_q4_convert_matches_golden() {
    run_base_parity("ltx_2_3_base_q4", 4);
}

/// Base LTX-2.3 distilled, **Q8** — byte-parity vs golden `ltx_2_3_base_q8`. Also the first end-to-end
/// validation of Q8 transformer quantization (eros is Q4-only).
#[test]
#[ignore = "needs ltx-2.3-22b-distilled.safetensors (~46 GB) + golden ltx_2_3_base_q8"]
fn base_q8_convert_matches_golden() {
    run_base_parity("ltx_2_3_base_q8", 8);
}
