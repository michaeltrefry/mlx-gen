//! Registry wiring + config-driven `load` for both Wan models.
//!
//! Verifies that `wan2_2_ti2v_5b` (dense 5B, `generate` still a WIP stub — sc-2680) and
//! `wan2_2_t2v_14b` (dual-expert MoE, `generate` fully wired) each self-register with the right
//! descriptor, that `load` reads the model's `config.json` (5B preset / dual-expert detection), that
//! the 14B loader rejects a non-dual config, and that the not-yet-wired sibling features (quant /
//! adapters / single-file source / precision override) are rejected for both.

use std::path::PathBuf;

use mlx_gen::{
    registry, AdapterKind, AdapterSpec, GenerationRequest, LoadSpec, Modality, Precision, Quant,
    WeightsSource,
};

use mlx_gen_wan::{MODEL_ID, MODEL_ID_T2V_14B};

/// The 5B's serialized `config.json` (the `convert_wan.py` schema; model_type ti2v + dim 3072).
const TI2V_5B_CONFIG: &str = r#"{
  "model_type": "ti2v",
  "model_version": "2.2",
  "patch_size": [1, 2, 2],
  "in_dim": 48,
  "dim": 3072,
  "ffn_dim": 14336,
  "out_dim": 48,
  "num_heads": 24,
  "num_layers": 30,
  "vae_z_dim": 48,
  "vae_stride": [4, 16, 16],
  "dual_model": false,
  "sample_shift": 5.0,
  "sample_steps": 40,
  "sample_guide_scale": 5.0,
  "sample_fps": 24,
  "max_area": 901120
}"#;

/// The A14B's serialized `config.json` (`convert_wan.py` schema; dual-expert T2V, dim 5120).
const T2V_14B_CONFIG: &str = r#"{
  "model_type": "t2v",
  "model_version": "2.2",
  "patch_size": [1, 2, 2],
  "in_dim": 16,
  "dim": 5120,
  "ffn_dim": 13824,
  "out_dim": 16,
  "num_heads": 40,
  "num_layers": 40,
  "vae_z_dim": 16,
  "vae_stride": [4, 8, 8],
  "dual_model": true,
  "boundary": 0.875,
  "sample_shift": 12.0,
  "sample_steps": 40,
  "sample_guide_scale": [3.0, 4.0],
  "sample_fps": 16,
  "frame_num": 81,
  "max_area": 0
}"#;

/// A throwaway model dir holding just `config.json` (`load` only reads config; `generate`'s heavy
/// weights aren't touched until called).
fn temp_model_dir(tag: &str) -> PathBuf {
    temp_model_dir_with(tag, TI2V_5B_CONFIG)
}

fn temp_model_dir_with(tag: &str, config: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("wan_s0_{}_{}", std::process::id(), tag));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), config).unwrap();
    dir
}

#[test]
fn wan_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID)
        .expect("wan2_2_ti2v_5b not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "wan2_2_ti2v_5b");
    assert_eq!(d.family, "wan");
    assert_eq!(d.modality, Modality::Video);
    // 5B uses real CFG + negative prompt, advertises a single image reference (TI2V), KV cache.
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_kv_cache);
    assert!(!d.capabilities.supports_lora);
    assert!(d.capabilities.samplers.contains(&"unipc"));
}

#[test]
fn load_reads_config_and_stubs_generate() {
    let dir = temp_model_dir("load");
    let g = registry::load(MODEL_ID, &LoadSpec::new(WeightsSource::Dir(dir.clone())))
        .expect("load should succeed (reads config.json)");
    assert_eq!(g.descriptor().id, MODEL_ID);

    // validate accepts a 32-aligned request with 1+4k frames; rejects sub-tile + bad frame counts.
    let ok = GenerationRequest {
        width: 704,
        height: 1280,
        frames: Some(81),
        ..Default::default()
    };
    assert!(g.validate(&ok).is_ok());
    let bad_size = GenerationRequest {
        width: 16,
        height: 1280,
        ..Default::default()
    };
    assert!(g.validate(&bad_size).is_err());
    let bad_frames = GenerationRequest {
        width: 704,
        height: 1280,
        frames: Some(80),
        ..Default::default()
    };
    assert!(g.validate(&bad_frames).is_err());

    // generate is an explicit WIP error until S1–S5.
    let mut noop = |_p| {};
    let err = g.generate(&ok, &mut noop).unwrap_err().to_string();
    assert!(err.contains("S1"), "expected WIP message, got: {err}");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_unwired_features() {
    let dir = temp_model_dir("reject");
    // Single-file source.
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::File(dir.join("config.json")))
    )
    .is_err());
    // Quantization (sc-2682).
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_err());
    // Precision override (the dense path runs f32 activations).
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    assert!(registry::load(MODEL_ID, &spec).is_err());
    // Adapters (sc-2683 / sc-2393).
    let adapters = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
    }];
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(adapters)
    )
    .is_err());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wan_t2v_14b_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID_T2V_14B)
        .expect("wan2_2_t2v_14b not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "wan2_2_t2v_14b");
    assert_eq!(d.family, "wan");
    assert_eq!(d.modality, Modality::Video);
    // Dual-expert CFG + negative prompt + KV cache; pure T2V (no image conditioning).
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_kv_cache);
    assert!(d.capabilities.conditioning.is_empty());
    assert!(!d.capabilities.supports_lora);
    assert!(d.capabilities.samplers.contains(&"unipc"));
    // H/W align to patch×vae_stride = 16 for the z16 VAE (vs 32 for the 5B's z48).
    assert_eq!(d.capabilities.min_size, 16);
}

#[test]
fn load_t2v_14b_reads_dual_config_and_wires_generate() {
    let dir = temp_model_dir_with("t2v14b", T2V_14B_CONFIG);
    let g = registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())),
    )
    .expect("load should succeed (reads dual config.json)");
    assert_eq!(g.descriptor().id, MODEL_ID_T2V_14B);

    // validate accepts a 16-aligned 1+4k-frame request; rejects sub-tile + bad frame counts.
    let ok = GenerationRequest {
        width: 512,
        height: 512,
        frames: Some(81),
        ..Default::default()
    };
    assert!(g.validate(&ok).is_ok());
    let bad_size = GenerationRequest {
        width: 8,
        height: 512,
        ..Default::default()
    };
    assert!(g.validate(&bad_size).is_err());
    let bad_frames = GenerationRequest {
        width: 512,
        height: 512,
        frames: Some(80),
        ..Default::default()
    };
    assert!(g.validate(&bad_frames).is_err());

    // generate IS wired (unlike the 5B stub) — it errors only by trying to open the absent weight
    // files, NOT with a "not yet wired" WIP message. (The real run needs the 54 GB checkpoint; see
    // the #[ignore] tests/s6_*.rs.)
    let mut noop = |_p| {};
    let err = g.generate(&ok, &mut noop).unwrap_err().to_string();
    assert!(
        !err.contains("not yet wired") && !err.contains("S1"),
        "14B generate must be wired (got a WIP stub message): {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_t2v_14b_rejects_non_dual_config_and_unwired_features() {
    // A single-model (Wan2.1) config is rejected by the dual-expert loader.
    let single = temp_model_dir_with("t2v14b_single", TI2V_5B_CONFIG);
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(single.clone()))
    )
    .is_err());
    std::fs::remove_dir_all(&single).ok();

    let dir = temp_model_dir_with("t2v14b_reject", T2V_14B_CONFIG);
    // Single-file source.
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::File(dir.join("config.json")))
    )
    .is_err());
    // Quantization (sc-2682) + adapters (sc-2683 / sc-2393) + precision override.
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_err());
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    assert!(registry::load(MODEL_ID_T2V_14B, &spec).is_err());
    let adapters = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
    }];
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(adapters)
    )
    .is_err());

    std::fs::remove_dir_all(&dir).ok();
}
