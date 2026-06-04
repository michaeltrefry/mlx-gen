//! Registry wiring + config-driven `load` for all Wan models.
//!
//! Verifies that `wan2_2_ti2v_5b` (dense 5B, `generate` fully wired — sc-2680),
//! `wan2_2_t2v_14b` (dual-expert MoE T2V, `generate` fully wired) and `wan2_2_i2v_14b` (dual-expert
//! MoE channel-concat I2V, `generate` fully wired) each self-register with the right descriptor, that
//! `load` reads the model's `config.json` (5B preset / dual-expert / i2v detection), that the 14B
//! loaders reject a mismatched config, that Q4/Q8 + LoRA/LoKr are accepted at load (applied in
//! generate), and that an invalid source / precision override is rejected for all.

use std::path::PathBuf;

use mlx_gen::{
    registry, AdapterKind, AdapterSpec, Conditioning, GenerationRequest, Image, LoadSpec, Modality,
    Precision, Quant, WeightsSource,
};

use mlx_gen_wan::{MODEL_ID, MODEL_ID_I2V_14B, MODEL_ID_T2V_14B};

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

/// The I2V-A14B's serialized `config.json` (`convert_wan.py` schema; dual-expert channel-concat,
/// in_dim 36, boundary 0.9, guide (3.5, 3.5), max_area 704×1280, z16 VAE).
const I2V_14B_CONFIG: &str = r#"{
  "model_type": "i2v",
  "model_version": "2.2",
  "patch_size": [1, 2, 2],
  "in_dim": 36,
  "dim": 5120,
  "ffn_dim": 13824,
  "out_dim": 16,
  "num_heads": 40,
  "num_layers": 40,
  "vae_z_dim": 16,
  "vae_stride": [4, 8, 8],
  "dual_model": true,
  "boundary": 0.9,
  "sample_shift": 5.0,
  "sample_steps": 40,
  "sample_guide_scale": [3.5, 3.5],
  "sample_fps": 16,
  "frame_num": 81,
  "max_area": 901120
}"#;

/// A **pre-quantized** T2V-A14B `config.json` — the dense config plus a `quantization` block
/// (`convert_wan.py` writes this when it packs the transformer). `load` must accept it (the snapshot
/// ships packed weights consumed by `from_weights`) without a `spec.quantize` override.
const T2V_14B_Q4_CONFIG: &str = r#"{
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
  "max_area": 0,
  "quantization": {"group_size": 64, "bits": 4}
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
    // LoRA (sc-2683) + LoKr (sc-2393) merge onto the single dense model at generate; Q4/Q8 (sc-2682)
    // via spec.quantize.
    assert!(d.capabilities.supports_lora);
    assert!(d.capabilities.supports_lokr);
    assert!(d.capabilities.samplers.contains(&"unipc"));
    // H/W align to patch×vae_stride = 32 for the z48 vae22 (spatial stride 16).
    assert_eq!(d.capabilities.min_size, 32);
}

#[test]
fn load_reads_config_and_wires_generate() {
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

    // generate IS wired (sc-2680, no longer a WIP stub) — with only config.json in the dir it errors
    // by trying to open the absent `t5_encoder.safetensors`, not by a scaffold/stub message.
    let mut noop = |_p| {};
    let err = g.generate(&ok, &mut noop).unwrap_err().to_string();
    assert!(
        !err.contains("scaffold") && !err.contains("not yet wired") && !err.contains("S1–S5"),
        "5B generate must be wired (got a WIP stub message): {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_rejects_bad_source_and_precision() {
    let dir = temp_model_dir("reject");
    // Single-file source.
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::File(dir.join("config.json")))
    )
    .is_err());
    // Precision override (the DiT runs bf16 GEMMs over an f32 residual — the parity regime).
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    assert!(registry::load(MODEL_ID, &spec).is_err());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_accepts_quant_and_adapters() {
    // Q4/Q8 (sc-2682) is now WIRED at load (the DiT is quantized lazily in generate); the e2e
    // numerics ride the shared WanTransformer::quantize path.
    let dir = temp_model_dir("quant");
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_ok());
    // LoRA/LoKr (sc-2683 / sc-2393) are accepted at load — the file is read + merged in generate
    // (which needs the real model weights), so load itself succeeds.
    let adapters = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    }];
    assert!(registry::load(
        MODEL_ID,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(adapters)
    )
    .is_ok());

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
    // LoRA (sc-2683) + LoKr (sc-2393) in generate, per-expert merge.
    assert!(d.capabilities.supports_lora);
    assert!(d.capabilities.supports_lokr);
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
    // Quantization (sc-2682) is now WIRED: Q4/Q8 is accepted at load (each expert is quantized
    // lazily in generate). The e2e numerics are gated by tests/quant_e2e_parity.rs.
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_ok());
    // Precision override (the experts run bf16 GEMMs over an f32 residual) is still rejected.
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    assert!(registry::load(MODEL_ID_T2V_14B, &spec).is_err());
    // LoKr (sc-2393) is now ACCEPTED at load — merged per expert at generate (the file is read then).
    let lokr = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lokr,
        pass_scales: None,
        moe_expert: None,
    }];
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(lokr)
    )
    .is_ok());
    // A LoRA adapter (incl. an MoE-expert-tagged one) is ACCEPTED at load (sc-2683); the per-expert
    // merge is deferred to generate (which needs the real expert weights), so load itself succeeds.
    let lora = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: Some(mlx_gen::MoeExpert::High),
    }];
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(lora)
    )
    .is_ok());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_t2v_14b_accepts_prequantized_snapshot_and_reconciles_spec() {
    // A pre-quantized snapshot (config.json carries a `quantization` block) loads WITHOUT a
    // spec.quantize override — `from_weights` builds the experts from the on-disk packed weights.
    let dir = temp_model_dir_with("t2v14b_q4", T2V_14B_Q4_CONFIG);
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone()))
    )
    .is_ok());
    // A matching spec.quantize is fine (redundant with the manifest).
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q4)
    )
    .is_ok());
    // A conflicting spec.quantize (Q8 vs the manifest's Q4) is rejected — the on-disk manifest is
    // authoritative; we never silently re-quantize at a different width.
    assert!(registry::load(
        MODEL_ID_T2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_err());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn wan_i2v_14b_is_registered() {
    let reg = registry::generators()
        .find(|r| (r.descriptor)().id == MODEL_ID_I2V_14B)
        .expect("wan2_2_i2v_14b not registered");
    let d = (reg.descriptor)();
    assert_eq!(d.id, "wan2_2_i2v_14b");
    assert_eq!(d.family, "wan");
    assert_eq!(d.modality, Modality::Video);
    // Dual-expert CFG + negative prompt + KV cache; a single image reference (channel-concat).
    assert!(d.capabilities.supports_guidance);
    assert!(d.capabilities.supports_negative_prompt);
    assert!(d.capabilities.supports_kv_cache);
    assert_eq!(
        d.capabilities.conditioning,
        vec![mlx_gen::ConditioningKind::Reference]
    );
    assert_eq!(d.capabilities.max_count, 1);
    // LoRA (sc-2683) + LoKr (sc-2393) in generate, per-expert merge.
    assert!(d.capabilities.supports_lora);
    assert!(d.capabilities.supports_lokr);
    assert_eq!(d.capabilities.min_size, 16);
}

/// A tiny solid-color RGB reference image for I2V validate tests.
fn dummy_image() -> Image {
    Image {
        width: 64,
        height: 64,
        pixels: vec![128u8; 64 * 64 * 3],
    }
}

#[test]
fn load_i2v_14b_reads_config_validates_and_wires_generate() {
    let dir = temp_model_dir_with("i2v14b", I2V_14B_CONFIG);
    let g = registry::load(
        MODEL_ID_I2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())),
    )
    .expect("load should succeed (reads i2v config.json)");
    assert_eq!(g.descriptor().id, MODEL_ID_I2V_14B);

    let with_ref = |extra: fn(&mut GenerationRequest)| {
        let mut req = GenerationRequest {
            width: 512,
            height: 512,
            frames: Some(81),
            conditioning: vec![Conditioning::Reference {
                image: dummy_image(),
                strength: None,
            }],
            ..Default::default()
        };
        extra(&mut req);
        req
    };

    // Accepts a 16-aligned 1+4k-frame request WITH a reference image.
    assert!(g.validate(&with_ref(|_| {})).is_ok());
    // Rejects a request WITHOUT a reference image (I2V requires the first frame).
    let no_ref = GenerationRequest {
        width: 512,
        height: 512,
        frames: Some(81),
        ..Default::default()
    };
    assert!(g.validate(&no_ref).is_err());
    // Rejects trim_first_frames (the conditioning `y` is built from num_frames).
    assert!(g
        .validate(&with_ref(|r| r.trim_first_frames = Some(1)))
        .is_err());
    // Still rejects sub-tile + bad frame counts.
    assert!(g.validate(&with_ref(|r| r.width = 8)).is_err());
    assert!(g.validate(&with_ref(|r| r.frames = Some(80))).is_err());

    // generate IS wired — it errors only by trying to open the absent weight files (or, here, by
    // requiring the reference at run time), NOT with a "not yet wired" WIP message.
    let mut noop = |_p| {};
    let err = g
        .generate(&with_ref(|_| {}), &mut noop)
        .unwrap_err()
        .to_string();
    assert!(
        !err.contains("not yet wired") && !err.contains("S1"),
        "i2v generate must be wired (got a WIP stub message): {err}"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn load_i2v_14b_rejects_non_i2v_config_and_unwired_features() {
    // A T2V (non-channel-concat) config is rejected by the i2v loader.
    let t2v = temp_model_dir_with("i2v14b_t2v", T2V_14B_CONFIG);
    assert!(registry::load(
        MODEL_ID_I2V_14B,
        &LoadSpec::new(WeightsSource::Dir(t2v.clone()))
    )
    .is_err());
    std::fs::remove_dir_all(&t2v).ok();

    let dir = temp_model_dir_with("i2v14b_reject", I2V_14B_CONFIG);
    // Single-file source.
    assert!(registry::load(
        MODEL_ID_I2V_14B,
        &LoadSpec::new(WeightsSource::File(dir.join("config.json")))
    )
    .is_err());
    // Quantization (sc-2682) is now WIRED: Q4/Q8 is accepted at load (each expert is quantized
    // lazily in generate). The e2e numerics are gated by tests/quant_e2e_parity.rs.
    assert!(registry::load(
        MODEL_ID_I2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_quant(Quant::Q8)
    )
    .is_ok());
    // Precision override (the experts run bf16 GEMMs over an f32 residual) is still rejected.
    let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
    spec.precision = Precision::Fp32;
    assert!(registry::load(MODEL_ID_I2V_14B, &spec).is_err());
    // LoKr (sc-2393) is now ACCEPTED at load — merged per expert at generate (the file is read then).
    let lokr = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lokr,
        pass_scales: None,
        moe_expert: None,
    }];
    assert!(registry::load(
        MODEL_ID_I2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(lokr)
    )
    .is_ok());
    // A LoRA adapter is ACCEPTED at load (sc-2683); the per-expert merge is deferred to generate.
    let lora = vec![AdapterSpec {
        path: dir.join("x.safetensors"),
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: Some(mlx_gen::MoeExpert::Low),
    }];
    assert!(registry::load(
        MODEL_ID_I2V_14B,
        &LoadSpec::new(WeightsSource::Dir(dir.clone())).with_adapters(lora)
    )
    .is_ok());

    std::fs::remove_dir_all(&dir).ok();
}
