//! Wan2.2 trainer e2e — the production `WanMoeTrainer` driven through the core registry exactly as
//! the SceneWorks worker will, across all three Wan trainers:
//!   * `wan2_2_t2v_14b` — dual-expert MoE (sc-3046).
//!   * `wan2_2_i2v_14b` — dual-expert MoE, channel-concat in_dim 36 / zero-`y` pad (sc-3279).
//!   * `wan2_2_ti2v_5b` — dense single-expert, z48 vae22 (sc-3279).
//!
//! `#[ignore]`d — each needs a converted bf16 snapshot (env-var override, else the SceneWorks /
//! mlx-gen cache default). Run:
//!   cargo test -p mlx-gen-wan --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/UMT5-encoded and cached (then the TE freed), the expert(s) train on their noise band(s), and
//! the adapter(s) reload through the REAL Wan inference merge (`merge_wan_adapters`) per expert.

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, MoeExpert, NetworkType, TrainingConfig,
    TrainingItem, TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_wan::merge_wan_adapters;

/// Resolve a snapshot dir from `$env_var`, else fall back to `default` under `$HOME`.
fn snapshot(env_var: &str, default: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_var) {
        return PathBuf::from(p);
    }
    PathBuf::from(std::env::var("HOME").unwrap()).join(default)
}

/// One trained expert's reload spec: its file suffix, the base weight file it merges into, and the
/// MoE tag (`None` for the dense single-file adapter).
struct ExpertFile {
    suffix: &'static str,
    weights_file: &'static str,
    expert: Option<MoeExpert>,
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(128, 128);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = dir.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }
    items
}

/// Merge a trained adapter file into a fresh weight map, asserting every target applies. `expert ==
/// None` is the dense single-file adapter (untagged; merges regardless of the expert arg).
fn assert_reloads(
    snapshot: &Path,
    file: &Path,
    expert: Option<MoeExpert>,
    weights_file: &str,
    n_targets: usize,
) {
    let mut w = Weights::from_file(snapshot.join(weights_file)).unwrap();
    let spec = AdapterSpec {
        path: file.to_path_buf(),
        scale: 1.0,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: expert,
    };
    let report = merge_wan_adapters(
        &mut w,
        std::slice::from_ref(&spec),
        expert.unwrap_or(MoeExpert::High),
    )
    .expect("trained LoRA should merge through the inference path");
    assert_eq!(
        report.applied, n_targets,
        "every trained {expert:?} target should merge"
    );
    assert!(
        report.skipped.is_empty(),
        "no {expert:?} key should be skipped: {:?}",
        report.skipped
    );
}

/// The shared lifecycle driver: load `model_id` from `snapshot`, train a tiny LoRA with `optimizer`,
/// assert the windowed loss falls, then reload each written adapter through `merge_wan_adapters`.
fn run_trainer_e2e(model_id: &str, snapshot: PathBuf, experts: &[ExpertFile], optimizer: &str) {
    let tmp = std::env::temp_dir().join(format!("{model_id}_{optimizer}_trainer_e2e"));
    let items = make_dataset(&tmp);

    let mut trainer = mlx_gen::load_trainer(
        model_id,
        &LoadSpec::new(WeightsSource::Dir(snapshot.clone())),
    )
    .unwrap_or_else(|e| panic!("{model_id} trainer should load: {e}"));

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-4,
        steps: 24,       // dual: 12/expert (alternating); dense: 24 on the single expert
        resolution: 256, // bucketed to 256
        save_every: 0,
        seed: 7,
        network_type: NetworkType::Lora,
        optimizer: optimizer.to_string(),
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut losses: Vec<f32> = Vec::new();
    let mut cached = 0u32;
    let out = trainer
        .train(&req, &mut |p| match p {
            TrainingProgress::Caching { current, .. } => cached = current,
            TrainingProgress::Training { loss, .. } => losses.push(loss),
            _ => {}
        })
        .expect("training should succeed");

    assert_eq!(cached, 2, "both dataset items should be cached");
    assert_eq!(out.steps, 24);
    assert_eq!(losses.len(), 24);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf losses (functional autograd is sane)"
    );
    let q = losses.len() / 4;
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    let (first_q, last_q) = (mean(&losses[..q]), mean(&losses[losses.len() - q..]));
    println!(
        "[{model_id}] cached {cached}; steps {}; loss-mean first-quarter {first_q:.5} -> last-quarter {last_q:.5}",
        out.steps
    );
    assert!(
        last_q < first_q * 0.95,
        "windowed loss-mean should fall on real data: {first_q:.5} -> {last_q:.5}"
    );

    // Locate each written adapter (dense: `swatch.safetensors`; dual: `swatch.{high,low}_noise...`).
    let out_dir = tmp.join("out");
    let primary = out.adapter_path.clone();
    let wp = Weights::from_file(&primary).unwrap();
    assert_eq!(wp.metadata("networkType"), Some("lora"));
    let n_targets = wp.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert!(n_targets > 0, "adapter should contain LoRA factors");
    assert!(
        wp.keys().any(|k| k.ends_with(".self_attn.q.lora_A.weight")),
        "adapter should carry native Wan attention keys"
    );

    for ef in experts {
        let file = if ef.suffix.is_empty() {
            out_dir.join("swatch.safetensors")
        } else {
            out_dir.join(format!("swatch.{}.safetensors", ef.suffix))
        };
        assert!(
            file.exists(),
            "expert adapter should be written: {}",
            file.display()
        );
        assert_reloads(&snapshot, &file, ef.expert, ef.weights_file, n_targets);
    }
    println!(
        "[{model_id}] e2e OK — {} adapter(s), {n_targets} targets/expert reload through merge_wan_adapters",
        experts.len()
    );
}

#[test]
#[ignore = "needs the converted Wan2.2-T2V-A14B MoE bf16 checkpoint"]
fn wan_t2v_a14b_trainer_trains_both_experts_and_reloads() {
    assert_eq!(mlx_gen_wan::MODEL_ID_T2V_14B, "wan2_2_t2v_14b");
    let snap = snapshot(
        "WAN_A14B_MODEL_DIR",
        ".cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16",
    );
    run_trainer_e2e("wan2_2_t2v_14b", snap, &moe_experts(), "adamw");
}

/// The dual-expert (high/low) reload spec for the A14B MoE trainers.
fn moe_experts() -> Vec<ExpertFile> {
    vec![
        ExpertFile {
            suffix: "high_noise",
            weights_file: "high_noise_model.safetensors",
            expert: Some(MoeExpert::High),
        },
        ExpertFile {
            suffix: "low_noise",
            weights_file: "low_noise_model.safetensors",
            expert: Some(MoeExpert::Low),
        },
    ]
}

/// Resolve the dense TI2V-5B snapshot (env override else the SceneWorks model dir).
fn ti2v_5b_snapshot() -> PathBuf {
    snapshot(
        "WAN_TI2V_5B_MODEL_DIR",
        "Library/Application Support/SceneWorks/data/models/mlx/wan_2_2_ti2v_5b",
    )
}

/// The dense single-file reload spec for the TI2V-5B trainer.
fn dense_expert() -> Vec<ExpertFile> {
    vec![ExpertFile {
        suffix: "",
        weights_file: "model.safetensors",
        expert: None,
    }]
}

#[test]
#[ignore = "needs the converted Wan2.2-I2V-A14B MoE bf16 checkpoint"]
fn wan_i2v_a14b_trainer_trains_both_experts_and_reloads() {
    assert_eq!(mlx_gen_wan::MODEL_ID_I2V_14B, "wan2_2_i2v_14b");
    let snap = snapshot(
        "WAN_I2V_A14B_MODEL_DIR",
        ".cache/mlx-gen-models/wan2_2_i2v_a14b_mlx_bf16",
    );
    run_trainer_e2e("wan2_2_i2v_14b", snap, &moe_experts(), "adamw");
}

#[test]
#[ignore = "needs the converted Wan2.2-TI2V-5B dense bf16 checkpoint (with z48 vae)"]
fn wan_ti2v_5b_trainer_trains_dense_and_reloads() {
    assert_eq!(mlx_gen_wan::MODEL_ID, "wan2_2_ti2v_5b");
    run_trainer_e2e(
        "wan2_2_ti2v_5b",
        ti2v_5b_snapshot(),
        &dense_expert(),
        "adamw",
    );
}

// sc-3048 — the ported Rose + Prodigy optimizers drive a real training run end-to-end (loss falls +
// the adapter reloads), on the fast dense TI2V-5B. The optimizers themselves are numerically
// validated against the torch reference in the core `train::optim` unit tests.
#[test]
#[ignore = "needs the converted Wan2.2-TI2V-5B dense bf16 checkpoint (with z48 vae)"]
fn wan_ti2v_5b_trainer_rose_optimizer() {
    run_trainer_e2e(
        "wan2_2_ti2v_5b",
        ti2v_5b_snapshot(),
        &dense_expert(),
        "rose",
    );
}

#[test]
#[ignore = "needs the converted Wan2.2-TI2V-5B dense bf16 checkpoint (with z48 vae)"]
fn wan_ti2v_5b_trainer_prodigy_optimizer() {
    run_trainer_e2e(
        "wan2_2_ti2v_5b",
        ti2v_5b_snapshot(),
        &dense_expert(),
        "prodigy",
    );
}
