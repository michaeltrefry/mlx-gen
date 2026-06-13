//! sc-5148 e2e — the production [`LensTrainer`] (the `Trainer` contract realized on the base
//! `microsoft/Lens` DiT), driven through the core registry exactly as the SceneWorks worker will.
//!
//! `#[ignore]`d — needs the real `microsoft/Lens` weights in the HF cache (or `LENS_SNAPSHOT`). Run:
//!   cargo test -p mlx-gen-lens --release --test trainer_e2e -- --ignored --nocapture
//!
//! Proves the full prepare→load→cache→train→save lifecycle: a tiny captioned PNG dataset is
//! VAE/caption-encoded and cached, AdamW training drives the flow-match loss down, and a PEFT adapter
//! is written that reloads through the REAL inference path (`apply_lens_adapters`, sc-3174) onto a
//! fresh Lens DiT — the round-trip contract, for both the LoRA and LoKr adapter kinds. This is the
//! native-MLX replacement for `lens_train_runner.py` (zero-Python north star, epic 3482).

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};

use mlx_gen_lens::adapters::apply_lens_adapters;
use mlx_gen_lens::dit::{LensDitConfig, LensTransformer};

/// The base `microsoft/Lens` snapshot directory (the model the trainer fine-tunes — sc-1583): the
/// `LENS_SNAPSHOT` override, else the newest snapshot in the HF cache.
fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("LENS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let base = PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("Lens snapshot dir {} (set LENS_SNAPSHOT)", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir() && p.join("transformer").is_dir())
        .expect("a microsoft/Lens snapshot with a transformer/ tree")
}

/// Two solid-colour swatch PNGs + captions in `dir`.
fn make_dataset(dir: &Path) -> Vec<TrainingItem> {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
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

/// A fresh dense Lens DiT for the round-trip reload (bf16, the trainer's default compute dtype).
fn fresh_transformer(root: &Path) -> LensTransformer {
    let w = Weights::from_dir(root.join("transformer")).unwrap();
    LensTransformer::from_weights(&w, &LensDitConfig::lens(), mlx_rs::Dtype::Bfloat16).unwrap()
}

/// The Lens DiT output `[1, seq, 128]` on a fixed (deterministic) latent + 4 captured text-feature
/// layers (the DiT's `num_text_layers`) at the res-64 grid (4×4) — used to compare the base vs the
/// adapter-installed forward (the "shifts output" acceptance signal). The inputs are seeded, so the
/// only difference between two calls is the installed adapter.
fn forward_output(t: &LensTransformer) -> mlx_rs::Array {
    let latent = 4usize; // res 64 → 64/16
    let hidden = mlx_rs::random::normal::<f32>(
        &[1, (latent * latent) as i32, 128],
        None,
        None,
        Some(&mlx_rs::random::key(1).unwrap()),
    )
    .unwrap()
    .as_dtype(mlx_rs::Dtype::Bfloat16)
    .unwrap();
    let feats: Vec<mlx_rs::Array> = (0..4)
        .map(|k| {
            mlx_rs::random::normal::<f32>(
                &[1, 8, 2880],
                None,
                None,
                Some(&mlx_rs::random::key(10 + k).unwrap()),
            )
            .unwrap()
            .as_dtype(mlx_rs::Dtype::Bfloat16)
            .unwrap()
        })
        .collect();
    let timestep = mlx_rs::Array::from_slice(&[0.5f32], &[1]);
    t.forward(&hidden, &feats, None, &timestep, 1, latent, latent)
        .unwrap()
}

/// Relative L2 distance `‖other − base‖ / ‖base‖` (computed in f32).
fn relative_l2(base: &mlx_rs::Array, other: &mlx_rs::Array) -> f32 {
    let b = base.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let o = other.as_dtype(mlx_rs::Dtype::Float32).unwrap();
    let num = mlx_rs::ops::subtract(&o, &b)
        .unwrap()
        .square()
        .unwrap()
        .sum(None)
        .unwrap()
        .sqrt()
        .unwrap()
        .item::<f32>();
    let den = b
        .square()
        .unwrap()
        .sum(None)
        .unwrap()
        .sqrt()
        .unwrap()
        .item::<f32>()
        .max(1e-6);
    num / den
}

/// Mean of the first vs last third of the loss trajectory — a timestep-noise-robust learning signal
/// (each step samples a fresh sigmoid timestep, so per-step loss is dominated by that variance; a
/// third-window averages it out). Thirds (not quarters) widen the windows for a steadier estimate.
fn windowed_means(losses: &[f32]) -> (f32, f32) {
    let w = (losses.len() / 3).max(1);
    let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
    (mean(&losses[..w]), mean(&losses[losses.len() - w..]))
}

#[test]
#[ignore = "needs real microsoft/Lens weights (~20B gpt-oss encoder; loads Q8)"]
fn lens_trainer_trains_and_writes_lora() {
    let root = snapshot();
    let tmp = std::env::temp_dir().join("lens_trainer_lora_e2e");
    let items = make_dataset(&tmp);

    // Reference the crate so its `inventory::submit!` registration links into this binary.
    assert_eq!(mlx_gen_lens::registry::MODEL_ID_BASE, "lens");

    // Load the trainer through the registry (validates self-registration), exactly like the worker.
    let mut trainer =
        mlx_gen::load_trainer("lens", &LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("lens trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        // 1e-4 (standard LoRA lr); 1e-3 overshoots the higher-capacity LoRA on this 2-image overfit.
        learning_rate: 1e-4,
        steps: 80,
        resolution: 64, // bucketed to 64 → 4×4 latent, fast
        save_every: 0,
        seed: 7,
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_lora.safetensors".to_string(),
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
    assert_eq!(out.steps, 80, "all micro-steps should run");
    assert_eq!(losses.len(), 80);
    assert!(losses.iter().all(|l| l.is_finite()), "no NaN/Inf losses");

    let (first_q, last_q) = windowed_means(&losses);
    println!("[lens-trainer] cached {cached}; loss trajectory {losses:?}");
    println!("[lens-trainer] loss-mean first-third {first_q:.5} -> last-third {last_q:.5}");
    assert!(
        last_q < first_q * 0.9,
        "windowed loss-mean should fall ≥10% on real data: {first_q:.5} -> {last_q:.5}"
    );

    // The produced adapter carries the PEFT keys + inference-reload metadata.
    assert!(out.adapter_path.exists(), "adapter file should be written");
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    assert!(
        w.keys()
            .any(|k| k == "transformer_blocks.0.attn.img_qkv.lora_A.weight"),
        "adapter should contain PEFT-keyed LoRA factors on the joint-attention projections"
    );
    // 48 dual-stream blocks × 4 projections (img_qkv/txt_qkv/to_out.0/to_add_out).
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert_eq!(n_targets, 192, "48 blocks × 4 joint-attention projections");

    // Round-trip: free the trainer's model, reload the adapter through the REAL inference path onto a
    // fresh DiT, confirm every target resolves and a forward is finite.
    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let mut t = fresh_transformer(&root);
    let base_out = forward_output(&t); // base DiT, no adapter
    let report = apply_lens_adapters(
        &mut t,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("LoRA adapter should reload through the inference path");
    assert_eq!(report.applied, n_targets, "every saved LoRA target reloads");
    assert!(
        report.unmatched_paths.is_empty(),
        "every LoRA target should resolve"
    );
    // The reloaded adapter must measurably SHIFT the DiT output vs the base (the story's literal
    // acceptance) — a trained, non-trivial adapter, not a no-op.
    let adapted_out = forward_output(&t);
    assert!(
        adapted_out.sum(None).unwrap().item::<f32>().is_finite(),
        "reloaded LoRA forward should be finite"
    );
    let shift = relative_l2(&base_out, &adapted_out);
    println!("[lens-trainer] LoRA e2e OK — {n_targets} targets reloaded; output shift (rel-L2) {shift:.4}");
    assert!(
        shift > 1e-3,
        "the trained LoRA must shift the DiT output beyond fp noise (rel-L2 {shift:.5})"
    );
}

#[test]
#[ignore = "needs real microsoft/Lens weights (~20B gpt-oss encoder; loads Q8)"]
fn lens_trainer_trains_with_gradient_checkpointing() {
    // sc-5170 — the same LoRA run as above but with `gradient_checkpointing = true`, driving the
    // checkpointed DiT forward through the full `train_impl` loop (not just `compute_loss_grads` in
    // isolation). Because the checkpointed grads are bit-identical to dense (proven in the lib harness
    // `checkpointed_grads_match_dense`), the run must train, converge, and round-trip exactly like the
    // dense LoRA run. This is the integration proof of the `train_impl` plumbing + the produced adapter.
    let root = snapshot();
    let tmp = std::env::temp_dir().join("lens_trainer_ckpt_e2e");
    let items = make_dataset(&tmp);

    let mut trainer =
        mlx_gen::load_trainer("lens", &LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("lens trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        learning_rate: 1e-4,
        steps: 80,
        resolution: 64,
        save_every: 0,
        seed: 7,
        gradient_checkpointing: true, // <-- sc-5170: checkpointed DiT forward + SDPA-segment ckpt off
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_ckpt_lora.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut losses: Vec<f32> = Vec::new();
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { loss, .. } = p {
                losses.push(loss);
            }
        })
        .expect("checkpointed training should succeed");

    assert_eq!(
        out.steps, 80,
        "all micro-steps should run under checkpointing"
    );
    assert!(losses.iter().all(|l| l.is_finite()), "no NaN/Inf losses");
    let (first_q, last_q) = windowed_means(&losses);
    println!("[lens-trainer-ckpt] loss-mean first-third {first_q:.5} -> last-third {last_q:.5}");
    assert!(
        last_q < first_q * 0.9,
        "checkpointed windowed loss-mean should fall ≥10%: {first_q:.5} -> {last_q:.5}"
    );

    // The produced adapter round-trips through the REAL inference path, same as the dense run.
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lora"));
    let n_targets = w.keys().filter(|k| k.ends_with(".lora_A.weight")).count();
    assert_eq!(n_targets, 192, "48 blocks × 4 joint-attention projections");

    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let mut t = fresh_transformer(&root);
    let base_out = forward_output(&t);
    let report = apply_lens_adapters(
        &mut t,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("checkpointed-run LoRA adapter should reload through the inference path");
    assert_eq!(report.applied, n_targets, "every saved LoRA target reloads");
    assert!(report.unmatched_paths.is_empty());
    let adapted_out = forward_output(&t);
    let shift = relative_l2(&base_out, &adapted_out);
    println!("[lens-trainer-ckpt] checkpointed e2e OK — {n_targets} targets; output shift (rel-L2) {shift:.4}");
    assert!(
        shift > 1e-3,
        "the checkpointed-trained LoRA must shift the DiT output beyond fp noise (rel-L2 {shift:.5})"
    );
}

#[test]
#[ignore = "needs real microsoft/Lens weights (~20B gpt-oss encoder; loads Q8)"]
fn lens_trainer_trains_and_reloads_lokr() {
    let root = snapshot();
    let tmp = std::env::temp_dir().join("lens_trainer_lokr_e2e");
    let items = make_dataset(&tmp);
    assert_eq!(mlx_gen_lens::registry::MODEL_ID_BASE, "lens");

    let mut trainer =
        mlx_gen::load_trainer("lens", &LoadSpec::new(WeightsSource::Dir(root.clone())))
            .expect("lens trainer should be registered");

    let config = TrainingConfig {
        rank: 8,
        alpha: 8.0,
        // LoKr's delta is `kron(w1, w2)` with both factors initialised small, so the effective
        // gradient magnitude is far smaller than LoRA's `B·A` — it needs a higher lr to learn at a
        // comparable rate in a bounded run (standard LyCORIS guidance; lr 1e-4 barely moves it,
        // whereas it converges stably at 1e-3 — the inverse of LoRA, which 1e-3 overshoots).
        learning_rate: 1e-3,
        steps: 80,
        resolution: 64,
        save_every: 0,
        seed: 7,
        network_type: NetworkType::Lokr, // <-- LyCORIS Kronecker adapter (sc-2218)
        decompose_factor: -1,            // balanced/auto
        ..Default::default()
    };
    let req = TrainingRequest {
        items,
        config,
        output_dir: tmp.join("out"),
        file_name: "swatch_lokr.safetensors".to_string(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };

    let mut losses: Vec<f32> = Vec::new();
    let out = trainer
        .train(&req, &mut |p| {
            if let TrainingProgress::Training { loss, .. } = p {
                losses.push(loss);
            }
        })
        .expect("LoKr training should succeed");

    assert_eq!(out.steps, 80);
    assert!(
        losses.iter().all(|l| l.is_finite()),
        "no NaN/Inf (kron autograd is sane)"
    );
    let (first_q, last_q) = windowed_means(&losses);
    println!("[lens-trainer-lokr] loss trajectory {losses:?}");
    println!("[lens-trainer-lokr] loss-mean first-third {first_q:.5} -> last-third {last_q:.5}");
    assert!(
        last_q < first_q * 0.9,
        "LoKr windowed loss-mean should fall ≥10%: {first_q:.5} -> {last_q:.5}"
    );

    // Adapter carries LoKr keys + metadata; one `lokr_w1` per trained target.
    let w = Weights::from_file(&out.adapter_path).unwrap();
    assert_eq!(w.metadata("networkType"), Some("lokr"));
    assert!(w.metadata("decomposeFactor").is_some());
    assert!(
        w.keys()
            .any(|k| k == "transformer_blocks.0.attn.img_qkv.lokr_w1"),
        "adapter should contain LoKr factor keys"
    );
    let n_targets = w.keys().filter(|k| k.ends_with(".lokr_w1")).count();
    assert_eq!(n_targets, 192, "48 blocks × 4 joint-attention projections");

    // Round-trip through the REAL inference path (parse_lokr → reconstruct_lokr_delta at bf16).
    let adapter_path = out.adapter_path.clone();
    drop(trainer);
    let mut t = fresh_transformer(&root);
    let base_out = forward_output(&t);
    let report = apply_lens_adapters(
        &mut t,
        &[AdapterSpec {
            path: adapter_path,
            scale: 1.0,
            kind: AdapterKind::Lokr,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .expect("LoKr adapter should reload through the inference path");
    assert_eq!(report.applied, n_targets, "every saved LoKr target reloads");
    assert!(
        report.unmatched_paths.is_empty(),
        "every LoKr target should resolve"
    );
    let adapted_out = forward_output(&t);
    assert!(
        adapted_out.sum(None).unwrap().item::<f32>().is_finite(),
        "reloaded LoKr forward should be finite"
    );
    let shift = relative_l2(&base_out, &adapted_out);
    println!("[lens-trainer-lokr] e2e OK — {n_targets} LoKr targets reloaded; output shift (rel-L2) {shift:.4}");
    assert!(
        shift > 1e-3,
        "the trained LoKr must shift the DiT output beyond fp noise (rel-L2 {shift:.5})"
    );
}
