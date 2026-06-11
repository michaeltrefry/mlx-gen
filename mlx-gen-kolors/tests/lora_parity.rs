//! sc-4733 — Kolors **inference** LoRA merge parity. Trains a tiny real LoRA via the registry
//! trainer (sc-4568), then reloads it through the inference adapter path (`Kolors::apply_lora` →
//! `apply_sdxl_adapters_with` on the Kolors U-Net) and gates the two invariants the SDXL-family
//! adapter merge guarantees:
//!   * **`scale = 0 ≡ base`** — merging `0·delta` into the dense f32 weights is byte-exact (`max|Δ|=0`)
//!     vs the un-adapted U-Net (the false-merge guard);
//!   * **`scale = 1` has effect** — the trained delta visibly moves the forward output.
//!
//! `#[ignore]`d — needs the real `Kwai-Kolors/Kolors-diffusers` snapshot (or `KOLORS_SNAPSHOT`) with
//! the materialized `tokenizer/tokenizer.json`. Run:
//!   cargo test -p mlx-gen-kolors --release --test lora_parity -- --ignored --nocapture
//!
//! Validated at **f32** (the SDXL merge dtype): `scale=0≡base` is byte-exact regardless of dtype
//! (`w + 0·delta = w`), but f32 keeps the `scale=1` forward free of fp16 chaos so the effect read is
//! clean. The merge surface is the registry's production **Complete** coverage.

use std::path::{Path, PathBuf};

use mlx_gen::{
    AdapterKind, AdapterSpec, CancelFlag, LoadSpec, NetworkType, TrainingConfig, TrainingItem,
    TrainingProgress, TrainingRequest, WeightsSource,
};
use mlx_gen_sdxl::{apply_sdxl_adapters_with, load_unet_kolors_dtype, LoraCoverage};
use mlx_rs::{ops, random, Array, Dtype};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// Train a tiny real LoRA (the sc-4568 path) and return its adapter file.
fn train_lora(tmp: &Path) -> PathBuf {
    // Force-link mlx-gen-kolors so its `inventory::submit!` trainer registration isn't dead-stripped
    // (this test reaches the trainer only via the generic `mlx_gen::load_trainer`, naming no
    // `mlx_gen_kolors::` symbol otherwise → "no trainer registered for id 'kolors'").
    assert_eq!(mlx_gen_kolors::MODEL_ID, "kolors");
    std::fs::create_dir_all(tmp).unwrap();
    let mut items = Vec::new();
    for (i, color) in [[200u8, 40, 40], [40, 80, 200]].iter().enumerate() {
        let mut img = image::RgbImage::new(96, 96);
        for px in img.pixels_mut() {
            *px = image::Rgb(*color);
        }
        let path = tmp.join(format!("img{i}.png"));
        img.save(&path).unwrap();
        items.push(TrainingItem {
            image_path: path,
            caption: format!("a solid colour swatch number {i}"),
        });
    }
    let mut trainer =
        mlx_gen::load_trainer("kolors", &LoadSpec::new(WeightsSource::Dir(snapshot()))).unwrap();
    let req = TrainingRequest {
        items,
        config: TrainingConfig {
            rank: 8,
            alpha: 8.0,
            learning_rate: 1e-3,
            steps: 16,
            resolution: 64,
            save_every: 0,
            seed: 7,
            network_type: NetworkType::Lora,
            decompose_factor: -1,
            ..Default::default()
        },
        output_dir: tmp.join("out"),
        file_name: "swatch_lora.safetensors".into(),
        trigger_words: vec![],
        cancel: CancelFlag::new(),
    };
    let out = trainer
        .train(&req, &mut |p: TrainingProgress| {
            if let TrainingProgress::Training { .. } = p {}
        })
        .expect("training should succeed");
    out.adapter_path
}

/// A fixed-input U-Net forward (Kolors conditioning shapes: ChatGLM context `[1,N,4096]`, pooled
/// `[1,4096]`, real-resolution `time_ids`).
fn forward(unet: &mlx_gen_sdxl::UNet2DConditionModel) -> Array {
    let mk = |shape: &[i32], seed: u64| {
        random::normal::<f32>(shape, None, None, Some(&random::key(seed).unwrap())).unwrap()
    };
    let x = mk(&[1, 8, 8, 4], 1);
    let cond = mk(&[1, 8, 4096], 2);
    let pooled = mk(&[1, 4096], 3);
    let row = [64.0f32, 64.0, 0.0, 0.0, 64.0, 64.0];
    let time_ids = Array::from_slice(&row, &[1, 6]);
    unet.forward(&x, 500.0, &cond, &pooled, &time_ids).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let d = ops::abs(ops::subtract(a, b).unwrap()).unwrap();
    ops::max(&d, None)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
        .item::<f32>()
}

#[test]
#[ignore = "needs real Kolors weights"]
fn kolors_inference_lora_scale0_is_base_and_scale1_has_effect() {
    let tmp = std::env::temp_dir().join("kolors_lora_parity");
    let _ = std::fs::remove_dir_all(&tmp);
    let adapter = train_lora(&tmp);

    let spec = |scale: f32| AdapterSpec {
        path: adapter.clone(),
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
        moe_expert: None,
    };

    // Base (un-adapted) reference.
    let base = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let eps_base = forward(&base);

    // scale = 0 ≡ base, byte-exact (the merge adds 0·delta to the dense weights).
    let mut unet0 = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    let r0 = apply_sdxl_adapters_with(&mut unet0, &[spec(0.0)], LoraCoverage::Complete).unwrap();
    assert!(
        r0.merged > 0,
        "scale-0 merge should still touch every target"
    );
    assert_eq!(r0.skipped_keys, 0, "no LoRA key should be skipped");
    let eps0 = forward(&unet0);
    let d0 = max_abs_diff(&eps0, &eps_base);
    println!(
        "[kolors-lora-parity] scale=0 vs base max|Δ|={d0:.3e} ({} merged)",
        r0.merged
    );
    assert_eq!(d0, 0.0, "scale=0 must be byte-exact to the un-adapted base");

    // scale = 1 moves the output (the trained delta has effect).
    let mut unet1 = load_unet_kolors_dtype(&snapshot(), Dtype::Float32).unwrap();
    apply_sdxl_adapters_with(&mut unet1, &[spec(1.0)], LoraCoverage::Complete).unwrap();
    let eps1 = forward(&unet1);
    let d1 = max_abs_diff(&eps1, &eps_base);
    println!("[kolors-lora-parity] scale=1 vs base max|Δ|={d1:.3e}");
    assert!(
        d1 > 0.0,
        "scale=1 trained LoRA must change the forward output"
    );
}
