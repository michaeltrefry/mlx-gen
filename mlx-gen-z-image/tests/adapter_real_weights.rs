//! sc-2602: end-to-end Z-Image LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real `Tongyi-MAI/Z-Image-Turbo` weights in the HF cache and the
//! adapter goldens produced by `tools/dump_z_image_adapter_golden.py` (gitignored, local). Run:
//!   cargo test -p mlx-gen-z-image --release --test adapter_real_weights -- --ignored --nocapture
//!
//! Three gates: (1) the key→module map resolves the FULL fork `ZImageLoRAMapping` target surface
//! against the real module tree; (2) the public `load(spec.with_adapters(…)).generate()` render
//! matches the fork's LoRA *and* LoKr golden (px>8); (3) a scale-0 adapter is a bit-exact no-op.

use std::path::PathBuf;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_z_image::{decoded_to_image, load_transformer};

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("ZIMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// (1) The top-level `AdaptableHost` resolves every fork `ZImageLoRAMapping` target against the
/// real module tree (30 layers + 2 noise_refiner + 2 context_refiner + 6 globals), and rejects
/// off-surface paths — no real weights for the *render*, but needs them to build the transformer.
#[test]
#[ignore = "needs real Z-Image weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    let block_targets = [
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out.0",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "adaLN_modulation.0",
    ];
    // Main + noise_refiner blocks expose all eight targets at every index.
    for (stack, n) in [("layers", 30usize), ("noise_refiner", 2)] {
        for i in 0..n {
            for tgt in block_targets {
                let p = format!("{stack}.{i}.{tgt}");
                assert!(resolves(&mut t, &p), "expected {p} to resolve");
            }
        }
    }
    // Context-refiner blocks have no timestep → attention + feed_forward only (adaLN is correctly
    // absent, mirroring the fork: the file never populates it).
    for i in 0..2 {
        assert!(resolves(
            &mut t,
            &format!("context_refiner.{i}.attention.to_q")
        ));
        assert!(resolves(
            &mut t,
            &format!("context_refiner.{i}.feed_forward.w2")
        ));
        assert!(
            !resolves(&mut t, &format!("context_refiner.{i}.adaLN_modulation.0")),
            "context blocks carry no adaLN"
        );
    }
    // The six global targets (trained-file naming).
    for p in [
        "all_x_embedder.2-1",
        "cap_embedder.1",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
        "all_final_layer.2-1.linear",
        "all_final_layer.2-1.adaLN_modulation.1",
    ] {
        assert!(resolves(&mut t, p), "expected global {p} to resolve");
    }
    // Off-surface paths must not resolve.
    for p in [
        "layers.30.attention.to_q",               // out of range
        "layers.0.attention.to_x",                // unknown proj
        "all_final_layer.2-1.adaLN_modulation.0", // final layer uses index 1, not 0
        "cap_embedder.0",                         // the RMSNorm, not a Linear
        "t_embedder.mlp.1",                       // the SiLU slot
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ routing map covers the full fork ZImageLoRAMapping surface");
}

fn render_with_adapter(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g =
        Weights::from_file(golden_dir().join(format!("z_image_{golden_kind}_golden.safetensors")))
            .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
        }]);
    }
    let generator = mlx_gen::load("z_image_turbo", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    let out = generator.generate(&req, &mut |_| {}).unwrap();
    match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let pixels = render_with_adapter(
        Some((&format!("z_image_{kind}_adapter.safetensors"), my_kind, 1.0)),
        kind,
    );
    let g = Weights::from_file(golden_dir().join(format!("z_image_{kind}_golden.safetensors")))
        .unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f64 / pixels.len() as f64;
    println!(
        "✓ {kind} adapter render: {differ}/{} px differ by >8 from the fork ({:.4}%)",
        pixels.len(),
        frac * 100.0
    );
    assert!(
        differ < pixels.len() / 20,
        "{kind} adapter render diverges from the fork: {differ} px ({:.3}%)",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real Z-Image weights + adapter golden"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real Z-Image weights + adapter golden"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// A scale-0 adapter must be a bit-exact no-op vs the no-adapter render (no regression).
#[test]
#[ignore = "needs real Z-Image weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render_with_adapter(None, "lora");
    let zero = render_with_adapter(
        Some(("z_image_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("✓ scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}
