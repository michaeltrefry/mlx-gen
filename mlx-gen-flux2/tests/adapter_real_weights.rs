//! sc-2646: end-to-end FLUX.2-klein-9b LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real FLUX.2-klein-9b snapshot (env `MLX_GEN_FLUX2_SNAPSHOT` or the HF
//! cache) and the adapter goldens from `tools/dump_flux2_adapter_golden.py` (gitignored, local):
//!   cd ~/repos/mflux && .venv/bin/python ~/repos/mlx-gen/tools/dump_flux2_adapter_golden.py
//!   cargo test -p mlx-gen-flux2 --test adapter_real_weights -- --ignored --nocapture
//!
//! Gates: (1) the key→module map resolves the FULL fork `Flux2LoRAMapping` surface (globals + 8
//! double × 12 + 24 single × 2) against the real module tree, and rejects off-surface; (2) the
//! public `load(spec.with_adapters(…)).generate()` render matches the fork's LoRA *and* LoKr golden
//! (px>8, below the cross-build f32 floor — the crate + the golden both run f32); (3) a scale-0
//! adapter is a bit-exact no-op; (4) scale-1 has a visible effect vs the no-adapter render.

use std::path::PathBuf;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_flux2::load_transformer;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("MLX_GEN_FLUX2_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").expect("HOME");
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-9b/snapshots");
    std::fs::read_dir(&snaps)
        .expect("snapshot dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tools/golden")
}

/// (1) The top-level `AdaptableHost` resolves every fork `Flux2LoRAMapping` diffusers target across
/// the real module tree (globals + 8 double blocks × 12 + 24 single blocks × 2), and rejects
/// off-surface paths (out-of-range blocks, klein-absent guidance linears, internal field names).
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    for p in [
        "x_embedder",
        "context_embedder",
        "proj_out",
        "norm_out.linear",
        "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.linear",
        "single_stream_modulation.linear",
        "time_guidance_embed.linear_1",
        "time_guidance_embed.linear_2",
    ] {
        assert!(resolves(&mut t, p), "global {p} should resolve");
    }
    let double_targets = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.linear_in",
        "ff.linear_out",
        "ff_context.linear_in",
        "ff_context.linear_out",
    ];
    for i in 0..8 {
        for tgt in double_targets {
            let p = format!("transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for i in 0..24 {
        for tgt in ["attn.to_qkv_mlp_proj", "attn.to_out"] {
            let p = format!("single_transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "transformer_blocks.8.attn.to_q", // out of range (8 double blocks: 0..7)
        "single_transformer_blocks.24.attn.to_out", // out of range (24 single blocks: 0..23)
        "time_guidance_embed.guidance_linear_1", // klein has no guidance embedding
        "transformer_blocks.0.attn.add_q", // internal field, not the file's add_q_proj
        "transformer_blocks.0.attn.qkv",  // not a FLUX.2 module
        "norm_out_linear",                // internal field name, not the dotted path
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ routing covers the full Flux2LoRAMapping surface (globals + 8×12 + 24×2) and rejects off-surface");
}

fn meta_u32(g: &Weights, k: &str) -> u32 {
    g.metadata(k).unwrap().parse().unwrap()
}

/// Render `flux2_klein_9b` txt2img with an optional adapter, at the golden's config.
fn render(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g =
        Weights::from_file(golden_dir().join(format!("flux2_{golden_kind}_golden.safetensors")))
            .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let (seed, steps) = (meta_u32(&g, "seed") as u64, meta_u32(&g, "steps"));
    let (w, h) = (meta_u32(&g, "width"), meta_u32(&g, "height"));

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
        }]);
    }
    let generator = mlx_gen::load("flux2_klein_9b", &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        ..Default::default()
    };
    match generator.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap().pixels,
        other => panic!("expected Images, got {other:?}"),
    }
}

fn px_gt8(a: &[u8], b: &[u8]) -> (usize, f64) {
    assert_eq!(a.len(), b.len(), "image size mismatch");
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count();
    (differ, differ as f64 / a.len() as f64 * 100.0)
}

/// (2) + (4): the public `load(adapter).generate()` render matches the fork golden (parity, below
/// the cross-build f32 floor) AND visibly differs from the no-adapter render (the adapter is real).
fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let adapter_file = format!("flux2_{kind}_adapter.safetensors");
    let pixels = render(Some((&adapter_file, my_kind, 1.0)), kind);
    let g =
        Weights::from_file(golden_dir().join(format!("flux2_{kind}_golden.safetensors"))).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let (differ, frac) = px_gt8(&pixels, &gimg.pixels);
    println!(
        "flux2 {kind} adapter render vs fork f32: {differ}/{} px>8 ({frac:.3}%)",
        pixels.len()
    );
    assert!(
        frac < 5.0,
        "flux2 {kind} adapter render diverges from the fork: {frac:.3}% px>8 (cross-build f32 floor is ~0.9%)"
    );

    // The adapter must actually change the image (guards a silently-dropped/no-op application).
    let base = render(None, kind);
    let (_, effect) = px_gt8(&pixels, &base);
    println!("flux2 {kind} adapter effect vs no-adapter: {effect:.2}% px>8");
    assert!(
        effect > 3.0,
        "flux2 {kind} adapter had no visible effect ({effect:.2}% px>8) — silently dropped?"
    );
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + adapter golden"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + adapter golden"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// (3) A scale-0 adapter is a bit-exact no-op vs the no-adapter render.
#[test]
#[ignore = "needs real FLUX.2-klein-9b weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render(None, "lora");
    let zero = render(
        Some(("flux2_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("flux2 scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}
