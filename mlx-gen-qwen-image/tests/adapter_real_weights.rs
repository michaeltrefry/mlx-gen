//! sc-2528: end-to-end Qwen-Image LoRA/LoKr adapter consumption against real weights.
//!
//! `#[ignore]`d — needs the real `Qwen/Qwen-Image` snapshot in the HF cache (env
//! `QWEN_IMAGE_SNAPSHOT`) and the adapter goldens from `tools/dump_qwen_adapter_golden.py`
//! (gitignored, local). Run:
//!   cargo test -p mlx-gen-qwen-image --release --test adapter_real_weights -- --ignored --nocapture
//!
//! Gates: (1) the key→module map resolves the FULL fork `QwenLoRAMapping` surface (60 blocks ×
//! attention + img/txt MLP) against the real module tree; (2) the public
//! `load(spec.with_adapters(…)).generate()` render matches the fork's LoRA *and* LoKr golden
//! (px>8); (3) a scale-0 adapter is a bit-exact no-op.

use std::path::PathBuf;

use mlx_gen::adapters::{AdaptableHost, Adapter};
use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, LoadSpec, WeightsSource,
};
use mlx_gen_qwen_image::{apply_qwen_adapters, decoded_to_image, loader};
use mlx_rs::ops::array_eq;
use mlx_rs::Array;

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("QWEN_IMAGE_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen-Image/snapshots");
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

/// Locate a file inside an HF-cache repo's `snapshots/<hash>/` dir (the first snapshot that has it).
fn hf_cache_file(repo_dir: &str, filename: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = PathBuf::from(home).join(format!(".cache/huggingface/hub/{repo_dir}/snapshots"));
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join(filename))
        .find(|p| p.exists())
}

/// The cached lightx2v Lightning LoRAs to probe — the 8-step T2I and Edit-2511 variants (the 4-step
/// variants share the identical per-block key structure). Skips any that aren't cached.
fn lightning_loras() -> Vec<(&'static str, PathBuf)> {
    let mut v = Vec::new();
    if let Some(p) = hf_cache_file(
        "models--lightx2v--Qwen-Image-Lightning",
        "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors",
    ) {
        v.push(("qwen-image-lightning-8step", p));
    }
    if let Some(p) = hf_cache_file(
        "models--lightx2v--Qwen-Image-Edit-2511-Lightning",
        "Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
    ) {
        v.push(("qwen-image-edit-2511-lightning-8step", p));
    }
    v
}

/// sc-2909: the acceleration-LoRA **viability** gate the story is gated on. The real lightx2v
/// Lightning LoRAs (T2I `Qwen-Image-Lightning` + `Qwen-Image-Edit-2511-Lightning`) load through
/// `apply_qwen_adapters` with **zero silent drops** — every one of the 720 per-block targets
/// (60 blocks × 12 modules: joint-attention q/k/v/out + add_q/k/v/to_add_out + img/txt MLP in/out)
/// resolves on the real 60-block tree. The merge math itself is the sc-2528 seam (already proven
/// bit-exact); this just confirms the Lightning files address modules the host map reaches.
#[test]
#[ignore = "needs real Qwen-Image weights + the cached lightx2v Lightning LoRAs"]
fn lightning_loras_apply_cleanly() {
    let loras = lightning_loras();
    assert!(
        !loras.is_empty(),
        "no cached Lightning LoRAs found — download e.g. lightx2v/Qwen-Image-Lightning"
    );
    for (label, path) in loras {
        // Fresh transformer per LoRA so each report reflects that file alone (no stacking).
        let mut t = loader::load_transformer(&snapshot()).unwrap();
        let report = apply_qwen_adapters(
            &mut t,
            &[AdapterSpec::new(path.clone(), 1.0, AdapterKind::Lora)],
        )
        .unwrap_or_else(|e| panic!("{label} ({}) failed to apply: {e}", path.display()));
        println!(
            "{label}: applied {} module(s), unmatched {:?}",
            report.applied, report.unmatched_paths
        );
        assert!(
            report.unmatched_paths.is_empty(),
            "{label}: {} unmatched target(s)",
            report.unmatched_paths.len()
        );
        assert_eq!(
            report.applied, 720,
            "{label}: expected 720 per-block modules (60 blocks × 12)"
        );
    }
}

/// (1) The top-level `AdaptableHost` resolves every fork `QwenLoRAMapping` target (all per-block:
/// the joint attention + the two stream MLPs; no globals) across the real 60-block tree, and
/// rejects off-surface paths.
#[test]
#[ignore = "needs real Qwen-Image weights"]
fn routing_map_covers_full_fork_surface() {
    let mut t = loader::load_transformer(&snapshot()).unwrap();
    let resolves = |t: &mut _, p: &str| -> bool {
        let segs: Vec<&str> = p.split('.').collect();
        AdaptableHost::adaptable_mut(t, &segs).is_some()
    };

    let targets = [
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_q_proj",
        "attn.add_k_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ];
    for i in 0..60 {
        for tgt in targets {
            let p = format!("transformer_blocks.{i}.{tgt}");
            assert!(resolves(&mut t, &p), "expected {p} to resolve");
        }
    }
    for p in [
        "transformer_blocks.60.attn.to_q",    // out of range
        "transformer_blocks.0.attn.to_out",   // missing .0
        "transformer_blocks.0.attn.add_q",    // internal name, not the file's add_q_proj
        "transformer_blocks.0.img_mlp.net.1", // gelu slot
        "img_in",                             // not a trained target
    ] {
        assert!(!resolves(&mut t, p), "expected {p} NOT to resolve");
    }
    println!("✓ routing map covers the full fork QwenLoRAMapping surface (60 blocks × 12 targets)");
}

fn render(adapter: Option<(&str, AdapterKind, f32)>, golden_kind: &str) -> Vec<u8> {
    let g = Weights::from_file(golden_dir().join(format!("qwen_{golden_kind}_golden.safetensors")))
        .unwrap();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("width").unwrap().parse().unwrap();
    let h: u32 = g.metadata("height").unwrap().parse().unwrap();

    let mut spec = LoadSpec::new(WeightsSource::Dir(snapshot()));
    if let Some((file, kind, scale)) = adapter {
        spec = spec.with_adapters(vec![AdapterSpec {
            path: golden_dir().join(file),
            scale,
            kind,
            pass_scales: None,
            moe_expert: None,
        }]);
    }
    let generator = mlx_gen::load("qwen_image", &spec).unwrap();
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

/// Count RGB8 pixels differing by >8 between two buffers.
fn px_gt8(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > 8)
        .count()
}

/// The base (no-adapter) render's px>8 vs the fork base golden, at the SAME config as the `kind`
/// adapter golden — the inherited bf16 toolchain drift floor the adapter render sits on. The base
/// golden (`qwen_image_golden.safetensors`) MUST be dumped at the adapter golden's
/// (seed, steps, size); a mismatch is a hard error (it would yield a bogus floor).
fn base_floor_px(kind: &str) -> usize {
    let ag =
        Weights::from_file(golden_dir().join(format!("qwen_{kind}_golden.safetensors"))).unwrap();
    let bg = Weights::from_file(golden_dir().join("qwen_image_golden.safetensors"))
        .expect("base golden qwen_image_golden.safetensors (dump it at the adapter config)");
    for k in ["seed", "steps", "width", "height"] {
        assert_eq!(
            ag.metadata(k),
            bg.metadata(k),
            "base golden {k} != adapter golden {k} — regenerate qwen_image_golden.safetensors at the adapter config"
        );
    }
    let pixels = render(None, kind);
    let bimg = decoded_to_image(bg.require("decoded").unwrap()).unwrap();
    px_gt8(&pixels, &bimg.pixels)
}

fn assert_matches_golden(kind: &str, my_kind: AdapterKind) {
    let pixels = render(
        Some((&format!("qwen_{kind}_adapter.safetensors"), my_kind, 1.0)),
        kind,
    );
    let g =
        Weights::from_file(golden_dir().join(format!("qwen_{kind}_golden.safetensors"))).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = px_gt8(&pixels, &gimg.pixels);

    // Floor-relative gate (sc-2718): the adapter render must not diverge from the fork by materially
    // more than the BASE render does — the inherited bf16 toolchain drift floor — i.e. the adapter
    // itself adds ~zero divergence (the residual is fork-faithful; scale-0 is bit-exact). The
    // `2×floor + 0.5%` cap allows the stronger-perturbation adapter image's larger content floor
    // while staying FAR tighter than a flat %. Measured @512²: Qwen base 0.05% / LoRA 0.02% /
    // LoKr 0.02%. (Replaces the old flat 5% guard, which was sized for the inflated 256² floor —
    // a small latent lets bf16 drift flip a large *fraction* of the few pixels; the floor collapses
    // ~30× at 512², so the goldens are now dumped at 512².)
    let base = base_floor_px(kind);
    let cap = base * 2 + pixels.len() / 200;
    let pct = |n: usize| n as f64 / pixels.len() as f64 * 100.0;
    println!(
        "✓ qwen {kind} adapter render: {differ} px>8 ({:.4}%); base floor {base} ({:.4}%); cap {cap} ({:.4}%)",
        pct(differ),
        pct(base),
        pct(cap),
    );
    assert!(
        differ <= cap,
        "qwen {kind} adapter render diverges beyond the base floor: {differ} px ({:.3}%) > cap {cap} px (base {base})",
        pct(differ),
    );
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter & base goldens (same config)"]
fn lora_render_matches_fork_golden() {
    assert_matches_golden("lora", AdapterKind::Lora);
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter & base goldens (same config)"]
fn lokr_render_matches_fork_golden() {
    assert_matches_golden("lokr", AdapterKind::Lokr);
}

/// Diagnostic (per the divergence-is-not-rounding rule): the Rust base render (no adapter) vs the
/// fork base golden at the adapter golden's config — the floor the LoRA/LoKr px>8 numbers inherit
/// and the `assert_matches_golden` cap is derived from. Needs `qwen_image_golden.safetensors`
/// dumped at the adapter config (`tools/dump_qwen_image_golden.py`; default 512²).
#[test]
#[ignore = "needs real Qwen-Image weights + base golden at the adapter config"]
fn base_render_drift_attributes_adapter_gap() {
    // The inherited bf16 toolchain floor (also folded into assert_matches_golden's cap).
    println!(
        "qwen base (no adapter) vs fork base: {} px>8",
        base_floor_px("lora")
    );
}

#[test]
#[ignore = "needs real Qwen-Image weights + adapter golden"]
fn scale_zero_adapter_is_noop() {
    let base = render(None, "lora");
    let zero = render(
        Some(("qwen_lora_adapter.safetensors", AdapterKind::Lora, 0.0)),
        "lora",
    );
    let differ = base.iter().zip(&zero).filter(|(a, b)| a != b).count();
    println!("✓ qwen scale-0 adapter no-op: {differ} px differ from the no-adapter render");
    assert_eq!(differ, 0, "scale-0 adapter must be a bit-exact no-op");
}

/// The single installed LoRA's `(a, b)` arrays, or panic.
fn lora_arrays(adapters: &[Adapter]) -> (Array, Array) {
    match adapters {
        [Adapter::Lora { a, b, .. }] => (a.clone(), b.clone()),
        _ => panic!("expected exactly one LoRA adapter, got {}", adapters.len()),
    }
}

/// sc-2618: a kohya `lora_unet_` file resolves the SAME modules and installs the byte-identical
/// adapter as the equivalent PEFT file, on the REAL Qwen 60-block tree. Drift guard (every kohya
/// path resolves), collision-free flattening, and a fused/off-surface key errors loudly.
#[test]
#[ignore = "needs real Qwen-Image weights"]
fn kohya_matches_peft_on_real_tree() {
    let none = None as Option<&std::collections::HashMap<String, String>>;
    let mut probe = loader::load_transformer(&snapshot()).unwrap();
    let paths = probe.adaptable_paths();
    assert!(!paths.is_empty(), "no kohya targets enumerated");
    for p in &paths {
        let segs: Vec<&str> = p.split('.').collect();
        assert!(
            AdaptableHost::adaptable_mut(&mut probe, &segs).is_some(),
            "drift: enumerated {p} does not resolve via adaptable_mut"
        );
    }
    let flat: std::collections::BTreeSet<String> =
        paths.iter().map(|p| p.replace('.', "_")).collect();
    assert_eq!(
        flat.len(),
        paths.len(),
        "two paths collide when flattened to a kohya stem"
    );

    // One on-disk spelling per module (drop a `.0` alias when its bare sibling is enumerated; a
    // no-op for Qwen, which has no aliases).
    let targets: Vec<String> = paths
        .iter()
        .filter(|p| match p.strip_suffix(".0") {
            Some(base) => !paths.iter().any(|q| q.as_str() == base),
            None => true,
        })
        .cloned()
        .collect();

    let r = 2i32;
    let mut kohya: Vec<(String, Array)> = Vec::new();
    let mut peft: Vec<(String, Array)> = Vec::new();
    for p in &targets {
        let segs: Vec<&str> = p.split('.').collect();
        let shape = AdaptableHost::adaptable_mut(&mut probe, &segs)
            .unwrap()
            .base_shape();
        let (out, inp) = (shape[0], shape[1]);
        let a = Array::from_slice(
            &(0..r * inp)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.001)
                .collect::<Vec<_>>(),
            &[r, inp],
        );
        let b = Array::from_slice(
            &(0..out * r)
                .map(|i| ((i % 11) as f32 - 5.0) * 0.001)
                .collect::<Vec<_>>(),
            &[out, r],
        );
        let alpha = Array::from_slice(&[4.0f32], &[1]);
        let stem = p.replace('.', "_");
        kohya.push((format!("lora_unet_{stem}.lora_down.weight"), a.clone()));
        kohya.push((format!("lora_unet_{stem}.lora_up.weight"), b.clone()));
        kohya.push((format!("lora_unet_{stem}.alpha"), alpha.clone()));
        peft.push((format!("transformer.{p}.lora_A.weight"), a));
        peft.push((format!("transformer.{p}.lora_B.weight"), b));
        peft.push((format!("transformer.{p}.alpha"), alpha));
    }
    let dir = std::env::temp_dir().join("mlx_gen_qwen_kohya_test");
    std::fs::create_dir_all(&dir).unwrap();
    let (kpath, ppath) = (dir.join("kohya.safetensors"), dir.join("peft.safetensors"));
    Array::save_safetensors(
        kohya
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &kpath,
    )
    .unwrap();
    Array::save_safetensors(
        peft.iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect::<Vec<_>>(),
        none,
        &ppath,
    )
    .unwrap();

    let mut tk = loader::load_transformer(&snapshot()).unwrap();
    let rk = apply_qwen_adapters(
        &mut tk,
        &[AdapterSpec {
            path: kpath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();
    assert_eq!(rk.applied, targets.len(), "kohya: not all targets applied");
    assert!(
        rk.unmatched_paths.is_empty(),
        "kohya unmatched: {:?}",
        rk.unmatched_paths
    );

    let mut tp = loader::load_transformer(&snapshot()).unwrap();
    apply_qwen_adapters(
        &mut tp,
        &[AdapterSpec {
            path: ppath,
            scale: 0.8,
            kind: AdapterKind::Lora,
            pass_scales: None,
            moe_expert: None,
        }],
    )
    .unwrap();

    for p in &targets {
        let segs: Vec<&str> = p.split('.').collect();
        let (ka, kb) = lora_arrays(
            AdaptableHost::adaptable_mut(&mut tk, &segs)
                .unwrap()
                .adapters(),
        );
        let (pa, pb) = lora_arrays(
            AdaptableHost::adaptable_mut(&mut tp, &segs)
                .unwrap()
                .adapters(),
        );
        assert!(
            array_eq(&ka, &pa, false).unwrap().item::<bool>()
                && array_eq(&kb, &pb, false).unwrap().item::<bool>(),
            "kohya and peft installed different adapters at {p}"
        );
    }
    println!(
        "✓ kohya ≡ peft across {} Qwen modules (byte-identical adapters)",
        targets.len()
    );

    // A fused/off-surface kohya key (Qwen has no fused `attn.qkv`) is surfaced and errors.
    let small = Array::from_slice(&[0.01f32], &[1, 1]);
    let opath = dir.join("kohya_offsurface.safetensors");
    Array::save_safetensors(
        vec![
            (
                "lora_unet_transformer_blocks_0_attn_qkv.lora_down.weight",
                &small,
            ),
            (
                "lora_unet_transformer_blocks_0_attn_qkv.lora_up.weight",
                &small,
            ),
        ],
        none,
        &opath,
    )
    .unwrap();
    let mut to = loader::load_transformer(&snapshot()).unwrap();
    assert!(
        apply_qwen_adapters(
            &mut to,
            &[AdapterSpec {
                path: opath,
                scale: 1.0,
                kind: AdapterKind::Lora,
                pass_scales: None,
                moe_expert: None,
            }],
        )
        .is_err(),
        "an off-surface kohya key must error, not silently drop"
    );
}
