//! sc-2639: SDXL LoRA merge parity vs the vendored `lora.py` (`apply_loras_to_unet`).
//!
//! `#[ignore]`d — needs the real SDXL snapshot, a real kohya LoRA (`latent-consistency/lcm-lora-sdxl`
//! in the HF cache), and the golden from `tools/dump_sdxl_lora_golden.py`.
//! Run: cargo test -p mlx-gen-sdxl --release --test lora_real_weights -- --ignored --nocapture
//!
//! Gates:
//! - `routing_surface_is_vendored_515` — the `AdaptableHost` routes exactly the vendored reachable
//!   surface (515 module paths), and `mid_block` is unreachable (faithful to the vendored naming gap).
//! - `lora_merge_count_matches_vendored` — merging the LCM-LoRA touches exactly 515 modules (== the
//!   vendored `touched`), the rest surfaced as skipped.
//! - `lora_render_matches_vendored` — the public `generate()` render with the LoRA merged matches
//!   the vendored merged-UNet golden within the cross-build forward residual (~0.005% px>8). The
//!   merge itself is provably bit-exact: the dump verifies the f32→f16→f32 delta equals the vendored
//!   f16 matmul for all 515 modules, so the residual is the pmetal-vs-wheel f32 GEMM 1-ULP (the same
//!   class as base T2I, slightly amplified by the extra delta matmul through the ancestral sampler).
//! - `scale_zero_lora_is_bit_exact_noop` — a scale-0 LoRA leaves the render bit-identical to base.

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_sdxl::{apply_sdxl_adapters, load_unet};
// Force-link the provider so its `inventory::submit!` registers `"sdxl"`.
use mlx_gen_sdxl as _;

// Production runs fp16 (sc-2721); the render gate uses the `float16=True` vendored-merge golden,
// dumped on MLX 0.31.2. Build: FLOAT16=1 <mlx-0.31.2 python> tools/dump_sdxl_lora_golden.py
const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/sdxl_lora_fp16_golden.safetensors"
);

fn snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("SDXL_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

/// The LoRA path is recorded in the golden metadata (so the test always matches the dumped golden).
fn lora_spec(g: &Weights, scale: f32) -> AdapterSpec {
    AdapterSpec {
        path: PathBuf::from(g.metadata("lora_path").expect("golden lora_path")),
        scale,
        kind: AdapterKind::Lora,
    }
}

fn render(spec: &LoadSpec, g: &Weights) -> Image {
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();
    let model = mlx_gen::load("sdxl", spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        negative_prompt: Some(g.metadata("negative").unwrap().to_string()),
        width: w,
        height: h,
        seed: Some(g.metadata("seed").unwrap().parse().unwrap()),
        steps: Some(g.metadata("steps").unwrap().parse().unwrap()),
        guidance: Some(g.metadata("cfg").unwrap().parse().unwrap()),
        ..Default::default()
    };
    match model.generate(&req, &mut |_| {}).unwrap() {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    }
}

#[test]
#[ignore = "needs the real SDXL snapshot"]
fn routing_surface_is_vendored_515() {
    let unet = load_unet(&snapshot()).unwrap();
    let paths = unet.lora_target_paths();
    assert_eq!(
        paths.len(),
        515,
        "routable LoRA surface must equal the vendored reachable count (515)"
    );
    // mid_block must be unreachable (the vendored naming gap this port reproduces — sc-2671).
    assert!(!paths.iter().any(|p| p.starts_with("mid_block")));
    // A representative down/up attention + proj + time_emb path is present.
    for want in [
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0",
        "up_blocks.0.attentions.0.proj_in",
        "down_blocks.0.resnets.0.time_emb_proj",
    ] {
        assert!(paths.iter().any(|p| p == want), "missing target {want}");
    }
    println!(
        "✓ SDXL LoRA routing surface = {} modules (vendored 515)",
        paths.len()
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA in the HF cache"]
fn lora_merge_count_matches_vendored() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let mut unet = load_unet(&snapshot()).unwrap();
    let report = apply_sdxl_adapters(&mut unet, &[lora_spec(&g, 1.0)]).unwrap();
    let vendored: usize = g.metadata("touched").unwrap().parse().unwrap();
    assert_eq!(
        report.merged, vendored,
        "merged count {} must equal the vendored touched {vendored}",
        report.merged
    );
    assert_eq!(report.merged, 515);
    // Every LCM-LoRA key is accounted for: 788 modules × 3 (down/up/alpha) = 2364; 515 merged
    // (×3 classified) + the rest surfaced as skipped — nothing silently dropped.
    assert_eq!(report.merged * 3 + report.skipped_keys, 2364);
    println!(
        "✓ LoRA merge: {} merged (== vendored {vendored}), {} keys skipped (surfaced)",
        report.merged, report.skipped_keys
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA + lora golden"]
fn lora_render_matches_vendored() {
    // The golden is the **vendored** 515-module merge; `model::load` defaults to the COMPLETE 809
    // surface (sc-2671), so opt into the vendored surface to compare apples-to-apples. (Run this
    // `#[ignore]` real-weights test on its own — it sets a process-global env.)
    std::env::set_var("SDXL_LORA_VENDORED", "1");
    let g = Weights::from_file(GOLDEN).unwrap();
    let spec =
        LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![lora_spec(&g, 1.0)]);
    let img = render(&spec, &g);
    std::env::remove_var("SDXL_LORA_VENDORED");

    let gpix: Vec<u8> = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();
    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden/rust_sdxl_lora.png");
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let px8 = img
        .pixels
        .iter()
        .zip(&gpix)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count() as f32
        / img.pixels.len() as f32;
    println!(
        "✓ LoRA render {}x{}: {:.3}% px>8 vs the vendored merge; saved {}",
        img.width,
        img.height,
        px8 * 100.0,
        out_path.display()
    );
    // The merge deltas are bit-exact to the vendored across all 515 modules (proven in the dump),
    // so this is the cross-build f32 forward residual (~0.005% px>8 measured), not a merge bug. Gate
    // generously below 0.1% — far under the 1% "real divergence" line; the merge math is exact.
    assert!(
        px8 < 0.001,
        "SDXL LoRA render diverged beyond the cross-build residual: {:.3}% px>8",
        px8 * 100.0
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot"]
fn peft_format_merges_end_to_end() {
    use mlx_rs::{Array, Dtype};

    // A real PEFT-format file (`base_model.model.unet.<dotted>.lora_A.default.weight`) targeting three
    // self-attention projections (all 640×640) of one block. Proves the PEFT classify→route→merge
    // path end-to-end on real weights (kohya validates the shared merge math; this validates the PEFT
    // key form, incl. the `.default` infix).
    let (rank, dim) = (4i32, 640i32);
    let a = Array::from_slice(&vec![0.01f32; (rank * dim) as usize], &[rank, dim])
        .as_dtype(Dtype::Float16)
        .unwrap();
    let b = Array::from_slice(&vec![0.02f32; (dim * rank) as usize], &[dim, rank])
        .as_dtype(Dtype::Float16)
        .unwrap();
    let alpha = Array::from_slice(&[4.0f32], &[1])
        .as_dtype(Dtype::Float16)
        .unwrap();
    let base = "base_model.model.unet.down_blocks.1.attentions.0.transformer_blocks.0.attn1";
    let mut names: Vec<String> = Vec::new();
    for leaf in ["to_q", "to_k", "to_v"] {
        names.push(format!("{base}.{leaf}.lora_A.default.weight"));
        names.push(format!("{base}.{leaf}.lora_B.default.weight"));
        names.push(format!("{base}.{leaf}.alpha"));
    }
    let arrs = [&a, &b, &alpha];
    let tensors: Vec<(&str, &Array)> = names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), arrs[i % 3]))
        .collect();
    let dir = std::env::temp_dir().join("mlx_gen_sdxl_peft_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("peft_lora.safetensors");
    Array::save_safetensors(tensors, None, &path).unwrap();

    let mut unet = load_unet(&snapshot()).unwrap();
    let report = apply_sdxl_adapters(
        &mut unet,
        &[AdapterSpec {
            path,
            scale: 1.0,
            kind: AdapterKind::Lora,
        }],
    )
    .unwrap();
    assert_eq!(
        report.merged, 3,
        "PEFT file should merge its 3 target modules"
    );
    assert_eq!(report.skipped_keys, 0, "all PEFT keys are reachable");
    println!(
        "✓ PEFT-format LoRA ({}.default.weight) merged 3/3 end-to-end",
        base
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + LCM-LoRA"]
fn scale_zero_lora_is_bit_exact_noop() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let base = render(&LoadSpec::new(WeightsSource::Dir(snapshot())), &g);
    let zero = render(
        &LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![lora_spec(&g, 0.0)]),
        &g,
    );
    let differ = base
        .pixels
        .iter()
        .zip(&zero.pixels)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differ, 0,
        "scale-0 LoRA must be a bit-exact no-op ({differ} px differ)"
    );
    println!("✓ scale-0 LoRA is a bit-exact no-op");
}
