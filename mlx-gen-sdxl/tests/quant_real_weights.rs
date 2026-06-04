//! sc-2641: SDXL Q4/Q8 quantization. The vendored SDXL path runs fp16 and doesn't quantize, so the
//! reference is a vendored-equivalent `nn.quantize` (bf16-cast, Linear-only, group 64) over the same
//! scope Rust uses (UNet + both CLIP encoders; VAE stays f32).
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the goldens from `tools/dump_sdxl_quant_golden.py`.
//! Run: cargo test -p mlx-gen-sdxl --release --test quant_real_weights -- --ignored --nocapture
//!
//! Gates: (1) **scales byte-match** — the loaded Q8/Q4 `wq`/`scales`/`biases` of a real
//! `down_blocks.1…attn1.to_q` are bit-exact to `mx.quantize(bf16_weight)`, proving the bf16-cast
//! packing is correct on **base-1.0** (the sc-1975 "Q8 broken on base-1.0" root cause — sc-2604).
//! (2)+(3) **render parity** — the full `load(Q).generate()` matches the vendored-equivalent Q8/Q4
//! render (tight: both quantize identically, so the chaos sampler stays on one trajectory).

use std::path::PathBuf;

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, Image, LoadSpec, Quant, WeightsSource};
use mlx_gen_sdxl as _;
use mlx_gen_sdxl::load_unet;
use mlx_rs::ops::eq;
use mlx_rs::{Array, Dtype};

fn golden(name: &str) -> Weights {
    Weights::from_file(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../tools/golden")
            .join(name),
    )
    .unwrap()
}

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

fn bf16(a: &Array) -> Array {
    a.as_dtype(Dtype::Bfloat16).unwrap()
}

fn all_eq(a: &Array, b: &Array) -> bool {
    a.shape() == b.shape() && eq(a, b).unwrap().all(None).unwrap().item::<bool>()
}

#[test]
#[ignore = "needs the real SDXL snapshot + quant scales golden"]
fn q8_q4_scales_byte_match() {
    let g = golden("sdxl_quant_scales_ref.safetensors");
    let probe: Vec<&str> = g.metadata("probe_path").unwrap().split('.').collect();

    for (bits, q) in [(8, "q8"), (4, "q4")] {
        let mut unet = load_unet(&snapshot()).unwrap();
        unet.quantize(bits).unwrap();
        let lin =
            AdaptableHost::adaptable_mut(&mut unet, &probe).expect("probe module is adaptable");
        let (wq, scales, biases, _bias, gs, b) =
            lin.quantized_params().expect("probe is quantized");
        assert_eq!(gs, 64);
        assert_eq!(b, bits);
        // The fork-equivalent reference is `mx.quantize(bf16_weight, 64, bits)` — bit-exact.
        assert!(
            all_eq(wq, g.require(&format!("wq_{q}")).unwrap()),
            "Q{bits} wq not byte-identical to mx.quantize(bf16)"
        );
        assert!(
            all_eq(scales, &bf16(g.require(&format!("scales_{q}")).unwrap())),
            "Q{bits} scales not byte-identical (the sc-1975/sc-2604 f32-checkpoint scales drift)"
        );
        assert!(
            all_eq(biases, &bf16(g.require(&format!("biases_{q}")).unwrap())),
            "Q{bits} biases not byte-identical"
        );
        println!(
            "✓ Q{bits} loaded scales/wq/biases byte-identical to mx.quantize(bf16) on base-1.0"
        );
    }
}

fn render_quant(q: Quant, g: &Weights) -> Image {
    let spec = LoadSpec::new(WeightsSource::Dir(snapshot())).with_quant(q);
    let model = mlx_gen::load("sdxl", &spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        negative_prompt: Some(g.metadata("negative").unwrap().to_string()),
        width: g.metadata("w").unwrap().parse().unwrap(),
        height: g.metadata("h").unwrap().parse().unwrap(),
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

fn px8(img: &Image, g: &Weights) -> f32 {
    let gpix: Vec<u8> = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();
    img.pixels
        .iter()
        .zip(&gpix)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count() as f32
        / img.pixels.len() as f32
}

fn render_gate(q: Quant, name: &str, tag: &str) {
    let g = golden(name);
    let img = render_quant(q, &g);
    let p = px8(&img, &g);
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/rust_sdxl_{tag}.png"));
    image::save_buffer(
        &out,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    println!(
        "✓ {tag} render {}x{}: {:.3}% px>8 vs the vendored-equivalent reference",
        img.width,
        img.height,
        p * 100.0
    );
    // Both engines quantize identically (scales byte-match, above), so the quantized forward is the
    // same trajectory — only the cross-build f32 residual remains (like base T2I).
    assert!(
        p < 0.001,
        "SDXL {tag} render diverged from the quant reference: {:.3}% px>8",
        p * 100.0
    );
}

#[test]
#[ignore = "needs the real SDXL snapshot + Q8 golden"]
fn q8_render_matches_reference() {
    render_gate(Quant::Q8, "sdxl_q8_fp16_golden.safetensors", "q8");
}

#[test]
#[ignore = "needs the real SDXL snapshot + Q4 golden"]
fn q4_render_matches_reference() {
    render_gate(Quant::Q4, "sdxl_q4_fp16_golden.safetensors", "q4");
}
