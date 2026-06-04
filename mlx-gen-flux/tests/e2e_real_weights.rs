//! sc-2345: end-to-end parity of the FLUX.1 port against a real-weights fork golden.
//!
//! `#[ignore]`d — needs the real `black-forest-labs/FLUX.1-{schnell,dev}` weights in the HF cache
//! and the golden produced by `tools/dump_flux_golden.py` (gitignored, local). `FLUX_VARIANT=dev`
//! (default schnell) selects the variant for both the dumper and this harness — the golden path,
//! model id, guidance, mu-shift, and T5 seq-length all follow. Run with:
//!   FLUX_VARIANT=dev MLX_GEN_FLUX_SNAPSHOT=<matching snapshot> \
//!     cargo test -p mlx-gen-flux --test e2e_real_weights -- --ignored --nocapture
//!
//! Stage tests feed the fork's own intermediates into each Rust stage to isolate it; the final
//! test drives the public `load(id, spec).generate(req)` API and compares the rendered image to
//! the fork's golden (px>8 fraction — the repo's parity bar, like the Z-Image/Qwen e2e tests).

use std::path::PathBuf;

use mlx_gen::image::decoded_to_image;
use mlx_gen::weights::Weights;
use mlx_gen::{GenerationOutput, GenerationRequest, LoadSpec, Progress, Quant, WeightsSource};
use mlx_gen_flux::{
    build_linear_sigmas, load_clip_encoder, load_t5_encoder, load_transformer, load_vae,
    unpack_latents, FluxVariant,
};
use mlx_rs::ops::{add, multiply};
use mlx_rs::{Array, Dtype};

/// Q8/Q4 verification. TWO checks:
/// (a) the quant GATE — feed the fork-Q golden's OWN embeds+init into the Rust transformer.quantize,
///     run the denoise on the fork's sigmas, and compare v0 to the fork-Q golden (isolates the
///     quantized transformer from the 256² sampler chaos);
/// (b) full public `load(spec.with_quant(Q)).generate()` render, saved for visual inspection.
/// The bf16 (non-quant) FLUX path is now pixel-parity (sc-2787), so the quant FULL-GENERATE residual
/// (schnell ~12% / dev <1% px>8 @256²) is NOT a quant bug — it's the core `adapters.rs` bf16→f32
/// activation upcast (the now-unneeded sc-2772 workaround) running the quantized *text-embedder* `qmm`
/// at f32 vs the fork's bf16, a ~1-bf16-ULP flip in the conditioning that the chaotic sampler
/// amplifies. `quantized_matmul` itself is fp32-accumulated/correct. v0 is the gate; removing that
/// upcast core-wide is sc-2719.
fn verify_quant(quant: Quant, bits: i32) {
    // bf16 quantized reference (QUANTIZE=N, no FLUX_PRECISION). Post-sc-2787 the Rust transformer
    // matches the fork's MIXED precision: the conditioning/modulation path (incl. the dev guidance
    // term `time_proj(guidance*1000)`) now ROUNDS in bf16 exactly like the fork, so the honest
    // reference is the production bf16 quantized golden. (Pre-2787 the Rust ran f32 conditioning, so
    // the gate used an `_f32`-precision Q golden to avoid conflating quant with the fork's bf16
    // modulation precision; that rationale is now obsolete — both sides round the modulation in bf16.)
    let g = Weights::from_file(golden_path(&format!("_q{bits}"))).unwrap();
    let stored: i32 = g.metadata("quantize").unwrap().parse().unwrap();
    assert_eq!(stored, bits, "golden dumped at a different bit-width");
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();
    let snap = snapshot();

    // (a) quant-transformer gate
    let mut t = load_transformer(&snap, variant()).unwrap();
    t.quantize(bits).unwrap();
    let vae = load_vae(&snap).unwrap();
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let steps = sigmas.len() - 1;
    let mut latents = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    let guid = guidance(&g); // 0.0 schnell / 3.5 dev — must match the golden's render
                             // Localizer (informational): the quantized substages vs the fork-Q golden's. text_embeddings0 is
                             // the modulation (quantized timestep + guidance + text embedders); hidden0/encoder0 the input
                             // embedders; single_img the full step-0 transformer. The Q golden's substages were dumped from
                             // the fork's quantized transformer (f32-precision), isolating quant from the bf16 activation path.
    for (name, arr) in t
        .forward_capture(&latents, &pe, &pooled, sigmas[0], guid, w, h)
        .unwrap()
    {
        if let Ok(gold) = g.require(&name) {
            println!(
                "  Q{bits} substage {name}: mean_rel={:.3e} peak_rel={:.3e}",
                mean_abs_rel(&f32a(&arr), gold),
                peak_rel(&f32a(&arr), gold)
            );
        }
    }
    // v0 = the quantized transformer's first-step velocity. This is the GATE: it verifies the
    // quantized transformer in isolation from the 256² sampler chaos that amplifies the (intentional)
    // sc-2604 bf16-vs-f32 scale difference over the denoise steps. Across schnell/dev × Q4/Q8 a
    // correct quant lands v0 ≤ ~3.1e-2; the 20-step final latents drift to 6e-2–1.6e-1 purely from
    // that chaotic amplification (NOT a quant defect) so they stay informational below.
    let mut v0_mr = f32::NAN;
    for i in 0..steps {
        let v = t
            .forward(&latents, &pe, &pooled, sigmas[i], guid, w, h)
            .unwrap();
        if i == 0 {
            v0_mr = mean_abs_rel(&f32a(&v), g.require("v0").unwrap());
            let v0_pr = peak_rel(&f32a(&v), g.require("v0").unwrap());
            println!("Q{bits} v0 vs fork-Q v0: mean_rel={v0_mr:.3e} peak_rel={v0_pr:.3e}");
        }
        let dt = sigmas[i + 1] - sigmas[i];
        latents = add(
            &latents,
            multiply(&v, Array::from_slice(&[dt], &[1])).unwrap(),
        )
        .unwrap();
    }
    let golden_lat = g.require("final_latents").unwrap();
    let lat_mr = mean_abs_rel(&f32a(&latents), golden_lat);
    let unpacked = unpack_latents(&latents, w, h).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "Q{bits} transformer gate (fork-Q embeds+init): v0 mean_rel={v0_mr:.3e} | 20-step latents mean_rel={lat_mr:.3e} (chaos-amplified)  decoded px>8={:.2}% vs fork-Q{bits}",
        frac * 100.0
    );
    // `quantized_matmul` is fp32-accumulated (correct on the NAX build); the quantized transformer's
    // step-0 velocity must match the fork's quantized transformer (isolated from sampler chaos).
    assert!(
        v0_mr < 6e-2,
        "Q{bits} quant transformer diverged at step 0: v0 mean_rel {v0_mr:.3e}"
    );

    // (b) full public quantized generate — coherence + save PNG
    let spec = LoadSpec::new(WeightsSource::Dir(snap)).with_quant(quant);
    let gen = mlx_gen::load(variant().id(), &spec).unwrap();
    let req = GenerationRequest {
        prompt: g.metadata("prompt").unwrap().to_string(),
        width: w,
        height: h,
        seed: Some(g.metadata("seed").unwrap().parse().unwrap()),
        steps: Some(steps as u32),
        ..Default::default()
    };
    let out = gen.generate(&req, &mut |_| {}).unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../tools/golden/rust_flux_{}_q{bits}.png",
        variant_slug()
    ));
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "Q{bits} full generate: {:.2}% px>8 vs fork-Q{bits} (incl. NAX build delta); saved {}",
        100.0 * differ as f32 / img.pixels.len() as f32,
        out_path.display()
    );
}

#[test]
#[ignore = "needs real FLUX.1 weights + bf16 Q8 golden (QUANTIZE=8)"]
fn e2e_q8_matches_fork() {
    verify_quant(Quant::Q8, 8);
}

#[test]
#[ignore = "needs real FLUX.1 weights + bf16 Q4 golden (QUANTIZE=4)"]
fn e2e_q4_matches_fork() {
    verify_quant(Quant::Q4, 4);
}

/// Which FLUX.1 variant this run targets: `FLUX_VARIANT=dev` → dev, else schnell. The golden file
/// names, the registered model id, the guidance/sigma-shift, and the T5 seq-length all follow.
fn variant() -> FluxVariant {
    match std::env::var("FLUX_VARIANT").as_deref() {
        Ok("dev") => FluxVariant::Dev,
        _ => FluxVariant::Schnell,
    }
}

fn variant_slug() -> &'static str {
    match variant() {
        FluxVariant::Schnell => "schnell",
        FluxVariant::Dev => "dev",
    }
}

/// `tools/golden/flux1_<variant><suffix>_golden.safetensors`. suffix: `""` bf16, `"_f32"` f32 ref,
/// `"_q8"`/`"_q4"` quantized.
fn golden_path(suffix: &str) -> String {
    format!(
        "{}/../tools/golden/flux1_{}{}_golden.safetensors",
        env!("CARGO_MANIFEST_DIR"),
        variant_slug(),
        suffix
    )
}

/// The fork golden to compare against. Post-sc-2787 the mlx-gen FLUX path matches the fork's
/// MIXED-precision reference (f32 latents/main-stream/T5, bf16 CLIP + conditioning), so the parity
/// target is the production **bf16** golden (the default, dumped with no `FLUX_PRECISION`). Set
/// `FLUX_GOLDEN=f32` to compare against the all-f32 reference instead (a diagnostic, not the target).
fn golden() -> Weights {
    let suffix = match std::env::var("FLUX_GOLDEN").as_deref() {
        Ok("f32") => "_f32",
        _ => "",
    };
    Weights::from_file(golden_path(suffix)).unwrap()
}

/// (width, height) from the golden metadata.
fn wh(g: &Weights) -> (u32, u32) {
    (
        g.metadata("w").unwrap().parse().unwrap(),
        g.metadata("h").unwrap().parse().unwrap(),
    )
}

/// Classifier-free guidance the golden was rendered with (0.0 for schnell, 3.5 for dev).
fn guidance(g: &Weights) -> f32 {
    g.metadata("guidance")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0.0)
}

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("MLX_GEN_FLUX_SNAPSHOT")
            .expect("set MLX_GEN_FLUX_SNAPSHOT to the matching FLUX.1 snapshot directory"),
    )
}

/// Peak-relative error `max|a-b| / max|b|` — the meaningful metric vs a bf16 golden.
fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let max_diff = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    max_diff / peak
}

fn mean_abs_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.reshape(&[n]).unwrap();
    let b = b.reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let mabs: f32 = b.iter().map(|y| y.abs()).sum::<f32>() / b.len() as f32;
    let md: f32 = a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    md / mabs
}

fn f32a(a: &Array) -> Array {
    a.as_dtype(Dtype::Float32).unwrap()
}

#[test]
#[ignore = "needs real FLUX.1 weights + local golden"]
fn e2e_tokenizer_matches_golden() {
    // The full pipeline TOKENIZES the prompt; every other test feeds the golden ids. This isolates
    // whether the Rust tokenizer reproduces the fork's t5/clip input_ids (sc-2787 full-pipeline gap).
    let g = golden();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let t5_tok = mlx_gen_flux::load_t5_tokenizer(&snapshot(), variant()).unwrap();
    let clip_tok = mlx_gen_flux::load_clip_tokenizer(&snapshot()).unwrap();
    let t5_ids = t5_tok.tokenize(&prompt).unwrap().input_ids;
    let clip_ids = clip_tok.tokenize(&prompt).unwrap().input_ids;
    let gi = |k: &str| {
        g.require(k)
            .unwrap()
            .as_dtype(Dtype::Int32)
            .unwrap()
            .as_slice::<i32>()
            .to_vec()
    };
    let ri = |a: &Array| a.as_dtype(Dtype::Int32).unwrap().as_slice::<i32>().to_vec();
    let (gt5, gclip) = (gi("t5_input_ids"), gi("clip_input_ids"));
    let (rt5, rclip) = (ri(&t5_ids), ri(&clip_ids));
    let t5_diff = rt5.iter().zip(&gt5).filter(|(a, b)| a != b).count();
    let clip_diff = rclip.iter().zip(&gclip).filter(|(a, b)| a != b).count();
    println!(
        "t5 ids: rust_len={} golden_len={} mismatches={t5_diff} | clip ids: rust_len={} golden_len={} mismatches={clip_diff}",
        rt5.len(),
        gt5.len(),
        rclip.len(),
        gclip.len()
    );
    if t5_diff > 0 {
        println!("  t5 rust[:12]  ={:?}", &rt5[..12.min(rt5.len())]);
        println!("  t5 golden[:12]={:?}", &gt5[..12.min(gt5.len())]);
    }
    if clip_diff > 0 {
        println!("  clip rust[:8]  ={:?}", &rclip[..8.min(rclip.len())]);
        println!("  clip golden[:8]={:?}", &gclip[..8.min(gclip.len())]);
    }
    assert_eq!(rt5.len(), gt5.len(), "t5 ids length");
    assert_eq!(rclip.len(), gclip.len(), "clip ids length");
    assert_eq!(t5_diff, 0, "t5 tokenizer ids diverge from the fork");
    assert_eq!(clip_diff, 0, "clip tokenizer ids diverge from the fork");
    println!("✓ tokenizer ids match the fork golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_t5_prompt_embeds_match_golden() {
    let g = golden();
    let t5 = load_t5_encoder(&snapshot()).unwrap();
    let out = t5.forward(g.require("t5_input_ids").unwrap()).unwrap();
    let golden = g.require("prompt_embeds").unwrap();
    assert_eq!(out.shape(), golden.shape(), "prompt_embeds shape");
    let pr = peak_rel(&f32a(&out), golden);
    let mr = mean_abs_rel(&f32a(&out), golden);
    println!(
        "T5 prompt_embeds: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        out.shape()
    );
    // Bit-exact vs the bf16 fork (T5 runs f32 internally via T5LayerNorm's f32 upcast; sc-2787).
    assert!(pr < 1e-4, "T5 prompt_embeds diverged: peak_rel {pr:.3e}");
    println!("✓ T5 prompt_embeds match the fork golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_clip_pooled_matches_golden() {
    let g = golden();
    let clip = load_clip_encoder(&snapshot()).unwrap();
    let out = clip.forward(g.require("clip_input_ids").unwrap()).unwrap();
    let golden = g.require("pooled_prompt_embeds").unwrap();
    assert_eq!(out.shape(), golden.shape(), "pooled shape");
    let pr = peak_rel(&f32a(&out), golden);
    let mr = mean_abs_rel(&f32a(&out), golden);
    println!(
        "CLIP pooled: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        out.shape()
    );
    // Bit-exact vs the bf16 fork (CLIP genuinely runs bf16; sc-2787).
    assert!(
        pr < 1e-4,
        "CLIP pooled diverged from the fork: peak_rel {pr:.3e}"
    );
    println!("✓ CLIP pooled matches the fork golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_transformer_v0_matches_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let guid = guidance(&g);
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let t = load_transformer(&snapshot(), variant()).unwrap();
    // The transformer runs the fork's mixed precision: f32 latents/embeds/main-stream, bf16
    // conditioning/modulation (sc-2787). The golden stores embeds as f32; `TimeTextEmbed` re-rounds
    // pooled+time+guidance to bf16 internally, matching the fork. `guid` is the golden's guidance
    // (0.0 schnell / 3.5 dev — dev sums a guidance embedding into time_text_embed).
    let init = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    let v = t
        .forward(&init, &pe, &pooled, sigmas[0], guid, w, h)
        .unwrap();
    let golden = g.require("v0").unwrap();
    assert_eq!(v.shape(), golden.shape(), "v0 shape");
    let pr = peak_rel(&f32a(&v), golden);
    let mr = mean_abs_rel(&f32a(&v), golden);
    println!(
        "transformer v0: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        v.shape()
    );
    // Bit-exact vs the bf16 fork after the RoPE/time_proj host→MLX fixes (sc-2787).
    assert!(
        pr < 1e-3,
        "transformer single forward diverged: peak_rel {pr:.3e}"
    );
    println!("✓ transformer single forward matches golden");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_vae_decode_matches_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let vae = load_vae(&snapshot()).unwrap();
    let latents = g.require("final_latents").unwrap();
    let unpacked = unpack_latents(latents, w, h).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    let golden = g.require("decoded").unwrap();
    assert_eq!(decoded.shape(), golden.shape(), "decoded shape");
    let pr = peak_rel(&decoded, golden);
    println!("VAE decoded: peak_rel={pr:.3e} shape={:?}", decoded.shape());
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(golden).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    println!(
        "✓ VAE+image: {}x{}, {} / {} px differ by >8",
        img.width,
        img.height,
        differ,
        img.pixels.len()
    );
    assert!(
        differ < img.pixels.len() / 50,
        "too many VAE pixel diffs: {differ}"
    );
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_transformer_substages_match_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let guid = guidance(&g);
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let t = load_transformer(&snapshot(), variant()).unwrap();
    let init = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    let caps = t
        .forward_capture(&init, &pe, &pooled, sigmas[0], guid, w, h)
        .unwrap();
    for (name, arr) in &caps {
        if let Ok(golden) = g.require(name) {
            let pr = peak_rel(&f32a(arr), golden);
            let mr = mean_abs_rel(&f32a(arr), golden);
            println!(
                "substage {name}: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
                arr.shape()
            );
        } else {
            println!("substage {name}: (no golden) shape={:?}", arr.shape());
        }
    }
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_rope_table_matches_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let txt_seq = g.require("prompt_embeds").unwrap().shape()[1] as usize; // 256 schnell / 512 dev
    let t = load_transformer(&snapshot(), variant()).unwrap();
    // [txt_seq + img_seq, 64] (img_seq = (h/16)*(w/16))
    let (cos, sin) = t
        .debug_rope(txt_seq, (h / 16) as usize, (w / 16) as usize)
        .unwrap();
    let seq = cos.shape()[0];
    let half = cos.shape()[1];
    // fork rope0 [1,1,seq,64,2,2]; flatten the 2x2 (= [cos,-sin,sin,cos]) → col0=cos, col2=sin.
    let r = g
        .require("rope0")
        .unwrap()
        .reshape(&[seq, half, 4])
        .unwrap();
    let pick = |col: i32| {
        r.take_axis(Array::from_slice(&[col], &[1]), 2)
            .unwrap()
            .reshape(&[seq, half])
            .unwrap()
    };
    let cos_f = pick(0); // 2x2 row-major [cos,-sin,sin,cos] → col0=cos
    let sin_f = pick(2); // col2=sin
    println!(
        "rope cos: peak_rel={:.3e} mean_rel={:.3e} | sin: peak_rel={:.3e} mean_rel={:.3e}",
        peak_rel(&cos, &cos_f),
        mean_abs_rel(&cos, &cos_f),
        peak_rel(&sin, &sin_f),
        mean_abs_rel(&sin, &sin_f)
    );
}

#[test]
#[ignore = "needs local golden"]
fn e2e_sigmas_match_golden() {
    // The one genuinely dev-specific code path: FLUX.1-dev applies the mu-shift to the linear
    // sigmas (schnell does not). Validate `build_linear_sigmas` directly against the fork's sigmas.
    let g = golden();
    let (w, h) = wh(&g);
    let golden_sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let steps = golden_sigmas.len() - 1;
    let sigmas = build_linear_sigmas(steps, w, h, variant().requires_sigma_shift()).unwrap();
    assert_eq!(sigmas.len(), golden_sigmas.len(), "sigma count");
    let max_abs = sigmas
        .iter()
        .zip(&golden_sigmas)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    println!(
        "scheduler ({}, shift={}): {} sigmas, max|Δ|={max_abs:.3e}",
        variant_slug(),
        variant().requires_sigma_shift(),
        sigmas.len()
    );
    assert!(
        max_abs < 1e-5,
        "sigmas diverge from the fork (max|Δ| {max_abs:.3e}) — scheduler/mu-shift bug"
    );
    println!("✓ scheduler sigmas match the fork golden");
}

#[test]
#[ignore = "needs local golden"]
fn e2e_init_noise_matches_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let init = mlx_gen_flux::create_noise(seed, w, h).unwrap();
    let golden = g.require("init").unwrap();
    assert_eq!(init.shape(), golden.shape(), "init shape");
    let pr = peak_rel(&f32a(&init), golden);
    println!("init noise: peak_rel={pr:.3e} shape={:?}", init.shape());
    assert!(
        pr < 1e-5,
        "Rust create_noise diverges from the fork RNG: peak_rel {pr:.3e}"
    );
    println!("✓ init noise matches the fork RNG");
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_denoise_loop_matches_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let guid = guidance(&g);
    let sigmas = g.require("sigmas").unwrap().as_slice::<f32>().to_vec();
    let steps = sigmas.len() - 1;
    let t = load_transformer(&snapshot(), variant()).unwrap();
    // Feed the fork's exact init + golden embeds + fork sigmas: isolates the loop from RNG/text.
    let mut latents = f32a(g.require("init").unwrap());
    let pe = f32a(g.require("prompt_embeds").unwrap());
    let pooled = f32a(g.require("pooled_prompt_embeds").unwrap());
    for i in 0..steps {
        let v = t
            .forward(&latents, &pe, &pooled, sigmas[i], guid, w, h)
            .unwrap();
        let dt = sigmas[i + 1] - sigmas[i];
        latents = add(
            &latents,
            multiply(&v, mlx_rs::Array::from_slice(&[dt], &[1])).unwrap(),
        )
        .unwrap();
    }
    let golden = g.require("final_latents").unwrap();
    let pr = peak_rel(&f32a(&latents), golden);
    let mr = mean_abs_rel(&f32a(&latents), golden);
    println!(
        "denoise final_latents: peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        latents.shape()
    );
    // Bit-exact (sc-2787): with the transformer + sigmas bit-exact, the full denoise loop on the
    // fork's own embeds reproduces the fork latents exactly (0.000e0). Tiny margin for safety.
    assert!(mr < 1e-3, "denoise loop diverged: mean_rel {mr:.3e}");

    // Decode these (golden-embed) latents to pixels — isolates transformer+denoise+VAE px>8 from the
    // text-encoder f32-vs-bf16 contribution that the full-pipeline test additionally includes.
    let vae = load_vae(&snapshot()).unwrap();
    let unpacked = unpack_latents(&latents, w, h).unwrap();
    let decoded = f32a(&vae.decode(&unpacked).unwrap());
    let img = decoded_to_image(&decoded).unwrap();
    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "✓ denoise(golden embeds)+VAE: {:.2}% px>8 vs fork (transformer/denoise/VAE only; the full \
         pipeline adds Rust's f32 T5/CLIP vs fork's bf16)",
        frac * 100.0
    );
}

#[test]
#[ignore = "needs real FLUX.1-schnell weights + local golden"]
fn e2e_single_stack_injected_matches_golden() {
    let g = golden();
    let (w, h) = wh(&g);
    let (lh, lw) = ((h / 16) as usize, (w / 16) as usize);
    let t = load_transformer(&snapshot(), variant()).unwrap();
    // Feed the fork's EXACT post-joint tensors — isolates the single stack from the 0.45% joint drift.
    let encoder = f32a(g.require("encoder_joint").unwrap());
    let hidden = f32a(g.require("joint_hidden").unwrap());
    let text_emb = f32a(g.require("text_embeddings0").unwrap());
    for (name, arr) in t
        .debug_single_block0(&encoder, &hidden, &text_emb, lh, lw)
        .unwrap()
    {
        let golden = g.require(&name).unwrap();
        println!(
            "  {name}: peak_rel={:.3e} mean_rel={:.3e}",
            peak_rel(&f32a(&arr), golden),
            mean_abs_rel(&f32a(&arr), golden)
        );
    }
    let b0 = t
        .debug_single_stack(&encoder, &hidden, &text_emb, lh, lw, 1)
        .unwrap();
    let gb0 = g.require("single_b0_img").unwrap();
    println!(
        "single block[0] (injected): peak_rel={:.3e} mean_rel={:.3e}",
        peak_rel(&f32a(&b0), gb0),
        mean_abs_rel(&f32a(&b0), gb0)
    );
    let out = t
        .debug_single_stack(&encoder, &hidden, &text_emb, lh, lw, 0)
        .unwrap();
    let golden = g.require("single_img").unwrap();
    let pr = peak_rel(&f32a(&out), golden);
    let mr = mean_abs_rel(&f32a(&out), golden);
    println!(
        "single stack (injected): peak_rel={pr:.3e} mean_rel={mr:.3e} shape={:?}",
        out.shape()
    );
}

/// Full prompt→image pipeline through the public Generator API vs the fork's **bf16** render.
///
/// Post-sc-2787 this is a genuine PIXEL-PARITY gate. With bf16 TE/conditioning matching the fork's
/// mixed precision + the RoPE/time_proj/sigma host→MLX fixes + the vendored CLIP tokenizer, and
/// goldens dumped on the version-matched mlx 0.31.2, every stage is bit-exact (T5/CLIP/v0/all
/// transformer substages/denoise = 0.000e0 — see the other tests), so the public render lands at the
/// VAE's tiny cross-build floor: **schnell ~0.007% / dev ~0.026% px>8**.
///
/// Historical note: this used to sit at ~32–42% px>8 and was rationalized as "FLUX is precision
/// chaotic." That was wrong — the gap was three host-vs-MLX tables (RoPE freqs, time_proj freqs,
/// `build_linear_sigmas` linspace, each ~1e-7 amplified by the 57-block stack) plus a CLIP-tokenizer
/// bug (GPT-2 byte-level instead of CLIP word-BPE → wrong pooled conditioning). All fixed.
#[test]
#[ignore = "needs real FLUX.1 weights + local golden (FLUX_VARIANT selects schnell/dev)"]
fn e2e_full_pipeline_matches_fork() {
    let g = golden();
    let snap = snapshot();
    let prompt = g.metadata("prompt").unwrap().to_string();
    let seed: u64 = g.metadata("seed").unwrap().parse().unwrap();
    let steps: u32 = g.metadata("steps").unwrap().parse().unwrap();
    let w: u32 = g.metadata("w").unwrap().parse().unwrap();
    let h: u32 = g.metadata("h").unwrap().parse().unwrap();

    let spec = LoadSpec::new(WeightsSource::Dir(snap));
    let generator = mlx_gen::load(variant().id(), &spec).unwrap();
    let req = GenerationRequest {
        prompt,
        width: w,
        height: h,
        seed: Some(seed),
        steps: Some(steps),
        // dev is guidance-distilled (validate() rejects guidance for schnell).
        guidance: variant().supports_guidance().then(|| guidance(&g)),
        ..Default::default()
    };
    let mut last_step = 0u32;
    let out = generator
        .generate(&req, &mut |p| {
            if let Progress::Step { current, .. } = p {
                last_step = last_step.max(current);
            }
        })
        .unwrap();
    let img = match out {
        GenerationOutput::Images(mut v) => v.pop().unwrap(),
        other => panic!("expected Images, got {other:?}"),
    };
    assert_eq!((img.width, img.height), (w, h), "image size");

    let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("../tools/golden/rust_flux_{}.png", variant_slug()));
    image::save_buffer(
        &out_path,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();

    let gimg = decoded_to_image(g.require("decoded").unwrap()).unwrap();
    let differ = img
        .pixels
        .iter()
        .zip(&gimg.pixels)
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 8)
        .count();
    let frac = differ as f32 / img.pixels.len() as f32;
    println!(
        "full pipeline (public generate): {}x{}; {:.3}% px>8 ({} / {}) vs fork; saved {}",
        img.width,
        img.height,
        frac * 100.0,
        differ,
        img.pixels.len(),
        out_path.display()
    );
    // Pixel-parity: every stage is bit-exact, so only the VAE's tiny cross-build residual remains
    // (observed schnell ~0.007% / dev ~0.026% px>8). 0.5% is a generous parity bound that still
    // catches any real regression (the old broken/chaotic states were 32–95%).
    assert!(
        frac < 5e-3,
        "full pipeline regressed from pixel-parity ({:.3}% px>8) — a stage is no longer bit-exact",
        frac * 100.0
    );
    println!(
        "✓ full FLUX.1-{} pipeline is pixel-parity with the fork ({:.3}% px>8 — VAE cross-build floor)",
        variant_slug(),
        frac * 100.0
    );
}
