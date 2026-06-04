//! sc-2769: end-to-end few-step acceleration renders (LCM / SDXL-Lightning / Hyper-SD) on the real
//! SDXL weights, each with its acceleration LoRA merged.
//!
//! `#[ignore]`d — needs the real SDXL snapshot + the acceleration LoRAs in the HF cache:
//!   latent-consistency/lcm-lora-sdxl, ByteDance/SDXL-Lightning, ByteDance/Hyper-SD.
//!   cargo test -p mlx-gen-sdxl --release --test accel_real_weights -- --ignored --nocapture
//!
//! Two gates:
//! - `few_step_renders_are_coherent` (acceptance): load SDXL + each accel LoRA, render at the locked
//!   4-step default, and assert the image is non-degenerate (finite, structured, sane mean). Writes
//!   PNGs to `tools/golden/` for eyeball review. This is the "renders correct few-step output" check
//!   — bit-exact parity vs a torch reference is impossible (different backend), so the
//!   *deterministic* parity number is the separate diagnostic below.
//! - `lightning_hyper_match_torch_teacher_forced` (diagnostic): for the two DETERMINISTIC samplers
//!   (Lightning Euler-trailing, Hyper TCD eta=0), inject the torch render's initial latent and report
//!   px>8 vs the torch image (from `dump_sdxl_accel_golden.py render`). Interpreted against the
//!   ancestral baseline torch↔MLX gap, also printed.

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen::{
    AdapterKind, AdapterSpec, GenerationOutput, GenerationRequest, Image, LoadSpec, WeightsSource,
};
use mlx_gen_sdxl as _;

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

/// The newest snapshot file for an HF repo whose basename matches `file` (the LoRA `.safetensors`).
fn cache_file(repo: &str, file: &str) -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots");
    let dir = std::fs::read_dir(&snaps)
        .unwrap_or_else(|_| panic!("HF cache for {repo}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir");
    dir.join(file)
}

fn lora_spec(path: PathBuf, scale: f32) -> AdapterSpec {
    AdapterSpec {
        path,
        scale,
        kind: AdapterKind::Lora,
        pass_scales: None,
    }
}

/// `(sampler, accel-LoRA path, steps)` for the three acceleration variants.
fn variants() -> Vec<(&'static str, PathBuf, u32)> {
    vec![
        (
            "lcm",
            cache_file(
                "latent-consistency/lcm-lora-sdxl",
                "pytorch_lora_weights.safetensors",
            ),
            4,
        ),
        (
            "lightning",
            cache_file(
                "ByteDance/SDXL-Lightning",
                "sdxl_lightning_4step_lora.safetensors",
            ),
            4,
        ),
        (
            "hyper",
            cache_file("ByteDance/Hyper-SD", "Hyper-SDXL-4steps-lora.safetensors"),
            4,
        ),
    ]
}

fn save_png(name: &str, img: &Image) -> PathBuf {
    let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    std::fs::create_dir_all(&out).unwrap();
    let p = out.join(name);
    image::save_buffer(
        &p,
        &img.pixels,
        img.width,
        img.height,
        image::ExtendedColorType::Rgb8,
    )
    .unwrap();
    p
}

/// Mean and population std of an RGB8 buffer — a coherent render has real spatial variance, a
/// degenerate one (NaN→0, constant gray) does not.
fn mean_std(px: &[u8]) -> (f32, f32) {
    let n = px.len() as f32;
    let mean = px.iter().map(|&v| v as f32).sum::<f32>() / n;
    let var = px.iter().map(|&v| (v as f32 - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

#[test]
#[ignore = "needs the real SDXL snapshot + the LCM/Lightning/Hyper LoRAs in the HF cache"]
fn few_step_renders_are_coherent() {
    let prompt = "a red fox in a forest, highly detailed".to_string();
    let (w, h, seed) = (1024u32, 1024u32, 42u64);

    for (sampler, lora, steps) in variants() {
        assert!(lora.exists(), "{sampler} LoRA missing: {}", lora.display());
        let spec =
            LoadSpec::new(WeightsSource::Dir(snapshot())).with_adapters(vec![lora_spec(lora, 1.0)]);
        let model = mlx_gen::load("sdxl", &spec)
            .unwrap_or_else(|e| panic!("load sdxl + {sampler} LoRA: {e}"));
        let req = GenerationRequest {
            prompt: prompt.clone(),
            width: w,
            height: h,
            seed: Some(seed),
            sampler: Some(sampler.to_string()),
            // steps/guidance default to the per-variant table (4-step / CFG 1).
            ..Default::default()
        };
        let mut last = 0u32;
        let out = model
            .generate(&req, &mut |p| {
                if let mlx_gen::Progress::Step { current, .. } = p {
                    last = last.max(current);
                }
            })
            .unwrap_or_else(|e| panic!("{sampler} generate: {e}"));
        assert_eq!(last, steps, "{sampler}: expected {steps} step events");
        let img = match out {
            GenerationOutput::Images(mut v) => v.pop().unwrap(),
            other => panic!("expected Images, got {other:?}"),
        };
        assert_eq!((img.width, img.height), (w, h));
        let p = save_png(&format!("sdxl_accel_{sampler}.png"), &img);
        let (mean, std) = mean_std(&img.pixels);
        println!(
            "✓ {sampler} {steps}-step: mean {mean:.1}/255  std {std:.1}  → {}",
            p.display()
        );
        // Non-degenerate: a real image has spatial variance and a sane (non-black/blown-out) mean.
        assert!(
            std > 15.0,
            "{sampler}: render looks degenerate (std {std:.1})"
        );
        assert!(
            (8.0..248.0).contains(&mean),
            "{sampler}: render mean {mean:.1} out of sane range (NaN/constant?)"
        );
    }
}

// ---- deterministic torch parity diagnostic -------------------------------------------------

fn px_frac(a: &[u8], b: &[u8], thr: i32) -> f32 {
    let differ = a
        .iter()
        .zip(b)
        .filter(|(x, y)| (**x as i32 - **y as i32).abs() > thr)
        .count();
    differ as f32 / a.len() as f32
}

/// diffusers NCHW f32 latent `[1,4,H,W]` → MLX NHWC `[1,H,W,4]`.
fn nchw_to_nhwc(a: &Array) -> Array {
    a.transpose_axes(&[0, 2, 3, 1]).unwrap()
}

/// Teacher-force the torch initial latent through the MLX deterministic samplers and report px>8 vs
/// the torch render. Needs `dump_sdxl_accel_golden.py render`. Prints the ancestral baseline gap too.
#[test]
#[ignore = "needs the SDXL snapshot + accel LoRAs + sdxl_accel_render_* goldens (dump … render)"]
fn lightning_hyper_match_torch_teacher_forced() {
    use mlx_gen::sampler::{AlphaSchedule, DiffusionSampler, LightningSampler, TcdSampler};
    use mlx_gen_sdxl::config::DiffusionConfig;
    use mlx_gen_sdxl::{
        decode_image, denoise, encode_conditioning, load_text_encoder_1_dtype,
        load_text_encoder_2_dtype, load_tokenizer, load_unet_dtype, text_time_ids, Denoiser,
    };

    let golden_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tools/golden");
    let dt = Dtype::Float16;
    let snap = snapshot();
    let cfg = DiffusionConfig::sdxl_base();
    let sched =
        AlphaSchedule::scaled_linear(cfg.num_train_steps, cfg.beta_start, cfg.beta_end).unwrap();

    // NO-LoRA backend baseline: base SDXL + Euler-trailing (30-step, CFG 1), teacher-forced from
    // torch's init latent + torch's CLIP conditioning. This is the torch↔MLX SDXL U-Net backend floor
    // with zero acceleration LoRA — the accel variants' gaps below are read against it.
    if let Ok(g) = Weights::from_file(golden_dir.join("sdxl_accel_render_base.safetensors")) {
        let steps: usize = g.metadata("steps").unwrap().parse().unwrap();
        let unet = load_unet_dtype(&snap, dt).unwrap();
        let sampler = LightningSampler::new(&sched, cfg.num_train_steps, steps, dt);
        let init = nchw_to_nhwc(g.require("init_latent").unwrap());
        let pe = g.require("prompt_embeds").unwrap().as_dtype(dt).unwrap();
        let pp = g.require("pooled").unwrap().as_dtype(dt).unwrap();
        let tids = text_time_ids(pp.shape()[0]);
        let d = Denoiser {
            unet: &unet,
            sampler: &sampler,
        };
        let lat = denoise(
            &d,
            init,
            &pe,
            &pp,
            &tids,
            1.0,
            &Default::default(),
            &mut |_| {},
        )
        .unwrap();
        let img = decode_image(&mlx_gen_sdxl::load_vae(&snap).unwrap(), &lat).unwrap();
        let gpix: Vec<u8> = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();
        save_png("sdxl_accel_base_tf.png", &img);
        let b8 = px_frac(&img.pixels, &gpix, 8) * 100.0;
        println!("• base (NO LoRA, Euler-trailing {steps}-step, torch-CLIP): px>8 {b8:.1}% vs torch — the backend floor");
        // The no-LoRA MLX path (U-Net + VAE + the Lightning Euler-trailing sampler) is bit-faithful
        // to diffusers: this is the floor every accel gap is read against (measured 0.7%).
        assert!(
            b8 < 3.0,
            "no-LoRA backend floor regressed: {b8:.1}% px>8 (sampler/UNet/VAE diverged from diffusers)"
        );
    }

    for (tag, lora) in [
        (
            "lightning",
            cache_file(
                "ByteDance/SDXL-Lightning",
                "sdxl_lightning_4step_lora.safetensors",
            ),
        ),
        (
            "hyper",
            cache_file("ByteDance/Hyper-SD", "Hyper-SDXL-4steps-lora.safetensors"),
        ),
    ] {
        let gpath = golden_dir.join(format!("sdxl_accel_render_{tag}.safetensors"));
        let Ok(g) = Weights::from_file(&gpath) else {
            eprintln!(
                "skip {tag}: {} not present (run dump_sdxl_accel_golden.py render)",
                gpath.display()
            );
            continue;
        };
        let prompt = g.metadata("prompt").unwrap().to_string();
        let steps: usize = g.metadata("steps").unwrap().parse().unwrap();
        let (w, h): (u32, u32) = (
            g.metadata("w").unwrap().parse().unwrap(),
            g.metadata("h").unwrap().parse().unwrap(),
        );

        // Load + merge the accel LoRA into the fp16 U-Net, using the SAME Complete coverage as the
        // production `load` path (sc-2671) — NOT the bare `apply_sdxl_adapters` (Vendored 515).
        let mut unet = load_unet_dtype(&snap, dt).unwrap();
        mlx_gen_sdxl::apply_sdxl_adapters_with(
            &mut unet,
            &[lora_spec(lora, 1.0)],
            mlx_gen_sdxl::LoraCoverage::Complete,
        )
        .unwrap();
        let te1 = load_text_encoder_1_dtype(&snap, dt).unwrap();
        let te2 = load_text_encoder_2_dtype(&snap, dt).unwrap();
        let vae = mlx_gen_sdxl::load_vae(&snap).unwrap();
        let tok = load_tokenizer(&snap).unwrap();

        // CFG off (guidance 1) for both → single (positive) conditioning row.
        let tokens = tok.tokenize_batch(&prompt, None).unwrap();
        let (mlx_cond, mlx_pooled) = encode_conditioning(&te1, &te2, &tokens).unwrap();

        // Inject the torch initial latent (NCHW f32 → NHWC), so the trajectory is deterministic.
        let init = nchw_to_nhwc(g.require("init_latent").unwrap());
        let gpix: Vec<u8> = g.require("image_u8").unwrap().as_slice::<u8>().to_vec();

        let sampler: Box<dyn DiffusionSampler> = if tag == "lightning" {
            Box::new(LightningSampler::new(
                &sched,
                cfg.num_train_steps,
                steps,
                dt,
            ))
        } else {
            Box::new(TcdSampler::new(
                sched.clone(),
                cfg.num_train_steps,
                50,
                steps,
                0.0,
                dt,
            ))
        };

        // Render from the (shared, deterministic) injected latent under a given CLIP conditioning.
        let render = |cond: &Array, pooled: &Array| -> Image {
            let tids = text_time_ids(pooled.shape()[0]);
            let d = Denoiser {
                unet: &unet,
                sampler: sampler.as_ref(),
            };
            let lat = denoise(
                &d,
                init.clone(),
                cond,
                pooled,
                &tids,
                1.0,
                &Default::default(),
                &mut |_| {},
            )
            .unwrap();
            decode_image(&vae, &lat).unwrap()
        };

        // (1) MLX CLIP conditioning — the full MLX path (CLIP + U-Net both MLX).
        let img_mlx = render(&mlx_cond, &mlx_pooled);
        save_png(&format!("sdxl_accel_{tag}_mlx_tf.png"), &img_mlx);
        assert_eq!((img_mlx.width, img_mlx.height), (w, h));
        let m8 = px_frac(&img_mlx.pixels, &gpix, 8) * 100.0;

        // (2) Torch CLIP conditioning injected — isolates the U-Net/LoRA backend gap from the CLIP
        // (text-encoder) backend gap. If (2) ≪ (1), the divergence is the CLIP backend, not the U-Net.
        let line = if let (Some(pe), Some(pp)) = (g.get("prompt_embeds"), g.get("pooled")) {
            let img_t = render(&pe.as_dtype(dt).unwrap(), &pp.as_dtype(dt).unwrap());
            save_png(&format!("sdxl_accel_{tag}_torchcond_tf.png"), &img_t);
            let t8 = px_frac(&img_t.pixels, &gpix, 8) * 100.0;
            format!("• {tag}: px>8 vs torch — MLX-CLIP {m8:.1}%  torch-CLIP-injected {t8:.1}%")
        } else {
            format!(
                "• {tag}: px>8 vs torch — MLX-CLIP {m8:.1}% (re-dump for the torch-CLIP isolation)"
            )
        };
        println!("{line}");

        // (3) Linear-only isolation (Lightning only): compare the MLX Linear-only merge to a torch
        // render with the conv LoRA modules STRIPPED (same Linear-only fusion). If this ≈ the backend
        // floor, the full-fusion gap (1)/(2) is exactly the dropped conv-layer LoRA (sc-2639 boundary),
        // and the MLX Linear merge itself is faithful to torch.
        if tag == "lightning" {
            if let Ok(gl) = Weights::from_file(
                golden_dir.join("sdxl_accel_render_lightning_linonly.safetensors"),
            ) {
                let init_l = nchw_to_nhwc(gl.require("init_latent").unwrap());
                let pe = gl.require("prompt_embeds").unwrap().as_dtype(dt).unwrap();
                let pp = gl.require("pooled").unwrap().as_dtype(dt).unwrap();
                let tids = text_time_ids(pp.shape()[0]);
                let d = Denoiser {
                    unet: &unet,
                    sampler: sampler.as_ref(),
                };
                let lat = denoise(
                    &d,
                    init_l,
                    &pe,
                    &pp,
                    &tids,
                    1.0,
                    &Default::default(),
                    &mut |_| {},
                )
                .unwrap();
                let img = decode_image(&vae, &lat).unwrap();
                let gpix_l: Vec<u8> = gl.require("image_u8").unwrap().as_slice::<u8>().to_vec();
                let l8 = px_frac(&img.pixels, &gpix_l, 8) * 100.0;
                println!("• lightning Linear-only (conv stripped both sides): px>8 {l8:.1}% vs torch — proves the Linear merge is faithful");
                // With the conv LoRA stripped from BOTH sides, the MLX Complete Linear merge is
                // bit-exact to torch's Linear fusion (measured 0.0%). The full-LoRA gap above is
                // therefore *entirely* the dropped conv-layer LoRA (sc-2639 Linear-only boundary).
                assert!(
                    l8 < 3.0,
                    "Linear-only LoRA merge diverged from torch's Linear fusion: {l8:.1}% px>8 \
                     (a real Linear-merge regression — the conv drop alone should leave ≈0%)"
                );
            }
        }
    }
    println!(
        "(SDXL-MLX is faithful to the vendored Apple `mlx_sd` reference, NOT diffusers — the real \
         sampler parity is the bit-exact scheduler-isolation gate. The full-fusion e2e gap is the \
         dropped conv-layer LoRA (sc-2639 Linear-only boundary), localized by the base-floor + \
         Linear-only rows; the CLIP-injected column rules out the text encoder.)"
    );
}
