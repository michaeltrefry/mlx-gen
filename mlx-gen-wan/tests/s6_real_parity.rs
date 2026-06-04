//! S6 **real-weight** end-to-end parity gate (`#[ignore]` — needs the 54 GB converted A14B
//! checkpoint, so it never runs in CI; the tiny seeded S1–S5 gates carry CI).
//!
//! This is the honest "real Mac e2e" for the Wan2.2-T2V-A14B port: it loads the **actual converted**
//! checkpoint and runs the genuine chain the product `Wan14b::generate` runs —
//!   real UMT5-XXL encode → per-expert `embed_text` → boundary-switched dual-expert `denoise_moe`
//!   over the two real 40-layer / dim-5120 experts → real z16 VAE decode
//! — comparing the final latents + decoded frames against a golden dumped from the `mlx_video`
//! Python reference on the same converted weights + the same injected noise
//! (`tools/dump_s6_real_fixtures.py`).
//!
//! Only the init noise is injected (seeded RNG isn't portable across the mlx-python / mlx-rs split);
//! everything else is recomputed in Rust, so this also re-checks **real-weight UMT5 parity** (the
//! Rust encode of the same prompt must match the dumped context) on top of the assembly.
//!
//! Run it (after `dump_s6_real_fixtures.py` wrote the fixture):
//! ```text
//! WAN_A14B_MODEL_DIR=~/.cache/mlx-gen-models/wan2_2_t2v_a14b_mlx_bf16 \
//! WAN_A14B_FIXTURE=/tmp/wan_a14b_s6.safetensors \
//!   cargo test -p mlx-gen-wan --test s6_real_parity -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen_wan::config::WanModelConfig;
use mlx_gen_wan::pipeline::denoise_moe;
use mlx_gen_wan::scheduler::SolverKind;
use mlx_gen_wan::{decode_to_frames, load_tokenizer, Expert, Umt5Encoder, WanTransformer, WanVae};

fn env_path(var: &str) -> Option<PathBuf> {
    std::env::var_os(var).map(|s| PathBuf::from(shellexpand_home(&s.to_string_lossy())))
}

/// Minimal `~` expansion (the env vars are typically written with a leading `~`).
fn shellexpand_home(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return format!("{}/{rest}", home.to_string_lossy());
        }
    }
    s.to_string()
}

fn diff(got: &[f32], exp: &[f32]) -> (f32, f64) {
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (g - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += e.abs() as f64;
    }
    (max_abs, sum_abs / sum_ref.max(1e-9))
}

fn prepend1(shape: &[i32]) -> Vec<i32> {
    let mut s = vec![1];
    s.extend_from_slice(shape);
    s
}

#[test]
#[ignore = "needs the 54 GB converted Wan2.2-T2V-A14B checkpoint (WAN_A14B_MODEL_DIR + WAN_A14B_FIXTURE)"]
fn wan_a14b_real_weight_e2e_matches_reference() {
    let model_dir = match env_path("WAN_A14B_MODEL_DIR") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_MODEL_DIR to the converted A14B model dir");
            return;
        }
    };
    let fixture = match env_path("WAN_A14B_FIXTURE") {
        Some(p) => p,
        None => {
            eprintln!("skip: set WAN_A14B_FIXTURE (run tools/dump_s6_real_fixtures.py first)");
            return;
        }
    };

    let cfg = WanModelConfig::from_model_dir(&model_dir).expect("read config.json");
    assert!(cfg.dual_model, "expected the dual-expert A14B config");
    let (low_gs, high_gs) = match cfg.sample_guide_scale {
        mlx_gen_wan::GuideScale::Dual { low, high } => (low, high),
        other => panic!("expected dual guide scale, got {other:?}"),
    };

    let fx = Weights::from_file(&fixture).expect("read fixture (run dump_s6_real_fixtures.py)");
    let noise = fx.require("noise").unwrap();
    let exp_ctx = fx.require("context").unwrap();
    let exp_ctx_null = fx.require("context_null").unwrap();
    let exp_lat = fx.require("final_latents").unwrap();
    let exp_vid = fx.require("video").unwrap();

    // The dumper's small-but-real geometry (must match dump_s6_real_fixtures.py).
    let prompt = "a red fox trotting across a snowy meadow at sunrise, cinematic";
    let steps = 6usize;
    let shift = cfg.sample_shift;

    // --- Real-weight UMT5 encode of the same prompt + the config negative prompt ---
    let tokenizer = load_tokenizer(model_dir.join("tokenizer.json"), cfg.text_len).unwrap();
    let t5_w = Weights::from_file(model_dir.join("t5_encoder.safetensors")).expect("t5 weights");
    let enc = Umt5Encoder::from_weights(&t5_w, &cfg).expect("umt5");
    let context = enc.encode(&tokenizer, prompt).unwrap();
    let context_null = enc.encode(&tokenizer, &cfg.sample_neg_prompt).unwrap();

    // Real-weight T5 parity (the encode is bit-exact in S1; here at full UMT5-XXL scale).
    assert_eq!(context.shape(), exp_ctx.shape(), "context shape");
    assert_eq!(
        context_null.shape(),
        exp_ctx_null.shape(),
        "context_null shape"
    );
    let (cx_max, cx_mr) = diff(context.as_slice::<f32>(), exp_ctx.as_slice::<f32>());
    let (cn_max, cn_mr) = diff(
        context_null.as_slice::<f32>(),
        exp_ctx_null.as_slice::<f32>(),
    );
    println!("[t5 context]      max|Δ|={cx_max:.3e} mean_rel={cx_mr:.3e}");
    println!("[t5 context_null] max|Δ|={cn_max:.3e} mean_rel={cn_mr:.3e}");
    drop(enc);
    drop(t5_w);

    // --- Both real experts + boundary-switched dual MoE denoise on the injected noise ---
    let low_w = Weights::from_file(model_dir.join("low_noise_model.safetensors")).expect("low");
    let high_w = Weights::from_file(model_dir.join("high_noise_model.safetensors")).expect("high");
    let low_dit = WanTransformer::from_weights(&low_w, &cfg).expect("low DiT");
    let high_dit = WanTransformer::from_weights(&high_w, &cfg).expect("high DiT");

    let low = Expert {
        transformer: &low_dit,
        ctx_cond: low_dit.embed_text(&context).unwrap(),
        ctx_uncond: Some(low_dit.embed_text(&context_null).unwrap()),
        guidance: low_gs,
    };
    let high = Expert {
        transformer: &high_dit,
        ctx_cond: high_dit.embed_text(&context).unwrap(),
        ctx_uncond: Some(high_dit.embed_text(&context_null).unwrap()),
        guidance: high_gs,
    };
    let boundary_timestep = cfg.boundary * cfg.num_train_timesteps as f32;

    let latents = denoise_moe(
        &low,
        &high,
        boundary_timestep,
        SolverKind::UniPC,
        cfg.num_train_timesteps,
        steps,
        shift,
        noise,
        &mut |i| println!("  step {i}/{steps}"),
    )
    .expect("denoise_moe");

    assert_eq!(latents.shape(), exp_lat.shape(), "final latent shape");
    let (la_max, la_mr) = diff(latents.as_slice::<f32>(), exp_lat.as_slice::<f32>());
    println!(
        "[a14b latents] shape={:?} max|Δ|={la_max:.3e} mean_rel={la_mr:.3e}",
        latents.shape()
    );
    drop(low_dit);
    drop(high_dit);
    drop(low_w);
    drop(high_w);

    // --- Real z16 VAE decode (compare the raw [-1,1] decode against the golden) ---
    let vae_w = Weights::from_file(model_dir.join("vae.safetensors")).expect("vae");
    let vae = WanVae::from_weights(&vae_w).expect("vae");
    let video = vae
        .decode(&latents.reshape(&prepend1(latents.shape())).unwrap())
        .unwrap();
    assert_eq!(video.shape(), exp_vid.shape(), "video shape");
    let (vid_max, vid_mr) = diff(video.as_slice::<f32>(), exp_vid.as_slice::<f32>());
    println!(
        "[a14b video]   shape={:?} max|Δ|={vid_max:.3e} mean_rel={vid_mr:.3e}",
        video.shape()
    );

    // Also exercise the product frame-assembly path (uint8 [F,H,W,3] → Vec<Image>).
    let frames_u8 = decode_to_frames(&vae, &latents).unwrap();
    let images = mlx_gen_wan::frames_to_images(&frames_u8).unwrap();
    assert_eq!(images.len(), exp_vid.shape()[2] as usize, "frame count");
    for img in &images {
        assert_eq!(
            img.pixels.len(),
            (img.width * img.height * 3) as usize,
            "frame pixel buffer size"
        );
    }
    println!(
        "[frames] {} frames @ {}x{}",
        images.len(),
        images[0].width,
        images[0].height
    );

    // T5 encode is bf16-GEMM → tiny cross-build drift; the 40-layer dual-expert stack compounds the
    // 0.31.1-vs-0.31.2 NAX kernel diff (S4/S5 envelope at tiny scale was 2e-2). A conversion or
    // wiring bug gives mean_rel ~O(1), not a few e-2 — these thresholds catch that decisively.
    assert!(cx_mr < 1e-2, "t5 context diverged: mean_rel={cx_mr:.3e}");
    assert!(
        cn_mr < 1e-2,
        "t5 context_null diverged: mean_rel={cn_mr:.3e}"
    );
    assert!(la_mr < 8e-2, "a14b latents diverged: mean_rel={la_mr:.3e}");
    assert!(vid_mr < 8e-2, "a14b video diverged: mean_rel={vid_mr:.3e}");
}
