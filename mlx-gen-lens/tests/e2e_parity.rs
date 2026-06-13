//! sc-3173 — end-to-end Lens-Turbo T2I parity vs the vendor `LensPipeline`.
//!
//! Runs the **full** Rust pipeline — the [`LensTokenizer`] (harmony render) → the gpt-oss
//! [`LensTextEncoder`] (capture + `txt_offset` slice) → the [`LensTransformer`] denoise (turbo
//! schedule + norm-rescaled CFG) → the Flux.2 [`vae`] decode — on the **same injected initial
//! latents** the torch golden used, and compares against the reference's final latents + decoded
//! image.
//!
//! The e2e is **cross-build** (MLX-Metal vs torch-CPU, both bf16): per-step bf16 op-order diverges
//! and accumulates over 48 DiT blocks × 4 steps, so the gate is **structural** (cosine) + coherence,
//! not bit-exact — the FLUX-hyper / cross-backend precedent. Injecting the reference's starting noise
//! removes the only *un*-reproducible source (the RNG); a wrong wiring (channel packing, offset slice,
//! CFG, timestep convention, …) would collapse the cosine, so a high cosine bounds the pipeline as
//! correct. The tokenizer is validated *inside* the e2e: the Rust render (with the golden's date) must
//! reproduce the golden's `input_ids` byte-for-byte.
//!
//! Run: `cargo test -p mlx-gen-lens --test e2e_parity -- --ignored --nocapture`

use mlx_rs::ops::{abs, max, multiply, subtract, sum};
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_lens::pipeline::LensPipeline;
use mlx_gen_lens::text::LensTokenizer;
use mlx_gen_lens::vae;

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/lens_e2e_golden.safetensors"
);

fn snapshot_root() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--microsoft--Lens-Turbo/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|_| panic!("snapshot dir {}", base.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .max()
        .expect("a snapshot")
}

fn meta_usize(g: &Weights, key: &str) -> usize {
    g.metadata(key).unwrap().parse().unwrap()
}
fn meta_f32(g: &Weights, key: &str) -> f32 {
    g.metadata(key).unwrap().parse().unwrap()
}

/// `max|a-b| / max|b|`.
fn peak_rel(got: &Array, want: &Array) -> f32 {
    let diff = abs(subtract(got, want).unwrap()).unwrap();
    let denom = max(abs(want).unwrap(), None).unwrap().item::<f32>();
    max(&diff, None).unwrap().item::<f32>() / denom.max(1e-12)
}

/// Cosine similarity over the flattened tensors.
fn cosine(got: &Array, want: &Array) -> f32 {
    let dot = sum(multiply(got, want).unwrap(), None)
        .unwrap()
        .item::<f32>();
    let na = sum(multiply(got, got).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    let nb = sum(multiply(want, want).unwrap(), None)
        .unwrap()
        .item::<f32>()
        .sqrt();
    dot / (na * nb).max(1e-12)
}

#[test]
#[ignore = "needs tools/golden/lens_e2e_golden.safetensors + the full Lens-Turbo snapshot (~50GB bf16 load)"]
fn lens_e2e_matches_reference() {
    let g = Weights::from_file(GOLDEN).expect("e2e golden");
    let (lat_h, lat_w) = (meta_usize(&g, "latent_h"), meta_usize(&g, "latent_w"));
    let num_steps = meta_usize(&g, "num_steps");
    let guidance = meta_f32(&g, "guidance");
    let prompt = g.metadata("prompt").unwrap();
    let negative = g.metadata("negative_prompt").unwrap_or_default();
    let date = g.metadata("current_date").unwrap();

    let snap = snapshot_root();

    // 1. Tokenizer cross-check (inside the e2e): the Rust harmony render with the golden's date must
    //    reproduce the golden's input_ids exactly — otherwise the encoder sees a different sequence.
    let tok =
        LensTokenizer::from_file(snap.join("tokenizer").join("tokenizer.json")).expect("tokenizer");
    let out = tok.encode(prompt, date).expect("encode prompt");
    let want_ids = g.require("input_ids").unwrap(); // [1, L] i32
    let got_ids = Array::from_slice(&out.ids, &[1, out.ids.len() as i32]);
    assert_eq!(
        got_ids.shape(),
        want_ids.shape(),
        "tokenizer length {:?} != golden {:?}",
        got_ids.shape(),
        want_ids.shape()
    );
    let id_mismatch = max(
        abs(subtract(&got_ids, want_ids.as_dtype(Dtype::Int32).unwrap()).unwrap()).unwrap(),
        None,
    )
    .unwrap()
    .item::<i32>();
    assert_eq!(id_mismatch, 0, "Rust tokenizer ids differ from the golden");

    // 2. Load the full pipeline (bf16 production) and run the real path with the injected latents.
    eprintln!("loading Lens pipeline (encoder MXFP4→bf16 + DiT bf16 + VAE f32)…");
    let pipe = LensPipeline::load(&snap, Dtype::Bfloat16).expect("load pipeline");

    let (features, mask) = pipe
        .encode_prompt(prompt, negative, date)
        .expect("encode_prompt");
    let init = g.require("init_latents").unwrap().clone(); // [1, seq, 128] f32

    eprintln!("denoising {num_steps} steps @ latent {lat_h}x{lat_w}…");
    let latents = pipe
        .denoise(
            &features,
            &mask,
            &init,
            lat_h,
            lat_w,
            num_steps,
            guidance,
            &mlx_gen::CancelFlag::default(),
            &mut |c, t| eprintln!("  step {c}/{t}"),
        )
        .expect("denoise");

    // 3. Compare the final latents (the tightest e2e signal: encoder + DiT + scheduler + CFG, pre-VAE).
    let got_lat = latents.as_dtype(Dtype::Float32).unwrap();
    let want_lat = g.require("final_latents").unwrap(); // [1, seq, 128] f32
    assert_eq!(
        got_lat.shape(),
        want_lat.shape(),
        "final-latent shape {:?} != {:?}",
        got_lat.shape(),
        want_lat.shape()
    );
    let lat_cos = cosine(&got_lat, want_lat);
    let lat_pr = peak_rel(&got_lat, want_lat);
    eprintln!("final latents: cosine {lat_cos:.5}  peak_rel {lat_pr:.3e}");

    // 4. Compare the decoded image (full e2e incl. the VAE shim).
    let decoded = vae::decode(pipe.vae(), &latents, lat_h, lat_w).unwrap(); // [1,H,W,3] NHWC [-1,1]
    let got_img = {
        // → [0,1] to match the golden's stored range.
        let half = Array::from_f32(0.5);
        let x = mlx_rs::ops::add(
            mlx_rs::ops::multiply(decoded.as_dtype(Dtype::Float32).unwrap(), &half).unwrap(),
            &half,
        )
        .unwrap();
        mlx_rs::ops::clip(&x, (0.0, 1.0)).unwrap()
    };
    let want_img = g.require("image").unwrap(); // [1,H,W,3] f32 in [0,1]
    assert_eq!(
        got_img.shape(),
        want_img.shape(),
        "image shape {:?} != {:?}",
        got_img.shape(),
        want_img.shape()
    );
    let img_cos = cosine(&got_img, want_img);
    let img_pr = peak_rel(&got_img, want_img);
    // Coherence floor: a degenerate (flat) render would have ~0 variance. The reference image's own
    // std is the yardstick; ours must be in the same ballpark (not collapsed to a constant).
    let std_of = |x: &Array| -> f32 {
        let m = mlx_rs::ops::mean(x, None).unwrap();
        let v = mlx_rs::ops::mean(
            multiply(subtract(x, &m).unwrap(), subtract(x, &m).unwrap()).unwrap(),
            None,
        )
        .unwrap()
        .item::<f32>();
        v.sqrt()
    };
    let (got_std, want_std) = (std_of(&got_img), std_of(want_img));
    eprintln!("image: cosine {img_cos:.5}  peak_rel {img_pr:.3e}  std got {got_std:.4} / ref {want_std:.4}");

    // Gates — structural (cross-build), not bit-exact.
    assert!(
        lat_cos > 0.90,
        "final-latent cosine {lat_cos:.5} ≤ 0.90 — wiring divergence, not bf16 noise"
    );
    assert!(
        img_cos > 0.90,
        "decoded-image cosine {img_cos:.5} ≤ 0.90 — wiring/VAE divergence"
    );
    assert!(
        got_std > 0.5 * want_std,
        "decoded image is near-flat (std {got_std:.4} vs ref {want_std:.4}) — not a coherent render"
    );
    eprintln!("ALL PASS");
}
