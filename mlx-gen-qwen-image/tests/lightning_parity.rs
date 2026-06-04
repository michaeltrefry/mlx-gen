//! sc-2909: Qwen-Image Lightning schedule-isolation parity vs **diffusers**.
//!
//! Qwen is flow-match, so the Lightning recipe is its own schedule under the `DiffusionSampler`
//! trait. The frozen fork has no dedicated Qwen Lightning sampler, so (as with the SDXL accel work,
//! sc-2769) the reference is the official lightx2v recipe as realized in diffusers'
//! `FlowMatchEulerDiscreteScheduler`. This gate feeds the SAME synthetic `(velocity, sample)` tensors
//! diffusers saw at each step and requires the Rust `FlowMatchSampler::lightning(n)` schedule + Euler
//! step to match diffusers to ~1e-6 (torch-f32 vs MLX-f32) — validating the Lightning sigmas free of
//! the transformer-backend confound. The end-to-end render (which DOES exercise the transformer +
//! LoRA) is a separate, qualitative gate (`tests/lightning_render_real_weights.rs`).
//!
//! `#[ignore]`d — needs the golden from `tools/dump_qwen_lightning_golden.py` (gitignored):
//!   /Users/michael/Repos/mflux/.venv/bin/python tools/dump_qwen_lightning_golden.py
//!   cargo test -p mlx-gen-qwen-image --test lightning_parity -- --ignored --nocapture

use mlx_gen::weights::Weights;
use mlx_gen::DiffusionSampler;
use mlx_gen_qwen_image::FlowMatchSampler;
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/qwen_lightning_sched_golden.safetensors"
);

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-6);
    a.iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()))
        / peak
}

#[test]
#[ignore = "needs tools/golden/qwen_lightning_sched_golden.safetensors (dump_qwen_lightning_golden.py)"]
fn lightning_schedule_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    for n in [4usize, 8] {
        let sampler = FlowMatchSampler::lightning(n);
        let key = format!("lightning_{n}");

        // 1. The sigma schedule (n + 1 entries incl. the trailing 0).
        let want_sigmas = g.require(&format!("{key}.sigmas")).unwrap();
        let ours: Vec<f32> = (0..=n).map(|i| sampler.sigma(i)).collect();
        let ours = Array::from_slice(&ours, &[(n + 1) as i32]);
        assert_eq!(ours.shape(), want_sigmas.shape(), "{key} sigma count");
        let sr = peak_rel(&ours, want_sigmas);
        println!("{key} sigmas peak_rel = {sr:.3e}");
        assert!(sr < 1e-5, "{key} sigmas diverged from diffusers: {sr:.3e}");

        // 2. The per-step deterministic Euler update on the synthetic tensors diffusers used.
        for i in 0..n {
            let v = g.require(&format!("{key}.v{i}")).unwrap();
            let x = g.require(&format!("{key}.x{i}")).unwrap();
            let out = sampler.step(v, x, i).unwrap();
            let want = g.require(&format!("{key}.det{i}")).unwrap();
            let pr = peak_rel(&out, want);
            println!("{key} step {i} det peak_rel = {pr:.3e}");
            assert!(
                pr < 1e-5,
                "{key} step {i} diverged from diffusers: {pr:.3e}"
            );
        }
    }
}
