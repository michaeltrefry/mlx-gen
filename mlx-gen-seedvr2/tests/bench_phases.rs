//! Phase-timing bench (manual: `cargo test -p mlx-gen-seedvr2 --test bench_phases --release -- --ignored --nocapture`).
//! Times encode / DiT / decode for per-frame (T=1) vs batched video (T=8) at 512², to locate the
//! bottleneck before optimizing. Not a correctness gate. Weight-gated.

use mlx_gen_seedvr2::config::DitConfig;
use mlx_gen_seedvr2::pipeline::Seedvr2Pipeline;
use mlx_rs::transforms::eval;
use mlx_rs::{Array, Dtype};

fn raw_dir() -> Option<std::path::PathBuf> {
    let base = std::path::Path::new(&std::env::var("HOME").unwrap())
        .join(".cache/huggingface/hub/models--numz--SeedVR2_comfyUI/snapshots");
    let snap = std::fs::read_dir(&base).ok()?.flatten().next()?.path();
    snap.join("seedvr2_ema_3b_fp16.safetensors")
        .exists()
        .then_some(snap)
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

#[test]
#[ignore = "manual perf bench"]
fn bench_phase_breakdown() {
    let Some(snap) = raw_dir() else {
        eprintln!("SKIP: raw checkpoint absent");
        return;
    };
    let pipe = Seedvr2Pipeline::load(
        &snap,
        "seedvr2_ema_3b_fp16.safetensors",
        &DitConfig::seedvr2_3b(),
        Dtype::Bfloat16,
    )
    .expect("load 3B bf16");
    let neg = pipe.neg_embed().expect("neg").clone();
    let ts = Array::from_f32(1000.0);

    for (t, label) in [(1i32, "per-frame T=1"), (8, "video T=8")] {
        let (h, w) = (512, 512);
        // synthetic preprocessed clip (1,3,T,H,W) in [-1,1]
        let n = (3 * t * h * w) as usize;
        let data: Vec<f32> = (0..n).map(|i| ((i % 255) as f32 / 127.5) - 1.0).collect();
        let processed = Array::from_slice(&data, &[1, 3, t, h, w])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();

        // warm-up (kernel compilation / weight residency)
        let _ = pipe.run_model_5d(&processed, &mk_noise(&pipe, &processed, 0), &neg, &ts, h, w);

        let reps = 3;
        let (mut enc, mut dit, mut dec) = (0f64, 0f64, 0f64);
        for r in 0..reps {
            let t0 = std::time::Instant::now();
            let latent = pipe.encode(&processed).unwrap();
            eval([&latent]).unwrap();
            enc += ms(t0.elapsed());

            let noise = mk_noise(&pipe, &processed, r as u64);
            let cond = Seedvr2Pipeline::condition(&latent).unwrap();
            let t1 = std::time::Instant::now();
            let latents = pipe.denoise(&noise, &cond, &neg, &ts).unwrap();
            eval([&latents]).unwrap();
            dit += ms(t1.elapsed());

            let t2 = std::time::Instant::now();
            let decoded = pipe.decode_crop_5d(&latents, h, w).unwrap();
            eval([&decoded]).unwrap();
            dec += ms(t2.elapsed());
        }
        let (enc, dit, dec) = (enc / reps as f64, dit / reps as f64, dec / reps as f64);
        let total = enc + dit + dec;
        eprintln!(
            "[{label}] total={total:.0}ms ({:.0}ms/frame)  encode={enc:.0}ms  DiT={dit:.0}ms ({:.0}%)  decode={dec:.0}ms ({:.0}%)",
            total / t as f64,
            100.0 * dit / total,
            100.0 * dec / total,
        );
    }
}

fn mk_noise(pipe: &Seedvr2Pipeline, processed: &Array, seed: u64) -> Array {
    use mlx_rs::random;
    let latent = pipe.encode(processed).unwrap();
    let sh = latent.shape();
    random::normal::<f32>(
        &[1, 16, sh[2], sh[3], sh[4]],
        None,
        None,
        Some(&random::key(seed).unwrap()),
    )
    .unwrap()
    .as_dtype(Dtype::Bfloat16)
    .unwrap()
}
