//! Kolors ControlNet parity (sc-3097).
//!
//! `#[ignore]`d: needs the `Kwai-Kolors/Kolors-ControlNet-Pose` snapshot (+ the Kolors snapshot &
//! `tokenizer.json` for the e2e gate) and `tools/dump_kolors_controlnet_golden.py`. Three gates:
//!
//!  - `kolors_controlnet_forward_matches_diffusers` (f32, the tight component gate): the Rust
//!    `ControlNet::forward` — fed the dumped latents / control image / ChatGLM3 context (4096, which
//!    the branch projects internally with its own `encoder_hid_proj`) / pooled / time_ids — matches
//!    diffusers' 9 down + mid residuals at `conditioning_scale=1.0`.
//!  - `kolors_controlnet_scale0_is_base` (f32): with `control_scale = 0` the residuals are exactly 0,
//!    so the denoise is **byte-identical** to plain T2I (`denoise_latents`) — proves the injection is
//!    non-destructive and correctly wired (no torch ref needed). Run at f32 because at bf16 the (zero)
//!    skip-residual adds in `forward_with_control` aren't bit-transparent and chaos-amplify (see the
//!    test body); f32 isolates the wiring.
//!  - the same test then renders with `control_scale > 0` and asserts the output is coherent AND
//!    actually differs from the scale-0 render (the control influences the image).
//!
//! Run: `cargo test -p mlx-gen-kolors --release --test controlnet_parity -- --ignored --nocapture`

use std::path::PathBuf;

use mlx_gen::weights::Weights;
use mlx_gen::{Image, WeightsSource};
use mlx_gen_kolors::unet::{load_controlnet, ControlNet};
use mlx_gen_kolors::Kolors;
use mlx_rs::{Array, Dtype};

const GOLDEN: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../tools/golden/kolors_controlnet_golden.safetensors"
);

fn kolors_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_SNAPSHOT") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-diffusers/snapshots");
    std::fs::read_dir(&snaps)
        .expect("HF cache snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn controlnet_snapshot() -> PathBuf {
    if let Ok(p) = std::env::var("KOLORS_CONTROLNET") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--Kwai-Kolors--Kolors-ControlNet-Pose/snapshots");
    std::fs::read_dir(&snaps)
        .expect("ControlNet snapshots dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("a snapshot dir")
}

fn rel(a: &Array, b: &Array) -> (f32, f32) {
    let n = b.shape().iter().product::<i32>();
    let (a, b) = (a.reshape(&[n]).unwrap(), b.reshape(&[n]).unwrap());
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-9);
    let mabs = (b.iter().map(|v| v.abs()).sum::<f32>() / b.len() as f32).max(1e-9);
    let max_d = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    let mean_d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max_d / peak, mean_d / mabs)
}

fn load_cn(dtype: Dtype) -> ControlNet {
    load_controlnet(&WeightsSource::Dir(controlnet_snapshot()), dtype).expect("load ControlNet")
}

#[test]
#[ignore = "needs the Kolors-ControlNet-Pose snapshot + tools/golden/kolors_controlnet_golden.safetensors"]
fn kolors_controlnet_forward_matches_diffusers() {
    let g = Weights::from_file(GOLDEN).unwrap();
    let t: f32 = g.metadata("timestep").unwrap().parse().unwrap();
    let num_down: usize = g.metadata("num_down").unwrap().parse().unwrap();
    let cn = load_cn(Dtype::Float32);

    let res = cn
        .forward(
            g.require("latents").unwrap(),
            g.require("control_image").unwrap(),
            t,
            g.require("context").unwrap(), // 4096 — projected internally by the CN's encoder_hid_proj
            g.require("pooled").unwrap(),
            g.require("time_ids").unwrap(),
            1.0,
        )
        .unwrap();

    assert_eq!(res.down.len(), num_down, "down residual count");
    let mut worst = 0f32;
    for (i, d) in res.down.iter().enumerate() {
        let (p, m) = rel(d, g.require(&format!("down{i}")).unwrap());
        println!("down{i}: peak_rel={p:.3e} mean_rel={m:.3e}");
        worst = worst.max(p);
    }
    let (pm, mm) = rel(&res.mid, g.require("mid").unwrap());
    println!("mid: peak_rel={pm:.3e} mean_rel={mm:.3e}");
    worst = worst.max(pm);
    // The torch-CPU-vs-MLX-Metal f32 floor: ~6e-3 peak / <4e-3 mean across the 9 down + mid
    // residuals (higher than the U-Net's ~5e-4 single-forward — the ControlNet adds the cond-image
    // conv stack + the small-magnitude zero-conv heads, so the same absolute f32 noise is a larger
    // *relative* error). All residuals are structurally coherent (mean_rel small); a real wiring /
    // projection bug blows past this by orders of magnitude.
    assert!(
        worst < 1.2e-2,
        "ControlNet residual peak_rel {worst:.3e} exceeds 1.2e-2 (wiring/projection bug)"
    );
    println!("✓ Kolors ControlNet forward matches diffusers (9 down + mid residuals, scale 1.0)");
}

#[test]
#[ignore = "needs the Kolors snapshot + tokenizer.json + Kolors-ControlNet-Pose snapshot"]
fn kolors_controlnet_scale0_is_base() {
    // **f32** for the byte-exact invariant. The zero-scale residuals are *exactly* 0 (verified), so
    // `denoise_control(scale=0)` must equal plain `denoise`. At f32 it is byte-identical; at bf16 it
    // is not, because `forward_with_control` structures the (zero) skip-residual adds differently from
    // `forward` and bf16 `add(x, 0)` is not bit-transparent for every value — a ~2e-2 per-step
    // perturbation that chaos-amplifies over the CFG trajectory (the same bf16-chaos property the T2I
    // gate documents), NOT a wiring defect. f32 isolates the wiring cleanly.
    let snap = kolors_snapshot();
    let kolors = Kolors::load(&snap, Dtype::Float32).expect("load Kolors");
    let cn = load_cn(Dtype::Float32);
    let (h, w, steps, cfg) = (512, 512, 8, 5.0);

    let pos = kolors
        .encode("a portrait of a person, studio lighting")
        .unwrap();
    let neg = kolors.encode("blurry, low quality").unwrap();

    // A deterministic control image (matches the dump's pattern family; any image works for scale 0).
    let mut px = vec![0u8; (h * w * 3) as usize];
    for y in 100..120 {
        for x in 80..430 {
            px[((y * w + x) * 3) as usize] = 255;
        }
    }
    let control = Image {
        width: w as u32,
        height: h as u32,
        pixels: px,
    };

    // Shared init noise so scale-0 control == plain T2I is a fair, deterministic comparison.
    mlx_rs::random::seed(7).unwrap();
    let init_noise =
        mlx_rs::random::normal::<f32>(&[1, h / 8, w / 8, 4], None, None, None).unwrap();

    let base = kolors
        .denoise_latents(&init_noise, &pos, &neg, steps, cfg, h, w)
        .unwrap();
    let s0 = kolors
        .denoise_controlnet_latents(
            &cn,
            &init_noise,
            &control,
            &pos,
            &neg,
            steps,
            cfg,
            0.0,
            h,
            w,
        )
        .unwrap();
    let (p0, _) = rel(&s0, &base);
    println!("scale-0 vs base (f32): peak_rel={p0:.3e}");
    let bytes_eq = {
        let n = base.shape().iter().product::<i32>();
        let a = s0.reshape(&[n]).unwrap();
        let b = base.reshape(&[n]).unwrap();
        a.as_slice::<f32>() == b.as_slice::<f32>()
    };
    assert!(
        bytes_eq,
        "control_scale=0 (f32) must be byte-identical to plain T2I (residual injection not zero-clean)"
    );
    println!("✓ control_scale=0 is byte-identical to plain T2I at f32 (injection wiring verified)");

    // scale > 0: the control must actually move the latents (and the render is coherent).
    let s_on = kolors
        .denoise_controlnet_latents(
            &cn,
            &init_noise,
            &control,
            &pos,
            &neg,
            steps,
            cfg,
            0.7,
            h,
            w,
        )
        .unwrap();
    let (pon, mon) = rel(&s_on, &base);
    println!("scale-0.7 vs base (f32): peak_rel={pon:.3e} mean_rel={mon:.3e}");
    assert!(
        mon > 1e-3,
        "control_scale=0.7 should perturb the latents vs base (mean_rel {mon:.3e} too small)"
    );
    let img = kolors.decode(&s_on).unwrap();
    assert!(
        img.pixels.iter().any(|&p| p > 16) && img.pixels.iter().any(|&p| p < 239),
        "degenerate ControlNet render"
    );
    println!("✓ Kolors ControlNet (scale>0) perturbs the output and renders coherently");
}
