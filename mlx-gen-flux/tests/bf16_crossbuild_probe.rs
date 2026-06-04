//! sc-2787 cross-build bf16 op probe. The mlx-gen FLUX bf16 paths (CLIP + AdaLN modulation) are
//! validated against goldens dumped on the mflux **wheel** (non-NAX MLX 0.31.0), while mlx-gen runs
//! the **NAX** build. f32 ops are bit-identical cross-build (proven: T5 prompt_embeds + the f32
//! transformer substages land at 0.000e0). This probe NAMES the residual in the bf16 paths: it loads
//! bf16 inputs + the wheel's bf16 matmul/SDPA outputs (`tools/dump_bf16_crossbuild_probe.py`) and
//! reruns the same ops on the NAX build, reporting the NAX-vs-wheel bf16 delta (expected ~1e-3 — a
//! sub-bf16-ULP reduction-order difference, NOT a code bug). Run:
//!   cargo test -p mlx-gen-flux --test bf16_crossbuild_probe -- --ignored --nocapture

use mlx_gen::nn::{gelu_tanh, silu};
use mlx_gen::weights::Weights;
use mlx_rs::fast::{layer_norm, rms_norm, scaled_dot_product_attention};
use mlx_rs::nn::gelu;
use mlx_rs::ops::{addmm, matmul, sigmoid};
use mlx_rs::{Array, Dtype};

fn mean_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum();
    let den: f32 = b.iter().map(|y| y.abs()).sum();
    num / den
}

fn peak_rel(a: &Array, b: &Array) -> f32 {
    let n = b.shape().iter().product::<i32>();
    let a = a.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let b = b.as_dtype(Dtype::Float32).unwrap().reshape(&[n]).unwrap();
    let (a, b) = (a.as_slice::<f32>(), b.as_slice::<f32>());
    let peak = b.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let md = a
        .iter()
        .zip(b)
        .fold(0f32, |m, (&x, &y)| m.max((x - y).abs()));
    md / peak
}

fn probe_path() -> String {
    format!(
        "{}/../tools/golden/bf16_crossbuild_probe.safetensors",
        env!("CARGO_MANIFEST_DIR")
    )
}

#[test]
#[ignore = "needs tools/dump_bf16_crossbuild_probe.py run on the mflux wheel first"]
fn bf16_matmul_sdpa_crossbuild_delta() {
    let g = Weights::from_file(probe_path()).unwrap();
    println!(
        "wheel golden mlx={}",
        g.metadata("mlx").unwrap_or("?")
    );
    for name in ["clip_proj", "clip_fc1", "adaln_gemv", "joint_qkv"] {
        let a = g.require(&format!("{name}_a")).unwrap();
        let b = g.require(&format!("{name}_b")).unwrap();
        let c_wheel = g.require(&format!("{name}_c")).unwrap();
        let c_nax = matmul(a, b).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
        println!(
            "matmul {name:11} {:?}x{:?}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
            a.shape(),
            b.shape(),
            mean_rel(&c_nax, c_wheel),
            peak_rel(&c_nax, c_wheel),
        );
    }
    // Biased linear = addmm(bias, x, Wᵀ) — the actual CLIP/AdaLN op.
    for name in ["clip_proj", "adaln_gemv"] {
        let x = g.require(&format!("addmm_{name}_x")).unwrap();
        let w = g.require(&format!("addmm_{name}_w")).unwrap();
        let bias = g.require(&format!("addmm_{name}_bias")).unwrap();
        let y_wheel = g.require(&format!("addmm_{name}_y")).unwrap();
        let y_nax = addmm(bias, x, w.t(), None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        println!(
            "addmm  {name:11} {:?}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
            x.shape(),
            mean_rel(&y_nax, y_wheel),
            peak_rel(&y_nax, y_wheel),
        );
    }

    let q = g.require("sdpa_q").unwrap();
    let k = g.require("sdpa_k").unwrap();
    let v = g.require("sdpa_v").unwrap();
    let o_nax = scaled_dot_product_attention(q, k, v, 1.0 / 8.0, None, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    println!(
        "sdpa_unmasked      {:?}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
        q.shape(),
        mean_rel(&o_nax, g.require("sdpa_o").unwrap()),
        peak_rel(&o_nax, g.require("sdpa_o").unwrap()),
    );
    let mask = g.require("sdpa_mask").unwrap();
    let om_nax = scaled_dot_product_attention(q, k, v, 1.0 / 8.0, mask, None)
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    println!(
        "sdpa_masked        {:?}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
        q.shape(),
        mean_rel(&om_nax, g.require("sdpa_om").unwrap()),
        peak_rel(&om_nax, g.require("sdpa_om").unwrap()),
    );

    // f32 GEMM + f32 SDPA: the transformer main stream. On the NAX build these run TF32 by default
    // (MLX_ENABLE_TF32=1) vs the wheel's true f32 — set MLX_ENABLE_TF32=0 when running this test to
    // see the gap collapse, confirming TF32 (not a code bug) is the transformer's f32 residual.
    let tf32 = std::env::var("MLX_ENABLE_TF32").unwrap_or_else(|_| "(default=1)".into());
    let fc = matmul(
        g.require("f32_mm_a").unwrap(),
        g.require("f32_mm_b").unwrap(),
    )
    .unwrap();
    println!(
        "f32_matmul [256,3072]x[3072,3072] (TF32={tf32}): NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
        mean_rel(&fc, g.require("f32_mm_c").unwrap()),
        peak_rel(&fc, g.require("f32_mm_c").unwrap()),
    );
    let fo = scaled_dot_product_attention(
        g.require("f32_sdpa_q").unwrap(),
        g.require("f32_sdpa_k").unwrap(),
        g.require("f32_sdpa_v").unwrap(),
        1.0 / (128.0_f32).sqrt(),
        None,
        None,
    )
    .unwrap();
    println!(
        "f32_sdpa   [1,24,256,128] (TF32={tf32}): NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
        mean_rel(&fo, g.require("f32_sdpa_o").unwrap()),
        peak_rel(&fo, g.require("f32_sdpa_o").unwrap()),
    );

    // Remaining transformer ops on the f32 main stream.
    let cmp = |label: &str, got: Array, gold_key: &str| {
        let gold = g.require(gold_key).unwrap();
        println!(
            "{label:18}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
            mean_rel(&got, gold),
            peak_rel(&got, gold),
        );
    };
    let tln = layer_norm(g.require("tln_x").unwrap(), None, None, 1e-6).unwrap();
    cmp("ln_affine_false", tln, "tln_y");
    let rn = rms_norm(
        g.require("rms_x").unwrap(),
        g.require("rms_w").unwrap(),
        1e-5,
    )
    .unwrap();
    cmp("rms_norm", rn, "rms_y");
    let ge = gelu(g.require("gelu_x").unwrap().clone()).unwrap();
    cmp("gelu_exact", ge, "gelu_exact");
    let gt = gelu_tanh(g.require("gelu_x").unwrap()).unwrap();
    cmp("gelu_tanh(approx)", gt, "gelu_approx");
    let sf = silu(g.require("gelu_x").unwrap()).unwrap();
    cmp("silu_f32", sf, "silu_f32");
    let sb = silu(g.require("silu_bf16_x").unwrap())
        .unwrap()
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    cmp("silu_bf16", sb, "silu_bf16_y");

    // EXACT joint-block f32 shapes.
    let ja = addmm(
        g.require("f32_addmm_b").unwrap(),
        g.require("f32_addmm_x")
            .unwrap()
            .reshape(&[256, 3072])
            .unwrap(),
        g.require("f32_addmm_w").unwrap().t(),
        None,
        None,
    )
    .unwrap();
    cmp("f32_addmm M=256", ja, "f32_addmm_y");
    let ff1 = addmm(
        g.require("f32_ff1_b").unwrap(),
        g.require("f32_ff1_x").unwrap(),
        g.require("f32_ff1_w").unwrap().t(),
        None,
        None,
    )
    .unwrap();
    cmp("f32_addmm ff1", ff1, "f32_ff1_y");
    let j512 = scaled_dot_product_attention(
        g.require("j512_q").unwrap(),
        g.require("j512_k").unwrap(),
        g.require("j512_v").unwrap(),
        1.0 / (128.0_f32).sqrt(),
        None,
        None,
    )
    .unwrap();
    cmp("f32_sdpa seq=512", j512, "j512_o");

    let ln_x = g.require("ln_x").unwrap();
    let ln_y = layer_norm(
        ln_x,
        Some(g.require("ln_w").unwrap()),
        Some(g.require("ln_b").unwrap()),
        1e-5,
    )
    .unwrap()
    .as_dtype(Dtype::Bfloat16)
    .unwrap();
    println!(
        "layer_norm {:?}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
        ln_x.shape(),
        mean_rel(&ln_y, g.require("ln_y").unwrap()),
        peak_rel(&ln_y, g.require("ln_y").unwrap()),
    );
    let sig_y = sigmoid(ln_x).unwrap().as_dtype(Dtype::Bfloat16).unwrap();
    println!(
        "sigmoid    {:?}: NAX-vs-wheel mean_rel={:.3e} peak_rel={:.3e}",
        ln_x.shape(),
        mean_rel(&sig_y, g.require("sig_y").unwrap()),
        peak_rel(&sig_y, g.require("sig_y").unwrap()),
    );
}
