//! # mlx-gen-sdxl
//!
//! The **Stable Diffusion XL** provider crate for [`mlx-gen`](mlx_gen). SDXL is a **U-Net**
//! generator (not a DiT like Z-Image/FLUX/Qwen), brought into Rust from Apple's vendored
//! `mlx-examples/stable_diffusion` path (`_vendor/mlx_sd/`, MIT) plus SceneWorks' LoRA merge — the
//! last Python image-inference path (sc-2400, epic 2337).
//!
//! Depends only on the `mlx-gen` core (nn primitives, adapters, weights, quant, the `Generator`
//! contract, the registry) and self-registers via `inventory` — linking this crate makes
//! `mlx_gen::load("sdxl", …)` resolve. The port reuses the core conv primitives already built for
//! the Z-Image VAE (`conv2d`, pytorch-compatible `group_norm`, `silu`, `upsample_nearest`) and the
//! shared `image`/`weights`/`quant`/`adapters` layers; it adds the SDXL-specific surfaces: the
//! `UNet2DConditionModel` (down/mid/up cross-attention blocks + time/`text_time` micro-conditioning
//! embeddings), the dual CLIP-L + OpenCLIP-bigG text encoders and their CLIP-BPE tokenizer, the
//! SDXL VAE, and the discrete Euler / Euler-Ancestral sampler with real classifier-free guidance.
//!
//! Parity target = the vendored fp16 reference (`StableDiffusionXL.generate_latents`), validated
//! stage-by-stage against goldens (see `tools/dump_sdxl_golden.py`).

pub mod adapters;
pub mod config;
pub mod inpaint;
pub mod ip_adapter;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod sampler;
pub mod text_encoder;
pub mod tokenizer;
pub mod training;
pub mod unet;
pub mod vae;
pub mod vision_encoder;

pub use adapters::{
    apply_sdxl_adapters, apply_sdxl_adapters_with, lora_delta, LoraCoverage, SdxlLoraReport,
};
pub use config::{
    BetaSchedule, ClipActivation, ClipTextConfig, DiffusionConfig, UNetConfig, VaeConfig,
};
pub use inpaint::{preprocess_mask, InpaintBlend};
pub use ip_adapter::{
    load_ip_kv_pairs, preprocess_clip_image, preprocess_clip_image_sized, IpImageEncoder,
    Resampler, ResamplerConfig,
};
pub use loader::{
    load_controlnet, load_ip_adapter, load_text_encoder_1, load_text_encoder_1_dtype,
    load_text_encoder_2, load_text_encoder_2_dtype, load_tokenizer, load_unet, load_unet_dtype,
    load_unet_kolors_dtype, load_unet_with_config, load_vae,
};
pub use model::{descriptor, load, Sdxl, MODEL_ID};
pub use pipeline::{
    decode_image, decoded_to_image, denoise, denoise_control, denoise_inpaint, denoise_ip,
    denoise_ip_control, encode_conditioning, encode_init_latents, preprocess_control_image,
    preprocess_init_image, seeded_prior, text_time_ids, ControlContext, Denoiser,
};
pub use sampler::EulerSampler;
pub use text_encoder::{ClipOutput, ClipTextEncoder};
pub use tokenizer::{ClipBpeTokenizer, PAD_ID};
pub use training::{load_trainer, SdxlTrainer};
pub use unet::{ControlNet, ControlResiduals, UNet2DConditionModel};
pub use vae::Autoencoder;
pub use vision_encoder::{ClipVisionEncoder, VisionConfig};

use std::sync::atomic::{AtomicBool, Ordering};

/// sc-2963 (rollout of the Wan sc-2957 template): when on, the UNet's remaining fusable elementwise
/// glue — the **SiLU** activations (`x·sigmoid(x)`: ResNet GN→SiLU, the time-embedding MLP, the output
/// head) — runs through `mx.compile`, fusing each into one kernel. The GEGLU/erf-GELU activations are
/// **already** `mx.compile`'d in core `nn` (`gelu_exact`/`gelu_quick`, sc-2721), and the GEGLU
/// `multiply` is a single op (no fusion to win), so SiLU is the only chain left.
///
/// ⚠️ SDXL is **fp16 and precision-load-bearing** ([[sdxl-fp16-sc2721]]): a fused fp16 kernel can round
/// differently from the same ops unfused (that 1-ULP gap is exactly why `gelu_exact` is compiled). The
/// reference runs SiLU eager, so the fp16 golden matches **eager** SiLU — fusing SiLU is only safe if
/// it is **bit-identical** to eager. It is: `tests/compile_parity.rs` proves `max|Δ| = 0` for the
/// compiled SiLU in fp16 AND f32 (the fused `sigmoid`+`multiply` rounds identically — unlike the
/// erf/divide GELU chain). So enabling it cannot move the golden. The VAE SiLUs (f32, once per
/// generation, outside the denoise loop) are left eager. **Enabled by the production denoise loop**
/// ([`pipeline::denoise`]); **off by default**.
static COMPILE_GLUE: AtomicBool = AtomicBool::new(false);

/// Enable/disable compiled elementwise glue (sc-2963). Process-global; set before the denoise loop.
pub fn set_compile_glue(on: bool) {
    COMPILE_GLUE.store(on, Ordering::Relaxed);
}

pub(crate) fn compile_glue() -> bool {
    COMPILE_GLUE.load(Ordering::Relaxed)
}

/// SiLU `x·sigmoid(x)` — one fused kernel when the sc-2963 glue toggle is on, else the eager core
/// [`mlx_gen::nn::silu`]. Bit-identical to eager in fp16 AND f32 (proven `max|Δ|=0`,
/// `tests/compile_parity.rs`), so it is golden-safe on the precision-load-bearing fp16 UNet.
pub(crate) fn silu_glue(x: &mlx_rs::Array) -> mlx_gen::Result<mlx_rs::Array> {
    use mlx_rs::ops::{multiply, sigmoid};
    if !compile_glue() {
        return mlx_gen::nn::silu(x);
    }
    let f = |x_: &mlx_rs::Array| -> std::result::Result<mlx_rs::Array, mlx_rs::error::Exception> {
        multiply(x_, &sigmoid(x_)?)
    };
    Ok(mlx_rs::transforms::compile::compile(f, true)(x)?)
}

#[cfg(test)]
mod sc2963 {
    use mlx_rs::{random, Array, Dtype};

    fn max_abs(a: &Array, b: &Array) -> f32 {
        let d = mlx_rs::ops::abs(mlx_rs::ops::subtract(a, b).unwrap()).unwrap();
        mlx_rs::ops::max(&d, None)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap()
            .item::<f32>()
    }

    // sc-2963 invariant: the compiled SiLU is **bit-identical** to the eager core SiLU in **fp16**
    // (the precision-load-bearing UNet dtype) AND f32 — `max|Δ|=0`. SDXL's fp16 golden matches the
    // eager (reference) SiLU, so a non-zero gap here would mean enabling compile regresses the golden
    // (it doesn't — unlike the erf-GELU chain, the fused `sigmoid`+`multiply` rounds identically).
    #[test]
    fn compiled_silu_bit_identical_to_eager_fp16_and_f32() {
        let k = random::key(0).unwrap();
        for dt in [Dtype::Float16, Dtype::Float32] {
            let x = random::normal::<f32>(&[4, 64, 64, 320], None, None, Some(&k))
                .unwrap()
                .as_dtype(dt)
                .unwrap();
            super::set_compile_glue(false);
            let eager = super::silu_glue(&x).unwrap();
            super::set_compile_glue(true);
            let compiled = super::silu_glue(&x).unwrap();
            super::set_compile_glue(false);
            assert_eq!(compiled.dtype(), dt, "silu_glue preserves dtype {dt:?}");
            let d = max_abs(&compiled, &eager);
            println!("[sdxl silu {dt:?}] max|Δ|={d:.3e}");
            assert_eq!(d, 0.0, "SDXL compiled SiLU diverged from eager in {dt:?}");
        }
    }
}
