//! # mlx-gen-lens
//!
//! The **Microsoft Lens / Lens-Turbo** provider crate for [`mlx-gen`](mlx_gen) — a native-MLX port
//! of the turbo-distilled text-to-image model that today runs only in a transformers-5 Python
//! sidecar venv (`/opt/lens-venv`). See epic 3164.
//!
//! Lens-Turbo is three components:
//!
//! 1. **Text encoder — gpt-oss-20b** (`GptOssForCausalLM`, subclassed `LensGptOssEncoder`): a
//!    24-layer **MoE** LLM (hidden 2880, 64 query / 8 KV heads, 32 experts top-4) with **attention
//!    sinks**, **alternating sliding/full attention**, **YaRN RoPE**, and clamped-SwiGLU experts.
//!    Used *encoder-only* — forward to layer 23, capture hidden states at layers `[5, 11, 17, 23]`,
//!    no LM head / KV cache / generation. The 32 expert stacks are MXFP4 in the checkpoint; the
//!    attention/router/embedding modules stay dense bf16 (`modules_to_not_convert`).
//! 2. **Denoising DiT — 48-layer dual-stream MMDiT** (`LensTransformer2DModel`): a near-twin of the
//!    `mlx-gen-qwen-image` MMDiT with a multi-layer text-feature front-end.
//! 3. **VAE — `AutoencoderKLFlux2`** — already ported in [`mlx_gen_flux2`]; only a thin Lens decode
//!    shim is new.
//!
//! ## Status
//!
//! Under construction (epic 3164). Shipped so far:
//! - **sc-3165** — crate scaffold, the [`config`] parser, and the gpt-oss **attention core**
//!   ([`text_encoder::gpt_oss::GptOssAttention`]): GQA + learned attention sinks + alternating
//!   sliding/full causal masks + YaRN RoPE.
//! - **sc-3166** — the **MoE** feed-forward ([`text_encoder::gpt_oss::GptOssMoe`]: top-k router +
//!   clamped-SwiGLU experts) + full **decoder-layer** assembly
//!   ([`text_encoder::gpt_oss::GptOssDecoderLayer`]), with MXFP4 expert dequant
//!   ([`text_encoder::mxfp4`]). Validated single-layer against `transformers.GptOssDecoderLayer`.
//! - **sc-3167** — the **harmony tokenizer + Lens chat-template** ([`text::LensTokenizer`]):
//!   byte-exact `input_ids` vs `LensPipeline._build_chat_inputs` (`txt_offset = 97`).
//! - **sc-3171** — the **encoder-only stack + multi-layer hidden capture**
//!   ([`text_encoder::encoder::LensTextEncoder`]): `embed_tokens` → 24 decoder layers with per-layer
//!   sliding/full masks → capture the layer outputs at `[5, 11, 17, 23]` → early-exit at the last
//!   selected layer. Validated end-to-end against the vendor `LensGptOssEncoder` on real bf16 weights.
//! - **sc-3168** — the **denoising DiT** ([`dit::LensTransformer`]): a 48-layer dual-stream MMDiT
//!   (multi-layer text front-end + fused-QKV joint attention with complex axial RoPE + AdaLN
//!   modulation + SwiGLU GateMLP + `AdaLayerNormContinuous`). Validated against the vendor
//!   `LensTransformer2DModel` on real f32 weights (block-0 peak_rel 1.2e-3, full-forward cosine 0.99998).
//! - **sc-3169** — the **VAE decode shim** ([`vae`]): the Lens latent space is the Flux.2 one, so
//!   `_decode` reduces to a reshape into the packed grid + the shared `mlx_gen_flux2::Flux2Vae`
//!   (bn de-normalize + 2×2 unpatchify + conv decode). Real-weight f32 vs `_decode` (peak_rel 8.7e-3,
//!   the conv-VAE floor).
//! - **sc-3170** — the **sampling schedule + CFG** ([`schedule`]): the Lens empirical-μ schedule is
//!   the core [`mlx_gen::FlowMatchEuler`] verbatim (same constants), plus the shifted-σ timestep
//!   convention and the norm-rescaled CFG. Bit-exact vs the diffusers scheduler (sigmas Δ 0, CFG 2e-7).
//!
//! - **sc-3173** — the **generate / e2e integration** ([`pipeline`] + [`resolution`] + [`registry`]):
//!   [`pipeline::LensPipeline`] ties the four components into one `generate` (tokenize → encode +
//!   `txt_offset` slice → align pos/neg → joint-CFG denoise → VAE decode), [`resolution`] ports the
//!   1024/1440 × 9-aspect buckets, and [`registry`] registers the `lens` + `lens_turbo` ids (4-step /
//!   g1 turbo vs 20-step / CFG-5 base). Validated end-to-end against the vendor `LensPipeline` on real
//!   bf16 weights with injected starting noise: decoded-image cosine **0.996**, final-latent cosine
//!   **0.979** (cross-build structural gate), coherent render (std matches the reference exactly).
//!
//! Still to come: the memory-efficient weight conversion / Q4-Q8 re-quant (sc-3172).

pub mod config;
pub mod dit;
pub mod pipeline;
pub mod registry;
pub mod resolution;
pub mod schedule;
pub mod text;
pub mod text_encoder;
pub mod vae;
